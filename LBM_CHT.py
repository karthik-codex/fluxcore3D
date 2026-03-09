"""
LBM-CHT 3D (Torch/CUDA)

What you asked for (implemented):
- 3D domain with a solid cube obstacle
- GPU accelerated (PyTorch on CUDA), vectorized; only small loops over lattice directions (D3Q19)
- Boundary options:
    * Temperature BC: fixed (inlet fixed tin; outlet zero-gradient; outer faces fixed ambient) OR periodic
    * Flow BC: inlet/outlet (x-direction forced flow) with periodic wrap in y/z OR fully periodic
- Solid cube thermal options:
    * heat source (qdot_total_W distributed over cube cells)
    * or "hot start" (initialize cube to T_hot and watch it cool)
    * or both (hot start + source)
- Snapshots stored into ONE .npz file (time-major arrays)
- PyVista 3D viewer:
    * slider for time frame
    * checkbox widgets to switch between fields
    * fixes the “frozen clim” issue by explicitly updating scalar_range on frame/field change
    
"""

import os
#msvc_path = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"
msvc_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"
#- CUDA torch extensions cached in C:\Users\Karthik\AppData\Local\torch_extensions\torch_extensions\Cache\py312_cu121
if os.path.isdir(msvc_path) and msvc_path not in os.environ["PATH"]:
    os.environ["PATH"] = msvc_path + ";" + os.environ["PATH"]

from tqdm import tqdm 
from datetime import datetime   
import pyvista as pv
import numpy as np    
import types
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn.functional as F
import trimesh

from pre_post.sdf_tracing import build_phi_from_stl_mm, build_ibb_from_sdf_gpu, build_phi_from_voxel_mask_mm
from pre_post.face_roles import resolve_face_roles, _face_ax_idx_fsign
from pre_post.gpu_voxelizer import patch_stl_voxelizer 
from pre_post.solid_assembly import SolidAssembly
# from solver.thermal_cuda_ext import patch_thermal_fused

try:
    from solver.nlevel_mg_cuda import NLevelGeometricMGSolver
except ImportError:
    from solver.nlevel_mg_jacobi import NLevelGeometricMGSolver
from solver.mg_pcg import patch_mg_pcg    
from solver.lbm_cuda_ext import LBMCUDAUpgrade   
from solver.flow_solver_upgrade import FlowSolverUpgrade
from solver.thermal_bc import SurfaceFluxBC, VolumetricHeatBC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

@dataclass
class FluidPropsSI:
    # Flow (SI)
    nu_m2_s: float = 1.0e-6          # kinematic viscosity (m^2/s)  water ~1e-6
    rho_kg_m3: float = 997.0         # density (kg/m^3)
    # Thermal (SI)
    k_W_mK: float = 0.6              # thermal conductivity (W/m/K)
    cp_J_kgK: float = 4182.0         # specific heat (J/kg/K)
    tin_C: float = 25.0              # inlet temperature (°C)

    @property
    def rho_cp_J_m3K(self) -> float:
        return self.rho_kg_m3 * self.cp_J_kgK

@dataclass
class SolidPropsSI:
    # Thermal (SI)
    k_W_mK: float = 400.0            # copper ~400
    rho_kg_m3: float = 8960.0
    cp_J_kgK: float = 385.0

    @property
    def rho_cp_J_m3K(self) -> float:
        return self.rho_kg_m3 * self.cp_J_kgK

@dataclass
class SimConfig3D:
    # Grid resolution (lattice)
    nx: int = 200
    ny: int = 200
    nz: int = 200

    # Physical domain size (SI, meters)
    Lx_m: float = 0.25
    Ly_m: float = 0.25
    Lz_m: float = 0.25

    # Flow BCs
    flow_bc: str = "inlet_outlet"  # "inlet_outlet" or "periodic"
    u_in_mps: float = 0.5          # inlet velocity in m/s (SI)
    flow_dir: str = "+X"           # <<< NEW: "+X","-X","+Y","-Y","+Z","-Z"
    transverse_walls: bool = True
    domain_walls:   List[str] = field(default_factory=list)
    domain_outlets: List[str] = field(default_factory=list)


    # Temperature BCs
    temp_bc: str = "fixed"         # "fixed" or "periodic"
    t_ambient_C: float = 25.0

    # Time stepping: choose dt from velocity unless explicitly set
    dt_phys_s: Optional[float] = None
    u_lat_target: float = 0.05     # target lattice inlet speed (keep < ~0.1)

    # --- NEW: thermal/source time step (SI) decoupled from flow mapping ---
    # If None, thermal uses dt_phys_s (current behavior).
    # If set, temperature evolution + source use dt_thermal_phys_s instead.
    dt_thermal_phys_s: Optional[float] = None   # <<< NEW

    # --- NEW: thermal solve mode ---
    thermal_mode: str = "transient"   # "transient" (one implicit step) or "steady" (iterate to converge)

    # Solid cube obstacle in lattice indices
    obstacle: Dict = None          # {"type":"cube","cx","cy","cz","wx","wy","wz"}

    # Solid init/heating
    solid_init_mode: str = "ambient"   # "ambient" or "hot"
    T_hot_C: float = 0.0
    heating_mode: str = "source"       # "off", "source", "hot", "hot+source"
    qdot_total_W: float = 0.0          # TOTAL power (W) deposited in solid region

    # IO / snapshots
    snap_stride: int = 200
    out_dir: str = "results"
    snapshot_file: str = "snapshots3d.npz"
    store_fields: Tuple[str, ...] = ("u", "v", "w", "speed", "rho", "T", "solid")

    # Monitoring
    print_stride: int = 200

    # Device
    device: str = "cuda"
    dtype: str = "float32"

    # --- NEW: stability controls for BGK ---
    tau_min: float = 0.53          # keep tau away from 0.5 (BGK stability)
    u_lat_max: float = 0.12        # clamp lattice speed (safety)

    collision: str = "trt"         # "bgk" or "trt" (later: "mrt"/"cumulant")
    trt_lambda: float = 0.1875 
    rho_out_lat: float = 1.0   # prescribed outlet density for pressure outlet (lattice units)

    rho_out_lat: float = 1.0   # prescribed outlet density for pressure outlet (lattice units)

    # --- Flow upgrade fields ---
    mrt_s_bulk: float = 1.4        # MRT energy channel relaxation rate
    mrt_s_ghost: float = 1.2       # MRT ghost mode relaxation rate
    outlet_bc_mode: str = "pressure"   # "pressure" (Zou/He) or "convective"
    sponge_length: int = 40         # cells near outlet; 0 = disabled
    sponge_strength: float = 0.3   # max extra relaxation at outlet face
    mrt_s_e: float         = 1.19    # Lallemand's canonical choice
    smag_Cs: float      = 0.10         # Smagorinsky constant (0.08-0.14 typical)

    use_mg_prec: bool = True
    mg_pre_smooth: int = 2
    mg_post_smooth: int = 2
    mg_coarse_iters: int = 20
    mg_omega: float = 0.6

    fluid_k0: int = 0    # first k-index where fluid channel starts (along Z)
    fluid_nz: int = 0    # number of k-cells in the fluid channel

class ObstacleFactory3D:
    @staticmethod
    def cube(nx: int, ny: int, nz: int, cx: int, cy: int, cz: int, wx: int, wy: int, wz: int) -> np.ndarray:
        solid = np.zeros((nx, ny, nz), dtype=bool)
        x0 = max(0, cx - wx // 2)
        x1 = min(nx, cx + (wx - wx // 2))
        y0 = max(0, cy - wy // 2)
        y1 = min(ny, cy + (wy - wy // 2))
        z0 = max(0, cz - wz // 2)
        z1 = min(nz, cz + (wz - wz // 2))
        solid[x0:x1, y0:y1, z0:z1] = True
        return solid

class STLVoxelizer:

    _ROT_MAP = {
        "+X": ([0, 1, 0], -90),
        "-X": ([0, 1, 0],  90),
        "+Y": ([1, 0, 0], -90),
        "-Y": ([1, 0, 0],  90),
        "+Z": ([0, 0, 1],   0),
        "-Z": ([1, 0, 0], 180),
    }

    def __init__(
        self,
        stl_path: str,
        build_dir: str,
        part_scale: float,
        pitch_mm: float,
        out_npz: str,
    ):
        """
        Parameters
        ----------
        stl_path   : path to .stl file
        build_dir  : "+X" | "-X" | "+Y" | "-Y" | "+Z" | "-Z"
                     which axis in the STL file is the build (up) direction
        part_scale : uniform scale applied to the loaded mesh
                     (e.g. 1.0 if STL is already in mm)
        pitch_mm   : voxel edge length in mm.
                     Use:  pitch_mm = (Lx_m / nx) * 1000
                     so that one voxel == one LBM lattice cell.
        out_npz    : path to save the output npz
        """
        self.pitch_mm  = float(pitch_mm)
        self.build_dir = build_dir
        self.scale     = float(part_scale)

        self._log(f"Loading STL: {stl_path}")
        self.mesh = self._load_and_orient(stl_path)

        b = self.mesh.bounds
        sx, sy, sz = (b[1] - b[0]).tolist()
        self._log(f"Part size after orient (W×D×H): {sx:.3f} × {sy:.3f} × {sz:.3f} mm")
        #self._log(f"Voxelizing at pitch = {self.pitch_mm:.4f} mm …")

        self.voxel_solid = self._voxelize()   # uint8 (Nx, Ny, Nz)
        gx, gy, gz = self.voxel_solid.shape
        n_solid = int(self.voxel_solid.sum())
        #self._log(f"Voxel grid: {gx}×{gy}×{gz}  |  solid cells: {n_solid}")

        np.savez(
            out_npz,
            voxel_solid = self.voxel_solid,          # uint8 (Nx, Ny, Nz)
            pitch_mm    = np.float64(self.pitch_mm),
            build_dir   = np.str_(build_dir),
            part_scale  = np.float64(self.scale),
            voxel_origin_mm=self.voxel_origin_mm,
        )
        #self._log(f"Saved → {out_npz}")

    # ── private ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        print(f"[PRE-PROC] {msg}")

    def _load_and_orient(self, path: str) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        mesh.apply_scale(self.scale)

        axis_vec, angle_deg = self._ROT_MAP[self.build_dir]
        R = trimesh.transformations.rotation_matrix(
            np.deg2rad(angle_deg), axis_vec, mesh.centroid
        )
        mesh.apply_transform(R)

        # Centre XY, ground Z=0 at bottom of part
        cx = float(np.mean(mesh.bounds[:, 0]))
        cy = float(np.mean(mesh.bounds[:, 1]))
        cz = float(mesh.bounds[0, 2])
        mesh.apply_translation([-cx, -cy, -cz])
        return mesh

    def _voxelize(self) -> np.ndarray:
        vox = self.mesh.voxelized(pitch=self.pitch_mm).fill()
        self.voxel_origin_mm = vox.transform[:3, 3].copy()
        return vox.matrix.astype(np.uint8)   # (Nx, Ny, Nz)

    def visualize_domain1(self, sim, surface_flux_bcs=None):
        """
        Visualize the simulation domain with:
        - Domain bounding box (wireframe)
        - Solid region (steelblue)
        - Surface flux voxels (red, one per registered SurfaceFluxBC)

        Parameters
        ----------
        sim               : LBMCHT3D_Torch instance
        surface_flux_bcs  : list[SurfaceFluxBC] | None
                            If None, reads from sim._surface_flux_bcs automatically.
                            Pass an explicit list to override.

        Example
        -------
            viz.visualize_domain(sim)
            # or with explicit BCs:
            viz.visualize_domain(sim, surface_flux_bcs=[chip_flux, cold_flux])
        """

        nx, ny, nz = sim.nx, sim.ny, sim.nz

        # ── Collect surface flux BCs ──────────────────────────────────────────
        if surface_flux_bcs is None:
            surface_flux_bcs = getattr(sim, '_surface_flux_bcs', [])

        # ── Build scalar field: 0=fluid, 1=solid, 2+=surface flux face ───────
        # Using values 2, 3, 4... allows multiple BCs to be shown in distinct
        # colors if needed, but we keep it simple: all flux faces = 2.
        domain_np = np.zeros((nx, ny, nz), dtype=np.uint8)
        solid_np  = sim.solid.cpu().numpy()
        domain_np[solid_np] = 1

        flux_colors = [
            ("red",        "Surface Flux"),
            ("orange",     "Surface Flux 2"),
            ("magenta",    "Surface Flux 3"),
            ("yellow",     "Surface Flux 4"),
        ]

        flux_masks = []
        for i, bc in enumerate(surface_flux_bcs):
            mask_np = bc.voxel_mask.cpu().numpy().astype(bool)
            flux_masks.append((mask_np, i, bc))
            # Overwrite solid label for these voxels with a unique index
            domain_np[mask_np] = 2 + i

        # ── PyVista grid ──────────────────────────────────────────────────────
        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(1, 1, 1),
            origin=(0, 0, 0),
        )
        # Cell data (one value per voxel — ImageData cell count = nx*ny*nz)
        grid.cell_data["region"] = domain_np.reshape(-1, order="F").astype(np.float32)

        pl = pv.Plotter()

        # ── Domain bounding box ───────────────────────────────────────────────
        box = pv.Box(bounds=(0, nx, 0, ny, 0, nz))
        pl.add_mesh(box, style="wireframe", color="black", line_width=1, label="Domain")

        # ── Solid (region == 1) ───────────────────────────────────────────────
        solid_mesh = grid.threshold([0.5, 1.5], scalars="region")
        if solid_mesh.n_cells > 0:
            pl.add_mesh(
                solid_mesh,
                color="steelblue",
                opacity=0.6,
                label="Solid",
            )

        # ── Surface flux faces (region == 2 + i) ─────────────────────────────
        for i, bc in enumerate(surface_flux_bcs):
            val = float(2 + i)
            flux_mesh = grid.threshold([val - 0.5, val + 0.5], scalars="region")
            if flux_mesh.n_cells == 0:
                continue

            color_name, _ = flux_colors[min(i, len(flux_colors) - 1)]
            q_total = bc.q_total_W(
                sim.rho_cp,
                sim.rho_cp_ref_J_m3K,
                sim.dx_phys_m,
                sim.dt_thermal_phys_s,
            )
            label = (
                f"Flux {bc.axis_str}  "
                f"q={bc.q_flux_W_m2/1e3:.1f} kW/m²  "
                f"Q≈{q_total:.1f} W  "
                f"({bc.n_voxels} vox)"
            )
            pl.add_mesh(
                flux_mesh,
                color=color_name,
                opacity=1.0,
                label=label,
            )

        # ── Axes labels in physical mm ────────────────────────────────────────
        dx_mm = float(sim.cfg.dx_mm)
        pl.add_text(
            f"Domain: {nx*dx_mm:.0f} × {ny*dx_mm:.0f} × {nz*dx_mm:.0f} mm  |  "
            f"voxel: {dx_mm:.2f} mm",
            position="upper_left",
            font_size=10,
            color="black",
        )

        pl.add_legend(bcolor=(0.95, 0.95, 0.95), border=True, size=(0.3, 0.25))
        pl.add_axes()
        pl.show()

    @staticmethod
    def visualize_domain(sim, surface_flux_bcs=None, solid_bodies=None, volumetric_heat_bcs=None, plotter=None):
            """
            Visualize domain with per-body color coding and heat flux faces.

            Parameters
            ----------
            sim               : LBMCHT3D_Torch instance
            surface_flux_bcs  : list[SurfaceFluxBC] | None  (defaults to sim._surface_flux_bcs)
            solid_bodies      : list of dicts, each:
                                {"npz": "heatsink.npz",
                                "offset_ijk": (i0,j0,k0),
                                "name": "heatsink",
                                "color": "steelblue"}
                                If None, all solid shown as single steelblue region.

            Example
            -------
            geometry.visualize_domain(
                sim,
                solid_bodies=[
                    {"npz": "heatsink.npz", "offset_ijk": (i0,j0,k0_hs),  "name": "heatsink", "color": "steelblue"},
                    {"npz": "CuSpray.npz",  "offset_ijk": (i0,j0,k0_cu),  "name": "CuSpray",  "color": "orange"},
                ],
            )
            """

            nx, ny, nz = sim.nx, sim.ny, sim.nz
            dx_mm = float(sim.cfg.dx_mm)

            # AFTER:
            if surface_flux_bcs is None:
                surface_flux_bcs = getattr(sim, '_surface_flux_bcs', [])
            if volumetric_heat_bcs is None:
                volumetric_heat_bcs = getattr(sim, '_volumetric_heat_bcs', [])

            # ── Build per-body label grid ─────────────────────────────────────────
            # 0 = fluid, 1..N = solid body index, N+1.. = flux faces
            region = np.zeros((nx, ny, nz), dtype=np.int32)

            body_colors = ["steelblue", "darkorange", "mediumseagreen", "mediumpurple",
                        "tomato", "gold", "deepskyblue", "sienna"]

            if solid_bodies is not None:
                for idx, body in enumerate(solid_bodies, start=1):
                    data    = np.load(body["npz"], allow_pickle=True)
                    vox     = data["voxel_solid"].astype(bool)
                    i0, j0, k0 = (int(x) for x in body["offset_ijk"])
                    Gx, Gy, Gz = vox.shape
                    di0 = max(i0,0); di1 = min(i0+Gx, nx)
                    dj0 = max(j0,0); dj1 = min(j0+Gy, ny)
                    dk0 = max(k0,0); dk1 = min(k0+Gz, nz)
                    si0=di0-i0; si1=si0+(di1-di0)
                    sj0=dj0-j0; sj1=sj0+(dj1-dj0)
                    sk0=dk0-k0; sk1=sk0+(dk1-dk0)
                    patch = vox[si0:si1, sj0:sj1, sk0:sk1]
                    region[di0:di1, dj0:dj1, dk0:dk1][patch] = idx
            else:
                # fallback: all solid as body 1
                region[sim.solid.cpu().numpy()] = 1
                solid_bodies = [{"name": "solid", "color": "steelblue"}]

            n_bodies = len(solid_bodies)
            for fi, bc in enumerate(surface_flux_bcs):
                mask_np = bc.voxel_mask.cpu().numpy().astype(bool)
                region[mask_np] = n_bodies + 1 + fi

            n_flux = len(surface_flux_bcs)
            for vi, bc in enumerate(volumetric_heat_bcs):
                mask_np = bc.voxel_mask.cpu().numpy().astype(bool)
                region[mask_np] = n_bodies + 1 + n_flux + vi

            # ── PyVista cell-data grid ────────────────────────────────────────────
            grid = pv.ImageData(
                dimensions=(nx+1, ny+1, nz+1),
                spacing=(1, 1, 1),
                origin=(0, 0, 0),
            )
            grid.cell_data["region"] = region.reshape(-1, order="F").astype(np.float32)

            pl = pv.Plotter()

            # Domain box
            box = pv.Box(bounds=(0, nx, 0, ny, 0, nz))
            pl.add_mesh(box, style="wireframe", color="black", line_width=2, label="Domain")

            # Per-body solid meshes
            for idx, body in enumerate(solid_bodies, start=1):
                color = body.get("color", body_colors[(idx-1) % len(body_colors)])
                name  = body.get("name", f"solid_{idx}")
                mesh  = grid.threshold([idx-0.5, idx+0.5], scalars="region")
                if mesh.n_cells > 0:
                    pl.add_mesh(mesh, color=color, opacity=0.7, label=name)

            # Surface flux BCs
            flux_colors_list = ["limegreen", "springgreen", "greenyellow"] 
            for fi, bc in enumerate(surface_flux_bcs):
                val   = float(n_bodies + 1 + fi)
                mesh  = grid.threshold([val-0.5, val+0.5], scalars="region")
                if mesh.n_cells == 0:
                    continue
                color = flux_colors_list[fi % len(flux_colors_list)]
                q_total = bc.q_flux_W_m2 * bc.n_voxels * (sim.dx_phys_m ** 2)
                label = (f"Flux {bc.axis_str}  "
                        f"q={bc.q_flux_W_m2/1e3:.2f} kW/m²  "
                        f"Q≈{q_total:.1f} W  ({bc.n_voxels} vox)")
                pl.add_mesh(mesh, color=color, opacity=1.0, label=label)

            vol_heat_colors = ["red", "magenta", "yellow", "cyan"]
            for vi, bc in enumerate(volumetric_heat_bcs):
                val  = float(n_bodies + 1 + n_flux + vi)
                mesh = grid.threshold([val-0.5, val+0.5], scalars="region")
                if mesh.n_cells == 0:
                    continue
                color = vol_heat_colors[vi % len(vol_heat_colors)]
                label = (f"Vol.Heat ({getattr(bc, 'name', f'src_{vi}')})  "
                        f"Q={bc.Q_watts:.1f} W  ({bc.n_voxels} vox)")
                pl.add_mesh(mesh, color=color, opacity=0.85, label=label)

            # Annotation
            fluid_k0 = getattr(sim.cfg, 'fluid_k0', 0)
            fluid_nz = getattr(sim.cfg, 'fluid_nz', nz)
            pl.add_text(
                f"Domain: {nx*dx_mm:.0f}×{ny*dx_mm:.0f}×{nz*dx_mm:.0f} mm  |  "
                f"voxel: {dx_mm:.2f} mm  |  "
                f"fluid: k={fluid_k0}–{fluid_k0+fluid_nz}  "
                f"({fluid_nz*dx_mm:.0f} mm)",
                position="upper_left", font_size=10, color="black",
            )

            pl.add_legend(bcolor=(0.95, 0.95, 0.95), border=True, size=(0.35, 0.30))
            pl.add_axes()
            if plotter is None:
                pl.show()   # only call show() when standalone
            else:
                pl.reset_camera(); pl.render()  # embedded — don't block


class LBMCHT3D_Torch:
    """
    D3Q19 BGK flow + 3D scalar advection-diffusion T with variable k, rho_cp on same grid.
    Solid cube:
      - bounce-back for flow (no-slip)
      - conduction via k/rho_cp in T equation
      - optional volumetric heat source or hot initial condition
    """

    # D3Q19 discrete velocities (cx,cy,cz)
    c_np = np.array([
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=int)

    # Weights for D3Q19
    # w0 = 1/3
    # w_axis = 1/18 (6 directions)
    # w_diag = 1/36 (12 directions)
    w_np = np.array(
        [1/3] + [1/18]*6 + [1/36]*12,
        dtype=np.float32
    )

    # Opposite directions index map for D3Q19 matching c_np above
    # 0->0
    # 1<->2, 3<->4, 5<->6
    # 7<->10, 8<->9, 11<->14, 12<->13, 15<->18, 16<->17
    opp_np = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], dtype=int)

    cs2 = 1.0 / 3.0

    def __init__(self, cfg: SimConfig3D, fluid: FluidPropsSI, solid: SolidPropsSI):
        self.cfg = cfg
        self.fluid = fluid
        self.solid_props = solid

        if cfg.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] cfg.device='cuda' but CUDA not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)

        self.dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        nx, ny, nz = self.nx, self.ny, self.nz

        # -------------------- NEW: SI<->lattice mapping --------------------
        # Uniform cubic cells required for consistent mapping.
        dx_x = cfg.dx_mm/1000
        dx_y = cfg.dx_mm/1000
        dx_z = cfg.dx_mm/1000
        dx_phys = float(dx_x)

        if (abs(dx_x - dx_y) / dx_phys > 1e-6) or (abs(dx_x - dx_z) / dx_phys > 1e-6):
            raise ValueError(
                f"Non-cubic cells: dx_x={dx_x}, dx_y={dx_y}, dx_z={dx_z}. "
                "Set nx,ny,nz proportional to Lx,Ly,Lz so dx matches."
            )

        self.dx_phys_m = dx_phys

        # Choose dt_phys from inlet velocity if not provided
        if cfg.dt_phys_s is None:
            u_in = float(cfg.u_in_mps)
            if u_in <= 0.0:
                raise ValueError("u_in_mps must be > 0 when dt_phys_s is None.")
            self.dt_phys_s = float(cfg.u_lat_target) * self.dx_phys_m / u_in
        else:
            self.dt_phys_s = float(cfg.dt_phys_s)

        # --- NEW: decouple thermal/source time from flow mapping ---
        self.dt_flow_phys_s = float(self.dt_phys_s)  # flow mapping time step
        if cfg.dt_thermal_phys_s is None:
            self.dt_thermal_phys_s = self.dt_flow_phys_s
        else:
            self.dt_thermal_phys_s = float(cfg.dt_thermal_phys_s)

        # lattice-time scale factor used by ALL thermal updates (dimensionless)
        self.thermal_dt_scale = self.dt_thermal_phys_s / (self.dt_flow_phys_s + 1e-30)

        # -----------------------------------------------------------

        # Derived lattice inlet speed (for stability reporting)
        self.u_in_lat = float(cfg.u_in_mps) * self.dt_phys_s / self.dx_phys_m

        if self.u_in_lat > 0.12:
            print(f"[WARN] u_in_lat={self.u_in_lat:.4f} is high for LBM. Consider smaller dt_phys_s or lower u_in_mps.")

        # Lattice viscosity from SI
        # nu_lat = nu_phys * dt_phys / dx_phys^2
        nu_lat = float(fluid.nu_m2_s) * self.dt_phys_s / (self.dx_phys_m ** 2)
        if nu_lat <= 0.0:
            raise ValueError("Computed nu_lat <= 0. Check nu_m2_s, dt_phys_s, dx_phys_m.")

        # --- Collision relaxation parameters ---
        tau_plus = 0.5 + nu_lat / self.cs2  # controls viscosity (same as BGK)

        # Enforce minimum tau for stability (this is not a clamp during runtime; it's a mapping feasibility requirement)
        # if tau_plus < float(cfg.tau_min):
        #     raise ValueError(
        #         f"tau_plus={tau_plus:.6g} < tau_min={cfg.tau_min}. "
        #         "Your SI<->LBM mapping is not feasible at this dx, dt, nu. "
        #         "Fix dx / dt / u_in or upgrade collision (TRT/MRT) and BCs."
        #     )

        self.tau_plus = float(tau_plus)
        self.omega_plus = torch.tensor(1.0 / self.tau_plus, device=self.device, dtype=self.dtype)

        if cfg.collision.lower() == "bgk":
            self.omega = self.omega_plus  # keep existing code path working

        elif cfg.collision.lower() == "trt":
            lam = float(cfg.trt_lambda)
            if lam <= 0.0:
                raise ValueError("trt_lambda must be > 0.")
            tau_minus = 0.5 + lam / (self.tau_plus - 0.5)
            self.tau_minus = float(tau_minus)
            self.omega_minus = torch.tensor(1.0 / self.tau_minus, device=self.device, dtype=self.dtype)
            self.omega = self.omega_plus  # sponge + fallback BGK paths use viscous rate
        elif cfg.collision.lower() == "mrt":          # ← ADD THIS BLOCK
            # MRT uses tau_plus for the viscous channels; omega is used by
            # sponge zone and any fallback BGK paths, so set it to omega_plus.
            # The actual per-mode relaxation rates are built in _init_mrt().
            self.omega = self.omega_plus
        elif cfg.collision.lower() == "mrt_smag":          # ← ADD THIS BLOCK
            # MRT uses tau_plus for the viscous channels; omega is used by
            # sponge zone and any fallback BGK paths, so set it to omega_plus.
            # The actual per-mode relaxation rates are built in _init_mrt().
            self.omega = self.omega_plus
        else:
            raise ValueError("cfg.collision must be 'bgk', 'trt', 'mrt', or 'mrt_smag'.")

        # Thermal reference scales (for consistent k/rho_cp)
        self.rho_cp_ref_J_m3K = float(fluid.rho_cp_J_m3K)  # use water as reference
        self.k_ref_W_mK = self.rho_cp_ref_J_m3K * (self.dx_phys_m ** 2) / self.dt_phys_s

        # ---------------------------------------------------------------


        # Constants
        self.c = torch.tensor(self.c_np, device=self.device, dtype=torch.int64)  # (19,3)
        self.cx = self.c[:, 0]
        self.cy = self.c[:, 1]
        self.cz = self.c[:, 2]
        self.w = torch.tensor(self.w_np, device=self.device, dtype=self.dtype)  # (19,)
        self.opp = torch.tensor(self.opp_np, device=self.device, dtype=torch.int64)

        # Masks
        self.solid = torch.zeros((nx, ny, nz), device=self.device, dtype=torch.bool)      # thermal solid (cube)
        self.solid_flow = torch.zeros((nx, ny, nz), device=self.device, dtype=torch.bool) # flow solids

        # Fields
        self.rho = torch.full((nx, ny, nz), 1.0, device=self.device, dtype=self.dtype)  # CHANGED
        self.u = torch.zeros((nx, ny, nz), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((nx, ny, nz), device=self.device, dtype=self.dtype)
        self.wv = torch.zeros((nx, ny, nz), device=self.device, dtype=self.dtype)

        # NEW: temperature ambient uses SI config name
        self.T = torch.full((nx, ny, nz), float(cfg.t_ambient_C), device=self.device, dtype=self.dtype)  # CHANGED
        self._surface_flux_bcs: list[SurfaceFluxBC] = []

        # -------------------- Materials in lattice-scaled form --------------------
        # Store dimensionless rho_cp_lat and k_lat so that:
        # (k_lat / rho_cp_lat) = (k_phys / rho_cp_phys) * dt_phys / dx_phys^2
        rho_cp_f_lat = float(fluid.rho_cp_J_m3K) / self.rho_cp_ref_J_m3K
        k_f_lat = float(fluid.k_W_mK) / self.k_ref_W_mK

        self.rho_cp = torch.full((nx, ny, nz), rho_cp_f_lat, device=self.device, dtype=self.dtype)
        self.k = torch.full((nx, ny, nz), k_f_lat, device=self.device, dtype=self.dtype)
        # -------------------------------------------------------------------------------

        # Populations f: (19, nx, ny, nz)
        self.f = torch.zeros((19, nx, ny, nz), device=self.device, dtype=self.dtype)

        # NEW: streaming buffer (same shape as f)
        self._f_stream = torch.empty_like(self.f)
        # NEW: bounce-back buffer
        self._f_bb = torch.empty_like(self.f)

        # Obstacle
        if cfg.obstacle is not None:
            self.set_obstacle(cfg.obstacle)

        # Init equilibrium
        self._initialize_equilibrium()

        # Init temperature BC
        self._apply_temperature_bcs()

        # Snapshot buffers (CPU)
        self._snapshots = None
        self._snap_count = 0

        print(f"[INIT] device={self.device}, dtype={self.dtype}, nx={nx}, ny={ny}, nz={nz}, omega={float(self.omega):.4f} \n")
        #print(f"[INIT] flow_bc={cfg.flow_bc}, temp_bc={cfg.temp_bc}, heating_mode={cfg.heating_mode}, qdot_total={cfg.qdot_total_W}")
        if cfg.obstacle:
            print(f"[INIT] obstacle={cfg.obstacle}")
        #print(f"[INIT] snap_stride={cfg.snap_stride}, snapshot_file={os.path.join(cfg.out_dir, cfg.snapshot_file)}")

    def set_obstacle(self, obstacle: Dict):
        typ = obstacle.get("type", "cube")
        if typ != "cube":
            raise NotImplementedError("Only cube obstacle implemented in this 3D prototype.")

        solid_np = ObstacleFactory3D.cube(
            self.nx, self.ny, self.nz,
            obstacle["cx"], obstacle["cy"], obstacle["cz"],
            obstacle["wx"], obstacle["wy"], obstacle["wz"],
        )
        self.solid = torch.from_numpy(solid_np).to(self.device, dtype=torch.bool)

        # --------------------------------------------------------------------------

        # Flow solids: just the cube for now (walls handled by BC choice)
        self.solid_flow = self.solid.clone()

        # Thermal properties inside cube
        # -------------------- NEW: solid material (SI -> lattice-scaled) --------------------
        rho_cp_s_lat = float(self.solid_props.rho_cp_J_m3K) / self.rho_cp_ref_J_m3K
        k_s_lat = float(self.solid_props.k_W_mK) / self.k_ref_W_mK

        self.rho_cp[self.solid] = rho_cp_s_lat
        self.k[self.solid] = k_s_lat
        # -------------------------------------------------------------------------------


        # Optional hot-start solid
        if self.cfg.solid_init_mode in ("hot",) or self.cfg.heating_mode in ("hot", "hot+source"):
            self.T[self.solid] = float(self.cfg.T_hot_C)  # CHANGED


        # Force velocities zero in solid
        self.u[self.solid_flow] = 0.0
        self.v[self.solid_flow] = 0.0
        self.wv[self.solid_flow] = 0.0

    def set_obstacle_from_stl(self, npz_path: str, offset_ijk: tuple, solid_props):
        """
        Loads the voxel grid saved by STLVoxelizer and stamps it into self.solid
        at the lattice position given by offset_ijk.

        Parameters
        ----------
        npz_path    : path to the npz created by STLVoxelizer
        offset_ijk  : (i0, j0, k0)  lattice indices where the STL voxel [0,0,0]
                    corner is placed.  Use this to position the part inside your
                    flow domain.
        solid_props : SolidPropsSI instance (same object you pass to LBMCHT3D_Torch)

        Checks
        ------
        Raises ValueError if the stored pitch doesn't match dx_phys_m (within 1%).
        Voxels that fall outside the domain are silently clipped / ignored.

        After this call the sim is in exactly the same state as after set_obstacle():
            self.solid        — Bool mask of solid cells
            self.solid_flow   — same mask (used for bounce-back)
            self.rho_cp / self.k — solid material values stamped in
            self.T            — optionally hot-started
            self.u/v/wv       — zeroed inside solid
        """

        data = np.load(npz_path, allow_pickle=True)
        vox_solid = data["voxel_solid"].astype(bool)   # (Gx, Gy, Gz)
        pitch_mm  = float(data["pitch_mm"])

        # Sanity-check pitch vs LBM dx
        dx_mm = self.dx_phys_m * 1000.0
        if abs(pitch_mm - dx_mm) / dx_mm > 0.01:
            raise ValueError(
                f"STL voxel pitch ({pitch_mm:.4f} mm) does not match LBM dx "
                f"({dx_mm:.4f} mm). Re-voxelize with pitch_mm = Lx_m/nx * 1000."
            )

        Gx, Gy, Gz = vox_solid.shape
        i0, j0, k0 = int(offset_ijk[0]), int(offset_ijk[1]), int(offset_ijk[2])

        # Destination slice in the LBM grid
        di0 = i0;            di1 = min(i0 + Gx, self.nx)
        dj0 = j0;            dj1 = min(j0 + Gy, self.ny)
        dk0 = k0;            dk1 = min(k0 + Gz, self.nz)

        # Source slice in the voxel grid
        si1 = di1 - i0
        sj1 = dj1 - j0
        sk1 = dk1 - k0

        solid_np = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        solid_np[di0:di1, dj0:dj1, dk0:dk1] = vox_solid[:si1, :sj1, :sk1]

        self.solid      = torch.from_numpy(solid_np).to(self.device, dtype=torch.bool)
        self.solid_flow = self.solid.clone()
        #self._apply_transverse_flow_walls()

        n_solid = int(self.solid.sum().item())
        print(f"[STL] solid cells stamped: {n_solid} "
            f"| offset=({i0},{j0},{k0}) | voxel grid={Gx}×{Gy}×{Gz}")

        # ── material properties (identical to set_obstacle) ──────────────────────
        rho_cp_s_lat = float(solid_props.rho_cp_J_m3K) / self.rho_cp_ref_J_m3K
        k_s_lat      = float(solid_props.k_W_mK)       / self.k_ref_W_mK

        self.rho_cp[self.solid] = rho_cp_s_lat
        self.k[self.solid]      = k_s_lat

        # ── optional hot-start ───────────────────────────────────────────────────
        if self.cfg.solid_init_mode == "hot" or self.cfg.heating_mode in ("hot", "hot+source"):
            self.T[self.solid] = float(self.cfg.T_hot_C)

        # ── zero velocity inside solid ───────────────────────────────────────────
        self.u[self.solid_flow]  = 0.0
        self.v[self.solid_flow]  = 0.0
        self.wv[self.solid_flow] = 0.0

    def _apply_transverse_flow_walls_OLD(self):
        if self.cfg.flow_bc != "inlet_outlet":
            return
        if not bool(getattr(self.cfg, "transverse_walls", False)):
            return

        flow_dir = self.cfg.flow_dir.upper().strip()
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        wall = torch.zeros_like(self.solid_flow)

        for d in (0, 1, 2):
            if d == axis:
                continue
            if d == 0:
                wall[0, :, :]  = True
                wall[-1, :, :] = True
            elif d == 1:
                wall[:, 0, :]  = True
                wall[:, -1, :] = True
            else:  # d == 2: restrict to fluid slab only
                wall[:, :, 0]  = True   # domain bottom — always needed
                wall[:, :, -1] = True   # domain top   — always needed
                k0 = int(getattr(self.cfg, 'fluid_k0', 0))
                kN = k0 + int(getattr(self.cfg, 'fluid_nz', self.nz)) - 1
                wall[:, :, k0] = True   # fluid slab floor (redundant when fluid_k0=0)
                wall[:, :, kN] = True   # fluid slab ceiling

        self.solid_flow |= wall
        self.u[self.solid_flow]  = 0.0
        self.v[self.solid_flow]  = 0.0
        self.wv[self.solid_flow] = 0.0

    def _apply_transverse_flow_walls2(self):
        """
        Stamps wall BCs on every face whose role is 'wall' per resolve_face_roles.
        Replaces the old transverse_walls hardcoded logic.
        """
        if self.cfg.flow_bc != "inlet_outlet":
            return

        roles = resolve_face_roles(self.cfg)
        if not any(r == 'wall' for r in roles.values()):
            return

        face_slice = {
            "+X": (slice(-1, None), slice(None),    slice(None)    ),
            "-X": (slice(0, 1),     slice(None),    slice(None)    ),
            "+Y": (slice(None),     slice(-1, None), slice(None)   ),
            "-Y": (slice(None),     slice(0, 1),     slice(None)   ),
            "+Z": (slice(None),     slice(None),    slice(-1, None) ),
            "-Z": (slice(None),     slice(None),    slice(0, 1)    ),
        }

        flow_axis = {"X": 0, "Y": 1, "Z": 2}[self.cfg.flow_dir[1].upper()]
        if flow_axis != 2:   # only relevant when Z is transverse
            k0 = int(getattr(self.cfg, 'fluid_k0', 0))
            kN = k0 + int(getattr(self.cfg, 'fluid_nz', self.nz)) - 1
            if k0 > 0:
                wall[:, :, k0] = True
            if kN < self.nz - 1:
                wall[:, :, kN] = True

        wall = torch.zeros_like(self.solid_flow)
        for face, role in roles.items():
            if role == 'wall':
                wall[face_slice[face]] = True

        self.solid_flow |= wall
        self.u [self.solid_flow] = 0.0
        self.v [self.solid_flow] = 0.0
        self.wv[self.solid_flow] = 0.0

    def _apply_transverse_flow_walls(self):
        if self.cfg.flow_bc != "inlet_outlet":
            return

        roles = resolve_face_roles(self.cfg)

        # Initialize wall tensor unconditionally so k0/kN block always has it
        wall = torch.zeros_like(self.solid_flow)

        face_slice = {
            "+X": (slice(-1, None), slice(None),    slice(None)    ),
            "-X": (slice(0, 1),     slice(None),    slice(None)    ),
            "+Y": (slice(None),     slice(-1, None), slice(None)   ),
            "-Y": (slice(None),     slice(0, 1),     slice(None)   ),
            "+Z": (slice(None),     slice(None),    slice(-1, None) ),
            "-Z": (slice(None),     slice(None),    slice(0, 1)    ),
        }

        for face, role in roles.items():
            if role == 'wall':
                wall[face_slice[face]] = True

        # Fluid slab floor/ceiling for Z-transverse cases (Y or X flow)
        flow_axis = {"X": 0, "Y": 1, "Z": 2}[self.cfg.flow_dir[1].upper()]
        if flow_axis != 2:
            k0 = int(getattr(self.cfg, 'fluid_k0', 0))
            kN = k0 + int(getattr(self.cfg, 'fluid_nz', self.nz)) - 1
            if k0 > 0:
                wall[:, :, k0] = True
            if kN < self.nz - 1:
                wall[:, :, kN] = True

        if not wall.any():
            return

        self.solid_flow |= wall
        self.u [self.solid_flow] = 0.0
        self.v [self.solid_flow] = 0.0
        self.wv[self.solid_flow] = 0.0

    def add_solid_body(
        self,
        npz_path:    str,
        offset_ijk:  tuple,
        solid_props: SolidPropsSI,
        name:        str = "solid",
    ):
        """
        Stamp one voxelized STL body (from STLVoxelizer npz) into self.solid.
        ACCUMULATES with OR — call this multiple times, one per body.
        Call finalize_solid_assembly() once after all bodies are added.

        Parameters
        ----------
        npz_path    : npz produced by STLVoxelizer
        offset_ijk  : (i0, j0, k0) — domain lattice index where voxel [0,0,0] is placed
        solid_props : SolidPropsSI for this body (sets per-cell k and rho_cp)
        name        : label for print output only
        """
        data     = np.load(npz_path, allow_pickle=True)
        vox      = data["voxel_solid"].astype(bool)   # (Gx, Gy, Gz)
        pitch_mm = float(data["pitch_mm"])
        dx_mm    = self.dx_phys_m * 1000.0

        if abs(pitch_mm - dx_mm) / dx_mm > 0.01:
            raise ValueError(
                f"[{name}] STL pitch {pitch_mm:.4f} mm != LBM dx {dx_mm:.4f} mm. "
                f"Re-voxelize with pitch_mm={dx_mm:.4f}."
            )

        Gx, Gy, Gz     = vox.shape
        i0, j0, k0     = int(offset_ijk[0]), int(offset_ijk[1]), int(offset_ijk[2])

        # Domain destination (clamped to grid bounds)
        di0 = max(i0, 0);       di1 = min(i0 + Gx, self.nx)
        dj0 = max(j0, 0);       dj1 = min(j0 + Gy, self.ny)
        dk0 = max(k0, 0);       dk1 = min(k0 + Gz, self.nz)

        # Corresponding source slice in voxel array
        si0 = di0 - i0;         si1 = si0 + (di1 - di0)
        sj0 = dj0 - j0;         sj1 = sj0 + (dj1 - dj0)
        sk0 = dk0 - k0;         sk1 = sk0 + (dk1 - dk0)

        if di1 <= di0 or dj1 <= dj0 or dk1 <= dk0:
            print(f"[{name}] WARNING: body is entirely outside domain — skipped.")
            return

        patch_np = vox[si0:si1, sj0:sj1, sk0:sk1]
        patch    = torch.from_numpy(patch_np).to(self.device, dtype=torch.bool)

        # Accumulate into global solid mask
        self.solid[di0:di1, dj0:dj1, dk0:dk1] |= patch

        # Stamp per-cell material properties
        rho_cp_lat = float(solid_props.rho_cp_J_m3K) / self.rho_cp_ref_J_m3K
        k_lat      = float(solid_props.k_W_mK)       / self.k_ref_W_mK

        dest_rho_cp = self.rho_cp[di0:di1, dj0:dj1, dk0:dk1]
        dest_k      = self.k     [di0:di1, dj0:dj1, dk0:dk1]
        dest_rho_cp[patch] = rho_cp_lat
        dest_k     [patch] = k_lat

        print(f"[{name}] stamped {int(patch.sum().item())} cells  "
            f"offset=({i0},{j0},{k0})  voxel_grid={Gx}×{Gy}×{Gz}  "
            f"k={solid_props.k_W_mK} W/mK")

    def finalize_solid_assembly(self):
        """
        Call ONCE after all add_solid_body() calls.
        Clones solid_flow, zeros velocity inside solid, optionally hot-starts.
        Then build phi/IBB as usual on the combined self.solid.
        """
        self.solid_flow = self.solid.clone()

        self.u [self.solid_flow] = 0.0
        self.v [self.solid_flow] = 0.0
        self.wv[self.solid_flow] = 0.0

        if self.cfg.solid_init_mode == "hot":
            self.T[self.solid] = float(self.cfg.T_hot_C)

        print(f"[Assembly] finalized — total solid cells: {int(self.solid.sum().item())}")

    @staticmethod  
    def solid_stack_mm(npz_path: str, dx_mm: float, axis: int = 2) -> float:
        thickness, _ = LBMCHT3D_Torch.solid_extent_from_npz(npz_path, axis)
        return thickness * dx_mm

    @staticmethod
    def solid_extent_from_npz(npz_path: str, axis: int = 2):
        """Returns (actual_thickness_in_cells, offset_within_grid) along axis."""
        vox = np.load(npz_path, allow_pickle=True)["voxel_solid"].astype(bool)
        idx = np.where(vox)[axis]
        lo = int(idx.min()); hi = int(idx.max())
        return (hi - lo + 1), lo

    @staticmethod
    def compute_domain_with_solid_stack(
        cfg,
        fluid_Lz_m:       float,
        solid_stack_Lz_m: float = 0.0,
        dx_mm:            float = 1.0,
    ):
        """
        Sets cfg.nz, cfg.fluid_k0, cfg.fluid_nz, cfg.Lz_m.
        Call AFTER setting cfg.Lx_m, cfg.Ly_m, cfg.nx, cfg.ny, cfg.dx_mm.

        solid_stack_Lz_m : total height (metres) of solid-only layers BELOW the fluid channel
                        (chip + TIM + whatever). 0 if heatsink base is at domain bottom.
        fluid_Lz_m       : height of the fluid channel (metres).
        """
        nz_solid = round(solid_stack_Lz_m * 1000.0 / dx_mm)
        nz_fluid = round(fluid_Lz_m       * 1000.0 / dx_mm)

        cfg.fluid_k0 = nz_solid
        cfg.fluid_nz = nz_fluid
        cfg.nz       = nz_solid + nz_fluid
        cfg.Lz_m     = cfg.nz * dx_mm / 1000.0

        print(f"[Domain] nz={cfg.nz}  fluid_k0={cfg.fluid_k0}  fluid_nz={cfg.fluid_nz}  "
            f"Lz={cfg.Lz_m*1000:.1f} mm  "
            f"solid_stack={nz_solid} cells  fluid={nz_fluid} cells")


    # LBM core
    # ---------------------------
    def _feq(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        """
        Supports:
        - rho,u,v,wv shape (nx,ny,nz)  -> returns (19,nx,ny,nz)
        - rho,u,v,wv shape (ny,nz)     -> returns (19,ny,nz)  (inlet plane)
        """

        # -------------------- NEW: normalize dims --------------------
        plane2d = (rho.ndim == 2)
        if plane2d:
            # (ny,nz) -> pretend it's (1,ny,nz) for broadcasting
            rho_ = rho[None, :, :]
            u_   = u[None, :, :]
            v_   = v[None, :, :]
            w_   = wv[None, :, :]
        else:
            rho_, u_, v_, w_ = rho, u, v, wv
        # ------------------------------------------------------------

        cu = 3.0 * (
            self.cx[:, None, None, None].to(self.dtype) * u_[None, :, :, :] +
            self.cy[:, None, None, None].to(self.dtype) * v_[None, :, :, :] +
            self.cz[:, None, None, None].to(self.dtype) * w_[None, :, :, :]
        )
        u2 = u_*u_ + v_*v_ + w_*w_
        feq = self.w[:, None, None, None] * rho_[None, :, :, :] * (1.0 + cu + 0.5*cu*cu - 1.5*u2[None, :, :, :])

        # -------------------- NEW: squeeze back if plane2d --------------------
        if plane2d:
            feq = feq[:, 0, :, :]  # (19,ny,nz)
        # ---------------------------------------------------------------------

        return feq

    def _initialize_equilibrium(self):
        self.f[:] = self._feq(self.rho, self.u, self.v, self.wv)

    def _macroscopic(self):
        self.rho = self.f.sum(dim=0)
        inv_rho = 1.0 / (self.rho + 1e-12)

        # D3Q19 momentum sums:
        # u = sum_i f_i * cx_i / rho
        # v = sum_i f_i * cy_i / rho
        # w = sum_i f_i * cz_i / rho
        self.u = (self.f * self.cx[:, None, None, None].to(self.dtype)).sum(dim=0) * inv_rho
        self.v = (self.f * self.cy[:, None, None, None].to(self.dtype)).sum(dim=0) * inv_rho
        self.wv = (self.f * self.cz[:, None, None, None].to(self.dtype)).sum(dim=0) * inv_rho

        self.u[self.solid_flow] = 0.0
        self.v[self.solid_flow] = 0.0
        self.wv[self.solid_flow] = 0.0

    def _collide(self):
        feq = self._feq(self.rho, self.u, self.v, self.wv)

        if self.cfg.collision.lower() == "bgk":
            self.f = self.f + self.omega * (feq - self.f)
            return

        # --- TRT ---
        # f_opp is f mirrored by opposite directions
        f_opp = self.f[self.opp, :, :, :]
        feq_opp = feq[self.opp, :, :, :]

        # Even/odd parts
        f_plus  = 0.5 * (self.f + f_opp)
        f_minus = 0.5 * (self.f - f_opp)

        feq_plus  = 0.5 * (feq + feq_opp)
        feq_minus = 0.5 * (feq - feq_opp)

        # Relax separately
        f_plus  = f_plus  + self.omega_plus  * (feq_plus  - f_plus)
        f_minus = f_minus + self.omega_minus * (feq_minus - f_minus)

        self.f = f_plus + f_minus

    def _stream_OLD(self):
        f_in = self.f
        f_out = self._f_stream

        if self.cfg.flow_bc == "periodic":
            for i in range(19):
                f_out[i] = torch.roll(
                    f_in[i],
                    shifts=(int(self.cx[i]), int(self.cy[i]), int(self.cz[i])),
                    dims=(0, 1, 2)
                )
            self.f, self._f_stream = self._f_stream, self.f
            return

        # -------------------- NEW: per-dimension periodic flags --------------------
        flow_dir = self.cfg.flow_dir.upper().strip()
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]
        transverse_walls = bool(getattr(self.cfg, "transverse_walls", False))

        # Only periodic in transverse directions if transverse_walls is False
        periodic_dim = [False, False, False]
        for d in (0, 1, 2):
            if d == axis:
                periodic_dim[d] = False
            else:
                periodic_dim[d] = (not transverse_walls)
        # -------------------------------------------------------------------------

        # keep previous values so boundary entries remain until BCs/bounce-back handle them
        f_out.copy_(f_in)

        c_comp = (self.cx, self.cy, self.cz)

        for i in range(19):
            fi = f_in[i]

            # Apply periodic rolls where enabled
            for d in (0, 1, 2):
                if periodic_dim[d]:
                    s = int(c_comp[d][i])
                    if s != 0:
                        fi = torch.roll(fi, shifts=s, dims=d)

            # Non-periodic shifts: compute src/dst slices for all dims at once
            sx = int(c_comp[0][i]); sy = int(c_comp[1][i]); sz = int(c_comp[2][i])

            if not periodic_dim[0] or not periodic_dim[1] or not periodic_dim[2]:
                # build slices
                def slc(s):
                    if s == 1:   return (slice(1, None), slice(0, -1))
                    if s == -1:  return (slice(0, -1), slice(1, None))
                    return (slice(None), slice(None))

                dx_dst, dx_src = slc(sx) if not periodic_dim[0] else (slice(None), slice(None))
                dy_dst, dy_src = slc(sy) if not periodic_dim[1] else (slice(None), slice(None))
                dz_dst, dz_src = slc(sz) if not periodic_dim[2] else (slice(None), slice(None))

                # If direction is non-periodic in a dim and shift would require outside values,
                # we simply don't fill those boundary cells (they stay as f_out copy for BC/bounce-back).
                f_out[i, dx_dst, dy_dst, dz_dst] = fi[dx_src, dy_src, dz_src]
            else:
                # fully periodic case (shouldn't happen here)
                f_out[i] = fi

        self.f, self._f_stream = self._f_stream, self.f

    def _stream(self):
        f_in  = self.f
        f_out = self._f_stream

        if self.cfg.flow_bc == "periodic":
            for i in range(19):
                f_out[i] = torch.roll(
                    f_in[i],
                    shifts=(int(self.cx[i]), int(self.cy[i]), int(self.cz[i])),
                    dims=(0, 1, 2))
            self.f, self._f_stream = self._f_stream, self.f
            return

        # Derive periodic flags from resolve_face_roles — single source of truth
        roles = resolve_face_roles(self.cfg)
        periodic_dim = [True, True, True]
        for face, role in roles.items():
            if role in ('inlet', 'outlet', 'wall'):
                ax = {"X": 0, "Y": 1, "Z": 2}[face[1]]
                periodic_dim[ax] = False

        f_out.copy_(f_in)
        c_comp = (self.cx, self.cy, self.cz)

        for i in range(19):
            fi = f_in[i]

            for d in (0, 1, 2):
                if periodic_dim[d]:
                    s = int(c_comp[d][i])
                    if s != 0:
                        fi = torch.roll(fi, shifts=s, dims=d)

            def slc(s):
                if s ==  1: return (slice(1, None), slice(0, -1))
                if s == -1: return (slice(0, -1),   slice(1, None))
                return (slice(None), slice(None))

            sx = int(c_comp[0][i])
            sy = int(c_comp[1][i])
            sz = int(c_comp[2][i])

            dx_dst, dx_src = slc(sx) if not periodic_dim[0] else (slice(None), slice(None))
            dy_dst, dy_src = slc(sy) if not periodic_dim[1] else (slice(None), slice(None))
            dz_dst, dz_src = slc(sz) if not periodic_dim[2] else (slice(None), slice(None))

            f_out[i, dx_dst, dy_dst, dz_dst] = fi[dx_src, dy_src, dz_src]

        self.f, self._f_stream = self._f_stream, self.f

    def _plane_f(self, axis: int, idx: int) -> torch.Tensor:
        # returns view with shape (19, n1, n2)
        if axis == 0:
            return self.f[:, idx, :, :]
        elif axis == 1:
            return self.f[:, :, idx, :]
        else:
            return self.f[:, :, :, idx]

    def _set_plane_f(self, axis: int, idx: int, fplane: torch.Tensor) -> None:
        if axis == 0:
            self.f[:, idx, :, :] = fplane
        elif axis == 1:
            self.f[:, :, idx, :] = fplane
        else:
            self.f[:, :, :, idx] = fplane

    def _set_plane_macro(self, axis: int, idx: int, rho_bc, u_bc, v_bc, w_bc) -> None:
        if axis == 0:
            self.rho[idx, :, :] = rho_bc; self.u[idx, :, :] = u_bc; self.v[idx, :, :] = v_bc; self.wv[idx, :, :] = w_bc
        elif axis == 1:
            self.rho[:, idx, :] = rho_bc; self.u[:, idx, :] = u_bc; self.v[:, idx, :] = v_bc; self.wv[:, idx, :] = w_bc
        else:
            self.rho[:, :, idx] = rho_bc; self.u[:, :, idx] = u_bc; self.v[:, :, idx] = v_bc; self.wv[:, :, idx] = w_bc
    # --------------------------------------------------------------------


    def _bounce_back(self):
        s = self.solid_flow
        if not s.any():
            return

        f_in = self.f
        f_out = self._f_bb

        # start from copy, then overwrite only solid sites
        f_out.copy_(f_in)

        for i in range(19):
            f_out[i, s] = f_in[int(self.opp[i]), s]

        self.f, self._f_bb = f_out, f_in

    def _bounce_back_post_stream(self):
        s = self.solid_flow
        if not s.any():
            return
        f_copy = self.f.clone()
        for i in range(19):
            self.f[i, s] = f_copy[int(self.opp[i]), s]

    def _apply_flow_bcs_OLD(self):
        if self.cfg.flow_bc == "periodic":
            return
        if self.cfg.flow_bc != "inlet_outlet":
            raise ValueError("flow_bc must be 'inlet_outlet' or 'periodic'.")

        # -------------------- axis + sign from cfg.flow_dir --------------------
        flow_dir = self.cfg.flow_dir.upper().strip()  # e.g. "+Y"
        sign = +1 if flow_dir[0] == "+" else -1
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        # plane shape (n1, n2) and inlet/outlet indices along chosen axis
        if axis == 0:
            n1, n2 = self.ny, self.nz
            inlet_idx, outlet_idx = (0, self.nx - 1) if sign == +1 else (self.nx - 1, 0)
            interior_out_idx = (outlet_idx - 1) if outlet_idx > 0 else (outlet_idx + 1)
        elif axis == 1:
            n1, n2 = self.nx, self.nz
            inlet_idx, outlet_idx = (0, self.ny - 1) if sign == +1 else (self.ny - 1, 0)
            interior_out_idx = (outlet_idx - 1) if outlet_idx > 0 else (outlet_idx + 1)
        else:
            n1, n2 = self.nx, self.ny
            inlet_idx, outlet_idx = (0, self.nz - 1) if sign == +1 else (self.nz - 1, 0)
            interior_out_idx = (outlet_idx - 1) if outlet_idx > 0 else (outlet_idx + 1)

        # >>> DEBUG: verify inlet/outlet planes are finite before any BC writes
        f_inlet_before  = self._plane_f(axis, inlet_idx)
        f_outlet_before = self._plane_f(axis, outlet_idx)
        if not torch.isfinite(f_inlet_before).all():
            raise FloatingPointError("Non-finite on inlet plane BEFORE BC")
        if not torch.isfinite(f_outlet_before).all():
            raise FloatingPointError("Non-finite on outlet plane BEFORE BC")


        c_comp = (self.cx, self.cy, self.cz)
        cax = c_comp[axis]  # component along flow axis
        # ----------------------------------------------------------------------

        # Helper: read macro planes (rho,u,v,w) with shape (n1,n2)
        def _plane_macro(axis_: int, idx_: int):
            if axis_ == 0:
                return (self.rho[idx_, :, :], self.u[idx_, :, :], self.v[idx_, :, :], self.wv[idx_, :, :])
            elif axis_ == 1:
                return (self.rho[:, idx_, :], self.u[:, idx_, :], self.v[:, idx_, :], self.wv[:, idx_, :])
            else:
                return (self.rho[:, :, idx_], self.u[:, :, idx_], self.v[:, :, idx_], self.wv[:, :, idx_])

        # Helper: fluid mask on a given plane (n1,n2) True=fluid
        def _plane_fluid_mask(axis_: int, idx_: int):
            # solid_flow True => solid. We want True=fluid.
            if axis_ == 0:
                return (~self.solid_flow[idx_, :, :])
            elif axis_ == 1:
                return (~self.solid_flow[:, idx_, :])
            else:
                return (~self.solid_flow[:, :, idx_])

        # -------------------------
        # ZOU/HE VELOCITY INLET
        # -------------------------
        u_in_val = float(self.u_in_lat) * float(sign)
        u_in_val = float(
            torch.clamp(
                torch.tensor(u_in_val, device=self.device, dtype=self.dtype),
                -float(self.cfg.u_lat_max),
                float(self.cfg.u_lat_max),
            ).item()
        )

        u_bc = torch.zeros((n1, n2), device=self.device, dtype=self.dtype)
        v_bc = torch.zeros((n1, n2), device=self.device, dtype=self.dtype)
        w_bc = torch.zeros((n1, n2), device=self.device, dtype=self.dtype)

        if axis == 0:
            u_bc[:] = u_in_val
        elif axis == 1:
            v_bc[:] = u_in_val
        else:
            w_bc[:] = u_in_val

        # Mask out solids on inlet plane (IMPORTANT: don't apply Zou/He on solids)
        m_in = _plane_fluid_mask(axis, inlet_idx)  # (n1,n2) bool
        u_bc[~m_in] = 0.0
        v_bc[~m_in] = 0.0
        w_bc[~m_in] = 0.0

        # Unknowns at inlet: directions pointing INTO domain from outside:
        # if inlet plane is at idx=0 -> missing cax=+1
        # if inlet plane is at idx=max -> missing cax=-1
        inlet_missing = (+1 if inlet_idx == 0 else -1)
        unknown_in = [i for i in range(19) if int(cax[i]) == inlet_missing]

        f0 = self._plane_f(axis, inlet_idx)  # (19,n1,n2)

        idx_c0 = [i for i in range(19) if int(cax[i]) == 0]
        idx_cneg = [i for i in range(19) if int(cax[i]) == -inlet_missing]

        sum_c0 = f0[idx_c0].sum(dim=0)
        sum_cneg = f0[idx_cneg].sum(dim=0)

        u_ax = (u_bc if axis == 0 else (v_bc if axis == 1 else w_bc))
        #rho_in = (sum_c0 + 2.0 * sum_cneg) / (1.0 - u_ax + 1e-12)
        rho_in = (sum_c0 + 2.0 * sum_cneg) / (1.0 - abs(u_in_val) + 1e-12)
        rho_in = torch.clamp(rho_in, 0.2, 5.0)

        self._regularized_reconstruct_plane(
            axis=axis, idx=inlet_idx,
            rho_bc=rho_in, u_bc=u_bc, v_bc=v_bc, w_bc=w_bc,
            unknown_ids=unknown_in,
            fluid_mask_2d=m_in,
        )


        # >>> DEBUG: after inlet BC
        f_inlet_after = self._plane_f(axis, inlet_idx)
        if not torch.isfinite(f_inlet_after).all():
            raise FloatingPointError("Non-finite on inlet plane AFTER inlet BC")



        # -------------------------
        # ZOU/HE PRESSURE OUTLET
        # -------------------------
        rho_out = torch.full((n1, n2), float(self.cfg.rho_out_lat), device=self.device, dtype=self.dtype)

        # Unknowns at outlet: directions coming from outside:
        # if outlet plane is at idx=max -> missing cax=-1
        # if outlet plane is at idx=0   -> missing cax=+1
        outlet_missing = (-1 if outlet_idx != 0 else +1)
        unknown_out = [i for i in range(19) if int(cax[i]) == outlet_missing]

        f1 = self._plane_f(axis, outlet_idx)

        idx_c0_o = idx_c0

        # Known outgoing populations at outlet
        if outlet_idx != 0:
            idx_known = [i for i in range(19) if int(cax[i]) == +1]
        else:
            idx_known = [i for i in range(19) if int(cax[i]) == -1]

        sum_c0_o = f1[idx_c0_o].sum(dim=0)
        sum_cpos_o = f1[idx_known].sum(dim=0)


        # u_ax_out = 1 - (sum_c0 + 2*sum_cpos)/rho_out  (standard Zou/He pressure outlet)
        #u_ax_out = 1.0 - (sum_c0_o + 2.0 * sum_cpos_o) / (rho_out + 1e-12)
        #u_ax_out = (sum_c0_o + 2.0 * sum_cpos_o) / (rho_out + 1e-12) - 1.0
        known_sign = +1 if outlet_idx != 0 else -1
        u_ax_out = known_sign * ((sum_c0_o + 2.0 * sum_cpos_o) / (rho_out + 1e-12) - 1.0)        

        # Tangential components: zero-gradient from interior neighbor plane
        rho_int, u_int, v_int, w_int = _plane_macro(axis, interior_out_idx)

        # Build outlet macro fields
        u_out = u_int.clone()
        v_out = v_int.clone()
        w_out = w_int.clone()

        # Set axial component from u_ax_out, keep tangentials from interior
        if axis == 0:
            u_out = u_ax_out
        elif axis == 1:
            v_out = u_ax_out
        else:
            w_out = u_ax_out

        rho_bc = rho_out.clone()

        # Backflow detection: velocity pointing into domain at outlet.
        back = (u_ax_out * float(sign) < 0.0)
        if back.any():
            # For backflow nodes, do NOT enforce pressure outlet; copy interior macro state
            rho_bc[back] = rho_int[back]
            u_out[back] = u_int[back]
            v_out[back] = v_int[back]
            w_out[back] = w_int[back]

        # Mask out solids on outlet plane (CRITICAL)
        m_out = _plane_fluid_mask(axis, outlet_idx)  # (n1,n2) bool
        u_out[~m_out] = 0.0
        v_out[~m_out] = 0.0
        w_out[~m_out] = 0.0

        if not hasattr(self, "_bc_stats_printed"):
            self._bc_stats_printed = False
        if (not self._bc_stats_printed):
            nb = int(back.sum().detach().cpu().item())
            nf = int(m_out.sum().detach().cpu().item())
            #print(f"[BC] outlet backflow nodes={nb}/{nf} (fluid only)")
            self._bc_stats_printed = True


        # Backflow nodes: do NOT apply pressure outlet regularization.
        # Copy distributions from interior neighbor plane (most robust).
        if back.any():
            f_out = self._plane_f(axis, outlet_idx)
            f_int = self._plane_f(axis, interior_out_idx)
            for qi in range(19):
                f_out[qi, back] = f_int[qi, back]
            self._set_plane_f(axis, outlet_idx, f_out)

        # Apply regularized pressure outlet only on non-backflow fluid nodes
        m_out2 = m_out & (~back)
        if m_out2.any():
            self._regularized_reconstruct_plane(
                axis=axis, idx=outlet_idx,
                rho_bc=rho_bc, u_bc=u_out, v_bc=v_out, w_bc=w_out,
                unknown_ids=unknown_out,
                fluid_mask_2d=m_out2,
            )

        # ---- NEW: localize outlet NaN source at one node ----

        # >>> DEBUG: after outlet BC
        f_outlet_after = self._plane_f(axis, outlet_idx)
        if not torch.isfinite(f_outlet_after).all():

            x, z = 13, 54
            y = outlet_idx  # 199 for your case

            fnode = self.f[:, x, y, z]          # adjust indexing if you store as (Q,nx,ny,nz)
            rho   = fnode.sum()
            ux    = (fnode * self.c[:,0]).sum() / rho
            uy    = (fnode * self.c[:,1]).sum() / rho
            uz    = (fnode * self.c[:,2]).sum() / rho

            print("[BC-DBG] solid?", bool(self.solid[x,y,z].item()),
                "rho=", float(rho),
                "u=", (float(ux), float(uy), float(uz)),
                "f_finite=", bool(torch.isfinite(fnode).all().item()))

        if not torch.isfinite(f_outlet_after).all():
            # print one bad node
            bad = ~torch.isfinite(f_outlet_after)
            ii, a, b = torch.nonzero(bad, as_tuple=True)  # ii in 0..18, (a,b) in plane coords
            print(f"[BC-NaN] outlet dir={int(ii[0])} plane_coords=({int(a[0])},{int(b[0])}) axis={axis} outlet_idx={outlet_idx}")
            raise FloatingPointError("Non-finite on outlet plane AFTER outlet BC")

    def _apply_flow_bcs(self):
        """
        Multi-face Zou/He dispatcher. Driven by resolve_face_roles.
        Handles arbitrary inlet/outlet/wall combinations for any flow direction.
        """
        if self.cfg.flow_bc == "periodic":
            return
        if self.cfg.flow_bc != "inlet_outlet":
            raise ValueError("flow_bc must be 'inlet_outlet' or 'periodic'.")

        roles     = resolve_face_roles(self.cfg)
        flow_dir  = self.cfg.flow_dir.upper().strip()
        flow_sign = +1 if flow_dir[0] == "+" else -1

        for face, role in roles.items():
            if role not in ('inlet', 'outlet'):
                continue
            ax, pidx, fsign = _face_ax_idx_fsign(self, face)
            if role == 'inlet':
                self._apply_single_plane_bc(ax, pidx, bc_type=0, fsign=flow_sign)
            else:
                self._apply_single_plane_bc(ax, pidx, bc_type=1, fsign=fsign)

    def _apply_single_plane_bc(self, axis: int, plane_idx: int,
                                bc_type: int, fsign: int):
        """
        Zou/He BC for one plane. Works for any of the 6 faces.
        bc_type : 0 = velocity inlet, 1 = pressure outlet
        fsign   : inlet → sign of inlet velocity along axis
                outlet → outward normal sign of this face (backflow detection)
        """
        interior_idx = (plane_idx - 1) if plane_idx > 0 else (plane_idx + 1)

        if   axis == 0: n1, n2 = self.ny, self.nz
        elif axis == 1: n1, n2 = self.nx, self.nz
        else:           n1, n2 = self.nx, self.ny

        cax = (self.cx, self.cy, self.cz)[axis]

        # At low face (idx=0): unknown dirs point +axis into domain
        # At high face (idx=n-1): unknown dirs point -axis into domain
        bc_at_low      = (plane_idx == 0)
        inlet_missing  = +1 if bc_at_low else -1
        known_cax_sign = -inlet_missing

        f_plane = self._plane_f(axis, plane_idx)   # (19, n1, n2) — live view

        if   axis == 0: m_fluid = ~self.solid_flow[plane_idx, :, :]
        elif axis == 1: m_fluid = ~self.solid_flow[:, plane_idx, :]
        else:           m_fluid = ~self.solid_flow[:, :, plane_idx]

        idx_c0      = [i for i in range(19) if int(cax[i]) == 0            ]
        idx_cknown  = [i for i in range(19) if int(cax[i]) == known_cax_sign]
        unknown_ids = [i for i in range(19) if int(cax[i]) == inlet_missing ]

        sum_c0     = f_plane[idx_c0    ].sum(dim=0)
        sum_cknown = f_plane[idx_cknown].sum(dim=0)

        if bc_type == 0:
            # ── Velocity inlet ────────────────────────────────────────────────
            u_in_val = float(self.u_in_lat) * float(fsign)
            u_in_val = float(torch.clamp(
                torch.tensor(u_in_val, device=self.device, dtype=self.dtype),
                -float(self.cfg.u_lat_max), float(self.cfg.u_lat_max)).item())

            u_bc = torch.zeros((n1, n2), device=self.device, dtype=self.dtype)
            v_bc = torch.zeros((n1, n2), device=self.device, dtype=self.dtype)
            w_bc = torch.zeros((n1, n2), device=self.device, dtype=self.dtype)
            if   axis == 0: u_bc[:] = u_in_val
            elif axis == 1: v_bc[:] = u_in_val
            else:           w_bc[:] = u_in_val

            rho_bc = (sum_c0 + 2.0 * sum_cknown) / (1.0 - abs(u_in_val) + 1e-12)
            rho_bc = torch.clamp(rho_bc, 0.2, 5.0)

        else:
            # ── Pressure outlet ───────────────────────────────────────────────
            rho_bc = torch.full((n1, n2), float(self.cfg.rho_out_lat),
                                device=self.device, dtype=self.dtype)

            u_ax_out = float(known_cax_sign) * (
                (sum_c0 + 2.0 * sum_cknown) / (rho_bc + 1e-12) - 1.0)

            # Backflow guard
            back = (u_ax_out * float(fsign) < 0.0)
            if back.any():
                f_int = self._plane_f(axis, interior_idx)
                for qi in range(19):
                    f_plane[qi, back] = f_int[qi, back]
                self._set_plane_f(axis, plane_idx, f_plane)
                m_fluid = m_fluid & ~back

            _, u_int, v_int, w_int = self._plane_macro_tuple(axis, interior_idx)
            u_bc = u_int.clone()
            v_bc = v_int.clone()
            w_bc = w_int.clone()
            if   axis == 0: u_bc = u_ax_out
            elif axis == 1: v_bc = u_ax_out
            else:           w_bc = u_ax_out

        u_bc[~m_fluid] = 0.0
        v_bc[~m_fluid] = 0.0
        w_bc[~m_fluid] = 0.0

        self._regularized_reconstruct_plane(
            axis=axis, idx=plane_idx,
            rho_bc=rho_bc, u_bc=u_bc, v_bc=v_bc, w_bc=w_bc,
            unknown_ids=unknown_ids, fluid_mask_2d=m_fluid)

    def _plane_macro_tuple(self, axis: int, idx: int):
        """(rho, u, v, w) plane views, shape (n1, n2)."""
        if   axis == 0:
            return self.rho[idx,:,:], self.u[idx,:,:], self.v[idx,:,:], self.wv[idx,:,:]
        elif axis == 1:
            return self.rho[:,idx,:], self.u[:,idx,:], self.v[:,idx,:], self.wv[:,idx,:]
        else:
            return self.rho[:,:,idx], self.u[:,:,idx], self.v[:,:,idx], self.wv[:,:,idx]

    def _feq_plane(self, rho2: torch.Tensor, u2: torch.Tensor, v2: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        """Equilibrium for a plane (ny,nz) -> (19,ny,nz)."""
        return self._feq(rho2, u2, v2, w2)  # your _feq already supports plane2d

    def _regularized_reconstruct_plane(
        self, *, axis: int, idx: int,
        rho_bc: torch.Tensor, u_bc: torch.Tensor, v_bc: torch.Tensor, w_bc: torch.Tensor,
        unknown_ids: list,
        fluid_mask_2d: Optional[torch.Tensor] = None,
    ):
        """
        Regularized boundary reconstruction:
        1) Provisional Zou/He reconstruct ONLY unknown directions
        2) Compute Π_neq from (f - f_eq)
        3) Rebuild full plane: f = f_eq + f_neq,reg  (2nd-order Hermite regularization)

        This strongly damps outlet oscillations compared to raw Zou/He. :contentReference[oaicite:1]{index=1}
        """
        # --- views ---
        f_plane = self._plane_f(axis, idx)  # (19,n1,n2) view
        m = fluid_mask_2d
        if m is None:
            # if you never call it without a mask, you can delete this branch
            m = torch.ones_like(rho_bc, dtype=torch.bool)

        # --- equilibrium on plane ---
        feq = self._feq_plane(rho_bc, u_bc, v_bc, w_bc)  # (19,n1,n2)
        feq_opp = feq[self.opp]
        fopp_plane = f_plane[self.opp]

        # --- (1) provisional Zou/He on unknown dirs (ONLY on fluid mask) ---
        for i in unknown_ids:
            f_plane[i, m] = fopp_plane[i, m] + (feq[i, m] - feq_opp[i, m])

        # --- (2) compute Π_neq = Σ_i (f_i - feq_i) c_iα c_iβ ---
        # components (19,)
        cx = self.cx.to(self.dtype)
        cy = self.cy.to(self.dtype)
        cz = self.cz.to(self.dtype)

        # (19,n1,n2) non-eq
        fneq = f_plane - feq

        # Build Π_neq components on mask
        # (n1,n2) each
        Pxx = (fneq * (cx[:, None, None] * cx[:, None, None])).sum(dim=0)
        Pyy = (fneq * (cy[:, None, None] * cy[:, None, None])).sum(dim=0)
        Pzz = (fneq * (cz[:, None, None] * cz[:, None, None])).sum(dim=0)
        Pxy = (fneq * (cx[:, None, None] * cy[:, None, None])).sum(dim=0)
        Pxz = (fneq * (cx[:, None, None] * cz[:, None, None])).sum(dim=0)
        Pyz = (fneq * (cy[:, None, None] * cz[:, None, None])).sum(dim=0)

        # zero out solids explicitly (so they don't pollute moments if mask has holes)
        Pxx = torch.where(m, Pxx, torch.zeros_like(Pxx))
        Pyy = torch.where(m, Pyy, torch.zeros_like(Pyy))
        Pzz = torch.where(m, Pzz, torch.zeros_like(Pzz))
        Pxy = torch.where(m, Pxy, torch.zeros_like(Pxy))
        Pxz = torch.where(m, Pxz, torch.zeros_like(Pxz))
        Pyz = torch.where(m, Pyz, torch.zeros_like(Pyz))

        # --- (3) regularized fneq for each direction: fneq_reg_i = w_i/(2 cs^4) * Q_i : Π_neq ---
        cs2 = self.cs2  # should be 1/3
        cs4 = cs2 * cs2
        wi = self.w.to(self.dtype)[:, None, None]  # (19,1,1)

        # Q tensors per direction (19,):
        Qxx = cx*cx - cs2
        Qyy = cy*cy - cs2
        Qzz = cz*cz - cs2
        Qxy = cx*cy
        Qxz = cx*cz
        Qyz = cy*cz

        # contraction (19,n1,n2)
        S = (
            Qxx[:, None, None]*Pxx[None, :, :] +
            Qyy[:, None, None]*Pyy[None, :, :] +
            Qzz[:, None, None]*Pzz[None, :, :] +
            2.0*(Qxy[:, None, None]*Pxy[None, :, :] +
                Qxz[:, None, None]*Pxz[None, :, :] +
                Qyz[:, None, None]*Pyz[None, :, :])
        )

        fneq_reg = wi * (S / (2.0*cs4 + 1e-30))

        # overwrite full plane on fluid nodes (regularized)
        for i in range(19):
            f_plane[i, m] = feq[i, m] + fneq_reg[i, m]

        # write back
        self._set_plane_f(axis, idx, f_plane)

        # update macros ONLY on fluid nodes
        if axis == 0:
            self.rho[idx, :, :][m] = rho_bc[m]; self.u[idx, :, :][m] = u_bc[m]; self.v[idx, :, :][m] = v_bc[m]; self.wv[idx, :, :][m] = w_bc[m]
        elif axis == 1:
            self.rho[:, idx, :][m] = rho_bc[m]; self.u[:, idx, :][m] = u_bc[m]; self.v[:, idx, :][m] = v_bc[m]; self.wv[:, idx, :][m] = w_bc[m]
        else:
            self.rho[:, :, idx][m] = rho_bc[m]; self.u[:, :, idx][m] = u_bc[m]; self.v[:, :, idx][m] = v_bc[m]; self.wv[:, :, idx][m] = w_bc[m]

    def _interp_bounce_back_from_post(self, f_post: torch.Tensor) -> None:
        """
        Bouzidi–Firdaouss–Lallemand (BFL) interpolated bounce-back.
        Uses post-collision PDFs (f_post) and overwrites streamed PDFs in self.f.

        For each boundary link (x_f, i):
        set f_{opp(i)}(x_f) using BFL interpolation with wall fraction q.
        """
        data = self.ibb_data
        if data.n_links == 0:
            raise ValueError("No IBB data available for bounce-back.")


        # Flatten for fast indexed gathers/scatters
        assert self.f.is_contiguous(), "self.f not contiguous; bounce-back may not write into self.f"
        assert f_post.is_contiguous(), "f_post not contiguous; unexpected layout"
        fS   = self.f.reshape(19, -1)        # streamed (will be modified)
        fPC  = f_post.reshape(19, -1)        # post-collision (read-only here)

        idx_f   = data.fluid_flat           # (N,)
        dirs    = data.dir_i                # (N,)
        opps    = data.dir_opp              # (N,)
        ge      = data.q_ge_half            # (N,) bool
        w1      = data.w_i                  # (N,)
        w2      = data.w_second             # (N,)

        # f_opp(x_f) from post-collision (this is the standard bounce-back source)
        f_opp_f = fPC[opps, idx_f]

        bb = torch.empty_like(f_opp_f)

        # q >= 0.5:
        # f_dir(x_f) = (1/(2q))*f_opp(x_f) + (1 - 1/(2q))*f_dir*(x_f)
        # (this reduces to halfway BB when q=0.5)
        bb[ge] = w1[ge] * f_opp_f[ge] + w2[ge] * fPC[dirs[ge], idx_f[ge]]

        # q < 0.5:
        # f_dir(x_f) = (2q)*f_opp(x_f) + (1 - 2q)*f_opp(x_ff)
        lt = ~ge
        if lt.any():
            idx_ff = data.ff_flat
            ff_ok  = data.ff_valid

            m = lt & ff_ok
            if m.any():
                f_opp_ff = fPC[opps[m], idx_ff[m]]
                bb[m] = w1[m] * f_opp_f[m] + w2[m] * f_opp_ff

            m_bad = lt & (~ff_ok)
            if m_bad.any():
                bb[m_bad] = f_opp_f[m_bad]  # fallback to halfway BB behavior for these links

        # IMPORTANT: write into the UNKNOWN population after streaming: f_dir(x_f)
        fS[dirs, idx_f] = bb

        self._reset_solid_populations()

    def _reset_solid_populations(self) -> None:
        """
        With link-wise IBB we do NOT rely on solid nodes to store reflections.
        Reset PDFs inside solid_flow each step to avoid garbage accumulation.
        """
        s = self.solid_flow
        if not s.any():
            return

        # equilibrium at rest (rho=1, u=v=w=0) -> feq = w_i * rho
        # (works for both BGK/TRT as a benign fill)
        rho1 = torch.ones((s.sum(),), device=self.device, dtype=self.dtype)
        # write per-direction constant; do it without allocations on full grid
        for i in range(19):
            self.f[i, s] = self.w[i]  # rho=1

    def step(self, do_thermal: bool = True) -> int:
        # macroscopic from f(t)
        self._macroscopic()

        # collide -> f_post (post-collision, pre-stream)
        self._collide()

        # persistent post-collision buffer
        if not hasattr(self, "_f_post"):
            self._f_post = torch.empty_like(self.f)
        self._f_post.copy_(self.f)

        # stream -> f_streamed
        self._stream()

        # obstacle BC FIRST (fills missing incoming populations near solids)
        if hasattr(self, "ibb_data") and self.ibb_data is not None and self.ibb_data.n_links > 0:
            self._interp_bounce_back_from_post(self._f_post)
        else:
            self._bounce_back_post_stream()

        # THEN inlet/outlet planes (Zou/He)
        self._apply_flow_bcs()

        # safety check (keep one, not 5)
        if not torch.isfinite(self.f).all():
            bad = torch.nonzero(~torch.isfinite(self.f), as_tuple=False)[0]
            print("[NaN-LOC] after BCs: idx(q,x,y,z)=", tuple(int(v) for v in bad.tolist()))
            raise FloatingPointError("Non-finite after BCs")

        # macroscopic from f(t+1)
        self._macroscopic()

        nsub_fluid, niter_solid = (0, 0)
        return (nsub_fluid, niter_solid)

    def run_flow_to_steady(
        self,
        max_steps: int = 20000,
        check_every: int = 200,
        umax_tol_rel: float = 5e-3,
        umax_min_checks: int = 5,
    ):
        """
        Flow-only iterations until umax stabilizes.
        Includes tqdm progress bar.
        """

        from tqdm import trange

        stable = 0
        umax_prev = None

        pbar = trange(max_steps, desc="FLOW", leave=True)

        for it in pbar:
            self.step(do_thermal=False)

            if (it % check_every) != 0:
                continue

            umax = float(
                torch.max(torch.sqrt(self.u*self.u + self.v*self.v + self.wv*self.wv))
                .detach().cpu().item()
            )

            pbar.set_postfix({"umax": f"{umax:.4f}", "stable": stable})

            if umax_prev is not None:
                rel = abs(umax - umax_prev) / max(abs(umax_prev), 1e-12)

                if rel < umax_tol_rel:
                    stable += 1
                else:
                    stable = 0

                if stable >= umax_min_checks:
                    pbar.close()
                    return {"flow_steps": it, "umax": umax, "stable_checks": stable}

            umax_prev = umax

        pbar.close()
        return {"flow_steps": max_steps, "umax": umax_prev, "stable_checks": stable}


    def _apply_temperature_bcs(self):
        if self.cfg.temp_bc == "periodic":
            return

        Tin = float(self.fluid.tin_C)

        flow_dir = self.cfg.flow_dir.upper().strip()
        sign = +1 if flow_dir[0] == "+" else -1
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        # inlet/outlet indices along flow axis
        if axis == 0:
            inlet_idx, outlet_idx = (0, self.nx - 1) if sign == +1 else (self.nx - 1, 0)
        elif axis == 1:
            inlet_idx, outlet_idx = (0, self.ny - 1) if sign == +1 else (self.ny - 1, 0)
        else:
            inlet_idx, outlet_idx = (0, self.nz - 1) if sign == +1 else (self.nz - 1, 0)

        # 1) Transverse DOMAIN walls: adiabatic (Neumann zero-grad)
        self._apply_neumann_zero_grad_domain_walls_inplace(self.T, axis=axis, inlet_idx=inlet_idx, outlet_idx=outlet_idx)

        # 2) Inlet Dirichlet Tin on inlet plane (only this is Dirichlet now)
        if axis == 0:
            self.T[inlet_idx, :, :] = Tin
        elif axis == 1:
            self.T[:, inlet_idx, :] = Tin
        else:
            self.T[:, :, inlet_idx] = Tin

        # 3) Outlet Neumann zero-gradient on outlet plane: copy from interior neighbor
        if axis == 0:
            nei = outlet_idx - 1 if outlet_idx > 0 else outlet_idx + 1
            self.T[outlet_idx, :, :] = self.T[nei, :, :]
        elif axis == 1:
            nei = outlet_idx - 1 if outlet_idx > 0 else outlet_idx + 1
            self.T[:, outlet_idx, :] = self.T[:, nei, :]
        else:
            nei = outlet_idx - 1 if outlet_idx > 0 else outlet_idx + 1
            self.T[:, :, outlet_idx] = self.T[:, :, nei]

        if not hasattr(self, "_tbcs_printed"):
            # Check one transverse face gradient (example: x=0 face when axis!=0)
            flow_dir = self.cfg.flow_dir.upper().strip()
            axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]
            if axis != 0:
                g = float((self.T[0,:,:] - self.T[1,:,:]).abs().max().detach().cpu().item())
                #print(f"[T-BC] transverse Neumann check: max|T(x0)-T(x1)|={g:.3e} K (should be ~0)")
            self._tbcs_printed = True

    def _apply_neumann_zero_grad_domain_walls_inplace(self, T: torch.Tensor, axis: int, inlet_idx: int, outlet_idx: int) -> None:
        """
        Apply zero-normal-gradient (Neumann) on all DOMAIN faces except:
        - inlet plane (Dirichlet, handled elsewhere)
        - outlet plane (Neumann, handled elsewhere but ok if included)
        This enforces: T_face = T_adjacent_interior
        """
        # X faces (i=0, i=nx-1) are transverse unless flow axis==0
        if axis != 0:
            T[0, :, :]  = T[1, :, :]
            T[-1, :, :] = T[-2, :, :]

        # Y faces
        if axis != 1:
            T[:, 0, :]  = T[:, 1, :]
            T[:, -1, :] = T[:, -2, :]

        # Z faces
        if axis != 2:
            T[:, :, 0]  = T[:, :, 1]
            T[:, :, -1] = T[:, :, -2]

    def _compute_heat_source(self) -> Optional[torch.Tensor]:
        """
        Returns src_step in [K / thermal-step].

        src_step = (qvol / rho_cp_phys) * dt_thermal_phys_s
        qvol = Q / Vsolid [W/m^3]
        rho_cp_phys [J/m^3/K]
        """
        mode = self.cfg.heating_mode
        if mode in ("off", "hot"):
            return None
        if mode not in ("source", "hot+source"):
            raise ValueError("heating_mode must be 'off','source','hot','hot+source'.")

        Q = float(self.cfg.qdot_total_W)
        if Q == 0.0:
            return None
        if not self.solid.any():
            return None

        solid_cells = int(self.solid.sum().detach().cpu().item())
        if solid_cells <= 0:
            return None

        # Solid volume: use STL/declared volume as you already intended (more accurate than voxel count if voxelization is coarse)
        Vstl = float(self.cfg.solid_vol) * 1e-9  # m^3
        if Vstl <= 0.0:
            return None

        qvol = Q / Vstl  # W/m^3

        # rho_cp stored dimensionless; convert to physical
        rho_cp_phys = self.rho_cp * self.rho_cp_ref_J_m3K  # J/m^3/K

        dt_th = float(self.dt_thermal_phys_s)

        src_step = torch.zeros_like(self.T)
        src_step[self.solid] = (qvol / (rho_cp_phys[self.solid] + 1e-30)) * dt_th  # K / thermal-step

        if not hasattr(self, "_src_printed"):
            src_max = float(src_step[self.solid].max().detach().cpu().item())
            Vsolid_vox = solid_cells * (self.dx_phys_m ** 3)
            # print(
            #     f"[HEAT] Q={Q:.6g} W | solid_cells={solid_cells} | "
            #     f"Vstl={Vstl:.6e} m^3 | Vvox={Vsolid_vox:.6e} m^3 | "
            #     f"qvol={qvol:.6e} W/m^3 | src_step_max={src_max:.6e} K/step"
            # )
            self._src_printed = True

        return src_step

    def add_surface_flux(self, bc: SurfaceFluxBC) -> None:
        """
        Register a SurfaceFluxBC to be included in every thermal solve.

        Example
        -------
            bc = SurfaceFluxBC(
                solid_mask    = sim.solid,
                axis          = "+Z",
                surface_L_mm  = 36.0,
                surface_W_mm  = 45.0,
                q_flux_W_m2   = 10_000.0,
                dx_mm         = sim.cfg.dx_mm,
            )
            sim.add_surface_flux(bc)
            print(f"Q_total = {bc.q_total_W(sim.rho_cp, sim.rho_cp_ref_J_m3K,
                                            sim.dx_phys_m, sim.dt_thermal_phys_s):.2f} W")
        """
        if not hasattr(self, '_surface_flux_bcs'):
            self._surface_flux_bcs = []
        self._surface_flux_bcs.append(bc)
        print(
            f"[SurfaceFlux] registered axis={bc.axis_str}, "
            f"q={bc.q_flux_W_m2:.4g} W/m², "
            f"n_voxels={bc.n_voxels}, "
            f"Q_total≈{bc.q_total_W(self.rho_cp, self.rho_cp_ref_J_m3K, self.dx_phys_m, self.dt_thermal_phys_s):.2f} W"
        )

    def clear_surface_fluxes(self) -> None:
        """Remove all registered SurfaceFluxBCs."""
        self._surface_flux_bcs = []

    def add_volumetric_heat(self, bc: VolumetricHeatBC) -> None:
        if not hasattr(self, '_volumetric_heat_bcs'):
            self._volumetric_heat_bcs = []
        self._volumetric_heat_bcs.append(bc)

    def _compute_volumetric_heat_sources(self) -> torch.Tensor:
        out = torch.zeros_like(self.T)
        if not hasattr(self, '_volumetric_heat_bcs') or not self._volumetric_heat_bcs:
            return out
        for bc in self._volumetric_heat_bcs:
            out.add_(bc.src_step(self.rho_cp, self.rho_cp_ref_J_m3K,
                                self.dx_phys_m, self.dt_thermal_phys_s))
        return out

    def _compute_surface_flux_sources(self) -> torch.Tensor:
        """
        Returns the summed src_step contribution [K / thermal-step] from ALL
        registered SurfaceFluxBC objects.

        Returns zero tensor if no BCs are registered.
        """
        out = torch.zeros_like(self.T)
        if not hasattr(self, '_surface_flux_bcs') or not self._surface_flux_bcs:
            return out
        for bc in self._surface_flux_bcs:
            out.add_(bc.src_step(
                rho_cp            = self.rho_cp,
                rho_cp_ref_J_m3K  = self.rho_cp_ref_J_m3K,
                dx_phys_m         = self.dx_phys_m,
                dt_thermal_phys_s = self.dt_thermal_phys_s,
            ))
        return out

    def _build_update_mask_thermal(self) -> torch.Tensor:
        """
        Unknowns mask (True = solve for this node).

        Dirichlet:
        - inlet plane only (Tin)
        Neumann:
        - outlet plane (zero-grad)
        - transverse domain walls (adiabatic)
        """
        if self.cfg.temp_bc == "periodic":
            return torch.ones_like(self.T, dtype=torch.bool)

        flow_dir = self.cfg.flow_dir.upper().strip()
        sign = +1 if flow_dir[0] == "+" else -1
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        if axis == 0:
            inlet_idx = 0 if sign == +1 else (self.nx - 1)
        elif axis == 1:
            inlet_idx = 0 if sign == +1 else (self.ny - 1)
        else:
            inlet_idx = 0 if sign == +1 else (self.nz - 1)

        m = torch.ones_like(self.T, dtype=torch.bool)

        # Freeze inlet only (Dirichlet)
        if axis == 0:
            m[inlet_idx, :, :] = False
        elif axis == 1:
            m[:, inlet_idx, :] = False
        else:
            m[:, :, inlet_idx] = False

        return m

    def _apply_temperature_bcs_inplace(self, T: torch.Tensor) -> None:
        if self.cfg.temp_bc == "periodic":
            return

        Tin = float(self.fluid.tin_C)

        flow_dir = self.cfg.flow_dir.upper().strip()
        sign = +1 if flow_dir[0] == "+" else -1
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        if axis == 0:
            inlet_idx, outlet_idx = (0, self.nx - 1) if sign == +1 else (self.nx - 1, 0)
        elif axis == 1:
            inlet_idx, outlet_idx = (0, self.ny - 1) if sign == +1 else (self.ny - 1, 0)
        else:
            inlet_idx, outlet_idx = (0, self.nz - 1) if sign == +1 else (self.nz - 1, 0)

        # 1) Transverse DOMAIN walls: adiabatic (Neumann)
        self._apply_neumann_zero_grad_domain_walls_inplace(T, axis=axis, inlet_idx=inlet_idx, outlet_idx=outlet_idx)

        # 2) Inlet Dirichlet Tin
        if axis == 0:
            T[inlet_idx, :, :] = Tin
        elif axis == 1:
            T[:, inlet_idx, :] = Tin
        else:
            T[:, :, inlet_idx] = Tin

        # 3) Outlet Neumann
        if axis == 0:
            nei = outlet_idx - 1 if outlet_idx > 0 else outlet_idx + 1
            T[outlet_idx, :, :] = T[nei, :, :]
        elif axis == 1:
            nei = outlet_idx - 1 if outlet_idx > 0 else outlet_idx + 1
            T[:, outlet_idx, :] = T[:, nei, :]
        else:
            nei = outlet_idx - 1 if outlet_idx > 0 else outlet_idx + 1
            T[:, :, outlet_idx] = T[:, :, nei]

    def _thermal_stencil_coeffs_steady(self):
        """
        Build neighbor coefficients for STEADY operator per THERMAL STEP:
            L_th(T) = diffusion_th(T) + convection_th(T)

        Returns:
            (aE,aW,aN,aS,aU,aD, diag) where diag = sum(a*)
        """
        dx = 1.0
        k = self.k
        rho_cp = self.rho_cp
        inv_rcp = 1.0 / (rho_cp + 1e-30)

        # Scale everything from "per flow-step" to "per thermal-step"
        s = float(self.thermal_dt_scale)

        periodic = (self.cfg.temp_bc == "periodic")

        if periodic:
            def E(A): return torch.roll(A, shifts=-1, dims=0)
            def W(A): return torch.roll(A, shifts=+1, dims=0)
            def N(A): return torch.roll(A, shifts=-1, dims=1)
            def S(A): return torch.roll(A, shifts=+1, dims=1)
            def U(A): return torch.roll(A, shifts=-1, dims=2)
            def D(A): return torch.roll(A, shifts=+1, dims=2)
        else:
            # NOTE: this branch only needs to be consistent with your chosen outlet Neumann.
            # Transverse wall Neumann is enforced by your explicit BC overwrite; if you want
            # perfect operator-consistency, mirror those walls here too (optional).
            flow_dir = self.cfg.flow_dir.upper().strip()
            sign = +1 if flow_dir[0] == "+" else -1
            axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

            if axis == 0:
                inlet_idx, outlet_idx = (0, self.nx - 1) if sign == +1 else (self.nx - 1, 0)
            elif axis == 1:
                inlet_idx, outlet_idx = (0, self.ny - 1) if sign == +1 else (self.ny - 1, 0)
            else:
                inlet_idx, outlet_idx = (0, self.nz - 1) if sign == +1 else (self.nz - 1, 0)

            def _shift_dim(A: torch.Tensor, dim: int, step: int) -> torch.Tensor:
                """
                Non-periodic neighbor fetch with Neumann ghosting consistent with thermal BCs:
                - outlet Neumann on flow axis outlet face
                - transverse domain walls Neumann on faces not aligned with flow axis
                """
                B = A.clone()

                if dim == 0:
                    if step == +1:
                        B[:-1, :, :] = A[1:, :, :]
                        # high-x face ghost
                        if (axis == 0 and outlet_idx == self.nx - 1) or (axis != 0):
                            B[-1, :, :] = A[-2, :, :]
                        else:
                            B[-1, :, :] = A[-1, :, :]
                    else:
                        B[1:, :, :] = A[:-1, :, :]
                        # low-x face ghost
                        if (axis == 0 and outlet_idx == 0) or (axis != 0):
                            B[0, :, :] = A[1, :, :]
                        else:
                            B[0, :, :] = A[0, :, :]
                    return B

                if dim == 1:
                    if step == +1:
                        B[:, :-1, :] = A[:, 1:, :]
                        if (axis == 1 and outlet_idx == self.ny - 1) or (axis != 1):
                            B[:, -1, :] = A[:, -2, :]
                        else:
                            B[:, -1, :] = A[:, -1, :]
                    else:
                        B[:, 1:, :] = A[:, :-1, :]
                        if (axis == 1 and outlet_idx == 0) or (axis != 1):
                            B[:, 0, :] = A[:, 1, :]
                        else:
                            B[:, 0, :] = A[:, 0, :]
                    return B

                # dim == 2
                if step == +1:
                    B[:, :, :-1] = A[:, :, 1:]
                    if (axis == 2 and outlet_idx == self.nz - 1) or (axis != 2):
                        B[:, :, -1] = A[:, :, -2]
                    else:
                        B[:, :, -1] = A[:, :, -1]
                else:
                    B[:, :, 1:] = A[:, :, :-1]
                    if (axis == 2 and outlet_idx == 0) or (axis != 2):
                        B[:, :, 0] = A[:, :, 1]
                    else:
                        B[:, :, 0] = A[:, :, 0]
                return B

            def E(A): return _shift_dim(A, dim=0, step=+1)
            def W(A): return _shift_dim(A, dim=0, step=-1)
            def N(A): return _shift_dim(A, dim=1, step=+1)
            def S(A): return _shift_dim(A, dim=1, step=-1)
            def U(A): return _shift_dim(A, dim=2, step=+1)
            def D(A): return _shift_dim(A, dim=2, step=-1)

        # --- diffusion (harmonic face k) ---
        kE, kW, kN, kS, kU, kD = E(k), W(k), N(k), S(k), U(k), D(k)
        kx_p = 2.0 * k * kE / (k + kE + 1e-30)
        kx_m = 2.0 * k * kW / (k + kW + 1e-30)
        ky_p = 2.0 * k * kN / (k + kN + 1e-30)
        ky_m = 2.0 * k * kS / (k + kS + 1e-30)
        kz_p = 2.0 * k * kU / (k + kU + 1e-30)
        kz_m = 2.0 * k * kD / (k + kD + 1e-30)

        # alpha_flow_lat = (k_lat/rho_cp_lat)  (already encodes dt_flow/dx^2)
        # alpha_th_lat   = s * alpha_flow_lat
        alphaE = s * (kx_p * inv_rcp)
        alphaW = s * (kx_m * inv_rcp)
        alphaN = s * (ky_p * inv_rcp)
        alphaS = s * (ky_m * inv_rcp)
        alphaU = s * (kz_p * inv_rcp)
        alphaD = s * (kz_m * inv_rcp)

        c = 1.0 / (dx * dx)
        aE_d = c * alphaE; aW_d = c * alphaW
        aN_d = c * alphaN; aS_d = c * alphaS
        aU_d = c * alphaU; aD_d = c * alphaD

        # --- convection (upwind) ---
        # u,v,w from LBM are "per flow-step"; convert to "per thermal-step"
        u = s * self.u
        v = s * self.v
        w = s * self.wv

        upx = torch.clamp(u, min=0.0); umx = torch.clamp(-u, min=0.0)
        upy = torch.clamp(v, min=0.0); umy = torch.clamp(-v, min=0.0)
        upz = torch.clamp(w, min=0.0); umz = torch.clamp(-w, min=0.0)

        adv = 1.0 / dx
        aW_a = adv * upx
        aE_a = adv * umx
        aS_a = adv * upy
        aN_a = adv * umy
        aD_a = adv * upz
        aU_a = adv * umz

        aE = aE_d + aE_a
        aW = aW_d + aW_a
        aN = aN_d + aN_a
        aS = aS_d + aS_a
        aU = aU_d + aU_a
        aD = aD_d + aD_a

        diag = (aE + aW + aN + aS + aU + aD)
        return aE, aW, aN, aS, aU, aD, diag

    def _build_T_dirichlet_field(self) -> torch.Tensor:
        """
        Dirichlet field for theta-splitting.
        With adiabatic transverse walls, the ONLY Dirichlet is inlet Tin.
        """
        Tin = float(self.fluid.tin_C)

        flow_dir = self.cfg.flow_dir.upper().strip()
        sign = +1 if flow_dir[0] == "+" else -1
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        # baseline: Tin everywhere (benign)
        T0 = torch.full_like(self.T, Tin)

        # inlet plane Tin (Dirichlet)
        if axis == 0:
            inlet_idx = 0 if sign == +1 else (self.nx - 1)
            T0[inlet_idx, :, :] = Tin
        elif axis == 1:
            inlet_idx = 0 if sign == +1 else (self.ny - 1)
            T0[:, inlet_idx, :] = Tin
        else:
            inlet_idx = 0 if sign == +1 else (self.nz - 1)
            T0[:, :, inlet_idx] = Tin

        return T0

    def solve_thermal_steady_only(
        self,
        # NOTE: `solvers` argument removed — no longer needed
        max_outer:         int   = 500,
        tol_dT_solid:      float = 1e-2,
        tol_dT_fluid:      float = 1e-2,
        tol_dTout_mean:    float = 5e-3,
        dt_scale_min:      float = 1.0,
        dt_scale_max:      float = 200.0,
        dt_scale_start:    float = 5.0,
        dt_ramp_factor:    float = 1.5,
        dt_backoff_factor: float = 0.5,
        dt_ramp_threshold: float = 0.5,
        max_mg_cycles:     int   = 20,
        tol_mg:            float = 1e-3,
        stable_steps:       int   = 5,
    ):

        max_thermal_dt_scale = 10.0
        if float(self.thermal_dt_scale) > max_thermal_dt_scale:
            import warnings
            warnings.warn(
                f"thermal_dt_scale={self.thermal_dt_scale:.1f} > {max_thermal_dt_scale}. "
                f"Set cfg.dt_thermal_phys_s <= {max_thermal_dt_scale * self.dt_flow_phys_s:.4f} s "
                f"to avoid MG ill-conditioning at this dx={self.dx_phys_m*1000:.2f} mm."
            )

        solid = self.solid
        fluid = ~solid

        flow_dir = self.cfg.flow_dir.upper().strip()
        sign     = +1 if flow_dir[0] == "+" else -1
        ax       = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]
        # store on self so _temperature_step_mg can read them
        self.sign = sign
        self.ax   = ax

        # ── Multi-outlet aware convergence monitor ────────────────────────────
        # Use resolve_face_roles to find all outlet faces.
        # This correctly handles -Z impingement (side exits) and any other config.
        import numpy as _np

        roles = resolve_face_roles(self.cfg)
        _outlet_planes = []   # list of (axis, idx) for each outlet face
        for _face, _role in roles.items():
            if _role == 'outlet':
                _fax = {"X": 0, "Y": 1, "Z": 2}[_face[1]]
                _fn  = (self.nx, self.ny, self.nz)[_fax]
                _idx = _fn - 1 if _face[0] == '+' else 0
                _outlet_planes.append((_fax, _idx))

        def _Tout():
            """Mean fluid temperature across all outlet faces."""
            vals = []
            for (_fax, _idx) in _outlet_planes:
                if   _fax == 0: Ts = self.T[_idx, :, :];  fs = fluid[_idx, :, :]
                elif _fax == 1: Ts = self.T[:, _idx, :];  fs = fluid[:, _idx, :]
                else:           Ts = self.T[:, :, _idx];  fs = fluid[:, :, _idx]
                if fs.any():
                    vals.append(float(Ts[fs].mean()))
            if vals:
                return float(_np.mean(vals))
            # Fallback: mean of all fluid cells (should rarely be needed)
            return float(self.T[fluid].mean()) if fluid.any() else float(self.T.mean())

        f_out_valid = False
        for (_fax, _idx) in _outlet_planes:
            if   _fax == 0: _fs = fluid[_idx, :, :]
            elif _fax == 1: _fs = fluid[:, _idx, :]
            else:           _fs = fluid[:, :, _idx]
            if _fs.any():
                f_out_valid = True
                break


        # if ax == 0:
        #     out_idx = self.nx - 1 if sign == +1 else 0
        #     f_out   = fluid[out_idx, :, :]
        #     def _Tout(): return self.T[out_idx, :, :]
        # elif ax == 1:
        #     out_idx = self.ny - 1 if sign == +1 else 0
        #     f_out   = fluid[:, out_idx, :]
        #     def _Tout(): return self.T[:, out_idx, :]
        # else:
        #     out_idx = self.nz - 1 if sign == +1 else 0
        #     f_out   = fluid[:, :, out_idx]
        #     def _Tout(): return self.T[:, :, out_idx]

        Tout_prev = None
        dt_use    = float(dt_scale_start)
        dT_s_inf  = float("inf")
        dT_f_inf  = float("inf")
        dTout     = float("inf")
        Tout_mean = float("nan")

        bar = tqdm(range(max_outer), desc="THERMAL", dynamic_ncols=True)
        stable = 0
        for outer in bar:
            Tprev = self.T.clone()

            # ── single pseudo-time step via N-level MG ─────────────────────
            info  = self._temperature_step_mg(
                dt         = float(dt_use),
                tol        = tol_mg,
                max_cycles = max_mg_cycles,
            )
            mg_it  = info.get("it",      max_mg_cycles)
            mg_rel = info.get("res_rel", float("inf"))
            lin_conv = info.get("converged", False)

            mg_diverged = info.get("diverged", False)
            if mg_diverged:
                if hasattr(self, '_mg_solver') and self._mg_solver._ready:
                    self._mg_solver._levels[0]['x'].zero_()
                dt_use = max(float(dt_scale_min), dt_backoff_factor * dt_use)
                bar.set_postfix({"BACKOFF": f"dt→{dt_use:.1f}", "mrel": f"{mg_rel:.2e}"})
                continue

            # Physical divergence guard (belt-and-suspenders)
            Tmax_new = float(self.T.max())
            if not torch.isfinite(self.T).all() or Tmax_new > 1e6:
                self.T = Tprev
                dt_use = max(float(dt_scale_min), dt_backoff_factor * dt_use)
                bar.set_postfix({
                    "BACKOFF": f"dt→{dt_use:.1f}",
                    "Tmax":    f"{Tmax_new:.2e}",
                })
                continue

            # ── dt ramping: increase when MG converged quickly ─────────────
            if mg_it < dt_ramp_threshold * max_mg_cycles:
                dt_use = min(float(dt_scale_max), dt_ramp_factor * dt_use)

            self._apply_temperature_bcs()

            dT       = (self.T - Tprev).abs()
            dT_s_inf = float(dT[solid].max()) if solid.any() else 0.0
            dT_f_inf = float(dT[fluid].max()) if fluid.any() else 0.0

            Tout_mean        = _Tout()
            #Tout_mean = float(To[f_out].mean()) if f_out.any() else float(To.mean())
            dTout     = float("inf") if Tout_prev is None else abs(Tout_mean - Tout_prev)
            Tout_prev = Tout_mean

            bar.set_postfix({
                "dT_s":  f"{dT_s_inf:.2e}",
                "dT_f":  f"{dT_f_inf:.2e}",
                "dTout": f"{dTout:.2e}",
                "mrel":  f"{mg_rel:.2e}",
                "mit":   mg_it,
                "Tmax":  f"{Tmax_new:.1f}",
                "dt":    f"{dt_use:.1f}",
                "stable": stable,
            })

            if (dT_s_inf < tol_dT_solid
                    and dT_f_inf < tol_dT_fluid
                    and dTout < tol_dTout_mean):
                stable += 1
                if stable >= stable_steps:
                    bar.close()
                    print(f"[THERMAL] converged at outer={outer+1}, "
                        f"Tmax={Tmax_new:.2f}°C, Tout={Tout_mean:.2f}°C")
                    return {
                        "outer_iters": outer + 1,
                        "Tmax":        Tmax_new,
                        "Tout_mean":   Tout_mean,
                        "dT_s_inf":    dT_s_inf,
                        "dT_f_inf":    dT_f_inf,
                        "dTout":       dTout,
                    }

        bar.close()
        print(f"[THERMAL] max_outer reached. Tmax={float(self.T.max()):.2f}°C")
        return {
            "outer_iters": max_outer,
            "Tmax":        float(self.T.max()),
            "Tout_mean":   Tout_mean,
            "dT_s_inf":    dT_s_inf,
            "dT_f_inf":    dT_f_inf,
            "dTout":       dTout,
        }

    def _temperature_step_mg(
        self,
        dt:         float,
        tol:        float = 1e-3,
        max_cycles: int   = 20,
    ) -> dict:
        """
        Advance temperature by one pseudo-time step dt using the standalone
        N-level geometric multigrid solver.

        Solves:  A theta = rhs,   A = I + dt * L_steady
                theta = T_new - T0  (T0 = Dirichlet lifting field)

        dt is the raw outer-loop pseudo-time step.
        s = thermal_dt_scale is baked into the MG coefficients at build time.
        src_step is in [K / thermal-step] (as returned by _compute_heat_source).
        """
        dt    = float(dt)
        T_old = self.T

        update_mask = self._build_update_mask_thermal()
        T0          = self._build_T_dirichlet_field()

        # src_vol = self._compute_heat_source()
        # if src_vol is None:
        #     src_vol = torch.zeros_like(T_old)

        # # Surface flux BCs (new — sums all registered SurfaceFluxBC objects)
        # src_surf = self._compute_surface_flux_sources()

        # # Combined source term
        # src_step = src_vol + src_surf

        # src_vol  = self._compute_heat_source()        # existing cfg-based source
        src_surf = self._compute_surface_flux_sources()
        src_vol = self._compute_volumetric_heat_sources()  # ← new
        src_step = (src_vol if src_vol is not None else torch.zeros_like(T_old)) + src_surf

        # ── create MG solver once ─────────────────────────────────────────────
        # self.ax and self.sign must be set before this is called.
        # solve_thermal_steady_only sets them at its top.
        if not hasattr(self, '_mg_solver'):
            self._mg_solver = NLevelGeometricMGSolver(
                device           = T_old.device,
                dtype            = T_old.dtype,
                flow_axis        = self.ax,
                flow_sign        = self.sign,
                thermal_dt_scale = float(self.thermal_dt_scale),
                omega            = float(getattr(self.cfg, 'mg_omega',        0.8)),
                n_pre            = int  (getattr(self.cfg, 'mg_pre_smooth',   2)),
                n_post           = int  (getattr(self.cfg, 'mg_post_smooth',  2)),
                n_coarse         = int  (getattr(self.cfg, 'mg_coarse_iters', 20)),
                min_coarse_cells = 4,
            )
            #patch_mg_pcg(self._mg_solver)            # ← add

        # u/v/wv: frozen after flow convergence; hierarchy rebuilt only on shape
        # change.  inv_diagM updated automatically when dt changes.
        self._mg_solver.build_if_needed(
            k      = self.k,
            rho_cp = self.rho_cp,
            u      = self.u,
            v      = self.v,
            w      = self.wv,      # field name in LBMCHT3D_Torch is wv not w
            mask   = update_mask,
            dt     = dt,
        )

        # lv0 = self._mg_solver._levels[0]
        # print(f"[DIAG] s={self._mg_solver.s:.4f}")
        # print(f"[DIAG] diagL max={float(lv0['diagL'].max()):.4e}  mean={float(lv0['diagL'].mean()):.4e}")
        # print(f"[DIAG] src_step max={float(src_step.abs().max()):.4e}")
        # print(f"[DIAG] inv_diagM max={float(lv0['inv_diagM'].max()):.4e}")
        # print(f"[DIAG] rho_cp min={float(self.rho_cp.min()):.4e}  max={float(self.rho_cp.max()):.4e}")

        # ── RHS:  rhs = (T_old + dt*src) - A(T0) ─────────────────────────────
        # A(T0) = T0 + dt * L_steady(T0)
        # L_apply_fine uses the same coefficients as the MG solve — no mismatch.
        LT0  = self._mg_solver.L_apply_fine(T0)
        rhs  = (T_old + dt * src_step) - (T0 + dt * LT0)
        rhs  = rhs.clone()
        rhs.mul_(update_mask.to(T_old.dtype))   # zero Dirichlet entries

        # ── warm-start from previous theta ────────────────────────────────────
        theta0 = (T_old - T0).clone()
        theta0.mul_(update_mask.to(T_old.dtype))

        # ── MG solve ──────────────────────────────────────────────────────────
        theta, mg_info = self._mg_solver.solve(
            b          = rhs,
            x0         = theta0,
            max_cycles = max_cycles,
            tol        = tol,
        )

        # ── SAFETY: reject solution if it diverged or produced unphysical values ──
        T_candidate = (T0 + theta).contiguous()
        T_cand_max  = float(T_candidate.max())
        T_cand_min  = float(T_candidate.min())
        Tin         = float(self.fluid.tin_C)
        physics_ok  = (
            torch.isfinite(T_candidate).all()
            and T_cand_max < Tin + 2000.0   # no more than 2000°C above inlet — reject otherwise
            and T_cand_min > Tin - 500.0    # no unphysical cold spots
        )

        if not physics_ok:
            # Do NOT update self.T — return with converged=False so outer loop backs off
            return {
                "converged": False,
                "it":        mg_info["it"],
                "res_rel":   mg_info["res_rel"],
                "r_rel":     float("inf"),
                "r_inf":     float("inf"),
                "src_inf":   float("nan"),
                "Tmax":      T_cand_max,
            }

        theta.mul_(update_mask.to(T_old.dtype))
        T_new = T_candidate
        self._apply_temperature_bcs_inplace(T_new)
        self.T = T_new

        # ── steady-state residual: L(T) → src/dt as T → T_ss ─────────────────
        with torch.no_grad():
            LT     = self._mg_solver.L_apply_fine(self.T)
            src_ss = src_step / dt
            r_ss   = LT - src_ss
            r_ss.mul_(update_mask.to(T_old.dtype))
            r_inf   = float(r_ss.abs().max())
            src_inf = float(src_ss[update_mask].abs().max()) if update_mask.any() else 1e-30
            r_rel   = r_inf / max(src_inf, 1e-30)

        return {
            "converged": mg_info["converged"],
            "it":        mg_info["it"],
            "res_rel":   mg_info["res_rel"],
            "r_rel":     r_rel,
            "r_inf":     r_inf,
            "src_inf":   src_inf,
            "Tmax":      float(self.T.max().detach().cpu().item()),
        }

    # ---------------------------
    # Snapshot storage (ONE file)
    # ---------------------------
    def _alloc_snapshot_buffers(self, nsteps: int):
        if self.cfg.snap_stride <= 0:
            self._snapshots = None
            return

        nsnaps = (nsteps // self.cfg.snap_stride) + 1
        nx, ny, nz = self.nx, self.ny, self.nz

        snaps = {}
        for f in self.cfg.store_fields:
            if f == "solid":
                snaps["solid"] = np.zeros((1, nx, ny, nz), dtype=np.uint8)
            else:
                snaps[f] = np.zeros((nsnaps, nx, ny, nz), dtype=np.float32 if self.dtype == torch.float32 else np.float64)

        snaps["_iters"] = np.zeros((nsnaps,), dtype=np.int64)
        self._snapshots = snaps
        self._snap_count = 0

        if "solid" in snaps:
            snaps["solid"][0] = self.solid.detach().cpu().numpy().astype(np.uint8)
        snaps["fluid_k0"] = np.array([getattr(self.cfg, 'fluid_k0', 0)], dtype=np.int64)

        # per-body masks (stored once, not per-frame)
        if hasattr(self, '_assembly') and hasattr(self._assembly, '_body_masks'):
            body_names = list(self._assembly._body_masks.keys())
            snaps["body_names"] = np.array(body_names)
            for name, mask in self._assembly._body_masks.items():
                snaps[f"solid_{name}"] = mask.cpu().numpy().astype(np.uint8)[np.newaxis]  # (1,nx,ny,nz)

        #print(f"[SNAP] allocating snapshot buffers: nsnaps={nsnaps} stride={self.cfg.snap_stride}")

    def _save_snapshot(self, it: int):
        if self._snapshots is None:
            return
        sidx = self._snap_count
        self._snapshots["_iters"][sidx] = it

        if "u" in self._snapshots:
            self._snapshots["u"][sidx] = self.u.detach().cpu().numpy()
        if "v" in self._snapshots:
            self._snapshots["v"][sidx] = self.v.detach().cpu().numpy()
        if "w" in self._snapshots:
            self._snapshots["w"][sidx] = self.wv.detach().cpu().numpy()
        if "speed" in self._snapshots:
            sp = torch.sqrt(self.u*self.u + self.v*self.v + self.wv*self.wv)
            self._snapshots["speed"][sidx] = sp.detach().cpu().numpy()
        if "rho" in self._snapshots:
            self._snapshots["rho"][sidx] = self.rho.detach().cpu().numpy()
        if "T" in self._snapshots:
            self._snapshots["T"][sidx] = self.T.detach().cpu().numpy()

        self._snap_count += 1

    def _flush_snapshots(self):
        if self._snapshots is None:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        out_path = os.path.join(self.cfg.out_dir, self.cfg.snapshot_file)

        ns = self._snap_count
        payload = {}
        for k, v in self._snapshots.items():
            if k == "solid":
                payload[k] = v
            elif k == "fluid_k0" or k == "body_names" or k.startswith("solid_"):
                payload[k] = v                 
            else:
                payload[k] = v[:ns]

        np.savez_compressed(out_path, **payload)
        iters = payload.get("_iters", None)
        # if iters is not None:
        #     print(f"[SNAP] iters={iters.tolist()}")  # NEW
        #print(f"[SNAP] wrote: {out_path} (frames={ns})")        

    @staticmethod
    def view_snapshots_pyvista_3d_single_npz(
        npz_path: str,
        fields: Optional[List[str]] = None,
        initial_field: str = "T",
        use_global_clim: bool = True,
        enable_slice: bool = True,
        slice_opacity: float = 1.0,
        enable_streamlines: bool = True,
        seed_mode: str = "inlet_plane",
        n_seeds: int = 150,
        inlet_x: int = 2,
        line_x: Tuple[float, float] = (5.0, 5.0),
        line_y: Tuple[float, float] = (0.0, 79.0),
        line_z: Tuple[float, float] = (0.0, 79.0),
        sphere_center: Tuple[float, float, float] = (10.0, 40.0, 40.0),
        sphere_radius: float = 10.0,
        streamline_max_time: float = 10000.0,
        streamline_initial_step: float = 1.5,
        flow_dir: Optional[str] = None,
        cmap = "turbo",
    ):
        import warnings

        #pv.global_theme.multi_samples = 8  # try 4, 8, or 16
        try:
            warnings.filterwarnings("ignore", category=pv.PyVistaDeprecationWarning)
        except Exception:
            pass

        if not os.path.isfile(npz_path):
            raise FileNotFoundError(npz_path)

        data = np.load(npz_path, allow_pickle=True)

        fields = [k for k in ("T", "rho", "speed") if k in data.files]
        if not fields:
            raise KeyError(f"None of T/rho/speed in {npz_path}")
        if initial_field not in fields:
            initial_field = fields[0]

        arr0 = data[initial_field]
        if arr0.ndim != 4:
            raise ValueError(f"Expected (nt,nx,ny,nz), got {arr0.shape}")
        nt, nx, ny, nz = arr0.shape

        flow_dir_use = "+X" if flow_dir is None else str(flow_dir).upper().strip()
        if flow_dir_use not in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            raise ValueError(f"flow_dir must be +X/-X/+Y/-Y/+Z/-Z; got '{flow_dir_use}'")
        sign = +1 if flow_dir_use[0] == "+" else -1
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir_use[1]]

        if "solid" not in data.files:
            raise KeyError(f"'solid' not in {npz_path}")
        s = data["solid"]
        solid_mask = (s[0] if s.ndim == 4 else s).astype(bool)
        if solid_mask.shape != (nx, ny, nz):
            raise ValueError("solid shape mismatch")
        solid_flat = solid_mask.reshape(-1, order="F")
        fluid_flat = ~solid_flat

        fluid_k0 = int(data["fluid_k0"]) if "fluid_k0" in data.files else 0
        if fluid_k0 > 0:
            solid_mask[:, :, :fluid_k0] = False   # blank out the walled-off zone
            solid_flat = solid_mask.reshape(-1, order="F")
            fluid_flat = ~solid_flat

        # body_names = list(data["body_names"]) if "body_names" in data.files else []
        # body_masks = {}
        # for name in body_names:
        #     key = f"solid_{name}"
        #     if key in data.files:
        #         m = data[key].astype(bool)
        #         if fluid_k0 > 0:
        #             m[:, :, :fluid_k0] = False
        #         body_masks[name] = m.reshape(-1, order="F")

        body_names = list(data["body_names"]) if "body_names" in data.files else []
        body_masks = {}
        for name in body_names:
            key = f"solid_{name}"
            if key in data.files:
                m = data[key].astype(bool)
                if fluid_k0 > 0:
                    m[:, :, :fluid_k0] = False
                body_masks[name] = m.reshape(-1, order="F")

        # ── Rebuild solid_flat from body masks only (excludes pad/buffer cells) ──
        # After the body_masks rebuild block:
        if body_masks:
            solid_flat_bodies = np.zeros(nx * ny * nz, dtype=bool)
            for m in body_masks.values():
                solid_flat_bodies |= m
            solid_mask = solid_flat_bodies.reshape((nx, ny, nz), order="F")
            solid_flat = solid_flat_bodies
            fluid_flat = ~solid_flat

        # ── Exclude sub-fluid zone from fluid rendering ──────────────────────────
        subfluid_flat = np.zeros(nx * ny * nz, dtype=bool)
        if fluid_k0 > 0:
            sub = np.zeros((nx, ny, nz), dtype=bool)
            sub[:, :, :fluid_k0] = True
            subfluid_flat = sub.reshape(-1, order="F")
            fluid_flat = fluid_flat & ~subfluid_flat   # fluid only above fluid_k0
        # else: keep original solid_flat from data["solid"] as fallback
    
        # Per-frame range from solid-only or fluid-only cells
        def _range_for_frame(field, frame_idx, fluid_mode):
            arr = data[field][frame_idx].reshape(-1, order="F").astype(np.float32)
            v = arr[fluid_flat] if fluid_mode else arr[solid_flat]
            v = v[np.isfinite(v)]
            if v.size == 0:
                return (0.0, 1.0)
            return (float(v.min()), float(v.max()))

        # Grids
        grid_pts = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1,1,1), origin=(0,0,0))
        scal0 = data[initial_field][0].reshape(-1, order="F").astype(np.float32)
        grid_pts.point_data["scalars"] = scal0.copy()
        grid_pts.set_active_scalars("scalars")

        grid_fluid_vol = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1,1,1), origin=(0,0,0))
        grid_fluid_vol.point_data["scalars"] = scal0.copy()
        grid_fluid_vol.set_active_scalars("scalars")

        grid_vec = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1,1,1), origin=(0,0,0))
        has_vec = all(k in data.files for k in ("u", "v", "w"))
        if has_vec:
            u0 = data["u"][0].reshape(-1, order="F")
            v0 = data["v"][0].reshape(-1, order="F")
            w0 = data["w"][0].reshape(-1, order="F")
            grid_vec.point_data["U"] = np.stack([u0, v0, w0], axis=1).copy()
            grid_vec.set_active_vectors("U")

        mask_grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1,1,1), origin=(0,0,0))
        mask_grid.point_data["mask"] = solid_flat.astype(np.uint8)
        solid_surface = mask_grid.contour([0.5], scalars="mask")

        state = {
            "field":      initial_field,
            "frame":      0,
            "fluid_mode": False,
            "stream_on":  bool(enable_streamlines and has_vec),
            "seed_mode":  seed_mode,
            "n_seeds":    int(n_seeds),
            "seed_cache": None,
            "seed_key":   None,
            "body_vis": {name: True for name in body_names},
        }
        rng = np.random.default_rng(0)

        pl = pv.Plotter(window_size=(1400, 900))
        #pl.enable_anti_aliasing("msaa")

        # try:
        #     pl.renderer.SetUseFXAA(False)  # prevent post-process blur
        # except Exception:
        #     pass
        
        box = pv.Box(bounds=(0, nx, 0, ny, 0, nz))
        #pl.add_mesh(box, style="wireframe", color="gray", line_width=2)

        actors = {"main": None, "stream": None, "main_sbar": None}
        for name in body_names:
            actors[f"body_{name}"] = None      

        # ── Named scalar bar keys (used for reliable remove-by-name) ─────────
        _FIELD_BAR  = "__field_bar__"
        _STREAM_BAR = "__stream_bar__"

        def _remove_sbar(name):
            """Remove a named scalar bar safely."""
            try:
                pl.remove_scalar_bar(name)
            except Exception:
                pass

        def _add_field_sbar(mapper, label, fmin, fmax):
            try:
                pl.remove_scalar_bar("Field")   # always same key → always found
            except Exception:
                pass
            pl.add_scalar_bar(
                title="Field",                  # fixed key, never changes
                mapper=mapper,
                n_labels=5,
                fmt="%.4g",
                vertical=True,
                position_x=0.91,
                position_y=0.10,
                width=0.04,
                height=0.75,
                title_font_size=1,
                label_font_size=18,
            )

        # Grid update helpers
        def _update_pts_scalars(field, frame_idx):
            flat = data[field][frame_idx].reshape(-1, order="F").astype(np.float32)
            grid_pts.point_data["scalars"][:] = flat
            try: grid_pts.GetPointData().GetArray("scalars").Modified()
            except Exception: pass
            grid_pts.Modified()


        def _update_fluid_vol_scalars(field, frame_idx, sentinel):
            flat = data[field][frame_idx].reshape(-1, order="F").astype(np.float32)
            out = grid_fluid_vol.point_data["scalars"]
            out[:] = flat
            out[solid_flat] = sentinel
            out[subfluid_flat] = sentinel       # ← add this line
            try: grid_fluid_vol.GetPointData().GetArray("scalars").Modified()
            except Exception: pass
            grid_fluid_vol.Modified()


        def _update_vectors(frame_idx):
            if not has_vec: return
            u = data["u"][frame_idx].reshape(-1, order="F")
            v = data["v"][frame_idx].reshape(-1, order="F")
            w = data["w"][frame_idx].reshape(-1, order="F")
            grid_vec.point_data["U"][:, 0] = u
            grid_vec.point_data["U"][:, 1] = v
            grid_vec.point_data["U"][:, 2] = w
            try: grid_vec.GetPointData().GetArray("U").Modified()
            except Exception: pass
            grid_vec.Modified()

        def _remove_actor(key):
            if actors[key] is not None:
                try: pl.remove_actor(actors[key])
                except Exception: pass
                actors[key] = None

        # Solid actor
        def _get_active_solid_flat():
            if not body_masks:
                return solid_flat
            active = np.zeros(len(solid_flat), dtype=bool)
            for name, mask in body_masks.items():
                if state["body_vis"].get(name, True):
                    active |= mask
            return active

        def _rebuild_solid_bodies(fmin, fmax):
            # remove all body actors
            for name in body_masks:
                _remove_actor(f"body_{name}")
            _remove_actor("main")  # fallback single-solid path
            
            if not body_masks:
                # fallback: original single solid
                _update_pts_scalars(state["field"], state["frame"])
                surf = solid_surface.sample(grid_pts)
                actors["main"] = pl.add_mesh(surf, scalars="scalars",
                    clim=(fmin, fmax), cmap=cmap, opacity=1.0, show_scalar_bar=False)
                _add_field_sbar(actors["main"].mapper, state["field"], fmin, fmax)
                return

            last_actor = None
            for name, mask in body_masks.items():
                if not state["body_vis"].get(name, True):
                    continue
                _update_pts_scalars(state["field"], state["frame"])
                body_grid = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
                body_grid.point_data["mask"] = mask.astype(np.uint8)
                body_surf = body_grid.contour([0.5], scalars="mask").sample(grid_pts)
                actors[f"body_{name}"] = pl.add_mesh(body_surf, scalars="scalars",
                    clim=(fmin, fmax), cmap=cmap, opacity=1.0, show_scalar_bar=False)
                last_actor = actors[f"body_{name}"]
            
            if last_actor is not None:
                _add_field_sbar(last_actor.mapper, state["field"], fmin, fmax)

        def _solid_clim():
            # clim from currently visible solid cells only
            active = _get_active_solid_flat()
            flat = data[state["field"]][state["frame"]].reshape(-1, order="F").astype(np.float32)
            v = flat[active & np.isfinite(flat)]
            if v.size == 0:
                return (0.0, 1.0)
            return (float(v.min()), float(v.max()))
        
        # Fluid volume actor
        def _opacity_tf(n=256):
            op = np.ones(n, dtype=np.float32)
            op[0] = 0.5
            return op

        def _rebuild_fluid(fmin, fmax):
            if actors.get("main_sbar") is not None:
                try: pl.remove_actor(actors["main_sbar"])
                except Exception: pass
                actors["main_sbar"] = None

            _remove_actor("main")
            sentinel = fmin - max(1.0, 0.01 * abs(fmax - fmin))
            _update_fluid_vol_scalars(state["field"], state["frame"], sentinel)
            vol = pl.add_volume(
                grid_fluid_vol, scalars="scalars",
                clim=(fmin, fmax), cmap=cmap,
                opacity=_opacity_tf(),
                show_scalar_bar=False,   # add manually — add_volume ignores scalar_bar_args
            )
            
            try:
                m = vol.mapper
                # stop VTK from auto-changing sample distance to meet FPS targets
                m.SetAutoAdjustSampleDistances(False)
            except Exception:
                pass

            # choose ONE of these depending on what your mapper exposes
            try:
                # smaller = better quality, slower; start with 0.5 then 0.25
                vol.mapper.SetSampleDistance(0.5)
            except Exception:
                pass

            try:
                # some VTK builds use ImageSampleDistance instead
                vol.mapper.SetImageSampleDistance(0.5)
            except Exception:
                pass           
            
            try:
                lut = vol.mapper.lookup_table
                lut.scalar_range = (fmin, fmax)
                lut.use_below_range_color = True
                lut.below_range_color = (0.0, 0.0, 0.0, 0.0)
            except Exception: pass
            try: vol.GetProperty().SetScalarOpacityUnitDistance(0.25)
            except Exception:
                try: vol.prop.SetScalarOpacityUnitDistance(0.25)
                except Exception: pass
            actors["main"] = vol
            #_add_field_sbar(vol.mapper, state["field"], fmin, fmax)

            try:
                pl.remove_scalar_bar("Field")
            except Exception:
                pass
            _dummy = pv.Sphere(radius=0.0001, center=(-99999, -99999, -99999))
            _dummy["_v"] = np.linspace(fmin, fmax, _dummy.n_points, dtype=np.float32)
            actors["main_sbar"] = pl.add_mesh(
                _dummy, scalars="_v", clim=(fmin, fmax), cmap=cmap,
                opacity=0.0, show_scalar_bar=True,
                scalar_bar_args=dict(
                    title="Field", n_labels=5, fmt="%.4g",
                    vertical=True, position_x=0.91, position_y=0.10,
                    width=0.04, height=0.75,
                    title_font_size=1, label_font_size=18,
                ),
            )

        # Streamlines
        def _seed_points(mode, n):
            # if mode == "inlet_plane":
            #     max_i = [nx-1, ny-1, nz-1][axis]
            #     idx = int(np.clip(inlet_x, 0, max_i))
            #     if axis == 0:
            #         yy = rng.integers(0, ny, n); zz = rng.integers(0, nz, n)
            #         return np.stack([np.full(n, idx), yy, zz], 1).astype(np.float32)
            #     elif axis == 1:
            #         xx = rng.integers(0, nx, n); zz = rng.integers(0, nz, n)
            #         return np.stack([xx, np.full(n, idx), zz], 1).astype(np.float32)
            #     else:
            #         xx = rng.integers(0, nx, n); yy = rng.integers(0, ny, n)
            #         return np.stack([xx, yy, np.full(n, idx)], 1).astype(np.float32)
            if mode == "inlet_plane":
                max_i = [nx-1, ny-1, nz-1][axis]
                idx = int(np.clip(inlet_x, 0, max_i))
                if axis == 0:
                    plane_fluid = ~solid_mask[idx, :, :]   # (ny, nz)
                elif axis == 1:
                    plane_fluid = ~solid_mask[:, idx, :]   # (nx, nz)
                else:
                    plane_fluid = ~solid_mask[:, :, idx]   # (nx, ny)
                fluid_coords = np.argwhere(plane_fluid)
                chosen = fluid_coords[rng.integers(0, len(fluid_coords), n)]
                if axis == 0:
                    return np.stack([np.full(n, idx), chosen[:,0], chosen[:,1]], 1).astype(np.float32)
                elif axis == 1:
                    return np.stack([chosen[:,0], np.full(n, idx), chosen[:,1]], 1).astype(np.float32)
                else:
                    return np.stack([chosen[:,0], chosen[:,1], np.full(n, idx)], 1).astype(np.float32)            
            elif mode == "line":
                t = np.linspace(0, 1, n, dtype=np.float32)
                return np.stack([line_x[0]+(line_x[1]-line_x[0])*t,
                                  line_y[0]+(line_y[1]-line_y[0])*t,
                                  line_z[0]+(line_z[1]-line_z[0])*t], 1)
            elif mode == "sphere":
                cx, cy, cz = sphere_center; r = float(sphere_radius)
                u = rng.random(n).astype(np.float32)
                v = rng.random(n).astype(np.float32)
                w = rng.random(n).astype(np.float32)
                phi = np.arccos(2*v - 1); rr = r*(w**(1/3))
                return np.stack([cx+rr*np.sin(phi)*np.cos(2*np.pi*u),
                                  cy+rr*np.sin(phi)*np.sin(2*np.pi*u),
                                  cz+rr*np.cos(phi)], 1).astype(np.float32)
            raise ValueError(f"seed_mode: got {mode!r}")

        def _rebuild_streamlines():
            _remove_actor("stream")
            _remove_sbar(_STREAM_BAR)
            if not state["stream_on"] or not has_vec:
                return
            n = int(max(1, state["n_seeds"]))
            key = (state["seed_mode"], n, inlet_x, flow_dir_use)
            if state["seed_key"] != key or state["seed_cache"] is None:
                state["seed_cache"] = _seed_points(state["seed_mode"], n)
                state["seed_key"] = key
            seeds = pv.PolyData(state["seed_cache"])
            try:
                sl = grid_vec.streamlines_from_source(
                    seeds, vectors="U", integration_direction="forward",
                    max_time=float(streamline_max_time),
                    initial_step_length=float(streamline_initial_step),
                    terminal_speed=0.0, max_steps=100000, integrator_type=45,
                )
            except TypeError:
                sl = grid_vec.streamlines_from_source(
                    seeds, vectors="U", integration_direction="forward",
                    max_length=float(streamline_max_time),
                    initial_step_length=float(streamline_initial_step),
                    terminal_speed=0.0, max_steps=100000, integrator_type=45,
                )
            if sl.n_points == 0:
                return
            if "U" in sl.point_data and sl.point_data["U"].ndim == 2:
                spd = np.linalg.norm(sl.point_data["U"], axis=1).astype(np.float32)
                sl["StreamSpeed"] = spd
                actors["stream"] = pl.add_mesh(
                    sl, scalars="StreamSpeed", cmap="turbo", line_width=2,
                    clim=(float(spd.min()), float(spd.max()) + 1e-10),
                    show_scalar_bar=False,
                )
                pl.add_scalar_bar(
                    title=_STREAM_BAR,
                    mapper=actors["stream"].mapper,
                    n_labels=4, fmt="%.3g", vertical=True,
                    position_x=0.84, position_y=0.10,
                    width=0.04, height=0.40,
                    title_font_size=1,
                    label_font_size=18,
                )
            else:
                actors["stream"] = pl.add_mesh(sl, color="white", line_width=2)

        # Master refresh
        def refresh(rebuild_stream=False, rebuild_main=False):
            if has_vec:
                _update_vectors(state["frame"])
            fmin, fmax = _range_for_frame(state["field"], state["frame"], state["fluid_mode"])
            if rebuild_main:
                if state["fluid_mode"]:
                    _rebuild_fluid(fmin, fmax)
                else:
                    _rebuild_solid_bodies(*_solid_clim())
            if rebuild_stream:
                _rebuild_streamlines()
            pl.render()

        # Sliders
        if nt > 1:
            def set_frame(val):
                state["frame"] = max(0, min(nt-1, int(round(val))))
                refresh(rebuild_stream=True, rebuild_main=True)
            pl.add_slider_widget(set_frame, rng=[0, nt-1], value=0, title="frame",
                                 pointa=(0.25, 0.04), pointb=(0.75, 0.04))

        seed_modes = ["inlet_plane", "line", "sphere"]

        def _set_seed_mode(val):
            state["seed_mode"] = seed_modes[int(round(val))]
            state["seed_cache"] = None; state["seed_key"] = None
            if state["stream_on"]:
                _rebuild_streamlines(); pl.render()

        pl.add_slider_widget(_set_seed_mode, rng=[0, 2],
                            value=seed_modes.index(state["seed_mode"]),
                            title="", pointa=(0.02, 0.67), pointb=(0.15, 0.67))
        pl.add_text("SEED MODE", position=(0.02, 0.75), font_size=14, viewport=True)

        def _set_nseeds(val):
            state["n_seeds"] = int(max(1, round(val)))
            state["seed_cache"] = None; state["seed_key"] = None
            if state["stream_on"]:
                _rebuild_streamlines(); pl.render()

        pl.add_slider_widget(_set_nseeds, rng=[10, 2000], value=int(n_seeds),
                            title="", pointa=(0.02, 0.52), pointb=(0.15, 0.52))
        pl.add_text("N SEEDS", position=(0.02, 0.60), font_size=14, viewport=True)

        # Checkboxes
        x0, y0, bs, rh = 12, 12, 18, 26
        row = 0

        # Bottom row: body checkboxes horizontal, show fluid at right
        cb_y    = 12          # y position in pixels
        cb_size = 18
        cb_gap  = 90          # pixels per body slot (size + label)
        x_start = 12

        body_cb_widgets = {}

        def _make_body_cb(name):
            def _cb(val):
                state["body_vis"][name] = bool(val)
                if not state["fluid_mode"]:
                    _rebuild_solid_bodies(*_solid_clim())
                    pl.render()
            return _cb

        for idx, name in enumerate(body_names):
            bx = x_start + idx * cb_gap
            w = pl.add_checkbox_button_widget(_make_body_cb(name), value=True,
                                            position=(bx, cb_y), size=cb_size)
            pl.add_text(name, position=(bx + cb_size + 10, cb_y+5), font_size=9)
            body_cb_widgets[name] = w

        # "show fluid" rightmost
        fluid_x = x_start + len(body_names) * cb_gap + 50

        def _toggle_fluid(val):
            state["fluid_mode"] = bool(val)
            if val:
                # hide all solid bodies
                for name in body_masks:
                    _remove_actor(f"body_{name}")
                _remove_actor("main")
                fmin, fmax = _range_for_frame(state["field"], state["frame"], True)
                _rebuild_fluid(fmin, fmax)
            else:
                _remove_actor("main")  # remove volume actor
                _rebuild_solid_bodies(*_solid_clim())
            refresh(rebuild_stream=True, rebuild_main=False)

        pl.add_checkbox_button_widget(_toggle_fluid, value=False,
                                    position=(fluid_x, cb_y), size=cb_size)
        pl.add_text("show fluid", position=(fluid_x + cb_size + 4, cb_y+ 10), font_size=9)
        row += 1

        if has_vec:
            def _toggle_stream(val):
                state["stream_on"] = bool(val)
                if val:
                    _rebuild_streamlines()
                else:
                    _remove_actor("stream")
                    _remove_sbar(_STREAM_BAR)
                    try: pl.remove_actor("__stream_label__")
                    except Exception: pass
                pl.render()
            pl.add_checkbox_button_widget(_toggle_stream, value=state["stream_on"],
                                          position=(x0, y0+row*rh), size=bs)
            pl.add_text("streamlines", position=(x0+bs+4, y0+row*rh), font_size=10)
            row += 1
        else:
            pl.add_text("streamlines disabled", position=(x0, y0+row*rh), font_size=10)
            row += 1

        pl.add_text("fields:", position=(x0, y0+row*rh), font_size=10)
        row += 1

        field_widgets = {}

        def _make_field_cb(fld):
            def _cb(val):
                if val:
                    for k, w in field_widgets.items():
                        if k != fld:
                            try: w.SetValue(False)
                            except Exception: pass
                    state["field"] = fld
                    refresh(rebuild_stream=False, rebuild_main=True)
            return _cb

        for fld in fields:
            pl.add_text(fld, position=(x0+bs+4, y0+row*rh), font_size=10)
            w = pl.add_checkbox_button_widget(_make_field_cb(fld),
                                              value=(fld == initial_field),
                                              position=(x0, y0+row*rh), size=bs)
            field_widgets[fld] = w
            row += 1

        refresh(rebuild_stream=state["stream_on"], rebuild_main=True)
        pl.add_axes()
        pl.show()


if __name__ == "__main__":
    
    cfg = SimConfig3D(
        Lx_m=0.035, Ly_m=0.035, Lz_m=0.035,
        #Lx_m=5.0, Ly_m=1.0, Lz_m=1.0,
        flow_bc="inlet_outlet",
        flow_dir="-Z",
        transverse_walls=True,
        u_in_mps=1.0,            # set your actual inlet speed here
        dt_thermal_phys_s=1.0,   # << choose thermal pseudo-time (SI seconds per outer step)
        temp_bc="fixed",
        t_ambient_C=25.0,
        heating_mode="off",
        qdot_total_W=0.0,
        solid_init_mode = "ambient",   # "ambient" or "hot"
        T_hot_C = 0.0,
        collision    = 'mrt_smag',
        outlet_bc_mode  = "convective", 
        snapshot_file = "heatksinkTPMS.npz"              
    )


    cfg.domain_walls  = ["-Z"]   # solid walls 
    cfg.domain_outlets = ["+X","+Y","-X","-Y",]  # additional pressure outlets 
    #cfg.transverse_walls = False  


    water = FluidPropsSI(
        nu_m2_s=1.0e-6,
        rho_kg_m3=997.0,
        k_W_mK=0.6,
        cp_J_kgK=4182.0,
        tin_C=25.0,
    )

    air = FluidPropsSI(
        nu_m2_s=15.0e-6,
        rho_kg_m3=1.25,
        k_W_mK=0.024,
        cp_J_kgK=1006.0,
        tin_C=25.0,
    )    

    air1 = FluidPropsSI(
        nu_m2_s=0.02,
        rho_kg_m3=1.0,
        k_W_mK=1.0,
        cp_J_kgK=50.0,
        tin_C=20.0,
    )    


    aluminum = SolidPropsSI(
        k_W_mK=237.0,
        rho_kg_m3=2700.0,
        cp_J_kgK=960.0,
    )


    aluminum1 = SolidPropsSI(
        k_W_mK=5.0,
        rho_kg_m3=1.0,
        cp_J_kgK=80.0,
    )

    copper = SolidPropsSI(
        k_W_mK=401.0,
        rho_kg_m3=8960.0,
        cp_J_kgK=385.0,
    )

    # Thermal Interface Material (typical silicone-based TIM pad)
    tim = SolidPropsSI(
        k_W_mK    = 6.0,       # W/m·K  — mid-range TIM pad (e.g. Bergquist GP3000)
        rho_kg_m3 = 2500.0,    # kg/m³
        cp_J_kgK  = 800.0,     # J/kg·K
    )

    # Transformer core — grain-oriented silicon steel (power magnetics)
    transformer_core = SolidPropsSI(
        k_W_mK    = 25.0,      # W/m·K  — laminated SiFe, through-plane ~25
        rho_kg_m3 = 7650.0,    # kg/m³
        cp_J_kgK  = 490.0,     # J/kg·K
    )

    ferrite_core = SolidPropsSI(
        k_W_mK    = 4.0,       # W/m·K  — MnZn ferrite typical
        rho_kg_m3 = 4800.0,    # kg/m³
        cp_J_kgK  = 700.0,     # J/kg·K
    )

    # Compute pitch to match LBM dx exactly
    dx_mm = 0.2# in millimeters
    cfg.dx_mm = dx_mm
    cfg.nx = round(cfg.Lx_m * 1000 / dx_mm)
    cfg.ny = round(cfg.Ly_m * 1000 / dx_mm)
    cfg.nz = round(cfg.Lz_m * 1000 / dx_mm)

    #patch_stl_voxelizer(STLVoxelizer)
    
    assembly = SolidAssembly(
        bodies=[
            dict(stl="geometries/TPMS_heatsink1.stl", npz="TPMS_heatsink.npz",
                name="heatsink", build_dir="+Z", material=aluminum,
                color="steelblue", role="fluid_base"),
            # dict(stl="geometries/Coldspray-heatsink/CuSpray.stl",  npz="CuSpray.npz",
            #     name="CuSpray",  build_dir="+Y", material=copper,
            #     color="darkorange", role="stack_below"),
            # dict(stl="geometries/Coldspray-heatsink/TIM.stl",  npz="TIM.npz",
            #     name="TIM",  build_dir="+Y", material=tim,
            #     color="gold", role="stack_below"),
            # dict(stl="geometries/Coldspray-heatsink/heatsource.stl",  npz="heatsource.npz",
            #     name="heatsource",  build_dir="+Y", material=transformer_core,
            #     color="sienna", role="stack_below"),                                
        ],
        fluid_Lz_m       = cfg.Lz_m,
        dx_mm            = cfg.dx_mm,
        cfg              = cfg,
        flow_dir   = cfg.flow_dir,    # or "+X", "-X", "-Y", "+Z", "-Z"
        j0_divisor = 2.5,
        STLVoxelizer_cls = STLVoxelizer,
        LBMCHT_cls       = LBMCHT3D_Torch,
    )

    assembly.build_domain()
    assembly.print_summary()
    cfg.obstacle = None              # don't use the cube factory

    sim = LBMCHT3D_Torch(cfg, air, aluminum)

    # _max_scale = 10.0
    # if sim.thermal_dt_scale > _max_scale:
    #     sim.dt_thermal_phys_s = _max_scale * sim.dt_flow_phys_s
    #     sim.thermal_dt_scale  = _max_scale
    #     print(f"[WARN] thermal_dt_scale clamped to {_max_scale}. "
    #         f"dt_thermal_phys_s → {sim.dt_thermal_phys_s:.4e} s")

    _target_scale = 3.0   # conservative — MG works reliably up to ~5
    sim.dt_thermal_phys_s = _target_scale * sim.dt_flow_phys_s
    sim.thermal_dt_scale  = _target_scale
    print(f"[THERMAL] dt_thermal_phys_s={sim.dt_thermal_phys_s:.4e} s  "
        f"(thermal_dt_scale={_target_scale})")

    assembly.stamp(sim)                   # add_solid_body × N + finalize + _solid_all
    sim._assembly = assembly 
    
    print("\n")
    phi_gpu = build_phi_from_voxel_mask_mm(sim.solid.cpu().numpy(), cfg.dx_mm, "cuda")
    sim.phi        = phi_gpu
    sim.solid      = (phi_gpu < 0.0)
    assembly.restore_thin_bodies(sim)     # merge back CuSpray lost by SDF

    build_ibb_from_sdf_gpu(sim, 
                           voxel_origin_mm=assembly._geo_objs["heatsink"].voxel_origin_mm,
                            pitch_mm=cfg.dx_mm, 
                            offset_ijk=assembly.specs[0].offset_ijk, 
                            c_np=sim.c_np,
                            opp_np=sim.opp_np
    )
    
    assembly.wall_off_subfluid(sim) 

    sim._apply_transverse_flow_walls()
    print("\n")

    # bc_chip = VolumetricHeatBC(
    #     solid_mask = assembly.get_solid_mask("heatsource"),
    #     Q_watts    = 100.0,
    #     dx_mm      = cfg.dx_mm,
    # )
    
    # sim.add_volumetric_heat(bc_chip)

    bc = SurfaceFluxBC(
        solid_mask    = assembly.get_solid_mask("heatsink"),          # (nx, ny, nz) bool CUDA tensor
        axis          = "-Z",               # "+X"/"-X"/"+Y"/"-Y"/"+Z"/"-Z"
        surface_L_mm  = 15.0,               # length of the heat source surface [mm]
        surface_W_mm  = 15.0,               # width  of the heat source surface [mm]
        q_flux_W_m2   = 55000, #20.0 / (400.0 * 200.0 * 1e-6),   # 20W over chip footprint,           # heat flux [W/m²], positive = into solid101.0 * 50.0 * 1e-6
        dx_mm         = cfg.dx_mm,      # voxel pitch [mm]
        center_mm     = assembly.get_body_center_mm("heatsink"),               # (u_mm, v_mm) in-plane center; None = solid centroid
    )

    sim.add_surface_flux(bc)

    STLVoxelizer.visualize_domain(sim, solid_bodies=assembly.viz_list())

    #Stage A: flow to steady (auto-stop)
    FlowSolverUpgrade.patch(sim)          # one call patches sim in-place
    LBMCUDAUpgrade.patch(sim)
    print("\n")

    from pre_post.tracer_patch import attach_tracer
    tracer = attach_tracer(sim, flow_dir=cfg.flow_dir, )
    from save_transient_animation import save_transient_animation
    save_transient_animation(sim, tracer, cfg,
        n_frames=200,
        steps_per_frame=40,      # 5000 total steps, flow still developing
        out_path="snapshots/flow_anim.zarr",
        warmup_steps=0,
    )

    flow_info = sim.run_flow_to_steady_v2(tol_u_ema=5e-3)

    therm_info = sim.solve_thermal_steady_only(
        max_outer          = 50000,
        tol_dT_solid       = 1e-2,
        tol_dT_fluid       = 1e-2,
        tol_dTout_mean     = 5e-3,
        dt_scale_min       = 1.0,
        dt_scale_max       = 200.0,
        dt_scale_start     = 2.0,
        dt_ramp_factor     = 1.5,
        dt_backoff_factor  = 0.5,
        dt_ramp_threshold  = 0.8,
        max_mg_cycles      = 15,
        tol_mg             = 5e-3,
        stable_steps       = 8,
    )
    assembly.print_thermal_diagnostics(sim)

    # Save one snapshot
    sim._alloc_snapshot_buffers(nsteps=0)
    sim._save_snapshot(0)
    sim._flush_snapshots()


    LBMCHT3D_Torch.view_snapshots_pyvista_3d_single_npz(
        npz_path=os.path.join(cfg.out_dir, cfg.snapshot_file),
        fields=["T", "speed", "u", "v", "w", "rho"],
        initial_field="T",
        seed_mode = 'inlet_plane',
        use_global_clim=True,
        flow_dir=cfg.flow_dir,
        cmap = "turbo",
    )
