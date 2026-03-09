"""
solid_assembly.py
=================
Clean interface for assembling multiple solid bodies into an LBM-CHT simulation.

Usage
-----
    from solid_assembly import SolidAssembly

    assembly = SolidAssembly(
        bodies=[
            dict(stl="heatsink.stl",  npz="heatsink.npz",  name="heatsink",
                 build_dir="+Y", material=aluminum, color="steelblue",
                 role="fluid_base"),          # base face at fluid_k0
            dict(stl="CuSpray.stl",   npz="CuSpray.npz",   name="CuSpray",
                 build_dir="+Y", material=copper,  color="darkorange",
                 role="stack_below"),          # stacked below fluid_base body
        ],
        fluid_Lz_m = 0.060,
        dx_mm      = dx_mm,
        cfg        = cfg,
        STLVoxelizer_cls = STLVoxelizer,
    )

    assembly.build_domain()          # compute_domain_with_solid_stack + voxelize
    assembly.stamp(sim)              # add_solid_body for all, finalize
    viz_bodies = assembly.viz_list() # list of dicts for STLVoxelizer.visualize_domain
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any

from pre_post.face_roles import resolve_face_roles, _face_ax_idx_fsign


@dataclass
class BodySpec:
    stl:       str             # path to STL file
    npz:       str             # output npz path
    name:      str             # display name
    build_dir: str             # STLVoxelizer orientation e.g. "+Y"
    material:  Any             # SolidPropsSI instance
    color:     str = "steelblue"
    role:      str = "fluid_base"  # "fluid_base" | "stack_below" | "stack_above" (future)
    part_scale: float = 1.0

    # Populated after voxelization
    vox_shape:  tuple = field(default_factory=tuple, init=False)  # (Gx, Gy, Gz)
    solid_thick: int  = field(default=0, init=False)  # actual solid cells along Z
    solid_pad:   int  = field(default=0, init=False)  # padding below solid within grid
    offset_ijk:  tuple = field(default_factory=tuple, init=False)

    @property
    def Gx(self): return self.vox_shape[0]
    @property
    def Gy(self): return self.vox_shape[1]
    @property
    def Gz(self): return self.vox_shape[2]


class SolidAssembly:
    """
    Manages N solid bodies, handles:
      - STL voxelization (via patched STLVoxelizer)
      - Padding-aware Z offsets (no gaps, no overlaps)
      - Domain sizing (compute_domain_with_solid_stack)
      - XY centering per body
      - sim.add_solid_body calls + finalize
      - SDF merge of thin bodies
      - visualize_domain dict generation
    """

    def __init__(
        self,
        bodies:            List[dict],
        fluid_Lz_m:        float,
        dx_mm:             float,
        cfg,
        STLVoxelizer_cls,
        LBMCHT_cls         = None,   # LBMCHT3D_Torch class (for static helpers)
        xy_anchor:         str = "center",  # "center" | "origin"
        j0_divisor:        float = 3.5,    # domain Y offset fraction
        flow_dir:    str = "+Y",
    ):
        self.specs      = [BodySpec(**b) for b in bodies]
        self.fluid_Lz_m = fluid_Lz_m
        self.dx_mm      = dx_mm
        self.cfg        = cfg
        self.VoxCls     = STLVoxelizer_cls
        self.LBM        = LBMCHT_cls
        self.xy_anchor  = xy_anchor
        self.j0_div     = j0_divisor
        self.flow_dir = flow_dir.strip().upper()
        

        self._geo_objs  = {}   # name → STLVoxelizer instance
        self._built     = False

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: voxelize all STLs + compute domain
    # ─────────────────────────────────────────────────────────────────────────

    def get_body_center_mm(self, name: str) -> tuple:
        """Returns (center_u_mm, center_v_mm) in the two cross-Z axes."""
        sp   = next(s for s in self.specs if s.name == name)
        i0, j0, k0 = sp.offset_ijk
        vox  = np.load(sp.npz, allow_pickle=True)["voxel_solid"].astype(bool)
        xi   = np.where(vox)[0]; yi = np.where(vox)[1]
        cx   = (xi.min() + xi.max()) / 2.0
        cy   = (yi.min() + yi.max()) / 2.0
        return ((i0 + cx) * self.dx_mm, (j0 + cy) * self.dx_mm)

    def build_domain(self):
        """Voxelize all STLs, compute domain size, compute all offsets."""

        # 1a. Voxelize
        for sp in self.specs:
            geo = self.VoxCls(
                stl_path   = sp.stl,
                build_dir  = sp.build_dir,
                part_scale = sp.part_scale,
                pitch_mm   = self.dx_mm,
                out_npz    = sp.npz,
            )
            self._geo_objs[sp.name] = geo
            sp.vox_shape = geo.voxel_solid.shape
            sp.solid_thick, sp.solid_pad = self._solid_extent(sp.npz, axis=2)

        # 1b. Find base body (fluid_base) — sets XY reference
        base = self._base_body()

        # 1c. Total solid stack height = sum of Gz for all stack_below bodies
        stack_cells = sum(
            sp.solid_thick for sp in self.specs if sp.role == "stack_below"
        ) + 1   # 1 thermal buffer cell below bottom solid face
        solid_stack_m = stack_cells * self.dx_mm / 1000.0

        # 1d. Compute domain
        if self.LBM is not None:
            self.LBM.compute_domain_with_solid_stack(
                self.cfg,
                fluid_Lz_m       = self.fluid_Lz_m,
                solid_stack_Lz_m = solid_stack_m,
                dx_mm            = self.dx_mm,
            )
        else:
            # fallback: direct cfg mutation
            nz_solid = round(solid_stack_m * 1000.0 / self.dx_mm)
            nz_fluid = round(self.fluid_Lz_m * 1000.0 / self.dx_mm)
            self.cfg.fluid_k0 = nz_solid
            self.cfg.fluid_nz = nz_fluid
            self.cfg.nz       = nz_solid + nz_fluid

        # 1e. Compute per-body offsets
        self._compute_offsets(base)
        self._built = True

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: stamp into sim
    # ─────────────────────────────────────────────────────────────────────────

    def stamp(self, sim):
        assert self._built, "Call build_domain() before stamp()"
        
        # Capture per-body masks by diffing solid before/after each add_solid_body
        self._body_masks = {}
        
        for sp in self.specs:
            solid_before = sim.solid.clone() if sim.solid is not None else None
            sim.add_solid_body(
                npz_path    = sp.npz,
                offset_ijk  = sp.offset_ijk,
                solid_props = sp.material,
                name        = sp.name,
            )
            if solid_before is not None:
                self._body_masks[sp.name] = sim.solid & ~solid_before
            else:
                self._body_masks[sp.name] = sim.solid.clone()

        sim.finalize_solid_assembly()
        sim._solid_all = sim.solid.clone()

    def get_solid_mask(self, name: str):
        """Returns the per-body bool mask on the sim device."""
        assert hasattr(self, '_body_masks'), "Call stamp() first"
        assert name in self._body_masks, f"Unknown body '{name}'. Available: {list(self._body_masks)}"
        return self._body_masks[name]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: SDF merge (call after phi rebuild)
    # ─────────────────────────────────────────────────────────────────────────

    def restore_thin_bodies(self, sim):
        """
        After sim.solid = (phi < 0), restore cells lost by SDF for thin bodies.
        Call this immediately after setting sim.solid from phi.
        """
        if hasattr(sim, "_solid_all"):
            sim.solid    |= sim._solid_all
            sim.solid_flow = sim.solid.clone()

    def wall_off_subfluid_OLD(self, sim):
        k0 = self.cfg.fluid_k0
        if k0 <= 0:
            return

        flow_dir = getattr(self.cfg, 'flow_dir', '+Y').upper().strip()
        axis = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]
        sign = +1 if flow_dir[0] == '+' else -1

        if axis == 2:
            # Z is the flow axis — walling off z<k0 would seal the outlet (-Z)
            # or inlet (+Z) plane.  For Z-flow the solid stack must not block
            # the flow boundaries.  Wall off only X/Y perimeter instead.
            # The solid bodies themselves are already stamped; just leave z-faces open.
            return

        # Default: wall off everything below fluid_k0 in Z (transverse to X/Y flow)
        sim.solid[:, :, :k0]      = True
        sim.solid_flow[:, :, :k0] = True
        if hasattr(sim, '_solid_all'):
            sim._solid_all[:, :, :k0] = True

    def wall_off_subfluid(self, sim):
        """
        Walls off sub-fluid solid stack (z < fluid_k0) for flow solver.

        X/Y flow: always wall — solid stack is below the fluid slab, perpendicular
                to flow, never an inlet or outlet.
        Z flow:   only wall if the low-Z face is NOT an outlet. If -Z is the primary
                outlet (e.g. impingement from top), leaving sub-fluid region open
                lets flow exit through the sides naturally; the stamped solid bodies
                themselves block inappropriate fluid paths.
        """
        k0 = self.cfg.fluid_k0
        if k0 <= 0:
            return

        flow_dir = getattr(self.cfg, 'flow_dir', '+Y').upper().strip()
        axis     = {"X": 0, "Y": 1, "Z": 2}[flow_dir[1]]

        if axis == 2:
            roles = resolve_face_roles(self.cfg)
            # If either Z face is an outlet, don't seal — flow needs that path
            if roles.get("-Z") == "outlet" or roles.get("+Z") == "outlet":
                return

        sim.solid      [:, :, :k0] = True
        sim.solid_flow [:, :, :k0] = True
        if hasattr(sim, '_solid_all'):
            sim._solid_all[:, :, :k0] = True

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: visualize_domain dict
    # ─────────────────────────────────────────────────────────────────────────

    def viz_list(self) -> List[dict]:
        """Returns list of dicts for STLVoxelizer.visualize_domain."""
        assert self._built, "Call build_domain() before viz_list()"
        return [
            {"npz": sp.npz, "offset_ijk": sp.offset_ijk,
             "name": sp.name, "color": sp.color}
            for sp in self.specs
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Offset + alignment helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _base_body(self) -> BodySpec:
        bases = [sp for sp in self.specs if sp.role == "fluid_base"]
        assert len(bases) == 1, "Exactly one body must have role='fluid_base'"
        return bases[0]

    # def _compute_offsets(self, base: BodySpec):
    #     cfg = self.cfg
    #     nx, ny = cfg.nx, cfg.ny
    #     fluid_k0 = cfg.fluid_k0

    #     # Base body: XY centering from domain, Z at fluid_k0 (pad-corrected)
    #     i0_base = int((nx - base.Gx) // 2)
    #     j0_base = int((ny - base.Gy) // self.j0_div)
    #     k0_base = fluid_k0 - base.solid_pad # base body stamped so its grid bottom aligns with fluid_k0
    #     # NOTE: heatsink pad handled by the fact that its first solid cell
    #     # is at k0_base + hs_pad inside the domain — this is fine,
    #     # the solid sits just above fluid_k0 by pad cells (air gap < 2 cells)
    #     base.offset_ijk = (i0_base, j0_base, k0_base)

    #     # Stack-below bodies: stacked from fluid_k0 downward
    #     k_cursor = fluid_k0   # tracks next available k for stack_below

    #     for sp in self.specs:
    #         if sp.role != "stack_below":
    #             continue

    #         # XY: center within base body footprint
    #         i0 = i0_base + (base.Gx - sp.Gx) // 2
    #         j0 = j0_base + (base.Gy - sp.Gy) // 2

    #         # Z: shift grid so TOP of solid aligns with k_cursor
    #         # top of solid within grid = sp.solid_pad + sp.solid_thick - 1
    #         # we want domain k of that cell = k_cursor - 1
    #         # → offset_k = k_cursor - 1 - (sp.solid_pad + sp.solid_thick - 1)
    #         #             = k_cursor - sp.solid_pad - sp.solid_thick
    #         k0 = k_cursor - sp.solid_pad - sp.solid_thick
    #         sp.offset_ijk = (i0, j0, k0)

    #         # Next body stacks below this one's grid bottom
    #         k_cursor = k_cursor - sp.solid_thick 

    def _compute_offsets(self, base: BodySpec):
        cfg = self.cfg
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        fluid_k0 = cfg.fluid_k0

        sign = +1 if self.flow_dir[0] == "+" else -1
        fax  = {"X": 0, "Y": 1, "Z": 2}[self.flow_dir[1]]   # flow axis index

        domain_dims = [nx, ny, nz]
        base_dims   = [base.Gx, base.Gy, base.Gz]

        # Compute base XYZ offset
        # Flow axis: place near inlet (sign=+1 → low index, sign=-1 → high index)
        # Cross axes: center in domain
        base_offset = [0, 0, 0]
        for ax in range(3):
            if ax == 2:
                # Z always = fluid_k0 - pad (stacking direction)
                base_offset[2] = fluid_k0 - base.solid_pad
            elif ax == fax:
                # Flow axis: offset toward inlet
                gap = domain_dims[ax] - base_dims[ax]
                if sign == +1:
                    # +flow: inlet at low index → place body at 1/j0_div from low
                    base_offset[ax] = int(gap // self.j0_div)
                else:
                    # -flow: inlet at high index → place body at 1/j0_div from high
                    base_offset[ax] = int(gap - gap // self.j0_div)
            else:
                # Cross axis: center
                base_offset[ax] = int((domain_dims[ax] - base_dims[ax]) // 2)

        i0_base, j0_base, k0_base = base_offset
        base.offset_ijk = (i0_base, j0_base, k0_base)

        # Stack-below bodies
        k_cursor = fluid_k0
        for sp in self.specs:
            if sp.role != "stack_below":
                continue
            i0 = i0_base + (base.Gx - sp.Gx) // 2
            j0 = j0_base + (base.Gy - sp.Gy) // 2
            k0 = k_cursor - sp.solid_pad - sp.solid_thick
            sp.offset_ijk = (i0, j0, k0)
            k_cursor = k_cursor - sp.solid_thick

    @staticmethod
    def _solid_extent(npz_path: str, axis: int = 2):
        """Returns (solid_thickness, pad_below) along axis."""
        vox = np.load(npz_path, allow_pickle=True)["voxel_solid"].astype(bool)
        idx = np.where(vox)[axis]
        if idx.size == 0:
            return 0, 0
        lo = int(idx.min()); hi = int(idx.max())
        return (hi - lo + 1), lo

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def print_summary(self):
        print(f"\n{'─'*60}")
        print(f"{'SolidAssembly Summary':^60}")
        print(f"{'─'*60}")
        print(f"  fluid_k0 = {self.cfg.fluid_k0}   fluid_nz = {self.cfg.fluid_nz}   nz = {self.cfg.nz}")
        print(f"{'─'*60}")
        for sp in self.specs:
            print(f"  [{sp.role:12s}] {sp.name:20s} "
                  f"grid={sp.Gx}×{sp.Gy}×{sp.Gz}  "
                  f"solid_thick={sp.solid_thick}  pad={sp.solid_pad}  "
                  f"offset={sp.offset_ijk}")
        print(f"{'─'*60}\n")

    def print_thermal_diagnostics(self, sim):
        """Print Tmax/Tmin/Tmean for each solid body after thermal solve."""
        assert hasattr(self, '_body_masks'), "Call stamp() first"
        T = sim.T

        print(f"\n{'─'*65}")
        print(f"{'Thermal Diagnostics':^65}")
        print(f"{'─'*65}")
        print(f"  {'Body':<20} {'Tmin [°C]':>10} {'Tmean [°C]':>11} {'Tmax [°C]':>10} {'Nvox':>8}")
        print(f"{'─'*65}")

        for sp in self.specs:
            mask = self._body_masks.get(sp.name)
            if mask is None or not mask.any():
                print(f"  {sp.name:<20} {'—':>10} {'—':>11} {'—':>10} {'0':>8}")
                continue
            T_body = T[mask]
            tmin  = float(T_body.min())
            tmean = float(T_body.mean())
            tmax  = float(T_body.max())
            nvox  = int(mask.sum())
            print(f"  {sp.name:<20} {tmin:>10.2f} {tmean:>11.2f} {tmax:>10.2f} {nvox:>8,}")

        # Overall
        solid_mask = sim.solid
        fluid_mask = ~solid_mask
        T_solid = T[solid_mask]
        T_fluid = T[fluid_mask]
        print(f"{'─'*65}")
        print(f"  {'ALL SOLID':<20} {float(T_solid.min()):>10.2f} {float(T_solid.mean()):>11.2f} {float(T_solid.max()):>10.2f} {int(solid_mask.sum()):>8,}")
        print(f"  {'ALL FLUID':<20} {float(T_fluid.min()):>10.2f} {float(T_fluid.mean()):>11.2f} {float(T_fluid.max()):>10.2f} {int(fluid_mask.sum()):>8,}")
        print(f"{'─'*65}\n")