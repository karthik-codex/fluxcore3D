"""
cht_worker.py
Exposes run_simulation_blocking() (pure function, no Qt) used by both
SimWorker (GUI thread) and cht_run.py (CLI).
"""
import os, traceback
import numpy as np
from pathlib import Path
#from PyQt5.QtCore import QObject, pyqtSignal

from UI_components.cht_constants   import FLUID_PRESETS, SOLID_PRESETS
from UI_components.cht_sim_imports import (
    HAS_SIM,
    SimConfig3D, FluidPropsSI, SolidPropsSI,
    STLVoxelizer, LBMCHT3D_Torch,
    LBMCUDAUpgrade, FlowSolverUpgrade,
    SurfaceFluxBC, VolumetricHeatBC,
    build_ibb_from_sdf_gpu, build_phi_from_voxel_mask_mm, unpatch_stl_voxelizer,
    patch_stl_voxelizer, SolidAssembly,
)


def run_simulation_blocking(params, log_fn=print, progress_fn=None, abort_fn=None):
    """
    Run full CHT pipeline synchronously.

    Parameters
    ----------
    params      : dict from ControlPanel.get_all_params() (or loaded .chtmdl)
    log_fn      : callable(str)    — status messages
    progress_fn : callable(int)    — 0-100 progress (optional)
    abort_fn    : callable()->bool — return True to stop (optional)

    Returns str path to output .npz, or raises RuntimeError.
    """
    pct = lambda n: progress_fn(n) if progress_fn else None
    aborted = lambda: abort_fn() if abort_fn else False

    dom          = params["domain"]
    project_name = (params.get("project_name") or "").strip() or "cht_result"
    snapshot_file = f"{project_name}.npz"

    log_fn("\u2699  Building SimConfig3D \u2026")
    cfg = SimConfig3D(
        Lx_m=dom["Lx"], Ly_m=dom["Ly"], Lz_m=dom["Lz"],
        flow_bc="inlet_outlet", flow_dir=params["flow_dir"],
        transverse_walls=True, u_in_mps=params["u_in"],
        dt_thermal_phys_s=1.0, temp_bc="fixed",
        t_ambient_C=params["t_amb_C"], heating_mode="off", qdot_total_W=0.0,
        solid_init_mode="ambient", T_hot_C=0.0,
        collision=params["collision"], outlet_bc_mode=params["outlet_bc"],
        snapshot_file=snapshot_file,
    )
    dx_mm = params["dx_mm"]
    cfg.dx_mm = dx_mm
    cfg.nx = round(dom["Lx"] * 1000 / dx_mm)
    cfg.ny = round(dom["Ly"] * 1000 / dx_mm)
    cfg.nz = round(dom["Lz"] * 1000 / dx_mm)
    cfg.obstacle = None

    cfg.domain_walls   = params.get("domain_walls",  [])
    cfg.domain_outlets = params.get("domain_outlets", [])
    log_fn(f"   Walls:   {cfg.domain_walls}")
    log_fn(f"   Outlets: {cfg.domain_outlets}")

    pct(5)

    fluid = FluidPropsSI(**FLUID_PRESETS[params["fluid"]])
    fluid.tin_C = params["t_in_C"]
    pct(8)

    if params.get("gpu_raytrace", True):
        log_fn("🔧  Patching STLVoxelizer (GPU ray tracing) …")
        patch_stl_voxelizer(STLVoxelizer)
    else:
        log_fn("⚠  GPU ray tracing off — restoring trimesh voxelizer")
        unpatch_stl_voxelizer(STLVoxelizer)
    bodies_dicts = [
        dict(stl=b.stl_path, npz=str(Path(b.stl_path).parent / f"{b.name}.npz"), name=b.name,
             build_dir=b.build_dir,
             material=SolidPropsSI(**SOLID_PRESETS[b.material]),
             color=b.color, role=b.role)
        for b in params["bodies"]
    ]
    pct(12)

    log_fn("\U0001f533  Building SolidAssembly \u2026")
    assembly = SolidAssembly(
        bodies=bodies_dicts, fluid_Lz_m=dom["Lz"], dx_mm=dx_mm, cfg=cfg,
        flow_dir=params["flow_dir"], j0_divisor=params["j0_divisor"],
        STLVoxelizer_cls=STLVoxelizer, LBMCHT_cls=LBMCHT3D_Torch,
    )
    assembly.build_domain(); assembly.print_summary()
    pct(22)

    default_solid = SolidPropsSI(**SOLID_PRESETS["Aluminum (LBM-scaled)"])
    log_fn("\U0001f527  Initialising LBM solver \u2026")
    sim = LBMCHT3D_Torch(cfg, fluid, default_solid)

    _target_scale = 3.0   # conservative — MG works reliably up to ~5
    sim.dt_thermal_phys_s = _target_scale * sim.dt_flow_phys_s
    sim.thermal_dt_scale  = _target_scale
    print(f"[THERMAL] dt_thermal_phys_s={sim.dt_thermal_phys_s:.4e} s  "
        f"(thermal_dt_scale={_target_scale})")

    assembly.stamp(sim); sim._assembly = assembly
    pct(30)

    log_fn("\U0001f4d0  Building SDF & IBB \u2026")
    phi_gpu = build_phi_from_voxel_mask_mm(sim.solid.cpu().numpy(), cfg.dx_mm, "cuda")
    sim.phi = phi_gpu; sim.solid = (phi_gpu < 0.0)
    assembly.restore_thin_bodies(sim)
    base_name = params["bodies"][0].name
    build_ibb_from_sdf_gpu(sim,
        voxel_origin_mm=assembly._geo_objs[base_name].voxel_origin_mm,
        pitch_mm=cfg.dx_mm, offset_ijk=assembly.specs[0].offset_ijk,
        c_np=sim.c_np, opp_np=sim.opp_np)
    assembly.wall_off_subfluid(sim)
    sim._apply_transverse_flow_walls()
    pct(40)

    bct = params["bc_type"]; bcp = params["bc_params"]
    if bct in ("surface_flux", "volumetric"):
        _apply_bc(sim, cfg, assembly, bct, bcp, log_fn)
    pct(45)

    if aborted(): raise RuntimeError("Aborted by user.")

    log_fn("\u26a1  Patching flow/thermal solvers \u2026")
    FlowSolverUpgrade.patch(sim); LBMCUDAUpgrade.patch(sim)
    log_fn("\U0001f4a7  Solving flow to steady-state \u2026")
    sim.run_flow_to_steady_v2(tol_u_ema=params["tol_u_ema"])
    pct(65)

    if aborted(): raise RuntimeError("Aborted by user.")

    # k0 = cfg.fluid_k0
    # if k0 > 0:
    #     sim.u[:, :, :k0]  = 0.0
    #     sim.v[:, :, :k0]  = 0.0
    #     sim.wv[:, :, :k0] = 0.0

    log_fn("\U0001f321  Solving thermal field \u2026")
    sim.solve_thermal_steady_only(
        max_outer=int(params["max_outer"]),
        tol_dT_solid=params["tol_dTs"], tol_dT_fluid=params["tol_dTf"],
        tol_dTout_mean=params["tol_dTs"],
        dt_scale_min=1.0, dt_scale_max=params["dt_scale_max"],
        dt_scale_start=2.0, dt_ramp_factor=1.5, dt_backoff_factor=0.5,
        dt_ramp_threshold=0.8, max_mg_cycles=params["max_mg"],
        tol_mg=5e-3, stable_steps=8,
    )
    assembly.print_thermal_diagnostics(sim)
    pct(90)

    if aborted(): raise RuntimeError("Aborted by user.")

    log_fn("\U0001f4be  Saving snapshot \u2026")
    sim._alloc_snapshot_buffers(nsteps=0)
    sim._save_snapshot(0)
    sim._flush_snapshots()
    out_npz = os.path.join(cfg.out_dir, cfg.snapshot_file)
    pct(100)
    log_fn(f"\u2705  Done  \u2192  {out_npz}")
    return out_npz


def _apply_bc(sim, cfg, assembly, bct, bcp, log_fn):
    import torch as _torch
    tgt  = bcp["solid_name"]
    spec = next((s for s in assembly.specs if s.name == tgt), None)
    if spec is None:
        raise RuntimeError(f"Body '{tgt}' not found in assembly.")
    vox      = np.load(spec.name + ".npz", allow_pickle=True)["voxel_solid"].astype(bool)
    dom_mask = np.zeros((cfg.nx, cfg.ny, cfg.nz), dtype=bool)
    i0,j0,k0 = (int(x) for x in spec.offset_ijk)
    Gx,Gy,Gz  = vox.shape
    di0=max(i0,0);di1=min(i0+Gx,cfg.nx); dj0=max(j0,0);dj1=min(j0+Gy,cfg.ny); dk0=max(k0,0);dk1=min(k0+Gz,cfg.nz)
    si0=di0-i0;si1=si0+(di1-di0); sj0=dj0-j0;sj1=sj0+(dj1-dj0); sk0=dk0-k0;sk1=sk0+(dk1-dk0)
    dom_mask[di0:di1,dj0:dj1,dk0:dk1] = vox[si0:si1,sj0:sj1,sk0:sk1]
    mask_t = _torch.from_numpy(dom_mask).to("cuda")
    if bct == "surface_flux":
        log_fn(f"\U0001f525  SurfaceFluxBC: {dom_mask.sum():,} cells")
        ctr = assembly.get_body_center_mm(tgt) if bcp.get("auto_center") else None
        sim.add_surface_flux(SurfaceFluxBC(
            solid_mask=mask_t, axis=bcp["axis"],
            surface_L_mm=bcp["L_mm"], surface_W_mm=bcp["W_mm"],
            q_flux_W_m2=bcp["q_flux"], dx_mm=cfg.dx_mm, center_mm=ctr))
    else:
        log_fn(f"\U0001f525  VolumetricHeatBC: {dom_mask.sum():,} cells")
        sim.add_volumetric_heat(VolumetricHeatBC(
            solid_mask=mask_t, Q_watts=bcp["Q_watts"], dx_mm=cfg.dx_mm))


# ── Qt wrapper ────────────────────────────────────────────────────────────────

# class SimWorker(QObject):
#     progress = pyqtSignal(str)
#     pct      = pyqtSignal(int)
#     finished = pyqtSignal(str)
#     error    = pyqtSignal(str)

#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#         self._abort = False

#     def abort(self): self._abort = True

#     def run(self):
#         if not HAS_SIM:
#             self.error.emit("Simulation modules not installed."); return
#         try:
#             out = run_simulation_blocking(
#                 self.params,
#                 log_fn      = self.progress.emit,
#                 progress_fn = self.pct.emit,
#                 abort_fn    = lambda: self._abort,
#             )
#             self.finished.emit(out)
#         except Exception as exc:
#             self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
