"""
backend/state.py — Default state definition and app.storage.user accessor.
Storage contains ONLY JSON-serialisable primitives (str, int, float, bool, list, dict).
"""
from __future__ import annotations

DEFAULT_STATE: dict = {
    # ── Domains ──────────────────────────────────────────────────────────────
    "Lx": 5.0, "Ly": 1.0, "Lz": 1.0,
    "j0_divisor": 2.5,
    "fluid": "Air (LBM-scaled)",
    "gpu_raytrace": True,
    "bodies": [],   # list[dict] — never store widget refs here

    # ── BCs ───────────────────────────────────────────────────────────────────
    "flow_dir": "+X",
    "u_in": 1.0,
    "t_in_C": 20.0,
    "t_amb_C": 20.0,
    # Face assignment
    "face_inlet":   "-X",
    "face_outlets": ["+X"],
    "face_walls":   ["+Y", "-Y", "+Z", "-Z"],
    # BC type
    "bc_type": "off",   # "off" | "surface_flux" | "volumetric"
    # Surface flux params
    "flux_solid": "", "flux_axis": "-Z",
    "flux_L_mm": 400.0, "flux_W_mm": 200.0,
    "flux_q": 1800.0, "flux_autocenter": True,
    # Volumetric params
    "vol_solid": "", "vol_Q": 50.0,
    # Voxelisation
    "dx_mm": 10.0,

    # ── Solver ────────────────────────────────────────────────────────────────
    "tol_u_ema": 0.005,
    "collision": "mrt_smag",
    "outlet_bc": "convective",
    "max_outer": 50000,
    "tol_dTs": 0.005,
    "tol_dTf": 0.005,
    "dt_scale_max": 100.0,
    "max_mg": 15,

    # ── Project / results ─────────────────────────────────────────────────────
    "project_name": "cht_result",
    "result_npz": "",
    "result_flow_dir": "+X",

    # ── Visualize ─────────────────────────────────────────────────────────────
    "vis_npz": "",
    "vis_field": "T",
    "vis_cmap": "inferno",
    "vis_fluid": False,
    "vis_fluid_opacity": 0.3,
    "vis_stream": True,
    "vis_nseeds": 150,
    "vis_seed_mode": "inlet_plane",
    "vis_low_quality": True,
    "vis_bodies": {},   # name → bool
}


def get_storage() -> dict:
    """Return app.storage.user, back-filling any missing keys from DEFAULT_STATE."""
    from nicegui import app as _app
    s = _app.storage.user
    for k, v in DEFAULT_STATE.items():
        if k not in s:
            import copy
            s[k] = copy.deepcopy(v)
    return s
