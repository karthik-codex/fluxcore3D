"""
backend/model_io.py — Constants mirrored from cht_constants.py + model save/load.
"""
from __future__ import annotations
import json
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

FLUID_PRESETS = {
    "Water":            dict(nu_m2_s=1.0e-6,  rho_kg_m3=997.0, k_W_mK=0.600, cp_J_kgK=4182.0, tin_C=25.0),
    "Air (physical)":   dict(nu_m2_s=15.0e-6, rho_kg_m3=1.25,  k_W_mK=0.024, cp_J_kgK=1006.0, tin_C=20.0),
    "Air (LBM-scaled)": dict(nu_m2_s=0.02,    rho_kg_m3=1.0,   k_W_mK=1.0,   cp_J_kgK=50.0,   tin_C=20.0),
}

SOLID_PRESETS = {
    "Aluminum (physical)":   dict(k_W_mK=237.0, rho_kg_m3=2700.0, cp_J_kgK=900.0),
    "Aluminum (LBM-scaled)": dict(k_W_mK=5.0,   rho_kg_m3=1.0,    cp_J_kgK=80.0),
    "Copper":                dict(k_W_mK=401.0, rho_kg_m3=8960.0, cp_J_kgK=385.0),
    "TIM (silicone pad)":    dict(k_W_mK=6.0,   rho_kg_m3=2500.0, cp_J_kgK=800.0),
    "Transformer core":      dict(k_W_mK=25.0,  rho_kg_m3=7650.0, cp_J_kgK=490.0),
    "Ferrite core":          dict(k_W_mK=4.0,   rho_kg_m3=4800.0, cp_J_kgK=700.0),
    "Steel":                 dict(k_W_mK=50.0,  rho_kg_m3=7850.0, cp_J_kgK=490.0),
}

BODY_COLORS  = ["steelblue", "darkorange", "gold", "sienna", "mediumpurple",
                "tomato", "limegreen", "hotpink", "deepskyblue", "coral"]
BODY_ROLES   = ["fluid_base", "stack_below", "stack_above", "fixed"]
BUILD_DIRS   = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
FLOW_DIRS    = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
FLUX_AXES    = ["-Z", "+Z", "-Y", "+Y", "-X", "+X"]
COLLISION_MODES = ["bgk", "trt", "mrt", "mrt_smag"]
OUTLET_MODES    = ["convective", "extrapolation", "neumann"]

MODEL_VERSION = "1.0"


# ── Model I/O ─────────────────────────────────────────────────────────────────

def build_state_dict(storage: dict) -> dict:
    """Convert app.storage.user → serialisable model dict."""
    bodies_raw = storage.get("bodies", [])
    return dict(
        version        = MODEL_VERSION,
        project_name   = storage.get("project_name", "cht_result"),
        domain         = dict(Lx=storage["Lx"], Ly=storage["Ly"], Lz=storage["Lz"]),
        j0_divisor     = storage["j0_divisor"],
        bodies         = [b if isinstance(b, dict) else b.to_dict() for b in bodies_raw],
        fluid          = storage["fluid"],
        t_in_C         = storage["t_in_C"],
        t_amb_C        = storage["t_amb_C"],
        flow_dir       = storage["flow_dir"],
        u_in           = storage["u_in"],
        collision      = storage["collision"],
        outlet_bc      = storage["outlet_bc"],
        bc_type        = storage["bc_type"],
        bc_params      = _build_bc_params(storage),
        dx_mm          = storage["dx_mm"],
        gpu_raytrace   = storage["gpu_raytrace"],
        tol_u_ema      = storage["tol_u_ema"],
        max_outer      = int(storage["max_outer"]),
        tol_dTs        = storage["tol_dTs"],
        tol_dTf        = storage["tol_dTf"],
        dt_scale_max   = storage["dt_scale_max"],
        max_mg         = int(storage["max_mg"]),
        face_states    = dict(
            inlet   = storage.get("face_inlet",   "-X"),
            outlets = storage.get("face_outlets", ["+X"]),
            walls   = storage.get("face_walls",   []),
        ),
        domain_walls   = storage.get("face_walls",   []),
        domain_outlets = storage.get("face_outlets", ["+X"]),
    )


def _build_bc_params(storage: dict) -> dict:
    bct = storage.get("bc_type", "off")
    if bct == "surface_flux":
        return dict(
            solid_name   = storage.get("flux_solid", ""),
            axis         = storage.get("flux_axis",  "-Z"),
            L_mm         = storage.get("flux_L_mm",  400.0),
            W_mm         = storage.get("flux_W_mm",  200.0),
            q_flux       = storage.get("flux_q",     1800.0),
            auto_center  = storage.get("flux_autocenter", True),
        )
    if bct == "volumetric":
        return dict(
            solid_name   = storage.get("vol_solid", ""),
            Q_watts      = storage.get("vol_Q",      50.0),
        )
    return {}


def save_model_file(storage: dict, filepath: str) -> bool:
    try:
        state = build_state_dict(storage)
        Path(filepath).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        print(f"[ModelIO] save failed: {exc}")
        return False


def load_model_file(filepath: str) -> dict | None:
    try:
        raw = json.loads(Path(filepath).read_text(encoding="utf-8"))
        return raw
    except Exception as exc:
        print(f"[ModelIO] load failed: {exc}")
        return None


def restore_storage_from_model(storage: dict, model: dict) -> None:
    """Write a loaded model dict back into app.storage.user."""
    dom = model.get("domain", {})
    if "Lx" in dom: storage["Lx"] = dom["Lx"]
    if "Ly" in dom: storage["Ly"] = dom["Ly"]
    if "Lz" in dom: storage["Lz"] = dom["Lz"]
    for key in ("j0_divisor", "fluid", "gpu_raytrace", "flow_dir", "u_in",
                "t_in_C", "t_amb_C", "dx_mm", "bc_type",
                "tol_u_ema", "max_outer", "tol_dTs", "tol_dTf",
                "dt_scale_max", "max_mg", "collision", "outlet_bc", "project_name"):
        if key in model:
            storage[key] = model[key]
    # bodies — keep as list of plain dicts
    raw_bodies = model.get("bodies", [])
    print(f"[ModelIO] restore: found {len(raw_bodies)} bodies in model file")
    for i, b in enumerate(raw_bodies):
        name = b.get("name", "?") if isinstance(b, dict) else getattr(b, "name", "?")
        stl  = b.get("stl_path", b.get("stl", "?")) if isinstance(b, dict) else getattr(b, "stl_path", "?")
        print(f"  [{i}] name={name}  stl={stl}")
        # Normalise legacy 'stl' key → 'stl_path'
        if isinstance(b, dict) and "stl_path" not in b and "stl" in b:
            b["stl_path"] = b["stl"]
    storage["bodies"] = [b if isinstance(b, dict) else b for b in raw_bodies]
    # bc_params
    bcp = model.get("bc_params", {})
    bct = model.get("bc_type", "off")
    storage["bc_type"] = bct
    if bct == "surface_flux":
        storage["flux_solid"]      = bcp.get("solid_name", "")
        storage["flux_axis"]       = bcp.get("axis",        "-Z")
        storage["flux_L_mm"]       = bcp.get("L_mm",        400.0)
        storage["flux_W_mm"]       = bcp.get("W_mm",        200.0)
        storage["flux_q"]          = bcp.get("q_flux",      1800.0)
        storage["flux_autocenter"] = bcp.get("auto_center",  True)
    elif bct == "volumetric":
        storage["vol_solid"] = bcp.get("solid_name", "")
        storage["vol_Q"]     = bcp.get("Q_watts",     50.0)
    # face states
    fs = model.get("face_states", {})
    if "inlet"   in fs: storage["face_inlet"]   = fs["inlet"]
    if "outlets" in fs: storage["face_outlets"]  = fs["outlets"]
    if "walls"   in fs: storage["face_walls"]    = fs["walls"]
