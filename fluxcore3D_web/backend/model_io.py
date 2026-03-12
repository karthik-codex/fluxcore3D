"""
backend/model_io.py — Constants mirrored from cht_constants.py + model save/load.
"""
from __future__ import annotations
import json
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

FLUID_PRESETS = {
    # ── Standard coolants ────────────────────────────────────────────────────
    "Water":
        dict(nu_m2_s=1.00e-6,  rho_kg_m3=997.0,  k_W_mK=0.600, cp_J_kgK=4182.0, tin_C=25.0),
    "EG/Water 50-50":           # ethylene glycol 50% by vol, 25°C
        dict(nu_m2_s=3.85e-6,  rho_kg_m3=1075.0, k_W_mK=0.410, cp_J_kgK=3300.0, tin_C=25.0),
    "PG/Water 50-50":           # propylene glycol 50% by vol, 25°C (food-safe)
        dict(nu_m2_s=5.00e-6,  rho_kg_m3=1040.0, k_W_mK=0.380, cp_J_kgK=3600.0, tin_C=25.0),
    "Mineral Oil":              # transformer/immersion grade
        dict(nu_m2_s=40.0e-6,  rho_kg_m3=870.0,  k_W_mK=0.130, cp_J_kgK=1900.0, tin_C=40.0),
    "PAO-6":                    # polyalphaolefin, data center cold plate
        dict(nu_m2_s=18.0e-6,  rho_kg_m3=820.0,  k_W_mK=0.140, cp_J_kgK=2100.0, tin_C=40.0),

    # ── Dielectric fluorocarbon coolants ─────────────────────────────────────
    "FC-72":                    # 3M Fluorinert, Tb=56°C, immersion/spray
        dict(nu_m2_s=0.38e-6,  rho_kg_m3=1680.0, k_W_mK=0.057, cp_J_kgK=1100.0, tin_C=25.0),
    "FC-40":                    # 3M Fluorinert, Tb=165°C, high-temp immersion
        dict(nu_m2_s=2.20e-6,  rho_kg_m3=1855.0, k_W_mK=0.062, cp_J_kgK=1100.0, tin_C=25.0),
    "Novec 649":                # 3M Novec, low-GWP FC-72 replacement
        dict(nu_m2_s=0.40e-6,  rho_kg_m3=1600.0, k_W_mK=0.059, cp_J_kgK=1103.0, tin_C=25.0),
    "HFE-7100":                 # 3M Novec 7100, HFE chemistry
        dict(nu_m2_s=0.61e-6,  rho_kg_m3=1510.0, k_W_mK=0.069, cp_J_kgK=1183.0, tin_C=25.0),

    # ── Gas ──────────────────────────────────────────────────────────────────
    "Air (physical)":
        dict(nu_m2_s=15.0e-6,  rho_kg_m3=1.225,  k_W_mK=0.0257, cp_J_kgK=1006.0, tin_C=20.0),
    "Air (LBM-scaled)":
        dict(nu_m2_s=0.02,     rho_kg_m3=1.0,    k_W_mK=1.0,    cp_J_kgK=50.0,   tin_C=20.0),
}

SOLID_PRESETS = {
    # ── Common metals ────────────────────────────────────────────────────────
    "Aluminum":
        dict(k_W_mK=237.0,  rho_kg_m3=2700.0, cp_J_kgK=900.0),
    "Copper":
        dict(k_W_mK=401.0,  rho_kg_m3=8960.0, cp_J_kgK=385.0),
    "Steel (carbon)":
        dict(k_W_mK=50.0,   rho_kg_m3=7850.0, cp_J_kgK=490.0),
    "Stainless Steel 316":
        dict(k_W_mK=16.2,   rho_kg_m3=8000.0, cp_J_kgK=500.0),
    "Tungsten":             # heat spreader, high-density shielding
        dict(k_W_mK=173.0,  rho_kg_m3=19300.0, cp_J_kgK=134.0),
    "Molybdenum":           # CTE-matched carrier for power devices
        dict(k_W_mK=138.0,  rho_kg_m3=10220.0, cp_J_kgK=251.0),

    # ── Semiconductor dies ───────────────────────────────────────────────────
    "Silicon (die)":
        dict(k_W_mK=148.0,  rho_kg_m3=2329.0, cp_J_kgK=700.0),
    "SiC (4H, substrate)":  # silicon carbide — SiC MOSFET, Schottky diode
        dict(k_W_mK=370.0,  rho_kg_m3=3210.0, cp_J_kgK=750.0),
    "GaN (bulk)":           # gallium nitride — GaN HEMT, RF, power
        dict(k_W_mK=130.0,  rho_kg_m3=6150.0, cp_J_kgK=490.0),
    "GaAs":                 # gallium arsenide — RF/microwave die
        dict(k_W_mK=46.0,   rho_kg_m3=5320.0, cp_J_kgK=350.0),
    "Diamond (CVD)":        # heat spreader for GaN-on-diamond
        dict(k_W_mK=1800.0, rho_kg_m3=3510.0, cp_J_kgK=520.0),

    # ── Ceramic substrates ───────────────────────────────────────────────────
    "AlN (standard, 170W)": # aluminum nitride DBC substrate
        dict(k_W_mK=170.0,  rho_kg_m3=3260.0, cp_J_kgK=740.0),
    "AlN (high grade, 230W)":
        dict(k_W_mK=230.0,  rho_kg_m3=3260.0, cp_J_kgK=740.0),
    "Si3N4":                # silicon nitride — toughest ceramic substrate
        dict(k_W_mK=30.0,   rho_kg_m3=3210.0, cp_J_kgK=700.0),
    "Al2O3 (96%)":          # alumina — low-cost DBC ceramic
        dict(k_W_mK=24.0,   rho_kg_m3=3720.0, cp_J_kgK=880.0),
    "BeO":                  # beryllia — high-k, toxic to machine
        dict(k_W_mK=250.0,  rho_kg_m3=2850.0, cp_J_kgK=1050.0),

    # ── Carbon-based spreaders ───────────────────────────────────────────────
    "Graphite (isotropic)":
        dict(k_W_mK=120.0,  rho_kg_m3=1800.0, cp_J_kgK=710.0),
    "Pyrolytic Graphite (in-plane)":  # vapor chamber spreader core
        dict(k_W_mK=1500.0, rho_kg_m3=2200.0, cp_J_kgK=710.0),

    # ── Die attach / bonding layers ──────────────────────────────────────────
    "Solder SAC305":        # Sn96.5Ag3Cu0.5, standard reflow
        dict(k_W_mK=57.0,   rho_kg_m3=7390.0, cp_J_kgK=230.0),
    "Indium solder":        # In100, low-temp, compliant
        dict(k_W_mK=82.0,   rho_kg_m3=7310.0, cp_J_kgK=233.0),
    "Sintered Silver":      # Ag sintering paste, high-power modules
        dict(k_W_mK=250.0,  rho_kg_m3=6000.0, cp_J_kgK=235.0),

    # ── Thermal interface materials ──────────────────────────────────────────
    "TIM (silicone pad, 6W)":
        dict(k_W_mK=6.0,    rho_kg_m3=2500.0, cp_J_kgK=800.0),
    "TIM (phase-change, 4W)":
        dict(k_W_mK=4.0,    rho_kg_m3=2300.0, cp_J_kgK=900.0),
    "Thermal Grease (high-k)":  # Shin-Etsu X-23 class
        dict(k_W_mK=10.0,   rho_kg_m3=2600.0, cp_J_kgK=750.0),
    "Thermal Epoxy":
        dict(k_W_mK=2.0,    rho_kg_m3=1800.0, cp_J_kgK=1050.0),

    # ── Magnetic / passive components ────────────────────────────────────────
    "Transformer core (GOES)": # grain-oriented electrical steel
        dict(k_W_mK=25.0,   rho_kg_m3=7650.0, cp_J_kgK=490.0),
    "Ferrite (MnZn)":
        dict(k_W_mK=4.0,    rho_kg_m3=4800.0, cp_J_kgK=700.0),
    "Ferrite (NiZn)":
        dict(k_W_mK=3.5,    rho_kg_m3=5000.0, cp_J_kgK=680.0),

    # ── Encapsulants / PCB ───────────────────────────────────────────────────
    "Epoxy mold compound":  # standard IC package mold
        dict(k_W_mK=0.8,    rho_kg_m3=1900.0, cp_J_kgK=1050.0),
    "FR4 (PCB)":
        dict(k_W_mK=0.3,    rho_kg_m3=1850.0, cp_J_kgK=1150.0),
    "Kapton (polyimide)":   # flex circuit, IMS dielectric
        dict(k_W_mK=0.12,   rho_kg_m3=1420.0, cp_J_kgK=1090.0),

    # ── LBM-scaled (for debugging / reduced-order runs) ─────────────────────
    "Aluminum (LBM-scaled)":
        dict(k_W_mK=5.0,    rho_kg_m3=1.0,    cp_J_kgK=80.0),
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
    import platform as _platform
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
    is_linux = _platform.system() != "Windows"
    for i, b in enumerate(raw_bodies):
        name = b.get("name", "?") if isinstance(b, dict) else getattr(b, "name", "?")
        stl  = b.get("stl_path", b.get("stl", "")) if isinstance(b, dict) else getattr(b, "stl_path", "")
        # Normalise legacy 'stl' key → 'stl_path'
        if isinstance(b, dict) and "stl_path" not in b and "stl" in b:
            b["stl_path"] = b["stl"]
        # Clear paths that don't exist on this machine (e.g. Windows paths loaded on Linux)
        if isinstance(b, dict):
            p = b.get("stl_path", "")
            is_windows_path = p.startswith(("C:\\", "D:\\", "C:/", "D:/")) or "\\" in p
            if p and not Path(p).exists():
                if is_linux and is_windows_path:
                    print(f"  [{i}] name={name}  stl=⚠ Windows path not found on Linux — cleared: {p}")
                else:
                    print(f"  [{i}] name={name}  stl=⚠ path not found — cleared: {p}")
                b["stl_path"] = ""
            else:
                print(f"  [{i}] name={name}  stl={stl or '(empty)'}")
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
