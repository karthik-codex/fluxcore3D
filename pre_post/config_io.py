import os
import json
from collections import OrderedDict
from typing import Any, Dict, Mapping, Tuple

try:
    import yaml  # pip install pyyaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# --- Category → flat-key mapping (loader flattens back to your original names) ---
_FLAT_MAP: Dict[str, Tuple[str, str]] = {
    # meta
    "material_name": ("meta", "material_name"),
    "dtype": ("meta", "dtype"),
    "tangent_stencil": ("meta", "tangent_stencil"),
    "ghost_mode": ("meta", "ghost_mode"),
    # paths
    "part_STL": ("paths", "part_STL"),
    "build_dir": ("paths", "build_dir"),
    "id_output": ("paths", "id_output"),
    # geometry
    "part_scale": ("geometry", "part_scale"),
    "layer_height": ("geometry", "layer_height"),
    "nodes_per_layer": ("geometry", "nodes_per_layer"),
    "sampling_type": ("geometry", "sampling_type"),
    # material
    "E": ("material", "E"),
    "nu": ("material", "nu"),
    "alpha": ("material", "alpha"),
    "rho": ("material", "rho"),
    # gravity
    "g": ("gravity", "g"),
    "gravity": ("gravity", "gravity"),
    # contact
    "mu": ("contact", "mu"),
    "gammaN": ("contact", "gammaN"),
    "gammaT": ("contact", "gammaT"),
    "cnt_qp_max": ("contact", "cnt_qp_max"),
    "cnt_bilateral_gap_scale": ("contact", "cnt_bilateral_gap_scale"),
    "g_smooth": ("contact", "g_smooth"),
    "cnt_kt_slip_reg": ("contact", "cnt_kt_slip_reg"),
    "cnt_alpha_lambda": ("contact", "cnt_alpha_lambda"),
    "cnt_AL_decay_zeta": ("contact", "cnt_AL_decay_zeta"),
    "cnt_mu_reg_eps_tau": ("contact", "cnt_mu_reg_eps_tau"),
    "cnt_slip_reg_eps": ("contact", "cnt_slip_reg_eps"),
    "cnt_E_ref": ("contact", "cnt_E_ref"),
    "cnt_nu_ref": ("contact", "cnt_nu_ref"),
    "eps_sign": ("contact", "eps_sign"),
    "cnt_gt_reg_eps": ("contact", "cnt_gt_reg_eps"),
    "cnt_friction_threshold": ("contact", "cnt_friction_threshold"),
    # damping
    "gamma_M": ("damping", "gamma_M"),
    "gamma_K": ("damping", "gamma_K"),
    "nr_relax_damping": ("damping", "nr_relax_damping"),
    "ptc_gamma0": ("damping", "ptc_gamma0"),
    "alpha_min": ("damping", "alpha_min"),
    "alpha_max": ("damping", "alpha_max"),
    "evp_alpha_damp": ("damping", "evp_alpha_damp"),
    # thermal
    "T_amb": ("thermal", "T_amb"),
    "T_plate": ("thermal", "T_plate"),
    "T_target": ("thermal", "T_target"),
    # solver
    "nr_max_iter": ("solver", "nr_max_iter"),
    "nr_tol": ("solver", "nr_tol"),
    "nr_rel_tol_base": ("solver", "nr_rel_tol_base"),
    "nr_abs_floor_smallR0": ("solver", "nr_abs_floor_smallR0"),
    "nr_abs_tol_hold": ("solver", "nr_abs_tol_hold"),
    "nr_ls_max": ("solver", "nr_ls_max"),
    "nr_cg_max_iter": ("solver", "nr_cg_max_iter"),
    "nr_cg_tol": ("solver", "nr_cg_tol"),
    "nr_solver_type": ("solver", "nr_solver_type"),
    "preconditioner": ("solver", "preconditioner"),
    "force_method": ("solver", "force_method"),
    "verbose": ("solver", "verbose"),
    "secant_predictor": ("solver", "secant_predictor"),
    # adaptivity
    "dt_init": ("adaptivity", "dt_init"),
    "dt_min": ("adaptivity", "dt_min"),
    "dt_max": ("adaptivity", "dt_max"),
    "dt_cap": ("adaptivity", "dt_cap"),
    "dt_hold1": ("adaptivity", "dt_hold1"),
    "dt_hold2": ("adaptivity", "dt_hold2"),
    "dt_hold3": ("adaptivity", "dt_hold3"),
    "grow_factor": ("adaptivity", "grow_factor"),
    "shrink_factor": ("adaptivity", "shrink_factor"),
    "max_retries": ("adaptivity", "max_retries"),
    "iter_easy": ("adaptivity", "iter_easy"),
    "iter_hard": ("adaptivity", "iter_hard"),
    "plateau_window": ("adaptivity", "plateau_window"),
    "plateau_tol": ("adaptivity", "plateau_tol"),
    # bool
    "render_3D": ("bool", "render_3D"),
    "print_log_file": ("bool", "print_log_file"),
    "disp_vectors": ("bool", "disp_vectors"),
    "initialize": ("bool", "initialize"),
    "run_model": ("bool", "run_model"),
    "model_preview": ("bool", "model_preview"),
    "log_output": ("bool", "log_output"),
}

# Preferred output order by category (controls OrderedDict order)
_CATEGORY_ORDER = [
    "meta", "paths", "geometry", "material",
    "gravity", "contact", "damping", "time",
    "thermal", "solver", "adaptivity", "bool"
]

def _read_any(path: str) -> Mapping[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {ext}")

def _get_nested(d: Mapping[str, Any], cat: str, key: str, default=Ellipsis):
    if cat not in d:
        if default is Ellipsis:
            raise KeyError(f"Missing category '{cat}' in config.")
        return default
    if key not in d[cat]:
        if default is Ellipsis:
            raise KeyError(f"Missing key '{key}' under '{cat}'.")
        return default
    return d[cat][key]

def _derive_defaults(flat: Dict[str, Any]) -> None:
    # part_name from part_STL
    part_stl = flat.get("part_STL", "")
    if part_stl and "part_name" not in flat:
        flat["part_name"] = os.path.basename(part_stl).split(".")[0]
    # Some sensible fallbacks if missing (optional)
    flat.setdefault("preconditioner", "None")
    flat.setdefault("verbose", True)

def load_config(path: str) -> "OrderedDict[str, Any]":
    """
    Loads a categorized YAML/JSON config and returns a flat, ordered dict
    matching your original keys (compatible with your class).
    Keys are ordered by the category sequence in _CATEGORY_ORDER.
    """
    data = _read_any(path)

    # Build a flat OrderedDict following category order
    out = OrderedDict()
    for category in _CATEGORY_ORDER:
        # find all flat keys that map to this category
        cat_keys = [k for k, (cat, _) in _FLAT_MAP.items() if cat == category]
        for k in cat_keys:
            cat, inner = _FLAT_MAP[k]
            # Some keys are optional; use Ellipsis to enforce strictness where needed
            default = None if k in {"alpha_const", "alpha_min", "alpha_max"} else Ellipsis
            try:
                out[k] = _get_nested(data, cat, inner, default=default)
            except KeyError as e:
                raise KeyError(f"{e} (while reading category '{category}', key '{k}')") from None

    # Derive & fill defaults/derived keys
    _derive_defaults(out)

    return out

# Convenience: JSON save of the flattened dict (keeps your preferred order)
def save_flat_json(cfg: Mapping[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# Example usage:
if __name__ == "__main__":
    cfg = load_config("config.yaml")         # or config.json
    # Now pass cfg straight to your class:
    # sim = MySimClass(cfg)
    print("Loaded keys (ordered):", list(cfg.keys()))
    print("part_name:", cfg.get("part_name"))
