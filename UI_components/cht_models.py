"""
cht_models.py  — project_name added
"""
from __future__ import annotations
import json
from pathlib import Path
from UI_components.cht_constants import BODY_COLORS, MODEL_VERSION


class SolidBodySpec:
    _counter = 0
    def __init__(self, stl_path="", name=None, build_dir="+Z",
                 material="Aluminum (LBM-scaled)", color=None, role="fluid_base"):
        SolidBodySpec._counter += 1
        self.stl_path  = stl_path
        self.name      = name or f"body_{SolidBodySpec._counter}"
        self.build_dir = build_dir
        self.material  = material
        self.color     = color or BODY_COLORS[(SolidBodySpec._counter-1) % len(BODY_COLORS)]
        self.role      = role
    @property
    def npz_name(self): return f"{self.name}.npz"
    def label(self): return f"{self.name}   [{self.material}]   \u00b7  {self.role}"
    def to_dict(self):
        return dict(stl_path=self.stl_path, name=self.name, build_dir=self.build_dir,
                    material=self.material, color=self.color, role=self.role)
    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.stl_path  = d.get("stl_path", "")
        obj.name      = d.get("name", "body")
        obj.build_dir = d.get("build_dir", "+Z")
        obj.material  = d.get("material", "Aluminum (LBM-scaled)")
        obj.color     = d.get("color", BODY_COLORS[0])
        obj.role      = d.get("role", "fluid_base")
        return obj

def build_state(domain, j0_divisor, bodies, fluid, t_in_C, t_amb_C,
                flow_dir, u_in, collision, outlet_bc, bc_type, bc_params,
                dx_mm, tol_u_ema, max_outer, tol_dTs, tol_dTf,
                dt_scale_max, max_mg, project_name="", gpu_raytrace=True,
                face_states=None, domain_walls=None, domain_outlets=None):
    return dict(version=MODEL_VERSION, project_name=project_name,
                domain=domain, j0_divisor=j0_divisor,
                bodies=[b.to_dict() for b in bodies],
                fluid=fluid, t_in_C=t_in_C, t_amb_C=t_amb_C,
                flow_dir=flow_dir, u_in=u_in,
                collision=collision, outlet_bc=outlet_bc,
                bc_type=bc_type, bc_params=bc_params,
                dx_mm=dx_mm, gpu_raytrace=gpu_raytrace, tol_u_ema=tol_u_ema, max_outer=max_outer,
                tol_dTs=tol_dTs, tol_dTf=tol_dTf,
                dt_scale_max=dt_scale_max, max_mg=max_mg,
                face_states=face_states or {},
                domain_walls=domain_walls or [],
                domain_outlets=domain_outlets or [])

def save_model(state, filepath):
    try:
        Path(filepath).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        print(f"[ModelIO] save failed: {exc}"); return False

def load_model(filepath):
    try:
        state = json.loads(Path(filepath).read_text(encoding="utf-8"))
        state["bodies"] = [SolidBodySpec.from_dict(d) for d in state.get("bodies", []) if isinstance(d, dict)]
        return state
    except Exception as exc:
        print(f"[ModelIO] load failed: {exc}"); return None
