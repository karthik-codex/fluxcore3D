import json
import torch

def _pwl_interp_1d(xq, xk, yk, clamp=True):
    # xq, xk, yk: tensors on same device/dtype; xk strictly increasing
    xq = xq.unsqueeze(-1)  # (..., 1)
    # clamp query
    if clamp:
        xq = torch.clamp(xq, xk[0], xk[-1])
    # find right bin indices
    idx = torch.searchsorted(xk, xq, right=True).clamp(min=1, max=xk.numel()-1)  # (...,1)
    x0 = xk[idx-1]; x1 = xk[idx]
    y0 = yk[idx-1]; y1 = yk[idx]
    t = (xq - x0) / (x1 - x0 + 1e-30)
    y = y0 + t * (y1 - y0)
    return y.squeeze(-1)

class MaterialDB:
    def __init__(self, json_path, device='cpu', dtype=torch.float64):
        with open(json_path, 'r') as f:
            self.cfg = json.load(f)
        self.device = device
        self.dtype = dtype

        # Pre-tensorize for speed
        self._tbl = {}
        for name, mat in self.cfg["materials"].items():
            entry = {}

            def to_tensor(arr):
                return torch.as_tensor(arr, device=device, dtype=dtype)

            for key in ("E", "nu", "alpha_CTE"):
                if key in mat:
                    entry[key] = {
                        "T": to_tensor(mat[key]["T"]),
                        "val": to_tensor(mat[key]["val"]),
                        "clamp": (mat[key].get("extrap", "clamp") == "clamp"),
                    }

            # density scaling curves
            rd = mat.get("relative_density_scaling", {})
            if "E_factor_vs_phi" in rd:
                entry["E_factor_vs_phi"] = {
                    "phi": to_tensor(rd["E_factor_vs_phi"]["phi"]),
                    "val": to_tensor(rd["E_factor_vs_phi"]["val"]),
                    "clamp": (rd["E_factor_vs_phi"].get("extrap", "clamp") == "clamp"),
                }

            # viscosity (not used yet)
            visc = mat.get("viscosity", None)
            if visc and visc.get("model","") == "arrhenius":
                entry["viscosity"] = {
                    "R": visc.get("gas_constant_R", 8.314e-3),
                    "A0": {
                        "T": to_tensor(visc["A0"]["T"]),
                        "val": to_tensor(visc["A0"]["val"]),
                        "clamp": (visc["A0"].get("extrap","clamp")=="clamp"),
                    },
                    "Q": {
                        "T": to_tensor(visc["Q"]["T"]),
                        "val": to_tensor(visc["Q"]["val"]),
                        "clamp": (visc["Q"].get("extrap","clamp")=="clamp"),
                    }
                }

            self._tbl[name] = entry

    def get_E(self, mat_name, T, rho_rel=None):
        t = self._tbl[mat_name]
        E = _pwl_interp_1d(T, t["E"]["T"], t["E"]["val"], clamp=t["E"]["clamp"])
        if (rho_rel is not None) and ("E_factor_vs_phi" in t):
            s = t["E_factor_vs_phi"]
            fac = _pwl_interp_1d(rho_rel, s["phi"], s["val"], clamp=s["clamp"])
            E = E * fac
        return E

    def get_nu(self, mat_name, T):
        t = self._tbl[mat_name]
        return _pwl_interp_1d(T, t["nu"]["T"], t["nu"]["val"], clamp=t["nu"]["clamp"])

    def get_alpha(self, mat_name, T):
        t = self._tbl[mat_name]
        return _pwl_interp_1d(T, t["alpha_CTE"]["T"], t["alpha_CTE"]["val"],
                              clamp=t["alpha_CTE"]["clamp"])
