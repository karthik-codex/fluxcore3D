# nlevel_mg.py
"""
Standalone N-level geometric multigrid solver for the CHT thermal system.

    A theta = b,    A = I + dt * L_steady

L_steady = harmonic-mean diffusion + first-order upwind convection.
s = thermal_dt_scale is baked into coefficients at build time.
dt passed to build_if_needed / solve is the RAW outer-loop pseudo-time step.

torch.compile notes:
  - Boolean masked assignment  x[~mask] = 0.0  causes graph breaks.
    Replaced everywhere with  x.mul_(mask_f)  where mask_f = mask.float().
  - The compiled smoother is a free function _smooth_step_fn that takes only
    tensors — no self, no Python control flow on tensor values.
  - Compilation happens once at the end of _update_inv_diag after all level
    tensors are ready. Recompilation is avoided because tensor shapes and
    dtypes never change after _build_hierarchy.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Compiled free function for one weighted-Jacobi smooth step.
# All inputs are plain tensors; no Python branching on tensor values.
# mask_f = mask.float()  (pre-computed, stored in level dict)
# ─────────────────────────────────────────────────────────────────────────────
def _smooth_step_fn(
    x:         Tensor,   # current iterate (modified in-place)
    b:         Tensor,   # RHS
    Lo:        Tensor,   # output buffer for M(x)  — written by _Mop before call
    inv_diagM: Tensor,   # omega / diagM, zeroed at Dirichlet nodes
    mask_f:    Tensor,   # mask as float32/64: 1.0=unknown, 0.0=Dirichlet
    r:         Tensor,   # residual buffer (modified in-place)
) -> None:
    """
    r = (b - Lo) * mask_f
    x = (x + inv_diagM * r) * mask_f
    Lo is M(x) already evaluated by the caller.
    """
    r.copy_(b)
    r.sub_(Lo)
    r.mul_(mask_f)
    x.addcmul_(inv_diagM, r)
    x.mul_(mask_f)


# Compile once at module import. On Windows this uses TorchInductor.
# mode="reduce-overhead" minimises Python/CUDA-launch overhead for repeated
# same-shape calls, which is exactly the smoother pattern.
try:
    _smooth_step_compiled = torch.compile(
        _smooth_step_fn, mode="aot_eager", fullgraph=True
    )
except Exception:
    # torch.compile unavailable (older PyTorch, or compile disabled)
    _smooth_step_compiled = _smooth_step_fn


class NLevelGeometricMGSolver:

    def __init__(
        self,
        device:           torch.device,
        dtype:            torch.dtype,
        flow_axis:        int,    # 0=X, 1=Y, 2=Z
        flow_sign:        int,    # +1 or -1
        thermal_dt_scale: float,  # s = dt_thermal_phys / dt_flow_phys
        omega:            float = 0.8,
        n_pre:            int   = 2,
        n_post:           int   = 2,
        n_coarse:         int   = 20,
        min_coarse_cells: int   = 4,
    ):
        self.device           = device
        self.dtype            = dtype
        self.ax               = int(flow_axis)
        self.sign             = int(flow_sign)
        self.s                = float(thermal_dt_scale)
        self.omega            = float(omega)
        self.n_pre            = int(n_pre)
        self.n_post           = int(n_post)
        self.n_coarse         = int(n_coarse)
        self.min_coarse_cells = int(min_coarse_cells)
        self._ready           = False
        self._dt_cache:    Optional[float] = None
        self._shape_cache: Optional[tuple] = None
        self._levels: list = []
        self._nlevels: int  = 0

    # ─────────────────────────────────────────────────────── public API ───

    def build_if_needed(
        self,
        k:      Tensor,
        rho_cp: Tensor,
        u:      Tensor,
        v:      Tensor,
        w:      Tensor,
        mask:   Tensor,
        dt:     float,
    ) -> None:
        shape = tuple(k.shape)
        dt    = float(dt)
        if not self._ready or self._shape_cache != shape:
            self._build_hierarchy(k, rho_cp, u, v, w, mask)
            # force inv_diag rebuild after new hierarchy
            self._dt_cache = None
        if self._dt_cache != dt:
            self._update_inv_diag(dt)   # also sets self._ready = True
            self._dt_cache = dt

    def solve(
        self,
        b:          Tensor,
        x0:         Optional[Tensor] = None,
        max_cycles: int   = 50,
        tol:        float = 1e-6,
    ) -> Tuple[Tensor, dict]:
        """
        Apply V-cycles until ||b - Ax|| / ||b|| < tol or max_cycles reached.
        Returns (x_solution, info_dict).
        info keys: 'converged', 'it', 'res_rel', 'res0_rel'.
        """
        assert self._ready, "Call build_if_needed() before solve()."

        lv0    = self._levels[0]
        mask_f = lv0['mask_f']

        x = lv0['x']
        if x0 is not None:
            x.copy_(x0)
            x.mul_(mask_f)
        else:
            x.zero_()

        lv0['b_buf'].copy_(b)
        lv0['b_buf'].mul_(mask_f)

        b_norm    = float(b.norm()) + 1e-30
        res_rel   = float("inf")
        res0_rel  = float("inf")
        converged = False

        for cyc in range(max_cycles):
            self._vcycle(0)

            self._Mop(x, 0)
            lv0['r'].copy_(lv0['b_buf'])
            lv0['r'].sub_(lv0['Lo'])
            res_rel = float(lv0['r'].norm()) / b_norm
            if cyc == 0:
                res0_rel = res_rel
            if res_rel < tol:
                converged = True
                break

        result = x.clone()
        result.mul_(mask_f)
        return result, {
            "converged": converged,
            "it":        cyc + 1,
            "res_rel":   res_rel,
            "res0_rel":  res0_rel,
        }

    def L_apply_fine(self, T: Tensor) -> Tensor:
        """
        Evaluate L_steady(T) at the fine level. Returns a new tensor.
        Used for RHS assembly in _temperature_step_mg.
        """
        assert self._ready, "Call build_if_needed() first."
        lv = self._levels[0]
        self._fill_nb_into(T, 0)
        Lo = torch.empty_like(T)
        Lo.copy_(lv['diagL'] * T)
        Lo.addcmul_(lv['aE'], lv['Te'], value=-1.0)
        Lo.addcmul_(lv['aW'], lv['Tw'], value=-1.0)
        Lo.addcmul_(lv['aN'], lv['Tn'], value=-1.0)
        Lo.addcmul_(lv['aS'], lv['Ts'], value=-1.0)
        Lo.addcmul_(lv['aU'], lv['Tu'], value=-1.0)
        Lo.addcmul_(lv['aD'], lv['Td'], value=-1.0)
        return Lo

    # ──────────────────────────────────────── hierarchy build ─────────────

    def _build_hierarchy(
        self,
        k: Tensor, rho_cp: Tensor,
        u: Tensor, v: Tensor, w: Tensor,
        mask: Tensor,
    ) -> None:
        self._levels = []
        dev  = self.device
        dtyp = self.dtype
        ax   = self.ax
        sign = self.sign

        k_l   = k
        rcp_l = rho_cp
        u_l   = u
        v_l   = v
        w_l   = w
        msk_l = mask      # bool

        while True:
            nx, ny, nz = k_l.shape

            if   ax == 0: out_idx = nx - 1 if sign == +1 else 0
            elif ax == 1: out_idx = ny - 1 if sign == +1 else 0
            else:         out_idx = nz - 1 if sign == +1 else 0

            dx = float(2 ** len(self._levels))

            aE, aW, aN, aS, aU, aD, diagL = self._build_coeffs_full(
                k_l, rcp_l, u_l, v_l, w_l, ax, out_idx, dx, self.s
            )

            shape  = (nx, ny, nz)
            # mask_f: 1.0 = unknown, 0.0 = Dirichlet.
            # Used with mul_() instead of x[~mask]=0.0 to avoid graph breaks.
            mask_f = msk_l.to(dtyp)

            lv = {
                'shape':    shape,
                'out_idx':  out_idx,
                'aE': aE, 'aW': aW, 'aN': aN,
                'aS': aS, 'aU': aU, 'aD': aD,
                'diagL':    diagL,
                # inv_diagM filled by _update_inv_diag
                'inv_diagM': torch.empty(shape, device=dev, dtype=dtyp),
                'mask':     msk_l,
                'mask_f':   mask_f,
                # V-cycle work buffers
                'x':     torch.empty(shape, device=dev, dtype=dtyp),
                'r':     torch.empty(shape, device=dev, dtype=dtyp),
                'b_buf': torch.empty(shape, device=dev, dtype=dtyp),
                'Lo':    torch.empty(shape, device=dev, dtype=dtyp),
                # neighbor buffers
                'Te': torch.empty(shape, device=dev, dtype=dtyp),
                'Tw': torch.empty(shape, device=dev, dtype=dtyp),
                'Tn': torch.empty(shape, device=dev, dtype=dtyp),
                'Ts': torch.empty(shape, device=dev, dtype=dtyp),
                'Tu': torch.empty(shape, device=dev, dtype=dtyp),
                'Td': torch.empty(shape, device=dev, dtype=dtyp),
            }
            self._levels.append(lv)

            if min(nx // 2, ny // 2, nz // 2) <= self.min_coarse_cells:
                break

            k_l   = self._restrict_2x(k_l)
            rcp_l = self._restrict_2x(rcp_l)
            u_l   = self._restrict_2x(u_l)
            v_l   = self._restrict_2x(v_l)
            w_l   = self._restrict_2x(w_l)
            # coarse mask: True only if ALL 8 fine children are unknowns
            msk_l = self._restrict_2x(msk_l.to(dtyp)).ge(0.999).bool()

        self._nlevels     = len(self._levels)
        self._shape_cache = tuple(k.shape)
        shapes_str = " → ".join(str(lv['shape']) for lv in self._levels)
        print(f"[MG] {self._nlevels}-level hierarchy: {shapes_str}")

    def _update_inv_diag(self, dt: float) -> None:
        """
        Recompute inv_diagM = omega / (1 + dt * diagL) at every level.
        Zeros Dirichlet nodes via mul_(mask_f).
        Called from build_if_needed whenever dt changes.
        Sets self._ready = True.
        """
        for lv in self._levels:
            diagM = (1.0 + dt * lv['diagL']).clamp_(min=1e-12)
            lv['inv_diagM'].copy_(self.omega / diagM)
            lv['inv_diagM'].mul_(lv['mask_f'])   # zero Dirichlet nodes
        self._ready = True

    # ─────────────────────────────────────────────────── V-cycle ──────────

    def _vcycle(self, level: int) -> None:
        """
        In-place V-cycle at `level`.
        On entry:  lv['x'] = current iterate, lv['b_buf'] = RHS.
        On exit:   lv['x'] updated in-place.
        """
        lv        = self._levels[level]
        mask_f    = lv['mask_f']
        inv_diagM = lv['inv_diagM']
        x         = lv['x']
        r         = lv['r']
        b         = lv['b_buf']
        Lo        = lv['Lo']

        if level == self._nlevels - 1:
            # ── coarsest level: Jacobi solve ─────────────────────────────
            for _ in range(self.n_coarse):
                self._Mop(x, level)
                _smooth_step_compiled(x, b, Lo, inv_diagM, mask_f, r)
            return

        # ── pre-smooth ────────────────────────────────────────────────────
        for _ in range(self.n_pre):
            self._Mop(x, level)
            _smooth_step_compiled(x, b, Lo, inv_diagM, mask_f, r)

        # ── compute fine residual ─────────────────────────────────────────
        self._Mop(x, level)
        r.copy_(b)
        r.sub_(Lo)
        r.mul_(mask_f)

        # ── restrict residual to coarse RHS ───────────────────────────────
        lv_c = self._levels[level + 1]
        self._restrict_into(r, lv_c['b_buf'])

        # ── coarse correction (starts from zero) ──────────────────────────
        lv_c['x'].zero_()
        self._vcycle(level + 1)

        # ── prolongate and add to fine iterate ────────────────────────────
        nx, ny, nz = lv['shape']
        e = self._prolong_2x(lv_c['x'], nx, ny, nz)
        x.add_(e)
        x.mul_(mask_f)

        # ── post-smooth ────────────────────────────────────────────────────
        for _ in range(self.n_post):
            self._Mop(x, level)
            _smooth_step_compiled(x, b, Lo, inv_diagM, mask_f, r)

    # ──────────────────────────────────────────── operator apply ──────────

    def _Mop(self, T: Tensor, level: int) -> None:
        """
        Compute M(T) = T + dt * L_steady(T) into self._levels[level]['Lo'].
        """
        lv = self._levels[level]
        self._fill_nb_into(T, level)
        Lo = lv['Lo']
        Lo.copy_(lv['diagL'] * T)
        Lo.addcmul_(lv['aE'], lv['Te'], value=-1.0)
        Lo.addcmul_(lv['aW'], lv['Tw'], value=-1.0)
        Lo.addcmul_(lv['aN'], lv['Tn'], value=-1.0)
        Lo.addcmul_(lv['aS'], lv['Ts'], value=-1.0)
        Lo.addcmul_(lv['aU'], lv['Tu'], value=-1.0)
        Lo.addcmul_(lv['aD'], lv['Td'], value=-1.0)
        Lo.mul_(self._dt_cache)
        Lo.add_(T)

    # ────────────────────────────────────────── neighbor fill ─────────────

    def _fill_nb_into(self, T: Tensor, level: int) -> None:
        """
        Fill pre-allocated neighbor buffers Te/Tw/Tn/Ts/Tu/Td for `level`.
        Ghosting matches _thermal_stencil_coeffs_steady exactly.
        """
        lv      = self._levels[level]
        ax      = self.ax
        out_idx = lv['out_idx']
        Te, Tw  = lv['Te'], lv['Tw']
        Tn, Ts  = lv['Tn'], lv['Ts']
        Tu, Td  = lv['Tu'], lv['Td']

        # ── X ─────────────────────────────────────────────────────────────
        Te[:-1, :, :] = T[1:,  :, :]
        Tw[1:,  :, :] = T[:-1, :, :]
        Te[-1,  :, :] = T[-2,  :, :]     # high-x Neumann always
        if ax == 0 and out_idx == 0:
            Tw[0, :, :] = T[1, :, :]     # flow axis, outlet at low-x → Neumann
        elif ax != 0:
            Tw[0, :, :] = T[1, :, :]     # transverse → Neumann
        else:
            Tw[0, :, :] = T[0, :, :]     # flow axis, inlet at low-x → copy-self

        # ── Y ─────────────────────────────────────────────────────────────
        Tn[:, :-1, :] = T[:, 1:,  :]
        Ts[:, 1:,  :] = T[:, :-1, :]
        Tn[:, -1,  :] = T[:, -2,  :]     # high-y Neumann always
        if ax == 1 and out_idx == 0:
            Ts[:, 0, :] = T[:, 1, :]
        elif ax != 1:
            Ts[:, 0, :] = T[:, 1, :]
        else:
            Ts[:, 0, :] = T[:, 0, :]

        # ── Z ─────────────────────────────────────────────────────────────
        Tu[:, :, :-1] = T[:, :, 1:]
        Td[:, :, 1:]  = T[:, :, :-1]
        Tu[:, :, -1]  = T[:, :, -2]      # high-z Neumann always
        if ax == 2 and out_idx == 0:
            Td[:, :, 0] = T[:, :, 1]
        elif ax != 2:
            Td[:, :, 0] = T[:, :, 1]
        else:
            Td[:, :, 0] = T[:, :, 0]

    # ─────────────────────────────────── coefficient builder (static) ─────

    @staticmethod
    def _build_coeffs_full(
        k:       Tensor,
        rho_cp:  Tensor,
        u:       Tensor,
        v:       Tensor,
        w:       Tensor,
        ax:      int,
        out_idx: int,
        dx:      float,
        s:       float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns (aE, aW, aN, aS, aU, aD, diagL) for

            L_steady(T) = diagL*T - aE*T_E - aW*T_W - aN*T_N
                                  - aS*T_S - aU*T_U - aD*T_D

        Diffusion:  harmonic-mean face k, coeff = s * k_face / (rho_cp * dx²)
        Convection: first-order upwind,   coeff = s * |u_face| / dx

        s = thermal_dt_scale is folded in here so that the outer loop uses
        raw dt and diagM = 1 + dt * diagL is consistent everywhere.
        """
        inv_rcp = 1.0 / (rho_cp + 1e-30)

        def _shift(A: Tensor, dim: int, step: int) -> Tensor:
            B = A.clone()
            if dim == 0:
                if step == +1:
                    B[:-1, :, :] = A[1:,  :, :]
                    B[-1,  :, :] = A[-2,  :, :]
                else:
                    B[1:,  :, :] = A[:-1, :, :]
                    if ax == 0 and out_idx == 0:
                        B[0, :, :] = A[1, :, :]
                    elif ax != 0:
                        B[0, :, :] = A[1, :, :]
                    else:
                        B[0, :, :] = A[0, :, :]
            elif dim == 1:
                if step == +1:
                    B[:, :-1, :] = A[:, 1:,  :]
                    B[:, -1,  :] = A[:, -2,  :]
                else:
                    B[:, 1:,  :] = A[:, :-1, :]
                    if ax == 1 and out_idx == 0:
                        B[:, 0, :] = A[:, 1, :]
                    elif ax != 1:
                        B[:, 0, :] = A[:, 1, :]
                    else:
                        B[:, 0, :] = A[:, 0, :]
            else:
                if step == +1:
                    B[:, :, :-1] = A[:, :, 1:]
                    B[:, :, -1]  = A[:, :, -2]
                else:
                    B[:, :, 1:]  = A[:, :, :-1]
                    if ax == 2 and out_idx == 0:
                        B[:, :, 0] = A[:, :, 1]
                    elif ax != 2:
                        B[:, :, 0] = A[:, :, 1]
                    else:
                        B[:, :, 0] = A[:, :, 0]
            return B

        kE = _shift(k, 0, +1);  kW = _shift(k, 0, -1)
        kN = _shift(k, 1, +1);  kS = _shift(k, 1, -1)
        kU = _shift(k, 2, +1);  kD = _shift(k, 2, -1)

        kxp = 2.0 * k * kE / (k + kE + 1e-30)
        kxm = 2.0 * k * kW / (k + kW + 1e-30)
        kyp = 2.0 * k * kN / (k + kN + 1e-30)
        kym = 2.0 * k * kS / (k + kS + 1e-30)
        kzp = 2.0 * k * kU / (k + kU + 1e-30)
        kzm = 2.0 * k * kD / (k + kD + 1e-30)

        cd   = s / (dx * dx)
        aE_d = cd * kxp * inv_rcp
        aW_d = cd * kxm * inv_rcp
        aN_d = cd * kyp * inv_rcp
        aS_d = cd * kym * inv_rcp
        aU_d = cd * kzp * inv_rcp
        aD_d = cd * kzm * inv_rcp

        ca   = s / dx
        upx  = torch.clamp( u, min=0.0);  umx = torch.clamp(-u, min=0.0)
        upy  = torch.clamp( v, min=0.0);  umy = torch.clamp(-v, min=0.0)
        upz  = torch.clamp( w, min=0.0);  umz = torch.clamp(-w, min=0.0)

        aW_a = ca * upx;  aE_a = ca * umx
        aS_a = ca * upy;  aN_a = ca * umy
        aD_a = ca * upz;  aU_a = ca * umz

        aE = aE_d + aE_a
        aW = aW_d + aW_a
        aN = aN_d + aN_a
        aS = aS_d + aS_a
        aU = aU_d + aU_a
        aD = aD_d + aD_a

        diagL = aE + aW + aN + aS + aU + aD
        return aE, aW, aN, aS, aU, aD, diagL

    # ──────────────────────────────── restriction / prolongation ──────────

    @staticmethod
    def _restrict_2x(x: Tensor) -> Tensor:
        n0, n1, n2 = x.shape
        n0e = (n0 // 2) * 2;  n1e = (n1 // 2) * 2;  n2e = (n2 // 2) * 2
        return (
            x[:n0e, :n1e, :n2e]
            .view(n0e // 2, 2, n1e // 2, 2, n2e // 2, 2)
            .mean(dim=(1, 3, 5))
        )

    def _restrict_into(self, src: Tensor, dst: Tensor) -> None:
        n0, n1, n2 = src.shape
        n0e = (n0 // 2) * 2;  n1e = (n1 // 2) * 2;  n2e = (n2 // 2) * 2
        dst.copy_(
            src[:n0e, :n1e, :n2e]
            .view(n0e // 2, 2, n1e // 2, 2, n2e // 2, 2)
            .mean(dim=(1, 3, 5))
        )

    @staticmethod
    def _prolong_2x(xc: Tensor, nx: int, ny: int, nz: int) -> Tensor:
        return F.interpolate(
            xc[None, None],
            size=(nx, ny, nz),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
