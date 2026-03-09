# nlevel_mg.py
"""
Standalone N-level geometric multigrid solver for the CHT thermal system.

    A theta = b,    A = I + dt * L_steady

L_steady = harmonic-mean diffusion + first-order upwind convection.
s = thermal_dt_scale is baked into coefficients at build time.
dt passed to build_if_needed / solve is the RAW outer-loop pseudo-time step.

Smoother: weighted Jacobi (optimal on GPU — RBGS has no benefit on CUDA).

Backend:
  If mg_cuda_ext loads successfully, _Mop and _smooth_step use the fused
  CUDA kernels from mg_kernels.cu.  This replaces ~20 separate kernel
  launches per _Mop call with 1, and the smooth step with 1 more.

  If the extension is unavailable (CPU, compile failure) the code falls back
  to the pure-PyTorch implementation transparently.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# Try to load the fused CUDA extension.
# mg_cuda_ext sets MG_CUDA_AVAILABLE and MG_CUDA at import time.
try:
    from solver.mg_cuda_ext import MG_CUDA, MG_CUDA_AVAILABLE
except ImportError:
    MG_CUDA           = None
    MG_CUDA_AVAILABLE = False


class NLevelGeometricMGSolver:

    def __init__(
        self,
        device:           torch.device,
        dtype:            torch.dtype,
        flow_axis:        int,
        flow_sign:        int,
        thermal_dt_scale: float,
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

        # Use fused CUDA kernels only when on CUDA and float32
        # (kernels are written for float32; float64 falls through to PyTorch)
        self._use_cuda_ext = (
            MG_CUDA_AVAILABLE
            and device.type == "cuda"
            and dtype == torch.float32
        )
        if self._use_cuda_ext:
            None #print("[MG] Using fused CUDA kernels for _Mop and smooth step.")
        else:
            print("[MG] Using PyTorch fallback for _Mop and smooth step.")

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
            self._dt_cache = None
        if self._dt_cache != dt:
            self._update_inv_diag(dt)
            self._dt_cache = dt

    def solve(
        self,
        b:          Tensor,
        x0:         Optional[Tensor] = None,
        max_cycles: int   = 50,
        tol:        float = 1e-6,
    ) -> Tuple[Tensor, dict]:
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

            # Divergence detection: abort if residual is growing or NaN
            if not (res_rel < float("inf")):   # catches NaN
                converged = False
                break
            if cyc > 2 and res_rel > 10.0 * res0_rel:  # diverging, not just stagnating
                converged = False
                break
            if res_rel < tol:
                converged = True
                break

        result = x.clone()
        result.mul_(mask_f)
        diverged = (not converged) and (
            not (res_rel < float("inf"))          # NaN
            or (cyc > 2 and res_rel > 10.0 * res0_rel)  # growing residual
        )
        return result, {
            "converged": converged,
            "diverged":  diverged,
            "it":        cyc + 1,
            "res_rel":   res_rel,
            "res0_rel":  res0_rel,
        }

    def L_apply_fine(self, T: Tensor) -> Tensor:
        """Evaluate L_steady(T) at fine level. Returns a new tensor."""
        assert self._ready
        lv = self._levels[0]
        Lo = torch.empty_like(T)
        if self._use_cuda_ext:
            self._Mop_cuda(T, 0, Lo)
            # Lo currently holds M(T) = T + dt*L(T); we want just L(T)
            # L(T) = (Lo - T) / dt
            Lo.sub_(T)
            Lo.div_(self._dt_cache)
        else:
            self._fill_nb_into(T, 0)
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
        msk_l = mask

        while True:
            nx, ny, nz = k_l.shape

            if   ax == 0: out_idx = nx - 1 if sign == +1 else 0
            elif ax == 1: out_idx = ny - 1 if sign == +1 else 0
            else:         out_idx = nz - 1 if sign == +1 else 0

            dx    = float(2 ** len(self._levels))
            shape = (nx, ny, nz)

            aE, aW, aN, aS, aU, aD, diagL = self._build_coeffs_full(
                k_l, rcp_l, u_l, v_l, w_l, ax, out_idx, dx, self.s
            )

            mask_f = msk_l.to(dtyp)

            lv = {
                'shape':    shape,
                'out_idx':  out_idx,
                'aE': aE, 'aW': aW, 'aN': aN,
                'aS': aS, 'aU': aU, 'aD': aD,
                'diagL':   diagL,
                'mask_f':  mask_f,
                # filled by _update_inv_diag
                'inv_diagM': torch.empty(shape, device=dev, dtype=dtyp),
                # V-cycle work buffers
                'x':     torch.empty(shape, device=dev, dtype=dtyp),
                'r':     torch.empty(shape, device=dev, dtype=dtyp),
                'b_buf': torch.empty(shape, device=dev, dtype=dtyp),
                'Lo':    torch.empty(shape, device=dev, dtype=dtyp),
                # neighbor buffers (used only by PyTorch fallback _Mop)
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
            msk_l = self._restrict_2x(msk_l.to(dtyp)).ge(0.999).bool()

        self._nlevels     = len(self._levels)
        self._shape_cache = tuple(k.shape)
        shapes_str = " → ".join(str(lv['shape']) for lv in self._levels)
        print(f"[MG] {self._nlevels}-level hierarchy: {shapes_str}")

    def _update_inv_diag(self, dt: float) -> None:
        for lv in self._levels:
            diagM = (1.0 + dt * lv['diagL']).clamp_(min=1e-12)
            lv['inv_diagM'].copy_(self.omega / diagM)
            lv['inv_diagM'].mul_(lv['mask_f'])
        self._ready = True

    # ─────────────────────────────────────────────────── V-cycle ──────────

    def _vcycle(self, level: int) -> None:
        lv        = self._levels[level]
        mask_f    = lv['mask_f']
        inv_diagM = lv['inv_diagM']
        x         = lv['x']
        r         = lv['r']
        b         = lv['b_buf']

        if level == self._nlevels - 1:
            for _ in range(self.n_coarse):
                self._Mop(x, level)
                self._smooth_step(x, b, lv['Lo'], inv_diagM, mask_f, r)
            return

        for _ in range(self.n_pre):
            self._Mop(x, level)
            self._smooth_step(x, b, lv['Lo'], inv_diagM, mask_f, r)

        self._Mop(x, level)
        r.copy_(b)
        r.sub_(lv['Lo'])
        r.mul_(mask_f)

        lv_c = self._levels[level + 1]
        self._restrict_into(r, lv_c['b_buf'])

        lv_c['x'].zero_()
        self._vcycle(level + 1)

        nx, ny, nz = lv['shape']
        e = self._prolong_2x(lv_c['x'], nx, ny, nz)
        x.add_(e, alpha=0.75)    # damping factor — prevents interface overshoot
        x.mul_(mask_f)

        for _ in range(self.n_post):
            self._Mop(x, level)
            self._smooth_step(x, b, lv['Lo'], inv_diagM, mask_f, r)

    # ───────────────────────────── operator apply (dispatched) ────────────

    def _Mop(self, T: Tensor, level: int) -> None:
        """Compute M(T) = T + dt*L(T) into levels[level]['Lo']."""
        if self._use_cuda_ext:
            self._Mop_cuda(T, level, self._levels[level]['Lo'])
        else:
            self._Mop_pytorch(T, level)

    def _Mop_cuda(self, T: Tensor, level: int, Lo: Tensor) -> None:
        """
        Single fused kernel: Lo = T + dt * L(T).
        Handles neighbor ghosting internally — no fill_nb_into needed.
        T and Lo must be contiguous float32 CUDA tensors.
        """
        lv = self._levels[level]
        MG_CUDA.mop_fused(
            T.contiguous(),
            lv['aE'], lv['aW'], lv['aN'],
            lv['aS'], lv['aU'], lv['aD'],
            lv['diagL'],
            Lo,
            float(self._dt_cache),
            int(self.ax),
            int(lv['out_idx']),
        )

    def _Mop_pytorch(self, T: Tensor, level: int) -> None:
        """PyTorch fallback. Fills neighbor buffers then evaluates stencil."""
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

    # ────────────────────────────── smooth step (dispatched) ──────────────

    def _smooth_step(
        self,
        x:         Tensor,
        b:         Tensor,
        Lo:        Tensor,
        inv_diagM: Tensor,
        mask_f:    Tensor,
        r:         Tensor,
    ) -> None:
        """x += inv_diagM * (b - Lo), then zero Dirichlet nodes via mask_f."""
        if self._use_cuda_ext:
            # Fused kernel: x += inv_diagM * (b - Lo).
            # inv_diagM is pre-zeroed at Dirichlet nodes so no mask needed.
            MG_CUDA.jacobi_smooth(x, b, Lo, inv_diagM)
        else:
            r.copy_(b)
            r.sub_(Lo)
            r.mul_(mask_f)
            x.addcmul_(inv_diagM, r)
            x.mul_(mask_f)

    # ────────────────────────────────────────── neighbor fill (fallback) ──

    def _fill_nb_into(self, T: Tensor, level: int) -> None:
        """Fill neighbor buffers. Used only by PyTorch fallback _Mop_pytorch."""
        lv      = self._levels[level]
        ax      = self.ax
        out_idx = lv['out_idx']
        Te, Tw  = lv['Te'], lv['Tw']
        Tn, Ts  = lv['Tn'], lv['Ts']
        Tu, Td  = lv['Tu'], lv['Td']

        Te[:-1, :, :] = T[1:,  :, :]
        Tw[1:,  :, :] = T[:-1, :, :]
        Te[-1,  :, :] = T[-2,  :, :]
        if ax == 0 and out_idx == 0:
            Tw[0, :, :] = T[1, :, :]
        elif ax != 0:
            Tw[0, :, :] = T[1, :, :]
        else:
            Tw[0, :, :] = T[0, :, :]

        Tn[:, :-1, :] = T[:, 1:,  :]
        Ts[:, 1:,  :] = T[:, :-1, :]
        Tn[:, -1,  :] = T[:, -2,  :]
        if ax == 1 and out_idx == 0:
            Ts[:, 0, :] = T[:, 1, :]
        elif ax != 1:
            Ts[:, 0, :] = T[:, 1, :]
        else:
            Ts[:, 0, :] = T[:, 0, :]

        Tu[:, :, :-1] = T[:, :, 1:]
        Td[:, :, 1:]  = T[:, :, :-1]
        Tu[:, :, -1]  = T[:, :, -2]
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
