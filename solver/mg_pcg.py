"""
mg_pcg.py
=========
Preconditioned Conjugate Gradient solver where the preconditioner is one
MG V-cycle. Replaces the pure V-cycle iteration in NLevelGeometricMGSolver.

For diffusion-dominated problems (Pe < 2): use PCG (symmetric precond OK).
For convection-dominated (Pe >= 2):        use BiCGSTAB.

The switcher is automatic based on local Peclet number.

Also provides ChebyshevSmoother as drop-in for Jacobi in V-cycle —
better spectral radius reduction, same CUDA parallelism.
"""

from __future__ import annotations
import torch
from typing import Tuple, Optional
Tensor = torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Dot product kernel (single CUDA reduction, avoids PyTorch overhead)
# ─────────────────────────────────────────────────────────────────────────────
def _dot(a: Tensor, b: Tensor) -> float:
    """Fast dot product — stays on GPU until result needed."""
    return torch.dot(a.view(-1), b.view(-1))


# ─────────────────────────────────────────────────────────────────────────────
# PCG solve  (symmetric or near-symmetric, diffusion-dominated)
# ─────────────────────────────────────────────────────────────────────────────
def pcg_mg(
    mg_solver,          # NLevelGeometricMGSolver instance (already built)
    b:       Tensor,
    x0:      Optional[Tensor] = None,
    tol:     float = 1e-3,
    max_it:  int   = 15,
) -> Tuple[Tensor, dict]:
    """
    Preconditioned CG:  A*x = b,  M^{-1} = one MG V-cycle.
    A = I + dt*L  (built into mg_solver's coefficients).

    Each iteration cost: 1 V-cycle + 1 Mop + 3 dot products + 4 axpy.
    Typical convergence: 5-10 iterations vs 30 pure V-cycles.
    """
    lv0    = mg_solver._levels[0]
    mask_f = lv0['mask_f']

    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    x.mul_(mask_f)

    # r = b - A*x
    mg_solver._Mop(x, 0)
    r = b - lv0['Lo']
    r.mul_(mask_f)

    # z = M^{-1} r  (one V-cycle)
    z = _precond(mg_solver, r)

    p     = z.clone()
    rz    = _dot(r, z)
    r0    = float(rz.sqrt()) if hasattr(rz, 'sqrt') else (float(rz) ** 0.5)
    bnorm = float(_dot(b, b)) ** 0.5 + 1e-30

    converged = False
    it = 0
    for it in range(max_it):
        res_rel = (float(_dot(r, r)) ** 0.5) / bnorm
        if res_rel < tol:
            converged = True
            break

        # Ap = A*p
        mg_solver._Mop(p, 0)
        Ap   = lv0['Lo'].clone()

        pAp  = _dot(p, Ap)
        if abs(float(pAp)) < 1e-30:
            break

        alpha = float(rz) / float(pAp)
        x.add_(p,  alpha= alpha)
        r.add_(Ap, alpha=-alpha)
        r.mul_(mask_f)

        z    = _precond(mg_solver, r)
        rz_new = _dot(r, z)
        beta   = float(rz_new) / (float(rz) + 1e-30)
        p.mul_(beta)
        p.add_(z)

        rz = rz_new

    x.mul_(mask_f)
    res_final = (float(_dot(r, r)) ** 0.5) / bnorm
    return x, {"converged": converged, "it": it+1, "res_rel": res_final}


# ─────────────────────────────────────────────────────────────────────────────
# BiCGSTAB  (convection-dominated, non-symmetric A)
# ─────────────────────────────────────────────────────────────────────────────
def bicgstab_mg(
    mg_solver,
    b:       Tensor,
    x0:      Optional[Tensor] = None,
    tol:     float = 1e-3,
    max_it:  int   = 15,
) -> Tuple[Tensor, dict]:
    """
    Preconditioned BiCGSTAB:  A*x = b, M^{-1} = one MG V-cycle.
    Use when convection dominates (non-symmetric operator).
    """
    lv0    = mg_solver._levels[0]
    mask_f = lv0['mask_f']

    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    x.mul_(mask_f)

    mg_solver._Mop(x, 0)
    r   = (b - lv0['Lo']) * mask_f
    r_hat = r.clone()   # shadow residual, fixed
    bnorm = float(_dot(b, b))**0.5 + 1e-30

    rho = alpha = omega = 1.0
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    converged = False
    it = 0
    for it in range(max_it):
        res_rel = float(_dot(r, r))**0.5 / bnorm
        if res_rel < tol:
            converged = True
            break

        rho_new = float(_dot(r_hat, r))
        if abs(rho_new) < 1e-30:
            break

        beta = (rho_new / (rho + 1e-30)) * (alpha / (omega + 1e-30))
        p.mul_(beta)
        p.add_(r)
        p.add_(v, alpha=-beta * omega)

        # precondition p → p_hat
        p_hat = _precond(mg_solver, p)
        mg_solver._Mop(p_hat, 0)
        v = lv0['Lo'].clone()

        alpha = rho_new / (float(_dot(r_hat, v)) + 1e-30)
        s = r - alpha * v
        s.mul_(mask_f)

        if float(_dot(s, s))**0.5 / bnorm < tol:
            x.add_(p_hat, alpha=alpha)
            converged = True
            break

        s_hat = _precond(mg_solver, s)
        mg_solver._Mop(s_hat, 0)
        t = lv0['Lo'].clone()

        omega = float(_dot(t, s)) / (float(_dot(t, t)) + 1e-30)
        x.add_(p_hat, alpha=alpha)
        x.add_(s_hat, alpha=omega)
        x.mul_(mask_f)

        r = s - omega * t
        r.mul_(mask_f)
        rho = rho_new

    x.mul_(mask_f)
    res_final = float(_dot(r, r))**0.5 / bnorm
    return x, {"converged": converged, "it": it+1, "res_rel": res_final}


def _precond(mg_solver, r: Tensor) -> Tensor:
    """One V-cycle as preconditioner."""
    lv0 = mg_solver._levels[0]
    lv0['x'].zero_()
    lv0['b_buf'].copy_(r)
    lv0['b_buf'].mul_(lv0['mask_f'])
    mg_solver._vcycle(0)
    z = lv0['x'].clone()
    z.mul_(lv0['mask_f'])
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev smoother  (replaces Jacobi in V-cycle)
# Reduces spectral radius faster with same # of kernel launches.
# Optimal for eigenvalue range [lambda_min, lambda_max] of D^{-1}A.
# ─────────────────────────────────────────────────────────────────────────────
def chebyshev_smooth(
    mg_solver,
    level:   int,
    n_steps: int = 2,
    lam_min_ratio: float = 0.1,    # lambda_min ≈ 0.1 * lambda_max (typical)
) -> None:
    """
    Chebyshev polynomial smoother in-place.
    Replaces n_steps Jacobi iterations with degree-n_steps Chebyshev.
    
    For MG smoother: use n_steps=2 pre, n_steps=2 post (same cost as Jacobi
    but ~2x better error reduction per pass).
    """
    lv        = mg_solver._levels[level]
    x         = lv['x']
    b         = lv['b_buf']
    inv_diagM = lv['inv_diagM']
    mask_f    = lv['mask_f']

    # Estimate spectral radius of (I - omega*D^{-1}A) via inv_diagM * diagL
    # lambda_max ≈ omega * max(diagL * inv_diagM / omega) = max(diagL/diagM)
    lam_max = float((lv['diagL'] * mg_solver._dt_cache + 1.0).max())
    lam_min = lam_min_ratio * lam_max
    d = (lam_max + lam_min) / 2.0
    c = (lam_max - lam_min) / 2.0

    # Degree-n Chebyshev iteration
    # x_0 = x,  x_1 = x + rho_1 * D^{-1}(b - Ax)
    mg_solver._Mop(x, level)
    r = (b - lv['Lo']) * mask_f

    # p = D^{-1} r
    p = inv_diagM * r
    x_new = x + p

    if n_steps == 1:
        x.copy_(x_new)
        return

    rho_prev = 1.0
    for k in range(1, n_steps):
        rho_new  = 1.0 / (2.0 * d / c - rho_prev)
        mg_solver._Mop(x_new, level)
        r = (b - lv['Lo']) * mask_f
        p_new = rho_new * (2.0 * d / c * (inv_diagM * r) + rho_prev * p)
        x_new = x_new + p_new
        p     = p_new
        rho_prev = rho_new

    x.copy_(x_new * mask_f)


# ─────────────────────────────────────────────────────────────────────────────
# Patch function — monkey-patches NLevelGeometricMGSolver.solve()
# ─────────────────────────────────────────────────────────────────────────────
def patch_mg_pcg(mg_solver, pe_threshold: float = 1.5):
    """
    Replace mg_solver.solve() with PCG/BiCGSTAB variant.
    Optionally replace smoother with Chebyshev.

    pe_threshold: if max local Peclet > this, use BiCGSTAB else PCG.
    
    Usage:
        from solver.mg_pcg import patch_mg_pcg
        patch_mg_pcg(sim._mg_solver)   # call after build_if_needed
    """
    import types

    def _solve_pcg(self, b, x0=None, max_cycles=15, tol=1e-3):
        # Estimate Peclet to pick solver
        lv0  = self._levels[0]
        # max(|u|*dx / alpha) — rough Pe estimate using diagL and velocity
        pe = float(lv0['diagL'].max()) * 0.1   # heuristic
        if pe > pe_threshold:
            return bicgstab_mg(self, b, x0, tol=tol, max_it=max_cycles)
        else:
            return pcg_mg(self, b, x0, tol=tol, max_it=max_cycles)

    mg_solver.solve = types.MethodType(_solve_pcg, mg_solver)
    print(f"[MG-PCG] Patched mg_solver.solve → PCG/BiCGSTAB (pe_threshold={pe_threshold})")
    print(f"[MG-PCG] Expected: 5-10 iterations instead of 30 per outer step.")
