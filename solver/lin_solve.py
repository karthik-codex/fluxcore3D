# linear_solvers.py
import os
import sys
import tempfile
import subprocess
import shutil
from typing import Callable, Optional, Tuple
import numpy as np
import torch
from scipy.sparse import csr_matrix, save_npz
from solver.amgx_worker import solve_once


class SparseLinearSolvers:
    """
    A self-contained collection of Krylov solvers + helper utilities.

    Usage:
        solvers = SparseLinearSolvers(
            device="cuda",
            amgx_config_path="path/to/config.json",   # optional
            amgx_env=dict(AMGX_DIR="...", CUDA_DIR="...")  # optional
        )
        x = solvers.solve(K_red, -R_red, method="bicgstab",
                          preconditioner=("physics", nv, mass_vals),
                          tol=1e-8, max_iter=500)
    """

    def __init__(
        self,
        device: str = "cuda",
        amgx_config_path: Optional[str] = None,
        amgx_env: Optional[dict] = None,
    ):
        self.device = device
        self.amgx_config_path = amgx_config_path
        self.amgx_env = amgx_env or {}

        # Lazy state for external libs (e.g., HYPRE handles) can live here.
        self._hypre_built = False

    # ---------------------------
    # Public unified entry point
    # ---------------------------
    def solve(
        self,
        K: torch.Tensor,
        rhs: torch.Tensor,
        method: str = "bicgstab",
        *,
        preconditioner: Tuple[str, int, torch.Tensor] = ("none", 0, torch.empty(0)),
        tol: float = 1e-8,
        max_iter: int = 500,
        restart: int = 80,
        lsq_tol: float = 1e-10,
        ilu_drop_tol: float = 1e-3,
        ilu_fill_factor: float = 10.0,
        amgx_config_path: Optional[str] = None,
        state_np1: dict = None,
    ) -> torch.Tensor:
        """
        method:
          - "bicgstab"
          - "gmres_gpu"            (torch GMRES with optional equilibration)
          - "gmres_cpu_ilu"        (SciPy GMRES + ILU precond on CPU)
          - "hypre"                (via hypre_py; expects hypre_py installed)
          - "amgx"                 (in a clean out-of-proc worker)
          - "normal_eq_cg"         (CG on normal equations)
        preconditioner:
          - ("none", _, _)
          - ("jacobi", _, _)
          - ("physics", nv, mass_vals)
        """
        method = method.lower()
        if method == "bicgstab":
            return self._solve_sparse_bicgstab(K, rhs, tol=tol, max_iter=max_iter, M=preconditioner)

        elif method == "gmres_gpu":
            # lightweight right-equilibrated GMRES you were using
            K_csr = K.to_sparse_csr()
            if K_csr.dtype != torch.float64:
                K_csr = K_csr.to(torch.float64)
            b = rhs.contiguous().to(dtype=K_csr.dtype, device=K_csr.device)

            A_mv, post, D = self._make_equilibrated_right_mv(K_csr)
            b_scaled = D * b  # <-- scale RHS
            # Optional left preconditioner (Jacobi / physics) for GMRES:
            M_inv = self._make_left_preconditioner(K, preconditioner, contact_state=state_np1)

            x, info = self._gmres(A_mv=A_mv, b=b_scaled, x0=None, M_inv=M_inv,
                                  tol=tol, max_iter=max_iter, restart=restart, verbose=False)
            return post(x)

        elif method == "gmres_cpu_ilu":
            return self._solve_sparse_gmres_ilu_cpu(K, rhs, tol=tol, max_iter=max_iter,
                                                    restart=restart, ilu_drop_tol=ilu_drop_tol,
                                                    ilu_fill_factor=ilu_fill_factor)

        elif method == "amgx":
            cfg = amgx_config_path or self.amgx_config_path
            if cfg is None:
                raise ValueError("[AMGX] Missing amgx_config_path.")
            return self._solve_sparse_amgx_outproc(K.to_sparse_csr(), rhs, cfg)

        elif method == "normal_eq_cg":
            return self._solve_sparse_normal_eq(K, rhs, tol=lsq_tol, max_iter=max_iter)

        else:
            raise ValueError(f"Unknown solver method: {method!r}")

    # ---------------------------
    # Preconditioners (closures)
    # ---------------------------
    def _make_left_preconditioner(
        self,
        K: torch.Tensor,
        M_tuple: Tuple[str, int, torch.Tensor],
        contact_state: dict = None,
    ):
        pre_type, nv, mass_vals = M_tuple
        pre_type = (pre_type or "none").lower()

        if pre_type == "jacobi":
            return self._build_diag_preconditioner_from_any(K)
        elif pre_type == "physics":
            return self._build_physics_preconditioner(K, nv, mass_vals, contact_state=contact_state)
        else:
            def M_inv_identity(v: torch.Tensor) -> torch.Tensor:
                return v
            return M_inv_identity

    def _build_physics_preconditioner1(
        self,
        K_red: torch.Tensor,
        nv: int,
        mass_vals: torch.Tensor,
        gamma_int_v: float = 1e-2,
        gamma_int_x: float = 1.0,
        contact_state: dict = None,
    ):
        """
        Physics-flavored diagonal left preconditioner.

        General case (v,x):
        v-block: Minv_v = 1 / (M + γ_v * |diag(K_vv)|)
        x-block: Minv_x = 1 / (γ_x * |diag(K_xx)| + ε)

        x-only case:
        Treat whole K as the x-block:
        Minv   = 1 / (γ_x * |diag(K)| + ε)

        Args
        ----
        K_red : sparse COO (coalesced) of shape (ndof, ndof)
        nv    : number of velocity DOFs (0 for x-only)
        mass_vals : (nv,) mass per v-DOF (ignored if nv==0)

        Returns
        -------
        M_inv: callable v -> precond(v)
        """
        K_red = K_red.coalesce()
        device, dtype = K_red.device, K_red.dtype
        ndof = K_red.size(0)

        row, col = K_red.indices()
        val = K_red.values()

        # diagonal magnitudes (cheap, robust)
        diag = torch.zeros(ndof, device=device, dtype=dtype)
        mask = (row == col)
        if mask.any():
            diag.index_add_(0, row[mask], val[mask])
        ad = diag.abs().clamp(min=1e-30)

        if nv <= 0 or nv >= ndof:
            # x-only (nv==0) or degenerate (nv==ndof → treat as v-only but safer to use x-form)
            # ε scales with diag to remain unit-consistent and avoid zero division
            eps_x = 1e-6 * float(ad.max())
            Minv = 1.0 / (gamma_int_x * ad + eps_x)

            def M_inv(v: torch.Tensor) -> torch.Tensor:
                return Minv * v

            return M_inv

        # (v,x) split
        diag_v = ad[:nv]
        diag_x = ad[nv:]

        # v-block: mass + small stiffness blend
        # mass_vals should be length nv; if not, fall back gracefully to diag_v
        if (mass_vals is None) or (mass_vals.numel() != nv):
            M_v = diag_v  # fallback; still stabilizing
        else:
            M_v = mass_vals.to(device=device, dtype=dtype)

        Minv_v = 1.0 / (M_v + gamma_int_v * diag_v)

        # x-block: stiffness-like diagonal with ε guard
        eps_x = 1e-6 * float(diag_x.max())
        Minv_x = 1.0 / (gamma_int_x * diag_x + eps_x)

        Minv = torch.cat([Minv_v, Minv_x], dim=0)

        def M_inv(v: torch.Tensor) -> torch.Tensor:
            return Minv * v

        return M_inv

    def _build_physics_preconditioner(
        self,
        K_red: torch.Tensor,
        nv: int,
        mass_vals: torch.Tensor,
        gamma_int_x: float = 1.0,
        contact_state: dict = None,
    ):
        K_red = K_red.coalesce()
        device, dtype = K_red.device, K_red.dtype
        ndof = K_red.size(0)

        row, col = K_red.indices()
        val = K_red.values()

        diag = torch.zeros(ndof, device=device, dtype=dtype)
        mask = (row == col)
        if mask.any():
            diag.index_add_(0, row[mask], val[mask])
        ad = diag.abs().clamp(min=1e-30)

        # --- Contact boost ---
        if contact_state is not None and contact_state.get("active", False):
            idx = contact_state["idx"]
            active_mask = contact_state.get("active_mask", None)
            
            rho_N = float(contact_state.get("rho_N", 0.0))
            A_i = contact_state.get("A_i", None)
            Vcnt = contact_state.get("Vcnt", None)
            
            if rho_N > 0 and A_i is not None and Vcnt is not None:
                if active_mask is not None and active_mask.any():
                    active_idx = idx[active_mask]
                    A_active = A_i[active_mask]
                    V_active = Vcnt[active_mask]
                else:
                    active_idx = idx
                    A_active = A_i
                    V_active = Vcnt
                    
                w_cnt = rho_N * A_active / (V_active + 1e-30)
                
                # Apply boost to all 3 DOFs of each active contact particle
                for d in range(3):
                    dof_idx = 3 * active_idx + d
                    ad[dof_idx] = ad[dof_idx] + w_cnt  # <-- THIS WAS MISSING

        # x-only path
        eps_x = 1e-6 * float(ad.max())
        Minv = 1.0 / (gamma_int_x * ad + eps_x)

        def M_inv(v: torch.Tensor) -> torch.Tensor:
            return Minv * v

        return M_inv
    
    def _build_diag_preconditioner_from_any(self, A: torch.Tensor, eps: float = 1e-12):
        assert A.size(0) == A.size(1), "A must be square."
        n = A.size(0)
        device, dtype = A.device, A.dtype

        if A.layout == torch.sparse_csr:
            crow = A.crow_indices()
            col = A.col_indices()
            val = A.values()
            row = torch.arange(n, device=device)
            row = torch.repeat_interleave(row, crow[1:] - crow[:-1])
            diag = torch.zeros(n, device=device, dtype=dtype)
            mask = (row == col)
            if mask.any():
                diag.index_add_(0, row[mask], val[mask])
        elif A.is_sparse:
            idx = A.indices()
            row, col, val = idx[0], idx[1], A.values()
            diag = torch.zeros(n, device=device, dtype=dtype)
            mask = (row == col)
            if mask.any():
                diag.index_add_(0, row[mask], val[mask])
        else:
            diag = torch.diag(A)

        M_inv_diag = 1.0 / torch.clamp(diag, min=eps)
        def M_inv(r: torch.Tensor) -> torch.Tensor:
            return r * M_inv_diag
        return M_inv

    # ---------------------------
    # Core solvers
    # ---------------------------
    def _solve_sparse_bicgstab(self, K, rhs, tol=1e-6, max_iter=500, M=("none", 0, torch.empty(0))):
        """BiCGSTAB with optional left preconditioner (jacobi | physics | none)."""
        K = K.coalesce()
        device, dtype = K.device, K.dtype
        n = K.size(0)

        pre_type, nv_passed, mass_vals_passed = M

        b = rhs.view(-1).to(device=device, dtype=dtype)
        x = torch.zeros(n, dtype=dtype, device=device)

        def A_mv(v):
            return torch.sparse.mm(K, v.view(-1,1)).view(-1)

        if pre_type == "jacobi":
            row, col = K.indices()
            val = K.values()
            diag = torch.zeros(n, dtype=dtype, device=device)
            # accumulate diagonal (in case of duplicates)
            mask = (row == col)           
    
            if mask.any():
                diag.index_add_(0, row[mask], val[mask])
                
            # --- SAFE DIAGONAL FLOOR (IMPORTANT CHANGE) ---
            diag_abs = diag.abs()
            max_diag = diag_abs.max() 
                            
            if max_diag.item() == 0.0:
                # completely hopeless diagonal -> don't precondition
                def M_inv(v):
                    return v
            else:
                # relative floor: e.g. 1e-6 of the max diagonal
                eps_rel = 1e-6
                eps = eps_rel * max_diag

                # clamp SMALL diagonals up to ±eps (preserve sign)
                diag_safe = torch.where(
                    diag_abs >= eps,
                    diag,
                    eps * diag.sign().where(diag_abs > 0, torch.ones_like(diag))  # sign or +1
                )

                Minv = 1.0 / diag_safe

                def M_inv(v):  # left preconditioning
                    return Minv * v
        elif pre_type == "physics":
            M_inv = self._build_physics_preconditioner(
                K,              # K_red
                nv_passed,             # size of v-block
                mass_vals_passed,      # lumped masses
                gamma_int_v=1e-2,
                gamma_int_x=1e-3
            )                
        else:
            def M_inv(v):  # identity
                return v

        r = b - A_mv(x)
        r_hat = r.clone()
        rho_old = torch.tensor(1.0, dtype=dtype, device=device)
        alpha = torch.tensor(1.0, dtype=dtype, device=device)
        omega = torch.tensor(1.0, dtype=dtype, device=device)
        v = torch.zeros_like(r)
        p = torch.zeros_like(r)

        norm_b = torch.norm(b)
        if norm_b.item() == 0.0:
            return torch.zeros_like(b)

        for _ in range(1, max_iter + 1):
            rho = torch.dot(r_hat, r)
            if abs(rho.item()) < 1e-30:
                break

            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

            y = M_inv(p)
            v = A_mv(y)

            denom = torch.dot(r_hat, v)
            if abs(denom.item()) < 1e-30:
                break
            alpha = rho / denom

            s = r - alpha * v
            nrm_s = torch.norm(s)
            if nrm_s.item() < tol * (norm_b.item() + 1e-30):
                x = x + alpha * y
                break

            z = M_inv(s)
            t = A_mv(z)

            tt = torch.dot(t, t)
            if tt.item() < 1e-30:
                omega = torch.tensor(0.0, dtype=dtype, device=device)
            else:
                omega = torch.dot(t, s) / tt

            x = x + alpha * y + omega * z
            r = s - omega * t

            if torch.norm(r).item() < tol * (norm_b.item() + 1e-30):
                break
            if abs(omega.item()) < 1e-30:
                break

            rho_old = rho

        return x

    # --- NEW: matrix-free BiCGSTAB (GPU) ---
    def bicgstab_matfree_OLD(
        self,
        A_mv,                       # callable: v -> A v
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        M_inv=None,                 # callable: v -> M^{-1} v
        *,
        tol: float = 1e-6,
        max_iter: int = 400,
    ):
        """
        Left-preconditioned BiCGSTAB for matrix-free operator.
        Solves: A x = b
        Uses: M_inv as left preconditioner (identity if None).
        """
        if M_inv is None:
            def M_inv(v): return v

        b = b.contiguous()
        x = torch.zeros_like(b) if x0 is None else x0.contiguous().clone()

        r = (b - A_mv(x)).contiguous()
        r_hat = r.clone()

        norm_b = torch.linalg.norm(b)
        if float(norm_b) == 0.0:
            return x, {"converged": True, "it": 0, "res_norm": 0.0}

        rho_old = torch.tensor(1.0, device=b.device, dtype=b.dtype)
        alpha   = torch.tensor(1.0, device=b.device, dtype=b.dtype)
        omega   = torch.tensor(1.0, device=b.device, dtype=b.dtype)

        v = torch.zeros_like(b)
        p = torch.zeros_like(b)

        # initial residual check (preconditioned)
        z = M_inv(r)
        res0 = torch.linalg.norm(z)
        if float(res0) <= tol * float(norm_b):
            return x, {"converged": True, "it": 0, "res_norm": float(res0)}

        converged = False
        res_norm = float(res0)

        for it in range(1, max_iter + 1):
            rho = torch.dot(r_hat, r)
            if float(rho.abs()) < 1e-30:
                break

            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

            y = M_inv(p)
            v = A_mv(y)

            denom = torch.dot(r_hat, v)
            if float(denom.abs()) < 1e-30:
                break

            alpha = rho / denom
            s = r - alpha * v

            z_s = M_inv(s)
            res_s = torch.linalg.norm(z_s)
            if float(res_s) <= tol * float(norm_b):
                x = x + alpha * y
                converged = True
                res_norm = float(res_s)
                break

            t = A_mv(z_s)
            tt = torch.dot(t, t)
            if float(tt.abs()) < 1e-30:
                break

            omega = torch.dot(t, s) / tt
            x = x + alpha * y + omega * z_s
            r = s - omega * t

            z = M_inv(r)
            res = torch.linalg.norm(z)
            res_norm = float(res)

            if res_norm <= tol * float(norm_b):
                converged = True
                break

            if float(omega.abs()) < 1e-30:
                break

            rho_old = rho

        return x, {"converged": bool(converged), "it": int(it), "res_norm": float(res_norm)}

    def bicgstab_matfree(
        self,
        A_mv,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        M_inv=None,
        *,
        tol: float = 1e-6,
        max_iter: int = 400,
        max_restarts: int = 5,
    ):
        if M_inv is None:
            def M_inv(v): return v

        b = b.contiguous()
        x = torch.zeros_like(b) if x0 is None else x0.contiguous().clone()

        norm_b = float(torch.linalg.norm(b))
        if norm_b == 0.0:
            return x, {"converged": True, "it": 0, "res_norm": 0.0, "res_rel": 0.0}

        # Compute initial true residual
        r = (b - A_mv(x)).contiguous()
        res0 = float(torch.linalg.norm(r))
        if res0 == 0.0:
            return x, {"converged": True, "it": 0, "res_norm": 0.0, "res_rel": 0.0}

        abs_tol   = tol * res0
        converged = False
        res_norm  = res0
        total_it  = 0
        n_restart = 0

        while total_it < max_iter:
            # ── (Re)initialize shadow residual and direction vectors ──────────────
            r_hat   = r.clone()
            rho_old = torch.tensor(1.0, device=b.device, dtype=b.dtype)
            alpha   = torch.tensor(1.0, device=b.device, dtype=b.dtype)
            omega   = torch.tensor(1.0, device=b.device, dtype=b.dtype)
            v       = torch.zeros_like(b)
            p       = torch.zeros_like(b)

            breakdown = False

            for it in range(1, max_iter - total_it + 1):
                total_it += 1

                rho = torch.dot(r_hat, r)

                # Breakdown: shadow residual lost orthogonality → restart
                if float(rho.abs()) < 1e-10 * float(torch.linalg.norm(r)) * float(torch.linalg.norm(r_hat)):
                    breakdown = True
                    break

                beta  = (rho / rho_old) * (alpha / omega)
                p     = r + beta * (p - omega * v)

                y     = M_inv(p)
                v     = A_mv(y)

                denom = torch.dot(r_hat, v)
                if float(denom.abs()) < 1e-30:
                    breakdown = True
                    break

                alpha = rho / denom
                s     = r - alpha * v

                res_s = float(torch.linalg.norm(s))
                if res_s <= abs_tol:
                    x         = x + alpha * y
                    res_norm  = res_s
                    converged = True
                    break

                z_s   = M_inv(s)
                t     = A_mv(z_s)

                tt = float(torch.dot(t, t))
                if abs(tt) < 1e-30:
                    breakdown = True
                    break

                omega    = torch.dot(t, s) / tt
                x        = x + alpha * y + omega * z_s
                r        = s - omega * t
                res_norm = float(torch.linalg.norm(r))

                if res_norm <= abs_tol:
                    converged = True
                    break

                if float(omega.abs()) < 1e-30:
                    breakdown = True
                    break

                rho_old = rho

            if converged:
                break

            if breakdown and n_restart < max_restarts:
                # Recompute r from scratch (avoids floating-point drift accumulation too)
                r = (b - A_mv(x)).contiguous()
                res_norm = float(torch.linalg.norm(r))
                n_restart += 1
                if res_norm <= abs_tol:
                    converged = True
                    break
                continue

            break   # max_iter or max_restarts exhausted

        return x, {
            "converged":   bool(converged),
            "it":          int(total_it),
            "res_norm":    float(res_norm),
            "res_rel":     float(res_norm / res0),
            "res0":        float(res0),
            "norm_b":      float(norm_b),
            "n_restart":   int(n_restart),
        }



    def _solve_sparse_normal_eq(self, K, rhs, tol=1e-10, max_iter=200):
        """CG on normal equations: K^T K x = K^T rhs (no explicit forming)."""
        K = K.coalesce()
        device, dtype = K.device, K.dtype

        if rhs.dim() == 2 and rhs.shape[1] == 1:
            rhs = rhs.view(-1)
        rhs = rhs.to(device=device, dtype=dtype)

        KT = K.transpose(0, 1).coalesce()
        b = torch.sparse.mm(KT, rhs.view(-1, 1)).view(-1)

        nb = torch.norm(b)
        if nb < 1e-20:
            return torch.zeros_like(b)

        def A_mv(v):
            Kv = torch.sparse.mm(K, v.view(-1, 1)).view(-1)
            return torch.sparse.mm(KT, Kv.view(-1, 1)).view(-1)

        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        for _ in range(max_iter):
            Ap = A_mv(p)
            denom = torch.dot(p, Ap)
            if abs(denom) < 1e-30:
                break
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if rs_new.sqrt() < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    def _solve_sparse_gmres_ilu_cpu(
        self,
        K_red: torch.Tensor,
        rhs: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 200,
        restart: int = 40,
        ilu_drop_tol: float = 1e-3,
        ilu_fill_factor: float = 10.0,
    ):
        """SciPy GMRES + ILU on CPU."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        Kc = K_red.coalesce()
        device_out = rhs.device
        dtype_out = rhs.dtype

        idx = Kc.indices()
        val = Kc.values()

        row = idx[0].cpu().numpy()
        col = idx[1].cpu().numpy()
        data = val.cpu().numpy()

        n = Kc.shape[0]
        K_csr = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

        ilu = spla.spilu(K_csr, drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)

        def Mv(x_np):
            return ilu.solve(x_np)

        M = spla.LinearOperator((n, n), matvec=Mv)
        A = spla.LinearOperator((n, n), matvec=lambda x: K_csr @ x)

        b_np = rhs.view(-1).detach().cpu().numpy()
        x_np, info = spla.gmres(A, b_np, M=M, tol=tol, restart=restart, maxiter=max_iter)

        if info != 0:
            print(f"[GMRES-ILU][WARN] info={info}, GMRES did not reach tol.")

        return torch.from_numpy(x_np).to(device=device_out, dtype=dtype_out)

    # ---------------------------
    # AMGX out-of-process solve
    # ---------------------------
    def _solve_sparse_amgx_outproc(self, K_csr: torch.Tensor, rhs: torch.Tensor, config_path: str):
        """
        Converts to SciPy CSR, saves to temp folder, launches worker that only loads
        CUDA+AMGX, and returns x on same device/dtype as rhs.
        """
        # Build SciPy CSR on CPU
        crow = K_csr.crow_indices().cpu().numpy().astype(np.int32, copy=False)
        ccol = K_csr.col_indices().cpu().numpy().astype(np.int32, copy=False)
        cval = K_csr.values().cpu().numpy().astype(np.float64, copy=False)
        n = int(K_csr.shape[0])
        A = csr_matrix((cval, ccol, crow), shape=(n, n))
        A.sort_indices()
        A.sum_duplicates()

        b_np = rhs.view(-1).detach().cpu().numpy().astype(np.float64, copy=False)

        td = tempfile.mkdtemp(prefix="amgx_case_")
        keep_dir = None
        try:
            A_npz = os.path.join(td, "A.npz")
            b_npy = os.path.join(td, "b.npy")
            x_npy = os.path.join(td, "x.npy")

            save_npz(A_npz, A, compressed=False)
            np.save(b_npy, b_np)

            worker = os.path.join(os.path.dirname(__file__), "amgx_worker.py")

            amgx_dir = self.amgx_env.get("AMGX_DIR")
            cuda_dir = self.amgx_env.get("CUDA_DIR")

            env = {
                "PATH": ((cuda_dir + ";") if cuda_dir else "") + ((amgx_dir + ";") if amgx_dir else "") + os.environ.get("PATH", ""),
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "",
                "CUDA_CACHE_DISABLE": "1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }

            py = sys.executable
            cmd = [
                py, worker,
                "--cfg", os.fspath(config_path),
                "--A", A_npz,
                "--b", b_npy,
                "--xout", x_npy,
                "--amgx_dir", amgx_dir or "",
                "--cuda_dir", cuda_dir or "",
            ]

            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=os.path.dirname(worker),
                timeout=None
            )

            if res.returncode != 0:
                keep_dir = td
                msg = [
                    f"[AMGX outproc] worker exit={res.returncode}",
                    f"[AMGX outproc] cmd: {' '.join(cmd)}",
                    f"[AMGX outproc] workdir: {td}",
                ]
                if res.stdout:
                    msg += ["[AMGX outproc] --- STDOUT ---", res.stdout.rstrip()]
                if res.stderr:
                    msg += ["[AMGX outproc] --- STDERR ---", res.stderr.rstrip()]
                raise RuntimeError("\n".join(msg))

            x_np = np.load(x_npy)
            shutil.rmtree(td, ignore_errors=True)

        except Exception:
            if keep_dir is None:
                keep_dir = td
            print(f"[AMGX outproc][KEEP] failure case saved at: {keep_dir}")
            raise

        return torch.from_numpy(x_np).to(device=rhs.device, dtype=rhs.dtype)


    # ---------------------------
    # GMRES flavors + helpers
    # ---------------------------
    def _gmres(
        self,
        A_mv,
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        M_inv=None,
        tol: float = 1e-8,
        max_iter: int = 200,
        restart: int = 50,
        verbose: bool = False,
    ):
        """Restarted left-preconditioned GMRES in torch (same as your working impl)."""
        device = b.device
        dtype = b.dtype
        n = b.numel()

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        if M_inv is None:
            def M_inv(r): return r

        r = (b - A_mv(x)).contiguous()
        z = M_inv(r).contiguous()
        beta0 = torch.linalg.norm(z)
        if beta0 == 0:
            return x, {"converged": True, "it": 0, "res_norm": 0.0}

        total_it = 0
        converged = False
        res_norm = beta0

        while total_it < max_iter and not converged:
            m = min(restart, max_iter - total_it)
            V = torch.zeros((n, m + 1), device=device, dtype=dtype)
            H = torch.zeros((m + 1, m), device=device, dtype=dtype)
            cs = torch.zeros(m, device=device, dtype=dtype)
            sn = torch.zeros(m, device=device, dtype=dtype)
            g = torch.zeros(m + 1, device=device, dtype=dtype)

            V[:, 0] = (z / beta0)
            g[0] = beta0

            inner_it = 0
            for j in range(m):
                inner_it = j + 1
                total_it += 1
                w = A_mv(V[:, j]).contiguous()
                w = M_inv(w).contiguous()

                # MGS + one reorthogonalization
                for _ in range(2):
                    for i in range(j + 1):
                        hij = torch.dot(V[:, i], w)
                        H[i, j] = H[i, j] + hij
                        w = w - hij * V[:, i]

                H[j + 1, j] = torch.linalg.norm(w)
                if H[j + 1, j] <= 1e-30:
                    if verbose:
                        print(f"[GMRES] happy breakdown | inner={j + 1}")
                    H[j + 1, j] = torch.zeros((), device=device, dtype=dtype)
                    m = inner_it
                    break

                V[:, j + 1] = (w / H[j + 1, j])

                for i in range(j):
                    tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = tmp

                h_jj = H[j, j]; h_j1j = H[j + 1, j]
                denom = torch.hypot(h_jj, h_j1j)
                if denom <= 0:
                    cs[j] = torch.tensor(1.0, device=device, dtype=dtype)
                    sn[j] = torch.tensor(0.0, device=device, dtype=dtype)
                else:
                    cs[j] = h_jj / denom
                    sn[j] = h_j1j / denom

                H[j, j] = cs[j] * h_jj + sn[j] * h_j1j
                H[j + 1, j] = torch.zeros((), device=device, dtype=dtype)

                tmp = cs[j] * g[j] + sn[j] * g[j + 1]
                g[j + 1] = -sn[j] * g[j]
                g[j] = tmp

                res_norm = g[j + 1].abs()
                if verbose:
                    print(f"[GMRES] it={total_it:4d}, inner={j + 1:2d}, ||M^-1 r||={res_norm.item():.3e}")

                if res_norm <= tol * beta0 or total_it >= max_iter:
                    m = inner_it
                    break

            y = torch.zeros(m, device=device, dtype=dtype)
            for i in range(m - 1, -1, -1):
                diag = H[i, i]
                if diag.abs() < 1e-30:
                    y[i] = 0.0
                else:
                    rhs = g[i] - torch.dot(H[i, i + 1:m], y[i + 1:m]) if i + 1 < m else g[i]
                    y[i] = rhs / diag

            x = (x + V[:, :m] @ y).contiguous()

            r = (b - A_mv(x)).contiguous()
            z = M_inv(r).contiguous()
            beta = torch.linalg.norm(z)
            res_norm = beta
            if verbose:
                print(f"[GMRES] restart done, ||M^-1 r||={beta.item():.3e}")
            if beta <= tol * beta0:
                converged = True
                break
            beta0 = beta

        info = {"converged": bool(converged), "it": int(total_it), "res_norm": float(res_norm)}
        return x, info

    # Right equilibration helper used by gmres_gpu
    def _make_equilibrated_right_mv1(self, K_csr: torch.Tensor):
        device, dtype = K_csr.device, K_csr.dtype

        crow, ccol, cval = K_csr.crow_indices(), K_csr.col_indices(), K_csr.values()
        n = K_csr.size(0)
        row_ids = torch.arange(n, device=device).repeat_interleave(crow[1:] - crow[:-1])
        row_abs_sum = torch.zeros(n, device=device, dtype=dtype)
        row_abs_sum.index_add_(0, row_ids, cval.abs())
        D = 1.0 / torch.clamp(row_abs_sum.sqrt(), min=1e-12)  # right scaling

        def right_mv(x):
            return torch.sparse.mm(K_csr, (D * x).unsqueeze(1)).squeeze(1)

        def post(y):
            return D * y

        return right_mv, post
    
    def _make_equilibrated_right_mv(self, K_csr: torch.Tensor):
        device, dtype = K_csr.device, K_csr.dtype
        crow, ccol, cval = K_csr.crow_indices(), K_csr.col_indices(), K_csr.values()
        n = K_csr.size(0)
        
        row_ids = torch.arange(n, device=device).repeat_interleave(crow[1:] - crow[:-1])
        row_abs_sum = torch.zeros(n, device=device, dtype=dtype)
        row_abs_sum.index_add_(0, row_ids, cval.abs())
        
        D = 1.0 / torch.clamp(row_abs_sum.sqrt(), min=1e-12)
        
        def scaled_mv(x):
            Dx = D * x
            KDx = torch.sparse.mm(K_csr, Dx.unsqueeze(1)).squeeze(1)
            return D * KDx
        
        def post(y):
            return D * y
        
        return scaled_mv, post, D  # <-- return D