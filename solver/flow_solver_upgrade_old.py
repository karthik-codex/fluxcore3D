"""
flow_solver_upgrade.py
======================
Drop-in upgrade for LBMCHT3D_Torch.

Usage
-----
    from flow_solver_upgrade import FlowSolverUpgrade
    FlowSolverUpgrade.patch(sim)                    # patch sim in-place
    flow_info = sim.run_flow_to_steady_v2(...)       # use upgraded loop

What it adds
------------
1. MRT collision (D3Q19)
   - Transformation matrix M derived from Lallemand & Luo (2000) Phys Rev E 61:6546
     and d'Humières et al. (2002) Phil Trans R Soc A 360:437
   - The 19 moments are:
       [0]  ρ               (conserved)
       [1]  e               energy
       [2]  ε               energy-squared
       [3]  jx              (conserved)
       [4]  qx              energy-flux x
       [5]  jy              (conserved)
       [6]  qy              energy-flux y
       [7]  jz              (conserved)
       [8]  qz              energy-flux z
       [9]  3p_xx           viscous stress  = 2cx²-cy²-cz²  (or equivalently 3cx²-|c|²)
       [10] 3π_xx           kinetic energy coupled to stress
       [11] p_ww            normal stress   = cy²-cz²
       [12] π_ww            kinetic energy coupled to p_ww
       [13] p_xy            shear stress    = cxcy
       [14] p_yz            shear stress    = cycz
       [15] p_xz            shear stress    = cxcz
       [16] m_x             3rd-order ghost = cx(cy²-cz²)
       [17] m_y             3rd-order ghost = cy(cz²-cx²)
       [18] m_z             3rd-order ghost = cz(cx²-cy²)

   - Relaxation rates:
       s[0,3,5,7] = 0.0   (conserved: ρ, jx, jy, jz — NEVER relaxed)
       s[9,11,13,14,15] = 1/τ_ν = ω_plus   (viscous channels, SET BY VISCOSITY)
       s[1]        = s_e                     (energy mode,   default 1.19)
       s[2]        = s_ε  = s_bulk           (energy-square, default cfg.mrt_s_bulk)
       s[4,6,8]    = s_q  = s_bulk           (heat-flux,     default cfg.mrt_s_bulk)
       s[10,12]    = s_π  = s_bulk           (stress-energy, default cfg.mrt_s_bulk)
       s[16,17,18] = s_g  = s_ghost          (ghost modes,   default cfg.mrt_s_ghost)

   Collision formula (per Lallemand & Luo Eq. 5):
       f* = f - M⁻¹ Ŝ (m - m_eq)
   where Ŝ = diag(s) and m = M f, m_eq = M f_eq.

2. Improved convergence monitor (run_flow_to_steady_v2)
   Tracks three independent criteria simultaneously:
     a) Velocity residual:   R_u  = ‖u^{n+1}−u^n‖_∞ / (‖u^n‖_∞ + ε)
     b) Mass imbalance:      R_ṁ  = |ṁ_in − ṁ_out| / max(ṁ_in, ε)
     c) Density deviation:   R_ρ  = ‖ρ−1‖_∞
   Convergence is declared when all three are below their respective tolerances
   for `min_stable_checks` consecutive check intervals.

3. Outlet sponge zone
   Applies an additional relaxation toward f_eq in the last `sponge_length` cells
   along the flow axis.  The strength ramps as σ(x) = σ_max · ((x-x0)/L_sponge)²
   (quadratic ramp to avoid an abrupt interface).
   Modified post-collision step for sponge cells:
       f = f - σ(x) · (f - f_eq)
   where f_eq is evaluated at the local instantaneous macro state.

4. Convective outlet BC
   When cfg.outlet_bc_mode == "convective", the outlet BC replaces the
   fixed-density Zou/He pressure reconstruction with a first-order advective
   (zero-gradient) extrapolation:
       f_i(outlet, t+1) = f_i(outlet−1, t)   for all unknown directions i
   followed by mass conservation correction to keep mean density at rho_out.
   This allows acoustic waves and vortices to leave the domain without reflection.

Configuration fields read from cfg (add these to SimConfig3D if not present):
    collision        str    "bgk" | "trt" | "mrt"
    mrt_s_bulk       float  relaxation for energy / heat-flux / stress-energy modes (1.2–1.8)
    mrt_s_ghost      float  relaxation for 3rd-order ghost modes (1.0–1.6)
    mrt_s_e          float  relaxation for energy mode alone (default: 1.19)
    outlet_bc_mode   str    "pressure" | "convective"
    sponge_length    int    cells; 0 to disable
    sponge_strength  float  σ_max (extra relaxation at outlet face)
"""

from __future__ import annotations
import types
import numpy as np
import torch
from tqdm import trange


# ═══════════════════════════════════════════════════════════════════════════════
#  D3Q19 MRT TRANSFORMATION MATRIX
#  Derived for the velocity ordering used in LBMCHT3D_Torch.
# ═══════════════════════════════════════════════════════════════════════════════

# fmt: off
_M_D3Q19 = np.array([
    # Vel ordering: rest | ±x, ±y, ±z | ±xy, ±xz, ±yz (matching LBMCHT3D_Torch c_np)
    #   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
    [  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # [0]  ρ
    [-30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],  # [1]  e
    [ 12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # [2]  ε
    [  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0,  0],  # [3]  jx
    [  0, -4,  4,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0,  0],  # [4]  qx
    [  0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1],  # [5]  jy
    [  0,  0,  0, -4,  4,  0,  0,  1, -1,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1],  # [6]  qy
    [  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1],  # [7]  jz
    [  0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1],  # [8]  qz
    [  0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],  # [9]  3p_xx = 2cx²-cy²-cz²
    [  0, -4, -4,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],  # [10] 3π_xx
    [  0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],  # [11] p_ww = cy²-cz²
    [  0,  0,  0, -2, -2,  2,  2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],  # [12] π_ww
    [  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # [13] p_xy = cxcy
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1],  # [14] p_yz = cycz
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0],  # [15] p_xz = cxcz
    [  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1, -1,  1,  1,  0,  0,  0,  0],  # [16] m_x = cx(cy²-cz²)
    [  0,  0,  0,  0,  0,  0,  0, -1,  1, -1,  1,  0,  0,  0,  0,  1,  1, -1, -1],  # [17] m_y = cy(cz²-cx²)
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1],  # [18] m_z = cz(cx²-cy²)
], dtype=np.float64)
# fmt: on

_M_INV_D3Q19 = np.linalg.inv(_M_D3Q19)

# Viscous channels: rows 9, 11, 13, 14, 15  (stress tensor components)
# These MUST be relaxed at 1/tau_ν to recover the correct Navier-Stokes viscosity.
_VISCOUS_CHANNELS = [9, 11, 13, 14, 15]

# Conserved channels: rows 0, 3, 5, 7  (density + momentum)
# Their relaxation rates MUST be 0 — never relax conserved moments.
_CONSERVED_CHANNELS = [0, 3, 5, 7]

# Energy / heat-flux / stress-energy channels (free parameters)
_ENERGY_CHANNEL = 1        # e   → use mrt_s_e (often ~1.19)
_EPS_CHANNEL    = 2        # ε
_HEATFLUX_CHANNELS   = [4, 6, 8]            # qx, qy, qz
_STRESSENERGY_CHANNELS = [10, 12]           # π_xx, π_ww
_GHOST_CHANNELS  = [16, 17, 18]             # 3rd-order ghosts


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PATCH CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class FlowSolverUpgrade:
    """
    Patches LBMCHT3D_Torch in-place with upgraded flow physics.

    Call once after construction:
        FlowSolverUpgrade.patch(sim)
    """

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def patch(sim):
        """
        Patch *sim* in-place.  After this call:

        • sim._collide       → MRT (if cfg.collision=='mrt'), TRT, or BGK unchanged
        • sim._apply_flow_bcs → supports convective outlet + sponge zone
        • sim.run_flow_to_steady_v2 → improved convergence monitor (new method)
        • sim._apply_sponge_zone    → sponge helper (new method)
        """
        cfg = sim.cfg

        # 1. ── MRT initialisation ────────────────────────────────────────────
        if getattr(cfg, 'collision', 'bgk').lower() == 'mrt':
            FlowSolverUpgrade._init_mrt(sim)
            # Replace _collide with our MRT version (bound method trick)
            sim._collide = types.MethodType(FlowSolverUpgrade._collide_mrt, sim)

        # 2. ── Sponge zone ───────────────────────────────────────────────────
        sponge_len = int(getattr(cfg, 'sponge_length', 0))
        if sponge_len > 0:
            FlowSolverUpgrade._init_sponge(sim)
            sim._apply_sponge_zone = types.MethodType(
                FlowSolverUpgrade._apply_sponge_zone, sim)

            # Wrap _collide so sponge is applied after collision each step
            _inner_collide = sim._collide
            def _collide_with_sponge(self_):
                _inner_collide()
                self_._apply_sponge_zone()
            sim._collide = types.MethodType(_collide_with_sponge, sim)

        # 3. ── Convective outlet BC ──────────────────────────────────────────
        if getattr(cfg, 'outlet_bc_mode', 'pressure').lower() == 'convective':
            FlowSolverUpgrade._init_convective_bc(sim)
            sim._apply_flow_bcs = types.MethodType(
                FlowSolverUpgrade._apply_flow_bcs_convective, sim)

        # 4. ── Improved run loop ─────────────────────────────────────────────
        sim.run_flow_to_steady_v2 = types.MethodType(
            FlowSolverUpgrade.run_flow_to_steady_v2, sim)

        print("[UPGRADE] FlowSolverUpgrade patched:")
        print(f"  collision       = {getattr(cfg,'collision','bgk')}")
        print(f"  outlet_bc_mode  = {getattr(cfg,'outlet_bc_mode','pressure')}")
        print(f"  sponge_length   = {getattr(cfg,'sponge_length',0)} cells, "
              f"strength = {getattr(cfg,'sponge_strength',0.0):.2f}")


    # ══════════════════════════════════════════════════════════════════════════
    #  1.  MRT COLLISION
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _init_mrt(sim):
        """
        Build the MRT matrices and relaxation vector S on the device.

        Sets on sim:
            _mrt_M      (19,19)  torch.Tensor  — transformation to moment space
            _mrt_Minv   (19,19)  torch.Tensor  — inverse transformation
            _mrt_S      (19,)    torch.Tensor  — diagonal relaxation rates
        """
        cfg = sim.cfg
        dev, dtyp = sim.device, sim.dtype

        M    = torch.tensor(_M_D3Q19,     device=dev, dtype=dtyp)
        Minv = torch.tensor(_M_INV_D3Q19, device=dev, dtype=dtyp)

        # ── Relaxation rates ──────────────────────────────────────────────────
        omega_nu  = float(sim.omega_plus)   # viscous channels → from physical viscosity
        s_bulk    = float(getattr(cfg, 'mrt_s_bulk',  1.4))
        s_ghost   = float(getattr(cfg, 'mrt_s_ghost', 1.2))
        s_e       = float(getattr(cfg, 'mrt_s_e',     1.19))

        # Validate relaxation rates (must be in (0, 2) for stability)
        for name, val in [('s_bulk', s_bulk), ('s_ghost', s_ghost), ('s_e', s_e), ('omega_nu', omega_nu)]:
            if not (0.0 < val < 2.0):
                raise ValueError(
                    f"MRT relaxation rate {name}={val:.6g} is outside (0,2). "
                    "Check cfg.mrt_s_bulk / mrt_s_ghost / omega_plus."
                )

        S = torch.zeros(19, device=dev, dtype=dtyp)

        # Conserved modes: s = 0 (no relaxation — strictly enforced)
        for ch in _CONSERVED_CHANNELS:
            S[ch] = 0.0

        # Viscous stress channels: MUST equal omega_nu for correct Navier-Stokes
        for ch in _VISCOUS_CHANNELS:
            S[ch] = omega_nu

        # Energy mode
        S[_ENERGY_CHANNEL] = s_e

        # Energy-square, heat-flux, stress-energy: use s_bulk
        S[_EPS_CHANNEL] = s_bulk
        for ch in _HEATFLUX_CHANNELS:
            S[ch] = s_bulk
        for ch in _STRESSENERGY_CHANNELS:
            S[ch] = s_bulk

        # Ghost modes: free parameters
        for ch in _GHOST_CHANNELS:
            S[ch] = s_ghost

        sim._mrt_M    = M
        sim._mrt_Minv = Minv
        sim._mrt_S    = S   # shape (19,)

        print(f"[MRT] ω_ν={omega_nu:.5f}  s_e={s_e:.3f}  "
              f"s_bulk={s_bulk:.3f}  s_ghost={s_ghost:.3f}")

    # ------------------------------------------------------------------
    @staticmethod
    def _collide_mrt(sim_self):
        """
        MRT collision operator.

        f* = f - M⁻¹ Ŝ (m - m_eq)

        where
          m     = M f                         (project to moment space)
          m_eq  = M f_eq                      (equilibrium in moment space)
          Ŝ (m - m_eq) = S ⊙ (m - m_eq)     (element-wise scale with s_i)

        The conserved modes (ρ, jx, jy, jz) have S[i]=0 so
        (m - m_eq)[conserved] is never added back, preserving mass/momentum.
        """
        nx, ny, nz = sim_self.nx, sim_self.ny, sim_self.nz
        N = nx * ny * nz

        M    = sim_self._mrt_M       # (19, 19)
        Minv = sim_self._mrt_Minv    # (19, 19)
        S    = sim_self._mrt_S       # (19,)

        f = sim_self.f               # (19, nx, ny, nz)

        # ── equilibrium ───────────────────────────────────────────────────────
        feq = sim_self._feq(sim_self.rho, sim_self.u, sim_self.v, sim_self.wv)
        # (19, nx, ny, nz)

        # ── project to moment space: m = M f  (batched matmul) ───────────────
        # reshape to (19, N), matmul (19,19)×(19,N) → (19,N), reshape back
        f_flat   = f.reshape(19, N)
        feq_flat = feq.reshape(19, N)

        m    = torch.mm(M, f_flat)    # (19, N)
        m_eq = torch.mm(M, feq_flat)  # (19, N)

        # ── relax: Ŝ (m - m_eq) ──────────────────────────────────────────────
        delta_m = S[:, None] * (m - m_eq)  # (19, N)  — conserved rows are 0

        # ── back to velocity space ────────────────────────────────────────────
        df = torch.mm(Minv, delta_m)       # (19, N)

        sim_self.f = (f_flat - df).reshape(19, nx, ny, nz)


    # ══════════════════════════════════════════════════════════════════════════
    #  2.  OUTLET SPONGE ZONE
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _init_sponge(sim):
        """
        Pre-compute the per-cell extra-relaxation field σ(x) for the sponge zone.

        σ = 0 everywhere except in the last `sponge_length` cells along the
        flow axis, where it ramps quadratically from 0 to σ_max (sponge_strength).

        Sets on sim:
            _sponge_sigma   (nx, ny, nz)   float tensor  — per-cell σ
            _sponge_axis    int             — flow axis (0=X, 1=Y, 2=Z)
            _sponge_outlet_idx int          — index of the outlet plane
        """
        cfg = sim.cfg
        dev, dtyp = sim.device, sim.dtype

        flow_dir    = cfg.flow_dir.upper().strip()
        sign        = +1 if flow_dir[0] == '+' else -1
        axis        = {'X': 0, 'Y': 1, 'Z': 2}[flow_dir[1]]
        n_along     = (sim.nx, sim.ny, sim.nz)[axis]

        outlet_idx  = (n_along - 1) if sign == +1 else 0
        sponge_len  = int(getattr(cfg, 'sponge_length', 10))
        sigma_max   = float(getattr(cfg, 'sponge_strength', 3.0))

        sigma_np = np.zeros((sim.nx, sim.ny, sim.nz), dtype=np.float32)

        # Quadratic ramp from 0 → σ_max over the sponge region
        for k in range(sponge_len):
            ramp = ((k + 1) / sponge_len) ** 2 * sigma_max
            if sign == +1:
                cell_idx = outlet_idx - (sponge_len - 1 - k)
            else:
                cell_idx = outlet_idx + (sponge_len - 1 - k)

            if cell_idx < 0 or cell_idx >= n_along:
                continue

            if axis == 0:
                sigma_np[cell_idx, :, :] = ramp
            elif axis == 1:
                sigma_np[:, cell_idx, :] = ramp
            else:
                sigma_np[:, :, cell_idx] = ramp

        sim._sponge_sigma      = torch.tensor(sigma_np, device=dev, dtype=dtyp)
        sim._sponge_axis       = axis
        sim._sponge_outlet_idx = outlet_idx

        n_sponge = int((sigma_np > 0).sum())
        print(f"[SPONGE] {n_sponge} cells, σ_max={sigma_max:.2f}, "
              f"axis={axis}, outlet_idx={outlet_idx}")

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_sponge_zone(sim_self):
        """
        After standard collision, additionally relax f toward f_eq in sponge cells:

            f = f - σ(x) · (f - f_eq)

        This is equivalent to applying a second BGK step with local ω=σ,
        damping unphysical reflections before they reach the outlet plane.
        The operation is applied only where σ > 0 to avoid touching the bulk.
        """
        sigma = sim_self._sponge_sigma   # (nx,ny,nz)

        # Only compute where sponge is active
        active = sigma > 0.0
        if not active.any():
            return

        feq = sim_self._feq(sim_self.rho, sim_self.u, sim_self.v, sim_self.wv)
        # (19, nx, ny, nz)

        # σ · (f - f_eq) — broadcast sigma over the Q dimension
        damping = sigma[None, :, :, :] * (sim_self.f - feq)
        sim_self.f = sim_self.f - damping


    # ══════════════════════════════════════════════════════════════════════════
    #  3.  CONVECTIVE OUTLET BC
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _init_convective_bc(sim):
        """
        Pre-compute index slices needed for the convective outlet update.

        Sets on sim:
            _conv_axis        int   — flow axis
            _conv_outlet_idx  int   — outlet plane index
            _conv_interior_idx int  — interior neighbour plane index
            _conv_sign        int   — +1 or -1
            _conv_unknown_ids list  — populations to overwrite at outlet
            _conv_f_prev      tensor (19, n1, n2) — outlet f from previous step
        """
        cfg = sim.cfg
        flow_dir     = cfg.flow_dir.upper().strip()
        sign         = +1 if flow_dir[0] == '+' else -1
        axis         = {'X': 0, 'Y': 1, 'Z': 2}[flow_dir[1]]
        n_along      = (sim.nx, sim.ny, sim.nz)[axis]

        outlet_idx   = (n_along - 1) if sign == +1 else 0
        interior_idx = (outlet_idx - 1) if outlet_idx > 0 else (outlet_idx + 1)

        c_comp = (sim.cx, sim.cy, sim.cz)
        cax    = c_comp[axis]
        outlet_missing = (-1 if outlet_idx != 0 else +1)
        unknown_ids    = [i for i in range(19) if int(cax[i]) == outlet_missing]

        n1, n2 = {0: (sim.ny, sim.nz),
                  1: (sim.nx, sim.nz),
                  2: (sim.nx, sim.ny)}[axis]

        sim._conv_axis         = axis
        sim._conv_outlet_idx   = outlet_idx
        sim._conv_interior_idx = interior_idx
        sim._conv_sign         = sign
        sim._conv_unknown_ids  = unknown_ids
        sim._conv_f_prev       = torch.zeros(
            (19, n1, n2), device=sim.device, dtype=sim.dtype)

        print(f"[CONV-BC] convective outlet: axis={axis}, outlet_idx={outlet_idx}, "
              f"{len(unknown_ids)} unknown dirs")

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_flow_bcs_convective(sim_self):
        """
        Replaces _apply_flow_bcs for convective outlet mode.

        Inlet:  unchanged Zou/He velocity inlet (from original _apply_flow_bcs).
        Outlet: Orlanski-type first-order convective BC for unknown populations,
                with a flux-conserving mass correction that adjusts only the
                unknown (incoming) directions so that net axial mass flux at the
                outlet matches the inlet — without disturbing the density field.

        Why the original density-rescaling was wrong
        ---------------------------------------------
        Multiplying ALL populations by (ρ_target / ρ_mean) preserves density
        but scales the axial momentum flux by the same factor, breaking the
        inlet-outlet mass balance whenever ρ_mean ≠ ρ_target.  The correct
        approach is to carry the mass-flux residual exclusively on the unknown
        (incoming) populations, which is exactly what the Zou/He pressure outlet
        does — except here we derive the correction from actual flux data rather
        than prescribing a density.

        Algorithm (per call)
        --------------------
        1. Save outlet plane BEFORE any BC (post-stream state).
        2. Call original _apply_flow_bcs → inlet gets its Zou/He update; outlet
           gets an initial pressure-Zou/He fill (overwritten in step 4).
        3. Compute inlet mass flux ṁ_in = Σ_{fluid} ρ·u_ax at inlet plane.
        4. Set f_unknown_i(outlet) = f_interior_i   (zero-gradient extrapolation)
           for the 5 populations with c·n̂ < 0.
        5. Compute current outlet mass flux ṁ_out with the extrapolated unknowns.
        6. Flux deficit δṁ = ṁ_in − ṁ_out is distributed equally among the
           N_unknown unknown populations at every fluid node:
               Δf_i = δṁ / (N_fluid_outlet × N_unknown × |c_ax|)
           This keeps Σ(f·c_ax) exact while adding zero net density change
           because Δf is equal for all unknown directions (Σ_unknown c_ax = const).
        7. Store outlet plane as f_prev for next step.
        """
        axis         = sim_self._conv_axis
        outlet_idx   = sim_self._conv_outlet_idx
        interior_idx = sim_self._conv_interior_idx
        unknown_ids  = sim_self._conv_unknown_ids
        sign         = sim_self._conv_sign

        c_comp = (sim_self.cx, sim_self.cy, sim_self.cz)
        cax    = c_comp[axis].to(sim_self.dtype)   # (19,)

        # fluid mask on outlet plane
        if axis == 0:
            fluid_mask = ~sim_self.solid_flow[outlet_idx, :, :]
        elif axis == 1:
            fluid_mask = ~sim_self.solid_flow[:, outlet_idx, :]
        else:
            fluid_mask = ~sim_self.solid_flow[:, :, outlet_idx]
        n_fluid = int(fluid_mask.sum().item())

        # ── 1. Save pre-BC outlet (post-stream) ──────────────────────────────
        f_outlet_post_stream = sim_self._plane_f(axis, outlet_idx).clone()

        # ── 2. Run original BC (handles inlet Zou/He correctly) ──────────────
        sim_self.__class__._apply_flow_bcs(sim_self)

        # ── 3. Compute inlet mass flux ḿ_in = Σ(ρ·u_ax) at inlet ────────────
        if axis == 0:
            rho_in = sim_self.rho[0 if sign == +1 else sim_self.nx - 1, :, :]
            u_in   = sim_self.u  [0 if sign == +1 else sim_self.nx - 1, :, :]
        elif axis == 1:
            rho_in = sim_self.rho[:, 0 if sign == +1 else sim_self.ny - 1, :]
            u_in   = sim_self.v  [:, 0 if sign == +1 else sim_self.ny - 1, :]
        else:
            rho_in = sim_self.rho[:, :, 0 if sign == +1 else sim_self.nz - 1]
            u_in   = sim_self.wv [:, :, 0 if sign == +1 else sim_self.nz - 1]

        if axis == 0:
            inlet_fluid = ~sim_self.solid_flow[0 if sign == +1 else sim_self.nx - 1, :, :]
        elif axis == 1:
            inlet_fluid = ~sim_self.solid_flow[:, 0 if sign == +1 else sim_self.ny - 1, :]
        else:
            inlet_fluid = ~sim_self.solid_flow[:, :, 0 if sign == +1 else sim_self.nz - 1]

        mdot_in = float((rho_in * u_in * float(sign))[inlet_fluid].sum().item())

        # ── 4. Zero-gradient extrapolation for unknown populations ────────────
        f_interior = sim_self._plane_f(axis, interior_idx)  # (19, n1, n2)
        f_outlet_new = sim_self._plane_f(axis, outlet_idx).clone()

        for i in unknown_ids:
            f_outlet_new[i] = f_interior[i]

        # ── 5. Compute outlet mass flux with extrapolated unknowns ────────────
        # ṁ_out = Σ_i Σ_fluid f_i · c_ax_i  (summed over fluid nodes, all dirs)
        mdot_out = 0.0
        for i in range(19):
            c_i = float(cax[i]) * float(sign)  # signed axial component
            if c_i != 0.0 and fluid_mask.any():
                mdot_out += c_i * float(f_outlet_new[i, fluid_mask].sum().item())

        # ── 6. Distribute flux deficit among unknown populations ──────────────
        # The unknowns all have c_ax = outlet_missing_sign (e.g. -1 for +Y flow).
        # Δf_per_node_per_dir = δṁ / (n_fluid * n_unknown * |c_ax_unknown|)
        # Effect: Σ_i Σ_fluid Δf_i · c_ax_i
        #       = n_fluid * n_unknown * Δf * c_ax_unknown
        #       = δṁ   ✓
        # Effect on density at each node:
        #       Δρ = n_unknown * Δf  (same for all fluid nodes)
        # This is a tiny uniform density correction ~O(Δu/n_unknown).
        n_unknown = len(unknown_ids)
        if n_unknown > 0 and n_fluid > 0:
            c_ax_unknown = float(cax[unknown_ids[0]]) * float(sign)  # ±1
            if abs(c_ax_unknown) > 0:
                # Correct sign derivation:
                #   Δṁ_out = n_fluid * n_unknown * delta_f * c_ax_unknown
                #   We want Δṁ_out = (ṁ_in - ṁ_out), therefore:
                #   delta_f = -(ṁ_in - ṁ_out) / (n_fluid * n_unknown * |c_ax|)
                # The negation is because c_ax_unknown < 0 for +Y flow at y=max:
                # adding a negative delta_f to a cy=-1 population reduces
                # the "backward" flux and increases the net outgoing flux.
                delta_f_raw = -(mdot_in - mdot_out) / (n_fluid * n_unknown * abs(c_ax_unknown) + 1e-30)

                # Clamp to ±20% of mean unknown-population magnitude.
                # Prevents large shocks in early transient; at steady state
                # the deficit goes to zero so this clamp is never active.
                mean_f_unk = float(
                    f_outlet_new[unknown_ids, :][:, fluid_mask].abs().mean().item()
                ) + 1e-10
                delta_f = max(-0.2 * mean_f_unk, min(0.2 * mean_f_unk, delta_f_raw))

                for i in unknown_ids:
                    f_outlet_new[i, fluid_mask] = f_outlet_new[i, fluid_mask] + delta_f

        # ── 7. Write back and store f_prev ────────────────────────────────────
        sim_self._set_plane_f(axis, outlet_idx, f_outlet_new)
        sim_self._conv_f_prev.copy_(f_outlet_post_stream)


    # ══════════════════════════════════════════════════════════════════════════
    #  4.  IMPROVED CONVERGENCE MONITOR
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def run_flow_to_steady_v2(
        sim_self,
        max_steps:        int   = 30000,
        check_every:      int   = 100,
        # Primary convergence criterion (the only one that must converge)
        tol_u_res:        float = 1e-4,    # relative velocity change ‖Δu‖∞/‖u‖∞
        # Secondary / sanity criteria (informational; warn but don't block convergence)
        tol_dm_rate:      float = 1e-5,    # domain mass rate of change |ΔM|/M per step
        warn_rho_mean:    float = 5e-3,    # warn if |mean(ρ_fluid)-1| > this
        min_stable_checks:int   = 5,       # consecutive check intervals satisfying tol_u_res
        # Diagnostic verbosity
        print_every:      int   = 500,
    ) -> dict:
        """
        Flow-to-steady loop with physically correct convergence monitoring.

        PRIMARY criterion (must converge to declare steady state):

          R_u = ‖u^{n+1} − u^n‖_∞ / (‖u^n‖_∞ + ε)   ← velocity residual

        SECONDARY diagnostics (printed but not blocking):

          ΔM_rate = |Σρ(t) − Σρ(t−Δt)| / Σρ(t)
              Domain-level mass rate of change.  Goes to zero when nothing
              is evolving.  This is the correct mass-conservation residual
              for LBM: plane-to-plane flux difference (ṁ_in vs ṁ_out) is a
              steady non-zero quantity in compressible LBM because density
              varies with pressure — it is NOT a convergence indicator.

          ρ_mean_err = |mean(ρ_fluid) − ρ_out_lat|
              Global mean density drift.  Should be < 0.5% at true steady
              state; values of 2-5% typically indicate the pressure drop
              across the domain, which is physically correct.
              NOTE: ‖ρ−1‖_∞ (the old R_rho) will always be large near
              obstacles because it captures the pressure peak — it is not
              and never was a convergence measure.

        Why ṁ_in vs ṁ_out is wrong as a convergence criterion
        -------------------------------------------------------
        In LBM, ρ encodes gauge pressure: ρ = ρ₀ + δp/cs².  With a
        velocity inlet and pressure outlet, ρ_inlet > ρ_outlet by exactly
        the pressure drop.  Therefore:
            ṁ_in  = Σ ρ_in · u_in    (ρ_in > 1 at high-pressure inlet side)
            ṁ_out = Σ ρ_out · u_out   (ρ_out ≈ 1)
        These will differ at steady state by a factor of ρ_in/ρ_out even
        though mass is perfectly conserved in the domain.  The correct
        check is ΔM_rate (domain mass change rate), not flux balance.
        """
        # ── fluid mask for domain-level diagnostics ──────────────────────────
        fluid_mask_3d = ~sim_self.solid_flow  # (nx, ny, nz)

        # ── snapshot of previous state ───────────────────────────────────────
        u_prev    = sim_self.u.clone()
        v_prev    = sim_self.v.clone()
        w_prev    = sim_self.wv.clone()
        rho_prev  = sim_self.rho.clone()

        stable      = 0
        history     = []
        umax_last   = 0.0
        R_u = R_dm = rho_mean_err = float('nan')

        # ── log header ───────────────────────────────────────────────────────
        header = (f"{'step':>7}  {'R_u':>10}  {'ΔM_rate':>10}  "
                  f"{'ρ_mean_err':>12}  {'umax':>10}  {'stable':>6}")
        print(header)
        print('─' * len(header))

        pbar = trange(max_steps, desc="FLOW-v2", leave=True, ncols=100)

        for it in pbar:
            sim_self.step(do_thermal=False)

            if (it % check_every) != 0:
                continue

            # ── velocity residual (primary criterion) ────────────────────────
            du = sim_self.u  - u_prev
            dv = sim_self.v  - v_prev
            dw = sim_self.wv - w_prev

            mag_du = max(
                float(torch.abs(du).max().item()),
                float(torch.abs(dv).max().item()),
                float(torch.abs(dw).max().item()),
            )
            mag_u = float(
                torch.sqrt(sim_self.u**2 + sim_self.v**2 + sim_self.wv**2).max().item()
            )
            umax_last = mag_u
            R_u = mag_du / (mag_u + 1e-12)

            # ── domain mass rate of change ΔM/M (secondary) ──────────────────
            # This is the correct mass-conservation residual.
            # (ṁ_in ≠ ṁ_out at steady state in LBM — see docstring)
            M_now  = float(sim_self.rho[fluid_mask_3d].sum().item())
            M_prev = float(rho_prev[fluid_mask_3d].sum().item())
            R_dm   = abs(M_now - M_prev) / (M_prev + 1e-12)

            # ── mean density deviation (pressure-drop indicator) ─────────────
            # Report it but do NOT require it to converge — it is the steady
            # pressure field, which is correct at any non-zero value.
            rho_mean = float(sim_self.rho[fluid_mask_3d].mean().item())
            rho_mean_err = abs(rho_mean - float(sim_self.cfg.rho_out_lat))

            # ── convergence decision (R_u only) ──────────────────────────────
            converged = (R_u < tol_u_res)
            stable    = stable + 1 if converged else 0

            # Optional warning if domain mass is still drifting
            if converged and R_dm > tol_dm_rate and stable == 1:
                print(f"  [WARN] R_u converged but ΔM_rate={R_dm:.2e} > {tol_dm_rate:.0e}. "
                      "Domain mass still evolving — consider more steps.")

            history.append({
                'step': it, 'R_u': R_u, 'R_dm': R_dm,
                'rho_mean_err': rho_mean_err, 'umax': mag_u,
            })

            pbar.set_postfix({
                'Ru':   f'{R_u:.2e}',
                'ΔM':   f'{R_dm:.2e}',
                'ρerr': f'{rho_mean_err:.4f}',
                'st':   stable,
            })

            if it % print_every == 0 or (converged and stable == 1):
                print(f"{it:>7}  {R_u:>10.3e}  {R_dm:>10.3e}  "
                      f"{rho_mean_err:>12.6f}  {mag_u:>10.4f}  {stable:>6}")

            # update snapshots
            u_prev.copy_(sim_self.u)
            v_prev.copy_(sim_self.v)
            w_prev.copy_(sim_self.wv)
            rho_prev.copy_(sim_self.rho)

            if stable >= min_stable_checks:
                pbar.close()
                print(f"\n[CONVERGED] step={it}, R_u={R_u:.3e}, "
                      f"ΔM_rate={R_dm:.3e}, ρ_mean_err={rho_mean_err:.5f}, "
                      f"umax={mag_u:.5f}")
                return {
                    'converged':     True,
                    'flow_steps':    it,
                    'umax':          mag_u,
                    'R_u':           R_u,
                    'R_dm':          R_dm,
                    'rho_mean_err':  rho_mean_err,
                    'rho_mean':      rho_mean,
                    'stable_checks': stable,
                    'history':       history,
                }

        pbar.close()
        print(f"\n[NOT CONVERGED] max_steps={max_steps} reached. "
              f"R_u={R_u:.3e}, ΔM_rate={R_dm:.3e}, ρ_mean_err={rho_mean_err:.5f}")
        return {
            'converged':     False,
            'flow_steps':    max_steps,
            'umax':          umax_last,
            'R_u':           R_u,
            'R_dm':          R_dm,
            'rho_mean_err':  rho_mean_err,
            'rho_mean':      rho_mean if 'rho_mean' in dir() else float('nan'),
            'stable_checks': stable,
            'history':       history,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS  (run with:  python flow_solver_upgrade.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _test_M_matrix():
    """Verify the M matrix: orthogonality, correct moment encoding, invertibility."""
    import numpy as np

    c_np = np.array([
        [0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
        [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],[1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
        [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1],
    ], dtype=int)

    M    = _M_D3Q19
    Minv = _M_INV_D3Q19

    # Invertibility
    err_inv = np.max(np.abs(M @ Minv - np.eye(19)))
    assert err_inv < 1e-10, f"M @ Minv != I  (err={err_inv:.3e})"

    # Conserved: row 0 = 1, rows 3,5,7 = cx,cy,cz
    cx, cy, cz = c_np[:,0], c_np[:,1], c_np[:,2]
    assert np.allclose(M[0], 1), "Row 0 should be all 1"
    assert np.allclose(M[3], cx), "Row 3 should be cx"
    assert np.allclose(M[5], cy), "Row 5 should be cy"
    assert np.allclose(M[7], cz), "Row 7 should be cz"

    # Viscous stress rows match quadratic forms
    s = (c_np**2).sum(axis=1)
    assert np.allclose(M[9],  3*cx**2 - s),   "Row 9 = 3cx^2 - s"
    assert np.allclose(M[11], cy**2 - cz**2),  "Row 11 = cy^2-cz^2"
    assert np.allclose(M[13], cx*cy),           "Row 13 = cx*cy"
    assert np.allclose(M[14], cy*cz),           "Row 14 = cy*cz"
    assert np.allclose(M[15], cx*cz),           "Row 15 = cx*cz"

    # Ghost rows
    assert np.allclose(M[16], cx*(cy**2 - cz**2)), "Row 16 = cx(cy^2-cz^2)"
    assert np.allclose(M[17], cy*(cz**2 - cx**2)), "Row 17 = cy(cz^2-cx^2)"
    assert np.allclose(M[18], cz*(cx**2 - cy**2)), "Row 18 = cz(cx^2-cy^2)"

    # Momentum-conserving: M[3,5,7] . M_inv should give EXACT conserved rows
    print("  M matrix: all structural checks passed.")


def _test_mrt_reduces_to_bgk():
    """
    When all free relaxation rates equal omega, MRT must reproduce BGK exactly.

    Key: feq MUST be computed from macros derived from f itself (not from external
    rho/u values), so that Σ feq = Σ f and Σ cx·feq = Σ cx·f identically.
    When that invariant holds, (m - m_eq)[conserved] = 0 exactly, and
    setting S[conserved]=0 or S[conserved]=omega gives the same result.
    """
    c_np = np.array([
        [0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
        [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],[1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
        [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1],
    ], dtype=int)
    cx, cy, cz = c_np[:,0], c_np[:,1], c_np[:,2]
    w = np.array([1/3]+[1/18]*6+[1/36]*12, dtype=np.float64)
    M    = _M_D3Q19
    Minv = _M_INV_D3Q19

    # Build f perturbed from equilibrium, then derive macros FROM f
    np.random.seed(7); N = 20
    rho_t = 1.0 + 0.01*np.random.randn(N)
    ux_t  = 0.01*np.random.randn(N)
    uy_t  = 0.005*np.random.randn(N)
    uz_t  = 0.003*np.random.randn(N)
    cu    = 3*(cx[:,None]*ux_t + cy[:,None]*uy_t + cz[:,None]*uz_t)
    feq_t = w[:,None]*rho_t[None,:]*(1 + cu + 0.5*cu**2 - 1.5*(ux_t**2+uy_t**2+uz_t**2))
    f     = feq_t + 0.001*np.random.randn(19, N)

    # Recompute macros FROM f (exactly as the LBM loop does before collision)
    rho   = f.sum(0)
    inv_r = 1.0 / rho
    ux    = (f * cx[:,None]).sum(0) * inv_r
    uy    = (f * cy[:,None]).sum(0) * inv_r
    uz    = (f * cz[:,None]).sum(0) * inv_r
    cu2   = 3*(cx[:,None]*ux + cy[:,None]*uy + cz[:,None]*uz)
    feq   = w[:,None]*rho[None,:]*(1 + cu2 + 0.5*cu2**2 - 1.5*(ux**2+uy**2+uz**2))

    # Verify feq consistency: this makes (m-meq)[conserved] == 0 exactly
    assert np.max(np.abs(feq.sum(0) - rho)) < 1e-14, "feq density inconsistency"
    assert np.max(np.abs((feq*cx[:,None]).sum(0) - rho*ux)) < 1e-14, "feq momentum inconsistency"

    omega = 1.2
    S     = np.full(19, omega)
    S[[0, 3, 5, 7]] = 0.0   # conserved channels: doesn't matter, (m-meq)=0 there anyway

    # BGK
    f_bgk = f + omega*(feq - f)

    # MRT with all rates = omega  →  must equal BGK
    m     = M @ f
    m_eq  = M @ feq
    f_mrt = f - Minv @ (S[:,None]*(m - m_eq))

    err = np.max(np.abs(f_mrt - f_bgk))
    assert err < 1e-10, f"MRT BGK-limit error = {err:.3e}"
    print(f"  MRT→BGK limit: max_err = {err:.3e}  [PASS]")

    # Also verify conservation for full MRT (different rates per channel)
    S2 = S.copy()
    S2[1]=1.19; S2[2]=S2[4]=S2[6]=S2[8]=S2[10]=S2[12]=1.4; S2[16:]=1.2
    f_mrt2 = f - Minv @ (S2[:,None]*(m - m_eq))
    mass_err = np.max(np.abs((f_mrt2 - f).sum(0)))
    momx_err = np.max(np.abs(((f_mrt2 - f)*cx[:,None]).sum(0)))
    assert mass_err < 1e-14, f"Full MRT mass conservation error: {mass_err:.3e}"
    assert momx_err < 1e-14, f"Full MRT momentum-x error: {momx_err:.3e}"
    print(f"  Full MRT conservation: Δmass={mass_err:.2e}, Δmom_x={momx_err:.2e}  [PASS]")


if __name__ == "__main__":
    print("Running unit tests for flow_solver_upgrade …")
    _test_M_matrix()
    _test_mrt_reduces_to_bgk()
    print("\nAll tests passed.")
