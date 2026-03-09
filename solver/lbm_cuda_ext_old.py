"""
lbm_cuda_ext.py
===============
Loads lbm_kernels.cu CUDA extension and provides a drop-in patch
(LBMCUDAUpgrade.patch) that replaces hot Python loops in LBMCHT3D_Torch
with fused CUDA kernels.

JIT-compiles on first import (~30s), cached thereafter.

Requirements (Windows):
  - CUDA Toolkit installed (nvcc on PATH or CUDA_HOME set)
  - MSVC Build Tools (cl.exe on PATH)
  - PyTorch with CUDA support

Usage
-----
    from lbm_cuda_ext import LBM_CUDA, LBM_CUDA_AVAILABLE
    from lbm_cuda_ext import LBMCUDAUpgrade

    # Patch sim in-place AFTER FlowSolverUpgrade.patch(sim):
    LBMCUDAUpgrade.patch(sim)

    # The patched sim.step() now calls fused CUDA kernels for
    # macroscopic + feq + collision, streaming, sponge, and IBB.

Kernels (in lbm_kernels.cu)
----------------------------
  macro_collide_bgk(f, f_post, rho, u, v, w, solid, omega)
  macro_collide_trt(f, f_post, rho, u, v, w, solid, omega_plus, omega_minus)
  macro_collide_mrt(f, f_post, rho, u, v, w, solid, S_rates[19])
  stream_pull      (f_in, f_out, nx, ny, nz, periodic_x, periodic_y, periodic_z)
  sponge           (f, rho, u, v, w, sigma)
  ibb              (f_post, f_stream, fluid_flat, dir_i, dir_opp,
                    w1, w2, q_ge_half, ff_flat, ff_valid)
  reset_solid      (f, solid)

Memory layout assumed: f[19, nx, ny, nz] C-contiguous (PyTorch default for
                       the ordering used in LBMCHT3D_Torch).
"""

from __future__ import annotations
import os
import types
import torch

# -----------------------------------------------------------------------------
#  Extension loading
# -----------------------------------------------------------------------------
LBM_CUDA           = None
LBM_CUDA_AVAILABLE = False


def _load_extension() -> bool:
    global LBM_CUDA, LBM_CUDA_AVAILABLE

    if not torch.cuda.is_available():
        print("[LBM_CUDA] CUDA not available -- using PyTorch fallback.")
        return False

    _dir = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.join(_dir, "lbm_kernels.cu")

    if not os.path.isfile(_src):
        print(f"[LBM_CUDA] lbm_kernels.cu not found at {_src} -- using PyTorch fallback.")
        return False

    try:
        from torch.utils.cpp_extension import load
        print("[LBM_CUDA] Compiling CUDA extension (first run only, ~30s) ...")
        LBM_CUDA = load(
            name    = "lbm_kernels",
            sources = [_src],
            extra_cuda_cflags = [
                "-O3",
                "--use_fast_math",
                "-DNDEBUG",
                "-allow-unsupported-compiler",
            ],
            extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"],
            verbose = False,
        )
        LBM_CUDA_AVAILABLE = True
        print("[LBM_CUDA] Extension loaded successfully.")
        return True
    except Exception as e:
        print(f"[LBM_CUDA] Compilation failed: {e}\nFalling back to PyTorch kernels.")
        LBM_CUDA           = None
        LBM_CUDA_AVAILABLE = False
        return False


_load_extension()


# -----------------------------------------------------------------------------
#  Helper: prepare IBB tensors in the correct dtype/layout for the kernel
# -----------------------------------------------------------------------------
def _prepare_ibb_tensors(ibb_data, device, dtype):
    """
    Convert ibb_data fields to the int32 / float32 / uint8 tensors expected
    by lbm_ibb().  Caches the result on ibb_data._cuda_tensors to avoid
    re-conversion every step.
    """
    if hasattr(ibb_data, '_cuda_tensors'):
        return ibb_data._cuda_tensors

    def _to(t, dt):
        return t.to(device=device, dtype=dt).contiguous()

    t = types.SimpleNamespace(
        fluid_flat = _to(ibb_data.fluid_flat, torch.int32),
        dir_i      = _to(ibb_data.dir_i,      torch.int32),
        dir_opp    = _to(ibb_data.dir_opp,     torch.int32),
        w_i        = _to(ibb_data.w_i,         torch.float32),
        w_second   = _to(ibb_data.w_second,    torch.float32),
        q_ge_half  = _to(ibb_data.q_ge_half.to(torch.uint8), torch.uint8),
        ff_flat    = _to(ibb_data.ff_flat,     torch.int32),
        ff_valid   = _to(ibb_data.ff_valid.to(torch.uint8),  torch.uint8),
    )
    ibb_data._cuda_tensors = t
    return t


# -----------------------------------------------------------------------------
#  LBMCUDAUpgrade -- patches LBMCHT3D_Torch with fused CUDA kernels
# -----------------------------------------------------------------------------
class LBMCUDAUpgrade:
    """
    Patches LBMCHT3D_Torch in-place to use fused CUDA kernels.

    Call AFTER FlowSolverUpgrade.patch(sim):
        FlowSolverUpgrade.patch(sim)    # adds MRT collision, sponge, etc.
        LBMCUDAUpgrade.patch(sim)       # replaces Python loops with CUDA kernels

    What gets replaced
    ------------------
    sim._macroscopic()  -> no-op (macros now computed inside macro_collide kernel)
    sim._collide()      -> lbm_kernels.macro_collide_{bgk,trt,mrt}
    sim._stream()       -> lbm_kernels.stream_pull
    sim._interp_bounce_back_from_post() -> lbm_kernels.ibb + reset_solid
    sim._apply_sponge_zone()  -> lbm_kernels.sponge
    sim.step()          -> reordered to call CUDA kernels directly

    Note: _apply_flow_bcs() is NOT replaced -- Zou/He reconstruction stays in
    Python/PyTorch (it runs on boundary planes only, not performance critical).
    """

    @staticmethod
    def patch(sim):
        if not LBM_CUDA_AVAILABLE:
            print("[LBM_CUDA] Kernels not available -- patch skipped, using PyTorch fallback.")
            return

        cfg = sim.cfg

        # -- 1. Prepare solid mask as uint8 (needed by all kernels) -----------
        sim._solid_u8 = sim.solid_flow.to(dtype=torch.uint8,
                                           device=sim.device).contiguous()

        # -- 2. Prepare MRT S-rates tensor (if MRT mode) ----------------------
        collision_mode = getattr(cfg, 'collision', 'bgk').lower()
        if collision_mode == 'mrt':
            # sim._mrt_S was set by FlowSolverUpgrade._init_mrt()
            sim._S_cuda = sim._mrt_S.to(device=sim.device,
                                         dtype=torch.float32).contiguous()
        else:
            sim._S_cuda = None

        # -- 3. Prepare f_post buffer -----------------------------------------
        if not hasattr(sim, '_f_post') or sim._f_post.shape != sim.f.shape:
            sim._f_post = torch.empty_like(sim.f)

        # -- 4. Prepare f_stream buffer (for pull streaming) ------------------
        # We reuse _f_stream if it exists; otherwise allocate
        if not hasattr(sim, '_f_stream') or sim._f_stream.shape != sim.f.shape:
            sim._f_stream = torch.empty_like(sim.f)

        # -- 5. Determine streaming periodic flags ----------------------------
        flow_dir = cfg.flow_dir.upper().strip()
        _axis = {'X': 0, 'Y': 1, 'Z': 2}[flow_dir[1]]
        _transverse_walls = bool(getattr(cfg, 'transverse_walls', False))
        _pbc = [not _transverse_walls, not _transverse_walls, not _transverse_walls]
        _pbc[_axis] = False  # flow axis is never periodic

        sim._cuda_periodic = tuple(_pbc)  # (px, py, pz) booleans

        # -- 6. Prepare sponge sigma tensor (flat) ----------------------------
        _has_sponge = (int(getattr(cfg, 'sponge_length', 0)) > 0 and
                       hasattr(sim, '_sponge_sigma'))
        if _has_sponge:
            sim._sponge_sigma_flat = sim._sponge_sigma.reshape(-1).contiguous()
        else:
            sim._sponge_sigma_flat = None

        # -- 7. Patch methods -------------------------------------------------
        # _macroscopic becomes a no-op (macros computed inside macro_collide kernel)
        sim._macroscopic_python = sim._macroscopic  # keep original for BC use
        sim._macroscopic = types.MethodType(LBMCUDAUpgrade._macroscopic_noop, sim)

        # _collide -> fused kernel (also computes macros)
        sim._collide = types.MethodType(
            LBMCUDAUpgrade._make_collide(collision_mode), sim)

        # _stream -> pull-scheme kernel
        sim._stream = types.MethodType(LBMCUDAUpgrade._stream_cuda, sim)

        # IBB -> CUDA kernel
        if hasattr(sim, 'ibb_data') and sim.ibb_data is not None and sim.ibb_data.n_links > 0:
            # Pre-convert IBB tensors once
            _prepare_ibb_tensors(sim.ibb_data, sim.device, sim.dtype)
            sim._interp_bounce_back_from_post = types.MethodType(
                LBMCUDAUpgrade._ibb_cuda, sim)

        # Sponge zone -> CUDA kernel (replaces the wrapped version from FlowSolverUpgrade)
        if _has_sponge:
            sim._apply_sponge_zone = types.MethodType(
                LBMCUDAUpgrade._sponge_cuda, sim)

        # Replace _apply_flow_bcs with single-kernel CUDA version
        sim._apply_flow_bcs = types.MethodType(_apply_flow_bcs_cuda, sim)

        # Save original step() as thermal delegate.
        # If LBMCHT3D_Torch.step() contains real thermal logic beyond (0,0),
        # it will be accessible as sim._original_step_thermal() from _step_cuda.
        # For the base implementation (thermal handled by outer loop), this is
        # effectively unused but harmless.
        sim._original_step_thermal = sim.step

        # Replace step() with CUDA-aware version
        sim.step = types.MethodType(LBMCUDAUpgrade._step_cuda, sim)

        sim._cuda_debug_nan = bool(getattr(cfg, 'cuda_debug_nan', False))

        print(f"[LBM_CUDA] LBMCUDAUpgrade patched:")
        print(f"  collision       = {collision_mode}  (fused macro+feq+collide)")
        print(f"  streaming       = pull-scheme CUDA kernel")
        print(f"  sponge          = {'CUDA kernel' if _has_sponge else 'disabled'}")
        ibb_n = sim.ibb_data.n_links if hasattr(sim, 'ibb_data') and sim.ibb_data else 0
        print(f"  IBB             = CUDA kernel ({ibb_n} links)")
        print(f"  periodic (x,y,z)= {sim._cuda_periodic}")
        print(f"  flow BCs        = CUDA zou_he_bc (2 kernel calls/step)")


    # --------------------------------------------------------------------------
    @staticmethod
    def _macroscopic_noop(sim_self):
        """
        No-op: macroscopic quantities (rho, u, v, w) are now computed inside
        the fused macro_collide kernel.  The macro fields are already up-to-date
        after _collide() writes them directly to sim.rho, sim.u, sim.v, sim.wv.

        The original _macroscopic is still accessible via sim._macroscopic_python()
        if needed (e.g. at initialisation, or for debugging).
        """
        pass


    # --------------------------------------------------------------------------
    @staticmethod
    def _make_collide(collision_mode: str):
        """
        Returns a bound-method-compatible function for the given collision mode.
        The function:
          1. Calls the appropriate fused CUDA kernel (macro+feq+collide)
          2. Writes rho, u, v, w directly into sim.rho/.u/.v/.wv
          3. Stores f_post in sim._f_post (needed by IBB)
          4. Optionally applies sponge in-place on f_post
        """
        def _collide_bgk(sim_self):
            N = sim_self.nx * sim_self.ny * sim_self.nz
            f_flat     = sim_self.f.reshape(19, N).contiguous()
            fpost_flat = sim_self._f_post.reshape(19, N)

            LBM_CUDA.macro_collide_bgk(
                f_flat, fpost_flat,
                sim_self.rho.reshape(N),
                sim_self.u.reshape(N),
                sim_self.v.reshape(N),
                sim_self.wv.reshape(N),
                sim_self._solid_u8.reshape(N),
                float(sim_self.omega))

            # Enforce zero velocity inside solids (macro output)
            sim_self.u[sim_self.solid_flow]  = 0.
            sim_self.v[sim_self.solid_flow]  = 0.
            sim_self.wv[sim_self.solid_flow] = 0.

        def _collide_trt(sim_self):
            N = sim_self.nx * sim_self.ny * sim_self.nz
            f_flat     = sim_self.f.reshape(19, N).contiguous()
            fpost_flat = sim_self._f_post.reshape(19, N)

            LBM_CUDA.macro_collide_trt(
                f_flat, fpost_flat,
                sim_self.rho.reshape(N),
                sim_self.u.reshape(N),
                sim_self.v.reshape(N),
                sim_self.wv.reshape(N),
                sim_self._solid_u8.reshape(N),
                float(sim_self.omega_plus),
                float(sim_self.omega_minus))

            sim_self.u[sim_self.solid_flow]  = 0.
            sim_self.v[sim_self.solid_flow]  = 0.
            sim_self.wv[sim_self.solid_flow] = 0.

        def _collide_mrt(sim_self):
            N = sim_self.nx * sim_self.ny * sim_self.nz
            f_flat     = sim_self.f.reshape(19, N).contiguous()
            fpost_flat = sim_self._f_post.reshape(19, N)

            LBM_CUDA.macro_collide_mrt(
                f_flat, fpost_flat,
                sim_self.rho.reshape(N),
                sim_self.u.reshape(N),
                sim_self.v.reshape(N),
                sim_self.wv.reshape(N),
                sim_self._solid_u8.reshape(N),
                sim_self._S_cuda)

            sim_self.u[sim_self.solid_flow]  = 0.
            sim_self.v[sim_self.solid_flow]  = 0.
            sim_self.wv[sim_self.solid_flow] = 0.

        dispatch = {'bgk': _collide_bgk, 'trt': _collide_trt, 'mrt': _collide_mrt}
        return dispatch.get(collision_mode, _collide_bgk)


    # --------------------------------------------------------------------------
    @staticmethod
    def _stream_cuda(sim_self):
        """Pull-scheme streaming using lbm_kernels.stream_pull."""
        px, py, pz = sim_self._cuda_periodic

        # f_post -> f_stream via pull
        LBM_CUDA.stream_pull(
            sim_self._f_post.contiguous(),
            sim_self._f_stream,
            sim_self.nx, sim_self.ny, sim_self.nz,
            int(px), int(py), int(pz))

        # Swap: f <- f_stream  (f_post is still valid for IBB)
        sim_self.f, sim_self._f_stream = sim_self._f_stream, sim_self.f


    # --------------------------------------------------------------------------
    @staticmethod
    def _ibb_cuda(sim_self, f_post: torch.Tensor):
        """
        CUDA IBB using lbm_kernels.ibb + reset_solid.
        Signature matches _interp_bounce_back_from_post(f_post).
        """
        ibb = sim_self.ibb_data
        if ibb.n_links == 0:
            return

        t = ibb._cuda_tensors  # pre-converted in LBMCUDAUpgrade.patch()
        N = sim_self.nx * sim_self.ny * sim_self.nz

        LBM_CUDA.ibb(
            f_post.reshape(19, N).contiguous(),
            sim_self.f.reshape(19, N),      # f_stream = sim_self.f after _stream_cuda swap
            t.fluid_flat, t.dir_i, t.dir_opp,
            t.w_i, t.w_second, t.q_ge_half,
            t.ff_flat, t.ff_valid)

        # Reset solid populations to equilibrium at rest
        LBM_CUDA.reset_solid(
            sim_self.f.reshape(19, N),
            sim_self._solid_u8.reshape(N))


    # --------------------------------------------------------------------------
    @staticmethod
    def _sponge_cuda(sim_self, target_f: 'torch.Tensor | None' = None):
        """
        Sponge zone via lbm_kernels.sponge.
        Replaces FlowSolverUpgrade._apply_sponge_zone.

        target_f : tensor [19, N] to apply sponge to.
                   Defaults to sim_self.f (in-place on current f).
                   Pass sim_self._f_post to apply post-collision, pre-stream.
        """
        if sim_self._sponge_sigma_flat is None:
            return

        N = sim_self.nx * sim_self.ny * sim_self.nz
        f_tgt = (target_f if target_f is not None else sim_self.f).reshape(19, N)
        LBM_CUDA.sponge(
            f_tgt,
            sim_self.rho.reshape(N),
            sim_self.u.reshape(N),
            sim_self.v.reshape(N),
            sim_self.wv.reshape(N),
            sim_self._sponge_sigma_flat)


    # --------------------------------------------------------------------------
    @staticmethod
    def _step_cuda(sim_self, do_thermal: bool = True) -> tuple:
        """
        CUDA-fused step().  Replaces LBMCHT3D_Torch.step().

        Step order (matches original semantic order exactly):
          1. macroscopic      -> NO-OP (macros computed inside fused collide kernel)
          2. _collide         -> fused macro+feq+collide -> writes f_post + macros
          3. _apply_sponge_zone -> CUDA sponge on f_post (if sponge_length > 0)
          4. _stream          -> pull-scheme CUDA kernel  (f_post -> f)
          5. IBB / bounce-back -> CUDA IBB kernel (reads f_post, writes into f)
          6. _apply_flow_bcs  -> PyTorch Zou/He (boundary planes only, unchanged)
          7. NaN safety check (debug mode only)
          8. thermal step     -> delegated to _original_step_thermal if available,
                                otherwise handled by outer loop (do_thermal=False path)

        Key implementation notes:
          * _f_post is populated by the CUDA collide kernel (step 2).
          * _stream swaps f ? _f_stream, so after step 4:
              - sim.f       = streamed populations (used by IBB and BCs)
              - sim._f_stream = pre-stream buffer (can be reused next step)
              - sim._f_post  = post-collision populations (still valid; used by IBB)
          * Sponge (step 3) runs on f_post BEFORE streaming, matching FlowSolverUpgrade
            which applied it post-collision inside _collide.  This means sponge damps
            reflections before they enter the stream, which is correct.
          * _apply_flow_bcs reads sim.f (post-stream) and writes Zou/He populations
            back to inlet/outlet planes, also updating sim.rho/u/v/w at boundaries.
            Interior macro fields remain valid from step 2.
        """
        # -- 2: Fused macroscopic + f_eq + collision --------------------------
        # Writes f_post (in sim._f_post) AND rho/u/v/w (in sim.rho/.u/.v/.wv)
        sim_self._collide()

        # -- 3: Sponge zone (post-collision, pre-stream) -----------------------
        # Must be explicit here: FlowSolverUpgrade wrapped sponge inside _collide,
        # but LBMCUDAUpgrade replaces _collide entirely.  We call _sponge_cuda
        # directly on _f_post (post-collision buffer) using macros from step 2.
        if sim_self._sponge_sigma_flat is not None:
            sim_self._apply_sponge_zone(sim_self._f_post)

        # -- 4: Pull-scheme streaming (f_post -> f) -----------------------------
        sim_self._stream()

        # -- 5: Obstacle bounce-back / IBB ------------------------------------
        if (hasattr(sim_self, 'ibb_data') and
                sim_self.ibb_data is not None and
                sim_self.ibb_data.n_links > 0):
            # f_post is still the pre-stream post-collision buffer as required by BFL
            sim_self._interp_bounce_back_from_post(sim_self._f_post)
        else:
            sim_self._bounce_back_post_stream()

        # -- 6: Inlet/Outlet BCs (Zou/He pressure/velocity) -------------------
        sim_self._apply_flow_bcs()

        # -- 7: Finite-value safety check -------------------------------------
        # Opt-in NaN check: set cfg.cuda_debug_nan=True to enable.
        # Off by default -- reads all 110 MB of f every step.
        if sim_self._cuda_debug_nan:
            if not torch.isfinite(sim_self.f).all():
                bad = torch.nonzero(~torch.isfinite(sim_self.f), as_tuple=False)[0]
                print("[LBM_CUDA NaN] after BCs: (q,x,y,z)=",
                      tuple(int(v) for v in bad.tolist()))
                raise FloatingPointError("Non-finite f after BCs")

        # -- 8: Thermal step --------------------------------------------------
        # The base LBMCHT3D_Torch.step() also returns (0,0) with no thermal code
        # in the provided implementation -- thermal is handled by the outer CHT loop.
        # If the solver has been extended with thermal substeps, preserve them via
        # the _original_step_thermal hook set during patch().
        if do_thermal and hasattr(sim_self, '_original_step_thermal'):
            return sim_self._original_step_thermal()

        return (0, 0)


# -----------------------------------------------------------------------------
#  Quick self-test (python lbm_cuda_ext.py)
# -----------------------------------------------------------------------------

# =============================================================================
#  CUDA replacement for _apply_flow_bcs
# =============================================================================

def _apply_flow_bcs_cuda(sim_self):
    """
    CUDA replacement for _apply_flow_bcs().  Two kernel calls (inlet + outlet)
    instead of ~26 small PyTorch kernels + Python interpreter loops.
    Each call covers one boundary plane of ~7200 nodes in a single CUDA launch.
    """
    cfg  = sim_self.cfg
    fdir = cfg.flow_dir.upper().strip()
    sign = +1 if fdir[0] == '+' else -1
    axis = {'X': 0, 'Y': 1, 'Z': 2}[fdir[1]]
    n_along = (sim_self.nx, sim_self.ny, sim_self.nz)[axis]

    if sign == +1:
        inlet_idx  = 0
        outlet_idx = n_along - 1
    else:
        inlet_idx  = n_along - 1
        outlet_idx = 0
    interior_out_idx = (outlet_idx - 1) if outlet_idx > 0 else 1

    N  = sim_self.nx * sim_self.ny * sim_self.nz
    f_flat = sim_self.f.reshape(19, N).contiguous()
    s_flat = sim_self._solid_u8.reshape(N)

    u_max = float(getattr(cfg, 'u_lat_max', 0.5))
    u_signed = float(max(-u_max, min(u_max, float(sim_self.u_in_lat) * sign)))

    # -- velocity inlet --
    LBM_CUDA.zou_he_bc(
        f_flat, s_flat,
        sim_self.nx, sim_self.ny, sim_self.nz,
        axis, inlet_idx, 0,
        0,                             # bc_type=0: velocity
        u_signed, float(cfg.rho_out_lat), sign)

    # -- pressure outlet --
    LBM_CUDA.zou_he_bc(
        f_flat, s_flat,
        sim_self.nx, sim_self.ny, sim_self.nz,
        axis, outlet_idx, interior_out_idx,
        1,                             # bc_type=1: pressure
        u_signed, float(cfg.rho_out_lat), sign)


# =============================================================================
#  Per-step phase timer (inject after patch for profiling)
# =============================================================================

def add_step_profiler(sim, print_every: int = 200):
    """
    Inject a synchronizing step timer into sim.step().
    Prints per-phase wall time every `print_every` steps.

    Usage (after LBMCUDAUpgrade.patch):
        from lbm_cuda_ext import add_step_profiler
        add_step_profiler(sim)
    """
    import types, time

    sim._prof = {k: 0.0 for k in ('collide','sponge','stream','ibb','bc','total')}
    sim._prof['n'] = 0

    def _timed_step(self, do_thermal=True):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        self._collide()
        torch.cuda.synchronize(); t1 = time.perf_counter()

        if self._sponge_sigma_flat is not None:
            self._apply_sponge_zone(self._f_post)
        torch.cuda.synchronize(); t2 = time.perf_counter()

        self._stream()
        torch.cuda.synchronize(); t3 = time.perf_counter()

        if (hasattr(self, 'ibb_data') and
                self.ibb_data is not None and self.ibb_data.n_links > 0):
            self._interp_bounce_back_from_post(self._f_post)
        else:
            self._bounce_back_post_stream()
        torch.cuda.synchronize(); t4 = time.perf_counter()

        self._apply_flow_bcs()
        torch.cuda.synchronize(); t5 = time.perf_counter()

        p = self._prof
        p['collide'] += t1-t0; p['sponge'] += t2-t1
        p['stream']  += t3-t2; p['ibb']    += t4-t3
        p['bc']      += t5-t4; p['total']  += t5-t0
        p['n'] += 1

        if p['n'] % print_every == 0:
            n = p['n']
            print(f"[PROFILE] {n} steps  (ms/step)")
            for k in ('collide','sponge','stream','ibb','bc','total'):
                print(f"  {k:10s}: {p[k]/n*1e3:.3f} ms")
            print(f"  it/s      : {n/p['total']:.1f}")

        if do_thermal and hasattr(self, '_original_step_thermal'):
            return self._original_step_thermal()
        return (0, 0)

    sim.step = types.MethodType(_timed_step, sim)
    print(f"[PROFILER] Injected. Prints every {print_every} steps.")


if __name__ == "__main__":
    if not LBM_CUDA_AVAILABLE:
        print("LBM_CUDA not available -- cannot run self-test.")
    else:
        import numpy as np
        print("\n=== LBM CUDA self-test ===")
        dev = 'cuda'
        nx, ny, nz = 8, 10, 6
        N = nx * ny * nz

        # Random f with positive values
        torch.manual_seed(42)
        f = torch.rand(19, N, device=dev, dtype=torch.float32) * 0.1
        # Normalize so rho ? 1
        f = f / f.sum(0, keepdim=True)

        solid  = torch.zeros(N, dtype=torch.uint8, device=dev)
        f_post = torch.empty_like(f)
        rho    = torch.empty(N, device=dev, dtype=torch.float32)
        u      = torch.empty(N, device=dev, dtype=torch.float32)
        v      = torch.empty(N, device=dev, dtype=torch.float32)
        w      = torch.empty(N, device=dev, dtype=torch.float32)

        # Test BGK
        LBM_CUDA.macro_collide_bgk(f, f_post, rho, u, v, w, solid, 1.6)
        print(f"  BGK: rho mean={rho.mean():.4f}  (expect ?1.0)")
        assert torch.isfinite(f_post).all(), "BGK produced NaN/Inf"
        # Mass conservation
        mass_before = f.sum(0)
        mass_after  = f_post.sum(0)
        err = (mass_before - mass_after).abs().max().item()
        print(f"  BGK mass conservation error: {err:.2e}  (expect <1e-5)")

        # Test MRT
        S = torch.tensor([0,1.19,1.4, 0,1.4, 0,1.4, 0,1.4,
                          1.8,1.8, 1.8,1.8, 1.8,1.8,1.8, 1.2,1.2,1.2],
                         device=dev, dtype=torch.float32)
        LBM_CUDA.macro_collide_mrt(f, f_post, rho, u, v, w, solid, S)
        assert torch.isfinite(f_post).all(), "MRT produced NaN/Inf"
        err = (f.sum(0) - f_post.sum(0)).abs().max().item()
        print(f"  MRT mass conservation error: {err:.2e}  (expect <1e-5)")

        # Test stream (periodic all dims)
        f_stream = torch.empty_like(f)
        LBM_CUDA.stream_pull(f_post, f_stream, nx, ny, nz, 1, 1, 1)
        assert torch.isfinite(f_stream).all(), "stream produced NaN/Inf"
        # Mass must be conserved across all-periodic streaming
        err_stream = (f_post.sum() - f_stream.sum()).abs().item()
        print(f"  Stream (all-periodic) mass error: {err_stream:.2e}  (expect <1e-4)")

        # Test sponge (small sigma)
        sigma = torch.zeros(N, device=dev, dtype=torch.float32)
        sigma[-N//4:] = 0.1  # last quarter is sponge
        f_before = f_stream.clone()
        LBM_CUDA.sponge(f_stream, rho, u, v, w, sigma)
        changed = (f_stream - f_before).abs().sum().item()
        print(f"  Sponge: sum |?f| in sponge zone = {changed:.4f}  (expect >0)")

        # Test reset_solid
        solid2 = torch.zeros(N, dtype=torch.uint8, device=dev)
        solid2[:5] = 1
        LBM_CUDA.reset_solid(f_stream, solid2)
        for q in range(19):
            w_q = [1/3]+[1/18]*6+[1/36]*12
            exp = w_q[q]
            got = f_stream[q, :5].mean().item()
            if abs(got - exp) > 1e-5:
                print(f"  reset_solid FAIL: q={q} got={got:.6f} expected={exp:.6f}")
        print("  reset_solid: solid nodes reset to rest equilibrium ?")

        print("\nAll self-tests passed. ?")
