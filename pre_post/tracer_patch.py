"""
tracer_patch.py
===============
Attaches a GPU passive scalar tracer to sim.

The tracer is advected by the LBM velocity field (u, v, w) every step
using first-order upwind scheme on GPU. This gives physically correct
smoke that flows through fins, accelerates in channels, recirculates
behind fins — no fake post-hoc advection.

Install: copy to pre_post/tracer_patch.py

Usage in your main script, AFTER LBMCUDAUpgrade.patch(sim):

    from pre_post.tracer_patch import attach_tracer
    tracer = attach_tracer(sim, flow_dir=cfg.flow_dir)
    tracer.warmup(n_steps=300)
"""

import torch
import types
import numpy as np


class TracerField:
    def __init__(self, sim, flow_dir="-Z", inject_depth=6, dissipation=0.9995):
        self.sim   = sim
        self.fd    = flow_dir.strip().upper()
        self.diss  = dissipation
        nx, ny, nz = sim.nx, sim.ny, sim.nz
        self.nx, self.ny, self.nz = nx, ny, nz

        self.field = torch.zeros(nx, ny, nz, device=sim.device, dtype=torch.float32)
        self.fluid = ~sim.solid_flow   # (nx,ny,nz) bool, never changes

        # inlet injection mask: fluid cells at the inlet face
        d = inject_depth
        m = torch.zeros(nx, ny, nz, dtype=torch.bool, device=sim.device)
        if   self.fd == "+Z": m[:, :, :d]    = True
        elif self.fd == "-Z": m[:, :, nz-d:] = True
        elif self.fd == "+Y": m[:, :d, :]    = True
        elif self.fd == "-Y": m[:, ny-d:, :] = True
        elif self.fd == "+X": m[:d, :, :]    = True
        elif self.fd == "-X": m[nx-d:, :, :] = True
        self._inlet = m & self.fluid

        print(f"[Tracer] flow_dir={flow_dir}  inject_depth={inject_depth}  "
              f"dissipation={dissipation}  inlet_voxels={int(self._inlet.sum()):,}")

    def inject(self):
        """Set tracer=1 at all inlet fluid cells. Call once per animation frame."""
        self.field[self._inlet] = 1.0

    def step(self):
        """
        One advection step using current sim.u, sim.v, sim.wv.
        Called automatically from inside sim.step() after the LBM kernel runs,
        so velocity is always up to date.
        Uses first-order upwind: stable for all LBM velocities (|u| << 1/sqrt(3)).
        """
        phi = self.field
        u   = self.sim.u    # (nx,ny,nz) already on GPU, updated by LBM kernel
        v   = self.sim.v
        w   = self.sim.wv

        # upwind differences per axis
        adv_x = torch.where(u >= 0,
                             phi - torch.roll(phi,  1, 0),
                             torch.roll(phi, -1, 0) - phi)
        adv_y = torch.where(v >= 0,
                             phi - torch.roll(phi,  1, 1),
                             torch.roll(phi, -1, 1) - phi)
        adv_z = torch.where(w >= 0,
                             phi - torch.roll(phi,  1, 2),
                             torch.roll(phi, -1, 2) - phi)

        self.field = torch.clamp(phi - (u*adv_x + v*adv_y + w*adv_z), 0.0, 1.0)
        self.field *= self.diss
        self.field[~self.fluid] = 0.0

    def warmup(self, n_steps=300):
        """
        Advance tracer n_steps without recording to pre-fill the domain.
        Call after flow is at steady state, before save_animation().
        """
        print(f"[Tracer] warming up {n_steps} steps to fill domain...")
        self.inject()
        for i in range(n_steps):
            if i % 30 == 0:
                self.inject()
            # call step directly — sim.step is NOT called here so flow stays frozen
            self.step()
        print(f"[Tracer] warmup done — max density: {float(self.field.max()):.4f}")

    def get(self):
        """Return (nx,ny,nz) float32 numpy array for Zarr storage."""
        return self.field.detach().cpu().numpy().astype(np.float32)


def attach_tracer(sim, flow_dir="-Z", inject_depth=6, dissipation=0.9995):
    """
    Creates a TracerField and wraps sim.step() so tracer.step() is called
    automatically after every LBM step.

    IMPORTANT: Call this AFTER LBMCUDAUpgrade.patch(sim) so we wrap
    _step_cuda (the final patched version), not the original Python step().
    The wrap just appends tracer.step() after whatever sim.step() does —
    it does not interfere with LBM kernels, IBB, BCs, or thermal logic.
    """
    tracer = TracerField(sim, flow_dir=flow_dir,
                         inject_depth=inject_depth, dissipation=dissipation)

    # Capture the current sim.step (= LBMCUDAUpgrade._step_cuda at this point)
    _lbm_step = sim.step

    def _step_with_tracer(self_, do_thermal=False):
        # Run the full LBM step (CUDA kernels, IBB, BCs)
        result = _lbm_step(do_thermal=do_thermal)
        # Advect tracer with the freshly updated velocity field
        tracer.step()
        return result

    sim.step = types.MethodType(_step_with_tracer, sim)
    print("[Tracer] attached — tracer.step() runs after every sim.step()")
    return tracer
