# surface_flux_bc.py
"""
SurfaceFluxBC — surface heat flux boundary condition for LBMCHT3D_Torch.

Usage
-----
    from surface_flux_bc import SurfaceFluxBC

    # Attach 10,000 W/m² on the +Z face of the heatsink base (36 × 45 mm chip footprint)
    bc = SurfaceFluxBC(
        solid_mask    = sim.solid,          # (nx, ny, nz) bool CUDA tensor
        axis          = "+Z",               # "+X"/"-X"/"+Y"/"-Y"/"+Z"/"-Z"
        surface_L_mm  = 36.0,               # length of the heat source surface [mm]
        surface_W_mm  = 45.0,               # width  of the heat source surface [mm]
        q_flux_W_m2   = 10_000.0,           # heat flux [W/m²], positive = into solid
        dx_mm         = sim.cfg.dx_mm,      # voxel pitch [mm]
        center_mm     = None,               # (u_mm, v_mm) in-plane center; None = solid centroid
    )

    sim.add_surface_flux(bc)

    # Optionally inspect which voxels were selected:
    print(f"[SurfaceFluxBC] {bc.n_voxels} face voxels, "
          f"Q_total={bc.q_total_W(sim.rho_cp, sim.rho_cp_ref_J_m3K, sim.dx_phys_m, sim.dt_thermal_phys_s):.2f} W")

Physics
-------
A voxel at the surface face receives energy flux q [W/m²] through one face of area dx².
    Q_cell = q * dx²  [W]
    dT/dt  = Q_cell / (rho_cp_phys * dx³) = q / (rho_cp_phys * dx)  [K/s]
    src_step[cell] = q / (rho_cp_phys * dx) * dt_thermal              [K / thermal-step]

This is identical in form to the volumetric source already in _compute_heat_source,
so it stacks additively into src_step with no solver changes needed.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch

Tensor = torch.Tensor


class VolumetricHeatBC:
    """
    Applies Q [Watts] uniformly across a specified solid voxel mask.

    src_step[cell] = (Q / V_solid) / rho_cp_phys[cell] * dt_thermal   [K/step]

    Usage
    -----
        bc = VolumetricHeatBC(
            solid_mask = sim.solid_CuSpray,   # bool tensor for that body only
            Q_watts    = 20.0,
            dx_mm      = cfg.dx_mm,
        )
        sim.add_volumetric_heat(bc)
    """

    def __init__(
        self,
        solid_mask: torch.Tensor,   # (nx,ny,nz) bool, only the target solid
        Q_watts:    float,
        dx_mm:      float,
    ):
        self.voxel_mask = solid_mask.bool()
        self.Q_watts    = float(Q_watts)
        self.dx_m       = float(dx_mm) * 1e-3
        self.n_voxels   = int(self.voxel_mask.sum().item())

        if self.n_voxels == 0:
            import warnings
            warnings.warn("[VolumetricHeatBC] solid_mask has zero voxels — no heat applied.")

        V = self.n_voxels * self.dx_m ** 3
        print(f"[VolumetricHeatBC] Q={Q_watts:.3g} W | "
              f"n_voxels={self.n_voxels} | V={V:.4e} m³ | "
              f"qvol={Q_watts/max(V,1e-30):.4e} W/m³")

    def src_step(
        self,
        rho_cp:            torch.Tensor,
        rho_cp_ref_J_m3K:  float,
        dx_phys_m:         float,
        dt_thermal_phys_s: float,
    ) -> torch.Tensor:
        out = torch.zeros_like(rho_cp)
        if self.n_voxels == 0:
            return out

        V_solid     = self.n_voxels * dx_phys_m ** 3          # m³
        qvol        = self.Q_watts / V_solid                   # W/m³
        rho_cp_phys = rho_cp[self.voxel_mask] * rho_cp_ref_J_m3K
        out[self.voxel_mask] = qvol / (rho_cp_phys + 1e-30) * dt_thermal_phys_s
        return out

    def q_total_W(self, *args, **kwargs) -> float:
        return self.Q_watts

    def update_Q(self, Q_watts: float) -> None:
        self.Q_watts = float(Q_watts)

class SurfaceFluxBC:
    """
    Pre-computes the voxel mask for a surface heat flux boundary condition.

    Parameters
    ----------
    solid_mask   : (nx, ny, nz) bool tensor on the target device.
                   Identifies which voxels belong to the solid receiving the flux.
    axis         : One of "+X", "-X", "+Y", "-Y", "+Z", "-Z".
                   "+Z" → flux enters the max-z face of the solid.
                   "-Z" → flux enters the min-z face.
    surface_L_mm : Physical length  of the heat source footprint [mm].
                   The first  in-plane axis dimension.
    surface_W_mm : Physical width   of the heat source footprint [mm].
                   The second in-plane axis dimension.
    q_flux_W_m2  : Heat flux magnitude [W/m²]. Positive = energy into the solid.
    dx_mm        : Voxel pitch [mm] (uniform cubic cells assumed).
    center_mm    : Optional (u_mm, v_mm) specifying the in-plane centre of the
                   heat source relative to the domain origin.
                   None → use the centroid of the detected surface voxels.

    In-plane axis convention
    ------------------------
    axis "+Z" / "-Z"  →  in-plane axes are X (L) and Y (W)
    axis "+Y" / "-Y"  →  in-plane axes are X (L) and Z (W)
    axis "+X" / "-X"  →  in-plane axes are Y (L) and Z (W)
    """

    # Map flow axis → (in-plane axis for L, in-plane axis for W)
    _PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def __init__(
        self,
        solid_mask:   Tensor,
        axis:         str,
        surface_L_mm: float,
        surface_W_mm: float,
        q_flux_W_m2:  float,
        dx_mm:        float,
        center_mm:    Optional[Tuple[float, float]] = None,
    ):
        axis = axis.strip().upper()
        assert len(axis) == 2 and axis[0] in ("+", "-") and axis[1] in ("X", "Y", "Z"), \
            f"axis must be one of '+X','-X','+Y','-Y','+Z','-Z'. Got: {axis!r}"

        self.axis_str     = axis
        self.sign         = +1 if axis[0] == "+" else -1
        self.ax           = {"X": 0, "Y": 1, "Z": 2}[axis[1]]
        self.q_flux_W_m2  = float(q_flux_W_m2)
        self.dx_mm        = float(dx_mm)
        self.surface_L_mm = float(surface_L_mm)
        self.surface_W_mm = float(surface_W_mm)

        # Build the voxel mask once at construction time.
        self.voxel_mask = self._build_voxel_mask(solid_mask, center_mm)
        self.n_voxels   = int(self.voxel_mask.sum().item())

        if self.n_voxels == 0:
            import warnings
            warnings.warn(
                f"[SurfaceFluxBC] axis={axis}: no solid voxels found in the "
                f"specified {surface_L_mm:.1f}×{surface_W_mm:.1f} mm footprint. "
                "Check your solid_mask and surface dimensions."
            )

    # ─────────────────────────────────────────────────────────── public ──

    def src_step(
        self,
        rho_cp:           Tensor,   # (nx,ny,nz) dimensionless lattice rho_cp
        rho_cp_ref_J_m3K: float,    # reference rho_cp [J/(m³·K)]
        dx_phys_m:        float,    # voxel pitch [m]
        dt_thermal_phys_s: float,   # thermal time step [s]
    ) -> Tensor:
        """
        Returns src_step contribution [K / thermal-step] for the surface flux.

        src_step[cell] = q_flux / (rho_cp_phys[cell] * dx) * dt_thermal

        Cells outside voxel_mask are zero.
        """
        out = torch.zeros_like(rho_cp)
        if self.n_voxels == 0:
            return out

        rho_cp_phys = rho_cp * rho_cp_ref_J_m3K          # [J/(m³·K)]
        coeff       = (
            self.q_flux_W_m2
            / (rho_cp_phys[self.voxel_mask] + 1e-30)
            / dx_phys_m
            * dt_thermal_phys_s
        )                                                  # [K/step]
        out[self.voxel_mask] = coeff
        return out

    def q_total_W(
        self,
        rho_cp:           Tensor,
        rho_cp_ref_J_m3K: float,
        dx_phys_m:        float,
        dt_thermal_phys_s: float,
    ) -> float:
        """
        Diagnostic: total heat power [W] applied by this BC.
        Q = q_flux * n_voxels * dx²
        """
        return self.q_flux_W_m2 * self.n_voxels * (dx_phys_m ** 2)

    def update_flux(self, q_flux_W_m2: float) -> None:
        """Change the flux magnitude at runtime (e.g. for transient loads)."""
        self.q_flux_W_m2 = float(q_flux_W_m2)

    def update_solid(
        self,
        solid_mask: Tensor,
        center_mm:  Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Rebuild the voxel mask from a new solid_mask.
        Call this if the solid geometry changes between runs.
        """
        self.voxel_mask = self._build_voxel_mask(solid_mask, center_mm)
        self.n_voxels   = int(self.voxel_mask.sum().item())

    # ─────────────────────────────────────────────────────────── private ─

    def _build_voxel_mask(
        self,
        solid_mask: Tensor,
        center_mm:  Optional[Tuple[float, float]],
    ) -> Tensor:
        """
        Detect surface voxels:
        1. Find the extreme solid voxel plane along `self.ax` (max for +, min for -).
        2. Intersect with `solid_mask` to keep only voxels that ARE solid.
        3. Filter by the L×W footprint centred at `center_mm` (or solid centroid).
        """
        device = solid_mask.device
        coords = solid_mask.nonzero(as_tuple=False)      # (N, 3), int64

        if coords.shape[0] == 0:
            return torch.zeros_like(solid_mask, dtype=torch.bool)

        # ── Step 1: extreme plane ─────────────────────────────────────────
        ax_coords = coords[:, self.ax]
        extreme   = int(ax_coords.max().item()) if self.sign == +1 \
                    else int(ax_coords.min().item())

        # Boolean mask selecting the extreme plane
        nx, ny, nz = solid_mask.shape
        plane_sel  = torch.zeros((nx, ny, nz), device=device, dtype=torch.bool)
        if   self.ax == 0: plane_sel[extreme, :, :] = True
        elif self.ax == 1: plane_sel[:, extreme, :] = True
        else:              plane_sel[:, :, extreme] = True

        surface = solid_mask & plane_sel                 # solid voxels on the face

        # ── Step 2: L×W footprint filter ─────────────────────────────────
        a1, a2    = self._PLANE_AXES[self.ax]
        dx_mm     = self.dx_mm
        half_L    = (self.surface_L_mm / 2.0) / dx_mm   # half-length in voxels
        half_W    = (self.surface_W_mm / 2.0) / dx_mm   # half-width  in voxels

        sv_coords = surface.nonzero(as_tuple=False)      # surface voxel coords

        if sv_coords.shape[0] == 0:
            return surface                               # no solid on that plane

        if center_mm is None:
            # Use centroid of the full body volume, not just the eroded surface plane
            # The surface plane loses edge voxels asymmetrically after SDF smoothing
            all_coords = coords  # already computed above as solid_mask.nonzero()
            c1 = all_coords[:, a1].float().mean().item()
            c2 = all_coords[:, a2].float().mean().item()
        else:
            c1 = float(center_mm[0]) / dx_mm
            c2 = float(center_mm[1]) / dx_mm

        # Build coordinate grids for the two in-plane axes
        # (only need per-axis 1-D vectors; broadcasting handles the rest)
        shape   = (nx, ny, nz)
        ranges  = [
            torch.arange(shape[i], device=device, dtype=torch.float32)
            for i in range(3)
        ]
        # Expand to 3-D with singleton dims for broadcasting
        g = []
        for i, r in enumerate(ranges):
            view_shape = [1, 1, 1]
            view_shape[i] = -1
            g.append(r.view(view_shape).expand(shape))

        in_bounds = (
            (g[a1] - c1).abs() <= half_L + 0.5          # +0.5 for boundary voxels
        ) & (
            (g[a2] - c2).abs() <= half_W + 0.5
        )

        return surface & in_bounds
