import torch

import torch
from dataclasses import dataclass
import trimesh
import numpy as np

@dataclass
class IBBData:
    n_links: int
    fluid_flat: torch.Tensor
    dir_i: torch.Tensor
    dir_opp: torch.Tensor
    q_ge_half: torch.Tensor
    w_i: torch.Tensor
    w_second: torch.Tensor
    ff_flat: torch.Tensor
    ff_valid: torch.Tensor

def build_links_from_mask_gpu(solid_geom: torch.Tensor,
                              c_np, opp_np) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # solid_geom: (nx,ny,nz) bool on CUDA
    device = solid_geom.device
    nx, ny, nz = solid_geom.shape
    fluid = ~solid_geom

    c = torch.as_tensor(c_np, device=device, dtype=torch.int64)      # (19,3)
    opp = torch.as_tensor(opp_np, device=device, dtype=torch.int64)  # (19,)

    link_flats = []
    link_dirs  = []
    link_ff    = []
    link_ff_ok = []

    # flat index helper
    def flat(i, j, k):
        return (i * (ny * nz) + j * nz + k)

    I, J, K = torch.meshgrid(
        torch.arange(nx, device=device),
        torch.arange(ny, device=device),
        torch.arange(nz, device=device),
        indexing="ij"
    )

    for q in range(1, 19):
        cx, cy, cz = int(c[q,0]), int(c[q,1]), int(c[q,2])

        # neighbor indices (no wrap). Build valid region mask.
        inb = I + cx
        jnb = J + cy
        knb = K + cz

        valid = (inb >= 0) & (inb < nx) & (jnb >= 0) & (jnb < ny) & (knb >= 0) & (knb < nz)

        # boundary link: current is fluid, neighbor is solid
        m = valid & fluid & solid_geom[inb.clamp(0,nx-1), jnb.clamp(0,ny-1), knb.clamp(0,nz-1)]
        if not m.any():
            continue

        # fluid node flat + direction id
        fflat = flat(I[m], J[m], K[m]).to(torch.int64)
        dirs  = torch.full_like(fflat, q, dtype=torch.int64)

        # x_ff = x_f - c_q (second fluid node opposite to wall)
        iff = I[m] - cx
        jff = J[m] - cy
        kff = K[m] - cz

        ff_ok = (iff >= 0) & (iff < nx) & (jff >= 0) & (jff < ny) & (kff >= 0) & (kff < nz) & fluid[iff, jff, kff]
        ff_flat = flat(iff.clamp(0,nx-1), jff.clamp(0,ny-1), kff.clamp(0,nz-1)).to(torch.int64)

        link_flats.append(fflat)
        link_dirs.append(dirs)
        link_ff.append(ff_flat)
        link_ff_ok.append(ff_ok)

    fluid_flat = torch.cat(link_flats, dim=0)
    dir_i      = torch.cat(link_dirs,  dim=0)
    ff_flat    = torch.cat(link_ff,    dim=0)
    ff_valid   = torch.cat(link_ff_ok, dim=0)

    dir_opp = opp[dir_i]

    return fluid_flat, dir_i, dir_opp, ff_flat, ff_valid


@torch.jit.script
def sdf_trilinear(phi: torch.Tensor,
                  origin: torch.Tensor,  # (3,) float32 world units (mm)
                  dx: float,
                  x: torch.Tensor        # (...,3) float32 world units
                  ) -> torch.Tensor:
    # phi shape: (nx, ny, nz) on GPU
    # x in same world coords as origin/dx
    g = (x - origin) / dx  # grid coords
    gx = g[..., 0].clamp(0.0, phi.shape[0] - 1.001)
    gy = g[..., 1].clamp(0.0, phi.shape[1] - 1.001)
    gz = g[..., 2].clamp(0.0, phi.shape[2] - 1.001)

    i0 = gx.floor().to(torch.int64); j0 = gy.floor().to(torch.int64); k0 = gz.floor().to(torch.int64)
    fx = (gx - i0.to(gx.dtype));      fy = (gy - j0.to(gy.dtype));      fz = (gz - k0.to(gz.dtype))

    i1 = (i0 + 1).clamp_max(phi.shape[0] - 1)
    j1 = (j0 + 1).clamp_max(phi.shape[1] - 1)
    k1 = (k0 + 1).clamp_max(phi.shape[2] - 1)

    c000 = phi[i0, j0, k0]
    c100 = phi[i1, j0, k0]
    c010 = phi[i0, j1, k0]
    c110 = phi[i1, j1, k0]
    c001 = phi[i0, j0, k1]
    c101 = phi[i1, j0, k1]
    c011 = phi[i0, j1, k1]
    c111 = phi[i1, j1, k1]

    cx00 = c000 + fx * (c100 - c000)
    cx10 = c010 + fx * (c110 - c010)
    cx01 = c001 + fx * (c101 - c001)
    cx11 = c011 + fx * (c111 - c011)

    cxy0 = cx00 + fy * (cx10 - cx00)
    cxy1 = cx01 + fy * (cx11 - cx01)

    return cxy0 + fz * (cxy1 - cxy0)

@torch.jit.script
def bfl_q_from_sdf(phi: torch.Tensor,
                   origin: torch.Tensor,   # (3,) mm
                   dx: float,
                   x0: torch.Tensor,       # (N,3) mm fluid cell centers
                   d: torch.Tensor,        # (N,3) unit directions
                   L: torch.Tensor,        # (N,) link lengths in mm
                   n_coarse: int = 6,
                   n_bisect: int = 10
                   ) -> torch.Tensor:
    # returns q in (0,1], with q=0.5 fallback when no crossing found
    N = x0.shape[0]
    # coarse samples t = [0, L] with n_coarse segments
    # Note: x0 should be in fluid => phi(x0) > 0; neighbor solid => phi(x0+L*d) < 0 ideally.
    t_vals = torch.linspace(0.0, 1.0, steps=n_coarse+1, device=x0.device, dtype=x0.dtype)  # (n+1,)
    t = (t_vals[None, :] * L[:, None])  # (N, n+1)
    xs = x0[:, None, :] + t[:, :, None] * d[:, None, :]  # (N,n+1,3)

    ph = sdf_trilinear(phi, origin, dx, xs.reshape(-1, 3)).reshape(N, -1)  # (N,n+1)

    # find first index where phi changes sign from + to -
    # assume start in fluid: ph[:,0] > 0
    s = ph > 0
    # crossing where s[k]=True and s[k+1]=False
    cross = s[:, :-1] & (~s[:, 1:])

    # default: halfway
    q = torch.full((N,), 0.5, device=x0.device, dtype=x0.dtype)

    any_cross = cross.any(dim=1)
    if any_cross.any():
        idx = cross.float().argmax(dim=1)  # first crossing segment (works because False=0)
        idx = idx.to(torch.int64)

        # bracket [t0,t1] for those rays
        t0 = t.gather(1, idx[:, None]).squeeze(1)
        t1 = t.gather(1, (idx+1)[:, None]).squeeze(1)

        # bisection refine root
        lo = t0.clone()
        hi = t1.clone()

        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            xm = x0 + mid[:, None] * d
            pm = sdf_trilinear(phi, origin, dx, xm)

            # want pm > 0 on fluid side, pm < 0 on solid side
            # if pm > 0, root is further (move lo); else move hi
            lo = torch.where(pm > 0, mid, lo)
            hi = torch.where(pm > 0, hi, mid)

        t_hit = hi
        q_hit = t_hit / (L + 1e-30)
        q = torch.where(any_cross, q_hit, q)

    # ensure valid range (numerically)
    return q


@torch.jit.script
def bfl_weights_from_q(q: torch.Tensor):
    ge = q >= 0.5
    w1 = torch.empty_like(q)
    w2 = torch.empty_like(q)

    # q >= 0.5
    qg = torch.where(ge, q, torch.ones_like(q))
    w1_ge = 1.0 / (2.0 * qg)
    w2_ge = 1.0 - w1_ge

    # q < 0.5
    ql = torch.where(~ge, q, 0.25 * torch.ones_like(q))
    w1_lt = 2.0 * ql
    w2_lt = 1.0 - 2.0 * ql

    w1 = torch.where(ge, w1_ge, w1_lt)
    w2 = torch.where(ge, w2_ge, w2_lt)
    return ge, w1, w2


def build_ibb_from_sdf_gpu(sim, voxel_origin_mm, pitch_mm, offset_ijk, c_np, opp_np):
    device = sim.device
    nx, ny, nz = sim.nx, sim.ny, sim.nz

    # domain origin in mesh coords (mm)
    origin_domain_mm = (torch.tensor(voxel_origin_mm, device=device, dtype=torch.float32)
                        - float(pitch_mm) * torch.tensor(offset_ijk, device=device, dtype=torch.float32))

    # links from STL-consistent solid mask
    fluid_flat, dir_i, dir_opp, ff_flat, ff_valid = build_links_from_mask_gpu(sim.solid, c_np, opp_np)
    N = int(fluid_flat.numel())

    # recover (i,j,k) from flat
    i = (fluid_flat // (ny * nz)).to(torch.float32)
    j = ((fluid_flat // nz) % ny).to(torch.float32)
    k = (fluid_flat % nz).to(torch.float32)

    # x0 in mm for each boundary link
    x0 = torch.stack([i, j, k], dim=1) * float(pitch_mm) + origin_domain_mm[None, :]  # (N,3)

    # direction unit vector d and link length L (mm)
    c = torch.tensor(c_np, device=device, dtype=torch.float32)  # (19,3)
    d_raw = c[dir_i]                                            # (N,3)
    cn = torch.linalg.norm(d_raw, dim=1)                        # (N,)
    d = d_raw / (cn[:, None] + 1e-30)
    L = float(pitch_mm) * cn                                    # (N,)

    # q from SDF (GPU)
    q = bfl_q_from_sdf(sim.phi, origin_domain_mm, float(pitch_mm), x0, d, L,
                       n_coarse=6, n_bisect=10)

    # weights
    ge, w1, w2 = bfl_weights_from_q(q)


    sim.ibb_data = IBBData(
        n_links=N,
        fluid_flat=fluid_flat.to(torch.int64),
        dir_i=dir_i.to(torch.int64),
        dir_opp=dir_opp.to(torch.int64),
        q_ge_half=ge,
        w_i=w1.to(torch.float32),
        w_second=w2.to(torch.float32),
        ff_flat=ff_flat.to(torch.int64),
        ff_valid=ff_valid,   # used only for q<0.5 branch
    )

    # diagnostics (single print)
    with torch.no_grad():
        qv = q[torch.isfinite(q)]
        print(f"[SDF-IBB] links={N} q[min,med,max]=({float(qv.min()):.4f},{float(qv.median()):.4f},{float(qv.max()):.4f}) "
              f"q<0.5={int((q<0.5).sum())}")


def build_phi_from_stl_mm(
    mesh: trimesh.Trimesh,
    nx: int, ny: int, nz: int,
    pitch_mm: float,
    origin_domain_mm: np.ndarray,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 250_000
):
    """
    Fast signed distance field builder (mm units).
    Uses ProximityQuery.signed_distance in chunks.
    """

    print("[SDF] Building signed distance field...")

    pq = trimesh.proximity.ProximityQuery(mesh)

    total_pts = nx * ny * nz
    phi = np.empty(total_pts, dtype=np.float32)

    for start in range(0, total_pts, chunk_size):
        end = min(start + chunk_size, total_pts)

        flat_ids = np.arange(start, end)

        i = flat_ids // (ny * nz)
        j = (flat_ids // nz) % ny
        k = flat_ids % nz

        pts = np.stack([i, j, k], axis=1).astype(np.float64)
        pts = origin_domain_mm[None, :] + pitch_mm * pts

        # ✅ Correct API
        d_signed = pq.signed_distance(pts)

        phi[start:end] = d_signed.astype(np.float32)

        if start % (5 * chunk_size) == 0:
            print(f"[SDF] {end}/{total_pts} points processed")

    phi = phi.reshape(nx, ny, nz)

    print("[SDF] Done.")
    print(f"[SDF] phi range: min={phi.min():.6f}, max={phi.max():.6f}")

    return torch.tensor(phi, device=device, dtype=dtype)

from scipy.ndimage import distance_transform_edt

def build_phi_from_voxel_mask_mm(
    voxel_solid: np.ndarray,   # bool (nx,ny,nz)
    pitch_mm: float,
    device="cuda",
    dtype=torch.float32
):
    """
    Extremely fast SDF from voxel mask.
    Exact for voxel geometry.
    """

    print("[SDF] Building SDF from voxel mask (distance transform)...")

    solid = voxel_solid.astype(bool)
    fluid = ~solid

    # Distance to nearest solid (for fluid region)
    dist_out = distance_transform_edt(fluid) * pitch_mm

    # Distance to nearest fluid (for solid region)
    dist_in  = distance_transform_edt(solid) * pitch_mm

    phi = dist_out
    phi[solid] = -dist_in[solid]

    print("[SDF] Done.")
    print(f"[SDF] phi range: min={phi.min():.6f}, max={phi.max():.6f}")

    return torch.tensor(phi, device=device, dtype=dtype)

