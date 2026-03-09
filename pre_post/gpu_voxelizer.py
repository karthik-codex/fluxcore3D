"""
gpu_voxelizer.py
================
Fast STL voxelizer for LBM grids. Drop-in replacement for trimesh subdivide.

Algorithm (GPU path)
--------------------
  Moller-Trumbore ray-triangle intersection in +Z direction.
  Parity is computed correctly: for each Z cell center z_k, count how many
  surface triangles are pierced at z < z_k.  Odd count => inside (solid).
  This avoids the edge double-counting bug of bin-based parity.

  Chunked over both rays and triangles to stay within VRAM.
  RTX 4090: heatsink 36x45mm at 0.1mm pitch (~18M cells) in ~0.5 s.

Usage
-----
    from gpu_voxelizer import patch_stl_voxelizer
    patch_stl_voxelizer(STLVoxelizer)                    # GPU (default)
    patch_stl_voxelizer(STLVoxelizer, method="pyvista")  # CPU fallback

    geometry = STLVoxelizer(stl_path=..., pitch_mm=0.1, ...)
"""

from __future__ import annotations
import numpy as np
import torch
from datetime import datetime


def _log(msg):
    print(f"[ {datetime.now().strftime('%I:%M:%S %p')} ] [GPUVox] {msg}")


# ---------------------------------------------------------------------------
#  Moller-Trumbore: Z-direction rays vs triangles
# ---------------------------------------------------------------------------

def _mt_intersect_z(ox, oy, v0, v1, v2, z_orig, eps=1e-7):
    """
    ox, oy : (R,)   ray origins in XY (ray dir = +Z)
    v0,v1,v2: (T,3) triangle vertices
    Returns hit_z (R, T): Z coordinate of hit, NaN if no intersection.
    """
    e1 = v1 - v0; e2 = v2 - v0           # (T, 3)
    h_x = -e2[:, 1]; h_y = e2[:, 0]      # (T,)  h = cross((0,0,1), e2)
    det = e1[:,0]*h_x + e1[:,1]*h_y       # (T,)
    valid = det.abs() > eps
    inv_det = torch.where(valid, 1.0/(det + (~valid).float()), torch.zeros_like(det))

    s_x = ox.unsqueeze(1) - v0[:,0].unsqueeze(0)   # (R, T)
    s_y = oy.unsqueeze(1) - v0[:,1].unsqueeze(0)

    u  = (s_x*h_x + s_y*h_y) * inv_det             # (R, T)
    q_z = s_x*e1[:,1] - s_y*e1[:,0]
    v_ = q_z * inv_det

    s_zt = z_orig - v0[:,2]                         # (T,)
    q_x = s_y*e1[:,2] - s_zt*e1[:,1]
    q_y = s_zt*e1[:,0] - s_x*e1[:,2]
    t   = (e2[:,0]*q_x + e2[:,1]*q_y + e2[:,2]*q_z) * inv_det

    hit = (valid & (u >= 0.) & (u <= 1.) & (v_ >= 0.) & (u+v_ <= 1.) & (t >= 0.))
    return torch.where(hit, z_orig + t, torch.full_like(t, float('nan')))


# ---------------------------------------------------------------------------
#  GPU voxelizer
# ---------------------------------------------------------------------------

def _repair_mesh(mesh, verbose=True):
    """Force-close a non-watertight mesh before voxelization."""
    import trimesh
    if not mesh.is_watertight:
        if verbose:
            _log(f"Mesh not watertight (open edges: {len(mesh.as_open_bounds())}). Repairing...")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        if not mesh.is_watertight:
            # Nuclear option: convex hull of the mesh — loses detail but always solid
            _log("WARNING: repair failed, using convex hull approximation")
            mesh = mesh.convex_hull
    return mesh

def gpu_voxelize_mesh(
    mesh,
    pitch_mm: float,
    device: str = "cuda",
    ray_chunk: int = 2048,
    tri_chunk: int = 32768,
    t_inner: int = 512,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (voxel_solid: uint8 (Nx,Ny,Nz),  voxel_origin_mm: float64 (3,))
    """
    dev = torch.device(device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    if dev.type == "cpu" and device == "cuda":
        if verbose: _log("WARNING: CUDA not available, using CPU (will be slow)")
    mesh = _repair_mesh(mesh, verbose=verbose)
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces,    dtype=np.int64)
    T = faces.shape[0]
    if verbose: _log(f"Mesh: {len(verts)} verts, {T} triangles | pitch={pitch_mm} mm")

    tv0 = torch.tensor(verts[faces[:,0]], dtype=torch.float32, device=dev)
    tv1 = torch.tensor(verts[faces[:,1]], dtype=torch.float32, device=dev)
    tv2 = torch.tensor(verts[faces[:,2]], dtype=torch.float32, device=dev)

    bmin = verts.min(0); bmax = verts.max(0)
    origin = bmin - pitch_mm * 0.5
    nx = int(np.ceil((bmax[0]-origin[0])/pitch_mm)) + 2
    ny = int(np.ceil((bmax[1]-origin[1])/pitch_mm)) + 2
    nz = int(np.ceil((bmax[2]-origin[2])/pitch_mm)) + 2
    if verbose: _log(f"Grid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.2f} M cells")

    z0 = float(origin[2])
    # Z cell centres (nz,) for parity counting
    z_cen = torch.tensor(z0 + (np.arange(nz)+0.5)*pitch_mm,
                         dtype=torch.float32, device=dev)

    # Tiny irrational perturbation to all ray origins -- prevents rays from
    # landing exactly on shared triangle edges (diagonal of each mesh face),
    # which would cause double-counting and spurious holes in the solid.
    PERTURB = 1.3e-5 * pitch_mm
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    ox_all = torch.tensor(origin[0] + gx.ravel()*pitch_mm + PERTURB*np.sqrt(2),
                          dtype=torch.float32, device=dev)
    oy_all = torch.tensor(origin[1] + gy.ravel()*pitch_mm + PERTURB*np.sqrt(3),
                          dtype=torch.float32, device=dev)
    ix_all = torch.tensor(gx.ravel(), dtype=torch.int32, device=dev)
    iy_all = torch.tensor(gy.ravel(), dtype=torch.int32, device=dev)
    n_rays = ox_all.shape[0]
    n_ch   = (n_rays + ray_chunk - 1) // ray_chunk

    voxels = np.zeros((nx, ny, nz), dtype=np.uint8)

    for r0 in range(0, n_rays, ray_chunk):
        r1 = min(r0+ray_chunk, n_rays); R = r1-r0
        ox = ox_all[r0:r1]; oy = oy_all[r0:r1]

        # Accumulate: number of triangle surfaces pierced BELOW each Z center
        below = torch.zeros(R, nz, dtype=torch.int16, device=dev)

        for t0 in range(0, T, tri_chunk):
            t1 = min(t0+tri_chunk, T)
            hz = _mt_intersect_z(ox, oy, tv0[t0:t1], tv1[t0:t1], tv2[t0:t1], z0)
            # hz: (R, Tc), NaN = miss
            Tc = hz.shape[1]
            for ti in range(0, Tc, t_inner):
                ti1 = min(ti+t_inner, Tc)
                hz_i  = hz[:, ti:ti1]                              # (R, t_inner)
                valid = ~torch.isnan(hz_i)
                if not valid.any(): continue
                # (R, t_inner, 1) < (1, 1, nz)  ->  (R, t_inner, nz)
                cnt = (hz_i.unsqueeze(2) < z_cen.view(1,1,nz)) & valid.unsqueeze(2)
                below += cnt.sum(dim=1).to(torch.int16)

        # Parity: odd number of surfaces below => inside
        solid = (below % 2 == 1)
        voxels[ix_all[r0:r1].cpu().numpy(),
               iy_all[r0:r1].cpu().numpy(), :] = solid.cpu().numpy().astype(np.uint8)

        # if verbose and ((r0//ray_chunk) % max(1, n_ch//10) == 0):
        #     _log(f"  {r1/n_rays*100:.0f}%  ({r1:,}/{n_rays:,})")

    if verbose:
        _log(f"Done.  Solid: {int(voxels.sum()):,}  ({voxels.mean()*100:.1f}% fill)")
    return voxels, origin.astype(np.float64)


# ---------------------------------------------------------------------------
#  PyVista fallback (CPU)
# ---------------------------------------------------------------------------

def pyvista_voxelize_mesh(mesh, pitch_mm: float, verbose: bool = True):
    """CPU voxelizer via PyVista/VTK.  Robust, no subdivision."""
    import pyvista as pv, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    mesh.export(tmp)
    pv_mesh = pv.read(tmp)
    os.unlink(tmp)

    if verbose: _log(f"Voxelizing with PyVista at pitch={pitch_mm} mm ...")
    vox = pv.voxelize(pv_mesh, density=pitch_mm, check_surface=False)

    b = np.array(pv_mesh.bounds)
    origin = np.array([b[0], b[2], b[4]], dtype=np.float64)
    nx = int(np.ceil((b[1]-b[0])/pitch_mm)) + 1
    ny = int(np.ceil((b[3]-b[2])/pitch_mm)) + 1
    nz = int(np.ceil((b[5]-b[4])/pitch_mm)) + 1

    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    pts  = np.array(vox.cell_centers().points)
    if len(pts):
        ix = np.clip(((pts[:,0]-origin[0])/pitch_mm).astype(int), 0, nx-1)
        iy = np.clip(((pts[:,1]-origin[1])/pitch_mm).astype(int), 0, ny-1)
        iz = np.clip(((pts[:,2]-origin[2])/pitch_mm).astype(int), 0, nz-1)
        grid[ix, iy, iz] = 1

    if verbose: _log(f"Done.  Solid: {int(grid.sum()):,}")
    return grid, origin


# ---------------------------------------------------------------------------
#  Drop-in patch for STLVoxelizer._voxelize
# ---------------------------------------------------------------------------

def patch_stl_voxelizer(cls, method: str = "gpu",
                         device: str = "cuda", ray_chunk: int = 2048):
    """
    Patch STLVoxelizer before instantiation to replace the broken
    trimesh-subdivide voxelizer.

    Parameters
    ----------
    method    : "gpu"     -- CUDA ray casting  (~0.5 s for 18M cells on 4090)
                "pyvista" -- PyVista/VTK CPU   (~30-60 s, robust fallback)
    device    : "cuda" or "cpu"
    ray_chunk : rays per GPU chunk; reduce to 512 if CUDA OOM

    Example
    -------
        from gpu_voxelizer import patch_stl_voxelizer
        patch_stl_voxelizer(STLVoxelizer)           # one line, before init

        geometry = STLVoxelizer(
            stl_path  = "geometries/stl/heatsink.stl",
            build_dir = "-Y",
            part_scale = 0.3,
            pitch_mm   = 0.1,
            out_npz    = "geometry.npz",
        )
    """
    if method == "gpu":
        def _vox(self):
            v, o = gpu_voxelize_mesh(self.mesh, self.pitch_mm,
                                     device=device, ray_chunk=ray_chunk)
            self.voxel_origin_mm = o
            return v
    else:
        def _vox(self):
            v, o = pyvista_voxelize_mesh(self.mesh, self.pitch_mm)
            self.voxel_origin_mm = o
            return v

    cls._voxelize = _vox
    # print(f"[gpu_voxelizer] STLVoxelizer._voxelize patched "
    #       f"(method={method}  device={device})")
    return cls

def _original_voxelize(self) -> np.ndarray:
    """Original trimesh-based voxelizer (CPU)."""
    vox = self.mesh.voxelized(pitch=self.pitch_mm).fill()
    self.voxel_origin_mm = vox.transform[:3, 3].copy()
    return vox.matrix.astype(np.uint8)


def unpatch_stl_voxelizer(cls):
    """Restore STLVoxelizer._voxelize to the original trimesh implementation."""
    cls._voxelize = _original_voxelize
    return cls


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import trimesh, time

    print("=== Box 10x20x5 mm at 0.5 mm pitch ===")
    mesh = trimesh.creation.box(extents=[10., 20., 5.])
    t0   = time.perf_counter()
    v, o = gpu_voxelize_mesh(mesh, pitch_mm=0.5, device="cuda")
    dt   = time.perf_counter() - t0
    inner = v[1:-1, 1:-1, 1:-1]
    print(f"Time: {dt:.2f} s | Grid: {v.shape} | "
          f"Interior solid: {inner.mean():.3f}  (expect 1.0)")

    print("\n=== Sphere r=5 mm at 0.5 mm pitch ===")
    mesh2 = trimesh.creation.icosphere(radius=5., subdivisions=4)
    v2, _ = gpu_voxelize_mesh(mesh2, pitch_mm=0.5, device="cuda", verbose=False)
    theo  = 4/3 * 3.14159 * 5.**3
    err   = abs(v2.sum() - theo) / theo * 100
    print(f"Volume: {v2.sum():.0f} mm^3  (theory {theo:.0f})  err={err:.1f}%")
