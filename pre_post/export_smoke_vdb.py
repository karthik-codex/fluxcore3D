"""
export_smoke_vdb.py
===================
Reads Zarr with real tracer field → writes per-frame VDB files for Blender.

Usage:
    python export_smoke_vdb.py snapshots/flow_anim.zarr --out exports/
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path
import numpy as np

try:
    import openvdb as vdb
except ImportError:
    print("ERROR: conda install -c conda-forge openvdb"); sys.exit(1)

try:
    import zarr
except ImportError:
    print("ERROR: pip install zarr"); sys.exit(1)

import warnings
import pyvista as pv
warnings.filterwarnings("ignore")


def load_zarr(path):
    store    = zarr.open(str(path), mode="r")
    attrs    = dict(store.attrs)
    nx, ny, nz = int(attrs["nx"]), int(attrs["ny"]), int(attrs["nz"])
    n_frames = int(attrs["n_frames"])
    dx_mm    = float(attrs.get("dx_mm", 1.0))
    flow_dir = str(attrs.get("flow_dir", "-Z"))
    fields   = store["fields"]

    body_names, body_masks = [], []
    if "masks" in store:
        for k in store["masks"]:
            arr = np.array(store["masks"][k]).astype(bool)
            body_names.append(k.replace("solid_", ""))
            body_masks.append(arr)

    solid = np.zeros((nx, ny, nz), dtype=bool)
    for bm in body_masks:
        solid |= bm
    fluid = ~solid

    has_tracer = "tracer" in fields

    def load_frame(i):
        out = {k: np.array(fields[k][i]).astype(np.float32)
               for k in ("u", "v", "w") if k in fields}
        if has_tracer:
            out["tracer"] = np.array(fields["tracer"][i]).astype(np.float32)
        return out

    print(f"[Zarr] {n_frames} frames | {nx}×{ny}×{nz} | has_tracer={has_tracer}")
    if not has_tracer:
        print("  WARNING: no tracer field found — re-run save_flow_animation.py to regenerate Zarr")
    return nx, ny, nz, n_frames, dx_mm, flow_dir, fluid, solid, body_names, body_masks, load_frame, has_tracer


def write_vdb(path, density, speed):
    d_grid = vdb.FloatGrid(0.0); d_grid.name = "density"
    d_grid.copyFromArray(density.astype(np.float32))

    s_grid = vdb.FloatGrid(0.0); s_grid.name = "speed"
    s_grid.copyFromArray(speed.astype(np.float32))

    vdb.write(str(path), [d_grid, s_grid])


def export_solids(nx, ny, nz, body_names, body_masks, solid_dir):
    solid_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, mask in zip(body_names, body_masks):
        try:
            g = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
            g.cell_data["m"] = mask.ravel(order="F").astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                surf = (g.threshold(0.5, scalars="m")
                          .extract_surface(algorithm="dataset_surface")
                          .clean().smooth(n_iter=40, relaxation_factor=0.08))
            if surf.n_points > 0:
                p = solid_dir / f"{name}.ply"
                surf.save(str(p))
                out.append({"name": name, "file": f"solids/{name}.ply"})
                print(f"  [solid] {name}: {surf.n_cells:,} faces")
        except Exception as e:
            print(f"  [solid] {name} failed: {e}")
    return out


def export_smoke(zarr_path, out_dir="exports"):
    import time
    from scipy.ndimage import gaussian_filter
    t0 = time.time()

    zarr_path = Path(zarr_path)
    out_dir   = Path(out_dir)
    vol_dir   = out_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz, n_frames, dx_mm, flow_dir, fluid, solid, \
        body_names, body_masks, load_frame, has_tracer = load_zarr(zarr_path)

    # global speed range for colour normalisation
    mid = load_frame(n_frames // 2)
    spd_mid = np.sqrt(mid["u"]**2 + mid["v"]**2 + mid["w"]**2)
    spd_max = float(np.percentile(spd_mid[fluid], 99)) + 1e-8
    print(f"  speed_max (99th pct) = {spd_max:.5f}")

    print("Exporting solid meshes...")
    solid_meta = export_solids(nx, ny, nz, body_names, body_masks, out_dir / "solids")

    try:
        from tqdm import tqdm
        frame_iter = tqdm(range(n_frames), unit="fr")
    except ImportError:
        frame_iter = range(n_frames)

    print(f"Writing {n_frames} VDB frames...")
    for fi in frame_iter:
        fr  = load_frame(fi)
        spd = np.sqrt(fr["u"]**2 + fr["v"]**2 + fr["w"]**2)

        if has_tracer:
            # real tracer: smooth slightly for volumetric look
            density = gaussian_filter(fr["tracer"], sigma=0.8)
            density[~fluid] = 0.0
            density = np.clip(density, 0.0, 1.0)
        else:
            # fallback: speed as density
            density = gaussian_filter(np.clip(spd / spd_max, 0, 1).astype(np.float32), sigma=1.2)
            density[~fluid] = 0.0

        spd_norm = np.clip(spd / spd_max, 0.0, 1.0).astype(np.float32)
        spd_norm[~fluid] = 0.0

        write_vdb(vol_dir / f"smoke_{fi+1:04d}.vdb", density, spd_norm)

    meta = {
        "nx": nx, "ny": ny, "nz": nz, "dx_mm": dx_mm,
        "n_frames": n_frames, "flow_dir": flow_dir,
        "vdb_dir": "volumes", "vdb_pattern": "smoke_####.vdb",
        "solids": solid_meta,
    }
    with open(out_dir / "meta_smoke.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Exporter] Done in {time.time()-t0:.0f}s → {vol_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="exports")
    args = ap.parse_args()
    export_smoke(args.input, args.out)
