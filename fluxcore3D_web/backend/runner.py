"""
backend/runner.py

Architecture:
  - Sim imports at module level → CUDA compiles once at startup, not on first click
  - Domain preview: SolidAssembly.build_domain() → stamp region array →
    PyVista threshold per body → surface mesh → .gltf file in static/preview/ →
    served over HTTP → scene.gltf(url) in browser (Three.js loads as file, no websocket cap)
  - Results viewer: same approach, temperature-colored GLTF per body
  - No point clouds. No websocket geometry limits. Works identically on localhost and AWS.
"""
from __future__ import annotations
import sys
import os
import threading
import traceback
import tempfile
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────

def _bootstrap_path() -> bool:
    here = Path(__file__).resolve().parent.parent
    for candidate in [here.parent, here.parent.parent]:
        if (candidate / "UI_components" / "cht_worker.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return True
    return False

_bootstrap_path()

# ── Static dirs ────────────────────────────────────────────────────────────────
_APP_DIR      = Path(__file__).resolve().parent.parent
_PREVIEW_DIR  = _APP_DIR / "static" / "preview"
_RESULTS_DIR  = _APP_DIR / "static" / "results"
_PROJECTS_DIR = _APP_DIR / "projects"
_SIM_OUT_DIR  = _APP_DIR / "sim_results"
_PROJECTS_DIR.mkdir(exist_ok=True)
_SIM_OUT_DIR.mkdir(exist_ok=True)
_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Persistent temp dir for voxel NPZs ────────────────────────────────────────
_VOXEL_DIR = Path(tempfile.mkdtemp(prefix="fc3d_vox_"))

# ── Eager sim imports — runs at app startup ────────────────────────────────────

HAS_SIM = False
SIM_IMPORT_ERROR = None
SimConfig3D = SolidPropsSI = STLVoxelizer = LBMCHT3D_Torch = None
SolidAssembly = patch_stl_voxelizer = unpatch_stl_voxelizer = None

try:
    from UI_components.cht_sim_imports import (
        HAS_SIM               as _HAS,
        SIM_IMPORT_ERROR      as _ERR,
        SimConfig3D           as _SimConfig3D,
        SolidPropsSI          as _SolidPropsSI,
        STLVoxelizer          as _STLVoxelizer,
        LBMCHT3D_Torch        as _LBMCHT,
        SolidAssembly         as _SolidAssembly,
        patch_stl_voxelizer   as _patch,
        unpatch_stl_voxelizer as _unpatch,
    )
    HAS_SIM               = _HAS
    SIM_IMPORT_ERROR      = _ERR
    SimConfig3D           = _SimConfig3D
    SolidPropsSI          = _SolidPropsSI
    STLVoxelizer          = _STLVoxelizer
    LBMCHT3D_Torch        = _LBMCHT
    SolidAssembly         = _SolidAssembly
    patch_stl_voxelizer   = _patch
    unpatch_stl_voxelizer = _unpatch
    if HAS_SIM:
        print("[runner] Sim modules loaded OK — CUDA extensions ready.")
    else:
        print(f"[runner] Sim modules unavailable: {SIM_IMPORT_ERROR}")
except Exception as _e:
    SIM_IMPORT_ERROR = _e
    print(f"[runner] Sim import failed: {_e}")

# ── Shared state ───────────────────────────────────────────────────────────────

_sim_state: dict = dict(running=False, pct=0, log=[], term_log=[], done=False,
                        aborted=False, result=None, error=None)

_preview_state: dict = dict(
    running=False, done=False, error=None, log=[],
    # body_meta: list of {name, color, gltf_url}  — no geometry, just HTTP URLs
    body_meta=None,
    grid_mm=None,   # [Wx, Wy, Wz] in mm
    flow_dir=None,
)


def abort_sim():
    _sim_state["aborted"] = True


def reset_sim_state():
    _sim_state.update(running=False, pct=0, log=[], term_log=[], done=False,
                      aborted=False, result=None, error=None)


def reset_preview_state():
    _preview_state.update(running=False, done=False, error=None,
                          log=[], body_meta=None, grid_mm=None, flow_dir=None)


# ── Simulation runner ──────────────────────────────────────────────────────────

def _blocking_run(params: dict):
    try:
        from UI_components.cht_worker import run_simulation_blocking
    except ImportError as e:
        msg = f"Cannot import simulation modules: {e}"
        print(f"[SIM ERROR] {msg}")
        _sim_state["error"] = msg
        _sim_state["done"]  = True
        return

    # ── Disable tqdm entirely — it hammers I/O and tanks GPU throughput ────────
    os.environ["TQDM_DISABLE"] = "1"

    # ── Stdout/stderr capture ─────────────────────────────────────────────────
    # term_log → big terminal panel (all stdout/stderr)
    # log      → small status log (high-level log_fn only)

    class _StreamCapture:
        """Tees to original stream AND appends to _sim_state['term_log'].
        Lines arriving via \r (tqdm progress overwrites) are silently dropped.
        Lines arriving via \n (real print statements) are kept.
        """
        def __init__(self, orig):
            self._orig = orig
            self._buf  = ""
        def write(self, text):
            self._orig.write(text)
            self._orig.flush()
            self._buf += text
            while "\n" in self._buf or "\r" in self._buf:
                nl = self._buf.find("\n")
                cr = self._buf.find("\r")
                # Pick whichever separator comes first
                if nl == -1 or (cr != -1 and cr < nl):
                    # \r line → tqdm progress overwrite → DROP
                    line, self._buf = self._buf.split("\r", 1)
                else:
                    # \n line → real print → KEEP
                    line, self._buf = self._buf.split("\n", 1)
                    line = line.rstrip()
                    if line:
                        _sim_state["term_log"].append(line)
        def flush(self):    self._orig.flush()
        def isatty(self):   return False
        def fileno(self):   return self._orig.fileno()

    _orig_out, _orig_err = sys.stdout, sys.stderr
    sys.stdout = _StreamCapture(_orig_out)
    sys.stderr = _StreamCapture(_orig_err)

    def log_fn(msg):
        # High-level: small log gets it, terminal gets it too
        _sim_state["log"].append(msg)
        _sim_state["term_log"].append(f"[SIM] {msg}")
    def pct_fn(n):    _sim_state["pct"] = n
    def abort_fn():   return _sim_state["aborted"]

    _sim_state["term_log"].append("[SIM] _blocking_run starting…")
    orig_cwd = os.getcwd()
    try:
        # cht_worker._apply_bc loads NPZs by relative path (spec.name + ".npz").
        # All NPZs live in _VOXEL_DIR, so keep cwd there for the entire sim run.
        os.chdir(str(_VOXEL_DIR))
        # Override output dir so results land in a persistent location
        project_name = params.get("project_name", "simulation")
        safe_proj = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
        sim_out = _SIM_OUT_DIR / safe_proj
        sim_out.mkdir(exist_ok=True)

        result = run_simulation_blocking(
            params, log_fn=log_fn, progress_fn=pct_fn, abort_fn=abort_fn)
        # Resolve to absolute path while cwd is still _VOXEL_DIR
        result = str(Path(result).resolve())

        # Copy NPZ to persistent sim_results/<project>/ so it survives temp dir cleanup
        try:
            import shutil as _shutil
            dest = sim_out / Path(result).name
            if Path(result).resolve() != dest.resolve():
                _shutil.copy2(result, dest)
            result = str(dest)
            print(f"[SIM] Results saved → {result}")
        except Exception as _ce:
            print(f"[SIM] Warning: could not copy result to projects dir: {_ce}")
        _sim_state["result"] = result
        print(f"[SIM] Done → {result}")
    except Exception as exc:
        full = traceback.format_exc()
        print(f"[SIM ERROR] {exc}\n{full}")
        _sim_state["error"] = f"{exc}\n\n{full}"
    finally:
        sys.stdout, sys.stderr = _orig_out, _orig_err
        os.chdir(orig_cwd)
        _sim_state["done"]    = True
        _sim_state["running"] = False


def launch_sim_thread(params: dict):
    reset_sim_state()
    _sim_state["running"] = True
    threading.Thread(target=_blocking_run, args=(params,), daemon=True).start()


# ── Preview runner ─────────────────────────────────────────────────────────────

_CSS_HEX = {
    "steelblue":"#4682B4","darkorange":"#FF8C00","gold":"#FFD700",
    "sienna":"#A0522D","mediumpurple":"#9370DB","tomato":"#FF6347",
    "limegreen":"#32CD32","hotpink":"#FF69B4","deepskyblue":"#00BFFF",
    "coral":"#FF7F50",
}
def _css_hex(c: str) -> str:
    return _CSS_HEX.get(c, c) if not c.startswith("#") else c


# ── Minimal GLTF writer ────────────────────────────────────────────────────────
# PyVista cannot write GLTF. We write it manually — it's just JSON + binary buffer.
# No trimesh/pygltflib dependency.

def _polydata_to_gltf(mesh, out_path: Path, color_rgb: tuple | None = None,
                      vertex_colors_u8=None, opacity: float = 1.0):
    """
    Write a PyVista PolyData mesh to a self-contained .gltf (JSON + embedded base64).

    mesh             : pyvista.PolyData with triangulated faces
    color_rgb        : (r,g,b) floats 0-1 for uniform color
    vertex_colors_u8 : np.ndarray shape (N,3) uint8 per-vertex RGB
    opacity          : 0.0-1.0, baked into GLTF material alphaMode
    """
    import numpy as np, json, base64

    # Triangulate and get arrays
    mesh = mesh.triangulate()
    verts = mesh.points.astype(np.float32)           # (N,3)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]         # (F,3) strip leading counts
    faces = faces.astype(np.uint32)

    n_verts = len(verts)
    n_faces = len(faces)

    # Binary buffer: positions then indices
    pos_bytes  = verts.tobytes()
    idx_bytes  = faces.tobytes()

    # Optional vertex colors
    col_bytes  = b""
    has_colors = vertex_colors_u8 is not None and len(vertex_colors_u8) == n_verts
    if has_colors:
        colors_f32 = vertex_colors_u8.astype(np.float32) / 255.0
        col_bytes  = colors_f32.tobytes()

    total_bytes = len(pos_bytes) + len(idx_bytes) + len(col_bytes)
    buf_b64 = base64.b64encode(pos_bytes + idx_bytes + col_bytes).decode()

    vmin = verts.min(axis=0).tolist()
    vmax = verts.max(axis=0).tolist()

    buffer_views = [
        {"buffer": 0, "byteOffset": 0,                                   "byteLength": len(pos_bytes), "target": 34962},  # ARRAY_BUFFER
        {"buffer": 0, "byteOffset": len(pos_bytes),                       "byteLength": len(idx_bytes), "target": 34963},  # ELEMENT_ARRAY_BUFFER
    ]
    accessors = [
        {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": n_verts,
         "type": "VEC3", "min": vmin, "max": vmax},
        {"bufferView": 1, "byteOffset": 0, "componentType": 5125, "count": n_faces * 3,
         "type": "SCALAR"},
    ]
    attributes = {"POSITION": 0}

    if has_colors:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": len(pos_bytes) + len(idx_bytes),
            "byteLength": len(col_bytes),
            "target": 34962,
        })
        accessors.append({
            "bufferView": 2, "byteOffset": 0,
            "componentType": 5126, "count": n_verts,
            "type": "VEC3",
        })
        attributes["COLOR_0"] = 2

    alpha = float(max(0.0, min(1.0, opacity)))
    alpha_mode = "BLEND" if alpha < 0.99 else "OPAQUE"
    if color_rgb and not has_colors:
        r, g, b = color_rgb
        material = {"pbrMetallicRoughness": {
            "baseColorFactor": [r, g, b, alpha],
            "metallicFactor": 0.1, "roughnessFactor": 0.7,
        }, "alphaMode": alpha_mode, "doubleSided": alpha < 0.99}
    else:
        material = {"pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, alpha],
            "metallicFactor": 0.1, "roughnessFactor": 0.7,
        }, "alphaMode": alpha_mode, "doubleSided": alpha < 0.99}

    gltf = {
        "asset": {"version": "2.0", "generator": "FluxCore3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": 1, "material": 0}]}],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": total_bytes,
                     "uri": f"data:application/octet-stream;base64,{buf_b64}"}],
    }
    out_path.write_text(json.dumps(gltf), encoding="utf-8")


def _blocking_preview(params: dict):
    """
    Pipeline (mirrors cht_main_window._preview_geometry + _render_domain_impl):

      1. patch/unpatch GPU ray-tracer
      2. SolidAssembly.build_domain()  →  per-body .npz in _VOXEL_DIR
      3. Stamp all bodies into integer region array (region==idx per body)
      4. PyVista ImageData → threshold([idx-0.5, idx+0.5]) per body → surface mesh
      5. mesh.save("body.gltf")  into static/preview/
      6. Store {name, color, gltf_url} in _preview_state["body_meta"]

    The browser loads each GLTF over HTTP (scene.gltf(url)).
    No geometry passes through the websocket → no size limit → works for 30M-voxel models.
    """
    import numpy as np

    def log(msg: str):
        _preview_state["log"].append(msg)
        print(f"[PREVIEW] {msg}")

    if not HAS_SIM:
        _preview_state["error"] = "Simulation modules not available."
        _preview_state["done"]  = True
        return

    try:
        import pyvista as pv
        from backend.model_io import SOLID_PRESETS

        dom    = params["domain"]
        dx_mm  = float(params["dx_mm"])
        nx     = round(dom["Lx"] * 1000 / dx_mm)
        ny     = round(dom["Ly"] * 1000 / dx_mm)
        nz_cfg = round(dom["Lz"] * 1000 / dx_mm)
        flow_d = params["flow_dir"]
        j0_div = float(params["j0_divisor"])

        def _as_dict(b):
            if isinstance(b, dict):
                return b
            return dict(stl_path=b.stl_path, name=b.name,
                        build_dir=getattr(b, "build_dir", "+Z"),
                        material=getattr(b, "material", "Aluminum (LBM-scaled)"),
                        color=getattr(b, "color", "steelblue"),
                        role=getattr(b, "role", "fluid_base"))

        body_dicts = []
        for b in params["bodies"]:
            try:
                bd = _as_dict(b)
                stl = bd.get("stl_path", "")
                if stl and not Path(stl).exists():
                    log(f"WARNING: STL not found, skipping body '{bd.get('name','?')}': {stl}")
                    continue
                body_dicts.append(bd)
            except (FileNotFoundError, OSError) as _e:
                name = getattr(b, "name", b.get("name","?") if isinstance(b,dict) else "?")
                log(f"WARNING: skipping body '{name}' — STL path error: {_e}")
        if not body_dicts:
            raise RuntimeError("No valid bodies with accessible STL files. Check STL paths.")
        log(f"{len(body_dicts)} bodies: {[b['name'] for b in body_dicts]}")

        # ── GPU ray-trace ─────────────────────────────────────────────────────
        if params.get("gpu_raytrace", True):
            patch_stl_voxelizer(STLVoxelizer)
        else:
            unpatch_stl_voxelizer(STLVoxelizer)

        # ── SimConfig3D ───────────────────────────────────────────────────────
        cfg = SimConfig3D(
            Lx_m=dom["Lx"], Ly_m=dom["Ly"], Lz_m=dom["Lz"],
            flow_bc="inlet_outlet", flow_dir=flow_d, transverse_walls=True,
            u_in_mps=float(params.get("u_in", 1.0)),
            dt_thermal_phys_s=1.0, temp_bc="fixed", t_ambient_C=20.0,
            heating_mode="off", qdot_total_W=0.0,
            solid_init_mode="ambient", T_hot_C=0.0,
        )
        cfg.dx_mm = dx_mm; cfg.nx = nx; cfg.ny = ny
        cfg.nz = nz_cfg; cfg.obstacle = None

        # ── Build assembly — NPZ paths in _VOXEL_DIR ──────────────────────────
        bodies_for_assembly = []
        for bd in body_dicts:
            mat_key  = bd.get("material", "Aluminum (LBM-scaled)")
            mat_dict = SOLID_PRESETS.get(mat_key, SOLID_PRESETS["Aluminum (LBM-scaled)"])
            bodies_for_assembly.append(dict(
                stl      = bd["stl_path"],
                npz      = str(_VOXEL_DIR / f"{bd['name']}.npz"),
                name     = bd["name"],
                build_dir= bd.get("build_dir", "+Z"),
                material = SolidPropsSI(**mat_dict),
                color    = bd.get("color", "steelblue"),
                role     = bd.get("role", "fluid_base"),
            ))

        orig_cwd = os.getcwd()
        log(f"Building domain  flow={flow_d}  grid={nx}×{ny}×{nz_cfg}  bodies={len(bodies_for_assembly)}")
        try:
            os.chdir(str(_VOXEL_DIR))
            assembly = SolidAssembly(
                bodies=bodies_for_assembly, fluid_Lz_m=dom["Lz"], dx_mm=dx_mm,
                cfg=cfg, flow_dir=flow_d, j0_divisor=j0_div,
                STLVoxelizer_cls=STLVoxelizer, LBMCHT_cls=LBMCHT3D_Torch,
            )
            assembly.build_domain()
        finally:
            os.chdir(orig_cwd)

        log(f"build_domain() done  fluid_k0={cfg.fluid_k0}  nz={cfg.nz}")

        nx_d, ny_d, nz_d = cfg.nx, cfg.ny, cfg.nz

        # ── Stamp region array (exactly as _render_domain_impl) ───────────────
        region = np.zeros((nx_d, ny_d, nz_d), dtype=np.int32)
        for idx, bd_viz in enumerate(assembly.viz_list(), start=1):
            npz = bd_viz["npz"]
            i0, j0, k0 = (int(x) for x in bd_viz["offset_ijk"])
            vox = np.load(npz, allow_pickle=True)["voxel_solid"].astype(bool)
            Gx, Gy, Gz = vox.shape
            # Clamp to domain bounds
            di0 = max(i0, 0); di1 = min(i0+Gx, nx_d)
            dj0 = max(j0, 0); dj1 = min(j0+Gy, ny_d)
            dk0 = max(k0, 0); dk1 = min(k0+Gz, nz_d)
            si0 = di0-i0; si1 = si0+(di1-di0)
            sj0 = dj0-j0; sj1 = sj0+(dj1-dj0)
            sk0 = dk0-k0; sk1 = sk0+(dk1-dk0)
            region[di0:di1, dj0:dj1, dk0:dk1][vox[si0:si1, sj0:sj1, sk0:sk1]] = idx
            log(f"  stamped {bd_viz['name']}  idx={idx}  cells={(vox[si0:si1,sj0:sj1,sk0:sk1]).sum()}")

        # ── PyVista ImageData → threshold per body → GLTF ────────────────────
        # spacing = dx_mm so coordinates are in mm, matching scene world space
        grid = pv.ImageData(
            dimensions=(nx_d+1, ny_d+1, nz_d+1),
            spacing=(dx_mm, dx_mm, dx_mm),
            origin=(0.0, 0.0, 0.0),
        )
        grid.cell_data["region"] = region.reshape(-1, order="F")

        # Clear old GLTF files from previous preview
        for f in _PREVIEW_DIR.glob("*.gltf"):
            try: f.unlink()
            except: pass
        for f in _PREVIEW_DIR.glob("*.bin"):
            try: f.unlink()
            except: pass

        body_meta = []
        for idx, bd_viz in enumerate(assembly.viz_list(), start=1):
            name  = bd_viz["name"]
            color = bd_viz["color"]
            safe  = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

            try:
                mesh = grid.threshold([idx - 0.5, idx + 0.5], scalars="region").extract_surface()
                if mesh.n_points == 0:
                    log(f"  {name}: threshold empty — skipping")
                    continue

                try:
                    mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.1)
                except Exception:
                    pass

                gltf_path = _PREVIEW_DIR / f"{safe}.gltf"
                r = int(_css_hex(color)[1:3], 16) / 255.0
                g = int(_css_hex(color)[3:5], 16) / 255.0
                b = int(_css_hex(color)[5:7], 16) / 255.0
                _polydata_to_gltf(mesh, gltf_path, color_rgb=(r, g, b))

                gltf_url = f"/static/preview/{safe}.gltf"
                body_meta.append(dict(name=name, color=color, gltf_url=gltf_url))
                log(f"  {name}: {mesh.n_points} pts → {gltf_path.name} ({gltf_path.stat().st_size//1024} KB)")

            except Exception as exc:
                log(f"  [WARN] {name}: GLTF export failed: {exc}")

        _preview_state["body_meta"] = body_meta
        _preview_state["grid_mm"]   = [nx_d * dx_mm, ny_d * dx_mm, nz_d * dx_mm]
        _preview_state["flow_dir"]  = flow_d
        log(f"Preview GTLFs ready — {len(body_meta)} bodies.")

    except Exception as exc:
        full_tb = traceback.format_exc()
        print(f"[PREVIEW] EXCEPTION:\n{full_tb}")
        _preview_state["error"] = f"{exc}\n\n{full_tb[-1200:]}"
    finally:
        _preview_state["done"]    = True
        _preview_state["running"] = False


def launch_preview_thread(params: dict):
    reset_preview_state()
    _preview_state["running"] = True
    threading.Thread(target=_blocking_preview, args=(params,), daemon=True).start()


# ── Results GLTF export ────────────────────────────────────────────────────────

def _polydata_lines_to_gltf(polydata, out_path: Path, vertex_colors_u8=None):
    """Export a PyVista PolyData with lines (streamlines) to GLTF mode=1."""
    import numpy as np, json, base64

    verts = polydata.points.astype(np.float32)
    # lines array: [count, i0, i1, count, i2, i3, ...]
    lines_raw = polydata.lines
    # Parse into pairs
    idx_pairs = []
    i = 0
    while i < len(lines_raw):
        n = lines_raw[i]; i += 1
        seg = lines_raw[i:i+n]; i += n
        for j in range(len(seg)-1):
            idx_pairs.append([seg[j], seg[j+1]])
    if not idx_pairs:
        return False
    indices = np.array(idx_pairs, dtype=np.uint32)

    pos_bytes = verts.tobytes()
    idx_bytes = indices.tobytes()
    col_bytes = b""
    has_colors = vertex_colors_u8 is not None and len(vertex_colors_u8) == len(verts)
    if has_colors:
        col_bytes = (vertex_colors_u8.astype(np.float32) / 255.0).tobytes()

    total = len(pos_bytes) + len(idx_bytes) + len(col_bytes)
    buf_b64 = base64.b64encode(pos_bytes + idx_bytes + col_bytes).decode()
    vmin = verts.min(axis=0).tolist(); vmax = verts.max(axis=0).tolist()

    buffer_views = [
        {"buffer":0,"byteOffset":0,"byteLength":len(pos_bytes),"target":34962},
        {"buffer":0,"byteOffset":len(pos_bytes),"byteLength":len(idx_bytes),"target":34963},
    ]
    accessors = [
        {"bufferView":0,"byteOffset":0,"componentType":5126,"count":len(verts),
         "type":"VEC3","min":vmin,"max":vmax},
        {"bufferView":1,"byteOffset":0,"componentType":5125,
         "count":len(indices)*2,"type":"SCALAR"},
    ]
    attributes = {"POSITION": 0}
    if has_colors:
        buffer_views.append({"buffer":0,"byteOffset":len(pos_bytes)+len(idx_bytes),
                              "byteLength":len(col_bytes),"target":34962})
        accessors.append({"bufferView":2,"byteOffset":0,"componentType":5126,
                          "count":len(verts),"type":"VEC3"})
        attributes["COLOR_0"] = 2

    gltf = {
        "asset":{"version":"2.0","generator":"FluxCore3D"},
        "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":attributes,"indices":1,
                                  "material":0,"mode":1}]}],  # mode 1 = LINES
        "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.3,0.8,1.0,1.0],
                       "metallicFactor":0.0,"roughnessFactor":1.0}}],
        "accessors":accessors,"bufferViews":buffer_views,
        "buffers":[{"byteLength":total,
                    "uri":f"data:application/octet-stream;base64,{buf_b64}"}],
    }
    out_path.write_text(json.dumps(gltf), encoding="utf-8")
    return True


def export_results_gltf(npz_path: str, storage: dict) -> dict:
    """
    Export per-body temperature-colored GLTF + optional fluid surface + streamlines.
    Replicates launch_results_viewer() logic from cht_main_window.py.

    Returns:
        {
          "bodies":      [{name, gltf_url, t_min, t_max, opacity?}],
          "streamlines": gltf_url or None,
          "fmin": float, "fmax": float,
          "field": str, "cmap": str,
        }
    """
    import numpy as np
    import matplotlib
    import pyvista as pv

    out = {"bodies": [], "streamlines": None,
           "fmin": 0.0, "fmax": 1.0, "field": "T", "cmap": "inferno"}

    try:
        data  = np.load(npz_path, allow_pickle=True)
        field = storage.get("vis_field", "T")
        if field not in data.files:
            field = "T"
        arr = data[field]
        F   = (arr[0] if arr.ndim == 4 else arr).astype(np.float32)
        nx, ny, nz = F.shape

        body_names = list(data["body_names"]) if "body_names" in data.files else []
        vis_bodies = storage.get("vis_bodies", {})
        # default: show all if vis_bodies is empty
        show_bodies = [n for n in body_names
                       if vis_bodies.get(n, True)] if vis_bodies else body_names

        # Use np.asarray().flat[0] to safely handle both 0-d and 1-element arrays
        # (NumPy version differences between machines cause int() to fail on 1-element arrays)
        def _scalar(arr): return np.asarray(arr).flat[0]
        fluid_k0   = int(_scalar(data["fluid_k0"])) if "fluid_k0" in data.files else 0
        flow_dir   = storage.get("result_flow_dir",
                     str(_scalar(data["flow_dir"])) if "flow_dir" in data.files else "+X")

        # ── Compute color range from SELECTED domains only (mirrors old GUI) ──
        body_masks = {}
        for name in body_names:
            key = f"solid_{name}"
            if key in data.files:
                m = data[key].astype(bool)
                if m.ndim == 4: m = m[0]
                body_masks[name] = m

        solid_union = np.zeros((nx,ny,nz), dtype=bool)
        for m in body_masks.values(): solid_union |= m

        sample_vals = []
        for name in show_bodies:
            if name in body_masks:
                sample_vals.append(F[body_masks[name]])
        show_fluid = storage.get("vis_fluid", False)
        if show_fluid:
            su_fluid = solid_union.copy()
            if fluid_k0 > 0: su_fluid[:,:,:fluid_k0] = True
            fluid_mask = ~su_fluid
            sample_vals.append(F[fluid_mask & np.isfinite(F)])

        filtered = [v[np.isfinite(v)] for v in sample_vals if len(v) > 0]
        if filtered:
            all_v = np.concatenate(filtered)
            fmin, fmax = float(all_v.min()), float(all_v.max())
        else:
            fmin, fmax = float(np.nanmin(F)), float(np.nanmax(F))
        if fmin == fmax: fmax = fmin + 1.0

        out["fmin"] = fmin; out["fmax"] = fmax
        out["field"] = field
        out["cmap"]  = storage.get("vis_cmap", "inferno")
        out["grid_size"] = [nx, ny, nz]

        cmap_name = storage.get("vis_cmap", "inferno")
        cmap = matplotlib.colormaps.get_cmap(cmap_name)

        # Clear old GLTF files
        for f in _RESULTS_DIR.glob("*.gltf"):
            try: f.unlink()
            except: pass
        for f in _RESULTS_DIR.glob("*.bin"):
            try: f.unlink()
            except: pass

        # ── PyVista shared grid for sampling ──────────────────────────────────
        # Use spacing=1 (voxel units) matching the old code
        grid_pts = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
        grid_pts.point_data["scalars"] = F.reshape(-1, order="F").astype(np.float32)

        # ── Per-body temperature-colored surfaces ─────────────────────────────
        # Use threshold+extract_surface (voxel faces) instead of contour (marching
        # cubes) — marching cubes on binary voxel masks produces noisy, artifact-
        # filled surfaces. The face-extraction approach gives clean blocky geometry
        # that exactly matches the simulation domain, then smooth_taubin rounds it.
        for name in show_bodies:
            if name not in body_masks: continue
            mask = body_masks[name]
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            try:
                body_grid = pv.ImageData(dimensions=(nx+1,ny+1,nz+1), spacing=(1,1,1), origin=(0,0,0))
                body_grid.cell_data["mask"] = mask.reshape(-1, order="F").astype(np.uint8)
                surf = body_grid.threshold(0.5, scalars="mask").extract_surface()
                if surf.n_points == 0: continue

                # Smooth to reduce blocky staircase appearance
                try:
                    surf = surf.smooth_taubin(n_iter=20, pass_band=0.1)
                except Exception:
                    pass  # smoothing optional

                # Sample temperature at surface vertices
                surf = surf.sample(grid_pts)
                t_vals = surf.point_data.get("scalars", None)
                if t_vals is None:
                    t_vals = np.full(surf.n_points, (fmin+fmax)/2, dtype=np.float32)
                t_norm = np.clip((t_vals - fmin) / (fmax - fmin), 0, 1)
                rgba   = cmap(t_norm)
                rgb_u8 = (rgba[:,:3]*255).astype(np.uint8)

                gltf_path = _RESULTS_DIR / f"{safe}.gltf"
                _polydata_to_gltf(surf, gltf_path, vertex_colors_u8=rgb_u8)
                out["bodies"].append(dict(
                    name=name, gltf_url=f"/static/results/{safe}.gltf",
                    t_min=float(t_vals.min()), t_max=float(t_vals.max()),
                ))
                print(f"[Results] {name}: {surf.n_points} pts  T={t_vals.min():.1f}–{t_vals.max():.1f}")
            except Exception as exc:
                import traceback as _tb
                print(f"[Results] {name} failed: {_tb.format_exc()[-300:]}")

        # ── Fluid surface ─────────────────────────────────────────────────────
        if show_fluid:
            try:
                fluid_opacity = float(storage.get("vis_fluid_opacity", 0.3))
                su_fluid = solid_union.copy()
                if fluid_k0 > 0: su_fluid[:,:,:fluid_k0] = True
                fluid_mask = ~su_fluid

                # Make a grid with the fluid region tagged, extract outer surface
                fg = pv.ImageData(dimensions=(nx+1,ny+1,nz+1), spacing=(1,1,1), origin=(0,0,0))
                fg.cell_data["fluid"] = fluid_mask.reshape(-1, order="F").astype(np.uint8)
                # T is point data — separate grid for sampling
                fg_pts = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
                fg_pts.point_data["T"] = F.reshape(-1, order="F").astype(np.float32)
                fluid_surf = fg.threshold(0.5, scalars="fluid").extract_surface()
                if fluid_surf.n_points > 0:
                    fluid_surf = fluid_surf.sample(fg_pts)
                    t_f = fluid_surf.point_data.get("T",
                          np.full(fluid_surf.n_points, (fmin+fmax)/2, dtype=np.float32))
                    t_norm = np.clip((t_f - fmin) / (fmax - fmin), 0, 1)
                    rgba   = cmap(t_norm)
                    rgb_u8 = (rgba[:,:3]*255).astype(np.uint8)
                    gltf_path = _RESULTS_DIR / "fluid_domain.gltf"
                    _polydata_to_gltf(fluid_surf, gltf_path, vertex_colors_u8=rgb_u8,
                                      opacity=fluid_opacity)
                    out["bodies"].append(dict(
                        name="Fluid domain",
                        gltf_url="/static/results/fluid_domain.gltf",
                        opacity=fluid_opacity,
                        t_min=float(t_f.min()), t_max=float(t_f.max()),
                    ))
                    print(f"[Results] Fluid: {fluid_surf.n_points} pts")
            except Exception as exc:
                import traceback as _tb
                print(f"[Results] Fluid failed: {_tb.format_exc()[-300:]}")

        # ── Streamlines ───────────────────────────────────────────────────────
        if storage.get("vis_stream", True) and all(
            k in data.files for k in ("u","v","w")
        ):
            try:
                n_seeds   = int(storage.get("vis_nseeds", 150))
                seed_mode = storage.get("vis_seed_mode", "inlet_plane")
                axis_map  = {"+X":0,"-X":0,"+Y":1,"-Y":1,"+Z":2,"-Z":2}
                flow_axis = axis_map.get(flow_dir, 0)

                u = (data["u"][0] if data["u"].ndim==4 else data["u"]).astype(np.float32)
                v = (data["v"][0] if data["v"].ndim==4 else data["v"]).astype(np.float32)
                w = (data["w"][0] if data["w"].ndim==4 else data["w"]).astype(np.float32)

                gv = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
                gv.point_data["U"] = np.stack([
                    u.reshape(-1,order="F"),
                    v.reshape(-1,order="F"),
                    w.reshape(-1,order="F")], axis=1)
                gv.set_active_vectors("U")

                rng = np.random.default_rng(42)
                if seed_mode == "inlet_plane":
                    idx_in = 2
                    if flow_axis == 0:
                        coords = np.argwhere(~solid_union[idx_in,:,:])
                        chosen = coords[rng.integers(0,len(coords),min(n_seeds,len(coords)))]
                        pts = np.stack([np.full(len(chosen),idx_in), chosen[:,0], chosen[:,1]],1).astype(np.float32)
                    elif flow_axis == 1:
                        coords = np.argwhere(~solid_union[:,idx_in,:])
                        chosen = coords[rng.integers(0,len(coords),min(n_seeds,len(coords)))]
                        pts = np.stack([chosen[:,0], np.full(len(chosen),idx_in), chosen[:,1]],1).astype(np.float32)
                    else:
                        coords = np.argwhere(~solid_union[:,:,idx_in])
                        chosen = coords[rng.integers(0,len(coords),min(n_seeds,len(coords)))]
                        pts = np.stack([chosen[:,0], chosen[:,1], np.full(len(chosen),idx_in)],1).astype(np.float32)
                else:
                    pts = (rng.random((n_seeds,3)) * np.array([nx,ny,nz])).astype(np.float32)

                seeds = pv.PolyData(pts)
                sl = gv.streamlines_from_source(
                    seeds, vectors="U", integration_direction="forward",
                    initial_step_length=1.5, terminal_speed=0.0,
                    max_steps=400,          # hard cap — 150 seeds × 400 = 60k pts max
                    integrator_type=45)

                # Subsample if still too large (browser limit ~100k pts for GLTF)
                MAX_SL_POINTS = 80_000
                if sl.n_points > MAX_SL_POINTS:
                    keep_ratio = MAX_SL_POINTS / sl.n_points
                    sl = sl.decimate_pro(1.0 - keep_ratio) if hasattr(sl, "decimate_pro") else sl

                if sl.n_points > 0 and sl.n_lines > 0:
                    U_sl = sl.point_data.get("U", None)
                    if U_sl is not None:
                        spd = np.linalg.norm(U_sl, axis=1).astype(np.float32)
                    else:
                        spd = np.zeros(sl.n_points, dtype=np.float32)
                    spd_norm = np.clip((spd - spd.min()) / max(spd.max()-spd.min(), 1e-10), 0, 1)
                    rgba_sl = matplotlib.colormaps.get_cmap("Spectral")(spd_norm)
                    rgb_sl  = (rgba_sl[:,:3]*255).astype(np.uint8)
                    sl_path = _RESULTS_DIR / "streamlines.gltf"
                    if _polydata_lines_to_gltf(sl, sl_path, vertex_colors_u8=rgb_sl):
                        out["streamlines"] = "/static/results/streamlines.gltf"
                        print(f"[Results] Streamlines: {sl.n_points} pts  {sl.n_lines} lines")
            except Exception as exc:
                import traceback as _tb
                print(f"[Results] Streamlines failed: {_tb.format_exc()[-400:]}")

    except Exception as exc:
        import traceback as _tb
        print(f"[Results export] FAILED: {_tb.format_exc()}")

    return out


# ── Legacy render_results_b64 (kept for fallback) ─────────────────────────────

def render_results_b64(npz_path: str, flow_dir: str = "+X",
                       field: str = "T", cmap: str = "inferno") -> str | None:
    import base64, io
    try:
        import numpy as np, pyvista as pv
        data = np.load(npz_path, allow_pickle=True)
        if field not in data.files: field = "T"
        arr = data[field]
        F   = (arr[0] if arr.ndim == 4 else arr).astype(np.float32)
        nx, ny, nz = F.shape
        body_names = list(data["body_names"]) if "body_names" in data.files else []
        body_masks = {}
        for name in body_names:
            key = f"solid_{name}"
            if key in data.files:
                m = data[key].astype(bool)
                if m.ndim == 4: m = m[0]
                body_masks[name] = m
        all_v = F[np.isfinite(F)]
        fmin, fmax = (float(all_v.min()), float(all_v.max())) if len(all_v) else (0.0, 1.0)
        if fmin == fmax: fmax += 1.0
        pl = pv.Plotter(off_screen=True, window_size=(900, 600))
        pl.set_background("#141414")
        grid_pts = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
        grid_pts.point_data["scalars"] = F.reshape(-1, order="F").astype(np.float32)
        added = [False]
        sbar  = dict(title=field, n_labels=5, fmt="%.4g", vertical=True,
                     position_x=0.91, position_y=0.10, width=0.04, height=0.75,
                     title_font_size=18, label_font_size=14, color="white")
        for name, mask in body_masks.items():
            bg = pv.ImageData(dimensions=(nx,ny,nz), spacing=(1,1,1), origin=(0,0,0))
            bg.point_data["mask"] = mask.reshape(-1, order="F").astype(np.uint8)
            surf = bg.contour([0.5], scalars="mask").sample(grid_pts)
            if surf.n_points == 0: continue
            pl.add_mesh(surf, scalars="scalars", cmap=cmap, clim=(fmin,fmax),
                        show_scalar_bar=not added[0],
                        scalar_bar_args=sbar if not added[0] else {})
            added[0] = True
        pl.add_axes(); pl.reset_camera()
        img = pl.screenshot(return_img=True); pl.close()
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.fromarray(img).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f"[runner] render_results_b64 failed: {exc}")
        return None
