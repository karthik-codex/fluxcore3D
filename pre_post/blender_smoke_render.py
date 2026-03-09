"""
blender_smoke_render.py
=======================
Blender 5.0 script: imports smoke VDB sequence + solid PLYs, renders animation.

Run:
    blender --background --python blender_smoke_render.py -- \
        --export-dir C:/path/to/exports \
        --out-dir    C:/path/to/renders \
        --width 1920 --height 1080

Or launch GUI (no --background) to adjust manually before rendering.
"""

import bpy, json, sys, math, mathutils
from pathlib import Path

def get_args():
    argv = sys.argv
    if "--" in argv: argv = argv[argv.index("--") + 1:]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir",       default="exports")
    ap.add_argument("--out-dir",          default="renders")
    ap.add_argument("--width",            type=int,   default=1920)
    ap.add_argument("--height",           type=int,   default=1080)
    ap.add_argument("--fps",              type=int,   default=24)
    ap.add_argument("--samples",          type=int,   default=64)
    ap.add_argument("--density-scale",    type=float, default=8.0)
    ap.add_argument("--emit-strength",    type=float, default=6.0)
    ap.add_argument("--solid-opacity",    type=float, default=0.55)
    ap.add_argument("--gui",              action="store_true")
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                bpy.data.cameras, bpy.data.volumes):
        for item in list(col): col.remove(item)


def setup_cycles(width, height, samples):
    sc = bpy.context.scene
    sc.render.engine                     = "CYCLES"
    sc.render.resolution_x               = width
    sc.render.resolution_y               = height
    sc.render.resolution_percentage      = 100
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform      = "Filmic"
    sc.view_settings.look                = "High Contrast"
    sc.view_settings.exposure            = -4.2

    cy = sc.cycles
    cy.samples               = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold    = 0.01
    cy.use_denoising         = True
    cy.volume_step_rate      = 0.1   # finer volume stepping = better smoke quality
    cy.volume_max_steps      = 1024
    try:    cy.denoiser = "OPTIX"
    except: cy.denoiser = "OPENIMAGEDENOISE"
    cy.device = "GPU"

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type in ("CUDA","OPTIX")
            if d.use: print(f"  [GPU] {d.name}")
    except Exception as e:
        print(f"  [GPU] warning: {e}")


def setup_world():
    w = bpy.data.worlds.new("W")
    bpy.context.scene.world = w
    try: w.use_nodes = True
    except: pass
    nt = w.node_tree; nt.nodes.clear()
    bg  = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs[0])
    bg.inputs["Color"].default_value    = (0.003, 0.005, 0.015, 1.0)
    bg.inputs["Strength"].default_value = 0.8


def add_camera(cx, cy, cz, d):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "Cam"
    bpy.context.scene.camera = cam
    cam.location = mathutils.Vector((cx, cy - d*1.4, cz + d*0.15))
    target    = mathutils.Vector((cx, cy, cz * 0.5))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z","Y").to_euler()
    cam.data.lens      = 60
    cam.data.clip_end  = d * 25


def add_lights(cx, cy, cz, d):
    def area(name, loc, energy, color, size=100.):
        bpy.ops.object.light_add(type="AREA", location=loc)
        o = bpy.context.active_object; o.name = name
        o.data.energy = energy; o.data.color = color[:3]; o.data.size = size
        direction = mathutils.Vector((cx,cy,cz)) - mathutils.Vector(loc)
        o.rotation_euler = direction.to_track_quat("-Z","Y").to_euler()
    area("Key",  (cx+d*.8, cy-d*.7, cz+d*.9), 80000, (1.0, 0.96, 0.88))
    area("Fill", (cx-d*.6, cy+d*.5, cz+d*.3), 20000, (0.4, 0.55, 1.0))
    area("Rim",  (cx,      cy+d*.9, cz-d*.1), 10000, (1.0, 0.5,  0.2))


BODY_COLORS = {
    "heatsink":   (0.50, 0.60, 0.72, 1.0),
    "CuSpray":    (0.72, 0.42, 0.08, 1.0),
    "TIM":        (0.60, 0.55, 0.10, 1.0),
    "heatsource": (0.80, 0.06, 0.02, 1.0),
    "solid":      (0.40, 0.50, 0.62, 1.0),
}


def make_solid_material(name, color, opacity):
    mat = bpy.data.materials.new(f"M_{name}")
    try: mat.use_nodes = True
    except: pass
    mat.blend_method = "BLEND" if opacity < 0.99 else "OPAQUE"
    nt = mat.node_tree; nt.nodes.clear()
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out  = nt.nodes.new("ShaderNodeOutputMaterial")

    # Special: heat source glows red
    if name == "heatsource":
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value    = (1.0, 0.15, 0.02, 1.0)
        emit.inputs["Strength"].default_value = 8.0
        mix  = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.7
        nt.links.new(emit.outputs[0], mix.inputs[1])
        nt.links.new(bsdf.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0],  out.inputs["Surface"])
    else:
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value   = 0.92
    bsdf.inputs["Roughness"].default_value  = 0.12
    bsdf.inputs["Alpha"].default_value      = opacity
    return mat


def make_smoke_material(density_scale, emit_strength):
    """
    Principled Volume shader with CONSTANT density (no VDB attribute nodes).
    ShaderNodeAttribute does not reliably read VDB float grids in Blender 5.0.
    Constant density guarantees visible smoke. Color ramp on speed still attempted.
    """
    mat = bpy.data.materials.new("SmokeMat")
    try: mat.use_nodes = True
    except: pass
    nt = mat.node_tree; nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")

    # Density FROM VDB grid — attribute_type=GEOMETRY is required in Blender 5.0
    attr_d = nt.nodes.new("ShaderNodeAttribute")
    attr_d.attribute_name = "density"
    attr_d.attribute_type = "GEOMETRY"
    mul_d = nt.nodes.new("ShaderNodeMath")
    mul_d.operation = "MULTIPLY"
    mul_d.inputs[1].default_value = density_scale
    nt.links.new(attr_d.outputs["Fac"], mul_d.inputs[0])
    nt.links.new(mul_d.outputs["Value"], vol.inputs["Density"])

    # Speed attribute for color
    attr_s = nt.nodes.new("ShaderNodeAttribute")
    attr_s.attribute_name = "speed"
    attr_s.attribute_type = "GEOMETRY"
    ramp   = nt.nodes.new("ShaderNodeValToRGB")
    cr     = ramp.color_ramp
    cr.interpolation = "LINEAR"
    cr.elements[0].position = 0.0;  cr.elements[0].color = (0.10, 0.05, 0.60, 1)  # deep blue
    cr.elements[1].position = 1.0;  cr.elements[1].color = (1.00, 0.90, 0.10, 1)  # yellow
    e1 = cr.elements.new(0.25); e1.color = (0.05, 0.50, 0.90, 1)  # cyan
    e2 = cr.elements.new(0.50); e2.color = (0.10, 0.85, 0.20, 1)  # green
    e3 = cr.elements.new(0.75); e3.color = (1.00, 0.40, 0.00, 1)  # orange

    nt.links.new(attr_s.outputs["Fac"],  ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"],  vol.inputs["Emission Color"])
    nt.links.new(ramp.outputs["Color"],  vol.inputs["Color"])  # scatter color too

    # Emission from speed
    mul_e = nt.nodes.new("ShaderNodeMath"); mul_e.operation = "MULTIPLY"
    mul_e.inputs[1].default_value = emit_strength
    nt.links.new(attr_s.outputs["Fac"], mul_e.inputs[0])
    nt.links.new(mul_e.outputs["Value"], vol.inputs["Emission Strength"])

    vol.inputs["Color"].default_value = (0.1, 0.4, 1.0, 1.0)
    vol.inputs["Anisotropy"].default_value = 0.3
    vol.inputs["Absorption Color"].default_value = (0.0, 0.02, 0.05, 1.0)

    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    return mat


def import_ply(path, name):
    bpy.ops.wm.ply_import(filepath=str(path))
    obj = bpy.context.active_object; obj.name = name
    for p in obj.data.polygons: p.use_smooth = True
    return obj


def import_vdb_first(vdb_dir, pattern):
    """Import first VDB frame to create the volume object."""
    first_vdb = vdb_dir / pattern.replace("####", "0001")
    if not first_vdb.exists():
        print(f"  ERROR: {first_vdb} not found"); return None
    bpy.ops.object.volume_import(filepath=str(first_vdb), files=[])
    vol_obj = bpy.context.active_object
    vol_obj.name = "Smoke"
    # Disable sequence mode — we swap filepath manually each frame
    try:
        vol_obj.data.is_sequence = False
    except Exception:
        pass
    print(f"  [VDB] imported {first_vdb.name}  active_voxels check via filepath swap")
    return vol_obj


def set_vdb_frame(vol_obj, vdb_dir, pattern, fi):
    """Swap the volume filepath to frame fi (1-indexed) before rendering."""
    fname = pattern.replace("####", f"{fi:04d}")
    fpath = str(vdb_dir / fname)
    vol_obj.data.filepath = fpath
    # Force Blender to reload the volume data
    vol_obj.data.sequence_mode = "CLIP" if hasattr(vol_obj.data, "sequence_mode") else None
    try:
        bpy.ops.object.select_all(action="DESELECT")
        vol_obj.select_set(True)
        bpy.context.view_layer.objects.active = vol_obj
    except Exception:
        pass


def main():
    import time
    args       = get_args()
    export_dir = Path(args.export_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = export_dir / "meta_smoke.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run export_smoke_vdb.py first.")
        sys.exit(1)
    with open(meta_path) as f: meta = json.load(f)

    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    cx, cy, cz = nx/2, ny/2, nz/2
    d          = (nx**2 + ny**2 + nz**2) ** 0.5
    n_frames   = meta["n_frames"]

    print(f"[BlenderSmoke] {nx}x{ny}x{nz} | {n_frames} frames | "
          f"{args.width}x{args.height} | samples={args.samples}")

    clear_scene()
    setup_cycles(args.width, args.height, args.samples)
    setup_world()
    add_camera(cx, cy, cz, d)
    add_lights(cx, cy, cz, d)

    # Set timeline
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end   = n_frames
    sc.render.fps  = args.fps

    # Import solid meshes (once, persistent)
    for s in meta.get("solids", []):
        ply = export_dir / s["file"]
        if not ply.exists(): continue
        obj = import_ply(ply, f"solid_{s['name']}")
        col = BODY_COLORS.get(s["name"], BODY_COLORS["solid"])
        obj.data.materials.append(
            make_solid_material(s["name"], col, args.solid_opacity))
        print(f"  [Solid] {s['name']}")

    # Import VDB smoke (first frame — filepath swapped per render below)
    vdb_dir   = export_dir / meta["vdb_dir"]
    pattern   = meta["vdb_pattern"]
    smoke_obj = import_vdb_first(vdb_dir, pattern)
    if smoke_obj:
        smoke_obj.data.materials.append(
            make_smoke_material(args.density_scale, args.emit_strength))
        print(f"  [Smoke] VDB loaded  dims={smoke_obj.dimensions[:]}")
    else:
        print("  ERROR: VDB import failed"); sys.exit(1)

    if args.gui:
        print("\n  [GUI mode] Scene ready. Press F12 to render.")
        return

    # Render all frames — swap VDB filepath explicitly before each render
    print(f"\nRendering {n_frames} frames...")
    times = []
    for fi in range(1, n_frames + 1):
        t0 = time.time()
        set_vdb_frame(smoke_obj, vdb_dir, pattern, fi)
        sc.frame_set(fi)
        sc.render.filepath = str(out_dir / f"frame_{fi-1:04d}.png")
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0; times.append(dt)
        eta = (sum(times)/len(times)) * (n_frames - fi)
        print(f"  frame {fi:04d}/{n_frames}  {dt:.1f}s  ETA {eta/60:.1f}min")

    print(f"\nDone. {sum(times)/60:.1f}min | {sum(times)/n_frames:.1f}s/frame avg")
    print(f"Encode: ffmpeg -framerate {args.fps} -i {out_dir}/frame_%04d.png "
          f"-c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p output.mp4")


if __name__ == "__main__":
    main()
