#!/usr/bin/env python3
"""
cht_run.py  --  CLI batch runner for CHT Workbench

Usage
-----
Single case:
    python cht_run.py case.chtmdl

Override project name (output .npz filename):
    python cht_run.py case.chtmdl --project my_run

Batch - multiple cases sequentially:
    python cht_run.py base.chtmdl high_flow.chtmdl fine_mesh.chtmdl

Per-case project names:
    python cht_run.py a.chtmdl b.chtmdl --projects run_A run_B

Auto-numbered outputs (stem_001, stem_002 ...):
    python cht_run.py *.chtmdl --auto-name

Change output directory:
    python cht_run.py *.chtmdl --workdir /results/batch_01

Priority for project name:  CLI flag > .chtmdl field > .chtmdl file stem
No display / GUI required.  CUDA must be available.
"""
import sys, os, argparse, time
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from UI_components.cht_models      import load_model
from UI_components.cht_sim_imports import HAS_SIM, SIM_IMPORT_ERROR
from UI_components.cht_worker      import run_simulation_blocking


def _bar(pct, w=40):
    filled = int(w * pct / 100)
    print(f"\r  [{'#'*filled}{'.'*(w-filled)}] {pct:3d}%", end="", flush=True)


def _run_one(chtmdl_path, project_override=None):
    path = Path(chtmdl_path)
    if not path.exists():
        return False, f"File not found: {path}"

    print(f"\n{'='*62}")
    print(f"  Case    : {path.name}")

    state = load_model(str(path))
    if state is None:
        return False, f"Could not parse {path}"

    params = state
    project_name = (
        project_override
        or params.get("project_name", "").strip()
        or path.stem
    )
    params["project_name"] = project_name

    dom = params["domain"]
    dx  = params["dx_mm"]
    nx  = round(dom["Lx"]*1000/dx); ny=round(dom["Ly"]*1000/dx); nz=round(dom["Lz"]*1000/dx)

    print(f"  Project : {project_name}  ->  {project_name}.npz")
    print(f"  Domain  : {dom['Lx']}x{dom['Ly']}x{dom['Lz']} m  |  dx={dx} mm  |  grid {nx}x{ny}x{nz}")
    print(f"  Bodies  : {[b.name for b in params['bodies']]}")
    print(f"  Flow    : {params['flow_dir']}  @  {params['u_in']} m/s")
    print(f"  BC      : {params['bc_type']}")
    print(f"{'='*62}")

    last = [0]
    def on_pct(p):
        if p != last[0]: last[0]=p; _bar(p)
    def log(msg):
        print(f"\n  {msg}", end="", flush=True)

    t0 = time.time()
    try:
        out = run_simulation_blocking(params, log_fn=log, progress_fn=on_pct)
        print(f"\n\n  DONE  {time.time()-t0:.1f}s  ->  {out}\n")
        return True, out
    except Exception as exc:
        print(f"\n\n  FAILED  {time.time()-t0:.1f}s  :  {exc}\n")
        return False, str(exc)


def main():
    ap = argparse.ArgumentParser(
        description="Run CHT simulations from .chtmdl files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("cases",     nargs="+",          metavar="FILE.chtmdl")
    ap.add_argument("--project", default=None,        help="Project name for single case")
    ap.add_argument("--projects",nargs="+",           help="Project names for batch (one per file)")
    ap.add_argument("--auto-name",action="store_true",help="Auto-number: stem_001, stem_002 ...")
    ap.add_argument("--workdir", default=None,        help="Change to this directory before running")
    args = ap.parse_args()

    if not HAS_SIM:
        print(f"[ERROR] Simulation modules unavailable: {SIM_IMPORT_ERROR}")
        sys.exit(1)

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        os.chdir(args.workdir)
        print(f"[workdir] {os.getcwd()}")

    n = len(args.cases)
    if args.auto_name:
        names = [f"{Path(c).stem}_{i+1:03d}" for i,c in enumerate(args.cases)]
    elif args.projects:
        if len(args.projects) != n:
            print(f"[ERROR] --projects needs {n} name(s), got {len(args.projects)}")
            sys.exit(1)
        names = args.projects
    elif args.project:
        if n > 1:
            print("[WARN] --project applies the same name to all cases")
        names = [args.project] * n
    else:
        names = [None] * n

    results = []
    for i,(case,proj) in enumerate(zip(args.cases,names),1):
        print(f"\n[{i}/{n}]", end="")
        ok, msg = _run_one(case, project_override=proj)
        results.append((case, ok, msg))

    print(f"\n{'='*62}")
    print(f"  SUMMARY  ({n} case{'s' if n!=1 else ''})")
    print(f"{'='*62}")
    for case, ok, msg in results:
        icon = "OK" if ok else "!!"
        tail = msg if ok else f"FAILED: {str(msg)[:55]}"
        print(f"  [{icon}]  {Path(case).name:40s}  {tail}")
    passed = sum(1 for _,ok,_ in results if ok)
    print(f"{'='*62}")
    print(f"  {passed}/{n} passed")
    sys.exit(0 if passed==n else 1)


if __name__ == "__main__":
    main()
