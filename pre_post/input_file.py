import os, sys, time
import hashlib
from datetime import datetime
import os, sys, time
import hashlib
from datetime import datetime
from CAD_engine.config import (
    HeatSinkConfig, BasePlateConfig, FinGeometryConfig,
    ArrayConfig, GridConfig, FinProfile, ArrayType,
)
from CAD_engine.assembly  import HeatSinkSDF
from CAD_engine.export    import to_lbm_dict, export_hdf5, export_npz, to_numpy
from CAD_engine.visualize import (plot_top_view, plot_side_view, plot_sdf_contour,
                         plot_overview, plot_3d_isosurface)

# PyVista 3-D viewer (optional — requires pip install pyvista)
try:
    from CAD_engine.pyvista_viz import (
        HeatSinkViewer,
        show_design, quick_slices, quick_dashboard,
        export_stl, export_vtk,
        compare_resolutions,
    )
    import matplotlib
    matplotlib.use("Agg")  
    import matplotlib.pyplot as plt
    _HAS_PYVISTA_VIZ = True
except ImportError:
    _HAS_PYVISTA_VIZ = False
from pre_post.config_io import load_config

from pre_post.sdf_tracing import build_phi_from_stl_mm, build_ibb_from_sdf_gpu, build_phi_from_voxel_mask_mm
from pre_post.gpu_voxelizer import patch_stl_voxelizer 
from pre_post.solid_assembly import SolidAssembly
try:
    from solver.nlevel_mg_cuda import NLevelGeometricMGSolver
except ImportError:
    from solver.nlevel_mg_jacobi import NLevelGeometricMGSolver
from solver.lbm_cuda_ext import LBMCUDAUpgrade   
from solver.flow_solver_upgrade import FlowSolverUpgrade
from solver.thermal_bc import SurfaceFluxBC, VolumetricHeatBC


#os.add_dll_directory(r"C:\Users\Karthik\Documents\GPU_SPH_BinderJet\AMGX\build\Release")

current_datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
Current_Directory = os.path.dirname(os.path.realpath('__file__'))

config = load_config("config/model_config.yaml")         # or config.json 
print("Loaded configuration from config/model_config.yaml")
ID = config.get("id_output", "")
PART_NAME = os.path.basename(config['part_STL']).split(".")[0] if 'part_STL' in config else "implicit_mpm_sim"
MODEL_RESULTS = f"results/{PART_NAME}_sim{ID}.npz" if PART_NAME else "results/implicit_mpm_sim.npz"
NODE_INPUT = f"geometries/pointcloud/{PART_NAME}_{config['nodes_per_layer']}NPL_{config['layer_height']}LH_nodes.npz"  # contains 'nodes_part', and your part_config blob if used
STL_PATH = f"geometries/stl/{PART_NAME}.stl" if PART_NAME else "geometries/stl/implicit_mpm_sim.stl"

log_file = f'results/{PART_NAME}_sim{ID}_{current_datetime}.log'

# Override the built-in print function with the custom one
if plot_config := config.get("log_output", True):
    import builtins
    original_print = builtins.print
    log_path = log_file  # ensure this is defined earlier
    def custom_print(*args, **kwargs):
        # Print to the originally intended destination (default: stdout)
        original_print(*args, **kwargs)
        log_kwargs = {k: v for k, v in kwargs.items() if k != "file"}
        try:
            with open(log_path, "a", encoding="utf-8", newline="") as f:
                original_print(*args, file=f, **log_kwargs)
        except Exception as e:
            # Don't crash your run just because logging failed
            original_print(f"[log-error] {e}", file=sys.stderr)

    builtins.print = custom_print

print("\n")
print("----------------------------------------------------------------------------------------------------")
print("----------------- GPU-NATIVE LATTICE BASED CONJUGATE HEAT TRANSFER SIMULATION TOOL -----------------")
print("--------------------------VERSION: 1.0 ----------- LAST UPDATE: 02/27/2026--------------------------")
print("-------------DEVELOPER: Karthik Rajan Venkatesan (Digital Manufacturing Specialist) ----------------")
print("--------------------------COPYRIGHT: (c) 2026 Eaton. All rights reserved. --------------------------")
print("----------------------------------------------------------------------------------------------------\n")

AMGX_DIR = os.path.join(Current_Directory, "AMGX", "build", "Release")
AMGX_CONFIG = os.path.join(Current_Directory, "config", "amgx_config_bicgstab_amg.json")

amgx_env = {
    "AMGX_DIR": AMGX_DIR,
    "CUDA_DIR": r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
}


cfg = HeatSinkConfig(
    base  = BasePlateConfig(
        length_x  = 0.030,   # 60 mm flow direction
        width_y   = 0.020,   # 40 mm lateral
        thickness = 0.003,   # 3 mm base plate
    ),
    fin   = FinGeometryConfig(
        profile       = FinProfile.NACA,
        naca_digits   = "0018",
        naca_chord    = 0.004,    # 9 mm chord
        naca_aoa_deg  = 0.0,      # aligned with flow
        height        = 0.005,    # 20 mm tall fin
        tip_width_ratio = 1.0,    # straight walls
        tip_radius    = 0.0002,   # 0.4 mm tip rounding
    ),
    array = ArrayConfig(
        array_type       = ArrayType.STAGGERED,  # staggered for better performance
        pitch_x          = 0.004,
        pitch_y          = 0.004,
        edge_margin_factor = 1.0,
        stagger_fraction = 0.5,
    ),
    grid  = GridConfig(target_dx=0.00005, device="cuda",
    ),
)

# hs     = HeatSinkSDF(cfg)
# result = hs.build()          # → dict with sdf, phase, bc_tags, coords

# viewer = HeatSinkViewer(result, cfg)
# #viewer.show_all()            # 4-panel dashboard

# #viewer.export_solid_stl("heatsink_fin.stl")
# show_design(result, cfg)

# Initialize node sampling
if config["initialize"]:
    if not os.path.exists(NODE_INPUT):
        None
        if config["model_preview"]:
            None        
    else:
        current_time = datetime.now().strftime("%I:%M:%S %p")
        print("[",current_time,"]...Model input file found. Skipping node sampling.")

if config["run_model"]:

    if not os.path.exists(NODE_INPUT):
        current_time = datetime.now().strftime("%I:%M:%S %p")
        print("[",current_time,"]...Model input file not found. Running node sampling first.")
        None

    # --- simulator ---
    # sim = TLSPHShrinkageSimulator(NODE_INPUT, MODEL_RESULTS, config=config, device='cuda', 
    #                               solvers=solvers, internal=internal_force, contact=contact_force) #, internal=internal_force)