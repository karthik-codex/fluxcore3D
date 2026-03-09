"""
cht_sim_imports.py
Centralised try/except block for all simulation backend imports.

Both cht_worker.py and cht_main_window.py import from here so the
try/except lives in exactly one place.

Exports
-------
HAS_SIM            : bool   — True when all sim modules loaded cleanly
SIM_IMPORT_ERROR   : Exception | None

If HAS_SIM is True the following names are also importable:
    SimConfig3D, FluidPropsSI, SolidPropsSI, STLVoxelizer, LBMCHT3D_Torch,
    NLevelGeometricMGSolver, patch_mg_pcg, LBMCUDAUpgrade, FlowSolverUpgrade,
    SurfaceFluxBC, VolumetricHeatBC,
    build_phi_from_stl_mm, build_ibb_from_sdf_gpu, build_phi_from_voxel_mask_mm,
    patch_stl_voxelizer, SolidAssembly
"""

import sys
import os

HAS_SIM          = False
SIM_IMPORT_ERROR = None

SimConfig3D = FluidPropsSI = SolidPropsSI = None
STLVoxelizer = LBMCHT3D_Torch = None
NLevelGeometricMGSolver = patch_mg_pcg = None
LBMCUDAUpgrade = FlowSolverUpgrade = None
SurfaceFluxBC = VolumetricHeatBC = None
build_phi_from_stl_mm = build_ibb_from_sdf_gpu = build_phi_from_voxel_mask_mm = None
patch_stl_voxelizer = unpatch_stl_voxelizer = SolidAssembly = None

try:
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    from LBM_CHT import (
        SimConfig3D, FluidPropsSI, SolidPropsSI,
        STLVoxelizer, LBMCHT3D_Torch,
    )
    try:
        from solver.nlevel_mg_cuda   import NLevelGeometricMGSolver
    except ImportError:
        from solver.nlevel_mg_jacobi import NLevelGeometricMGSolver

    from solver.mg_pcg              import patch_mg_pcg
    from solver.lbm_cuda_ext        import LBMCUDAUpgrade
    from solver.flow_solver_upgrade import FlowSolverUpgrade
    from solver.thermal_bc                 import SurfaceFluxBC, VolumetricHeatBC
    from pre_post.sdf_tracing       import (
        build_phi_from_stl_mm,
        build_ibb_from_sdf_gpu,
        build_phi_from_voxel_mask_mm,
    )
    from pre_post.gpu_voxelizer     import patch_stl_voxelizer, unpatch_stl_voxelizer
    from pre_post.solid_assembly    import SolidAssembly

    HAS_SIM = True

except Exception as _e:
    SIM_IMPORT_ERROR = _e
    print(f"[WARN] Simulation import failed — {type(_e).__name__}: {_e}")
