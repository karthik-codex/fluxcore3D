# mg_cuda_ext.py
"""
Loads the mg_kernels.cu CUDA extension via torch.utils.cpp_extension.load().

JIT-compiles on first import and caches the result.  Subsequent imports use
the cached .pyd/.so without recompilation.

Requirements (Windows):
  - CUDA Toolkit installed (nvcc on PATH, or set CUDA_HOME)
  - MSVC (Visual Studio Build Tools) — cl.exe on PATH
  - PyTorch with CUDA support (torch.cuda.is_available() == True)

Usage:
    from mg_cuda_ext import MG_CUDA, MG_CUDA_AVAILABLE
    if MG_CUDA_AVAILABLE:
        MG_CUDA.mop_fused(T, aE, aW, aN, aS, aU, aD, diagL, Lo, dt, ax, out_idx)
        MG_CUDA.jacobi_smooth(x, b, Lo, inv_diagM)
"""

import os
import torch
from torch.utils.cpp_extension import load

MG_CUDA           = None
MG_CUDA_AVAILABLE = False

def _load_extension() -> bool:
    global MG_CUDA, MG_CUDA_AVAILABLE

    if not torch.cuda.is_available():
        print("[MG_CUDA] CUDA not available — using PyTorch fallback.")
        return False

    _dir = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.join(_dir, "mg_kernels.cu")

    if not os.path.isfile(_src):
        print(f"[MG_CUDA] mg_kernels.cu not found at {_src} — using PyTorch fallback.")
        return False

    try:
        print("[MG_CUDA] Compiling CUDA extension (first run only, ~30s) ...")
        MG_CUDA = load(
            name    = "mg_kernels",
            sources = [_src],
            # Windows needs /O2 and no -ffast-math
            extra_cuda_cflags = [
                "-O3",
                "--use_fast_math",
                "-DNDEBUG",
                "-allow-unsupported-compiler",
            ],
            extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"],
            verbose = False,
        )
        MG_CUDA_AVAILABLE = True
        print("[MG_CUDA] Extension loaded successfully.")
        return True
    except Exception as e:
        print(f"[MG_CUDA] Compilation failed: {e}\nFalling back to PyTorch.")
        MG_CUDA           = None
        MG_CUDA_AVAILABLE = False
        return False


_load_extension()
