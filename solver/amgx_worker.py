# amgx_worker.py — fully self-contained AMGX worker (no imports from your project)
# Loads CUDA + cuSPARSE + AMGX DLLs explicitly, then solves A x = b from SciPy .npz
#
# Usage:
#   python amgx_worker.py --cfg <cfg.json> --A <A.npz> --b <b.npy> --xout <x.npy> \
#                         [--amgx_dir "<...>\AMGX\build\Release"] \
#                         [--cuda_dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"]

import os
import sys
import argparse
import ctypes
import numpy as np
from scipy.sparse import load_npz, csr_matrix

def add_dir(p):
    if p and os.path.isdir(p):
        os.add_dll_directory(p)
        return True
    return False

def load_dll(path):
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(f"Failed to load DLL: {path}\n{e}")

def solve_once(cfg_path, A_path, b_path, xout_path, amgx_dir, cuda_dir):
    # 0) DLL search paths
    add_dir(amgx_dir)
    add_dir(cuda_dir)

    # ---- NEW: try MPI dir if present (msmpi.dll dependency) ----
    mpi_dir = r"C:\Program Files\Microsoft MPI\Bin"
    if os.path.isdir(mpi_dir):
        add_dir(mpi_dir)

    # 1) Force DLL load order (match probe)
    #    nvcuda is driver, cudart is runtime, then cusparse, then amgx core.
    #    Let Windows loader find nvcuda in System32; cudart/cusparse in cuda_dir.
    load_dll("nvcuda.dll")  # from System32
    load_dll(os.path.join(cuda_dir, "cudart64_12.dll"))
    load_dll(os.path.join(cuda_dir, "cusparse64_12.dll"))

    # ---- NEW: also load cublas + cusolver (your DLL lists them) ----
    load_dll(os.path.join(cuda_dir, "cublas64_12.dll"))
    load_dll(os.path.join(cuda_dir, "cusolver64_11.dll"))

    # 2) Load AMGX core + wrapper from amgx_dir
    amgx_core = load_dll(os.path.join(amgx_dir, "amgxsh.dll"))
    wrapper   = load_dll(os.path.join(amgx_dir, "amgx_wrapper.dll"))

    # 3) Bind C exports
    wrapper.amgx_global_init.restype     = ctypes.c_int
    wrapper.amgx_global_finalize.restype = ctypes.c_int
    wrapper.amgx_solve_csr.restype       = ctypes.c_int
    wrapper.amgx_solve_csr.argtypes = [
        ctypes.c_int,                           # n
        ctypes.c_int,                           # nnz
        ctypes.POINTER(ctypes.c_int),           # row_ptr
        ctypes.POINTER(ctypes.c_int),           # col_ind
        ctypes.POINTER(ctypes.c_double),        # vals
        ctypes.POINTER(ctypes.c_double),        # rhs
        ctypes.POINTER(ctypes.c_double),        # solution
        ctypes.c_char_p                         # config_file
    ]

    # 4) Initialize AMGX
    rc = wrapper.amgx_global_init()
    if rc != 0:
        raise RuntimeError(f"amgx_global_init() failed: {rc}")

    try:
        # 5) Load inputs
        K = load_npz(A_path).tocsr()
        if not isinstance(K, csr_matrix):
            K = csr_matrix(K)
        K.sort_indices()
        K.sum_duplicates()

        b = np.load(b_path).astype(np.float64).reshape(-1)
        n, m = K.shape
        if n != m:
            raise RuntimeError(f"K is not square: {n}x{m}")
        if b.shape[0] != n:
            raise RuntimeError(f"rhs length {b.shape[0]} != {n}")

        indptr  = np.ascontiguousarray(K.indptr.astype(np.int32,  copy=False))
        indices = np.ascontiguousarray(K.indices.astype(np.int32, copy=False))
        data    = np.ascontiguousarray(K.data.astype(np.float64,  copy=False))
        nnz = int(data.size)
        if int(indptr[-1]) != nnz:
            raise RuntimeError(f"CSR error: indptr[-1]={int(indptr[-1])} != nnz={nnz}")

        x = np.zeros(n, dtype=np.float64)

        # 6) Call AMGX
        row_ptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        col_ind_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        vals_ptr    = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rhs_ptr     = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        x_ptr       = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        cfg = os.fspath(cfg_path).encode("utf-8")

        print(f"[WORKER] n={n}, nnz={nnz}, col[min,max]=[{indices.min() if indices.size else 0},{indices.max() if indices.size else 0}]")
        ret = wrapper.amgx_solve_csr(
            int(n), int(nnz),
            row_ptr_ptr, col_ind_ptr, vals_ptr,
            rhs_ptr, x_ptr, cfg
        )
        if ret != 0:
            raise RuntimeError(f"amgx_solve_csr failed with code {ret}")

        # 7) Save
        np.save(xout_path, x)

    finally:
        wrapper.amgx_global_finalize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",   required=True)
    ap.add_argument("--A",     required=True)
    ap.add_argument("--b",     required=True)
    ap.add_argument("--xout",  required=True)
    ap.add_argument("--amgx_dir", default=r"C:\Users\Immortal Machine\Documents\GPU_SPH_BinderJet\AMGX\build\Release")
    ap.add_argument("--cuda_dir", default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    args = ap.parse_args()

    # Keep worker clean & predictable
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[WORKER] CWD={os.getcwd()}")
    print(f"[WORKER] CFG={args.cfg}")
    print(f"[WORKER] A={args.A}")
    print(f"[WORKER] b={args.b}")
    print(f"[WORKER] xout={args.xout}")
    print(f"[WORKER] AMGX_DIR={args.amgx_dir}")
    print(f"[WORKER] CUDA_DIR={args.cuda_dir}")

    solve_once(args.cfg, args.A, args.b, args.xout, args.amgx_dir, args.cuda_dir)

if __name__ == "__main__":
    main()
