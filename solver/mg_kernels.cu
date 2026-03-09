// mg_kernels.cu
//
// Fused CUDA kernels for N-level geometric multigrid thermal solver.
//
// Kernels:
//   mop_fused_kernel  : Computes M(T) = T + dt * L(T) in a single pass.
//                       Fuses _fill_nb_into + _Mop (was ~20 kernel launches).
//                       Each thread: reads T at 7 stencil points via index
//                       arithmetic, reads 7 coefficient fields, writes Lo.
//
//   jacobi_smooth_kernel : x += inv_diagM * (b - Lo).
//                          Called after mop_fused_kernel.
//
// Memory layout: C-contiguous (nx, ny, nz), row-major.
//   linear index: idx = i*(ny*nz) + j*nz + k
//
// Boundary ghosting (Neumann, matches _fill_nb_into exactly):
//   High face of every axis: ghost = adjacent interior cell (Neumann).
//   Low  face, flow axis, outlet at 0:   Neumann (ghost = cell 1).
//   Low  face, flow axis, inlet  at 0:   copy-self (ghost = cell 0).
//   Low  face, transverse axis:          Neumann (ghost = cell 1).
//
// Compile via torch.utils.cpp_extension (see mg_cuda_ext.py).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Neumann ghost index for the LOW face of a given axis.
//   ax       : flow axis (0,1,2)
//   out_idx  : outlet face index on the flow axis
//   this_ax  : which axis we are computing the ghost for
//   n        : extent of this_ax
// Returns the source index (in this_ax coordinate) for the ghost at position 0.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ int low_ghost(int ax, int out_idx, int this_ax, int n) {
    // outlet Neumann: flow axis, outlet at low face
    if (ax == this_ax && out_idx == 0) return 1;
    // transverse Neumann
    if (ax != this_ax)                 return 1;
    // flow axis, inlet at low face → copy-self
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// mop_fused_kernel
//
// Lo[idx] = T[idx] + dt * L(T)[idx]
// L(T)[idx] = diagL[idx]*T[idx]
//           - aE[idx]*T[iE,j,k] - aW[idx]*T[iW,j,k]
//           - aN[idx]*T[i,jN,k] - aS[idx]*T[i,jS,k]
//           - aU[idx]*T[i,j,kU] - aD[idx]*T[i,j,kD]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mop_fused_kernel(
    const float* __restrict__ T,
    const float* __restrict__ aE,
    const float* __restrict__ aW,
    const float* __restrict__ aN,
    const float* __restrict__ aS,
    const float* __restrict__ aU,
    const float* __restrict__ aD,
    const float* __restrict__ diagL,
          float* __restrict__ Lo,
    float dt,
    int nx, int ny, int nz,
    int ax, int out_idx        // flow axis and outlet face index
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (idx >= total) return;

    const int nynz = ny * nz;
    const int i = idx / nynz;
    const int j = (idx % nynz) / nz;
    const int k = idx % nz;

    // ── X neighbors ────────────────────────────────────────────────────────
    // high-x (i == nx-1): Neumann → mirror to i-1
    const int iE = (i < nx - 1) ? i + 1 : i - 1;
    // low-x (i == 0): ghost determined by axis + outlet position
    const int iW = (i > 0) ? i - 1 : low_ghost(ax, out_idx, 0, nx);

    // ── Y neighbors ────────────────────────────────────────────────────────
    const int jN = (j < ny - 1) ? j + 1 : j - 1;
    const int jS = (j > 0) ? j - 1 : low_ghost(ax, out_idx, 1, ny);

    // ── Z neighbors ────────────────────────────────────────────────────────
    const int kU = (k < nz - 1) ? k + 1 : k - 1;
    const int kD = (k > 0) ? k - 1 : low_ghost(ax, out_idx, 2, nz);

    const float TC = T[idx];
    const float TE = T[iE * nynz + j * nz + k];
    const float TW = T[iW * nynz + j * nz + k];
    const float TN = T[i  * nynz + jN * nz + k];
    const float TS = T[i  * nynz + jS * nz + k];
    const float TU = T[i  * nynz + j  * nz + kU];
    const float TD = T[i  * nynz + j  * nz + kD];

    const float LT = diagL[idx] * TC
                   - aE[idx] * TE - aW[idx] * TW
                   - aN[idx] * TN - aS[idx] * TS
                   - aU[idx] * TU - aD[idx] * TD;

    Lo[idx] = TC + dt * LT;
}

// ─────────────────────────────────────────────────────────────────────────────
// jacobi_smooth_kernel
//
// x[idx] += inv_diagM[idx] * (b[idx] - Lo[idx])
// inv_diagM is pre-zeroed at Dirichlet nodes so no masking needed.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void jacobi_smooth_kernel(
          float* __restrict__ x,
    const float* __restrict__ b,
    const float* __restrict__ Lo,
    const float* __restrict__ inv_diagM,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] += inv_diagM[idx] * (b[idx] - Lo[idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// C++ wrapper functions called from Python via torch.utils.cpp_extension.
// All tensors must be CUDA, float32, contiguous.
// ─────────────────────────────────────────────────────────────────────────────

#define BLOCK_SIZE 256

torch::Tensor mop_fused(
    torch::Tensor T,
    torch::Tensor aE,
    torch::Tensor aW,
    torch::Tensor aN_t,    // aN — can't use aN as name (conflicts on some compilers)
    torch::Tensor aS,
    torch::Tensor aU,
    torch::Tensor aD,
    torch::Tensor diagL,
    torch::Tensor Lo,      // output buffer, pre-allocated, same shape as T
    double dt,
    int64_t ax,
    int64_t out_idx
) {
    TORCH_CHECK(T.is_cuda(),       "T must be a CUDA tensor");
    TORCH_CHECK(T.is_contiguous(), "T must be contiguous");
    TORCH_CHECK(T.scalar_type() == torch::kFloat32, "T must be float32");

    const int nx = T.size(0);
    const int ny = T.size(1);
    const int nz = T.size(2);
    const int total = nx * ny * nz;

    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mop_fused_kernel<<<blocks, BLOCK_SIZE>>>(
        T.data_ptr<float>(),
        aE.data_ptr<float>(),
        aW.data_ptr<float>(),
        aN_t.data_ptr<float>(),
        aS.data_ptr<float>(),
        aU.data_ptr<float>(),
        aD.data_ptr<float>(),
        diagL.data_ptr<float>(),
        Lo.data_ptr<float>(),
        static_cast<float>(dt),
        nx, ny, nz,
        static_cast<int>(ax),
        static_cast<int>(out_idx)
    );

    return Lo;
}

void jacobi_smooth(
    torch::Tensor x,
    torch::Tensor b,
    torch::Tensor Lo,
    torch::Tensor inv_diagM
) {
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    const int n = x.numel();
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    jacobi_smooth_kernel<<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        b.data_ptr<float>(),
        Lo.data_ptr<float>(),
        inv_diagM.data_ptr<float>(),
        n
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 module registration
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mop_fused",      &mop_fused,      "Fused MG stencil operator M(T) = T + dt*L(T)");
    m.def("jacobi_smooth",  &jacobi_smooth,  "Fused Jacobi smooth step: x += inv_diagM*(b-Lo)");
}
