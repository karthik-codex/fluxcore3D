// thermal_kernels.cu
//
// Fused CUDA kernels for thermal solve pipeline.
//
// Kernels:
//   thermal_rhs_kernel         : Fuses RHS construction + theta0 init in one pass.
//   thermal_reconstruct_kernel : Fuses T_new = T0 + theta + all Neumann face BCs.
//   convergence_metrics_kernel : Parallel reduction for dT_solid_max, dT_fluid_max
//                                without CPU sync (async, write to device scalars).
//   physics_check_kernel       : Async finite+range validity, no stall unless needed.
//
// Memory layout: C-contiguous (nx, ny, nz).
//   linear index: idx = i*(ny*nz) + j*nz + k

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 256

// ─────────────────────────────────────────────────────────────────────────────
// thermal_rhs_kernel
//
// Single pass that produces both:
//   rhs[idx]    = mask[idx] * ((T_old[idx] + dt*src[idx]) - LT0[idx])
//   theta0[idx] = mask[idx] * (T_old[idx] - T0[idx])
//
// LT0 is the output of mop_fused(T0,...) = T0 + dt*L(T0), passed in directly.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void thermal_rhs_kernel(
    const float* __restrict__ T_old,
    const float* __restrict__ T0,
    const float* __restrict__ src_step,
    const float* __restrict__ LT0,
    const float* __restrict__ mask_f,
          float* __restrict__ rhs,
          float* __restrict__ theta0,
    float dt, int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float m  = mask_f[idx];
    const float to = T_old[idx];
    const float t0 = T0[idx];

    rhs[idx]    = m * ((to + dt * src_step[idx]) - LT0[idx]);
    theta0[idx] = m * (to - t0);
}

// ─────────────────────────────────────────────────────────────────────────────
// thermal_reconstruct_kernel
//
// Fuses T_new = T0 + theta  +  all Neumann BC face overwrites.
//
// Face priority: if a cell sits on a Neumann boundary it copies from its
// interior neighbor (same logic as _apply_temperature_bcs_inplace).
// Interior cells: val = T0[idx] + theta[idx].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void thermal_reconstruct_kernel(
    const float* __restrict__ T0,
    const float* __restrict__ theta,
          float* __restrict__ T_new,
    int nx, int ny, int nz,
    int ax, int out_idx
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (idx >= total) return;

    const int nynz = ny * nz;
    const int i = idx / nynz;
    const int j = (idx % nynz) / nz;
    const int k = idx % nz;

    float val = T0[idx] + theta[idx];
    bool  is_neumann = false;
    int   src_idx    = idx;

    // ── Check each face in priority order ─────────────────────────────────
    // High faces: always Neumann
    if (i == nx - 1) {
        is_neumann = true; src_idx = (i-1)*nynz + j*nz + k;
    } else if (i == 0 && ax != 0) {
        is_neumann = true; src_idx = 1*nynz + j*nz + k;
    } else if (i == 0 && ax == 0 && out_idx == 0) {
        is_neumann = true; src_idx = 1*nynz + j*nz + k;
    }

    if (!is_neumann) {
        if (j == ny - 1) {
            is_neumann = true; src_idx = i*nynz + (j-1)*nz + k;
        } else if (j == 0 && ax != 1) {
            is_neumann = true; src_idx = i*nynz + 1*nz + k;
        } else if (j == 0 && ax == 1 && out_idx == 0) {
            is_neumann = true; src_idx = i*nynz + 1*nz + k;
        }
    }

    if (!is_neumann) {
        if (k == nz - 1) {
            is_neumann = true; src_idx = i*nynz + j*nz + (k-1);
        } else if (k == 0 && ax != 2) {
            is_neumann = true; src_idx = i*nynz + j*nz + 1;
        } else if (k == 0 && ax == 2 && out_idx == 0) {
            is_neumann = true; src_idx = i*nynz + j*nz + 1;
        }
    }

    if (is_neumann) val = T0[src_idx] + theta[src_idx];

    T_new[idx] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// convergence_metrics_kernel + reduce_max_kernel
//
// Two-pass async parallel reduction: no CPU sync needed until result is read.
// solid_mask: uint8 (1=solid, 0=fluid)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void convergence_metrics_kernel(
    const float*   __restrict__ T_new,
    const float*   __restrict__ T_old,
    const uint8_t* __restrict__ solid_mask,
          float*   __restrict__ partial_solid,
          float*   __restrict__ partial_fluid,
    int n
) {
    __shared__ float sh_solid[BLOCK_SIZE];
    __shared__ float sh_fluid[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float vs = 0.0f, vf = 0.0f;
    if (idx < n) {
        float dt = fabsf(T_new[idx] - T_old[idx]);
        if (solid_mask[idx]) vs = dt; else vf = dt;
    }
    sh_solid[tid] = vs; sh_fluid[tid] = vf;
    __syncthreads();

    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_solid[tid] = fmaxf(sh_solid[tid], sh_solid[tid+s]);
            sh_fluid[tid] = fmaxf(sh_fluid[tid], sh_fluid[tid+s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_solid[blockIdx.x] = sh_solid[0];
        partial_fluid[blockIdx.x] = sh_fluid[0];
    }
}

__global__ void reduce_max_kernel(
    const float* __restrict__ partials,
          float* __restrict__ result,
    int n
) {
    __shared__ float sh[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    sh[tid] = (idx < n) ? partials[idx] : 0.0f;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid+s]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)result, __float_as_int(sh[0]));
}

// ─────────────────────────────────────────────────────────────────────────────
// physics_check_kernel
//
// Sets invalid[0]=1 if any cell is non-finite or out of [T_min, T_max].
// Does NOT stall CPU — caller reads flag only when truly needed.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void physics_check_kernel(
    const float* __restrict__ T,
          int*   __restrict__ invalid,
    float T_min, float T_max, int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = T[idx];
    if (!isfinite(v) || v < T_min || v > T_max) atomicOr(invalid, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// C++ wrappers (also include mg_kernels definitions here so one .cu = one ext)
// ─────────────────────────────────────────────────────────────────────────────

// ── Forward declarations for mop_fused / jacobi_smooth (from mg_kernels.cu)
// These are compiled together in the same extension — see mg_cuda_ext.py.

void thermal_rhs_launch(
    torch::Tensor T_old, torch::Tensor T0,
    torch::Tensor src_step, torch::Tensor LT0,
    torch::Tensor mask_float,
    torch::Tensor rhs, torch::Tensor theta0,
    double dt
) {
    const int n = T_old.numel();
    thermal_rhs_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        T_old.data_ptr<float>(), T0.data_ptr<float>(),
        src_step.data_ptr<float>(), LT0.data_ptr<float>(),
        mask_float.data_ptr<float>(),
        rhs.data_ptr<float>(), theta0.data_ptr<float>(),
        (float)dt, n
    );
}

torch::Tensor thermal_reconstruct_launch(
    torch::Tensor T0, torch::Tensor theta,
    int64_t nx, int64_t ny, int64_t nz,
    int64_t ax, int64_t out_idx
) {
    auto T_new = torch::empty_like(T0);
    const int total = nx*ny*nz;
    thermal_reconstruct_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        T0.data_ptr<float>(), theta.data_ptr<float>(), T_new.data_ptr<float>(),
        (int)nx, (int)ny, (int)nz, (int)ax, (int)out_idx
    );
    return T_new;
}

std::vector<torch::Tensor> convergence_metrics_launch(
    torch::Tensor T_new, torch::Tensor T_old, torch::Tensor solid_u8
) {
    const int n = T_new.numel();
    const int blocks = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(T_new.device());
    auto ps = torch::zeros({blocks}, opts);
    auto pf = torch::zeros({blocks}, opts);
    convergence_metrics_kernel<<<blocks, BLOCK_SIZE>>>(
        T_new.data_ptr<float>(), T_old.data_ptr<float>(),
        solid_u8.data_ptr<uint8_t>(), ps.data_ptr<float>(), pf.data_ptr<float>(), n
    );
    auto rs = torch::zeros({1}, opts);
    auto rf = torch::zeros({1}, opts);
    const int b2 = (blocks+BLOCK_SIZE-1)/BLOCK_SIZE;
    reduce_max_kernel<<<b2, BLOCK_SIZE>>>(ps.data_ptr<float>(), rs.data_ptr<float>(), blocks);
    reduce_max_kernel<<<b2, BLOCK_SIZE>>>(pf.data_ptr<float>(), rf.data_ptr<float>(), blocks);
    return {rs, rf};
}

torch::Tensor physics_check_launch(torch::Tensor T, double T_min, double T_max) {
    const int n = T.numel();
    auto flag = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(T.device()));
    physics_check_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        T.data_ptr<float>(), flag.data_ptr<int>(), (float)T_min, (float)T_max, n
    );
    return flag;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("thermal_rhs",         &thermal_rhs_launch,          "Fused RHS+theta0");
    m.def("thermal_reconstruct", &thermal_reconstruct_launch,   "Fused T_new + Neumann BCs");
    m.def("convergence_metrics", &convergence_metrics_launch,   "Async dT_solid/fluid max");
    m.def("physics_check",       &physics_check_launch,         "Async finite+range check");
}
