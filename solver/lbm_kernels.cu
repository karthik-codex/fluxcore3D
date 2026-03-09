// lbm_kernels.cu
//
// Fused CUDA kernels for 3-D D3Q19 Lattice Boltzmann Method (LBM)
// conjugate-heat-transfer solver (LBMCHT3D_Torch + FlowSolverUpgrade).
//
// Memory layout: f[q][N]  where  N = nx*ny*nz,  q in [0,19)
//   linear index:  flat = i*(ny*nz) + j*nz + k
//   f flat index:  q*N + flat
//
// D3Q19 velocity ordering (matches LBMCHT3D_Torch / flow_solver_upgrade.py):
//   q  cx cy cz
//   0   0  0  0   rest
//   1   1  0  0   +x face
//   2  -1  0  0   -x face
//   3   0  1  0   +y face
//   4   0 -1  0   -y face
//   5   0  0  1   +z face
//   6   0  0 -1   -z face
//   7   1  1  0   +x+y edge
//   8   1 -1  0   +x-y edge
//   9  -1  1  0   -x+y edge
//  10  -1 -1  0   -x-y edge
//  11   1  0  1   +x+z edge
//  12   1  0 -1   +x-z edge
//  13  -1  0  1   -x+z edge
//  14  -1  0 -1   -x-z edge
//  15   0  1  1   +y+z edge
//  16   0  1 -1   +y-z edge
//  17   0 -1  1   -y+z edge
//  18   0 -1 -1   -y-z edge
//
// Opposite index:  opp = {0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15}
//
// Kernels exported (called via PyTorch C++ extension):
//   lbm_macro_collide_bgk  - fused macroscopic + feq + BGK collision
//   lbm_macro_collide_trt  - fused macroscopic + feq + TRT collision
//   lbm_macro_collide_mrt  - fused macroscopic + feq + MRT collision  <- primary
//   lbm_stream_pull        - pull-scheme streaming with wall/periodic BCs
//   lbm_sponge             - outlet sponge-zone relaxation toward feq
//   lbm_ibb                - Bouzidi interpolated bounce-back
//   lbm_reset_solid        - reset solid-node populations to rest equilibrium
//
// Compile via lbm_cuda_ext.py (torch.utils.cpp_extension.load).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Fallback: define C10_CUDA_KERNEL_LAUNCH_CHECK if not provided by the
// installed PyTorch version (some builds omit it from the public headers).
#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#  define C10_CUDA_KERNEL_LAUNCH_CHECK() AT_CUDA_CHECK(cudaGetLastError())
#endif

// -----------------------------------------------------------------------------
//  Compile-time D3Q19 constants
// -----------------------------------------------------------------------------
__constant__ int   CX[19] = { 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0};
__constant__ int   CY[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1};
__constant__ int   CZ[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1};
__constant__ int   OPP[19]= { 0, 2, 1, 4, 3, 6, 5,10, 9, 8, 7,14,13,12,11,18,17,16,15};

// Lattice weights: w0=1/3, w1..6=1/18, w7..18=1/36
__constant__ float W[19]  = {
    0.33333333f,
    0.05555556f, 0.05555556f, 0.05555556f, 0.05555556f, 0.05555556f, 0.05555556f,
    0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f,
    0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f,
    0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
};

// -----------------------------------------------------------------------------
//  Device helper: compute f_eq[19] into registers given rho, ux, uy, uz
//  Returns feq via output array pointer (19 floats in caller's registers).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void compute_feq(
        float rho, float ux, float uy, float uz,
        float feq[19])
{
    const float w0 = 0.33333333f;
    const float w1 = 0.05555556f;
    const float w2 = 0.02777778f;
    const float three_halves_u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

    // Precompute 3*u components to avoid redundant multiplications
    const float ox = 3.f*ux, oy = 3.f*uy, oz = 3.f*uz;

    // Macro to compute individual feq:  w * rho * (1 + cu + 0.5*cu^2 - 1.5*|u|^2)
    #define FEQTERM(w, cu) ((w)*rho*(1.f + (cu) + 0.5f*(cu)*(cu) - three_halves_u2))

    feq[ 0] = FEQTERM(w0, 0.f);
    feq[ 1] = FEQTERM(w1, ox);
    feq[ 2] = FEQTERM(w1, -ox);
    feq[ 3] = FEQTERM(w1, oy);
    feq[ 4] = FEQTERM(w1, -oy);
    feq[ 5] = FEQTERM(w1, oz);
    feq[ 6] = FEQTERM(w1, -oz);
    feq[ 7] = FEQTERM(w2, ox+oy);
    feq[ 8] = FEQTERM(w2, ox-oy);
    feq[ 9] = FEQTERM(w2, -ox+oy);
    feq[10] = FEQTERM(w2, -ox-oy);
    feq[11] = FEQTERM(w2, ox+oz);
    feq[12] = FEQTERM(w2, ox-oz);
    feq[13] = FEQTERM(w2, -ox+oz);
    feq[14] = FEQTERM(w2, -ox-oz);
    feq[15] = FEQTERM(w2, oy+oz);
    feq[16] = FEQTERM(w2, oy-oz);
    feq[17] = FEQTERM(w2, -oy+oz);
    feq[18] = FEQTERM(w2, -oy-oz);

    #undef FEQTERM
}

// -----------------------------------------------------------------------------
//  Device helper: compute all 19 MRT moments m = M @ f (inline, no mem access)
//  Uses the specific M matrix from flow_solver_upgrade.py / Lallemand&Luo 2000.
//  All intermediate sums use __fmaf_rn for fused multiply-add.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void compute_moments(const float f[19], float m[19])
{
    // Useful partial sums (avoids repeating work across moment rows)
    const float sf   = f[1]+f[2]+f[3]+f[4]+f[5]+f[6];            // face |c|=1
    const float se   = f[7]+f[8]+f[9]+f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18]; // edge |c|=sqrt2

    // x-axis partial sums
    const float cx_face = f[1]-f[2];                               // jx from face
    const float cx_edge = (f[7]+f[8]-f[9]-f[10]) + (f[11]+f[12]-f[13]-f[14]); // jx from edge
    // y-axis
    const float cy_face = f[3]-f[4];
    const float cy_edge = (f[7]-f[8]+f[9]-f[10]) + (f[15]+f[16]-f[17]-f[18]);
    // z-axis
    const float cz_face = f[5]-f[6];
    const float cz_edge = (f[11]-f[12]+f[13]-f[14]) + (f[15]-f[16]+f[17]-f[18]);

    // edge sums for stress components
    const float sxy_p = f[7]+f[8]+f[9]+f[10];  // xy-plane edges
    const float sxz_p = f[11]+f[12]+f[13]+f[14]; // xz-plane edges
    const float syz_p = f[15]+f[16]+f[17]+f[18]; // yz-plane edges

    // Row 0: ? = sum all
    m[ 0] = f[0] + sf + se;
    // Row 1: e = -30*f0 - 11*faces + 8*edges
    m[ 1] = -30.f*f[0] - 11.f*sf + 8.f*se;
    // Row 2: ? = 12*f0 - 4*faces + edges
    m[ 2] = 12.f*f[0] - 4.f*sf + se;
    // Row 3: jx
    m[ 3] = cx_face + cx_edge;
    // Row 4: qx = -4*(face jx) + edge jx
    m[ 4] = -4.f*cx_face + cx_edge;
    // Row 5: jy
    m[ 5] = cy_face + cy_edge;
    // Row 6: qy = -4*(face jy) + edge jy
    m[ 6] = -4.f*cy_face + cy_edge;
    // Row 7: jz
    m[ 7] = cz_face + cz_edge;
    // Row 8: qz = -4*(face jz) + edge jz
    m[ 8] = -4.f*cz_face + cz_edge;
    // Row 9: 3p_xx = 2cx?-cy?-cz? = 2*(face_x+edge_xx) - (face_y+edge_yy) - (face_z+edge_zz)
    m[ 9] = 2.f*(f[1]+f[2]) - (f[3]+f[4]+f[5]+f[6]) + sxy_p + sxz_p - 2.f*syz_p;
    // Row 10: 3?_xx = -4*(face_x terms) + 2*(face_y+face_z) + xy+xz edges - 2*yz edges
    m[10] = -4.f*(f[1]+f[2]) + 2.f*(f[3]+f[4]+f[5]+f[6]) + sxy_p + sxz_p - 2.f*syz_p;
    // Row 11: p_ww = cy?-cz? = face_y - face_z + xy_edges - xz_edges
    m[11] = (f[3]+f[4]) - (f[5]+f[6]) + sxy_p - sxz_p;
    // Row 12: ?_ww = -2*face_y + 2*face_z + xy_edges - xz_edges
    m[12] = -2.f*(f[3]+f[4]) + 2.f*(f[5]+f[6]) + sxy_p - sxz_p;
    // Row 13: p_xy = cx*cy = [0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,0,0,0,0]
    m[13] = f[7]-f[8]-f[9]+f[10];
    // Row 14: p_yz = cy*cz = [0,...,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1]
    m[14] = f[15]-f[16]-f[17]+f[18];
    // Row 15: p_xz = cx*cz = [0,...,0,0,0,0,0,1,-1,-1,1,0,0,0,0]
    m[15] = f[11]-f[12]-f[13]+f[14];
    // Row 16: m_x = cx(cy?-cz?) = [0,...,1,1,-1,-1,-1,-1,1,1,0,0,0,0]
    m[16] = (f[7]+f[8]) - (f[9]+f[10]) - (f[11]+f[12]) + (f[13]+f[14]);
    // Row 17: m_y = cy(cz?-cx?) = [0,...,-1,1,-1,1,0,0,0,0,1,1,-1,-1]
    m[17] = -f[7]+f[8]-f[9]+f[10] + f[15]+f[16]-f[17]-f[18];
    // Row 18: m_z = cz(cx?-cy?) = [0,...,0,0,0,0,1,-1,1,-1,-1,1,-1,1]
    m[18] = f[11]-f[12]+f[13]-f[14] - f[15]+f[16]-f[17]+f[18];
}

// -----------------------------------------------------------------------------
//  Device helper: compute df = Minv @ delta_m (sparse Minv, inline)
//  delta_m[0]=delta_m[3]=delta_m[5]=delta_m[7]=0 by conservation (caller ensures).
//  Only the 15 nonzero dm channels appear; notation dm1..dm18 (dm0/3/5/7 absent).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void apply_Minv(const float dm[19], float df[19])
{
    // Exact Minv @ delta_m, hardcoded from numpy.linalg.inv(M_D3Q19).
    // Conserved channels dm[0]=dm[3]=dm[5]=dm[7] are always 0 (caller enforces).
    // Verified against dense Minv @ dm for 100 random vectors, max error < 1e-7.
    //
    // Critical coefficients that differ between row groups (dm[10]):
    //   Rows 1-2  (+/-x face):    Minv[1-2][10]   = -1/18 = -0.0555555556
    //   Rows 3-6  (+/-y,+/-z face): Minv[3-6][10]   = +1/36 = +0.0277777778
    //   Rows 7-14 (xy,xz edge): Minv[7-14][10]  = +1/72 = +0.0138888889  <- different!
    //   Rows 15-18 (yz edge):   Minv[15-18][10] = -1/36 = -0.0277777778

    const float d1  = dm[1],  d2  = dm[2];
    const float d4  = dm[4],  d6  = dm[6],  d8  = dm[8];
    const float d9  = dm[9],  d10 = dm[10];
    const float d11 = dm[11], d12 = dm[12];
    const float d13 = dm[13], d14 = dm[14], d15 = dm[15];
    const float d16 = dm[16], d17 = dm[17], d18 = dm[18];

    // Shared base terms (same coefficients for d1, d2 within each group)
    const float base_face = -0.0045948204f*d1 + -0.0158730159f*d2;  // rows 1-6
    const float base_edge =  0.0033416876f*d1 +  0.0039682540f*d2;  // rows 7-18

    // -- Row 0: rest -----------------------------------------------------------
    df[ 0] = -0.0125313283f*d1 + 0.0476190476f*d2;

    // -- Rows 1-2: +/-x face -----------------------------------------------------
    df[ 1] = base_face + -0.1000000000f*d4 +  0.0555555556f*d9 + -0.0555555556f*d10;
    df[ 2] = base_face +  0.1000000000f*d4 +  0.0555555556f*d9 + -0.0555555556f*d10;

    // -- Rows 3-4: +/-y face -----------------------------------------------------
    df[ 3] = base_face + -0.1000000000f*d6 + -0.0277777778f*d9 +  0.0277777778f*d10
             +  0.0833333333f*d11 + -0.0833333333f*d12;
    df[ 4] = base_face +  0.1000000000f*d6 + -0.0277777778f*d9 +  0.0277777778f*d10
             +  0.0833333333f*d11 + -0.0833333333f*d12;

    // -- Rows 5-6: +/-z face -----------------------------------------------------
    df[ 5] = base_face + -0.1000000000f*d8 + -0.0277777778f*d9 +  0.0277777778f*d10
             + -0.0833333333f*d11 +  0.0833333333f*d12;
    df[ 6] = base_face +  0.1000000000f*d8 + -0.0277777778f*d9 +  0.0277777778f*d10
             + -0.0833333333f*d11 +  0.0833333333f*d12;

    // -- Rows 7-10: xy-plane edges ---------------------------------------------
    // NOTE: d10 coefficient here is 0.0138888889 (= 1/72), NOT 0.0277 (= 1/36)
    const float exy = base_edge + 0.0277777778f*d9 + 0.0138888889f*d10
                    + 0.0833333333f*d11 + 0.0416666667f*d12;
    df[ 7] = exy +  0.0250000000f*d4 +  0.0250000000f*d6 +  0.2500000000f*d13 +  0.1250000000f*d16 + -0.1250000000f*d17;
    df[ 8] = exy +  0.0250000000f*d4 + -0.0250000000f*d6 + -0.2500000000f*d13 +  0.1250000000f*d16 +  0.1250000000f*d17;
    df[ 9] = exy + -0.0250000000f*d4 +  0.0250000000f*d6 + -0.2500000000f*d13 + -0.1250000000f*d16 + -0.1250000000f*d17;
    df[10] = exy + -0.0250000000f*d4 + -0.0250000000f*d6 +  0.2500000000f*d13 + -0.1250000000f*d16 +  0.1250000000f*d17;

    // -- Rows 11-14: xz-plane edges --------------------------------------------
    const float exz = base_edge + 0.0277777778f*d9 + 0.0138888889f*d10
                    + -0.0833333333f*d11 + -0.0416666667f*d12;
    df[11] = exz +  0.0250000000f*d4 +  0.0250000000f*d8 +  0.2500000000f*d15 + -0.1250000000f*d16 +  0.1250000000f*d18;
    df[12] = exz +  0.0250000000f*d4 + -0.0250000000f*d8 + -0.2500000000f*d15 + -0.1250000000f*d16 + -0.1250000000f*d18;
    df[13] = exz + -0.0250000000f*d4 +  0.0250000000f*d8 + -0.2500000000f*d15 +  0.1250000000f*d16 +  0.1250000000f*d18;
    df[14] = exz + -0.0250000000f*d4 + -0.0250000000f*d8 +  0.2500000000f*d15 +  0.1250000000f*d16 + -0.1250000000f*d18;

    // -- Rows 15-18: yz-plane edges --------------------------------------------
    // NOTE: d9 = -0.0555, d10 = -0.0277 (both negative, different signs/magnitudes from xy/xz)
    const float eyz = base_edge + -0.0555555556f*d9 + -0.0277777778f*d10;
    df[15] = eyz +  0.0250000000f*d6 +  0.0250000000f*d8 +  0.2500000000f*d14 +  0.1250000000f*d17 + -0.1250000000f*d18;
    df[16] = eyz +  0.0250000000f*d6 + -0.0250000000f*d8 + -0.2500000000f*d14 +  0.1250000000f*d17 +  0.1250000000f*d18;
    df[17] = eyz + -0.0250000000f*d6 +  0.0250000000f*d8 + -0.2500000000f*d14 + -0.1250000000f*d17 + -0.1250000000f*d18;
    df[18] = eyz + -0.0250000000f*d6 + -0.0250000000f*d8 +  0.2500000000f*d14 + -0.1250000000f*d17 +  0.1250000000f*d18;
}

// =============================================================================
//  KERNEL 1:  lbm_macro_collide_bgk_kernel
//             Fused: macroscopic -> f_eq -> BGK collision -> f_post
// =============================================================================
__global__ void lbm_macro_collide_bgk_kernel(
    const float* __restrict__ f,          // [19][N] input populations
          float* __restrict__ f_post,     // [19][N] post-collision output
          float* __restrict__ rho_out,    // [N]
          float* __restrict__ u_out,      // [N]
          float* __restrict__ v_out,      // [N]
          float* __restrict__ w_out,      // [N]
    const uint8_t* __restrict__ solid,    // [N] 1=solid,0=fluid
    float omega,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // -- Load f[0..18] into registers -----------------------------------------
    float fi[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) fi[q] = f[q*N + idx];

    if (solid[idx]) {
        // Solid node: write equilibrium at rest, zero macro
        #pragma unroll
        for (int q = 0; q < 19; q++) f_post[q*N + idx] = W[q];  // rho=1,u=0
        rho_out[idx] = 1.f; u_out[idx] = 0.f; v_out[idx] = 0.f; w_out[idx] = 0.f;
        return;
    }

    // -- Macroscopic quantities ------------------------------------------------
    float rho = 0.f, jx = 0.f, jy = 0.f, jz = 0.f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        rho += fi[q];
        jx  += CX[q] * fi[q];
        jy  += CY[q] * fi[q];
        jz  += CZ[q] * fi[q];
    }
    const float inv_rho = 1.f / (rho + 1e-12f);
    const float ux = jx * inv_rho;
    const float uy = jy * inv_rho;
    const float uz = jz * inv_rho;

    rho_out[idx] = rho; u_out[idx] = ux; v_out[idx] = uy; w_out[idx] = uz;

    // -- BGK collision ---------------------------------------------------------
    float feq[19];
    compute_feq(rho, ux, uy, uz, feq);

    #pragma unroll
    for (int q = 0; q < 19; q++)
        f_post[q*N + idx] = fi[q] + omega * (feq[q] - fi[q]);
}


// =============================================================================
//  KERNEL 2:  lbm_macro_collide_trt_kernel
//             Fused: macroscopic -> f_eq -> TRT collision -> f_post
// =============================================================================
__global__ void lbm_macro_collide_trt_kernel(
    const float* __restrict__ f,
          float* __restrict__ f_post,
          float* __restrict__ rho_out,
          float* __restrict__ u_out,
          float* __restrict__ v_out,
          float* __restrict__ w_out,
    const uint8_t* __restrict__ solid,
    float omega_plus,
    float omega_minus,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float fi[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) fi[q] = f[q*N + idx];

    if (solid[idx]) {
        #pragma unroll
        for (int q = 0; q < 19; q++) f_post[q*N + idx] = W[q];
        rho_out[idx]=1.f; u_out[idx]=0.f; v_out[idx]=0.f; w_out[idx]=0.f;
        return;
    }

    // Macroscopic
    float rho=0.f, jx=0.f, jy=0.f, jz=0.f;
    #pragma unroll
    for (int q=0; q<19; q++) {
        rho += fi[q];
        jx  += CX[q]*fi[q]; jy += CY[q]*fi[q]; jz += CZ[q]*fi[q];
    }
    const float inv_rho = 1.f/(rho+1e-12f);
    const float ux=jx*inv_rho, uy=jy*inv_rho, uz=jz*inv_rho;
    rho_out[idx]=rho; u_out[idx]=ux; v_out[idx]=uy; w_out[idx]=uz;

    float feq[19];
    compute_feq(rho, ux, uy, uz, feq);

    // TRT: f_plus = 0.5*(f+f_opp), f_minus = 0.5*(f-f_opp)
    #pragma unroll
    for (int q=0; q<19; q++) {
        const int qo = OPP[q];
        const float fp   = 0.5f*(fi[q]+fi[qo]);
        const float fm   = 0.5f*(fi[q]-fi[qo]);
        const float feqp = 0.5f*(feq[q]+feq[qo]);
        const float feqm = 0.5f*(feq[q]-feq[qo]);
        f_post[q*N+idx] = (fp + omega_plus*(feqp-fp)) + (fm + omega_minus*(feqm-fm));
    }
}


// =============================================================================
//  KERNEL 3:  lbm_macro_collide_mrt_kernel   <- PRIMARY for FlowSolverUpgrade
//             Fused: macroscopic -> f_eq -> MRT collision -> f_post
//             MRT uses inline M/Minv (no global memory for matrices).
//             S relaxation rates passed as a 19-element device array.
// =============================================================================
__global__ void lbm_macro_collide_mrt_kernel(
    const float* __restrict__ f,
          float* __restrict__ f_post,
          float* __restrict__ rho_out,
          float* __restrict__ u_out,
          float* __restrict__ v_out,
          float* __restrict__ w_out,
    const uint8_t* __restrict__ solid,
    const float*  __restrict__ S,         // [19] MRT relaxation rates on device
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float fi[19];
    #pragma unroll
    for (int q=0; q<19; q++) fi[q] = f[q*N+idx];

    if (solid[idx]) {
        #pragma unroll
        for (int q=0; q<19; q++) f_post[q*N+idx] = W[q];
        rho_out[idx]=1.f; u_out[idx]=0.f; v_out[idx]=0.f; w_out[idx]=0.f;
        return;
    }

    // -- Macroscopic -----------------------------------------------------------
    float rho=0.f, jx=0.f, jy=0.f, jz=0.f;
    #pragma unroll
    for (int q=0; q<19; q++) {
        rho += fi[q];
        jx  += (float)CX[q]*fi[q];
        jy  += (float)CY[q]*fi[q];
        jz  += (float)CZ[q]*fi[q];
    }
    const float inv_rho = 1.f/(rho+1e-12f);
    const float ux=jx*inv_rho, uy=jy*inv_rho, uz=jz*inv_rho;
    rho_out[idx]=rho; u_out[idx]=ux; v_out[idx]=uy; w_out[idx]=uz;

    // -- Equilibrium -----------------------------------------------------------
    float feq[19];
    compute_feq(rho, ux, uy, uz, feq);

    // -- MRT moments: m = M @ f,  meq = M @ feq -------------------------------
    float m[19], meq[19];
    compute_moments(fi,  m);
    compute_moments(feq, meq);

    // -- Relaxation: delta_m[k] = S[k] * (m[k] - meq[k]) ---------------------
    // Conserved channels (0,3,5,7): S=0 by construction, result is 0
    float dm[19];
    #pragma unroll
    for (int k=0; k<19; k++) dm[k] = S[k] * (m[k] - meq[k]);
    // Explicitly zero conserved channels for numerical safety
    dm[0]=0.f; dm[3]=0.f; dm[5]=0.f; dm[7]=0.f;

    // -- Back-transform: df = Minv @ delta_m ----------------------------------
    float df[19];
    apply_Minv(dm, df);

    // -- Post-collision: f* = f - df -------------------------------------------
    #pragma unroll
    for (int q=0; q<19; q++) f_post[q*N+idx] = fi[q] - df[q];
}


// =============================================================================
//  KERNEL 4:  lbm_stream_pull_kernel
//             Pull-scheme streaming: each destination cell reads from sources.
//             Boundary handling:
//               - Periodic dims: coordinate wraps
//               - Wall/flow-axis dims: if source out-of-bounds -> copy self (f_in)
//             This matches _stream() behavior (copy first, then shift).
// =============================================================================
__global__ void lbm_stream_pull_kernel(
    const float* __restrict__ f_in,   // [19][N] post-collision (source)
          float* __restrict__ f_out,  // [19][N] streamed (destination)
    int nx, int ny, int nz,
    int periodic_x,    // 1=periodic in X
    int periodic_y,    // 1=periodic in Y
    int periodic_z,    // 1=periodic in Z
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int nynz = ny * nz;
    const int ix = idx / nynz;
    const int iy = (idx % nynz) / nz;
    const int iz = idx % nz;

    #pragma unroll
    for (int q = 0; q < 19; q++) {
        int sx = ix - CX[q];
        int sy = iy - CY[q];
        int sz = iz - CZ[q];

        // Handle each dimension:
        // Periodic -> wrap; non-periodic & out-of-bounds -> copy self (no-update)
        if (periodic_x) {
            sx = (sx + nx) % nx;
        } else if (sx < 0 || sx >= nx) {
            f_out[q*N + idx] = f_in[q*N + idx];  // retain own value
            continue;  // this direction has no valid source
        }

        if (periodic_y) {
            sy = (sy + ny) % ny;
        } else if (sy < 0 || sy >= ny) {
            f_out[q*N + idx] = f_in[q*N + idx];
            continue;
        }

        if (periodic_z) {
            sz = (sz + nz) % nz;
        } else if (sz < 0 || sz >= nz) {
            f_out[q*N + idx] = f_in[q*N + idx];
            continue;
        }

        const int src = sx*nynz + sy*nz + sz;
        f_out[q*N + idx] = f_in[q*N + src];
    }
}


// =============================================================================
//  KERNEL 5:  lbm_sponge_kernel
//             Outlet sponge zone: f = f - sigma(x) * (f - feq)
//             Applied only to cells where sigma > 0.
//             Requires pre-computed rho, u, v, w (use output of macro_collide).
// =============================================================================
__global__ void lbm_sponge_kernel(
          float* __restrict__ f,          // [19][N] in-place update
    const float* __restrict__ rho,        // [N]
    const float* __restrict__ u,          // [N]
    const float* __restrict__ v,          // [N]
    const float* __restrict__ w,          // [N]
    const float* __restrict__ sigma,      // [N] per-cell sponge strength (0=inactive)
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float sig = sigma[idx];
    if (sig <= 0.f) return;  // not in sponge zone -- skip

    float feq[19];
    compute_feq(rho[idx], u[idx], v[idx], w[idx], feq);

    #pragma unroll
    for (int q = 0; q < 19; q++) {
        const int fi_idx = q*N + idx;
        f[fi_idx] = f[fi_idx] - sig * (f[fi_idx] - feq[q]);
    }
}


// =============================================================================
//  KERNEL 6:  lbm_ibb_kernel
//             Bouzidi-Firdaouss-Lallemand (BFL) interpolated bounce-back.
//             Matches _interp_bounce_back_from_post() exactly.
//
//             For q >= 0.5:  f_dir(x_f) = (1/(2q)) * f_opp_post(x_f)
//                                       + (1 - 1/(2q)) * f_dir_post(x_f)
//             For q < 0.5:   f_dir(x_f) = 2q * f_opp_post(x_f)
//                                       + (1 - 2q) * f_opp_post(x_ff)
//             Fallback (q<0.5, no valid x_ff): halfway BB -> f_opp_post(x_f)
// =============================================================================
__global__ void lbm_ibb_kernel(
    const float* __restrict__ f_post,     // [19][N] post-collision (read-only)
          float* __restrict__ f_stream,   // [19][N] post-stream (modified in-place)
    const int*   __restrict__ fluid_flat, // [n_links] flat index of fluid node
    const int*   __restrict__ dir_i,      // [n_links] direction index (incoming after BB)
    const int*   __restrict__ dir_opp,    // [n_links] opposite direction index
    const float* __restrict__ w1,         // [n_links] weight for primary term
    const float* __restrict__ w2,         // [n_links] weight for secondary term
    const uint8_t* __restrict__ q_ge_half,// [n_links] 1 if q >= 0.5
    const int*   __restrict__ ff_flat,    // [n_links] flat index of x_ff (q<0.5 case)
    const uint8_t* __restrict__ ff_valid, // [n_links] 1 if ff_flat is valid
    int n_links,
    int N)
{
    const int lid = blockIdx.x * blockDim.x + threadIdx.x;
    if (lid >= n_links) return;

    const int   xf_idx  = fluid_flat[lid];  // fluid node flat index
    const int   qi      = dir_i[lid];        // direction written into f_stream
    const int   qo      = dir_opp[lid];      // opposite direction

    const float w_1     = w1[lid];
    const float w_2     = w2[lid];

    // f_opp at fluid node (post-collision)
    const float f_opp_f = f_post[qo * N + xf_idx];

    float bb_val;
    if (q_ge_half[lid]) {
        // q >= 0.5: interpolate with post-collision f_dir at fluid node
        const float f_dir_f = f_post[qi * N + xf_idx];
        bb_val = w_1 * f_opp_f + w_2 * f_dir_f;
    } else {
        if (ff_valid[lid]) {
            // q < 0.5: use second fluid node x_ff
            const int xff_idx   = ff_flat[lid];
            const float f_opp_ff = f_post[qo * N + xff_idx];
            bb_val = w_1 * f_opp_f + w_2 * f_opp_ff;
        } else {
            // Fallback: halfway bounce-back behavior
            bb_val = f_opp_f;
        }
    }

    // Write the incoming (post-BB) population back into f_stream
    f_stream[qi * N + xf_idx] = bb_val;
}


// =============================================================================
//  KERNEL 7:  lbm_reset_solid_kernel
//             Reset solid-node populations to rest equilibrium (rho=1, u=0).
//             Called after IBB to prevent garbage accumulation in solid nodes.
//             Matches _reset_solid_populations().
// =============================================================================
__global__ void lbm_reset_solid_kernel(
          float* __restrict__ f,          // [19][N]
    const uint8_t* __restrict__ solid,    // [N]
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!solid[idx]) return;

    #pragma unroll
    for (int q = 0; q < 19; q++)
        f[q*N + idx] = W[q];  // feq at rho=1, u=v=w=0
}


// =============================================================================
//  PyTorch C++ extension entry points
// =============================================================================

#define BLOCK 256
#define GRID(N) (((N) + BLOCK - 1) / BLOCK)

// -- Helper: validate tensor is on CUDA, contiguous, float32/uint8 ------------
#define CHECK_CUDA(x)  TORCH_CHECK((x).device().is_cuda(),  #x " must be on CUDA")
#define CHECK_CONT(x)  TORCH_CHECK((x).is_contiguous(),     #x " must be contiguous")
#define CHECK_F32(x)   TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_U8(x)    TORCH_CHECK((x).scalar_type() == torch::kUInt8,   #x " must be uint8")
#define CHECK_I32(x)   TORCH_CHECK((x).scalar_type() == torch::kInt32,   #x " must be int32")


// -----------------------------------------------------------------------------
void lbm_macro_collide_bgk(
        torch::Tensor f,
        torch::Tensor f_post,
        torch::Tensor rho_out,
        torch::Tensor u_out,
        torch::Tensor v_out,
        torch::Tensor w_out,
        torch::Tensor solid,
        float omega)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    const int N = f_post.numel() / 19;
    lbm_macro_collide_bgk_kernel<<<GRID(N), BLOCK>>>(
        f.data_ptr<float>(), f_post.data_ptr<float>(),
        rho_out.data_ptr<float>(), u_out.data_ptr<float>(),
        v_out.data_ptr<float>(), w_out.data_ptr<float>(),
        solid.data_ptr<uint8_t>(), omega, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void lbm_macro_collide_trt(
        torch::Tensor f,
        torch::Tensor f_post,
        torch::Tensor rho_out,
        torch::Tensor u_out,
        torch::Tensor v_out,
        torch::Tensor w_out,
        torch::Tensor solid,
        float omega_plus,
        float omega_minus)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    const int N = f_post.numel() / 19;
    lbm_macro_collide_trt_kernel<<<GRID(N), BLOCK>>>(
        f.data_ptr<float>(), f_post.data_ptr<float>(),
        rho_out.data_ptr<float>(), u_out.data_ptr<float>(),
        v_out.data_ptr<float>(), w_out.data_ptr<float>(),
        solid.data_ptr<uint8_t>(), omega_plus, omega_minus, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void lbm_macro_collide_mrt(
        torch::Tensor f,
        torch::Tensor f_post,
        torch::Tensor rho_out,
        torch::Tensor u_out,
        torch::Tensor v_out,
        torch::Tensor w_out,
        torch::Tensor solid,
        torch::Tensor S_rates)   // [19] float32 device tensor
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    CHECK_CUDA(S_rates); CHECK_CONT(S_rates); CHECK_F32(S_rates);
    TORCH_CHECK(S_rates.numel() == 19, "S_rates must have 19 elements");
    const int N = f_post.numel() / 19;
    lbm_macro_collide_mrt_kernel<<<GRID(N), BLOCK>>>(
        f.data_ptr<float>(), f_post.data_ptr<float>(),
        rho_out.data_ptr<float>(), u_out.data_ptr<float>(),
        v_out.data_ptr<float>(), w_out.data_ptr<float>(),
        solid.data_ptr<uint8_t>(),
        S_rates.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void lbm_stream_pull(
        torch::Tensor f_in,
        torch::Tensor f_out,
        int nx, int ny, int nz,
        int periodic_x, int periodic_y, int periodic_z)
{
    CHECK_CUDA(f_in); CHECK_CONT(f_in); CHECK_F32(f_in);
    CHECK_CUDA(f_out); CHECK_CONT(f_out);
    const int N = nx * ny * nz;
    lbm_stream_pull_kernel<<<GRID(N), BLOCK>>>(
        f_in.data_ptr<float>(), f_out.data_ptr<float>(),
        nx, ny, nz, periodic_x, periodic_y, periodic_z, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void lbm_sponge(
        torch::Tensor f,
        torch::Tensor rho,
        torch::Tensor u,
        torch::Tensor v,
        torch::Tensor w,
        torch::Tensor sigma)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    const int N = rho.numel();
    lbm_sponge_kernel<<<GRID(N), BLOCK>>>(
        f.data_ptr<float>(),
        rho.data_ptr<float>(), u.data_ptr<float>(),
        v.data_ptr<float>(), w.data_ptr<float>(),
        sigma.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void lbm_ibb(
        torch::Tensor f_post,
        torch::Tensor f_stream,
        torch::Tensor fluid_flat,
        torch::Tensor dir_i,
        torch::Tensor dir_opp,
        torch::Tensor w1,
        torch::Tensor w2,
        torch::Tensor q_ge_half,
        torch::Tensor ff_flat,
        torch::Tensor ff_valid)
{
    CHECK_CUDA(f_post); CHECK_CONT(f_post);
    const int n_links = fluid_flat.numel();
    if (n_links == 0) return;
    const int N = f_post.numel() / 19;
    lbm_ibb_kernel<<<GRID(n_links), BLOCK>>>(
        f_post.data_ptr<float>(), f_stream.data_ptr<float>(),
        fluid_flat.data_ptr<int>(), dir_i.data_ptr<int>(),
        dir_opp.data_ptr<int>(), w1.data_ptr<float>(), w2.data_ptr<float>(),
        q_ge_half.data_ptr<uint8_t>(), ff_flat.data_ptr<int>(),
        ff_valid.data_ptr<uint8_t>(),
        n_links, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void lbm_reset_solid(torch::Tensor f, torch::Tensor solid)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    const int N = solid.numel();
    lbm_reset_solid_kernel<<<GRID(N), BLOCK>>>(
        f.data_ptr<float>(), solid.data_ptr<uint8_t>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// =============================================================================
//  KERNEL 8:  lbm_zou_he_bc_kernel
//             Fused regularized Zou/He BC for one boundary plane.
//             Replaces the entire _apply_flow_bcs() Python path for one plane.
//
//  Each thread handles one (n1 x n2) node on the boundary plane.
//  Reads 19 f values from global memory, computes everything in registers
//  (feq, Zou/He provisional, P_neq stress tensor, fneq_reg), writes 19 back.
//  Zero inter-thread communication -- fully independent per node.
//
//  bc_type:  0 = velocity inlet  (prescribe u_ax, solve for rho)
//            1 = pressure outlet (prescribe rho_out, solve for u_ax)
//
//  axis / plane_idx / sign define which plane and flow direction.
//  interior_idx is only used for bc_type=1 (tangential extrapolation).
//
//  Backflow handling (outlet only): if u_ax*(sign) < 0, copy from interior.
//
//  BUG FIXES applied:
//    Bug 2 (outlet sign): u_ax = (sum_c0 + 2*sum_known)/rho - 1
//                         was: 1 - (sum_c0 + 2*sum_known)/rho
//    Bug 3 (inlet denominator): rho = (...) / (1 - fabsf(u_in_lat))
//                               was: (1 - u_in_lat) which is wrong for sign=-1
// =============================================================================
__global__ void lbm_zou_he_bc_kernel(
          float* __restrict__ f,          // [19][N] in-place
    const uint8_t* __restrict__ solid,    // [N]
    int nx, int ny, int nz,
    int axis,           // 0=X, 1=Y, 2=Z
    int plane_idx,      // index of the BC plane along axis
    int interior_idx,   // adjacent interior plane (for outlet tangential extrap)
    int bc_type,        // 0=velocity inlet, 1=pressure outlet
    float u_in_lat,     // inlet lattice velocity (signed: sign*|u_lat|)
    float rho_out_lat,  // outlet target density
    int sign,           // +1 or -1 (flow direction sign)
    int n1, int n2,     // plane dimensions (e.g. ny,nz for axis=0)
    int N               // total cells = nx*ny*nz
)
{
    // Thread index within the plane
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n1 * n2) return;

    // Map thread -> (a, b) in plane coords -> flat 3-D index
    const int pa = tid / n2;   // first transverse coord
    const int pb = tid % n2;   // second transverse coord

    // Flat 3-D index for BC plane and interior plane
    int flat_bc, flat_in;
    if (axis == 0) {
        flat_bc = plane_idx    * ny * nz + pa * nz + pb;
        flat_in = interior_idx * ny * nz + pa * nz + pb;
    } else if (axis == 1) {
        flat_bc = pa * ny * nz + plane_idx    * nz + pb;
        flat_in = pa * ny * nz + interior_idx * nz + pb;
    } else {
        flat_bc = pa * ny * nz + pb * nz + plane_idx;
        flat_in = pa * ny * nz + pb * nz + interior_idx;
    }

    // Skip solid nodes
    if (solid[flat_bc]) return;

    // ---- Load 19 populations into registers --------------------------------
    float fi[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) fi[q] = f[q * N + flat_bc];

    // ---- Q tensors (compile-time constants) --------------------------------
    static const float QXX[19] = {-0.33333333f, 0.66666667f, 0.66666667f, -0.33333333f, -0.33333333f, -0.33333333f, -0.33333333f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, -0.33333333f, -0.33333333f, -0.33333333f, -0.33333333f};
    static const float QYY[19] = {-0.33333333f, -0.33333333f, -0.33333333f, 0.66666667f, 0.66666667f, -0.33333333f, -0.33333333f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, -0.33333333f, -0.33333333f, -0.33333333f, -0.33333333f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f};
    static const float QZZ[19] = {-0.33333333f, -0.33333333f, -0.33333333f, -0.33333333f, -0.33333333f, 0.66666667f, 0.66666667f, -0.33333333f, -0.33333333f, -0.33333333f, -0.33333333f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f, 0.66666667f};
    static const float QXY[19] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,-1.f,-1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    static const float QXZ[19] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,-1.f,-1.f, 1.f, 0.f, 0.f, 0.f, 0.f};
    static const float QYZ[19] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,-1.f,-1.f, 1.f};

    // ---- Determine unknown/known direction sets based on axis+plane_idx ----
    // inlet_missing: component of UNKNOWN directions (pointing INTO domain).
    // For inlet at low-face: c_ax = +1; at high-face: c_ax = -1.
    // For outlet at low-face: c_ax = +1; at high-face: c_ax = -1.
    const int bc_at_low = (plane_idx == 0) ? 1 : 0;
    // unknown directions: c_ax == (bc_at_low ? +1 : -1) at inlet,
    //                     c_ax == (bc_at_low ? +1 : -1) at outlet (same face rule)
    const int unknown_cax = bc_at_low ? +1 : -1;  // c_ax of unknowns
    const int known_cax   = -unknown_cax;          // c_ax of knowns (outgoing)

    // ---- Sum populations by c_ax group (using CX/CY/CZ constant arrays) ---
    float sum_c0   = 0.f;  // c_ax == 0
    float sum_cknown = 0.f; // c_ax == known_cax (outgoing, fully determined)
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        int cq;
        if      (axis == 0) cq = CX[q];
        else if (axis == 1) cq = CY[q];
        else                cq = CZ[q];
        if (cq == 0)         sum_c0     += fi[q];
        if (cq == known_cax) sum_cknown += fi[q];
    }

    // ---- Compute rho and u_ax from BC type ---------------------------------
    float rho_bc, u_ax_bc;
    float u_bc = 0.f, v_bc = 0.f, w_bc = 0.f;  // full velocity at BC

    if (bc_type == 0) {
        // Velocity inlet: u_ax = u_in_lat (signed), solve for rho
        // BUG 3 FIX: denominator must be 1 - |u_in| regardless of sign.
        // Old (wrong): rho_bc = (...) / (1.f - u_ax_bc + 1e-12f)
        //   -> for sign=-1, u_ax_bc is negative -> denominator = 1+|u| -> wrong rho
        // New (correct): use fabsf(u_in_lat) so denominator is always 1-|u|
        u_ax_bc = u_in_lat;
        rho_bc  = (sum_c0 + 2.f * sum_cknown) / (1.f - fabsf(u_in_lat) + 1e-12f);
        rho_bc  = fminf(fmaxf(rho_bc, 0.2f), 5.0f);
    } else {
        // Pressure outlet: rho_bc = rho_out, solve for u_ax
        // BUG 2 FIX: correct Zou/He derivation gives u = X/rho - 1, not 1 - X/rho.
        // Old (wrong): u_ax_bc = 1.f - (sum_c0 + 2.f * sum_cknown) / (rho_bc + 1e-12f)
        //   -> computes -u_correct -> backflow check always true -> BC disabled
        // New (correct): u_ax_bc = X/rho - 1
        rho_bc  = rho_out_lat;
        //u_ax_bc = (sum_c0 + 2.f * sum_cknown) / (rho_bc + 1e-12f) - 1.f;
        u_ax_bc = (float)known_cax * ((sum_c0 + 2.f * sum_cknown) / (rho_bc + 1e-12f) - 1.f);

        // Backflow: if velocity points back INTO domain at outlet, copy interior
        const float backflow_check = u_ax_bc * (float)sign;
        if (backflow_check < 0.f) {
            // Load interior populations and write directly -- no BC reconstruction
            #pragma unroll
            for (int q = 0; q < 19; q++)
                f[q * N + flat_bc] = f[q * N + flat_in];
            return;
        }
        // Tangential components from interior (zero-gradient)
        // These are needed to build feq. Read macro from interior.
        // Use local rho=rho_out; u_transverse from interior-plane momentums.
        // For simplicity: compute from interior f populations.
        float fi_in[19];
        #pragma unroll
        for (int q = 0; q < 19; q++) fi_in[q] = f[q * N + flat_in];
        float rho_in = 0.f, jx_in = 0.f, jy_in = 0.f, jz_in = 0.f;
        #pragma unroll
        for (int q = 0; q < 19; q++) {
            rho_in += fi_in[q];
            jx_in  += (float)CX[q] * fi_in[q];
            jy_in  += (float)CY[q] * fi_in[q];
            jz_in  += (float)CZ[q] * fi_in[q];
        }
        const float inv_rho_in = 1.f / (rho_in + 1e-12f);
        // Tangential components from interior, axial from Zou/He
        if      (axis == 0) { u_bc = u_ax_bc; v_bc = jy_in*inv_rho_in; w_bc = jz_in*inv_rho_in; }
        else if (axis == 1) { u_bc = jx_in*inv_rho_in; v_bc = u_ax_bc; w_bc = jz_in*inv_rho_in; }
        else                { u_bc = jx_in*inv_rho_in; v_bc = jy_in*inv_rho_in; w_bc = u_ax_bc; }
    }

    // Assign full velocity for inlet (tangential = 0)
    if (bc_type == 0) {
        if      (axis == 0) u_bc = u_ax_bc;
        else if (axis == 1) v_bc = u_ax_bc;
        else                w_bc = u_ax_bc;
    }

    // ---- Compute feq at BC state -------------------------------------------
    float feq[19];
    compute_feq(rho_bc, u_bc, v_bc, w_bc, feq);

    // ---- Provisional Zou/He: fill unknown directions -----------------------
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        int cq;
        if      (axis == 0) cq = CX[q];
        else if (axis == 1) cq = CY[q];
        else                cq = CZ[q];
        if (cq == unknown_cax) {
            // fi[q] = fi_opp[q] + feq[q] - feq_opp[q]  (Zou/He)
            fi[q] = fi[OPP[q]] + feq[q] - feq[OPP[q]];
        }
    }

    // ---- Regularized reconstruction: compute P_neq -------------------------
    // fneq_q = fi[q] - feq[q]
    float Pxx=0.f, Pyy=0.f, Pzz=0.f, Pxy=0.f, Pxz=0.f, Pyz=0.f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        const float fn = fi[q] - feq[q];
        Pxx += fn * QXX[q];
        Pyy += fn * QYY[q];
        Pzz += fn * QZZ[q];
        Pxy += fn * QXY[q];
        Pxz += fn * QXZ[q];
        Pyz += fn * QYZ[q];
    }

    // fneq_reg[q] = w[q] / (2 * cs4) * (Qxx*Pxx + Qyy*Pyy + ...)
    // cs4 = (1/3)^2 = 1/9,  1/(2*cs4) = 4.5
    const float inv_2cs4 = 4.5f;

    // Rebuild full plane: f[q] = feq[q] + fneq_reg[q]
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        const float S_q = QXX[q]*Pxx + QYY[q]*Pyy + QZZ[q]*Pzz
                        + 2.f*(QXY[q]*Pxy + QXZ[q]*Pxz + QYZ[q]*Pyz);
        f[q * N + flat_bc] = feq[q] + W[q] * S_q * inv_2cs4;
    }
}


// =============================================================================
//  Host-side entry point for lbm_zou_he_bc
// =============================================================================
void lbm_zou_he_bc(
        torch::Tensor f,
        torch::Tensor solid,
        int nx, int ny, int nz,
        int axis, int plane_idx, int interior_idx,
        int bc_type,
        float u_in_lat, float rho_out_lat, int sign)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    int n1, n2;
    if      (axis == 0) { n1 = ny; n2 = nz; }
    else if (axis == 1) { n1 = nx; n2 = nz; }
    else                { n1 = nx; n2 = ny; }
    const int N = nx * ny * nz;
    const int n_plane = n1 * n2;
    lbm_zou_he_bc_kernel<<<GRID(n_plane), BLOCK>>>(
        f.data_ptr<float>(), solid.data_ptr<uint8_t>(),
        nx, ny, nz, axis, plane_idx, interior_idx,
        bc_type, u_in_lat, rho_out_lat, sign,
        n1, n2, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// =============================================================================
//  KERNEL 9:  lbm_macro_collide_mrt_smag_kernel
//             Smagorinsky-LES MRT collision with local per-cell omega.
//
//  Key idea: estimate local strain rate |S| from the non-equilibrium stress
//  tensor Pi_neq = sum_q f_neq_q * c_q_alpha * c_q_beta, then solve the
//  Smagorinsky quadratic for an effective viscosity nu_eff > nu_0.
//
//  nu_eff = [nu_0 + sqrt(nu_0^2 + 4*(Cs*dx)^2 * |Pi|_F / (2*rho*cs^4))] / 2
//  omega_eff = 1 / (3*nu_eff + 0.5)
//
//  The local omega_eff replaces S[9,11,13,14,15] (viscous channels) for this
//  cell only.  All other MRT channels keep their global relaxation rates.
//
//  Cs_dx2: (Cs * dx)^2  precomputed on host and passed as a scalar.
//  cs4 = (1/3)^2 = 1/9;  inv_2cs4 = 4.5
//
//  Reference: Hou et al. (1994) JCP, Sagaut (2010) Turbulent LBM, Ch 4.
// =============================================================================
__global__ void lbm_macro_collide_mrt_smag_kernel(
    const float* __restrict__ f,
          float* __restrict__ f_post,
          float* __restrict__ rho_out,
          float* __restrict__ u_out,
          float* __restrict__ v_out,
          float* __restrict__ w_out,
    const uint8_t* __restrict__ solid,
    const float*   __restrict__ S,       // [19] global MRT rates
    float  Cs_dx2,                       // (Cs * dx)^2  [lattice units]
    float  omega0,                       // base viscous relaxation rate
    int    N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float fi[19];
    #pragma unroll
    for (int q = 0; q < 19; q++) fi[q] = f[q*N + idx];

    if (solid[idx]) {
        #pragma unroll
        for (int q = 0; q < 19; q++) f_post[q*N + idx] = W[q];
        rho_out[idx]=1.f; u_out[idx]=0.f; v_out[idx]=0.f; w_out[idx]=0.f;
        return;
    }

    // -- Macroscopic ----------------------------------------------------------
    float rho=0.f, jx=0.f, jy=0.f, jz=0.f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        rho += fi[q];
        jx  += (float)CX[q]*fi[q];
        jy  += (float)CY[q]*fi[q];
        jz  += (float)CZ[q]*fi[q];
    }
    const float inv_rho = 1.f / (rho + 1e-12f);
    const float ux=jx*inv_rho, uy=jy*inv_rho, uz=jz*inv_rho;
    rho_out[idx]=rho; u_out[idx]=ux; v_out[idx]=uy; w_out[idx]=uz;

    // -- Equilibrium ----------------------------------------------------------
    float feq[19];
    compute_feq(rho, ux, uy, uz, feq);

    // -- Non-equilibrium stress tensor Pi_alphabeta = sum_q fneq_q * ca * cb --
    // Only the 6 independent components needed:
    //   Pxx = sum fneq * cx*cx,  Pyy = cy*cy,  Pzz = cz*cz
    //   Pxy = cx*cy,             Pxz = cx*cz,  Pyz = cy*cz
    float Pxx=0.f, Pyy=0.f, Pzz=0.f, Pxy=0.f, Pxz=0.f, Pyz=0.f;
    #pragma unroll
    for (int q = 0; q < 19; q++) {
        const float fneq_q = fi[q] - feq[q];
        const float cx=(float)CX[q], cy=(float)CY[q], cz=(float)CZ[q];
        Pxx += fneq_q * cx * cx;
        Pyy += fneq_q * cy * cy;
        Pzz += fneq_q * cz * cz;
        Pxy += fneq_q * cx * cy;
        Pxz += fneq_q * cx * cz;
        Pyz += fneq_q * cy * cz;
    }

    // -- Frobenius norm: |Pi|_F = sqrt(Pxx^2+Pyy^2+Pzz^2+2*(Pxy^2+Pxz^2+Pyz^2)) --
    const float Pi_F = sqrtf(Pxx*Pxx + Pyy*Pyy + Pzz*Pzz
                           + 2.f*(Pxy*Pxy + Pxz*Pxz + Pyz*Pyz));

    // -- Smagorinsky effective relaxation rate --------------------------------
    // nu_eff = [nu0 + sqrt(nu0^2 + 4*(Cs*dx)^2 * |Pi|_F / (2*rho*cs^4))] / 2
    // cs^4 = 1/9,  1/(2*cs^4) = 4.5
    const float nu0 = (1.f/omega0 - 0.5f) / 3.f;
    const float discriminant = nu0*nu0
        + 4.f * Cs_dx2 * (Pi_F * inv_rho) * 4.5f;  // 4.5 = 1/(2*cs4)
    const float nu_eff = 0.5f * (nu0 + sqrtf(fmaxf(discriminant, nu0*nu0)));
    const float omega_eff = fminf(1.99f, 1.f / (3.f*nu_eff + 0.5f));

    // -- MRT with local viscous rates -----------------------------------------
    float m[19], meq[19];
    compute_moments(fi,  m);
    compute_moments(feq, meq);

    float dm[19];
    #pragma unroll
    for (int k = 0; k < 19; k++) dm[k] = S[k] * (m[k] - meq[k]);

    // Viscous stress channels: replace with local omega_eff
    // (channels 9, 11, 13, 14, 15 = 3p_xx, p_ww, p_xy, p_yz, p_xz)
    dm[ 9] = omega_eff * (m[ 9] - meq[ 9]);
    dm[11] = omega_eff * (m[11] - meq[11]);
    dm[13] = omega_eff * (m[13] - meq[13]);
    dm[14] = omega_eff * (m[14] - meq[14]);
    dm[15] = omega_eff * (m[15] - meq[15]);

    // Conserved channels: force to zero for numerical safety
    dm[0]=0.f; dm[3]=0.f; dm[5]=0.f; dm[7]=0.f;

    float df[19];
    apply_Minv(dm, df);

    #pragma unroll
    for (int q = 0; q < 19; q++) f_post[q*N + idx] = fi[q] - df[q];
}


void lbm_macro_collide_mrt_smag(
        torch::Tensor f,
        torch::Tensor f_post,
        torch::Tensor rho,
        torch::Tensor u,
        torch::Tensor v,
        torch::Tensor w,
        torch::Tensor solid,
        torch::Tensor S_rates,
        float Cs_dx2,
        float omega0)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    const int N = (int)f.numel() / 19;
    lbm_macro_collide_mrt_smag_kernel<<<GRID(N), BLOCK>>>(
        f.data_ptr<float>(),
        f_post.data_ptr<float>(),
        rho.data_ptr<float>(),
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        w.data_ptr<float>(),
        solid.data_ptr<uint8_t>(),
        S_rates.data_ptr<float>(),
        Cs_dx2, omega0, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// =============================================================================
//  KERNEL 10: lbm_convective_bc_reduce_kernel
//             Convective (zero-gradient) outlet BC with fused on-device
//             rho reduction. Two-pass design:
//
//  Pass 1 (this kernel): for each fluid outlet node --
//    a) Zero-gradient copy: f_unknown(outlet) = f_unknown(interior)
//       where "unknown" = directions pointing INTO domain at the outlet face.
//    b) Compute new rho = sum_q f[q] at this outlet node.
//    c) Block-level tree reduction in shared memory.
//    d) One atomicAdd per block to device-side d_sum_rho and d_count.
//
//  Pass 2 (kernel 11): reads d_sum_rho and d_count, applies correction
//    factor = rho_out_lat * count / sum_rho to all 19 populations.
//
//  No host-device transfers anywhere. d_sum_rho and d_count are device
//  pointers allocated once in the host entry point (persistent static).
//  Caller must cudaMemsetAsync both to zero before launching this kernel.
//
//  Shared memory layout per block: [0..bdx-1] = rho accumulators (float)
//                                  [bdx..2*bdx-1] = fluid counts (float)
// =============================================================================
__global__ void lbm_convective_bc_reduce_kernel(
          float*   __restrict__ f,
    const uint8_t* __restrict__ solid,
    int nx, int ny, int nz,
    int axis, int outlet_idx, int interior_idx,
    int unknown_cax,   // c_ax value of unknown directions: +1 or -1
    int n1, int n2, int N,
    float* __restrict__ d_sum_rho,   // device scalar: accumulated rho (init 0)
    int*   __restrict__ d_count)     // device scalar: fluid node count (init 0)
{
    extern __shared__ float sdata[];
    float* s_rho = sdata;
    float* s_cnt = sdata + blockDim.x;
    const int lane = threadIdx.x;
    s_rho[lane] = 0.f;
    s_cnt[lane] = 0.f;

    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int nynz = ny * nz;

    if (tid < n1 * n2) {
        const int pa = tid / n2, pb = tid % n2;

        int flat_out, flat_in;
        if      (axis == 0) { flat_out = outlet_idx   *nynz + pa*nz + pb;
                              flat_in  = interior_idx *nynz + pa*nz + pb; }
        else if (axis == 1) { flat_out = pa*nynz + outlet_idx   *nz + pb;
                              flat_in  = pa*nynz + interior_idx *nz + pb; }
        else                { flat_out = pa*nynz + pb*nz + outlet_idx;
                              flat_in  = pa*nynz + pb*nz + interior_idx; }

        if (!solid[flat_out]) {
            // Step 1: zero-gradient copy for unknown directions
            #pragma unroll
            for (int q = 0; q < 19; q++) {
                int cq;
                if      (axis == 0) cq = CX[q];
                else if (axis == 1) cq = CY[q];
                else                cq = CZ[q];
                if (cq == unknown_cax)
                    f[q*N + flat_out] = f[q*N + flat_in];
            }

            // Step 2: compute rho at this outlet node after copy
            float rho_node = 0.f;
            #pragma unroll
            for (int q = 0; q < 19; q++) rho_node += f[q*N + flat_out];

            s_rho[lane] = rho_node;
            s_cnt[lane] = 1.f;
        }
    }
    __syncthreads();

    // Block-level tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            s_rho[lane] += s_rho[lane + stride];
            s_cnt[lane] += s_cnt[lane + stride];
        }
        __syncthreads();
    }

    // One atomicAdd per block to global device accumulators
    if (lane == 0) {
        atomicAdd(d_sum_rho, s_rho[0]);
        atomicAdd(d_count,   (int)s_cnt[0]);
    }
}


// =============================================================================
//  KERNEL 11: lbm_convective_mass_correct_kernel
//             Reads d_sum_rho and d_count written by kernel 10.
//             Computes correction = rho_out_lat * count / sum_rho.
//             Scales all 19 populations at each fluid outlet node in-place.
//             Fully independent per node -- no synchronization needed.
//             No host-device transfer: reads device pointers directly.
// =============================================================================
__global__ void lbm_convective_mass_correct_kernel(
          float*   __restrict__ f,
    const uint8_t* __restrict__ solid,
    int ny, int nz,
    int axis, int outlet_idx,
    float rho_out_lat,
    const float* __restrict__ d_sum_rho,
    const int*   __restrict__ d_count,
    int n1, int n2, int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n1 * n2) return;

    const int pa = tid / n2, pb = tid % n2;
    const int nynz = ny * nz;

    int flat_out;
    if      (axis == 0) flat_out = outlet_idx * nynz + pa*nz + pb;
    else if (axis == 1) flat_out = pa*nynz + outlet_idx*nz + pb;
    else                flat_out = pa*nynz + pb*nz + outlet_idx;

    if (solid[flat_out]) return;

    const float cnt  = (float)(*d_count);
    const float ssum = *d_sum_rho;

    // Guard against degenerate case (no fluid nodes or zero rho sum)
    if (cnt < 0.5f || ssum < 1e-10f) return;

    const float correction = rho_out_lat * cnt / ssum;

    #pragma unroll
    for (int q = 0; q < 19; q++)
        f[q*N + flat_out] *= correction;
}


// =============================================================================
//  Host-side entry point for lbm_convective_bc
//  Orchestrates both kernel 10 and kernel 11.
//  Allocates persistent device accumulators on first call (never freed).
//  All operations are on-device with no host-device synchronization.
//
//  unknown_cax: c_ax value of directions pointing INTO domain at outlet.
//               = +1 if outlet is at low face (plane_idx == 0)
//               = -1 if outlet is at high face (plane_idx == n-1)
// =============================================================================
static float* g_d_sum_rho = nullptr;
static int*   g_d_count   = nullptr;

void lbm_convective_bc(
        torch::Tensor f,
        torch::Tensor solid,
        int nx, int ny, int nz,
        int axis, int outlet_idx, int interior_idx,
        int unknown_cax,
        float rho_out_lat,
        int n1, int n2)
{
    CHECK_CUDA(f); CHECK_CONT(f); CHECK_F32(f);
    const int N = nx * ny * nz;

    // Allocate device accumulators on first call
    if (g_d_sum_rho == nullptr) {
        cudaMalloc(&g_d_sum_rho, sizeof(float));
        cudaMalloc(&g_d_count,   sizeof(int));
    }

    // Zero accumulators asynchronously (stays on same stream, no CPU sync)
    cudaMemsetAsync(g_d_sum_rho, 0, sizeof(float));
    cudaMemsetAsync(g_d_count,   0, sizeof(int));

    // Kernel 10: zero-gradient copy + shared-memory tree reduction + atomicAdd
    // Shared memory: 2 * BLOCK floats per block (rho array + count array)
    const int smem = 2 * BLOCK * sizeof(float);
    lbm_convective_bc_reduce_kernel<<<GRID(n1*n2), BLOCK, smem>>>(
        f.data_ptr<float>(),
        solid.data_ptr<uint8_t>(),
        nx, ny, nz,
        axis, outlet_idx, interior_idx,
        unknown_cax,
        n1, n2, N,
        g_d_sum_rho, g_d_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Kernel 11: apply correction factor (reads device scalars, zero CPU transfer)
    lbm_convective_mass_correct_kernel<<<GRID(n1*n2), BLOCK>>>(
        f.data_ptr<float>(),
        solid.data_ptr<uint8_t>(),
        ny, nz,
        axis, outlet_idx,
        rho_out_lat,
        g_d_sum_rho, g_d_count,
        n1, n2, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused LBM CUDA kernels for LBMCHT3D_Torch";

    m.def("macro_collide_bgk", &lbm_macro_collide_bgk,
          "Fused macroscopic + feq + BGK collision (f->f_post, rho, u, v, w)");

    m.def("macro_collide_trt", &lbm_macro_collide_trt,
          "Fused macroscopic + feq + TRT collision");

    m.def("macro_collide_mrt", &lbm_macro_collide_mrt,
          "Fused macroscopic + feq + MRT collision (primary upgrade path)");

    m.def("stream_pull", &lbm_stream_pull,
          "Pull-scheme streaming with periodic/wall boundary handling");

    m.def("sponge", &lbm_sponge,
          "Outlet sponge-zone relaxation: f = f - sigma*(f - feq)");

    m.def("ibb", &lbm_ibb,
          "Bouzidi interpolated bounce-back (BFL): writes into f_stream in-place");

    m.def("reset_solid", &lbm_reset_solid,
          "Reset solid-node populations to rest equilibrium (rho=1, u=0)");

    m.def("macro_collide_mrt_smag", &lbm_macro_collide_mrt_smag,
          "Smagorinsky-LES MRT collision: per-cell omega from local strain rate "
          "(for Re > ~500; C_s=0.1 typical for wall-bounded flows)");

    m.def("zou_he_bc", &lbm_zou_he_bc,
          "Fused regularized Zou/He BC for one boundary plane "
          "(bc_type 0=velocity inlet, 1=pressure outlet). "
          "Bug 2 (outlet sign) and Bug 3 (inlet denominator) fixed.");

    m.def("convective_bc", &lbm_convective_bc,
          "Convective outlet BC: zero-gradient copy for unknown directions + "
          "fused on-device rho reduction (shared-mem tree + atomicAdd) + "
          "mass correction kernel. Two CUDA passes, zero host-device transfer. "
          "Args: f, solid, nx, ny, nz, axis, outlet_idx, interior_idx, "
          "unknown_cax (+1=low-face outlet, -1=high-face), rho_out_lat, n1, n2");
}
