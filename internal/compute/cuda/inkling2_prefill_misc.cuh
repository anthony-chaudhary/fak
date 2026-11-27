// inkling2_prefill_misc.cuh: fp32 prefill versions of the small elementwise and
// reduction kernels of the batched inkling2_loop prefill (RMS rows, SITU, controller
// down/apply, short conv, depth mix).
//
// Prefill only needs greedy parity with the fp32 llama.cpp oracle, not bit identity
// with the decode path, so reductions are fp32 trees and transcendentals are accurate
// fp32 (tanhf/expf); FP64 runs at 1/64 rate on consumer Blackwell. Each launcher keeps
// the signature of the launcher it replaces and falls back to it for shapes the fast
// path does not cover. Included after inkling2_prefill.cuh by cuda_kernels.cu.

static __device__ __forceinline__ bool ink2m_al16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

static __device__ __forceinline__ float ink2m_block_sum(float v, float *sh) {
  v = ink2_warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int)(blockDim.x >> 5);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = lane < nw ? sh[lane] : 0.f;
  return ink2_warp_sum(v);
}

// ---------------------------------------------------------------- RMS rows
#define INK2M_RMS_THREADS 256
#define INK2M_RMS_VEC 4

// One block per row; the row stays in registers between the sum of squares and the scale.
__global__ void __launch_bounds__(INK2M_RMS_THREADS) k_ink2m_rms(const float *x, const float *w, float *y, int n,
                                                                 float eps) {
  __shared__ float sh[32];
  const float *xr = x + (size_t)blockIdx.x * n;
  float *yr = y + (size_t)blockIdx.x * n;
  const int n4 = n >> 2;
  float4 v[INK2M_RMS_VEC];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < INK2M_RMS_VEC; ++j) {
    const int i = threadIdx.x + j * INK2M_RMS_THREADS;
    v[j] = i < n4 ? reinterpret_cast<const float4 *>(xr)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    ss = fmaf(v[j].x, v[j].x, ss);
    ss = fmaf(v[j].y, v[j].y, ss);
    ss = fmaf(v[j].z, v[j].z, ss);
    ss = fmaf(v[j].w, v[j].w, ss);
  }
  ss = ink2m_block_sum(ss, sh);
  const float s = 1.f / sqrtf(ss / (float)n + eps);
#pragma unroll
  for (int j = 0; j < INK2M_RMS_VEC; ++j) {
    const int i = threadIdx.x + j * INK2M_RMS_THREADS;
    if (i >= n4) break;
    float4 o = make_float4(__fmul_rn(v[j].x, s), __fmul_rn(v[j].y, s), __fmul_rn(v[j].z, s), __fmul_rn(v[j].w, s));
    if (w) {
      const float4 wv = reinterpret_cast<const float4 *>(w)[i];
      o = make_float4(__fmul_rn(o.x, wv.x), __fmul_rn(o.y, wv.y), __fmul_rn(o.z, wv.z), __fmul_rn(o.w, wv.w));
    }
    reinterpret_cast<float4 *>(yr)[i] = o;
  }
}

extern "C" int fcuda_ink2p_rms(const float *x, const float *w, float *y, int rows, int n, float eps) {
  const bool vec = (n & 3) == 0 && n <= 4 * INK2M_RMS_VEC * INK2M_RMS_THREADS && ((uintptr_t)x & 15) == 0 &&
                   ((uintptr_t)y & 15) == 0 && ((uintptr_t)w & 15) == 0;
  if (!vec) return fcuda_ink2_rms_rows(x, w, y, rows, n, eps);
  FCUDA_TS("k_ink2m_rms", rows, k_ink2m_rms<<<rows, INK2M_RMS_THREADS, 0, g_stream>>>(x, w, y, n, eps));
  return ink2_last_error();
}

// ---------------------------------------------------------------- SITU
static __device__ __forceinline__ float ink2m_situ(float g, float u, float beta, float lb) {
  const float a = __fmul_rn(__fmul_rn(beta, tanhf(__fdiv_rn(g, beta))), __frcp_rn(__fadd_rn(1.f, expf(-g))));
  if (lb > 0.f) u = __fmul_rn(lb, tanhf(__fdiv_rn(u, lb)));
  return __fmul_rn(a, u);
}

// blockIdx.y is the routed slot; each thread owns 4 consecutive columns.
__global__ void k_ink2m_situ4(const float *gu, float *act, int width, int actStride, float beta, float lb) {
  const int j4 = blockIdx.x * blockDim.x + threadIdx.x;
  if (j4 * 4 >= width) return;
  const float *row = gu + (size_t)blockIdx.y * 2 * width;
  const float4 g = reinterpret_cast<const float4 *>(row)[j4];
  const float4 u = reinterpret_cast<const float4 *>(row + width)[j4];
  const float4 o = make_float4(ink2m_situ(g.x, u.x, beta, lb), ink2m_situ(g.y, u.y, beta, lb),
                               ink2m_situ(g.z, u.z, beta, lb), ink2m_situ(g.w, u.w, beta, lb));
  reinterpret_cast<float4 *>(act + (size_t)blockIdx.y * actStride)[j4] = o;
}

__global__ void k_ink2m_situ1(const float *gu, float *act, int width, int actStride, float beta, float lb) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= width) return;
  const float *row = gu + (size_t)blockIdx.y * 2 * width;
  act[(size_t)blockIdx.y * actStride + j] = ink2m_situ(row[j], row[width + j], beta, lb);
}

extern "C" int fcuda_ink2p_situ_v2(const float *gu, float *act, int k, int width, int actStride, double beta,
                                   double lb) {
  const float fb = (float)beta, fl = (float)lb;
  if ((width & 3) == 0 && (actStride & 3) == 0 && ((uintptr_t)gu & 15) == 0 && ((uintptr_t)act & 15) == 0) {
    const dim3 grid((width / 4 + 127) / 128, k);
    FCUDA_TS("k_ink2m_situ4", grid, k_ink2m_situ4<<<grid, 128, 0, g_stream>>>(gu, act, width, actStride, fb, fl));
  } else {
    const dim3 grid((width + 255) / 256, k);
    FCUDA_TS("k_ink2m_situ1", grid, k_ink2m_situ1<<<grid, 256, 0, g_stream>>>(gu, act, width, actStride, fb, fl));
  }
  return ink2_last_error();
}

// ---------------------------------------------------------------- controller down
// g[t][r] = rms_scale(cur_t) * W[r,0:n].cur_t + rms_scale(msg_t) * W[r,n:2n].msg_t as a
// tiled fp32 GEMM: a block owns 16 tokens x 32 ranks and walks K in 64-wide chunks
// staged k-major in smem; each warp takes 8 k of every chunk with a 4x4 register tile,
// and the 8 warp partials are summed at the end. Token norms ride the X staging.
#define INK2M_CD_TOK 16
#define INK2M_CD_RK 32
#define INK2M_CD_KC 64
#define INK2M_CD_XP 20
#define INK2M_CD_WP 36

#define INK2M_CTRL_MAX 3
struct Ink2PCtrlSet {
  const float *cur[INK2M_CTRL_MAX], *msg[INK2M_CTRL_MAX], *down[INK2M_CTRL_MAX];
  const float *gain[INK2M_CTRL_MAX], *forget[INK2M_CTRL_MAX];
  float *g, *out;
};

__global__ void __launch_bounds__(256) k_ink2m_ctrl_down(Ink2PCtrlSet cs, int n, int rank, int T) {
  __shared__ __align__(16) float xs[INK2M_CD_KC * INK2M_CD_XP];
  __shared__ __align__(16) float ws[INK2M_CD_KC * INK2M_CD_WP];
  __shared__ float red[8 * INK2M_CD_TOK * INK2M_CD_RK];
  __shared__ float ssh[2][8][INK2M_CD_TOK];
  const int j = blockIdx.z;
  const float *cur = cs.cur[j], *msg = cs.msg[j], *W = cs.down[j];
  float *g = cs.g + (size_t)j * T * rank;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t0 = blockIdx.x * INK2M_CD_TOK, r0 = blockIdx.y * INK2M_CD_RK;
  // staging roles: X float4 (token xt, k quad xq); W float4 x2 (rank wr, k quads wq, wq+8)
  const int xt = tid & 15, xq = tid >> 4;
  const int wr = tid & 31, wq = tid >> 5;
  const bool xok = t0 + xt < T;
  // compute roles: tokens tq*4..+3, ranks rq*4..+3, k = warp*8 .. +7 of the chunk
  const int tq = lane >> 3, rq = lane & 7;
  float acc[2][4][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b) acc[h][a][b] = 0.f;
  float ss[2] = {0.f, 0.f};
  const int nck = n / INK2M_CD_KC;
  const float *wrow = W + (size_t)(r0 + wr) * 2 * n;
  float4 px, pw0, pw1;
  auto fetch = [&](int c) {
    const int h = c >= nck, k0 = (h ? c - nck : c) * INK2M_CD_KC;
    const float *src = h ? msg : cur;
    px = xok ? *reinterpret_cast<const float4 *>(src + (size_t)(t0 + xt) * n + k0 + xq * 4)
             : make_float4(0.f, 0.f, 0.f, 0.f);
    const float *wp = wrow + (size_t)h * n + k0;
    pw0 = *reinterpret_cast<const float4 *>(wp + wq * 4);
    pw1 = *reinterpret_cast<const float4 *>(wp + (wq + 8) * 4);
  };
  fetch(0);
  for (int c = 0; c < 2 * nck; ++c) {
    const int h = c >= nck;
    __syncthreads();
    {
      float *xd = xs + (xq * 4) * INK2M_CD_XP + xt;
      xd[0] = px.x;
      xd[INK2M_CD_XP] = px.y;
      xd[2 * INK2M_CD_XP] = px.z;
      xd[3 * INK2M_CD_XP] = px.w;
      ss[h] = fmaf(px.x, px.x, ss[h]);
      ss[h] = fmaf(px.y, px.y, ss[h]);
      ss[h] = fmaf(px.z, px.z, ss[h]);
      ss[h] = fmaf(px.w, px.w, ss[h]);
      float *wd = ws + (wq * 4) * INK2M_CD_WP + wr;
      wd[0] = pw0.x;
      wd[INK2M_CD_WP] = pw0.y;
      wd[2 * INK2M_CD_WP] = pw0.z;
      wd[3 * INK2M_CD_WP] = pw0.w;
      wd += 32 * INK2M_CD_WP;
      wd[0] = pw1.x;
      wd[INK2M_CD_WP] = pw1.y;
      wd[2 * INK2M_CD_WP] = pw1.z;
      wd[3 * INK2M_CD_WP] = pw1.w;
    }
    __syncthreads();
    if (c + 1 < 2 * nck) fetch(c + 1);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int k = warp * 8 + kk;
      const float4 xv = *reinterpret_cast<const float4 *>(xs + k * INK2M_CD_XP + tq * 4);
      const float4 wv = *reinterpret_cast<const float4 *>(ws + k * INK2M_CD_WP + rq * 4);
      const float xa[4] = {xv.x, xv.y, xv.z, xv.w}, wa[4] = {wv.x, wv.y, wv.z, wv.w};
      if (h) {
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
          for (int b = 0; b < 4; ++b) acc[1][a][b] = fmaf(xa[a], wa[b], acc[1][a][b]);
      } else {
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
          for (int b = 0; b < 4; ++b) acc[0][a][b] = fmaf(xa[a], wa[b], acc[0][a][b]);
      }
    }
  }
  // token norms: threads xt and xt+16 of every warp share a token
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float v = ss[h] + __shfl_xor_sync(0xffffffff, ss[h], 16);
    if (lane < 16) ssh[h][warp][lane] = v;
  }
  float dot[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    __syncthreads();
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b) red[(warp * INK2M_CD_TOK + tq * 4 + a) * INK2M_CD_RK + rq * 4 + b] = acc[h][a][b];
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int o = tid + e * 256;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) s += red[w * INK2M_CD_TOK * INK2M_CD_RK + o];
      dot[h][e] = s;
    }
  }
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int o = tid + e * 256, tl = o / INK2M_CD_RK, r = o % INK2M_CD_RK;
    if (t0 + tl >= T) continue;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      s0 += ssh[0][w][tl];
      s1 += ssh[1][w][tl];
    }
    const float sc = 1.f / sqrtf(s0 / (float)n + 1e-6f), sm = 1.f / sqrtf(s1 / (float)n + 1e-6f);
    g[(size_t)(t0 + tl) * rank + r0 + r] = fmaf(sc, dot[0][e], sm * dot[1][e]);
  }
}

extern "C" int fcuda_ink2p_ctrl_down_v2(const float *cur, const float *msg, const float *W, float *g, int n, int rank,
                                        int T) {
  if (n % INK2M_CD_KC != 0 || rank % INK2M_CD_RK != 0 || ((uintptr_t)cur & 15) || ((uintptr_t)msg & 15) ||
      ((uintptr_t)W & 15))
    return fcuda_ink2p_ctrl_down(cur, msg, W, g, n, rank, T);
  const dim3 grid((T + INK2M_CD_TOK - 1) / INK2M_CD_TOK, rank / INK2M_CD_RK, 1);
  // Small-T batches produce too few 16x32 tiles to fill the SMs; fcuda_ink2p_ctrl_down launches rank x T blocks.
  if ((size_t)grid.x * grid.y < 16) return fcuda_ink2p_ctrl_down(cur, msg, W, g, n, rank, T);
  Ink2PCtrlSet cs = {};
  cs.cur[0] = cur; cs.msg[0] = msg; cs.down[0] = W; cs.g = g;
  FCUDA_TS("k_ink2m_ctrl_down", grid, k_ink2m_ctrl_down<<<grid, 256, 0, g_stream>>>(cs, n, rank, T));
  return ink2_last_error();
}

// ---------------------------------------------------------------- controller apply
// out[t][i] = cur - tanh(forgetW[i].g_t)*cur + tanh(gainW[i].g_t)*msg over a tile of 64
// channels x 32 tokens; the gain/forget/g slices are staged rank-major once per block
// and each thread owns 4 channels x 2 tokens. The rank dot keeps the serial fmaf order.
#define INK2M_CA_CH 64
#define INK2M_CA_TOK 32
#define INK2M_CA_WP 68
#define INK2M_CA_GP 36

__global__ void __launch_bounds__(256) k_ink2m_ctrl_apply(Ink2PCtrlSet cs, int n, int rank, int T) {
  extern __shared__ __align__(16) float ash[];
  float *gws = ash, *fws = ash + (size_t)rank * INK2M_CA_WP, *gs = ash + (size_t)rank * 2 * INK2M_CA_WP;
  const int j = blockIdx.z;
  const float *cur = cs.cur[j], *msg = cs.msg[j], *gainW = cs.gain[j], *forgetW = cs.forget[j];
  const float *g = cs.g + (size_t)j * T * rank;
  float *out = cs.out + (size_t)j * T * n;
  const int tid = threadIdx.x;
  const int i0 = blockIdx.x * INK2M_CA_CH, t0 = blockIdx.y * INK2M_CA_TOK;
  const int rq4 = rank >> 2;
  for (int idx = tid; idx < INK2M_CA_CH * rq4; idx += 256) {
    const int ch = idx % INK2M_CA_CH, rq = idx / INK2M_CA_CH;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
    if (i0 + ch < n) {
      a = *reinterpret_cast<const float4 *>(gainW + (size_t)(i0 + ch) * rank + rq * 4);
      b = *reinterpret_cast<const float4 *>(forgetW + (size_t)(i0 + ch) * rank + rq * 4);
    }
    float *gd = gws + (rq * 4) * INK2M_CA_WP + ch, *fd = fws + (rq * 4) * INK2M_CA_WP + ch;
    gd[0] = a.x;
    gd[INK2M_CA_WP] = a.y;
    gd[2 * INK2M_CA_WP] = a.z;
    gd[3 * INK2M_CA_WP] = a.w;
    fd[0] = b.x;
    fd[INK2M_CA_WP] = b.y;
    fd[2 * INK2M_CA_WP] = b.z;
    fd[3 * INK2M_CA_WP] = b.w;
  }
  for (int idx = tid; idx < INK2M_CA_TOK * rank; idx += 256) {
    const int r = idx % rank, tl = idx / rank;
    gs[r * INK2M_CA_GP + tl] = t0 + tl < T ? g[(size_t)(t0 + tl) * rank + r] : 0.f;
  }
  __syncthreads();
  const int cq = tid & 15, tp = tid >> 4;
  float gn[2][4], fg[2][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) gn[a][b] = fg[a][b] = 0.f;
#pragma unroll 8
  for (int r = 0; r < rank; ++r) {
    const float4 gw = *reinterpret_cast<const float4 *>(gws + r * INK2M_CA_WP + cq * 4);
    const float4 fw = *reinterpret_cast<const float4 *>(fws + r * INK2M_CA_WP + cq * 4);
    const float2 gv = *reinterpret_cast<const float2 *>(gs + r * INK2M_CA_GP + tp * 2);
    const float ga[4] = {gw.x, gw.y, gw.z, gw.w}, fa[4] = {fw.x, fw.y, fw.z, fw.w}, gt[2] = {gv.x, gv.y};
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        gn[a][b] = fmaf(ga[b], gt[a], gn[a][b]);
        fg[a][b] = fmaf(fa[b], gt[a], fg[a][b]);
      }
  }
  const int i = i0 + cq * 4;
  if (i >= n) return;
#pragma unroll
  for (int a = 0; a < 2; ++a) {
    const int tok = t0 + tp * 2 + a;
    if (tok >= T) break;
    const size_t o = (size_t)tok * n + i;
    const float4 c4 = *reinterpret_cast<const float4 *>(cur + o), m4 = *reinterpret_cast<const float4 *>(msg + o);
    const float ca[4] = {c4.x, c4.y, c4.z, c4.w}, ma[4] = {m4.x, m4.y, m4.z, m4.w};
    float r4[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float f = tanhf(fg[a][b]), th = tanhf(gn[a][b]);
      r4[b] = __fadd_rn(__fsub_rn(ca[b], __fmul_rn(f, ca[b])), __fmul_rn(th, ma[b]));
    }
    *reinterpret_cast<float4 *>(out + o) = make_float4(r4[0], r4[1], r4[2], r4[3]);
  }
}

extern "C" int fcuda_ink2p_ctrl_apply_v2(const float *cur, const float *msg, const float *g, const float *gainW,
                                         const float *forgetW, float *out, int n, int rank, int T) {
  if ((n & 3) || (rank & 3) || ((uintptr_t)cur & 15) || ((uintptr_t)msg & 15) || ((uintptr_t)out & 15) ||
      ((uintptr_t)gainW & 15) || ((uintptr_t)forgetW & 15))
    return fcuda_ink2p_ctrl_apply(cur, msg, g, gainW, forgetW, out, n, rank, T);
  Ink2PCtrlSet cs = {};
  cs.cur[0] = cur; cs.msg[0] = msg; cs.gain[0] = gainW; cs.forget[0] = forgetW;
  cs.g = const_cast<float *>(g); cs.out = out;
  const dim3 grid((n + INK2M_CA_CH - 1) / INK2M_CA_CH, (T + INK2M_CA_TOK - 1) / INK2M_CA_TOK, 1);
  const size_t smem = (size_t)rank * (2 * INK2M_CA_WP + INK2M_CA_GP) * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2m_ctrl_apply, smem);
  if (oe) return oe;
  FCUDA_TS("k_ink2m_ctrl_apply", grid, k_ink2m_ctrl_apply<<<grid, 256, smem, g_stream>>>(cs, n, rank, T));
  return ink2_last_error();
}

extern "C" int fcuda_ink2p_ctrl_batch_v2(const float *const *ptrs, int nj, float *g, float *out, int n, int rank,
                                          int T) {
  if (nj < 1 || nj > INK2M_CTRL_MAX || n % INK2M_CD_KC != 0 || rank % INK2M_CD_RK != 0 || (n & 3) ||
      (rank & 3)) return (int)cudaErrorInvalidValue;
  Ink2PCtrlSet cs = {};
  cs.g = g; cs.out = out;
  for (int j = 0; j < nj; ++j) {
    cs.cur[j] = ptrs[j]; cs.msg[j] = ptrs[3 + j]; cs.down[j] = ptrs[6 + j];
    cs.gain[j] = ptrs[9 + j]; cs.forget[j] = ptrs[12 + j];
    if (((uintptr_t)cs.cur[j] & 15) || ((uintptr_t)cs.msg[j] & 15) || ((uintptr_t)cs.down[j] & 15) ||
        ((uintptr_t)cs.gain[j] & 15) || ((uintptr_t)cs.forget[j] & 15)) return (int)cudaErrorInvalidValue;
  }
  const dim3 dg((T + INK2M_CD_TOK - 1) / INK2M_CD_TOK, rank / INK2M_CD_RK, nj);
  FCUDA_TS("k_ink2m_ctrl_down", dg, k_ink2m_ctrl_down<<<dg, 256, 0, g_stream>>>(cs, n, rank, T));
  const dim3 ag((n + INK2M_CA_CH - 1) / INK2M_CA_CH, (T + INK2M_CA_TOK - 1) / INK2M_CA_TOK, nj);
  const size_t smem = (size_t)rank * (2 * INK2M_CA_WP + INK2M_CA_GP) * sizeof(float);
  const int oe = ink2p_smem_optin((const void *)k_ink2m_ctrl_apply, smem);
  if (oe) return oe;
  FCUDA_TS("k_ink2m_ctrl_apply", ag, k_ink2m_ctrl_apply<<<ag, 256, smem, g_stream>>>(cs, n, rank, T));
  return ink2_last_error();
}

// ---------------------------------------------------------------- short conv
// out[t] depends only on x[t-past..t] (history for t < past) and dyn[t], so a block
// owns 32 channels x 32 tokens with the inputs (plus past leading rows) staged in smem.
// Only the token-tile-0 blocks read the history, so they also write the final history
// after their staging barrier. Per-output fp32 op order is k_ink2p_conv's. x must not
// alias out.
#define INK2M_CV_CH 32
#define INK2M_CV_TOK 32

template <int KN>
__global__ void __launch_bounds__(256) k_ink2m_conv(const float *x, const float *w, const float *dyn, float *hist,
                                                    float *out, const float *resid, int n, int heads, float norm,
                                                    int T) {
  constexpr int kn = KN;
  __shared__ float xs[INK2M_CV_TOK + INK2P_MAX_KN][INK2M_CV_CH];
  const int lc = threadIdx.x & 31, row = threadIdx.x >> 5;
  const int c = blockIdx.x * INK2M_CV_CH + lc, t0 = blockIdx.y * INK2M_CV_TOK;
  const int past = kn - 1;
  const bool cok = c < n;
  for (int r = row; r < INK2M_CV_TOK + past; r += 8) {
    const int tok = t0 - past + r;
    float v = 0.f;
    if (cok && tok < T) v = tok >= 0 ? x[(size_t)tok * n + c] : hist[(size_t)c * past + (tok + past)];
    xs[r][lc] = v;
  }
  __syncthreads();
  if (!cok) return;
  if (blockIdx.y == 0 && row < past) {
    const int idx = T - past + row;
    hist[(size_t)c * past + row] = idx >= 0 ? x[(size_t)idx * n + c] : xs[T + row][lc];
  }
  float wv[KN];
#pragma unroll
  for (int j = 0; j < kn; ++j) wv[j] = w[(size_t)c * kn + j];
  const int hd = dyn ? c / (n / heads) : 0;
  const int dstride = heads * kn;
#pragma unroll
  for (int j = 0; j < INK2M_CV_TOK / 8; ++j) {
    const int tl = row + j * 8, tok = t0 + tl;
    if (tok >= T) break;
    const float xc = xs[past + tl][lc];
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < past; ++q) acc = __fadd_rn(acc, __fmul_rn(xs[tl + q][lc], wv[q]));
    acc = __fadd_rn(acc, __fmul_rn(xc, wv[past]));
    if (dyn) {
      const float *dr = dyn + (size_t)tok * dstride + hd * kn;
#pragma unroll
      for (int lag = 0; lag < kn; ++lag) {
        const float xv = lag > 0 ? xs[past + tl - lag][lc] : xc;
        acc = __fadd_rn(acc, __fmul_rn(dr[lag], xv));
      }
    }
    float o = __fmul_rn(__fadd_rn(xc, acc), norm);
    if (resid) o = __fadd_rn(resid[(size_t)tok * n + c], o);
    out[(size_t)tok * n + c] = o;
  }
}

extern "C" int fcuda_ink2p_conv_v2(const float *x, const float *w, const float *dyn, float *hist, float *out,
                                   const float *resid, int n, int heads, int kn, float norm, int T) {
  if (kn < 1 || kn > INK2P_MAX_KN) return (int)cudaErrorInvalidValue;
  const dim3 grid((n + INK2M_CV_CH - 1) / INK2M_CV_CH, (T + INK2M_CV_TOK - 1) / INK2M_CV_TOK);
  switch (kn) {
#define INK2M_CONV_CASE(K)                                                                                         case K:                                                                                                           FCUDA_TS("k_ink2m_conv", grid, k_ink2m_conv<K><<<grid, 256, 0, g_stream>>>(x, w, dyn, hist, out, resid, n, heads,                                                                               norm, T));                            break;
    INK2M_CONV_CASE(1)
    INK2M_CONV_CASE(2)
    INK2M_CONV_CASE(3)
    INK2M_CONV_CASE(4)
    INK2M_CONV_CASE(5)
    INK2M_CONV_CASE(6)
    INK2M_CONV_CASE(7)
    INK2M_CONV_CASE(8)
#undef INK2M_CONV_CASE
  }
  return ink2_last_error();
}

// ---------------------------------------------------------------- depth mix
// k_ink2p_mix with the states held in registers across both passes (one read of every
// state), 384 threads per token, fp32 reductions and fp32 softmax over the NS states.
#define INK2M_MIX_THREADS 384
#define INK2M_MIX_VEC 2

template <int NS, int NQ>
__device__ void ink2m_mix_body(const Ink2States &st, size_t toff, const float *queries, float *out, size_t qstride,
                               int n) {
  constexpr int NV = NS * (1 + NQ);
  __shared__ float part[INK2M_MIX_THREADS / 32][NV];
  __shared__ float tot[NV];
  __shared__ float scale[NS];
  __shared__ float wts[NQ][NS];
  const int n4 = n >> 2;
  float4 xv[INK2M_MIX_VEC][NS];
  float acc[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) acc[v] = 0.f;
#pragma unroll
  for (int j = 0; j < INK2M_MIX_VEC; ++j) {
    const int i = threadIdx.x + j * INK2M_MIX_THREADS;
    const bool ok = i < n4;
    float4 qv[NQ];
#pragma unroll
    for (int q = 0; q < NQ; ++q)
      qv[q] = ok ? reinterpret_cast<const float4 *>(queries + (size_t)q * n)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      const float4 x = ok ? reinterpret_cast<const float4 *>(st.p[s] + toff)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
      xv[j][s] = x;
      acc[s] = fmaf(x.x, x.x, fmaf(x.y, x.y, fmaf(x.z, x.z, fmaf(x.w, x.w, acc[s]))));
#pragma unroll
      for (int q = 0; q < NQ; ++q)
        acc[NS + s * NQ + q] =
            fmaf(x.x, qv[q].x, fmaf(x.y, qv[q].y, fmaf(x.z, qv[q].z, fmaf(x.w, qv[q].w, acc[NS + s * NQ + q]))));
    }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const float x = ink2_warp_sum(acc[v]);
    if (lane == 0) part[warp][v] = x;
  }
  __syncthreads();
  if (threadIdx.x < NV) {
    float t = 0.f;
    for (int w = 0; w < INK2M_MIX_THREADS / 32; ++w) t += part[w][threadIdx.x];
    tot[threadIdx.x] = t;
  }
  __syncthreads();
  if (threadIdx.x < NS) scale[threadIdx.x] = 1.f / sqrtf(tot[threadIdx.x] / (float)n + 1e-6f);
  __syncthreads();
  if (threadIdx.x < NQ) {
    const int q = threadIdx.x;
    const float inv = 1.f / sqrtf((float)n);
    float sc[NS], mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      sc[s] = __fmul_rn(scale[s], tot[NS + s * NQ + q]) * inv;
      mx = fmaxf(mx, sc[s]);
    }
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      sc[s] = expf(sc[s] - mx);
      sum += sc[s];
    }
#pragma unroll
    for (int s = 0; s < NS; ++s) wts[q][s] = sc[s] / sum;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < INK2M_MIX_VEC; ++j) {
    const int i = threadIdx.x + j * INK2M_MIX_THREADS;
    if (i >= n4) break;
    float4 xs[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s)
      xs[s] = make_float4(__fmul_rn(xv[j][s].x, scale[s]), __fmul_rn(xv[j][s].y, scale[s]),
                          __fmul_rn(xv[j][s].z, scale[s]), __fmul_rn(xv[j][s].w, scale[s]));
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        const float wq = wts[q][s];
        a.x = __fadd_rn(a.x, __fmul_rn(wq, xs[s].x));
        a.y = __fadd_rn(a.y, __fmul_rn(wq, xs[s].y));
        a.z = __fadd_rn(a.z, __fmul_rn(wq, xs[s].z));
        a.w = __fadd_rn(a.w, __fmul_rn(wq, xs[s].w));
      }
      reinterpret_cast<float4 *>(out + (size_t)q * qstride + toff)[i] = a;
    }
  }
}

template <int NQ>
__global__ void __launch_bounds__(INK2M_MIX_THREADS) k_ink2m_mix(Ink2States st, int ns, const float *queries,
                                                                 float *out, int n, int T) {
  const size_t toff = (size_t)blockIdx.x * n, qs = (size_t)T * n;
  switch (ns) {
  case 1: ink2m_mix_body<1, NQ>(st, toff, queries, out, qs, n); break;
  case 2: ink2m_mix_body<2, NQ>(st, toff, queries, out, qs, n); break;
  case 3: ink2m_mix_body<3, NQ>(st, toff, queries, out, qs, n); break;
  case 4: ink2m_mix_body<4, NQ>(st, toff, queries, out, qs, n); break;
  case 5: ink2m_mix_body<5, NQ>(st, toff, queries, out, qs, n); break;
  case 6: ink2m_mix_body<6, NQ>(st, toff, queries, out, qs, n); break;
  case 7: ink2m_mix_body<7, NQ>(st, toff, queries, out, qs, n); break;
  default: ink2m_mix_body<8, NQ>(st, toff, queries, out, qs, n); break;
  }
}

extern "C" int fcuda_ink2p_mix_v2(const float *const *states, int ns, const float *queries, int nq, float *out, int n,
                                  int T) {
  if (ns < 1 || ns > INK2_MAX_STATES || nq < 1 || nq > INK2_MAX_QUERIES || T < 1) return (int)cudaErrorInvalidValue;
  bool vec = (n & 3) == 0 && n <= 4 * INK2M_MIX_VEC * INK2M_MIX_THREADS && ((uintptr_t)queries & 15) == 0 &&
             ((uintptr_t)out & 15) == 0;
  for (int i = 0; i < ns; ++i) vec = vec && ((uintptr_t)states[i] & 15) == 0;
  if (!vec) return fcuda_ink2p_mix(states, ns, queries, nq, out, n, T);
  Ink2States st;
  for (int i = 0; i < INK2_MAX_STATES; ++i) st.p[i] = i < ns ? states[i] : nullptr;
  switch (nq) {
  case 1: FCUDA_TS("k_ink2m_mix", T, k_ink2m_mix<1><<<T, INK2M_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  case 2: FCUDA_TS("k_ink2m_mix", T, k_ink2m_mix<2><<<T, INK2M_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  case 3: FCUDA_TS("k_ink2m_mix", T, k_ink2m_mix<3><<<T, INK2M_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  default: FCUDA_TS("k_ink2m_mix", T, k_ink2m_mix<4><<<T, INK2M_MIX_THREADS, 0, g_stream>>>(st, ns, queries, out, n, T)); break;
  }
  return ink2_last_error();
}
