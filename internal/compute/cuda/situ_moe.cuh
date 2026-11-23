// situ_moe.cuh — the elementwise halves of a routed situ-activation expert MLP
// (inkling2_loop). Every multiply/add is an explicit _rn intrinsic so nvcc cannot
// contract it into an FMA, matching the host forward's rounding order (float32
// scale, float64 activation, float32 weighted sum). CUDA's double exp/tanh are not
// Go's math.Exp/math.Tanh, so an activation can land one float32 ulp apart: the
// fused path is held to the llama.cpp greedy oracle, not to host bit identity.

// k_situ_act: act[p*actStride + j] = situ(guScale[p] * gu[p*guStride + j],
// guScale[p] * gu[p*guStride + width + j]) for j < width, 0 for width <= j < actStride.
__global__ void k_situ_act(const float *gu, int guStride, const float *guScale, float *act,
                           int actStride, int width, int picks, double beta, double linBeta) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int p = blockIdx.y;
  if (p >= picks || j >= actStride) return;
  if (j >= width) {
    act[(size_t)p * actStride + j] = 0.0f;
    return;
  }
  float s = guScale[p];
  double g = (double)__fmul_rn(gu[(size_t)p * guStride + j], s);
  double u = (double)__fmul_rn(gu[(size_t)p * guStride + width + j], s);
  double sig = __ddiv_rn(1.0, __dadd_rn(1.0, exp(-g)));
  double a = __dmul_rn(__dmul_rn(beta, tanh(__ddiv_rn(g, beta))), sig);
  if (linBeta > 0) u = __dmul_rn(linBeta, tanh(__ddiv_rn(u, linBeta)));
  act[(size_t)p * actStride + j] = (float)__dmul_rn(a, u);
}

// k_situ_combine: out[j] = sum over p in order of weight[p] * (dnScale[p] * d[p*dStride + j]).
__global__ void k_situ_combine(const float *d, int dStride, const float *dnScale,
                               const float *weight, float *out, int hidden, int picks) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= hidden) return;
  float acc = 0.0f;
  for (int p = 0; p < picks; p++)
    acc = __fadd_rn(acc, __fmul_rn(weight[p], __fmul_rn(d[(size_t)p * dStride + j], dnScale[p])));
  out[j] = acc;
}

extern "C" void fcuda_situ_act_f32(const float *dGU, int guStride, const float *dGUScale,
                                   float *dAct, int actStride, int width, int picks, double beta,
                                   double linBeta) {
  dim3 grid((actStride + 255) / 256, picks);
  FCUDA_TS("k_situ_act", grid, k_situ_act<<<grid, 256, 0, g_stream>>>(
      dGU, guStride, dGUScale, dAct, actStride, width, picks, beta, linBeta));
}

extern "C" void fcuda_situ_combine_f32(const float *dD, int dStride, const float *dDnScale,
                                       const float *dWeight, float *dOut, int hidden, int picks) {
  FCUDA_TS("k_situ_combine", (hidden + 255) / 256, k_situ_combine<<<(hidden + 255) / 256, 256, 0, g_stream>>>(
      dD, dStride, dDnScale, dWeight, dOut, hidden, picks));
}
