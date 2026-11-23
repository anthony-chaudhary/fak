// nvfp4_warp_gemv.cuh — warp-per-row NVFP4 W4A16 decode GEMV. Same per-element math as
// k_nvfp4_w4a16_f32 (E2M1 x E4M3FN block scale, BF16-RNE activation, _rn products), but the
// K reduction is split across the 32 lanes and tree-summed, so it does NOT reproduce the
// reference kernel's serial order. Each lane owns whole 16-value scale blocks: one 8-byte
// packed load, one scale byte and four float4 activation loads per block, coalesced across the
// warp. Needs in % 16 == 0, an 8-byte-aligned packed row base and a 16-byte-aligned X row.

__global__ void k_nvfp4_w4a16_warp_f32(const uint8_t *Packed, const uint8_t *Scales,
                                       const float *X, float *Y, int out, int in,
                                       float globalScale) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int p = blockIdx.y;
  if (row >= out) return;
  const uint8_t *wrow = Packed + (size_t)row * (size_t)(in / 2);
  const uint8_t *srow = Scales + (size_t)row * (size_t)(in / 16);
  const float *xrow = X + (size_t)p * (size_t)in;
  float sum = 0.0f;
  for (int b = lane; b < in / 16; b += 32) {
    const uint2 q = *reinterpret_cast<const uint2 *>(wrow + (size_t)b * 8u);
    const float s = nv4_e4m3fn(srow[b]);
    const float4 *xv = reinterpret_cast<const float4 *>(xrow + (size_t)b * 16u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 v = xv[i];
      const uint32_t word = i < 2 ? q.x : q.y;
      const float xs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int e = (i * 4 + j) & 7;
        const float weight = __fmul_rn(nv4_e2m1((word >> (4 * e)) & 0xfu), s);
        sum = __fadd_rn(sum, __fmul_rn(weight, nv4_bf16_rne(xs[j])));
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
  if (lane == 0) Y[(size_t)p * (size_t)out + row] = __fmul_rn(sum, globalScale);
}

extern "C" int fcuda_nvfp4_w4a16_warp_f32(const uint8_t *dPacked, const uint8_t *dScales,
                                          const float *dX, float *dY, int out, int in, int P,
                                          float globalScale) {
  if (!dPacked || !dScales || !dX || !dY || out <= 0 || in <= 0 || P <= 0 || P > 65535 ||
      (in & 15) != 0 || ((uintptr_t)dPacked & 7u) != 0 || ((uintptr_t)dX & 15u) != 0 ||
      globalScale < 0.0f || !isfinite(globalScale))
    return (int)cudaErrorInvalidValue;
  dim3 grid((unsigned int)((out + 7) / 8), (unsigned int)P);
  FCUDA_TS("k_nvfp4_w4a16_warp_f32", grid, k_nvfp4_w4a16_warp_f32<<<grid, 256, 0, g_stream>>>(
      dPacked, dScales, dX, dY, out, in, globalScale));
  return (int)cudaGetLastError();
}
