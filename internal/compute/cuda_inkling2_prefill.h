#ifndef FAK_CUDA_INKLING2_PREFILL_H
#define FAK_CUDA_INKLING2_PREFILL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* inkling2_loop batched prompt prefill ops (cuda/inkling2_prefill.cuh): token t of a
 * T-token chunk sits at position pos0 + t; activations are [T][width] rows. */
int fcuda_ink2p_mix(const float *const *states, int ns, const float *queries, int nq, float *out, int n, int T);
int fcuda_ink2p_gemm_f32(const float *W, const float *X, float *Y, int rows, int cols, int T);
int fcuda_ink2p_ctrl_down(const float *cur, const float *msg, const float *W, float *g, int n, int rank, int T);
int fcuda_ink2p_ctrl_apply(const float *cur, const float *msg, const float *g, const float *gainW,
    const float *forgetW, float *out, int n, int rank, int T);
int fcuda_ink2p_conv(const float *x, const float *w, const float *dyn, float *hist, float *out,
    const float *resid, int n, int heads, int kn, float norm, int T);
int fcuda_ink2p_qkv_post(float *q, float *r, const float *k, const float *v, const float *qclip,
    const float *rclip, float *kcache, float *vcache, int heads, int kvHeads, int headDim, int relLen,
    float qkClip, float relClip, float eps, int global, float alpha, float floorTok, int cap, int pos0, int T);
int fcuda_ink2p_attend(const float *q, const float *r, const float *sinks, const float *relBias,
    const float *kcache, const float *vcache, float *out, int heads, int kvHeads, int headDim, int relLen,
    int extent, int window, int cap, float scale, int pos0, int T);
int fcuda_ink2p_route(const float *logits, const float *bias, int *ids, float *wts, int nexp, int k,
    float wscale, int T);
int fcuda_ink2p_combine(const float *down, const float *wts, float *out, int k, int n, int T);
int fcuda_ink2p_bucket(const int *ids, int npairs, int nexp, int *counts, int *offsets, int *order, int *tiles,
    int *ntiles);
int fcuda_ink2p_gather_pairs(const int *ids, const int *order, const float *X, int xDiv, int xStride,
    int *sortedIds, float *sortedX, int width, int npairs);
int fcuda_ink2p_scatter_pairs(const int *order, const float *sortedY, float *Y, int rows, int npairs);
int fcuda_ink2p_max_tiles(int npairs, int nexp);
int fcuda_ink2p_expert_nvfp4(const uint8_t *W, const uint8_t *scales, const int *offsets, const int *order,
    const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride, float *Y, int rows,
    int in, const float *escale);
int fcuda_ink2p_expert_iq(int fmt, const uint8_t *W, const int *offsets, const int *order, const int *tiles,
    const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride, float *Y, int rows, int in,
    const float *escale);

#ifdef __cplusplus
}
#endif
#endif
