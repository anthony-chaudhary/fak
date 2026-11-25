#ifndef FAK_CUDA_INKLING2_PREFILL_H
#define FAK_CUDA_INKLING2_PREFILL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* inkling2_loop batched prompt prefill ops (cuda/inkling2_prefill.cuh): token t of a
 * T-token chunk sits at position pos0 + t; activations are [T][width] rows. */
int fcuda_ink2p_mix(const float *const *states, int ns, const float *queries, int nq, float *out, int n, int T);
/* fp32 prefill variants (cuda/inkling2_prefill_misc.cuh); each falls back to its original for shapes it does not cover. */
int fcuda_ink2p_mix_v2(const float *const *states, int ns, const float *queries, int nq, float *out, int n, int T);
int fcuda_ink2p_rms(const float *x, const float *w, float *y, int rows, int n, float eps);
int fcuda_ink2p_situ_v2(const float *gu, float *act, int k, int width, int actStride, double beta, double lb);
int fcuda_ink2p_ctrl_down_v2(const float *cur, const float *msg, const float *W, float *g, int n, int rank, int T);
int fcuda_ink2p_ctrl_apply_v2(const float *cur, const float *msg, const float *g, const float *gainW,
                              const float *forgetW, float *out, int n, int rank, int T);
int fcuda_ink2p_conv_v2(const float *x, const float *w, const float *dyn, float *hist, float *out,
                        const float *resid, int n, int heads, int kn, float norm, int T);
int fcuda_ink2p_gemm_f32(const float *W, const float *X, float *Y, int rows, int cols, int T);
int fcuda_ink2p_ctrl_down(const float *cur, const float *msg, const float *W, float *g, int n, int rank, int T);
int fcuda_ink2p_ctrl_apply(const float *cur, const float *msg, const float *g, const float *gainW,
    const float *forgetW, float *out, int n, int rank, int T);
int fcuda_ink2p_conv(const float *x, const float *w, const float *dyn, float *hist, float *out,
    const float *resid, int n, int heads, int kn, float norm, int T);
int fcuda_ink2p_conv_log(const float *x, const float *w, const float *dyn, float *hist, float *log,
    float *out, const float *resid, int n, int heads, int kn, float norm, int T);
int fcuda_ink2p_combine_conv_log(const float *down, const float *wts, int k, const float *w, float *hist, float *log,
    float *out, const float *resid, int n, int kn, float norm, int T);
int fcuda_ink2p_conv_commit(float *hist, const float *log, int channels, int past, int L, int keep);
int fcuda_ink2p_conv_commit_dev(float *hist, const float *log, int channels, int past, int L, const int *keepDev);
int fcuda_ink2p_qkv_post_dev(float *q, float *r, const float *k, const float *v, const float *qclip,
    const float *rclip, float *kcache, float *vcache, int heads, int kvHeads, int headDim, int relLen,
    float qkClip, float relClip, float eps, int global, float alpha, float floorTok, int cap, const int *posDev, int T);
int fcuda_ink2p_attend_dev(const float *q, const float *r, const float *sinks, const float *relBias,
    const float *kcache, const float *vcache, float *out, int heads, int kvHeads, int headDim, int relLen,
    int extent, int window, int cap, float scale, const int *posDev, int T);
int fcuda_ink2_graph_end_slot(int slot);
/* DSpark draft ops (cuda/inkling2_dspark.cuh). */
int fcuda_ink2d_fuse(const float *const *f, const float *a, int taps, const float *w, float *out, int rows, int n,
    float eps);
int fcuda_ink2d_qk(const float *x, const float *w, float *dst, int rows, int heads, int dim, const int *posDev,
    int cap, int ring, float eps, double theta);
int fcuda_ink2d_ring_store(const float *src, float *ring, int rows, int width, const int *posDev, int cap);
int fcuda_ink2d_attend(const float *q, const float *kb, const float *vb, const float *kc, const float *vc, float *out,
    int rows, int heads, int kvHeads, int dim, const int *posDev, int window, int cap);
int fcuda_ink2d_silu_mul(const float *g, const float *u, float *out, int n);
int fcuda_ink2v_q5k_rows(const unsigned char *W, const float *X, float *Y, int out, int in, int rows);
int fcuda_ink2p_ctrl_multi(const float *const *cur, const float *const *msg, const float *const *W,
    const float *const *gainW, const float *const *forgetW, float *const *g, float *const *out, int J, int n,
    int rank, int T);
int fcuda_ink2v_mix(const float *const *states, int ns, const float *queries, int nq, const float *const *norms,
    float eps, float *out, int n, int T);
int fcuda_ink2v_q6k_rows(const unsigned char *W, const float *X, float *Y, int out, int in, int rows);
int fcuda_ink2p_conv_pair_log(const float *const *x, const float *const *w, const float *const *dyn,
    float *const *hist, float *const *log, float *const *out, const float *norm, int n, int heads, int kn, int T);
int fcuda_ink2v_repeat_rows(const float *x, float *out, int n, int rows, int k);
int fcuda_ink2v_q8_rows(const int8_t *W, const float *Wscale, const float *X, const float *gate, void *q, int quantize,
    float *Y, int out, int in, int rows, int block);
int fcuda_ink2d_topm(const float *logits, int rows, int rowStride, int vocab, int M, float *candV, int *candI,
    int *out);
int fcuda_ink2d_markov_in(const float *state, const unsigned short *w1, const int *anchor, const int *idx, int step,
    const float *h, const float *frnnRing, const int *posDev, int cap, int rank, int hidden, float *in);
int fcuda_ink2d_markov_gate(const float *raw, const float *bias, float *state, float *latent, int rank, int step);
int fcuda_ink2d_markov_pick_q5k(const unsigned char *W, int in, const float *base, const int *cand, int M,
    const float *z, const float *zb, int *idx, int step);
int fcuda_ink2_graph_launch_slot(int slot);
int fcuda_ink2p_qkv_post(float *q, float *r, const float *k, const float *v, const float *qclip,
    const float *rclip, float *kcache, float *vcache, int heads, int kvHeads, int headDim, int relLen,
    float qkClip, float relClip, float eps, int global, float alpha, float floorTok, int cap, int pos0, int T);
/* fp16 tensor-core tiled attention (cuda/inkling2_prefill_attn.cuh); same contract, falls back to the original off its fast shape. */
int fcuda_ink2p_attend_v2(const float *q, const float *r, const float *sinks, const float *relBias,
    const float *kcache, const float *vcache, float *out, int heads, int kvHeads, int headDim, int relLen,
    int extent, int window, int cap, float scale, int pos0, int T);
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
/* int8 tensor-core grouped NVFP4 GEMM (cuda/inkling2_prefill_nvfp4.cuh); same contract. */
int fcuda_ink2p_expert_nvfp4_v2(const uint8_t *W, const uint8_t *scales, const int *offsets, const int *order,
    const int *tiles, const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride, float *Y, int rows,
    int in, const float *escale);
/* int8 tensor-core grouped IQ/Q3_K GEMM (cuda/inkling2_prefill_iq.cuh); same contract. */
int fcuda_ink2p_expert_iq_v2(int fmt, const uint8_t *W, const int *offsets, const int *order, const int *tiles,
    const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride, float *Y, int rows, int in,
    const float *escale);
int fcuda_ink2p_expert_iq(int fmt, const uint8_t *W, const int *offsets, const int *order, const int *tiles,
    const int *ntiles, int maxTiles, const float *X, int xDiv, int xStride, float *Y, int rows, int in,
    const float *escale);

#ifdef __cplusplus
}
#endif
#endif
