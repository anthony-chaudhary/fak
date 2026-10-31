// Qualification-only checked adapter. Reuse the existing optional terminal
// stream/allocator lifetime seam, not a new pool or pointer-identity cache.
static unsigned long long g_q4k_qkv_qualification_operations = 0;
extern "C" unsigned long long fcuda_q4k_qkv_qualification_operations(void) {
    return g_q4k_qkv_qualification_operations;
}
extern "C" int fcuda_q4k_qkv_qualification_state(void) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
    if (g_capture_open) return FCUDA_Q4K_MMVQ_REFUSED;
    return v41_checked_stream_state();
#else
    return FCUDA_Q4K_MMVQ_REFUSED;
#endif
}
extern "C" int fcuda_q4k_qkv_qualification_finish(void) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
    return v41_checked_stream_finish();
#else
    return -3;
#endif
}
extern "C" int fcuda_q4k_qkv_qualification_f32(
    const uint8_t *wQ, const uint8_t *wK, const uint8_t *wV,
    const float *x, float *yQ, float *yK, float *yV,
    int outQ, int outK, int outV, int in, void *workspace, size_t workspaceBytes) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
    int status = fcuda_q4k_qkv_qualification_state();
    if (status != 0) return status;
    const uint8_t *weights[3] = {wQ, wK, wV};
    float *outputs[3] = {yQ, yK, yV};
    const int widths[3] = {outQ, outK, outV};
    if (!fak_q4k_shared_input::valid_ranges(weights, x, outputs, widths, in, workspace, workspaceBytes))
        return FCUDA_Q4K_MMVQ_REFUSED;
    // Exact live allocation bases only: no foreign device, managed memory or
    // interior alias can be smuggled through a forged Go tensor descriptor.
    status = v41_checked_stream_owned_bytes(x, (uint64_t)in * sizeof(float));
    if (status != 0) return status;
    status = v41_checked_stream_owned_bytes(workspace, (uint64_t)(in / 32) * 36);
    if (status != 0) return status;
    for (int i = 0; i < 3; ++i) {
        status = v41_checked_stream_owned_bytes(weights[i], (uint64_t)widths[i] * (in / 256) * 144);
        if (status != 0) return status;
        status = v41_checked_stream_owned_bytes(outputs[i], (uint64_t)widths[i] * sizeof(float));
        if (status != 0) return status;
    }
    status = fcuda_q4k_qkv_shared_input_f32(wQ, wK, wV, x, yQ, yK, yV,
                                         outQ, outK, outV, in, workspace, workspaceBytes);
    // Even an immediate launch failure can follow already enqueued reads. A
    // failed checked finish closes the existing global allocator/free seam,
    // retaining every outstanding allocation, not only these eight operands.
    const int completed = v41_checked_stream_finish();
    if (completed == 0 && status == 0) ++g_q4k_qkv_qualification_operations;
    return completed != 0 ? completed : status;
#else
    return FCUDA_Q4K_MMVQ_REFUSED;
#endif
}
