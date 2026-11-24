// A four-vector register tile reuses each decoded weight across prompt rows.
struct PackedPanel {
    uint count;
    uint input[4], output[4];
};

#define PANEL_ARGS PACKED_ARGS, constant PackedPanel &panel [[buffer(4)]]

#define PACKED_PANEL_KERNEL(TYPE, BLOCK) \
kernel void panel_##TYPE(PANEL_ARGS) { \
    row = row * 4 + simd; \
    if (row >= shape.rows) return; \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + shape.byte_offset) + \
            size_t(row) * (shape.cols / BLOCK); \
    float4 acc[4] = {}; \
    for (uint group = lane; group < shape.cols / 16; group += 32) { \
        float4x4 values; \
        dequantize_##TYPE(w + group / (BLOCK/16), group % (BLOCK/16), values); \
        for (uint p = 0; p < panel.count; p++) { \
            device const float4 *v = (device const float4 *)(x + panel.input[p] + 16*group); \
            for (int i = 0; i < 4; i++) acc[p] = fma(values[i], v[i], acc[p]); \
        } \
    } \
    for (uint p = 0; p < panel.count; p++) { \
        float total = simd_sum(acc[p].x + acc[p].y + acc[p].z + acc[p].w); \
        if (lane == 0) y[panel.output[p]+row] = total; \
    } \
}

PACKED_PANEL_KERNEL(q8_0, 32)
PACKED_PANEL_KERNEL(q3_K, 256)
PACKED_PANEL_KERNEL(q4_K, 256)
PACKED_PANEL_KERNEL(q5_K, 256)
PACKED_PANEL_KERNEL(q6_K, 256)
PACKED_PANEL_KERNEL(iq2_xs, 256)
PACKED_PANEL_KERNEL(iq2_s, 256)
PACKED_PANEL_KERNEL(iq3_xxs, 256)
PACKED_PANEL_KERNEL(iq3_s, 256)
PACKED_PANEL_KERNEL(iq4_nl, 32)
PACKED_PANEL_KERNEL(nvfp4, 64)

#define SCALAR_PANEL_KERNEL(NAME, T) \
kernel void panel_##NAME(PANEL_ARGS) { \
    row = row * 4 + simd; \
    if (row >= shape.rows) return; \
    device const T *w = (device const T *)(raw + shape.byte_offset) + size_t(row) * shape.cols; \
    float acc[4] = {}; \
    for (uint col = lane; col < shape.cols; col += 32) { \
        float value = decode_scalar(w[col]); \
        for (uint p = 0; p < panel.count; p++) \
            acc[p] = fma(value, x[panel.input[p]+col], acc[p]); \
    } \
    for (uint p = 0; p < panel.count; p++) { \
        float total = simd_sum(acc[p]); \
        if (lane == 0) y[panel.output[p]+row] = total; \
    } \
}

SCALAR_PANEL_KERNEL(f32, float)
SCALAR_PANEL_KERNEL(f16, half)
SCALAR_PANEL_KERNEL(bf16, ushort)
