// Each lane decodes sixteen contiguous weights directly into registers.
// Four simdgroups share a threadgroup and compute independent output rows.
struct PackedShape { uint rows, cols, byte_offset; };

#define PACKED_ARGS \
    device const uchar *raw [[buffer(0)]], \
    device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], \
    constant PackedShape &shape [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

#define ROUTED_ARGS \
    device const uchar *raw [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], constant PackedShape &shape [[buffer(3)]], \
    device const uint *ids [[buffer(4)]], constant uint &xstride [[buffer(5)]], \
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

#define MV_ENTRY_POINTS(TYPE) \
kernel void mv_##TYPE(PACKED_ARGS) { \
    row = row*4 + simd; \
    if (row < shape.rows) row_##TYPE(raw+shape.byte_offset, x, y+row, shape.cols, row, lane); \
} \
kernel void routed_##TYPE(ROUTED_ARGS) { \
    uint row = group.x*4 + simd; \
    if (row >= shape.rows) return; \
    size_t wrow = size_t(ids[group.y])*shape.rows + row; \
    row_##TYPE(raw+shape.byte_offset, x + group.y*xstride, y + group.y*shape.rows + row, shape.cols, wrow, lane); \
}

#define PACKED_KERNEL(TYPE, BLOCK) \
inline void row_##TYPE(device const uchar *raw, device const float *x, device float *y, uint cols, size_t row, uint lane) { \
    device const block_##TYPE *w = (device const block_##TYPE *)raw + row*(cols/BLOCK); \
    float4 acc = 0; \
    for (uint group = lane; group < cols / 16; group += 32) { \
        float4x4 values; \
        dequantize_##TYPE(w + group / (BLOCK/16), group % (BLOCK/16), values); \
        device const float4 *v = (device const float4 *)(x + 16*group); \
        for (int i = 0; i < 4; i++) acc = fma(values[i], v[i], acc); \
    } \
    float total = simd_sum(acc.x + acc.y + acc.z + acc.w); \
    if (lane == 0) *y = total; \
} \
MV_ENTRY_POINTS(TYPE)

PACKED_KERNEL(q8_0, 32)
PACKED_KERNEL(q3_K, 256)
PACKED_KERNEL(q4_K, 256)
PACKED_KERNEL(q5_K, 256)
PACKED_KERNEL(q6_K, 256)
PACKED_KERNEL(iq2_xs, 256)
PACKED_KERNEL(iq2_s, 256)
PACKED_KERNEL(iq3_xxs, 256)
PACKED_KERNEL(iq3_s, 256)
PACKED_KERNEL(iq4_nl, 32)
PACKED_KERNEL(nvfp4, 64)

inline float decode_scalar(float v) { return v; }
inline float decode_scalar(half v) { return float(v); }
inline float decode_scalar(ushort v) { return as_type<float>(uint(v) << 16); }

#define SCALAR_KERNEL(NAME, T) \
inline void row_##NAME(device const uchar *raw, device const float *x, device float *y, uint cols, size_t row, uint lane) { \
    device const T *w = (device const T *)raw + row*cols; \
    float sum = 0; \
    for (uint col = lane; col < cols; col += 32) sum = fma(decode_scalar(w[col]), x[col], sum); \
    float total = simd_sum(sum); \
    if (lane == 0) *y = total; \
} \
MV_ENTRY_POINTS(NAME)

SCALAR_KERNEL(f32, float)
SCALAR_KERNEL(f16, half)
SCALAR_KERNEL(bf16, ushort)
