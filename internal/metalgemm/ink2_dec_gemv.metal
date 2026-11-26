// Routed decode products that apply the per-expert scale at the store. Each
// keeps the routed_ kernel's lane mapping (a SIMD group per output row, lane l
// owning the 16-weight groups l, l+32, ...), its fma order and its simd_sum, so
// total*scale[id] is the one multiply of the same sum that the separate scale
// pass makes.
struct Ink2DecShape { uint rows, cols, byte_offset, xstride, scaled, row_bits; };

#define INK2_DEC_ARGS \
    device const uchar *raw [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], device const uint *ids [[buffer(3)]], \
    device const float *scale [[buffer(4)]], constant Ink2DecShape &p [[buffer(5)]], \
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

#define INK2_DEC_STORE(total, id, row) \
    if (lane == 0) y[group.y*p.rows + row] = p.scaled != 0 ? total * scale[id] : total

#define INK2_DEC_ENTRY(NAME) \
kernel void ink2_dec_routed_##NAME(INK2_DEC_ARGS) { \
    uint row = (group.x << p.row_bits) + simd; \
    if (row >= p.rows) return; \
    uint id = ids[group.y]; \
    float total = simd_sum(ink2_dec_dot_##NAME(raw + p.byte_offset, x + group.y*p.xstride, p.cols, \
                                               size_t(id)*p.rows + row, lane)); \
    INK2_DEC_STORE(total, id, row); \
}

#define INK2_DEC_BLOCK(TYPE, BLOCK) \
inline float ink2_dec_dot_##TYPE(device const uchar *raw, device const float *x, uint cols, size_t row, uint lane) { \
    device const block_##TYPE *w = (device const block_##TYPE *)raw + row*(cols/BLOCK); \
    float4 acc = 0; \
    for (uint group = lane; group < cols / 16; group += 32) { \
        float4x4 values; \
        dequantize_##TYPE(w + group / (BLOCK/16), group % (BLOCK/16), values); \
        device const float4 *v = (device const float4 *)(x + 16*group); \
        for (int i = 0; i < 4; i++) acc = fma(values[i], v[i], acc); \
    } \
    return acc.x + acc.y + acc.z + acc.w; \
} \
INK2_DEC_ENTRY(TYPE)

INK2_DEC_BLOCK(q8_0, 32)
INK2_DEC_BLOCK(q3_K, 256)
INK2_DEC_BLOCK(q4_K, 256)
INK2_DEC_BLOCK(q5_K, 256)
INK2_DEC_BLOCK(q6_K, 256)
INK2_DEC_BLOCK(iq2_xs, 256)
INK2_DEC_BLOCK(iq2_s, 256)
INK2_DEC_BLOCK(iq3_xxs, 256)
INK2_DEC_BLOCK(iq3_s, 256)
INK2_DEC_BLOCK(iq4_nl, 32)
INK2_DEC_BLOCK(nvfp4, 64)

#define INK2_DEC_SCALAR(NAME, T) \
inline float ink2_dec_dot_##NAME(device const uchar *raw, device const float *x, uint cols, size_t row, uint lane) { \
    device const T *w = (device const T *)raw + row*cols; \
    float sum = 0; \
    for (uint col = lane; col < cols; col += 32) sum = fma(decode_scalar(w[col]), x[col], sum); \
    return sum; \
} \
INK2_DEC_ENTRY(NAME)

INK2_DEC_SCALAR(f32, float)
INK2_DEC_SCALAR(f16, half)
INK2_DEC_SCALAR(bf16, ushort)

// NVFP4 with the E2M1 codes decoded in registers instead of from the table.
// Twice each magnitude is a nibble of one constant and the scale is halved,
// both exactly, so every product is the table's product.
inline float4 ink2_dec_e2m1(uint4 n) {
    int4 v = int4((uint4(0xC8643210u) >> ((n & 7u) << 2u)) & 15u);
    return float4(select(v, -v, (n & 8u) != 0u));
}

inline float4x4 ink2_dec_nvfp4_group(device const block_nvfp4 *b, uint gi) {
    uint sc = b->d[gi] & 127u;
    uint e = sc >> 3, m = sc & 7u;
    float hd = sc == 127u ? 0.0f : e == 0u ? float(m) / 1024.f : as_type<float>(((e + 119u) << 23) | (m << 20));
    device const packed_uchar4 *q = (device const packed_uchar4 *)b->qs + 2u*gi;
    uint4 b0 = uint4(uchar4(q[0])), b1 = uint4(uchar4(q[1]));
    return float4x4(ink2_dec_e2m1(b0 & 15u) * hd, ink2_dec_e2m1(b1 & 15u) * hd,
                    ink2_dec_e2m1(b0 >> 4u) * hd, ink2_dec_e2m1(b1 >> 4u) * hd);
}

inline float ink2_dec_dot_nvfp4w(device const uchar *raw, device const float *x, uint cols, size_t row, uint lane) {
    device const block_nvfp4 *w = (device const block_nvfp4 *)raw + row*(cols/64);
    float4 acc = 0;
    for (uint g = lane; g < cols / 16; g += 32) {
        float4x4 values = ink2_dec_nvfp4_group(w + g/4, g%4);
        device const float4 *v = (device const float4 *)(x + 16*g);
        for (int i = 0; i < 4; i++) acc = fma(values[i], v[i], acc);
    }
    return acc.x + acc.y + acc.z + acc.w;
}
INK2_DEC_ENTRY(nvfp4w)

// The register decode two steps at a time: both groups' loads issue before
// the first fma, which still runs in step order.
inline float ink2_dec_dot_nvfp4u(device const uchar *raw, device const float *x, uint cols, size_t row, uint lane) {
    device const block_nvfp4 *w = (device const block_nvfp4 *)raw + row*(cols/64);
    uint n = cols / 16;
    float4 acc = 0;
    for (uint g = lane; g < n; g += 64) {
        uint h = g + 32;
        bool more = h < n;
        float4x4 a = ink2_dec_nvfp4_group(w + g/4, g%4);
        float4x4 b = ink2_dec_nvfp4_group(w + (more ? h : g)/4, (more ? h : g)%4);
        device const float4 *va = (device const float4 *)(x + 16*g);
        device const float4 *vb = (device const float4 *)(x + 16*(more ? h : g));
        float4 xa0 = va[0], xa1 = va[1], xa2 = va[2], xa3 = va[3];
        float4 xb0 = vb[0], xb1 = vb[1], xb2 = vb[2], xb3 = vb[3];
        acc = fma(a[0], xa0, acc); acc = fma(a[1], xa1, acc); acc = fma(a[2], xa2, acc); acc = fma(a[3], xa3, acc);
        if (more) {
            acc = fma(b[0], xb0, acc); acc = fma(b[1], xb1, acc); acc = fma(b[2], xb2, acc); acc = fma(b[3], xb3, acc);
        }
    }
    return acc.x + acc.y + acc.z + acc.w;
}
INK2_DEC_ENTRY(nvfp4u)

// The register decode with the slot's activation staged once in threadgroup
// memory for all the threadgroup's rows.
#define INK2_DEC_XMAX 4096
kernel void ink2_dec_routed_nvfp4t(INK2_DEC_ARGS, uint2 at [[thread_position_in_threadgroup]],
                                   uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float4 xs[INK2_DEC_XMAX / 4];
    device const float4 *xv = (device const float4 *)(x + group.y*p.xstride);
    for (uint i = at.x; i < p.cols / 4; i += threads.x) xs[i] = xv[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = (group.x << p.row_bits) + simd;
    if (row >= p.rows) return;
    uint id = ids[group.y];
    device const block_nvfp4 *w = (device const block_nvfp4 *)(raw + p.byte_offset) + (size_t(id)*p.rows + row)*(p.cols/64);
    float4 acc = 0;
    for (uint g = lane; g < p.cols / 16; g += 32) {
        float4x4 values = ink2_dec_nvfp4_group(w + g/4, g%4);
        for (int i = 0; i < 4; i++) acc = fma(values[i], xs[4*g + i], acc);
    }
    float total = simd_sum(acc.x + acc.y + acc.z + acc.w);
    INK2_DEC_STORE(total, id, row);
}

// Routed products with the column count fixed at compile time and RS rows per
// SIMD group. Every load is named before its fma instead of waiting on a
// runtime trip count, the rows share each step's activation loads, and each
// row keeps lane l's groups l, l+32, ... in step order, so every sum is the
// routed_ kernel's. NVFP4 has none: it timed no faster this way.
#define INK2_DEC_ROWS(COLS_PER_ROW) \
    uint row0 = ((group.x << p.row_bits) + simd) * RS; \
    if (row0 >= p.rows) return; \
    uint id = ids[group.y]; \
    device const float4 *xv = (device const float4 *)(x + group.y*p.xstride); \
    constexpr uint G = COLS / 16, STEPS = (G + 31) / 32, TAIL = G % 32; \
    bool tail = TAIL == 0 || lane < TAIL; \
    size_t rowAt[RS]; \
    for (uint r = 0; r < RS; r++) rowAt[r] = (size_t(id)*p.rows + min(row0 + r, p.rows - 1)) * (COLS_PER_ROW)

#define INK2_DEC_ROWS_STORE \
    for (uint r = 0; r < RS; r++) { \
        float total = simd_sum(acc[r].x + acc[r].y + acc[r].z + acc[r].w); \
        if (row0 + r < p.rows) INK2_DEC_STORE(total, id, row0 + r); \
    }

#define INK2_DEC_CBLOCK(TYPE, BLOCK) \
template <uint COLS, uint RS> \
kernel void ink2_dec_routed_##TYPE##c(INK2_DEC_ARGS) { \
    INK2_DEC_ROWS(COLS / BLOCK); \
    device const block_##TYPE *w = (device const block_##TYPE *)(raw + p.byte_offset); \
    float4 acc[RS]; \
    for (uint r = 0; r < RS; r++) acc[r] = 0; \
    for (uint s = 0; s < STEPS; s++) { \
        if (s + 1 == STEPS && !tail) break; \
        uint g = lane + 32*s; \
        float4 v0 = xv[4*g], v1 = xv[4*g + 1], v2 = xv[4*g + 2], v3 = xv[4*g + 3]; \
        for (uint r = 0; r < RS; r++) { \
            float4x4 values; \
            dequantize_##TYPE(w + rowAt[r] + g / (BLOCK/16), g % (BLOCK/16), values); \
            acc[r] = fma(values[0], v0, acc[r]); acc[r] = fma(values[1], v1, acc[r]); \
            acc[r] = fma(values[2], v2, acc[r]); acc[r] = fma(values[3], v3, acc[r]); \
        } \
    } \
    INK2_DEC_ROWS_STORE \
}

INK2_DEC_CBLOCK(iq2_xs, 256)
INK2_DEC_CBLOCK(iq3_xxs, 256)
INK2_DEC_CBLOCK(iq3_s, 256)
INK2_DEC_CBLOCK(iq4_nl, 32)

#define INK2_DEC_C(TYPE, COLS, RS) \
template [[host_name("ink2_dec_routed_" #TYPE "c" #COLS "r" #RS)]] kernel void ink2_dec_routed_##TYPE##c<COLS, RS>(INK2_DEC_ARGS);

INK2_DEC_C(iq2_xs, 2304, 2)
INK2_DEC_C(iq3_xxs, 2304, 2)
INK2_DEC_C(iq3_s, 2304, 2)
INK2_DEC_C(iq4_nl, 1152, 2)
