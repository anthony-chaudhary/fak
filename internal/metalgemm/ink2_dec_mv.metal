// Scalar-weight decode matvecs with the mv_ kernels' arithmetic: a SIMD group
// per output row, lane l owning columns l, l+32, ..., the same fma order per lane
// and the same simd_sum. The column count is a template argument, so each lane's
// steps are named at compile time and a chunk of U steps issues all of its loads
// before the first of its fmas; mv_'s runtime-bounded loop issues one at a time.
struct Ink2MvShape { uint rows, byte_offset, row_bits; };

#define INK2_MV_ARGS \
    device const uchar *raw [[buffer(0)]], device const float *x [[buffer(1)]], \
    device float *y [[buffer(2)]], constant Ink2MvShape &p [[buffer(3)]], \
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

#define INK2_MV_SCALAR(NAME, T) \
template <uint COLS, uint U> \
inline float ink2_mv_dot_##NAME(device const uchar *raw, device const float *x, size_t row, uint lane) { \
    static_assert(COLS % 32 == 0, "every lane takes the same number of columns"); \
    constexpr uint S = COLS / 32; \
    static_assert(S % U == 0, "the chunks tile each lane's steps"); \
    device const T *w = (device const T *)raw + row*COLS; \
    float sum = 0; \
    for (uint s0 = 0; s0 < S; s0 += U) { \
        float a[U], b[U]; \
        for (uint u = 0; u < U; u++) { \
            uint col = lane + 32*(s0 + u); \
            a[u] = decode_scalar(w[col]); \
            b[u] = x[col]; \
        } \
        for (uint u = 0; u < U; u++) { \
            sum = fma(a[u], b[u], sum); \
        } \
    } \
    return sum; \
}

INK2_MV_SCALAR(f32, float)
INK2_MV_SCALAR(f16, half)
INK2_MV_SCALAR(bf16, ushort)

#define INK2_MV_KERNEL(NAME) \
template <uint COLS, uint U> \
kernel void ink2_dec_mv_##NAME(INK2_MV_ARGS) { \
    uint row = (group << p.row_bits) + simd; \
    if (row >= p.rows) return; \
    float total = simd_sum(ink2_mv_dot_##NAME<COLS, U>(raw + p.byte_offset, x, row, lane)); \
    if (lane == 0) y[row] = total; \
}

INK2_MV_KERNEL(f32)
INK2_MV_KERNEL(f16)
INK2_MV_KERNEL(bf16)

typedef decltype(ink2_dec_mv_f32<2304, 24>) ink2_dec_mv_t;

#define INK2_MV_AT(NAME, COLS) \
template [[host_name("ink2_dec_mv_" #NAME "_" #COLS)]] kernel ink2_dec_mv_t ink2_dec_mv_##NAME<COLS, 24>;

#define INK2_MV_COLS(COLS) INK2_MV_AT(f32, COLS) INK2_MV_AT(f16, COLS) INK2_MV_AT(bf16, COLS)

INK2_MV_COLS(2304)
INK2_MV_COLS(3072)
