package compute

// Ink2Ref addresses a float32 element offset inside a device tensor. A zero Ref (no
// tensor) passes a null pointer, for the optional operands of an op.
type Ink2Ref struct {
	T   Tensor
	Off int
}

// Ink2At returns a reference to element off of t.
func Ink2At(t Tensor, off int) Ink2Ref { return Ink2Ref{T: t, Off: off} }

// Ink2AttnParams are the per-layer attention scalars of the inkling2_loop forward.
type Ink2AttnParams struct {
	Heads, KVHeads, HeadDim, RelLen int
	QKClip, RelClip, Eps            float32
	Global                          bool
	Alpha, Floor, Scale             float32
}

// Ink2Situ selects the fused SITU activation of an expert input: x holds Width gate
// values followed by Width up values per slot.
type Ink2Situ struct {
	Width            int
	Beta, LinearBeta float64
}

// Ink2Controls are the operands of up to four loop controllers run together:
// out_j = cur_j - tanh(F_j g_j)*cur_j + tanh(G_j g_j)*msg_j, g_j = Down_j [rms(cur_j), rms(msg_j)].
type Ink2Controls struct {
	Cur, Msg, Down, Gain, Forget []Ink2Ref
}

// Ink2KVConv is the k and v short conv of one layer visit: tap weights, optional
// dynamic taps, the slot history rows, kernel width and output scales.
type Ink2KVConv struct {
	KW, VW, KDyn, VDyn, KHist, VHist Ink2Ref
	Kn                               int
	KNorm, VNorm                     float32
}

// Inkling2Device is the op set of the resident inkling2_loop decode forward: every
// op enqueues on the device stream and nothing is read back except through the
// explicit Ink2Read* fences. Buffers come from Ink2Alloc and live until Free.
type Inkling2Device interface {
	Ink2Alloc(n int) (Tensor, error)
	Ink2Write(dst Ink2Ref, src []float32) error
	Ink2Read(src Ink2Ref, n int) ([]float32, error)
	Ink2ReadInts(src Ink2Ref, n int) ([]int32, error)

	// Ink2MatVec is y = W·x for one resident dense weight of any MatMul dtype.
	Ink2MatVec(w Tensor, x, y Ink2Ref) error
	// Ink2MatVecQ8 is Ink2MatVec for a Q8_0 weight with x's quantized codes kept in q
	// (in/4 + in/32 floats); quantize=false reuses codes an earlier call on the same x
	// wrote. gate (optional, needs quantize) multiplies sigmoid(gate) into x as it is
	// quantized. ok is false (nothing launched) for any other dtype.
	Ink2MatVecQ8(w Tensor, x, gate, q Ink2Ref, quantize bool, y Ink2Ref) (ok bool, err error)
	// Ink2ExpertMatVec multiplies k routed slots through one stacked expert tensor
	// whose rows are [experts*expertRows, in]; ids holds the k expert indices on
	// the device. scale (optional) is a per-expert output multiplier; situ (optional)
	// makes x each slot's [gate | up] row and multiplies its SITU activation instead.
	// ok is false (nothing launched) for a dtype without an indexed kernel.
	Ink2ExpertMatVec(stack Tensor, expertRows int, ids, x Ink2Ref, xStride int, y Ink2Ref, k int, scale Ink2Ref, situ *Ink2Situ) (ok bool, err error)

	Ink2RMS(x, w, y Ink2Ref, rows, n int, eps float32) error
	Ink2CtrlPair(cur, msg, pair Ink2Ref, n int) error
	// Ink2Mix writes nq depth mixes of states; with norms (one weight per query, a
	// zero Ref for none) each mix row is RMS-normalized (eps) before it is written.
	Ink2Mix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n int) error
	// Ink2Control runs len(ctl.Cur) loop controllers of one layer visit into the rows of
	// out ([j][n]) through the rank-sized scratch g ([j][rank]); with norms each output
	// row is then RMS-normalized (eps) by its weight.
	Ink2Control(ctl Ink2Controls, g, out Ink2Ref, n, rank int, norms []Ink2Ref, eps float32) error
	Ink2Conv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error
	// Ink2CombineConv is Ink2Conv without dynamic taps whose input is Ink2Combine(down, wts, k).
	Ink2CombineConv(down, wts Ink2Ref, k int, w, hist, out, resid Ink2Ref, n, kn int, norm float32) error
	// Ink2QKVPost clips q/r/k, normalizes v and appends k/v to the slot cache; cv
	// (optional) first runs the k and v short convs on the raw projections.
	Ink2QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error
	Ink2Attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error
	Ink2SigmoidMul(x, gate Ink2Ref, n int) error
	Ink2Route(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32) error
	Ink2ExpertScale(y, ids, scale Ink2Ref, k, rows int) error
	Ink2Situ(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error
	Ink2Combine(down, wts, out Ink2Ref, k, n int) error
	Ink2Add(a, b, out Ink2Ref, n int) error
	Ink2SoftcapArgmax(logits Ink2Ref, n int, softcap float32, outIdx Ink2Ref) error

	// Ink2GraphBegin opens a capture of the enqueued ops; Ink2GraphEnd keeps the
	// instantiated graph; Ink2GraphLaunch replays it.
	Ink2GraphBegin() error
	Ink2GraphEnd() error
	Ink2GraphLaunch() error
	Ink2GraphReset()
}

// Inkling2HalfKVDevice is the decode QKV post and attend over f16 global-slot caches;
// local rings stay fp32. A backend without it keeps every cache fp32.
type Inkling2HalfKVDevice interface {
	// Ink2QKVPostHalf is Ink2QKVPost with kcache and vcache holding __half elements.
	Ink2QKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error
	// Ink2AttendHalf is Ink2Attend over f16 kcache and vcache.
	Ink2AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error
}

// Ink2FP8KVCodeFloats returns the float32 element count to pass to Ink2Alloc
// for one FP8 E4M3FN K or V cache of capacity cap (cap * kvHeads * headDim bytes).
func Ink2FP8KVCodeFloats(cap, kvHeads, headDim int) int {
	return (cap*kvHeads*headDim + 3) / 4
}

// Ink2FP8KVScaleFloats returns the float32 element count to pass to Ink2Alloc
// for one per-(token, kvHead) F32 scale buffer of capacity cap (cap * kvHeads floats).
func Ink2FP8KVScaleFloats(cap, kvHeads int) int {
	return cap * kvHeads
}

// Inkling2FP8KVDevice is the decode and prefill QKV post and attend over
// per-(token, kvHead) OCP FP8 E4M3FN global-slot caches with F32 absmax scales;
// local SWA rings stay fp32.
//
// Buffer layout per global slot:
//   - kcache, vcache: uint8 E4M3FN codes laid out [cap][kvHeads][headDim],
//     allocated with Ink2Alloc(Ink2FP8KVCodeFloats(cap, kvHeads, headDim)).
//   - kscale, vscale: float32 per-(token, kvHead) scales laid out [cap][kvHeads],
//     allocated with Ink2Alloc(Ink2FP8KVScaleFloats(cap, kvHeads)).
//     Dequant(pos, kvh, d) = decode_e4m3fn(code[pos, kvh, d]) * scale[pos, kvh].
//   - overflow (optional; zero Ink2Ref disables): 1-int32 device flag allocated
//     with Ink2Alloc(1); set to non-zero via atomicOr if any K/V element is
//     non-finite or overflows FP8 scale range.
type Inkling2FP8KVDevice interface {
	// Ink2PFP8KVReady reports why the fp8 prefill ops cannot run in this process, before any is called.
	Ink2PFP8KVReady() error
	// Ink2QKVPostFP8 is Ink2QKVPost writing per-(token, kvHead) E4M3FN codes to kcache/vcache
	// and F32 scales to kscale/vscale.
	Ink2QKVPostFP8(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error
	// Ink2AttendFP8 is Ink2Attend reading E4M3FN kcache/vcache with F32 kscale/vscale.
	Ink2AttendFP8(q, r, sinks, relBias, kcache, vcache, kscale, vscale, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error
	// Ink2PQKVPostFP8 is Ink2PQKVPost writing per-(token, kvHead) E4M3FN codes and F32 scales for T tokens starting at pos0.
	Ink2PQKVPostFP8(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error
	// Ink2PAttendFP8 is Ink2PAttend reading E4M3FN kcache/vcache with F32 kscale/vscale for T tokens starting at pos0.
	Ink2PAttendFP8(q, r, sinks, relBias, kcache, vcache, kscale, vscale, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error
}

// Ink2NVFP4KVCodeFloats returns the float32 element count to pass to Ink2Alloc
// for one NVFP4 packed E2M1 K or V cache of capacity cap (cap * kvHeads * (headDim/2) bytes).
func Ink2NVFP4KVCodeFloats(cap, kvHeads, headDim int) int {
	return (cap*kvHeads*(headDim/2) + 3) / 4
}

// Ink2NVFP4KVScaleFloats returns the float32 element count to pass to Ink2Alloc
// for one per-16-element E4M3FN block-scale buffer of capacity cap (cap * kvHeads * (headDim/16) bytes).
func Ink2NVFP4KVScaleFloats(cap, kvHeads, headDim int) int {
	return (cap*kvHeads*(headDim/16) + 3) / 4
}

// Inkling2NVFP4KVDevice is the decode and prefill QKV post and attend over
// block-16 NVFP4 (E2M1 codes + E4M3FN block scales) global-slot caches;
// local SWA rings stay fp32. Requires headDim % 16 == 0.
//
// Buffer layout per global slot:
//   - kcache, vcache: uint8 packed E2M1 codes laid out [cap][kvHeads][headDim/2]
//     (low nibble = even dim 2*i, high nibble = odd dim 2*i+1), allocated with
//     Ink2Alloc(Ink2NVFP4KVCodeFloats(cap, kvHeads, headDim)).
//   - kscale, vscale: uint8 OCP E4M3FN per-16-element block scales laid out
//     [cap][kvHeads][headDim/16], allocated with
//     Ink2Alloc(Ink2NVFP4KVScaleFloats(cap, kvHeads, headDim)).
//     Dequant(pos, kvh, d) = decode_e2m1(nibble[pos, kvh, d]) * decode_e4m3fn(scale[pos, kvh, d/16]).
//   - overflow (optional; zero Ink2Ref disables): 1-int32 device flag allocated
//     with Ink2Alloc(1); set to non-zero via atomicOr if any K/V element is
//     non-finite or any block scale exceeds E4M3FN max (amax16/6.0 > 448.0).
type Inkling2NVFP4KVDevice interface {
	// Ink2PNVFP4KVReady reports why the nvfp4 prefill ops cannot run in this process, before any is called.
	Ink2PNVFP4KVReady() error
	// Ink2QKVPostNVFP4 is Ink2QKVPost writing block-16 E2M1 codes to kcache/vcache
	// and E4M3FN block scales to kscale/vscale.
	Ink2QKVPostNVFP4(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error
	// Ink2AttendNVFP4 is Ink2Attend reading block-16 E2M1 kcache/vcache with E4M3FN kscale/vscale.
	Ink2AttendNVFP4(q, r, sinks, relBias, kcache, vcache, kscale, vscale, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error
	// Ink2PQKVPostNVFP4 is Ink2PQKVPost writing block-16 E2M1 codes and E4M3FN scales for T tokens starting at pos0.
	Ink2PQKVPostNVFP4(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error
	// Ink2PAttendNVFP4 is Ink2PAttend reading block-16 E2M1 kcache/vcache with E4M3FN scales for T tokens starting at pos0.
	Ink2PAttendNVFP4(q, r, sinks, relBias, kcache, vcache, kscale, vscale, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error
}

// Inkling2AsymKVDevice is the decode and prefill QKV post and attend over
// asymmetric global-slot caches (K = block-16 FP8 E4M3FN with E4M3FN block scales,
// V = block-16 NVFP4 E2M1 with E4M3FN block scales); local SWA rings stay fp32.
type Inkling2AsymKVDevice interface {
	Ink2PAsymKVReady() error
	Ink2QKVPostAsym(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error
	Ink2AttendAsym(q, r, sinks, relBias, kcache, vcache, kscale, vscale, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error
	Ink2PQKVPostAsym(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error
	Ink2PAttendAsym(q, r, sinks, relBias, kcache, vcache, kscale, vscale, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error
}
