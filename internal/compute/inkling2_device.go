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
	// Ink2CtrlDown is g = Down*[rms(cur), rms(msg)] for an F32 [rank, 2n] Down.
	Ink2CtrlDown(cur, msg, w, g Ink2Ref, n, rank int) error
	Ink2CtrlApply(cur, msg, g, gainW, forgetW, out Ink2Ref, n, rank int) error
	Ink2Conv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error
	Ink2QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int) error
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
