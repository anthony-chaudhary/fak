package compute

// Ink2SeqStrides are the element strides between the C sequences of a
// sequence-batched op: sequence i's operand is the given Ref advanced by stride*i.
// Weights, clip scales and the relative bias are shared and take no stride. Scores
// is in floats and must be even (the scores rows are float64).
type Ink2SeqStrides struct {
	Q, R, K, V, KDyn, VDyn, Sinks, KCache, VCache, Hist, Scores, Out, Meta, X, Resid int
}

// Fields returns the strides in the order the device launchers take them.
func (st Ink2SeqStrides) Fields() [15]int64 {
	return [15]int64{int64(st.Q), int64(st.R), int64(st.K), int64(st.V), int64(st.KDyn), int64(st.VDyn),
		int64(st.Sinks), int64(st.KCache), int64(st.VCache), int64(st.Hist), int64(st.Scores), int64(st.Out),
		int64(st.Meta), int64(st.X), int64(st.Resid)}
}

// Inkling2BatchDevice runs the per-sequence state ops of a lockstep batched decode
// for C sequences in one launch per kernel, the sequence being an extra grid axis.
// Row i of every op computes exactly what the matching Inkling2Device op computes on
// sequence i's operands alone (same kernel body, same reduction order), so a batched
// step is bit-identical to looping the single-sequence op over the rows.
type Inkling2BatchDevice interface {
	// Ink2BQKVPost is Ink2QKVPost over C sequences (q, r, k, v, kcache, vcache, meta,
	// cv's dyn taps and histories advance by st).
	Ink2BQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv, st Ink2SeqStrides, C int) error
	// Ink2BAttend is Ink2Attend over C sequences, each with its own scores rows.
	Ink2BAttend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int, st Ink2SeqStrides, C int) error
	// Ink2BConv is Ink2Conv (no dynamic taps, one head) over C sequences (x, hist,
	// out and resid advance by st.X, st.Hist, st.Out and st.Resid).
	Ink2BConv(x, w, hist, out, resid Ink2Ref, n, kn int, norm float32, st Ink2SeqStrides, C int) error
	// Ink2BSoftcapArgmax is Ink2SoftcapArgmax over C logit rows ld floats apart into
	// outIdx [C][4] ints (index at 0 of each row).
	Ink2BSoftcapArgmax(logits Ink2Ref, n, ld int, softcap float32, outIdx Ink2Ref, C int) error
}
