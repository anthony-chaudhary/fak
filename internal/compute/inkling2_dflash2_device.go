package compute

// Inkling2DFlash2Device is the device op set of the DFlash2 draft block beside
// a resident inkling2 target. Weights are BF16 [rows, cols] tensors.
type Inkling2DFlash2Device interface {
	// Ink2DFLinear is y = x W^T for up to 8 rows; x holds 1 to 4 column segments.
	Ink2DFLinear(x []Ink2Ref, w Tensor, y Ink2Ref, tokens int) error
	// Ink2DFHead is y[t] = W x[t] over up to 8 rows of a packed head weight,
	// y row t at t*rows of W; ok is false for a weight it does not cover.
	Ink2DFHead(w Tensor, x, y Ink2Ref, tokens int) (bool, error)
	// Ink2DFConv is one side of the draft's dynamic short conv over a block.
	Ink2DFConv(x, dyn Ink2Ref, base Tensor, resid, out Ink2Ref, rows, hidden, group, kn, side int) error
	// Ink2DFAttend is the draft's non-causal block attention over a context ring.
	Ink2DFAttend(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error
	// Ink2DFSelect is the codebook candidate selector's greedy chain.
	Ink2DFSelect(logits Ink2Ref, rows, rowStride, vocab, k int, proj Ink2Ref, pred, succ Tensor,
		anchorDev Ink2Ref, softcap float32, scratch, path, cand Ink2Ref) error
	// Ink2DFQKRope RMS-normalizes each head by w and applies neox RoPE at position
	// *posDev+base+row, into row order or ring slot position%cap.
	Ink2DFQKRope(x, w, dst, posDev Ink2Ref, rows, heads, dim, cap, base int, ring bool, eps float32, theta float64) error
	// Ink2DFRingStore copies src row r into ring slot (*posDev+base+r)%cap.
	Ink2DFRingStore(src, ring, posDev Ink2Ref, rows, width, cap, base int) error
	// Ink2DFSiluMul is out = silu(g) * u.
	Ink2DFSiluMul(g, u, out Ink2Ref, n int) error
}

// Inkling2DFlash2FusedDevice is the optional fused launches of the draft
// block; each writes what its separate Inkling2DFlash2Device ops write.
type Inkling2DFlash2FusedDevice interface {
	// Ink2DFLinearMulti is Ink2DFLinear of up to three weights of equal
	// columns over one input, in one launch: y[i] = x w[i]^T.
	Ink2DFLinearMulti(x []Ink2Ref, w []Tensor, y []Ink2Ref, tokens int) error
	// Ink2DFLinearSiluMul is y = silu(x gate^T) * (x up^T), Ink2DFLinear twice
	// and Ink2DFSiluMul; scratch ([tokens][rows]) is used only on the fallback.
	Ink2DFLinearSiluMul(x Ink2Ref, gate, up Tensor, y, scratch Ink2Ref, tokens int) error
}
