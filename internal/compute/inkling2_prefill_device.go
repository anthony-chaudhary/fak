package compute

// Inkling2PrefillDevice batches the Inkling2Device decode ops along the token axis
// for a prompt chunk: token t of a T-token chunk sits at position pos0+t and every
// activation is [T][width] rows. Ops that are already row-batched (Ink2RMS rows,
// Ink2SigmoidMul, Ink2Situ and Ink2ExpertMatVec over T*k slots) are reused as is.
type Inkling2PrefillDevice interface {
	// Ink2PMatMul is Y[T,out] = X[T,in] times W transposed for one resident dense weight.
	Ink2PMatMul(w Tensor, x, y Ink2Ref, T int) error
	// Ink2PMix writes query-major [nq][T][n].
	Ink2PMix(states []Ink2Ref, queries Ink2Ref, nq int, out Ink2Ref, n, T int) error
	Ink2PCtrlDown(cur, msg, w, g Ink2Ref, n, rank, T int) error
	Ink2PCtrlApply(cur, msg, g, gainW, forgetW, out Ink2Ref, n, rank, T int) error
	// Ink2PConv walks the chunk in token order and leaves the final history in hist.
	Ink2PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error
	Ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error
	Ink2PAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error
	Ink2PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error
	Ink2PCombine(down, wts, out Ink2Ref, k, n, T int) error
	// Ink2PExpertGrouped multiplies npairs routed (token, slot) pairs through one
	// stacked expert tensor grouped by expert, so each expert block is decoded once
	// per tile of pairs; pair i reads x row (i/xDiv)*xStride and writes y row i.
	// scratch holds Ink2PGroupScratch(npairs, nexp) ints; bucketed reuses the bucket an
	// earlier call left in scratch for the same ids. ok is false (nothing
	// launched) for a dtype without a grouped kernel (NVFP4 runs a BF16 tensor-core MMA,
	// the IQ and Q3_K stacks an FP32 tile kernel).
	Ink2PExpertGrouped(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch Ink2Ref, bucketed bool) (ok bool, err error)
	// Ink2PExpertSorted runs the decode expert GEMV over the npairs pairs in expert
	// order (pair inputs gathered into sortX, outputs through sortY), so consecutive
	// slots reuse one expert's rows from cache. sortX holds npairs*xStride floats,
	// sortY npairs*expertRows, sortIds npairs ints. ok is false for a dtype without
	// an indexed kernel.
	Ink2PExpertSorted(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch, sortIds, sortX, sortY Ink2Ref) (ok bool, err error)
}

// Ink2PGroupScratch is the int count of the grouped expert GEMM bucket scratch.
func Ink2PGroupScratch(npairs, nexp int) int {
	tiles := npairs/8 + nexp
	return nexp + (nexp + 1) + npairs + 2*tiles + 1
}
