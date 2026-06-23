package compute

// dsa.go — the OPTIONAL device seam for GLM-MoE-DSA's sparse attention.
//
// GLM-5.2's attention is sparse: a learned indexer scores every cached key, top-k selects
// which keys a query attends, and the softmax(scale·q·k)·V runs over ONLY that selected set
// (model.glmDsaAttendCached). The dense projections feeding it already route through the core
// Backend.MatMul (the #86 device path). This seam moves the remaining piece — the attention
// compute over the selected keys — onto the device too, so a GLM-5.2 forward on a device
// backend runs its attention math on the kernel, not host-resident.
//
// It is an OPTIONAL capability discovered by type-assertion (exactly like CollectiveBackend):
// the forward asserts the Backend for DSASparseBackend and falls back to the host loop when a
// backend does not implement it, so adding it is a backend method, never a forward-loop edit,
// and a backend without it (an early metal/vulkan build) is simply not used for this op.

// DSASparseBackend is the optional capability a backend implements to run GLM-MoE-DSA's
// sparse attention — the per-head softmax(scale·q·k)·V over the host-SELECTED causal key set —
// on the device instead of host-resident in model.glmDsaAttendCached.
//
// WHY the selection stays on the host (by design, not omission): the f64 index-score dots +
// top-k pick WHICH keys attend. Computing that selection on the host keeps the selected set
// bit-identical CPU↔device, so the device's only divergence from the host is the f32 reduction
// order over the SAME keys — the Approx class the dense GEMM and flash-attention lanes already
// live in — never a different key set (a single flipped top-k entry would diverge the output
// far past a reduction-order cosine, so this boundary is what keeps the witness argmax-exact).
// The genuinely sparse control flow (scoring + selection + the O(topK) key gather) is small
// next to the attention FLOPs this op moves to the kernel.
//
// The cpu-ref implements it as the EXACT arithmetic of glmDsaAttendCached's inner loop
// (single-accumulator dot·scale, softmaxInPlace, in-order ΣwV — the same dot/softmaxInPlace the
// model uses), so adopting the seam leaves the cpu-ref forward argmax-exact with the host path;
// a device implements it with its own kernel (the cuda backend: k_dsa_sparse_attend, Approx,
// held to cudaDsaSparseAttnCosineMin).
type DSASparseBackend interface {
	Backend
	// DSASparseAttend runs GLM-DSA sparse attention for ONE query position over nSel gathered,
	// causal, already-SELECTED keys/values. Layout (matching glmDsaAttendCached's cache):
	//   q    [nH*qkHead]            — this position's per-head query (post-RoPE)
	//   selK [nSel, nH*qkHead]      — host-gathered selected key rows, head h at i*nH*qkHead+h*qkHead
	//   selV [nSel, nH*vHead]       — host-gathered selected value rows, head h at i*nH*vHead+h*vHead
	// For each head h: scores[i] = scale·dot(q_h, selK_i_h), softmax over i, then
	//   out[h*vHead:(h+1)*vHead] = Σ_i softmax(scores)[i]·selV_i_h.
	// Returns [nH*vHead] (the attnConcat fed to o_proj). qkHead = qkNope+qkRope (key/query width),
	// vHead the value width — they DIFFER under MLA, so this op carries both.
	DSASparseAttend(q, selK, selV Tensor, nSel, nH, qkHead, vHead int, scale float32) Tensor
}

// dsaSparseAttendHost is the byte-for-byte CPU reference: the exact inner loop of
// model.glmDsaAttendCached, reading from the gathered contiguous selK/selV instead of the cache.
// The cpu-ref backend's DSASparseAttend delegates here, and every device kernel is held Approx
// against it. Kept package-level so a backend's host fallback (or a test) can reuse the reference
// reduction order. It uses the package dot / softmaxInPlace, which are byte-identical to the
// model's, so cpu-ref sparse attention is argmax-exact with the all-host forward.
func dsaSparseAttendHost(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) []float32 {
	out := make([]float32, nH*vHead)
	for h := 0; h < nH; h++ {
		qh := q[h*qkHead : (h+1)*qkHead]
		scores := make([]float32, nSel)
		for i := 0; i < nSel; i++ {
			kh := selK[i*nH*qkHead+h*qkHead : i*nH*qkHead+(h+1)*qkHead]
			scores[i] = dot(qh, kh) * scale
		}
		softmaxInPlace(scores)
		oh := out[h*vHead : (h+1)*vHead]
		for i := 0; i < nSel; i++ {
			vh := selV[i*nH*vHead+h*vHead : i*nH*vHead+(h+1)*vHead]
			w := scores[i]
			for d := 0; d < vHead; d++ {
				oh[d] += w * vh[d]
			}
		}
	}
	return out
}

// DSASparseAttend on the cpu-ref is the Reference implementation: byte-for-byte
// glmDsaAttendCached's inner loop. (The cpu-ref is the floor every device is checked against.)
func (c *cpuBackend) DSASparseAttend(q, selK, selV Tensor, nSel, nH, qkHead, vHead int, scale float32) Tensor {
	out := dsaSparseAttendHost(c.f32(q), c.f32(selK), c.f32(selV), nSel, nH, qkHead, vHead, scale)
	return c.result([]int{nH * vHead}, out)
}
