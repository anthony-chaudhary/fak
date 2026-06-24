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

// DSAIndexBackend is the optional capability a backend implements to run GLM-MoE-DSA's learned
// indexer SCORING + top-k SELECTION — the last GLM-5.2 compute still host-resident even when the
// dense projections and the sparse-attention compute already run on the kernel. The host loop it
// replaces is model.glmDsaIndexStep's per-key score (decode) / model.dsaIndexScores+dsaTopKIndices
// (prefill): for each causal key, score = Σ_h weights[h]·relu(scale·dot(index_q[h], index_k)), then
// keep the top-k key POSITIONS (score descending, ties broken by lower position — the exact
// dsaTopKIndices order).
//
// WHY this can move to the device AND keep the witness argmax-exact (the honesty boundary that kept
// the selection host-side until now): the score is a small dot over IndexHeadDim (8 in the fixture,
// O(128) in the real model) accumulated in f64 — the SAME reduction the host uses — so the device
// scores match the host f64 scores bit-closely, and the top-k is taken with the IDENTICAL total order
// (score desc, position asc). The selected key SET is therefore bit-identical CPU↔device — there is
// no flipped top-k entry — so a GLM-5.2 forward whose selection runs on the kernel attends exactly
// the keys the host would, and the downstream sparse-attention stays argmax-exact. The cost is the
// f64 score dots (cheap: A100 has native f64; IndexHeadDim is tiny), paid to make selection-stability
// PROVABLE rather than gated. This is the difference from the f32-GEMM lanes, which are Approx by
// design: the indexer drives a DISCRETE selection, so it must be reduction-faithful, not cosine-close.
type DSAIndexBackend interface {
	Backend
	// DSAIndexSelect scores nKeys cached index-keys against one query's nH index-heads and returns
	// the top-k selected key positions (length min(topK, nValid)) for that query at queryPos.
	//   indexQ  [nH*indexDim]        — this query's per-head indexer query (post-RoPE, post-norm)
	//   indexK  [nKeys*indexDim]     — the cached per-key indexer keys (post-RoPE, post-norm), key
	//                                  k at k*indexDim; key position k is causally valid iff k<=queryPos
	//   weights [nH]                 — this query's per-head indexer weights (already scaled)
	// Score(k) = Σ_h weights[h]·relu(scale·Σ_d indexQ[h*indexDim+d]·indexK[k*indexDim+d]); keys with
	// position > queryPos are masked out. Returns the top-k positions, score descending, ties by lower
	// position (dsaTopKIndices order). The returned slice is host-resident (a small index list).
	DSAIndexSelect(indexQ, indexK, weights Tensor, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int
}

// dsaIndexSelectHost is the byte-for-byte CPU reference for the indexer score + top-k: the exact
// arithmetic of model.glmDsaIndexStep's per-key loop (f64 dot, relu, weighted sum) followed by the
// dsaTopKIndices total order (score descending, ties by lower position). Every device kernel is held
// against it; because the device accumulates the score dot in f64 too, the selected positions match
// bit-for-bit (selection-stable), not merely cosine-close.
func dsaIndexSelectHost(indexQ, indexK, weights []float32, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int {
	cands := make([]dsaIndexCand, 0, nKeys)
	for k := 0; k < nKeys; k++ {
		if k > queryPos {
			continue
		}
		key := indexK[k*indexDim : (k+1)*indexDim]
		var score float64
		for h := 0; h < nH; h++ {
			qh := indexQ[h*indexDim : (h+1)*indexDim]
			var hd float64
			for d := 0; d < indexDim; d++ {
				hd += float64(qh[d]) * float64(key[d])
			}
			hs := hd * float64(scale)
			if hs < 0 {
				hs = 0
			}
			score += float64(weights[h]) * hs
		}
		cands = append(cands, dsaIndexCand{pos: k, score: score})
	}
	// Stable order: score descending, ties by lower position (the dsaTopKIndices tie-break).
	sortDSAIndexCands(cands)
	n := topK
	if n > len(cands) {
		n = len(cands)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = cands[i].pos
	}
	return out
}

// dsaIndexCand pairs a key position with its indexer score for the top-k sort.
type dsaIndexCand struct {
	pos   int
	score float64
}

// sortDSAIndexCands sorts by score descending, ties by lower position — an insertion sort to keep
// the cpu-ref dependency-free and the order byte-identical to model.dsaTopKIndices' sort.SliceStable.
func sortDSAIndexCands(c []dsaIndexCand) {
	for i := 1; i < len(c); i++ {
		j := i
		for j > 0 && dsaIndexCandLess(c[j], c[j-1]) {
			c[j], c[j-1] = c[j-1], c[j]
			j--
		}
	}
}

func dsaIndexCandLess(a, b dsaIndexCand) bool {
	if a.score == b.score {
		return a.pos < b.pos
	}
	return a.score > b.score
}

// DSAIndexSelect on the cpu-ref is the Reference implementation: byte-for-byte the host indexer
// score + top-k. (The cpu-ref is the selection floor every device is checked against.)
func (c *cpuBackend) DSAIndexSelect(indexQ, indexK, weights Tensor, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int {
	return dsaIndexSelectHost(c.f32(indexQ), c.f32(indexK), c.f32(weights), nKeys, nH, indexDim, queryPos, topK, scale)
}
