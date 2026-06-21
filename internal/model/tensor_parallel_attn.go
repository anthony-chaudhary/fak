package model

import "fmt"

// tensor_parallel_attn.go — the SECOND Megatron tensor-parallel building block:
// multi-head attention sharded across ranks. tensor_parallel.go proved the FFN block
// (gate/up column-parallel, down row-parallel); this proves attention, completing the
// "tensor parallelism within a layer" pair.
//
// Multi-head attention is EMBARRASSINGLY head-parallel: each head's scores, softmax, and
// weighted value-sum depend only on that head's own Q/K/V — never on another head. So
// sharding the heads across ranks needs NO collective inside the attention itself; the
// only cross-rank reduction is the OUTPUT projection, which is row-parallel over the
// concatenated head dimension (exactly one AllReduce per attention block, the same shape
// as the FFN's down projection). This is why Megatron shards attention by head.
//
// The decomposition (with GQA, query head h -> kv head h/grp, grp = nH/nKV):
//   - COLUMN-parallel Q/K/V: rank r owns a contiguous band of KV-head GROUPS — kv heads
//     [Lo,Hi) and the query heads [Lo*grp, Hi*grp) that map to them. Its q/k/v projection
//     rows are the contiguous slices for those heads (a column-parallel matmul slice), so
//     each projected head equals the full unsharded projection's same head bit-for-bit.
//   - Per-head attention runs locally on the rank's heads — identical math to the unsharded
//     run for each head, so the rank's attention-output band equals the unsharded
//     attention-output's matching head slice BIT-FOR-BIT.
//   - ROW-parallel output projection: o_proj is [hidden, nH*headDim]; rank r holds the
//     columns for its query heads and computes a PARTIAL [hidden] from its own attention
//     output; AllReduceSum across ranks gives the final output. This reassociates the
//     reduction, so the final output matches the unsharded reference within the documented
//     ~1e-6 round-off, never claimed bit-exact (parallel.go discipline) — identical to the
//     FFN. TensorParallelAttentionReference pins the AllReduce's rank order bit-exactly.
//
// HONESTY — what this primitive does and does NOT prove. The reference it is gated against
// (referenceAttention) is the SAME per-head/projection code (attentionOutputBand) run
// UNSHARDED over the full head range. So the gates prove SHARDING-INVARIANCE — that
// splitting the heads across ranks and AllReduce-ing the o_proj reproduces the unsharded
// result — NOT that this attention kernel matches HF or fak's live attnSeq. This standalone
// kernel is intentionally minimal: it models scaled-dot-product causal attention + GQA
// only, and OMITS RoPE, attention softcap, ALiBi bias, qk-norm, the AttnOutputGate, and
// q/k/v/o biases that fak's real attnSeq (forward.go) applies. Those are per-head/elementwise
// and orthogonal to the head-sharding correctness proven here; reproducing fak's ACTUAL
// attention is the job of the wiring step, which must gate the sharded path against the
// existing HF oracle. Like TensorParallelFFN this is a STANDALONE primitive over explicit
// weights, not wired into the live Forward path.

// attnHeadBand is the per-rank head assignment derived from a kv-head shard: kv heads
// [KVLo,KVHi) and the query heads [QLo,QHi) that map onto them under GQA.
type attnHeadBand struct {
	KVLo, KVHi int
	QLo, QHi   int
}

// attnBandForShard derives a rank's query/kv head band from its kv-head shard. The plan
// shards the nKV dimension; query heads follow because head h attends kv head h/grp.
func attnBandForShard(s TPShard, grp int) attnHeadBand {
	return attnHeadBand{KVLo: s.Lo, KVHi: s.Hi, QLo: s.Lo * grp, QHi: s.Hi * grp}
}

// headCausalAttention computes one query head's causal attention output over a sequence:
// for each position t, scores over keys [0,t] (scaled dot), softmax, then the weighted
// value sum. qSeq/kSeq/vSeq are the per-position head vectors (length headDim each). This
// is the same per-head formulation as the model's reference attention (swa_test.go), with
// no sliding window. It is head-local: identical whether the head is computed in the
// unsharded run or on a shard, which is what makes the sharded attention output bit-exact.
func headCausalAttention(qSeq, kSeq, vSeq [][]float32, headDim int, scale float32) [][]float32 {
	seq := len(qSeq)
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		scores := make([]float32, t+1)
		for j := 0; j <= t; j++ {
			scores[j] = dot(qSeq[t], kSeq[j]) * scale
		}
		softmaxInPlace(scores)
		o := make([]float32, headDim)
		for j := 0; j <= t; j++ {
			wj := scores[j]
			vh := vSeq[j]
			for d := 0; d < headDim; d++ {
				o[d] += wj * vh[d]
			}
		}
		out[t] = o
	}
	return out
}

// attentionOutputBand projects X through the q/k/v weights for the head band [qLo,qHi) ×
// kv band [kvLo,kvHi) and returns the per-position concatenated attention output for those
// query heads — the band a single rank owns. qW is [nH*headDim, hidden] row-major (so query
// head h is rows [h*headDim,(h+1)*headDim)); kW/vW are [nKV*headDim, hidden]. The returned
// rows have length (qHi-qLo)*headDim. Shared by the unsharded reference (full band) and each
// rank (its band), so a rank's output equals the full-band reference's slice bit-for-bit
// (a head-index disjointness witness — same code over a sub-range vs the full range).
func attentionOutputBand(qW, kW, vW []float32, X [][]float32, band attnHeadBand, grp, hidden, headDim int, scale float32) [][]float32 {
	seq := len(X)
	// Project the band's q heads and kv heads at every position (column-parallel slices).
	qProj := make([][]float32, seq) // [seq][(QHi-QLo)*headDim]
	kProj := make([][]float32, seq) // [seq][(KVHi-KVLo)*headDim]
	vProj := make([][]float32, seq)
	qRows := (band.QHi - band.QLo) * headDim
	kvRows := (band.KVHi - band.KVLo) * headDim
	qSlice := shardWeightRows(qW, hidden, band.QLo*headDim, band.QHi*headDim)
	kSlice := shardWeightRows(kW, hidden, band.KVLo*headDim, band.KVHi*headDim)
	vSlice := shardWeightRows(vW, hidden, band.KVLo*headDim, band.KVHi*headDim)
	for t := 0; t < seq; t++ {
		qProj[t] = matRows(qSlice, X[t], qRows, hidden)
		kProj[t] = matRows(kSlice, X[t], kvRows, hidden)
		vProj[t] = matRows(vSlice, X[t], kvRows, hidden)
	}
	// Per-head causal attention over the band; query head h (global) uses kv head h/grp.
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		out[t] = make([]float32, qRows)
	}
	nQ := band.QHi - band.QLo
	for hl := 0; hl < nQ; hl++ {
		gh := band.QLo + hl               // global query head
		localKV := (gh / grp) - band.KVLo // local kv-head index within this band
		qSeq := make([][]float32, seq)
		kSeq := make([][]float32, seq)
		vSeq := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			qSeq[t] = qProj[t][hl*headDim : (hl+1)*headDim]
			kSeq[t] = kProj[t][localKV*headDim : (localKV+1)*headDim]
			vSeq[t] = vProj[t][localKV*headDim : (localKV+1)*headDim]
		}
		ho := headCausalAttention(qSeq, kSeq, vSeq, headDim, scale)
		for t := 0; t < seq; t++ {
			copy(out[t][hl*headDim:(hl+1)*headDim], ho[t])
		}
	}
	return out
}

// TensorParallelAttention computes multi-head causal self-attention sharded across `plan`
// over the KV-head dimension (plan.Dim must == nKV): each rank owns a band of KV-head
// groups, projects+attends its heads locally (column-parallel Q/K/V, no collective), and
// the output projection is row-parallel with ONE AllReduceSum. The result matches the
// unsharded reference (referenceAttention) within the o_proj AllReduce round-off, while
// each rank's attention-output band equals the unsharded band bit-for-bit (see the file
// header for what "reference" means here — sharding-invariance, not HF-correctness).
//
// Shapes: qW [nH*headDim, hidden]; kW,vW [nKV*headDim, hidden]; oW [hidden, nH*headDim]; X
// [seq][hidden]; result [seq][hidden]. nH must be a multiple of nKV (GQA). The rank count
// need NOT divide nKV — NewTPPlan tiles the nKV kv-head groups near-evenly, so a rank may
// own more groups than another. A nil collective defaults to LocalCollective.
func TensorParallelAttention(qW, kW, vW, oW []float32, X [][]float32, hidden, nH, nKV, headDim int, scale float32, plan TPPlan, coll Collective) ([][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if nH <= 0 || nKV <= 0 || headDim <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("model: TensorParallelAttention bad dims nH=%d nKV=%d headDim=%d hidden=%d", nH, nKV, headDim, hidden)
	}
	if nH%nKV != 0 {
		return nil, fmt.Errorf("model: TensorParallelAttention nH=%d not a multiple of nKV=%d (GQA grouping)", nH, nKV)
	}
	if plan.Dim != nKV {
		return nil, fmt.Errorf("model: TensorParallelAttention plan.Dim = %d, want nKV = %d (shard whole KV-head groups)", plan.Dim, nKV)
	}
	qHeadDim := nH * headDim
	if len(qW) != qHeadDim*hidden {
		return nil, fmt.Errorf("model: TensorParallelAttention qW len = %d, want nH*headDim*hidden = %d", len(qW), qHeadDim*hidden)
	}
	if len(kW) != nKV*headDim*hidden || len(vW) != nKV*headDim*hidden {
		return nil, fmt.Errorf("model: TensorParallelAttention kW/vW len = %d/%d, want nKV*headDim*hidden = %d", len(kW), len(vW), nKV*headDim*hidden)
	}
	if len(oW) != hidden*qHeadDim {
		return nil, fmt.Errorf("model: TensorParallelAttention oW len = %d, want hidden*nH*headDim = %d", len(oW), hidden*qHeadDim)
	}
	for t := range X {
		if len(X[t]) != hidden {
			return nil, fmt.Errorf("model: TensorParallelAttention X row %d len = %d, want hidden %d", t, len(X[t]), hidden)
		}
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	seq := len(X)
	rankPartials := tensorParallelAttentionPartials(qW, kW, vW, oW, X, hidden, nH, nKV, headDim, scale, plan)
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		parts := make([][]float32, len(plan.Shards))
		for r := range plan.Shards {
			parts[r] = rankPartials[r][t]
		}
		reduced, err := coll.AllReduceSum(parts)
		if err != nil {
			return nil, fmt.Errorf("model: TensorParallelAttention o_proj AllReduce at position %d: %w", t, err)
		}
		out[t] = reduced
	}
	return out, nil
}

// tensorParallelAttentionPartials computes each rank's [seq][hidden] o_proj partial — the
// per-rank result the attention block's per-position AllReduceSum combines. Factored out so
// TensorParallelAttention (Collective reduction) and TensorParallelAttentionReference
// (rank-order sum) share the IDENTICAL per-rank computation and differ ONLY in the
// reduction, which the bit-exact reference pins. Assumes validated shapes.
func tensorParallelAttentionPartials(qW, kW, vW, oW []float32, X [][]float32, hidden, nH, nKV, headDim int, scale float32, plan TPPlan) [][][]float32 {
	grp := nH / nKV
	qHeadDim := nH * headDim
	seq := len(X)
	rankPartials := make([][][]float32, len(plan.Shards)) // [rank][seq][hidden]
	for r, s := range plan.Shards {
		band := attnBandForShard(s, grp)
		attnOut := attentionOutputBand(qW, kW, vW, X, band, grp, hidden, headDim, scale)
		oCols := shardWeightColumns(oW, hidden, qHeadDim, band.QLo*headDim, band.QHi*headDim)
		part := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			part[t] = matRows(oCols, attnOut[t], hidden, (band.QHi-band.QLo)*headDim)
		}
		rankPartials[r] = part
	}
	return rankPartials
}

// TensorParallelAttentionReference is the shard-grouped bit-exact oracle for
// TensorParallelAttention: the identical per-rank o_proj partials, summed per position in
// rank order directly rather than through the Collective. Pinning TensorParallelAttention
// == this at max|Δ|=0 proves the collective reduces the o_proj partials in rank order
// (the row-parallel bit-exactness contract), independently of the loose vs-reference
// round-off bound. Validation mirrors TensorParallelAttention.
func TensorParallelAttentionReference(qW, kW, vW, oW []float32, X [][]float32, hidden, nH, nKV, headDim int, scale float32, plan TPPlan) ([][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if nH <= 0 || nKV <= 0 || headDim <= 0 || hidden <= 0 || nH%nKV != 0 || plan.Dim != nKV {
		return nil, fmt.Errorf("model: TensorParallelAttentionReference bad dims/plan (nH=%d nKV=%d headDim=%d hidden=%d planDim=%d)", nH, nKV, headDim, hidden, plan.Dim)
	}
	rankPartials := tensorParallelAttentionPartials(qW, kW, vW, oW, X, hidden, nH, nKV, headDim, scale, plan)
	seq := len(X)
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		parts := make([][]float32, len(plan.Shards))
		for r := range plan.Shards {
			parts[r] = rankPartials[r][t]
		}
		out[t] = sumPartialsRankOrder(parts)
	}
	return out, nil
}

// referenceAttentionBandSlice exposes attentionOutputBand for the FULL head range so a test
// can assert a rank's band equals the full-band reference's matching slice. It is the
// unsharded attention output for query heads [0,nH) — the per-head outputs a sharded run
// reproduces (same code, full range vs sub-range: a head-index disjointness witness).
func referenceAttentionBandSlice(qW, kW, vW []float32, X [][]float32, hidden, nH, nKV, headDim int, scale float32) [][]float32 {
	grp := nH / nKV
	return attentionOutputBand(qW, kW, vW, X, attnHeadBand{KVLo: 0, KVHi: nKV, QLo: 0, QHi: nH}, grp, hidden, headDim, scale)
}

// referenceAttention computes UNSHARDED multi-head causal attention end to end — full
// projection, per-head attention, then a SINGLE o_proj matmul over the whole nH*headDim
// (one fdot per output, NOT a per-rank reassociated sum) — so the test's "TP vs reference"
// comparison is against a genuinely unsharded run, and the row-parallel round-off is
// measured honestly. NOTE: this shares its attention/projection core with the sharded path
// (attentionOutputBand), so the gate proves sharding-invariance, not HF-correctness (see
// file header). Returns [seq][hidden].
func referenceAttention(qW, kW, vW, oW []float32, X [][]float32, hidden, nH, nKV, headDim int, scale float32) [][]float32 {
	attnOut := referenceAttentionBandSlice(qW, kW, vW, X, hidden, nH, nKV, headDim, scale)
	seq := len(X)
	qHeadDim := nH * headDim
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		out[t] = matRows(oW, attnOut[t], hidden, qHeadDim)
	}
	return out
}
