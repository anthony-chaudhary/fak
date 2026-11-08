package model

// kimik31_dsa_forward.go — the Kimi-K3.1 Gated DSA layer forward, host oracle
// (ticket 1611-kimi-k31 leaf L6). See kimik31_dsa.go for the pinned
// recurrence, its references, and the state/param contracts.
//
// PER-TOKEN CONTRACT (x is the ALREADY input-layernorm'd layer input,
// [Hidden]; the composite owns attn_norm/res_mix, exactly as L5's KDA step
// takes normalized inputs and llama.cpp feeds build_dsa_layer its
// post-attn_norm cur):
//  1. qr = rmsnorm(WQa*x, WQaNorm, RMSEps); Qcur = WQb*qr
//     ([NumHeads x (nope+rope)]); the rope halves carry NO positional
//     transform (mla_use_nope, asserted at conversion time).
//  2. lat = WKvA*x ([KVLoRA+QkRope]); the cache row is
//     concat(rmsnorm(lat[:KVLoRA], WKvANorm, RMSEps), lat[KVLoRA:]).
//  3. ik = LayerNorm(WIdxK*x, IdxKNormW, IdxKNormB, IdxEps) — mean-
//     subtracting LayerNorm, NOT RMS — appended cell-aligned with the
//     latent row. The current cell participates in its own selection
//     (llama.cpp caches ik before scoring against get_k).
//  4. iq = WIdxQ*qr ([IdxHeads x IdxDim]); iw = WIdxProj*x pre-scaled by
//     1/sqrt(IdxDim*IdxHeads); score[t] = sum_h iw[h]*ReLU(iq[h].ik[t]).
//  5. sel = SelectKimiK31DSATopK(score, TopK): top min(TopK, nCells) by
//     (score desc, index asc). At step time every cached cell is causally
//     valid (cells only ever exist for past-or-current tokens), so the
//     additive-causal-mask intersection of build_top_k_mask is identity
//     here; PrefillSeq scans token-by-token, which reproduces the masked
//     batched prefill exactly.
//  6. Absorbed MLA over sel only: per head h, qNope/qPe are the two halves
//     of Qcur[h]; the K absorption replaces the shared compressed-key
//     projection with a per-head query-side fold
//     qAbs = WKb[h]^T * qNope ([KVLoRA], llama.cpp line 547
//     q_nope_absorbed = mul_mat(wk_b, q_nope)), then
//     s_t = kqScale*(qAbs.lat_t + qPe.kpe_t) with kqScale =
//     1/sqrt(nope+rope) applied INSIDE the softmax (F32, max-subtracted);
//     out_h = sum_t w_t*(WVb[h]*lat_t) (V projected from the cached latent,
//     the absorbed form, build_attn_mha v_mla=wv_b line 631).
//  7. Optional gate: out *= sigmoid(WGate*x) (reads the normed input x,
//     full-rank over the attention output), then WO*out.
//
// NUMERIC ORDER (pinned for the parity witness): projections via
// kdaMatVecMul (serial F32 inner products); the K absorption accumulates
// qAbs[j] over the nope dim serially; indexer scores accumulate heads outer
// / dims inner, skipping non-positive ReLU terms (adding +0 is bit-exact);
// attention weights accumulate cells ascending; value projection computes
// the full F32 inner product per (cell, channel) BEFORE the weighted sum.
// The independent float64 transcription (kimik31_dsa_naive_test.go) mirrors
// this order.

import (
	"fmt"
	"math"
)

// ForwardKimiK31DSAStep runs one Gated DSA layer over one token. x is the
// already input-layernorm'd layer input ([Hidden]); st carries the cell-
// aligned latent + indexer-key caches (built with NewKimiK31DSALayerState
// matching p's geometry); it returns the layer output ([Hidden], post
// o_proj) and updates st in place. Params must have passed
// ValidateKimiK31DSAParams.
func ForwardKimiK31DSAStep(st *KimiK31DSALayerState, p *KimiK31DSAParams, x []float32) []float32 {
	if st == nil || p == nil {
		panic("model: kimik31 dsa step: nil state or params")
	}
	if len(x) != p.Hidden {
		panic(fmt.Sprintf("model: kimik31 dsa step: input length %d != Hidden %d", len(x), p.Hidden))
	}
	if st.LatentDim != p.KVLoRA+p.QkRope || st.IdxDim != p.IdxDim {
		panic(fmt.Sprintf("model: kimik31 dsa step: state dims (%d,%d) != params ((kvlora+rope)%d, idxdim %d)",
			st.LatentDim, st.IdxDim, p.KVLoRA+p.QkRope, p.IdxDim))
	}

	// 1. The q latent; the indexer queries read the SAME normalized latent.
	qr := kdaMatVecMul(p.WQa, x, p.QLoRA, p.Hidden)
	rmsnormInto(qr, qr, p.WQaNorm, p.RMSEps)
	qFull := kdaMatVecMul(p.WQb, qr, p.NumHeads*(p.QkNope+p.QkRope), p.QLoRA)

	// 2-3. The cache row (normalized compressed part || raw PE) and the
	// indexer key (true LayerNorm), appended cell-aligned.
	latent, ik := kimiK31DSACacheRow(p, x)
	st.Append(latent, ik)

	// 4-5. Lightning-indexer scores over ALL cached cells, then top-k.
	iq := kdaMatVecMul(p.WIdxQ, qr, p.IdxHeads*p.IdxDim, p.QLoRA)
	iw := kdaMatVecMul(p.WIdxProj, x, p.IdxHeads, p.Hidden)
	preScale := float32(1.0 / math.Sqrt(float64(p.IdxHeads*p.IdxDim)))
	for i := range iw {
		iw[i] *= preScale
	}
	scores := kimiK31DSAIndexerScores(st, p, iq, iw)
	sel := SelectKimiK31DSATopK(scores, p.TopK)

	// 6-7. Sparse absorbed MLA over the selected cells, gate, o_proj.
	return kimiK31DSAAttendAndOut(st, p, qFull, sel, x)
}

// ForwardKimiK31DSAStepWithSelection is the shared-indexer seam: it appends
// the cache row for x and runs the sparse absorbed MLA over the caller-
// supplied selection instead of computing the lightning indexer. K3.1 ships
// every DSA layer with its own indexer (is_indexer_full all-1, verified
// against the official checkpoint), so L8 uses this only for GGUFs whose
// LLM_KV_ATTENTION_INDEXER_TYPES marks a layer shared: it passes the
// previous DSA layer's sel. sel must be non-empty, sorted ascending, with
// every entry in [0, Len) AFTER the append (the current cell included).
func ForwardKimiK31DSAStepWithSelection(st *KimiK31DSALayerState, p *KimiK31DSAParams, x []float32, sel []int) []float32 {
	if st == nil || p == nil {
		panic("model: kimik31 dsa step-with-selection: nil state or params")
	}
	if len(x) != p.Hidden {
		panic(fmt.Sprintf("model: kimik31 dsa step-with-selection: input length %d != Hidden %d", len(x), p.Hidden))
	}
	if st.LatentDim != p.KVLoRA+p.QkRope || st.IdxDim != p.IdxDim {
		panic(fmt.Sprintf("model: kimik31 dsa step-with-selection: state dims (%d,%d) != params ((kvlora+rope)%d, idxdim %d)",
			st.LatentDim, st.IdxDim, p.KVLoRA+p.QkRope, p.IdxDim))
	}
	if len(sel) == 0 {
		panic("model: kimik31 dsa step-with-selection: empty selection")
	}

	// Cache row first: the selection may reference the current cell.
	latent, ik := kimiK31DSACacheRow(p, x)
	st.Append(latent, ik)

	n := st.Len()
	prev := -1
	for _, t := range sel {
		if t < prev || t >= n {
			panic(fmt.Sprintf("model: kimik31 dsa step-with-selection: sel entry %d out of ordered range [0,%d)", t, n))
		}
		prev = t
	}

	// Shared layers skip the indexer but still need Qcur for the MLA.
	qr := kdaMatVecMul(p.WQa, x, p.QLoRA, p.Hidden)
	rmsnormInto(qr, qr, p.WQaNorm, p.RMSEps)
	qFull := kdaMatVecMul(p.WQb, qr, p.NumHeads*(p.QkNope+p.QkRope), p.QLoRA)
	out := make([]float32, p.NumHeads*p.VDim)
	kimiK31DSASparseMLA(st, p, qFull, sel, out)
	return kimiK31DSAGateAndOut(p, out, x)
}

// kimiK31DSACacheRow builds the two cache rows for one token: the absorbed-
// MLA latent concat(rmsnorm(kv_cmpr), k_pe) and the LayerNorm'd indexer key.
func kimiK31DSACacheRow(p *KimiK31DSAParams, x []float32) (latent, ik []float32) {
	latRaw := kdaMatVecMul(p.WKvA, x, p.KVLoRA+p.QkRope, p.Hidden)
	kvCmprN := make([]float32, p.KVLoRA)
	rmsnormInto(kvCmprN, latRaw[:p.KVLoRA], p.WKvANorm, p.RMSEps)
	latent = make([]float32, p.KVLoRA+p.QkRope)
	copy(latent, kvCmprN)
	copy(latent[p.KVLoRA:], latRaw[p.KVLoRA:])
	ik = layernorm(kdaMatVecMul(p.WIdxK, x, p.IdxDim, p.Hidden),
		p.IdxKNormW, p.IdxKNormB, p.IdxEps)
	return latent, ik
}

// ForwardKimiK31DSAPrefillSeq runs T tokens causally through the layer and
// returns the outputs for all T tokens ([T*Hidden]). The oracle scans
// token-by-token (each step sees exactly the causally-valid prefix), which
// reproduces llama.cpp's batched prefill with the additive causal mask.
func ForwardKimiK31DSAPrefillSeq(st *KimiK31DSALayerState, p *KimiK31DSAParams, xSeq []float32, T int) []float32 {
	if len(xSeq) != T*p.Hidden {
		panic(fmt.Sprintf("model: kimik31 dsa prefill: xSeq length %d != T*Hidden %d", len(xSeq), T*p.Hidden))
	}
	hid := p.Hidden
	out := make([]float32, T*hid)
	for t := 0; t < T; t++ {
		o := ForwardKimiK31DSAStep(st, p, xSeq[t*hid:(t+1)*hid])
		copy(out[t*hid:(t+1)*hid], o)
	}
	return out
}

// kimiK31DSAIndexerScores computes score[t] = sum_h iw[h]*ReLU(iq[h].ik[t])
// for every cached cell t. Heads accumulate outer, dims inner; non-positive
// ReLU terms are skipped (adding +0.0 is bit-exact in IEEE-754).
func kimiK31DSAIndexerScores(st *KimiK31DSALayerState, p *KimiK31DSAParams, iq, iw []float32) []float32 {
	n := st.Len()
	scores := make([]float32, n)
	for tau := 0; tau < n; tau++ {
		ikRow := st.IndexK[tau*p.IdxDim : (tau+1)*p.IdxDim]
		var s float32
		for h := 0; h < p.IdxHeads; h++ {
			iqRow := iq[h*p.IdxDim : (h+1)*p.IdxDim]
			var d float32
			for j := 0; j < p.IdxDim; j++ {
				d += iqRow[j] * ikRow[j]
			}
			if d > 0 {
				s += iw[h] * d
			}
		}
		scores[tau] = s
	}
	return scores
}

// kimiK31DSAAttendAndOut runs the sparse absorbed MLA over sel (ascending)
// and finishes the layer: optional sigmoid gate on the normed input x, then
// o_proj. It assumes the cache row for x was already appended and qFull is
// the current token's [NumHeads x (nope+rope)] query.
func kimiK31DSAAttendAndOut(st *KimiK31DSALayerState, p *KimiK31DSAParams, qFull []float32, sel []int, x []float32) []float32 {
	out := make([]float32, p.NumHeads*p.VDim)
	kimiK31DSASparseMLA(st, p, qFull, sel, out)
	return kimiK31DSAGateAndOut(p, out, x)
}

// kimiK31DSASparseMLA writes the per-head attention outputs into out
// ([NumHeads x VDim]): per head, the K absorption qAbs = WKb[h]^T*qNope,
// scores over the selected cells with kqScale = 1/sqrt(nope+rope) inside a
// max-subtracted F32 softmax, then the absorbed value projection
// out_h = sum_t w_t*(WVb[h]*lat_t).
func kimiK31DSASparseMLA(st *KimiK31DSALayerState, p *KimiK31DSAParams, qFull []float32, sel []int, out []float32) {
	headK := p.QkNope + p.QkRope
	kqScale := float32(1.0 / math.Sqrt(float64(headK)))
	vbHeadRows := p.VDim * p.KVLoRA
	for h := 0; h < p.NumHeads; h++ {
		qBase := h * headK
		qNope := qFull[qBase : qBase+p.QkNope]
		qPe := qFull[qBase+p.QkNope : qBase+headK]

		// K absorption: fold the per-head compressed-key projection into the
		// query side (llama.cpp line 547). WKb is [out=nope x in=KVLoRA] per
		// head, so qAbs[j] = sum_i WKb[h*nope+i, j]*qNope[i].
		qAbs := make([]float32, p.KVLoRA)
		for j := 0; j < p.KVLoRA; j++ {
			var s float32
			for i := 0; i < p.QkNope; i++ {
				s += p.WKb[(h*p.QkNope+i)*p.KVLoRA+j] * qNope[i]
			}
			qAbs[j] = s
		}

		nSel := len(sel)
		wts := make([]float32, nSel)
		for i, tau := range sel {
			row := tau * st.LatentDim
			s := dot(qAbs, st.Latent[row:row+p.KVLoRA]) +
				dot(qPe, st.Latent[row+p.KVLoRA:row+st.LatentDim])
			wts[i] = s * kqScale
		}
		kimiK31DSASoftmax(wts)

		wbOff := h * vbHeadRows
		for v := 0; v < p.VDim; v++ {
			var acc float32
			for i, tau := range sel {
				wi := wts[i]
				if wi == 0 {
					continue
				}
				row := tau * st.LatentDim
				wRow := wbOff + v*p.KVLoRA
				var vp float32
				for kv := 0; kv < p.KVLoRA; kv++ {
					vp += p.WVb[wRow+kv] * st.Latent[row+kv]
				}
				acc += wi * vp
			}
			out[h*p.VDim+v] = acc
		}
	}
}

// kimiK31DSASoftmax applies a max-subtracted softmax in place (F32
// accumulation; the kq scale is already folded into the input, matching
// ggml_soft_max_ext where the scale multiplies the scores before the exp).
func kimiK31DSASoftmax(s []float32) {
	if len(s) == 0 {
		return
	}
	m := s[0]
	for _, v := range s[1:] {
		if v > m {
			m = v
		}
	}
	var sum float32
	for i, v := range s {
		s[i] = float32(math.Exp(float64(v - m)))
		sum += s[i]
	}
	inv := float32(1.0 / float64(sum))
	for i := range s {
		s[i] *= inv
	}
}

// kimiK31DSAGateAndOut applies the optional sigmoid output gate (full-rank
// over the attention output, reading the normed input x — llama.cpp line
// 514 "the output gate reads the *normed* layer input") and the o_proj.
func kimiK31DSAGateAndOut(p *KimiK31DSAParams, out, x []float32) []float32 {
	if p.WGate != nil {
		g := kdaMatVecMul(p.WGate, x, p.NumHeads*p.VDim, p.Hidden)
		for i := range out {
			out[i] *= sigmoidf(g[i])
		}
	}
	return kdaMatVecMul(p.WO, out, p.Hidden, p.NumHeads*p.VDim)
}
