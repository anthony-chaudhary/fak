package model

import "sync"

func appendLayerKV(cache *KVCache, layer int, rawKey, key, value []float32) ([]float32, []float32) {
	cache.Kraw[layer] = append(cache.Kraw[layer], rawKey...)
	cache.K[layer] = append(cache.K[layer], key...)
	cache.V[layer] = append(cache.V[layer], value...)
	return cache.K[layer], cache.V[layer]
}

func preparePrefillAttention(cache *KVCache, layer int, rawKey, key, value []float32, cfg Config, rows, heads, headDim int) (keys, values []float32, window int, output []float32) {
	keys, values = appendLayerKV(cache, layer, rawKey, key, value)
	window = cfg.windowForLayer(layer)
	output = make([]float32, rows*heads*headDim)
	return
}

// prefill_attn.go — the balanced, allocation-free batched causal GQA attention used by the
// prefill paths. It replaces the per-token parFor that profiling (FAK_QPROFILE) exposed as
// the second-largest prefill cost — ~27% of Q8 prefill, despite attention being ~25x fewer
// MACs than the projection GEMMs. Two structural defects caused that:
//
//  1. TRIANGULAR LOAD IMBALANCE. Causal token t attends over base+t+1 positions, so its
//     work grows linearly in t. A contiguous parFor(P) hands worker W-1 the high-t tokens,
//     which do ~P/chunk more work than worker 0's low-t tokens; every core then waits on
//     that one slow chunk. Splitting (token,head) work units ROUND-ROBIN over the flat
//     u=t*nH+h index instead gives every worker an even spread of token positions, so the
//     per-worker totals match to within one unit (perfect balance is unnecessary; the
//     residual is the single most-expensive (t,h) unit, ~microseconds).
//
//  2. PER-(token,head) ALLOCATION. The old loop did `scores := make([]float32, nPos)` inside
//     the (t,h) loop — P*nH (=2304 at P=256) heap slices per layer, 30 layers per prefill.
//     Here each worker keeps ONE scratch buffer of the max width and reslices it per unit.
//
// The math is byte-for-byte the old computation — same scoreDot, same softmax, same
// in-order V accumulation per output — so the Q8 correctness gate (argmax-exact vs the HF
// oracle, logit-cosine vs f32) and the f32 oracle rungs are unaffected; only WHICH core
// computes WHICH (t,h) output, and the scratch lifetime, change.

// attnPrefillInto computes batched causal GQA attention for a prefill panel of P tokens into
// attnOut ([P, nH*hd], assumed zeroed by the caller). Q is [P, nH*hd]; Kl/Vl are the full KV
// cache as flat [nPos, nKV*hd] (stride w=nKV*hd). base is the number of cached positions
// before this panel (0 for a fresh prefill), so token t attends over base+t+1 positions.
// scoreDot is the (q·k) kernel (fdot in the fast path, dot in the legacy A/B path). grp is
// the GQA group size (nH/nKV) — passed in so it matches cfg.GroupSize() exactly. W is the
// layer's sliding-window bound (-1 = full causal, the default): query t (absolute base+t)
// then attends only keys [j0, base+t], j0=max(0, base+t-W+1). The prefill cache is
// contiguous (pos[j]==j), so the index is the absolute position; W=-1 keeps j0=0 exactly,
// reducing every byte to the pre-SWA loop. attnCap is the optional Gemma2 attention
// score soft-cap; zero keeps Llama-family scores unchanged.
func attnPrefillInto(attnOut, Q, Kl, Vl []float32, P, base, nH, hd, w, grp, W, layer int, scale, attnCap float32, scoreDot func(a, b []float32) float32, obs AttnObserver) {
	units := P * nH
	maxPos := base + P // widest scores row in this panel

	work := func(wkr, nw int) {
		scores := make([]float32, maxPos) // one scratch per worker, resliced per unit
		for u := wkr; u < units; u += nw {
			t := u / nH
			h := u % nH
			nPos := base + t + 1
			j0 := windowLoContig(nPos, base+t, W)
			kvh := h / grp
			qh := packedHead(Q, t, nH*hd, h, hd)
			sc := scores[:nPos-j0]
			fillAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
			softcapInPlace(sc, attnCap)
			softmaxInPlace(sc)
			if obs != nil { // #852: prefill token t sits at absolute position base+t
				emitAttnRow(obs, layer, base+t, h, j0, sc)
			}
			out := packedHead(attnOut, t, nH*hd, h, hd)
			accumulateAttentionValues(out, Vl, sc, j0, nPos, w, kvh, hd)
		}
	}

	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 1 {
		work(0, 1)
		return
	}
	var wg sync.WaitGroup
	for k := 0; k < nw; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			work(k, nw)
		}(k)
	}
	wg.Wait()
}

// saxpy does out += a*x over the full length of out (== len(x)). Split into 8 independent
// updates so the compiler can keep the lanes in flight (the per-output accumulation order is
// unchanged vs the old scalar `out[d] += wj*vh[d]` loop: each out[d] still receives exactly
// one a*x[d] term per call, in the same call sequence). hd is a multiple of 8 (64) here.
func saxpy(out, x []float32, a float32) {
	n := len(out)
	i := 0
	for ; i+8 <= n; i += 8 {
		out[i] += a * x[i]
		out[i+1] += a * x[i+1]
		out[i+2] += a * x[i+2]
		out[i+3] += a * x[i+3]
		out[i+4] += a * x[i+4]
		out[i+5] += a * x[i+5]
		out[i+6] += a * x[i+6]
		out[i+7] += a * x[i+7]
	}
	for ; i < n; i++ {
		out[i] += a * x[i]
	}
}

// packedHead returns one head from a row-major panel whose rows have rowWidth
// elements. Keeping the stride arithmetic here prevents the decode, prefill, and
// verification attention loops from each transcribing the same slice bounds.
func packedHead(panel []float32, row, rowWidth, head, headDim int) []float32 {
	start := row*rowWidth + head*headDim
	return panel[start : start+headDim]
}

func vectorHead(vector []float32, head, headDim int) []float32 {
	return packedHead(vector, 0, len(vector), head, headDim)
}

func packedHead3(panel []float32, row, rowWidth, firstHead, headDim int) ([]float32, []float32, []float32) {
	return packedHead(panel, row, rowWidth, firstHead, headDim),
		packedHead(panel, row, rowWidth, firstHead+1, headDim),
		packedHead(panel, row, rowWidth, firstHead+2, headDim)
}

func zeroPackedHeads(panel []float32, row, rowWidth, firstHead, heads, headDim int) {
	for head := firstHead; head < firstHead+heads; head++ {
		clear(packedHead(panel, row, rowWidth, head, headDim))
	}
}

func scoreScratchHead(scratch [][]float32, worker, groupSize, groupHead, n int) []float32 {
	return scratch[worker*groupSize+groupHead][:n]
}

func scoreScratchHead3(scratch [][]float32, worker, groupSize, n int) ([]float32, []float32, []float32) {
	return scoreScratchHead(scratch, worker, groupSize, 0, n),
		scoreScratchHead(scratch, worker, groupSize, 1, n),
		scoreScratchHead(scratch, worker, groupSize, 2, n)
}

func fillAttentionScores(scores, query, keys []float32, first, end, rowWidth, kvHead, headDim int, scale float32, scoreDot func(a, b []float32) float32) {
	for pos := first; pos < end; pos++ {
		scores[pos-first] = scoreDot(query, packedHead(keys, pos, rowWidth, kvHead, headDim)) * scale
	}
}

func fillAttentionScores3(scores0, scores1, scores2, query0, query1, query2, keys []float32, first, end, rowWidth, kvHead, headDim int, scale float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32)) {
	for pos := first; pos < end; pos++ {
		s0, s1, s2 := scoreDot3(query0, query1, query2, packedHead(keys, pos, rowWidth, kvHead, headDim))
		i := pos - first
		scores0[i], scores1[i], scores2[i] = s0*scale, s1*scale, s2*scale
	}
}

func fillSoftmaxAttentionScores(scores, query, keys []float32, first, end, rowWidth, kvHead, headDim int, scale float32, scoreDot func(a, b []float32) float32) {
	fillAttentionScores(scores, query, keys, first, end, rowWidth, kvHead, headDim, scale, scoreDot)
	softmaxInPlace(scores)
}

func fillSoftmaxAttentionScores3(scores0, scores1, scores2, query0, query1, query2, keys []float32, first, end, rowWidth, kvHead, headDim int, scale float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32)) {
	fillAttentionScores3(scores0, scores1, scores2, query0, query1, query2, keys, first, end, rowWidth, kvHead, headDim, scale, scoreDot3)
	softmaxInPlace(scores0)
	softmaxInPlace(scores1)
	softmaxInPlace(scores2)
}

func accumulateAttentionValues(out, values, scores []float32, first, end, rowWidth, kvHead, headDim int) {
	for pos := first; pos < end; pos++ {
		saxpy(out, packedHead(values, pos, rowWidth, kvHead, headDim), scores[pos-first])
	}
}

// accumulateAttentionGroup preserves the generic GQA traversal: stream each V
// head once per position, then update every query head that shares it. Keeping
// position and dimension outside the group loop avoids rereading the KV cache
// groupSize times while preserving each output element's position order.
func accumulateAttentionGroup(outPanel []float32, outRow, outRowWidth, firstHead, groupSize, headDim int, values []float32, scores [][]float32, scoreBase, first, end, valueRowWidth, kvHead int) {
	for pos := first; pos < end; pos++ {
		value := packedHead(values, pos, valueRowWidth, kvHead, headDim)
		i := pos - first
		for dim, x := range value {
			for groupHead := 0; groupHead < groupSize; groupHead++ {
				out := packedHead(outPanel, outRow, outRowWidth, firstHead+groupHead, headDim)
				out[dim] += scores[scoreBase+groupHead][i] * x
			}
		}
	}
}

func accumulateAttentionValues3(out0, out1, out2, values, scores0, scores1, scores2 []float32, first, end, rowWidth, kvHead, headDim int, useSIMD bool) {
	for pos := first; pos < end; pos++ {
		value := packedHead(values, pos, rowWidth, kvHead, headDim)
		i := pos - first
		if useSIMD {
			saxpy3(out0, out1, out2, value, scores0[i], scores1[i], scores2[i])
		} else {
			saxpy3scalar(out0, out1, out2, value, scores0[i], scores1[i], scores2[i])
		}
	}
}

func accumulatePackedAttentionValues3(outPanel []float32, outRow, outRowWidth, firstHead, headDim int, values, scores0, scores1, scores2 []float32, first, end, valueRowWidth, kvHead int, useSIMD bool) {
	out0, out1, out2 := packedHead3(outPanel, outRow, outRowWidth, firstHead, headDim)
	accumulateAttentionValues3(out0, out1, out2, values, scores0, scores1, scores2, first, end, valueRowWidth, kvHead, headDim, useSIMD)
}

func saxpy3(out0, out1, out2, x []float32, a0, a1, a2 float32) {
	if saxpy3Fast(out0, out1, out2, x, a0, a1, a2) {
		return
	}
	saxpy3scalar(out0, out1, out2, x, a0, a1, a2)
}

func saxpy3scalar(out0, out1, out2, x []float32, a0, a1, a2 float32) {
	n := len(out0)
	i := 0
	for ; i+8 <= n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		out0[i] += a0 * x0
		out1[i] += a1 * x0
		out2[i] += a2 * x0
		out0[i+1] += a0 * x1
		out1[i+1] += a1 * x1
		out2[i+1] += a2 * x1
		out0[i+2] += a0 * x2
		out1[i+2] += a1 * x2
		out2[i+2] += a2 * x2
		out0[i+3] += a0 * x3
		out1[i+3] += a1 * x3
		out2[i+3] += a2 * x3
		out0[i+4] += a0 * x4
		out1[i+4] += a1 * x4
		out2[i+4] += a2 * x4
		out0[i+5] += a0 * x5
		out1[i+5] += a1 * x5
		out2[i+5] += a2 * x5
		out0[i+6] += a0 * x6
		out1[i+6] += a1 * x6
		out2[i+6] += a2 * x6
		out0[i+7] += a0 * x7
		out1[i+7] += a1 * x7
		out2[i+7] += a2 * x7
	}
	for ; i < n; i++ {
		v := x[i]
		out0[i] += a0 * v
		out1[i] += a1 * v
		out2[i] += a2 * v
	}
}
