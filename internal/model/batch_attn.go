package model

import "sync"

// attnDecodeBatch computes per-user causal attention for ONE batched decode step. For each
// user b, the query Q[b] attends over user b's OWN full KV cache (caches[b].K[l]/V[l]); the
// result [nH*hd] is written into attnOut[b] (which the caller has zeroed). Users are fully
// independent — own cache, own length — so the work parallelises across the flat
// (user,kv-head) unit index. Each unit computes the grp query heads sharing one V head, then
// streams each value vector once while updating those grp output heads. For every individual
// output element, the j=0..nPos-1 accumulation order is unchanged, so StepBatch stays
// bit-identical to serial Step. This is the decode analogue of attnPrefillInto.
//
// W is the per-layer sliding-window bound (cfg.windowForLayer): W<0 (the default) is full
// causal attention and reduces the loops byte-for-byte to the pre-SWA path; W>=0 masks the
// score/V loops to the contiguous visible suffix (windowLoContig). This SWA mask layers on
// top of the GQA-fuse + saxpy3-SIMD perf branches — they coexist.
func attnDecodeBatch(attnOut, Q []float32, caches []*KVCache, l, B, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32), scoreScratch [][]float32, obs AttnObserver) [][]float32 {
	nKV := nH / grp
	units := B * nKV
	maxPos := 0
	for b := 0; b < B; b++ {
		if n := len(caches[b].K[l]) / w; n > maxPos {
			maxPos = n
		}
	}
	useSaxpy3SIMD := B >= attnSaxpy3SIMDMinBatch
	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 0 {
		return scoreScratch
	}
	scoreScratch = grow2D(scoreScratch, nw*grp, maxPos)
	work := func(wkr, nw int) {
		for u := wkr; u < units; u += nw {
			b := u / nKV
			kvh := u % nKV
			c := caches[b]
			Kl, Vl := c.K[l], c.V[l]
			nPos := len(Kl) / w
			// SWA read-time mask: this user's query (its just-appended K row, at absolute
			// position nPos-1 since the cache is contiguous and was appended at Cache.Len())
			// attends only keys in the window. j0=0 (full causal) when W<0.
			j0 := windowLoContig(nPos, nPos-1, W)
			visible := nPos - j0
			if attnGQAFuse && grp == 3 && scoreDot3 != nil {
				h0 := kvh * grp
				q0, q1, q2 := packedHead3(Q, b, nH*hd, h0, hd)
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, visible)
				fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0, q1, q2, Kl, j0, nPos, w, kvh, hd, scale, scoreDot3)
				if obs != nil { // #852: query is the just-appended row at abs pos nPos-1
					emitAttnRow(obs, l, nPos-1, h0+0, j0, sc0)
					emitAttnRow(obs, l, nPos-1, h0+1, j0, sc1)
					emitAttnRow(obs, l, nPos-1, h0+2, j0, sc2)
				}
			} else {
				for g := 0; g < grp; g++ {
					h := kvh*grp + g
					qh := packedHead(Q, b, nH*hd, h, hd)
					sc := scoreScratchHead(scoreScratch, wkr, grp, g, visible)
					fillSoftmaxAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
					if obs != nil { // #852: query is the just-appended row at abs pos nPos-1
						emitAttnRow(obs, l, nPos-1, h, j0, sc)
					}
				}
			}
			// Clear this kv-head's query-head output rows before the value accumulation
			// below adds into them. Both score branches above leave attnOut untouched, and
			// the fused branch's h0 is kvh*grp, so the rows to clear are the same set either
			// way — one loop, not one per branch.
			zeroPackedHeads(attnOut, b, nH*hd, kvh*grp, grp, hd)
			if grp == 3 {
				h0 := kvh * grp
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, visible)
				accumulatePackedAttentionValues3(attnOut, b, nH*hd, h0, hd, Vl, sc0, sc1, sc2, j0, nPos, w, kvh,
					useSaxpy3SIMD && visible >= attnSaxpy3SIMDMinPos)
				continue
			}
			accumulateAttentionGroup(attnOut, b, nH*hd, kvh*grp, grp, hd, Vl, scoreScratch, wkr*grp, j0, nPos, w, kvh)
		}
	}

	if nw <= 1 {
		work(0, 1)
		return scoreScratch
	}
	parFor(nw, nw, func(lo, hi int) {
		for k := lo; k < hi; k++ {
			work(k, nw)
		}
	})
	return scoreScratch
}

// attnPrefillMultiInto computes causal GQA attention for a rectangular multi-sequence
// prefill panel. Rows are laid out [user][token], with P new tokens per user. Each row
// attends only to that user's own cache prefix plus earlier rows from the same user; other
// users' K/V rows are never visible. This is the multi-agent analogue of attnPrefillInto.
//
// W is the per-layer sliding-window bound (cfg.windowForLayer): W<0 is full causal and the
// score/V loops reduce byte-for-byte to the pre-SWA path; W>=0 masks each query to the
// contiguous visible suffix (windowLoContig). scoreScratch is a reusable per-worker softmax
// scratch (one row per worker); it is grown as needed and returned so the caller can pool it
// across layers/calls (pass nil to allocate fresh). attnOut is written per (row,head) output
// slice via saxpy, which `+=`-accumulates, so the caller MUST zero any region attnOut covers
// before calling — load-bearing when attnOut is a reused buffer.
func attnPrefillMultiInto(attnOut, Q []float32, caches []*KVCache, baseB []int, layer, P, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreScratch [][]float32) [][]float32 {
	B := len(caches)
	units := B * P * nH
	maxPos := 0
	for b := 0; b < B; b++ {
		if n := baseB[b] + P; n > maxPos {
			maxPos = n
		}
	}
	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 0 {
		return scoreScratch
	}
	scoreScratch = grow2D(scoreScratch, nw, maxPos)

	work := func(wkr, nw int) {
		scores := scoreScratch[wkr][:maxPos]
		for u := wkr; u < units; u += nw {
			row := u / nH
			h := u % nH
			b := row / P
			t := row % P
			c := caches[b]
			Kl, Vl := c.K[layer], c.V[layer]
			nPos := baseB[b] + t + 1
			// SWA read-time mask: query (absolute position baseB[b]+t) over the contiguous
			// prefill cache. j0=0 (full causal) when W<0.
			j0 := windowLoContig(nPos, baseB[b]+t, W)
			kvh := h / grp
			qh := packedHead(Q, row, nH*hd, h, hd)
			sc := scores[:nPos-j0]
			fillAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
			softmaxInPlace(sc)
			out := packedHead(attnOut, row, nH*hd, h, hd)
			accumulateAttentionValues(out, Vl, sc, j0, nPos, w, kvh, hd)
		}
	}

	if nw == 1 {
		work(0, 1)
		return scoreScratch
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
	return scoreScratch
}

// attnPrefillMultiGQAInto is the GQA-fused prefill analogue of attnDecodeBatch's fast path.
// It accepts the same sliding-window W bound as attnPrefillMultiInto; W<0 is full causal.
func attnPrefillMultiGQAInto(attnOut, Q []float32, caches []*KVCache, baseB []int, layer, P, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32), scoreScratch [][]float32) [][]float32 {
	B := len(caches)
	nKV := nH / grp
	units := B * P * nKV
	maxPos := 0
	for b := 0; b < B; b++ {
		if n := baseB[b] + P; n > maxPos {
			maxPos = n
		}
	}
	nw := currentWorkerCount()
	if nw > units {
		nw = units
	}
	if nw <= 0 {
		return scoreScratch
	}
	scoreScratch = grow2D(scoreScratch, nw*grp, maxPos)
	useSaxpy3SIMD := B >= attnSaxpy3SIMDMinBatch

	work := func(wkr, nw int) {
		for u := wkr; u < units; u += nw {
			row := u / nKV
			kvh := u % nKV
			b := row / P
			t := row % P
			c := caches[b]
			Kl, Vl := c.K[layer], c.V[layer]
			nPos := baseB[b] + t + 1
			j0 := windowLoContig(nPos, baseB[b]+t, W)
			span := nPos - j0
			if attnGQAFuse && grp == 3 && scoreDot3 != nil {
				h0 := kvh * grp
				q0, q1, q2 := packedHead3(Q, row, nH*hd, h0, hd)
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, span)
				fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0, q1, q2, Kl, j0, nPos, w, kvh, hd, scale, scoreDot3)
			} else {
				for g := 0; g < grp; g++ {
					h := kvh*grp + g
					qh := packedHead(Q, row, nH*hd, h, hd)
					sc := scoreScratchHead(scoreScratch, wkr, grp, g, span)
					fillSoftmaxAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
				}
			}
			if grp == 3 {
				h0 := kvh * grp
				sc0, sc1, sc2 := scoreScratchHead3(scoreScratch, wkr, grp, span)
				accumulatePackedAttentionValues3(attnOut, row, nH*hd, h0, hd, Vl, sc0, sc1, sc2, j0, nPos, w, kvh,
					useSaxpy3SIMD && span >= attnSaxpy3SIMDMinPos)
				continue
			}
			accumulateAttentionGroup(attnOut, row, nH*hd, kvh*grp, grp, hd, Vl, scoreScratch, wkr*grp, j0, nPos, w, kvh)
		}
	}

	if nw == 1 {
		work(0, 1)
		return scoreScratch
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
	return scoreScratch
}
