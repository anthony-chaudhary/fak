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
	nw := numWorkers
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
				q0 := Q[b*nH*hd+h0*hd : b*nH*hd+(h0+1)*hd]
				q1 := Q[b*nH*hd+(h0+1)*hd : b*nH*hd+(h0+2)*hd]
				q2 := Q[b*nH*hd+(h0+2)*hd : b*nH*hd+(h0+3)*hd]
				sc0 := scoreScratch[wkr*grp+0][:visible]
				sc1 := scoreScratch[wkr*grp+1][:visible]
				sc2 := scoreScratch[wkr*grp+2][:visible]
				for j := j0; j < nPos; j++ {
					kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
					s0, s1, s2 := scoreDot3(q0, q1, q2, kh)
					i := j - j0
					sc0[i] = s0 * scale
					sc1[i] = s1 * scale
					sc2[i] = s2 * scale
				}
				softmaxInPlace(sc0)
				softmaxInPlace(sc1)
				softmaxInPlace(sc2)
				if obs != nil { // #852: query is the just-appended row at abs pos nPos-1
					emitAttnRow(obs, l, nPos-1, h0+0, j0, sc0)
					emitAttnRow(obs, l, nPos-1, h0+1, j0, sc1)
					emitAttnRow(obs, l, nPos-1, h0+2, j0, sc2)
				}
				for g := 0; g < grp; g++ {
					h := h0 + g
					out := attnOut[b*nH*hd+h*hd : b*nH*hd+(h+1)*hd]
					for i := range out {
						out[i] = 0
					}
				}
			} else {
				for g := 0; g < grp; g++ {
					h := kvh*grp + g
					qh := Q[b*nH*hd+h*hd : b*nH*hd+(h+1)*hd]
					sc := scoreScratch[wkr*grp+g][:visible]
					for j := j0; j < nPos; j++ {
						kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
						sc[j-j0] = scoreDot(qh, kh) * scale
					}
					softmaxInPlace(sc)
					if obs != nil { // #852: query is the just-appended row at abs pos nPos-1
						emitAttnRow(obs, l, nPos-1, h, j0, sc)
					}
					out := attnOut[b*nH*hd+h*hd : b*nH*hd+(h+1)*hd]
					for i := range out {
						out[i] = 0
					}
				}
			}
			if grp == 3 {
				out0 := attnOut[b*nH*hd+(kvh*grp+0)*hd : b*nH*hd+(kvh*grp+1)*hd]
				out1 := attnOut[b*nH*hd+(kvh*grp+1)*hd : b*nH*hd+(kvh*grp+2)*hd]
				out2 := attnOut[b*nH*hd+(kvh*grp+2)*hd : b*nH*hd+(kvh*grp+3)*hd]
				sc0 := scoreScratch[wkr*grp+0][:visible]
				sc1 := scoreScratch[wkr*grp+1][:visible]
				sc2 := scoreScratch[wkr*grp+2][:visible]
				if !useSaxpy3SIMD || visible < attnSaxpy3SIMDMinPos {
					for j := j0; j < nPos; j++ {
						vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
						i := j - j0
						a0, a1, a2 := sc0[i], sc1[i], sc2[i]
						d := 0
						for ; d+8 <= hd; d += 8 {
							x0, x1, x2, x3 := vh[d], vh[d+1], vh[d+2], vh[d+3]
							x4, x5, x6, x7 := vh[d+4], vh[d+5], vh[d+6], vh[d+7]
							out0[d] += a0 * x0
							out1[d] += a1 * x0
							out2[d] += a2 * x0
							out0[d+1] += a0 * x1
							out1[d+1] += a1 * x1
							out2[d+1] += a2 * x1
							out0[d+2] += a0 * x2
							out1[d+2] += a1 * x2
							out2[d+2] += a2 * x2
							out0[d+3] += a0 * x3
							out1[d+3] += a1 * x3
							out2[d+3] += a2 * x3
							out0[d+4] += a0 * x4
							out1[d+4] += a1 * x4
							out2[d+4] += a2 * x4
							out0[d+5] += a0 * x5
							out1[d+5] += a1 * x5
							out2[d+5] += a2 * x5
							out0[d+6] += a0 * x6
							out1[d+6] += a1 * x6
							out2[d+6] += a2 * x6
							out0[d+7] += a0 * x7
							out1[d+7] += a1 * x7
							out2[d+7] += a2 * x7
						}
						for ; d < hd; d++ {
							v := vh[d]
							out0[d] += a0 * v
							out1[d] += a1 * v
							out2[d] += a2 * v
						}
					}
					continue
				}
				for j := j0; j < nPos; j++ {
					vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
					i := j - j0
					saxpy3(out0, out1, out2, vh, sc0[i], sc1[i], sc2[i])
				}
				continue
			}
			for j := j0; j < nPos; j++ {
				vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				for d := 0; d < hd; d++ {
					v := vh[d]
					for g := 0; g < grp; g++ {
						h := kvh*grp + g
						out := attnOut[b*nH*hd+h*hd : b*nH*hd+(h+1)*hd]
						out[d] += scoreScratch[wkr*grp+g][j-j0] * v
					}
				}
			}
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
	nw := numWorkers
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
			qh := Q[row*nH*hd+h*hd : row*nH*hd+(h+1)*hd]
			sc := scores[:nPos-j0]
			for j := j0; j < nPos; j++ {
				kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				sc[j-j0] = scoreDot(qh, kh) * scale
			}
			softmaxInPlace(sc)
			out := attnOut[row*nH*hd+h*hd : row*nH*hd+(h+1)*hd]
			for j := j0; j < nPos; j++ {
				vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				saxpy(out, vh, sc[j-j0])
			}
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
	nw := numWorkers
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
				q0 := Q[row*nH*hd+h0*hd : row*nH*hd+(h0+1)*hd]
				q1 := Q[row*nH*hd+(h0+1)*hd : row*nH*hd+(h0+2)*hd]
				q2 := Q[row*nH*hd+(h0+2)*hd : row*nH*hd+(h0+3)*hd]
				sc0 := scoreScratch[wkr*grp+0][:span]
				sc1 := scoreScratch[wkr*grp+1][:span]
				sc2 := scoreScratch[wkr*grp+2][:span]
				for j := j0; j < nPos; j++ {
					kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
					s0, s1, s2 := scoreDot3(q0, q1, q2, kh)
					i := j - j0
					sc0[i] = s0 * scale
					sc1[i] = s1 * scale
					sc2[i] = s2 * scale
				}
				softmaxInPlace(sc0)
				softmaxInPlace(sc1)
				softmaxInPlace(sc2)
				out0 := attnOut[row*nH*hd+(h0+0)*hd : row*nH*hd+(h0+1)*hd]
				out1 := attnOut[row*nH*hd+(h0+1)*hd : row*nH*hd+(h0+2)*hd]
				out2 := attnOut[row*nH*hd+(h0+2)*hd : row*nH*hd+(h0+3)*hd]
				if !useSaxpy3SIMD || span < attnSaxpy3SIMDMinPos {
					for j := j0; j < nPos; j++ {
						vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
						i := j - j0
						a0, a1, a2 := sc0[i], sc1[i], sc2[i]
						for d, v := range vh {
							out0[d] += a0 * v
							out1[d] += a1 * v
							out2[d] += a2 * v
						}
					}
					continue
				}
				for j := j0; j < nPos; j++ {
					vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
					i := j - j0
					saxpy3(out0, out1, out2, vh, sc0[i], sc1[i], sc2[i])
				}
				continue
			}

			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				qh := Q[row*nH*hd+h*hd : row*nH*hd+(h+1)*hd]
				sc := scoreScratch[wkr*grp+g][:span]
				for j := j0; j < nPos; j++ {
					kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
					sc[j-j0] = scoreDot(qh, kh) * scale
				}
				softmaxInPlace(sc)
			}
			if grp == 3 {
				h0 := kvh * grp
				out0 := attnOut[row*nH*hd+(h0+0)*hd : row*nH*hd+(h0+1)*hd]
				out1 := attnOut[row*nH*hd+(h0+1)*hd : row*nH*hd+(h0+2)*hd]
				out2 := attnOut[row*nH*hd+(h0+2)*hd : row*nH*hd+(h0+3)*hd]
				sc0 := scoreScratch[wkr*grp+0][:span]
				sc1 := scoreScratch[wkr*grp+1][:span]
				sc2 := scoreScratch[wkr*grp+2][:span]
				if !useSaxpy3SIMD || span < attnSaxpy3SIMDMinPos {
					for j := j0; j < nPos; j++ {
						vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
						i := j - j0
						a0, a1, a2 := sc0[i], sc1[i], sc2[i]
						for d, v := range vh {
							out0[d] += a0 * v
							out1[d] += a1 * v
							out2[d] += a2 * v
						}
					}
					continue
				}
				for j := j0; j < nPos; j++ {
					vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
					i := j - j0
					saxpy3(out0, out1, out2, vh, sc0[i], sc1[i], sc2[i])
				}
				continue
			}
			for j := j0; j < nPos; j++ {
				vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				i := j - j0
				for d, v := range vh {
					for g := 0; g < grp; g++ {
						h := kvh*grp + g
						out := attnOut[row*nH*hd+h*hd : row*nH*hd+(h+1)*hd]
						out[d] += scoreScratch[wkr*grp+g][i] * v
					}
				}
			}
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
