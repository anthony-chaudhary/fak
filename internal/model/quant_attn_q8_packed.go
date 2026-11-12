package model

// The block reconstruction below adapts the Q8_0 row dequantizer from
// ggml-org/llama.cpp at 83078fec0db82d6b5a00d9599062c38c39145755 (MIT).
// Copyright (c) 2023-2026 The ggml authors. License and notices:
// THIRD_PARTY_NOTICES.md ("llama.cpp").
// Fak uses fp32 scales rather than the donor's fp16 scales. Each worker reconstructs
// one head row into bounded F32 scratch, reused by every sharing query head.
// No full-layer K/V materialization or query quantization is performed.

// attnDecodeQ8Packed consumes contiguous B=1 host Q8 KV with the existing optimized
// Q8-weight decode's native fdot/fdot3, softmax and value accumulation policy.
// It is not for QSA, ALiBi, sinks, softcaps, sparse positions, or device KV.
// Caller owns exclusive cache access through worker join; output, query, score
// scratch and cache storage must be disjoint. False is returned only before any
// mutation; selected faults propagate and must never trigger host replay.
// The tile is O(workers*headDim) F32, not zero F32 scratch. This replaces the failed
// scalar packed reduction; arithmetic equivalence still requires qualification.
func attnDecodeQ8Packed(out, query []float32, cache *KVCache, layer, nH, hd, w, grp, window int, scale float32, scratch [][]float32, obs AttnObserver) ([][]float32, bool) {
	if cache == nil || !cache.quantized() || layer < 0 || layer >= len(cache.kQ8) || layer >= len(cache.vQ8) ||
		nH <= 0 || hd <= 0 || grp <= 0 || nH%grp != 0 || hd%kvQ8_0GroupSize != 0 ||
		nH > len(query)/hd || nH > len(out)/hd || w <= 0 || w/hd != nH/grp || w%hd != 0 {
		return scratch, false
	}
	keys, values := &cache.kQ8[layer], &cache.vQ8[layer]
	if keys.width != w || values.width != w || len(keys.codes)%w != 0 ||
		len(values.codes) != len(keys.codes) || len(keys.scales) != len(keys.codes)/kvQ8_0GroupSize ||
		len(values.scales) != len(keys.scales) {
		return scratch, false
	}
	nPos := len(keys.codes) / w
	first := windowLoContig(nPos, nPos-1, window)
	visible := nPos - first
	if visible <= 0 {
		return scratch, false
	}
	nKV := nH / grp
	nw := currentWorkerCount()
	if nw < 1 {
		nw = 1
	}
	if nw > nKV {
		nw = nKV
	}
	scratch = grow2D(scratch, nw*grp, visible)
	// One head row per worker, reused for K then V. Unlike whole-layer expansion,
	// this allocation is independent of context length and number of cache heads.
	tiles := make([]float32, nw*hd)
	scoreDot3 := fdot3scalar
	if attnFdot3SIMD {
		scoreDot3 = fdot3SIMD
	}
	useSaxpy3SIMD := 1 >= attnSaxpy3SIMDMinBatch && visible >= attnSaxpy3SIMDMinPos
	work := func(worker int) {
		tile := tiles[worker*hd : (worker+1)*hd]
		for kvh := worker; kvh < nKV; kvh += nw {
			// Decode K once per position, not once per query head. Native dot
			// callbacks retain their reduction tree and SIMD selection unchanged.
			for pos := first; pos < nPos; pos++ {
				q8PackedHeadTile(tile, keys, pos*w+kvh*hd)
				i := pos - first
				if attnGQAFuse && grp == 3 {
					q0, q1, q2 := packedHead3(query, 0, nH*hd, kvh*grp, hd)
					s0, s1, s2 := scoreDot3(q0, q1, q2, tile)
					scratch[worker*grp][i] = s0 * scale
					scratch[worker*grp+1][i] = s1 * scale
					scratch[worker*grp+2][i] = s2 * scale
				} else {
					for g := 0; g < grp; g++ {
						h := kvh*grp + g
						scratch[worker*grp+g][i] = fdot(query[h*hd:(h+1)*hd], tile) * scale
					}
				}
			}
			for g := 0; g < grp; g++ {
				scores := scratch[worker*grp+g][:visible]
				softmaxInPlace(scores)
				if obs != nil {
					emitAttnRow(obs, layer, nPos-1, kvh*grp+g, first, scores)
				}
			}
			zeroPackedHeads(out, 0, nH*hd, kvh*grp, grp, hd)
			for pos := first; pos < nPos; pos++ {
				q8PackedHeadTile(tile, values, pos*w+kvh*hd)
				i := pos - first
				if grp == 3 {
					out0, out1, out2 := packedHead3(out, 0, nH*hd, kvh*grp, hd)
					s0, s1, s2 := scratch[worker*grp][i], scratch[worker*grp+1][i], scratch[worker*grp+2][i]
					if useSaxpy3SIMD {
						saxpy3(out0, out1, out2, tile, s0, s1, s2)
					} else {
						saxpy3scalar(out0, out1, out2, tile, s0, s1, s2)
					}
				} else {
					// Exact generic accumulateAttentionGroup traversal/expression;
					// do not add an explicit product rounding or substitute saxpy.
					for dim, x := range tile {
						for g := 0; g < grp; g++ {
							head := packedHead(out, 0, nH*hd, kvh*grp+g, hd)
							head[dim] += scratch[worker*grp+g][i] * x
						}
					}
				}
			}
		}
	}
	if nw == 1 {
		work(0)
	} else {
		parFor(nw, nw, func(lo, hi int) {
			for worker := lo; worker < hi; worker++ {
				work(worker)
			}
		})
	}
	return scratch, true
}

// Reconstruct exactly the same fp32 elements as decodeRangeInto, loading each
// block scale once. The complete head tile is available to native vector kernels.
func q8PackedHeadTile(tile []float32, rows *kvPackedRow, base int) {
	for off := 0; off < len(tile); off += kvQ8_0GroupSize {
		d := rows.scales[(base+off)/kvQ8_0GroupSize]
		codes := rows.codes[base+off : base+off+kvQ8_0GroupSize]
		for j, code := range codes {
			tile[off+j] = float32(code) * d
		}
	}
}
