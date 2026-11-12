package model

import (
	"math"
	"sync"
	"unsafe"
)

// Adapted from ggml-org/llama.cpp at 74bca3950054eed1795f525cb9cf06bfe796492d (MIT).
// License and notices: THIRD_PARTY_NOTICES.md ("llama.cpp").
// The borrowed unit is the multi-query/KV tile, packed K/V reuse, and running
// maximum/sum/output rescale. FAK keeps its existing fdot and saxpy primitives,
// F32 activations, causal/window indexing and softcap expression. This is not
// the donor's SIMD GEMM, quantized KV, sink, or ALiBi implementation.
//
// Online normalization changes reduction order. It is qualification-only, not
// bit-exact attention or a measured speedup. Validation and packing are included
// in the caller's existing attention phase; no global cache or retained owner.
const (
	qwen35AttentionQueryTile = 8
	qwen35AttentionKVTile    = 32
)

// qwen35PrefillTiledAttentionInto may borrow only a builtin producer's dead,
// exclusively owned qf allocation. false means NO writes to any argument.
// Success overwrites the full output and joins all workers before returning.
// Observer/device/custom-callback routes retain their original implementation.
func qwen35PrefillTiledAttentionInto(out, q, k, v, scratch []float32, rows, base, heads, dim, kvWidth, group, window int, scale, cap float32, obs AttnObserver) bool {
	if !qwen35PrefillTiledAttentionQualification {
		return false
	}
	return qwen35PrefillTiledAttentionRun(out, q, k, v, scratch, rows, base, heads, dim, kvWidth, group, window, scale, cap, obs, currentWorkerCount())
}

func qwen35PrefillTiledAttentionRun(out, q, k, v, scratch []float32, rows, base, heads, dim, kvWidth, group, window int, scale, cap float32, obs AttnObserver, workers int) bool {
	if obs != nil || rows < 2 || base < 0 || heads < 1 || dim < 1 || group < 1 || heads%group != 0 || window == 0 || workers < 1 {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if base > maxInt-rows || heads > maxInt/dim || rows > maxInt/heads {
		return false
	}
	kvHeads, width, positions := heads/group, heads*dim, base+rows
	if kvWidth != kvHeads*dim || rows > len(q)/width || rows > len(out)/width || positions > len(k)/kvWidth || positions > len(v)/kvWidth {
		return false
	}
	// The division guards cover every scratch product, without extending a
	// short loan to its capacity. The original worker budget is never reduced
	// merely to make the candidate fit.
	const fixed = qwen35AttentionQueryTile*qwen35AttentionKVTile + 2*qwen35AttentionQueryTile
	const perDim = 2*qwen35AttentionKVTile + qwen35AttentionQueryTile
	if dim > (maxInt-fixed)/perDim {
		return false
	}
	workerWidth := perDim*dim + fixed
	queries := rows * group // bounded by checked rows*heads
	tiles := queries / qwen35AttentionQueryTile
	if queries%qwen35AttentionQueryTile != 0 {
		tiles++
	}
	units := kvHeads * tiles // tiles <= rows*group, hence <= rows*heads
	if workers > rows*heads {
		workers = rows * heads
	}
	if units < workers || workers > len(scratch)/workerWidth {
		return false
	}
	out, q = out[:rows*width], q[:rows*width]
	k, v = k[:positions*kvWidth], v[:positions*kvWidth]
	scratch = scratch[:workers*workerWidth]
	for _, read := range [][]float32{q, k, v} {
		if qwen35AttentionRowsOverlap(out, read) || qwen35AttentionRowsOverlap(scratch, read) {
			return false
		}
	}
	if qwen35AttentionRowsOverlap(out, scratch) {
		return false
	}
	// Preflight ALL exceptional-data refusals before either scratch or output
	// publication. Returning false after a partial tile would corrupt the
	// legacy zero-output oracle. Bounds conservatively exclude possible F32
	// dot/scale or unnormalized-value overflow; unsupported inputs use the
	// untouched legacy path, including its NaN/Inf behavior.
	qmax, qok := qwen35AttentionFiniteMax(q)
	kmax, kok := qwen35AttentionFiniteMax(k)
	vmax, vok := qwen35AttentionFiniteMax(v)
	if !qok || !kok || !vok || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || math.IsNaN(float64(cap)) || math.IsInf(float64(cap), 0) {
		return false
	}
	const safe = float64(math.MaxFloat32) / 4
	dotBound := qmax * kmax * float64(dim)
	if dotBound > safe || dotBound*math.Abs(float64(scale)) > safe || vmax*float64(positions) > safe {
		return false
	}

	work := func(worker int) {
		buf := scratch[worker*workerWidth : (worker+1)*workerWidth]
		packedK := buf[:qwen35AttentionKVTile*dim]
		packedV := buf[len(packedK) : 2*len(packedK)]
		scores := buf[2*len(packedK) : 2*len(packedK)+qwen35AttentionQueryTile*qwen35AttentionKVTile]
		acc := buf[2*len(packedK)+len(scores) : 2*len(packedK)+len(scores)+qwen35AttentionQueryTile*dim]
		maxima := buf[len(buf)-2*qwen35AttentionQueryTile : len(buf)-qwen35AttentionQueryTile]
		sums := buf[len(buf)-qwen35AttentionQueryTile:]
		for unit := worker; unit < units; {
			kvHead, tile := unit/tiles, unit%tiles
			queryLo := tile * qwen35AttentionQueryTile
			count := queries - queryLo
			if count > qwen35AttentionQueryTile {
				count = qwen35AttentionQueryTile
			}
			clear(acc[:count*dim])
			clear(sums[:count])
			for i := 0; i < count; i++ {
				maxima[i] = float32(math.Inf(-1))
			}
			firstToken := queryLo / group
			lastToken := (queryLo + count - 1) / group
			keyLo := windowLoContig(base+firstToken+1, base+firstToken, window)
			keyEnd := base + lastToken + 1
			for keyTile := keyLo; keyTile < keyEnd; {
				n := keyEnd - keyTile
				if n > qwen35AttentionKVTile {
					n = qwen35AttentionKVTile
				}
				// Each source K/V row is packed ONCE for all query rows in this
				// tile, not once per head/token. Tail rows are never read.
				for j := 0; j < n; j++ {
					start := (keyTile+j)*kvWidth + kvHead*dim
					copy(packedK[j*dim:(j+1)*dim], k[start:start+dim])
					copy(packedV[j*dim:(j+1)*dim], v[start:start+dim])
				}
				for i := 0; i < count; i++ {
					query := queryLo + i
					token, head := query/group, kvHead*group+query%group
					end := base + token + 1
					start := windowLoContig(end, base+token, window)
					lo, hi := start-keyTile, end-keyTile
					if lo < 0 {
						lo = 0
					}
					if hi > n {
						hi = n
					}
					if lo >= hi {
						continue
					}
					qs := q[token*width+head*dim : token*width+(head+1)*dim]
					sc := scores[i*qwen35AttentionKVTile+lo : i*qwen35AttentionKVTile+hi]
					for j := lo; j < hi; j++ {
						sc[j-lo] = fdot(qs, packedK[j*dim:(j+1)*dim]) * scale
					}
					// Preserve FAK's multiply-then-softcap operation order.
					softcapInPlace(sc, cap)
					m := maxima[i]
					for _, x := range sc {
						if x > m {
							m = x
						}
					}
					alpha := float32(0)
					if !math.IsInf(float64(maxima[i]), -1) {
						alpha = float32(math.Exp(float64(maxima[i] - m)))
					}
					row := acc[i*dim : (i+1)*dim]
					for d := range row {
						row[d] *= alpha
					}
					sum := sums[i] * alpha
					for j, x := range sc {
						weight := float32(math.Exp(float64(x - m)))
						sum += weight
						saxpy(row, packedV[(lo+j)*dim:(lo+j+1)*dim], weight)
					}
					maxima[i], sums[i] = m, sum
				}
				keyTile += n
			}
			for i := 0; i < count; i++ {
				query := queryLo + i
				token, head := query/group, kvHead*group+query%group
				dst := out[token*width+head*dim : token*width+(head+1)*dim]
				for d, x := range acc[i*dim : (i+1)*dim] {
					dst[d] = x / sums[i]
				}
			}
			if units-unit <= workers {
				break
			}
			unit += workers
		}
	}
	// Synchronous, disjoint worker loans; cancellation remains at the existing
	// joined layer boundary, with no background owner or partial fallback.
	// Match the original attention worker lifetime instead of holding the
	// process-wide parFor dispatch mutex across a long attention operation.
	if workers == 1 {
		work(0)
		return true
	}
	var joined sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		joined.Add(1)
		go func(worker int) {
			defer joined.Done()
			work(worker)
		}(worker)
	}
	joined.Wait()
	return true
}

func qwen35AttentionFiniteMax(x []float32) (float64, bool) {
	var max float64
	for _, v := range x {
		f := math.Abs(float64(v))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return 0, false
		}
		if f > max {
			max = f
		}
	}
	return max, true
}

func qwen35AttentionRowsOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	ap := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bp := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if ap <= bp {
		return bp-ap < uintptr(len(a))*4
	}
	return ap-bp < uintptr(len(b))*4
}
