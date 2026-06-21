package model

// batch.go — MULTI-USER batched decode: the aggregate-throughput lane the baseline doc
// (MODEL-BASELINE-RESULTS.md) explicitly scoped OUT as "vLLM's regime, not fak's claim".
//
// The lever, stated precisely. Batch-1 decode is MEMORY-BANDWIDTH-bound at 0.50 flop/byte
// (the profiler's verdict, Act 1): per generated token the kernel re-streams ALL the weight
// bytes (~537 MB f32 / ~150 MB Q8), and time ≈ weight_bytes ÷ bandwidth, almost independent
// of how much arithmetic those weights drive. So serving ONE user wastes the machine: the
// weights are streamed to compute a single token's worth of MACs, then thrown away.
//
// Multi-user batching fixes exactly that. Stack ONE decode token from each of B independent
// users into a [B, *] panel and run each of the seven weight matmuls + the LM head as ONE
// GEMM over that panel (matMulBatch / qGemm8). Each weight row is now read ONCE and reused
// across all B users — the same arithmetic-intensity move that makes prefill fast, applied
// to the batch dimension instead of the token dimension. The bottleneck byte-stream is
// amortised B-fold, so AGGREGATE throughput (tokens/sec across all users) scales ~linearly
// with B until the GEMM becomes compute-bound, then plateaus at the compute roofline. This
// is "continuous batching" — the single biggest throughput multiplier in LLM serving — done
// in-kernel over kernel-OWNED per-user KV caches.
//
// What is per-user and what is shared. The seven projection GEMMs + the head are SHARED
// (one weight stream, B rows). Attention is PER-USER: user b's query attends only to user
// b's own KVCache (its own history, its own length), so there is zero cross-user mixing —
// the caches stay independent objects the context-MMU can still Evict/Clone per user.
//
// The bit-identity contract (f32). matMulBatch's row b is, by construction, bit-for-bit
// equal to parMatRows(weight, panel_row_b) — same fdot, same i-order (TestParallelMatchesSerial
// already pins this). The per-user attention here replays tokenHidden's EXACT scalar
// arithmetic (dot for scores, in-order V accumulation). So StepBatch's per-user logits are
// bit-for-bit identical to running each user through the serial Session.Step — proven by
// TestBatchedDecodeMatchesSerial (Float32bits equality on logits AND every user's KV cache).
// Batching changes only WHICH tokens share a weight load, never a single rounding.
//
// The Q8 path (stepBatchQ) reuses the register-blocked tile GEMM (qGemm8) the prefill path
// uses, so — exactly like prefill — it is NOT bit-identical to the serial qdot8 decode
// kernel (the tile reduces in a different lane order) but clears the same honest Q8 gate:
// argmax-exact + logit-cosine vs the f32 path (TestBatchedDecodeQMatchesF32). The f32 KV
// cache it builds is the same object either way, so Evict/Clone are untouched.

import (
	"os"
	"strconv"
	"sync"
)

const batchRectPrefillMaxTokens = 512

var attnGQAFuse = initAttnGQAFuse()

func initAttnGQAFuse() bool {
	switch os.Getenv("FAK_QATTN_GQA") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

var attnSaxpy3SIMDMinPos = initAttnSaxpy3SIMDMinPos()

func initAttnSaxpy3SIMDMinPos() int {
	if s := os.Getenv("FAK_SAXPY3_SIMD_MINPOS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 0 {
			return n
		}
	}
	return 1
}

var attnSaxpy3SIMDMinBatch = initAttnSaxpy3SIMDMinBatch()

func initAttnSaxpy3SIMDMinBatch() int {
	if s := os.Getenv("FAK_SAXPY3_SIMD_MINB"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n
		}
	}
	// Long-context batch decode spends most of its non-GEMM time accumulating V for the
	// three GQA query heads; the amd64 helper wins even at small B and can still be
	// disabled/tuned with FAK_SAXPY3_SIMD_MINB.
	return 1
}

var attnFdot3SIMD = initAttnFdot3SIMD()

func initAttnFdot3SIMD() bool {
	switch os.Getenv("FAK_FDOT3_SIMD") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

var attnFdot3SIMDMinBatch = initAttnFdot3SIMDMinBatch()

func initAttnFdot3SIMDMinBatch() int {
	if s := os.Getenv("FAK_FDOT3_SIMD_MINB"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n
		}
	}
	return 64
}

var qFastSwiGLU = initQFastSwiGLU()

func initQFastSwiGLU() bool {
	switch os.Getenv("FAK_Q_FAST_SWIGLU") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

func swigluQInPlace(g, u []float32) {
	if qFastSwiGLU {
		swigluFastInPlace(g, u)
		return
	}
	swigluInPlace(g, u)
}

func batchRectFastPathOK(cfg Config, quant bool) bool {
	return batchPreNormFastPathOK(cfg, quant)
}

func batchDecodeFastPathOK(cfg Config, quant bool) bool {
	return batchPreNormFastPathOK(cfg, quant)
}

// batchPreNormFastPathOK reports whether the shared-weight panel GEMM lane (rect
// prefill + decode) covers this config, or whether it must fall back to the per-user
// serial Session path. The fast lane hardcodes the standard PreNorm q/k/v attention +
// Llama-shaped FFN; every excluded axis below is an attention/FFN/RoPE/topology shape
// that lane does not model. isGLMMoeDsa is excluded because GLM-5.2 attention is MLA +
// Dynamic Sparse Attention (q_a/q_b/kv_a/kv_b + a learned indexer), NOT q_proj/k_proj/
// v_proj — routing it here reads tensors it does not have. The real GLM-5.2 is also MoE
// (already excluded), but a DENSE glm_moe_dsa (the synthetic / pipelinegen form) is only
// caught by this explicit guard, so the exclusion keys on the attention arch, not on the
// incidental MoE property.
func batchPreNormFastPathOK(cfg Config, quant bool) bool {
	if cfg.BlockTopology != PreNorm ||
		cfg.IsMoE() ||
		cfg.isGLMMoeDsa() ||
		cfg.DenseMLP ||
		cfg.Alibi ||
		cfg.IsQwen35Hybrid() ||
		cfg.AttnOutputGate ||
		cfg.AttnSoftcap != 0 ||
		cfg.hasLayerSpecificRopeTheta() {
		return false
	}
	if quant {
		return q8FastPreNormOK(cfg)
	}
	return true
}

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
func attnDecodeBatch(attnOut, Q []float32, caches []*KVCache, l, B, nH, hd, w, grp, W int, scale float32, scoreDot func(a, b []float32) float32, scoreDot3 func(a, b, c, x []float32) (float32, float32, float32), scoreScratch [][]float32) [][]float32 {
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

// BatchSession decodes B independent user sequences in lockstep, sharing one weight stream
// per layer across all B. Each user is a full Session with its OWN kernel-owned KVCache, so
// per-user prefill, eviction, and prefix-clone all still work; the batch only fuses the
// decode-step matmuls. Users may sit at different absolute positions (different history
// lengths) — attention reads each user's own cache length.
type BatchSession struct {
	M    *Model
	Seqs []*Session // B per-user sessions, each owns its KVCache and absolute position

	// Quant routes the batched step through the Q8_0 tile-GEMM lane (stepBatchQ); the model
	// must have had Quantize() called. Mirrors Session.Quant. The f32 default is byte-for-byte
	// the proven path.
	Quant bool

	// scratch is one reused Q8 activation panel for the quantized path: the per-step panels
	// (Xn → q/k/v, attnOut → o, Xn2 → gate/up, G → down, Xnorm → head) are consumed
	// sequentially, so a single growable buffer serves them all and avoids re-allocating a
	// panel every decode step (the same hygiene prefillBatchedQ uses). nil in the f32 path.
	scratch *q8Panel

	// dbuf holds the reused decode-step output/intermediate buffers. Every
	// projection output and panel is a fixed [B, width] shape each step, so one grown-once
	// buffer per role replaces the ~B-scaled MB of per-step allocation that BenchmarkStepBatchQ
	// measured (34 MB/step at B=32, 133 MB at B=128) — pure GC pressure, no numerics change.
	// The result is bit-identical to the allocating path. nil until the first batched decode step.
	dbuf *batchDecodeBuf

	// pbuf holds reused buffers for rectangular Q8 prefill (the repeated private-result
	// ingestion phase in fleetserve). PrefillEach still returns fresh final logits; these
	// buffers only replace per-layer temporaries that are fully consumed before return.
	pbuf *batchRectPrefillBuf
}

// batchDecodeBuf is the per-BatchSession reused-buffer set for batched decode. All fields
// are grown once to the batch's shape and overwritten every step. Logits is the one buffer the
// step RETURNS (out[b] aliases it), so per the StepBatch contract a caller must consume the
// returned logits before the next StepBatch call (every in-tree caller — GenerateBatch and the
// benchmarks — does).
type batchDecodeBuf struct {
	X, Xn, Q, K, V, attn, O, Xn2, G, U, Down, Xnorm, Logits []float32
	scores                                                  [][]float32
	pos                                                     []int
	cos, sin                                                [][]float32
	caches                                                  []*KVCache
	out                                                     [][]float32
}

type batchRectPrefillBuf struct {
	X, Xn, Q, K, V, attn, O, Xn2, G, U, Down, Xnorm []float32
	base                                            []int
	caches                                          []*KVCache
	cos, sin                                        [][]float32
	scores                                          [][]float32
}

// grow returns b resliced to length n, reallocating only when cap is short. The returned slice
// is NOT zeroed (every use below fully overwrites it before reading).
func grow(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n, growCap(n))
	}
	return b[:n]
}

func growCap(n int) int {
	return n + n/8 + 64
}

func grow2D(b [][]float32, rows, cols int) [][]float32 {
	if cap(b) < rows {
		b = make([][]float32, rows)
	} else {
		b = b[:rows]
	}
	for i := range b {
		b[i] = grow(b[i], cols)
	}
	return b
}

func growInts(b []int, n int) []int {
	if cap(b) < n {
		return make([]int, n)
	}
	return b[:n]
}

func growCaches(b []*KVCache, n int) []*KVCache {
	if cap(b) < n {
		return make([]*KVCache, n)
	}
	return b[:n]
}

func growLogitRows(b [][]float32, n int) [][]float32 {
	if cap(b) < n {
		return make([][]float32, n)
	}
	return b[:n]
}

// NewBatchSession starts a B-user batch, each user with a fresh KV cache.
func (m *Model) NewBatchSession(n int) *BatchSession {
	bs := &BatchSession{M: m, Seqs: make([]*Session, n)}
	for i := range bs.Seqs {
		bs.Seqs[i] = m.NewSession()
	}
	return bs
}

// NewBatchFromPrefix starts an n-user batch where every user's KV cache is a CLONE of an
// already-computed prefix — a shared system prompt + tool schemas prefilled ONCE, then
// spliced into all n agents. This is the cross-agent KV-reuse path the fleet exists for:
// n agents that share a long prefix pay the prefix prefill a SINGLE time plus n cheap
// deep-copies, where a per-slot serving engine with no cross-request KV sharing (llama.cpp)
// must prefill that prefix n times to decode the n agents concurrently. Each clone is exact
// (KVCache.Clone), so every user is bit-identical to one that prefilled the prefix itself —
// the same R14 prefix-reuse property TestKVPrefixReuseMatchesRecompute proves for one
// session, now fanned out across the batch. The returned users sit at the prefix's length,
// ready for StepBatch on their first generated token.
func (m *Model) NewBatchFromPrefix(prefix *KVCache, n int) *BatchSession {
	return m.NewBatchFromPrefixReserve(prefix, n, 0)
}

// NewBatchFromPrefixReserve is NewBatchFromPrefix with per-user cache capacity reserved
// for the known decode/result tail. It preserves the same exact cloned prefix but avoids
// append-triggered prefix re-copies during the measured fleet run.
func (m *Model) NewBatchFromPrefixReserve(prefix *KVCache, n, extraPositions int) *BatchSession {
	bs := &BatchSession{M: m, Seqs: make([]*Session, n)}
	for i := range bs.Seqs {
		bs.Seqs[i] = &Session{M: m, Cache: prefix.CloneWithReserve(extraPositions)}
	}
	return bs
}

// SetQuant turns on the Q8_0 lane for every user in the batch (call after Model.Quantize()).
func (bs *BatchSession) SetQuant(q bool) {
	bs.Quant = q
	for _, s := range bs.Seqs {
		s.Quant = q
	}
}

// Reserve grows every user's KV cache for extra future positions without changing contents.
func (bs *BatchSession) Reserve(extraPositions int) {
	for _, s := range bs.Seqs {
		s.Cache.Reserve(extraPositions)
	}
}

// N is the number of users in the batch.
func (bs *BatchSession) N() int { return len(bs.Seqs) }

// PrefillEach ingests each user's (possibly distinct) prompt into that user's own cache and
// returns each user's last-token logits — the distribution over its first generated token.
// Prefill is per-user (prompts have different lengths); the throughput win this file is about
// is in the DECODE phase (StepBatch), which is the memory-bound regime an agent loop lives in.
func (bs *BatchSession) PrefillEach(prompts [][]int) [][]float32 {
	if len(prompts) != len(bs.Seqs) {
		panic("model: PrefillEach prompt count != batch size")
	}
	if P, ok := rectangularPrefillLen(prompts); ok && batchRectFastPathOK(bs.M.Cfg, bs.Quant) {
		if bs.Quant {
			return bs.prefillEachRectQ(prompts, P, true)
		}
		return bs.prefillEachRectF32(prompts, P, true)
	}
	out := make([][]float32, len(prompts))
	for b, p := range prompts {
		out[b] = bs.Seqs[b].Prefill(p)
	}
	return out
}

// PrefillEachNoLogits ingests each user's prompt into its own cache and intentionally skips
// final-token logits when the rectangular fast path can do so. Fleet result-ingest uses this:
// the tool/result tokens must extend KV state, but their post-prefill next-token distribution
// is discarded before the next decode turn starts. Non-PreNorm topologies (and the
// non-rectangular case) fall back to the per-session topology-aware PrefillNoLogits.
func (bs *BatchSession) PrefillEachNoLogits(prompts [][]int) {
	if len(prompts) != len(bs.Seqs) {
		panic("model: PrefillEachNoLogits prompt count != batch size")
	}
	if P, ok := rectangularNoLogitsPrefillLen(prompts); ok && batchRectFastPathOK(bs.M.Cfg, bs.Quant) {
		if bs.Quant {
			bs.prefillEachRectQ(prompts, P, false)
			return
		}
		bs.prefillEachRectF32(prompts, P, false)
		return
	}
	for b, p := range prompts {
		bs.Seqs[b].PrefillNoLogits(p)
	}
}

func rectangularNoLogitsPrefillLen(prompts [][]int) (int, bool) {
	if len(prompts) == 0 {
		return 0, false
	}
	P := len(prompts[0])
	if P == 0 || P > batchRectPrefillMaxTokens {
		return 0, false
	}
	for _, p := range prompts[1:] {
		if len(p) != P {
			return 0, false
		}
	}
	return P, true
}

func rectangularPrefillLen(prompts [][]int) (int, bool) {
	if len(prompts) < 2 {
		return 0, false
	}
	P := len(prompts[0])
	if P == 0 || P > batchRectPrefillMaxTokens {
		return 0, false
	}
	for _, p := range prompts[1:] {
		if len(p) != P {
			return 0, false
		}
	}
	return P, true
}

func (bs *BatchSession) prefillEachRectF32(prompts [][]int, P int, wantLogits bool) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B, N := len(prompts), len(prompts)*P

	baseB, caches, cosN, sinN := bs.rectPrefillGeometry(P)
	embed := m.embedRows()
	X := make([]float32, N*H)
	for b, p := range prompts {
		for t, id := range p {
			row := b*P + t
			copy(X[row*H:(row+1)*H], embed[id*H:(id+1)*H])
			scaleEmbedInPlace(X[row*H:(row+1)*H], cfg) // Gemma; no-op for Llama
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }

		Xn := make([]float32, N*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wIn, eps, cfg))
			}
		})

		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, N)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, N)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, N)
		for row := 0; row < N; row++ {
			m.applyProjBias(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], V[row*w:(row+1)*w])
			m.applyLayerQKNorm(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w])
		}

		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.Kraw[l] = append(c.Kraw[l], K[row*w:(row+1)*w]...)
			}
		}
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				ropeRowQKInto(Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], cosN[row], sinN[row], hd, nH, nKV)
			}
		})
		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.K[l] = append(c.K[l], K[row*w:(row+1)*w]...)
				c.V[l] = append(c.V[l], V[row*w:(row+1)*w]...)
			}
		}

		// F32 prefill keeps the plain (allocating) attention path: the windowed
		// attnPrefillMultiInto covers both full-causal (W<0) and SWA layers. The fresh make
		// below starts attnOut zeroed (the saxpy accumulation requires it), and nil scratch
		// means no pooling here — the pooled GQA-fused path is the Q8 hot lane only.
		attnOut := make([]float32, N*nH*hd)
		attnPrefillMultiInto(attnOut, Q, caches, baseB, l, P, nH, hd, w, grp, cfg.windowForLayer(l), scale, dot, nil)

		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, N)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(O[row*H:(row+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := make([]float32, N*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn2[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wPost, eps, cfg))
			}
		})
		I := cfg.IntermediateSize
		G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, I, H, N)
		U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, I, H, N)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(G[row*I:(row+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[row*I:(row+1)*I], lp("mlp.up_proj.bias"))
		}
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, I, N)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(Down[row*H:(row+1)*H], lp("mlp.down_proj.bias"))
		}
		for i := range X {
			X[i] += Down[i]
		}
	}

	bs.finishRectPrefillPositions(baseB, P)
	if !wantLogits {
		return nil
	}
	Xnorm := make([]float32, B*H)
	normW := m.tensor("model.norm.weight")
	for b := 0; b < B; b++ {
		row := b*P + P - 1
		copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[row*H:(row+1)*H], normW, eps, cfg))
	}
	Logits := matMulBatch(m.lmHead(), Xnorm, cfg.VocabSize, H, B)
	out := splitLogits(Logits, B, cfg.VocabSize)
	for b := range out {
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	return out
}

func (bs *BatchSession) prefillEachRectQ(prompts [][]int, P int, wantLogits bool) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B, N := len(prompts), len(prompts)*P
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	if bs.pbuf == nil {
		bs.pbuf = &batchRectPrefillBuf{}
	}
	pb := bs.pbuf

	baseB := growInts(pb.base, B)
	pb.base = baseB
	caches := growCaches(pb.caches, B)
	pb.caches = caches
	cosN := grow2D(pb.cos, N, hd/2)
	pb.cos = cosN
	sinN := grow2D(pb.sin, N, hd/2)
	pb.sin = sinN
	inv := cachedInvFreq(cfg, 0)
	for b, s := range bs.Seqs {
		baseB[b] = s.Cache.Len()
		caches[b] = s.Cache
		for t := 0; t < P; t++ {
			row := b*P + t
			ropeRowInto(cosN[row], sinN[row], inv, baseB[b]+t)
		}
	}
	embed := m.embedRows()
	X := grow(pb.X, N*H)
	pb.X = X
	for b, p := range prompts {
		for t, id := range p {
			row := b*P + t
			copy(X[row*H:(row+1)*H], embed[id*H:(id+1)*H])
			scaleEmbedInPlace(X[row*H:(row+1)*H], cfg) // Gemma; no-op for Llama
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }
		ql := m.q8Layer(l)

		Xn := grow(pb.Xn, N*H)
		pb.Xn = Xn
		wIn := m.tensor(lp("input_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wIn, eps, cfg))
				} else {
					rmsnormInto(Xn[row*H:(row+1)*H], X[row*H:(row+1)*H], wIn, eps)
				}
			}
		})
		quantizeBatchPanelInto(bs.scratch, Xn, N, H)
		// Fused q/k/v: one quantized Xn panel drives three tile GEMMs into pooled dsts (perf,
		// numerically identical to three separate qGemm8 calls). Bias + QK-norm are applied
		// via the config-driven helpers so non-Llama archs (AttentionBias, QKNorm) are correct.
		Q := grow(pb.Q, N*nH*hd)
		pb.Q = Q
		K := grow(pb.K, N*w)
		pb.K = K
		V := grow(pb.V, N*w)
		pb.V = V
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.qProj, Y: Q},
			qgemm8Target{qt: ql.kProj, Y: K},
			qgemm8Target{qt: ql.vProj, Y: V},
		)
		for row := 0; row < N; row++ {
			m.applyProjBias(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], V[row*w:(row+1)*w])
			m.applyLayerQKNorm(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w])
		}

		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.Kraw[l] = append(c.Kraw[l], K[row*w:(row+1)*w]...)
			}
		}
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				ropeRowQKInto(Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], cosN[row], sinN[row], hd, nH, nKV)
			}
		})
		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.K[l] = append(c.K[l], K[row*w:(row+1)*w]...)
				c.V[l] = append(c.V[l], V[row*w:(row+1)*w]...)
			}
		}

		// Attention. pb.attn is a reused buffer and the helper += accumulates into it, so it
		// must be cleared first. The GQA-fused helper carries the layer window bound.
		attnOut := grow(pb.attn, N*nH*hd)
		pb.attn = attnOut
		clear(attnOut)
		scoreDot3 := fdot3scalar
		if attnFdot3SIMD && B >= attnFdot3SIMDMinBatch {
			scoreDot3 = fdot3SIMD
		}
		pb.scores = attnPrefillMultiGQAInto(attnOut, Q, caches, baseB, l, P, nH, hd, w, grp, cfg.windowForLayer(l), scale, fdot, scoreDot3, pb.scores)

		quantizeBatchPanelInto(bs.scratch, attnOut, N, nH*hd)
		O := grow(pb.O, N*H)
		pb.O = O
		qGemm8Into(ql.oProj, bs.scratch, O)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(O[row*H:(row+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := grow(pb.Xn2, N*H)
		pb.Xn2 = Xn2
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[row*H:(row+1)*H], X[row*H:(row+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		quantizeBatchPanelInto(bs.scratch, Xn2, N, H)
		// Fused gate/up GEMM (perf) into pooled dsts, then the config-driven activation. For
		// Llama act==silu so `act(G)*U` is byte-identical to swigluInPlace(G,U); for Gemma it
		// is the correct GeGLU. The fused GEMM is orthogonal to the activation choice.
		G := grow(pb.G, N*I)
		pb.G = G
		U := grow(pb.U, N*I)
		pb.U = U
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.gateProj, Y: G},
			qgemm8Target{qt: ql.upProj, Y: U},
		)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(G[row*I:(row+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[row*I:(row+1)*I], lp("mlp.up_proj.bias"))
		}
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		quantizeBatchPanelInto(bs.scratch, G, N, I)
		Down := grow(pb.Down, N*H)
		pb.Down = Down
		qGemm8Into(ql.downProj, bs.scratch, Down)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(Down[row*H:(row+1)*H], lp("mlp.down_proj.bias"))
		}
		for i := range X {
			X[i] += Down[i]
		}
	}

	bs.finishRectPrefillPositions(baseB, P)
	if !wantLogits {
		return nil
	}
	Xnorm := grow(pb.Xnorm, B*H)
	pb.Xnorm = Xnorm
	normW := m.tensor("model.norm.weight")
	for b := 0; b < B; b++ {
		row := b*P + P - 1
		if cfg.NormGain1p || cfg.LayerNorm {
			copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[row*H:(row+1)*H], normW, eps, cfg))
		} else {
			rmsnormInto(Xnorm[b*H:(b+1)*H], X[row*H:(row+1)*H], normW, eps)
		}
	}
	quantizeBatchPanelInto(bs.scratch, Xnorm, B, H)
	Logits := qGemm8(m.q8(m.headName()), bs.scratch)
	out := splitLogits(Logits, B, cfg.VocabSize)
	for b := range out {
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	return out
}

func (bs *BatchSession) rectPrefillGeometry(P int) ([]int, []*KVCache, [][]float32, [][]float32) {
	B := len(bs.Seqs)
	baseB := make([]int, B)
	caches := make([]*KVCache, B)
	cosN := make([][]float32, B*P)
	sinN := make([][]float32, B*P)
	for b, s := range bs.Seqs {
		baseB[b] = s.Cache.Len()
		caches[b] = s.Cache
		for t := 0; t < P; t++ {
			row := b*P + t
			cosN[row], sinN[row] = ropeRow(bs.M.Cfg, baseB[b]+t)
		}
	}
	return baseB, caches, cosN, sinN
}

func (bs *BatchSession) finishRectPrefillPositions(baseB []int, P int) {
	for b, s := range bs.Seqs {
		for t := 0; t < P; t++ {
			s.Cache.pos = append(s.Cache.pos, baseB[b]+t)
		}
	}
}

func splitLogits(logits []float32, B, vocab int) [][]float32 {
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		out[b] = logits[b*vocab : (b+1)*vocab]
	}
	return out
}

// StepBatch decodes ONE token for each user (ids[b] is user b's already-chosen token) and
// returns B next-token logit vectors. In the f32 path each user's logits are bit-for-bit
// identical to bs.Seqs[b].Step(ids[b]) run serially; the only difference is that the weight
// stream is shared across all B users instead of re-streamed per user.
func (bs *BatchSession) StepBatch(ids []int) [][]float32 {
	if len(ids) != len(bs.Seqs) {
		panic("model: StepBatch id count != batch size")
	}
	if len(ids) == 1 || !batchDecodeFastPathOK(bs.M.Cfg, bs.Quant) {
		if len(ids) == 1 {
			return [][]float32{bs.Seqs[0].Step(ids[0])}
		}
		out := make([][]float32, len(ids))
		for b, id := range ids {
			out[b] = bs.Seqs[b].Step(id)
		}
		return out
	}
	if bs.Quant {
		return bs.stepBatchQ(ids)
	}
	return bs.stepBatchF32(ids)
}

// GenerateBatch greedily decodes up to n tokens for every user in lockstep after their
// prompts, returning each user's generated ids. A user that emits EOS stops contributing new
// tokens (its slot is re-fed its own EOS so the batch geometry stays rectangular — cheap, and
// it keeps the per-user output bit-identical to serial Generate for the non-EOS users). This
// is STATIC batching (fixed B for the run); the per-step primitive (StepBatch) is exactly
// what a continuous-batching scheduler would call after admitting/evicting users between steps.
func (bs *BatchSession) GenerateBatch(prompts [][]int, n int) [][]int {
	B := len(bs.Seqs)
	logits := bs.PrefillEach(prompts)
	out := make([][]int, B)
	done := make([]bool, B)
	// eosSlot is the id re-fed into a finished slot to keep the batch rectangular. The
	// first EOS id (scalar or list head) is fine for this; stop detection uses isEOS.
	eosSlot := bs.M.Cfg.EOSTokenID
	if len(bs.M.Cfg.EOSTokenIDs) > 0 {
		eosSlot = bs.M.Cfg.EOSTokenIDs[0]
	}
	next := make([]int, B)
	for i := 0; i < n; i++ {
		anyLive := false
		for b := 0; b < B; b++ {
			if done[b] {
				next[b] = eosSlot // keep slot rectangular; its logits are ignored once done
				continue
			}
			t := argmaxF32(logits[b])
			out[b] = append(out[b], t)
			next[b] = t
			if bs.M.Cfg.IsEOS(t) {
				done[b] = true
			} else {
				anyLive = true
			}
		}
		if !anyLive {
			break
		}
		logits = bs.StepBatch(next)
	}
	return out
}

// stepBatchF32 is the f32 multi-user decode step. Structure mirrors tokenHidden exactly,
// hoisted to operate on a [B, *] panel: the projections/MLP/head become matMulBatch GEMMs
// (weight read once, reused across all B), attention stays per-user over each user's cache.
func (bs *BatchSession) stepBatchF32(ids []int) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B := len(ids)
	if bs.dbuf == nil {
		bs.dbuf = &batchDecodeBuf{}
	}
	db := bs.dbuf

	// Each user decodes at its OWN absolute position (its current cache length), so RoPE is
	// per-user. (Captured before any append; pos is recorded at the end, matching tokenHidden.)
	posB := growInts(db.pos, B)
	db.pos = posB
	cosB := grow2D(db.cos, B, hd/2)
	db.cos = cosB
	sinB := grow2D(db.sin, B, hd/2)
	db.sin = sinB
	inv := cachedInvFreq(cfg, 0)
	for b := 0; b < B; b++ {
		posB[b] = bs.Seqs[b].Cache.Len()
		ropeRowInto(cosB[b], sinB[b], inv, posB[b])
	}

	// embedding lookup: X is [B, H], one working hidden row per user.
	embed := m.embedRows()
	X := make([]float32, B*H)
	for b, id := range ids {
		copy(X[b*H:(b+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[b*H:(b+1)*H], cfg) // Gemma; no-op for Llama
	}
	caches := growCaches(db.caches, B)
	db.caches = caches
	for b := 0; b < B; b++ {
		caches[b] = bs.Seqs[b].Cache
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }

		// pre-attn RMSNorm, per user.
		Xn := make([]float32, B*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		for b := 0; b < B; b++ {
			copy(Xn[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wIn, eps, cfg))
		}

		// batched q/k/v projections: one weight stream, B rows.
		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, B)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, B)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, B)
		for b := 0; b < B; b++ {
			m.applyProjBias(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w], V[b*w:(b+1)*w])
			m.applyLayerQKNorm(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w])
		}

		// per-user RoPE + append k/v to each user's own cache (cheap serial pass; the append
		// mutates each user's distinct cache so it can't be raced). The per-user single-row
		// rotate-and-stash is the same one the decode block funnels through (ropeRowQK).
		for b := 0; b < B; b++ {
			qb := Q[b*nH*hd : (b+1)*nH*hd]
			kb := K[b*w : (b+1)*w]
			vb := V[b*w : (b+1)*w]
			c := caches[b]
			c.Kraw[l] = append(c.Kraw[l], kb...) // pre-RoPE, for lossless eviction
			ropeRowQKInto(qb, kb, cosB[b], sinB[b], hd, nH, nKV)
			c.K[l] = append(c.K[l], kb...)
			c.V[l] = append(c.V[l], vb...)
		}
		// causal GQA attention, each user over its own cache (parallel, allocation-light).
		attnOut := make([]float32, B*nH*hd)
		db.scores = attnDecodeBatch(attnOut, Q, caches, l, B, nH, hd, w, grp, cfg.windowForLayer(l), scale, dot, nil, db.scores)

		// batched output projection + residual.
		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, B)
		for i := range X {
			X[i] += O[i]
		}

		// batched MLP (SwiGLU) + residual.
		Xn2 := make([]float32, B*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		for b := 0; b < B; b++ {
			copy(Xn2[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wPost, eps, cfg))
		}
		I := cfg.IntermediateSize
		G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, I, H, B)
		U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, I, H, B)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, I, B)
		for i := range X {
			X[i] += Down[i]
		}
	}

	// final norm per user; record each user's new absolute position; batched LM head.
	normW := m.tensor("model.norm.weight")
	Xnorm := make([]float32, B*H)
	for b := 0; b < B; b++ {
		copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], normW, eps, cfg))
		bs.Seqs[b].Cache.pos = append(bs.Seqs[b].Cache.pos, posB[b])
	}
	// the 113 MB tied-embedding head streamed ONCE for all B users — the single biggest
	// per-token weight, and so the single biggest batching beneficiary at decode.
	Logits := matMulBatch(m.lmHead(), Xnorm, cfg.VocabSize, H, B)
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		out[b] = Logits[b*cfg.VocabSize : (b+1)*cfg.VocabSize]
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	return out
}

// qgemmBatch quantizes a [B, width] activation panel into the session's reused scratch and
// runs the register-blocked Q8_0 tile GEMM against the named weight, returning [B, out].
func (bs *BatchSession) qgemmBatch(name string, X []float32, B, width int) []float32 {
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	quantizeBatchPanelInto(bs.scratch, X, B, width)
	return qGemm8(bs.M.q8(name), bs.scratch)
}

// qgemmBatchInto is qgemmBatch writing the GEMM result into a caller-provided dst (reused
// across decode steps). Bit-identical to qgemmBatch.
func (bs *BatchSession) qgemmBatchInto(name string, X []float32, B, width int, dst []float32) {
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	quantizeBatchPanelInto(bs.scratch, X, B, width)
	qGemm8Into(bs.M.q8(name), bs.scratch, dst)
}

func (bs *BatchSession) qgemmBatchTensorInto(qt *q8Tensor, X []float32, B, width int, dst []float32) {
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	quantizeBatchPanelInto(bs.scratch, X, B, width)
	qGemm8Into(qt, bs.scratch, dst)
}

// stepBatchQ is the Q8_0 multi-user decode step: the structural twin of stepBatchF32 with the
// projections + head run as quantized tile GEMMs over the batch panel. Attention is the same
// f32 math over the f32 KV cache (identical to tokenHiddenQ). Not bit-identical to the serial
// qdot8 decode (the tile reduces in a different order) but clears the same Q8 gate the prefill
// path does — argmax-exact + cosine vs f32 (TestBatchedDecodeQMatchesF32).
func (bs *BatchSession) stepBatchQ(ids []int) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B := len(ids)

	if bs.dbuf == nil {
		bs.dbuf = &batchDecodeBuf{}
	}
	db := bs.dbuf

	posB := growInts(db.pos, B)
	db.pos = posB
	cosB := grow2D(db.cos, B, hd/2)
	db.cos = cosB
	sinB := grow2D(db.sin, B, hd/2)
	db.sin = sinB
	inv := cachedInvFreq(cfg, 0)
	for b := 0; b < B; b++ {
		posB[b] = bs.Seqs[b].Cache.Len()
		ropeRowInto(cosB[b], sinB[b], inv, posB[b])
	}

	embed := m.embedRows()
	X := grow(db.X, B*H)
	db.X = X
	for b, id := range ids {
		copy(X[b*H:(b+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[b*H:(b+1)*H], cfg) // Gemma; no-op for Llama
	}
	caches := growCaches(db.caches, B)
	db.caches = caches
	for b := 0; b < B; b++ {
		caches[b] = bs.Seqs[b].Cache
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }
		ql := m.q8Layer(l)

		Xn := grow(db.Xn, B*H)
		db.Xn = Xn
		wIn := m.tensor(lp("input_layernorm.weight"))
		for b := 0; b < B; b++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				copy(Xn[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wIn, eps, cfg))
			} else {
				rmsnormInto(Xn[b*H:(b+1)*H], X[b*H:(b+1)*H], wIn, eps)
			}
		}
		// q/k/v share one quantized panel of Xn (built once, reused across the three GEMMs).
		if bs.scratch == nil {
			bs.scratch = &q8Panel{}
		}
		quantizeBatchPanelInto(bs.scratch, Xn, B, H)
		Q := grow(db.Q, B*nH*hd)
		db.Q = Q
		K := grow(db.K, B*w)
		db.K = K
		V := grow(db.V, B*w)
		db.V = V
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.qProj, Y: Q},
			qgemm8Target{qt: ql.kProj, Y: K},
			qgemm8Target{qt: ql.vProj, Y: V},
		)
		for b := 0; b < B; b++ {
			m.applyProjBias(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w], V[b*w:(b+1)*w])
			m.applyLayerQKNorm(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w])
		}

		for b := 0; b < B; b++ {
			qb := Q[b*nH*hd : (b+1)*nH*hd]
			kb := K[b*w : (b+1)*w]
			vb := V[b*w : (b+1)*w]
			c := caches[b]
			c.Kraw[l] = append(c.Kraw[l], kb...)
			ropeRowQKInto(qb, kb, cosB[b], sinB[b], hd, nH, nKV)
			c.K[l] = append(c.K[l], kb...)
			c.V[l] = append(c.V[l], vb...)
		}
		// attention is the same f32 math over the f32 KV cache as the f32 lane.
		attnOut := grow(db.attn, B*nH*hd)
		db.attn = attnOut
		scoreDot3 := fdot3scalar
		if attnFdot3SIMD && B >= attnFdot3SIMDMinBatch {
			scoreDot3 = fdot3SIMD
		}
		db.scores = attnDecodeBatch(attnOut, Q, caches, l, B, nH, hd, w, grp, cfg.windowForLayer(l), scale, fdot, scoreDot3, db.scores)

		O := grow(db.O, B*H)
		db.O = O
		bs.qgemmBatchTensorInto(ql.oProj, attnOut, B, nH*hd, O)
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := grow(db.Xn2, B*H)
		db.Xn2 = Xn2
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		for b := 0; b < B; b++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				copy(Xn2[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wPost, eps, cfg))
			} else {
				rmsnormInto(Xn2[b*H:(b+1)*H], X[b*H:(b+1)*H], wPost, eps)
			}
		}
		I := cfg.IntermediateSize
		quantizeBatchPanelInto(bs.scratch, Xn2, B, H)
		G := grow(db.G, B*I)
		db.G = G
		U := grow(db.U, B*I)
		db.U = U
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.gateProj, Y: G},
			qgemm8Target{qt: ql.upProj, Y: U},
		)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := grow(db.Down, B*H)
		db.Down = Down
		bs.qgemmBatchTensorInto(ql.downProj, G, B, I, Down)
		for i := range X {
			X[i] += Down[i]
		}
	}

	normW := m.tensor("model.norm.weight")
	Xnorm := grow(db.Xnorm, B*H)
	db.Xnorm = Xnorm
	for b := 0; b < B; b++ {
		if cfg.NormGain1p || cfg.LayerNorm {
			copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], normW, eps, cfg))
		} else {
			rmsnormInto(Xnorm[b*H:(b+1)*H], X[b*H:(b+1)*H], normW, eps)
		}
		bs.Seqs[b].Cache.pos = append(bs.Seqs[b].Cache.pos, posB[b])
	}
	Logits := grow(db.Logits, B*cfg.VocabSize)
	db.Logits = Logits
	bs.qgemmBatchTensorInto(m.q8Head(), Xnorm, B, H, Logits)
	out := growLogitRows(db.out, B)
	db.out = out
	for b := 0; b < B; b++ {
		out[b] = Logits[b*cfg.VocabSize : (b+1)*cfg.VocabSize]
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	return out
}
