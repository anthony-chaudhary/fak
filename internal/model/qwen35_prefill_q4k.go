package model

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// qwen35_prefill_q4k.go is the resident-Q4_K twin of qwen35_prefill.go (the Q8 hybrid
// Gated-DeltaNet fresh-prefill path). It exists for the same reason prefillBatchedQ4K
// (prefill_q4k.go) is the twin of prefillBatchedQ: when Qwen3.6-27B is loaded through the
// resident-Q4_K path (FAK_Q4K=1), the generic batched-Q4_K prefill refuses the hybrid arch
// (q8PrefillNeedsTokenLoop → IsQwen35Hybrid), so prompt processing fell back to the
// per-token blockStep loop — re-streaming every weight P times. On the M3 Pro that pins
// Qwen3.6 prefill at ~0.5 tok/s (22 tok in ~46 s) regardless of how compact the decode
// stream is. This path keeps the GDN recurrence but batches each layer's projection/MLP
// GEMMs over the prompt panel, exactly as the Q8 hybrid path does, closing
// QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P3's open "qwen35-hybrid falls back to the per-token
// loop" item for the resident-Q4_K lane.
//
// It differs from prefillQwen35HybridQHidden in exactly ONE way — each projection GEMM
// dispatches by resident format via `proj`, the same per-weight dispatch prefillBatchedQ4K
// uses:
//
//   - q4kw-resident (the identity-normalized q4_k_m majority: self_attn v_proj/o_proj,
//     mlp gate/up/down) → q4kGemm on the f32 activation, each super-block dequantized ONCE
//     and reused across all P prompt tokens.
//   - everything else (self_attn q/k, and EVERY linear_attn.* projection — these are
//     reordered/unpermuted for qwen35, so ResidentQ4KEligible keeps them out of q4kw; plus
//     any Q6_K weight) → q8GemmDispatch on a Q8-quantized activation panel: CPU qGemm8 by
//     default, Metal Q8 GEMM when MetalQ4K is enabled (#1087).
//
// Everything else — the conv1d causal scan, the per-head Gated-DeltaNet recurrence, the
// L2/RMS norms, RoPE, the causal GQA over the f32 KV cache, SwiGLU, the residuals — is the
// identical f32 math prefillQwen35HybridQ runs, copied verbatim so the recurrence has a
// single proven reference. The cache it builds is the same f32 object (Kraw pre-RoPE, K
// post-RoPE, V, pos, plus the linearAttnCache conv/recurrent state).
//
// Correctness contract vs the per-token Q4K decode path (tokenHiddenQ via sessionQ4KKernel):
// sessionQ4KKernel.mul resolves a projection by name with the IDENTICAL order proj uses
// (q4kw first → q4kMatRows, else → qGemm8/qMatRows on m.q8). So per weight, both paths take
// the same kernel: the q4_k_m majority is bit-identical on CPU (q4kGemm == q4kMatRows per
// (o,t), TestQ4KGemmMatchesMatRows) and approximate on Metal; the Q8 minority differs only by
// the documented Q8 deferred-reduction/FMA-rounding drift the Q8 hybrid path's own gate already
// covers. The recurrence is the same f32 math fed by those projections. Pinned by
// TestPrefillQwen35HybridQ4KMatchesTokenLoop and the Metal Q4K/Q8 gates.

// q4kQwen35HybridPrefillOK gates the batched resident-Q4_K hybrid prefill. It is the same
// architecture gate the Q8 hybrid path uses (q8Qwen35HybridPrefillOK) — the resident-Q4_K
// path covers the identical Qwen3.5/3.6 hybrid family; only the projection kernel differs.
func q4kQwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return q8Qwen35HybridPrefillOK(cfg, promptLen)
}

// hybridQ4KProj is the per-weight projection dispatch shared by every GEMM in the batched
// resident-Q4_K hybrid prefill: q4kw-resident -> q4kGemm on the raw f32 activation Xf;
// otherwise -> q8GemmDispatch on the pre-quantized Q8 panel Xq (CPU qGemm8 by default,
// Metal Q8 GEMM when MetalQ4K is enabled). Mirrors prefillBatchedQ4K's `proj`.
type hybridQ4KProj func(name string, Xf []float32, Xq *q8Panel) []float32

// hybridQ4KGroup runs a group of projections that share one activation panel (Xf f32, Xq the
// pre-quantized Q8 panel) through the batched one-command-buffer q4_k GEMM group, filling any
// non-grouped member per-weight via the shared proj. Results are returned in `names` order and are
// identical to calling proj per name — just fewer Metal command buffers (the prefill-wall lever).
type hybridQ4KGroup func(names []string, Xf []float32, Xq *q8Panel) [][]float32

func (s *Session) prefillQwen35HybridQ4K(ids []int) []float32 {
	return s.headResident(s.prefillQwen35HybridQ4KHidden(ids))
}

func (s *Session) prefillQwen35HybridQ4KNoLogits(ids []int) {
	_ = s.prefillQwen35HybridQ4KHidden(ids)
}

func (s *Session) prefillQwen35HybridQ4KHidden(ids []int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	P := len(ids)
	if P == 0 {
		return nil
	}
	profile := os.Getenv("FAK_QPROFILE") != ""
	var start time.Time
	if profile {
		start = time.Now()
	}
	var gemmTime time.Duration
	// Sub-buckets of gemmTime, split by which resident kernel actually served each projection,
	// so the next prefill-optimization decision is evidence-based rather than assuming the q4_k
	// Metal GEMM dominates. q4kTime is the q4_k-majority GEMM (Metal q4_k dequant-GEMM under
	// -tags fakmetal + MetalQ4K, else CPU q4kGemm) INCLUDING the grouped one-command-buffer
	// GEMMGroup roundtrip; q8Time is the Q8 minority (full-attn q/k + every linear_attn.*),
	// which on the 36 GiB Mac is OOM-gated onto the CPU qGemm8 path (metalQ8UploadAllowed=false)
	// and is the suspected real prefill wall; q6kTime is the resident Q6_K/Q5_K matmul weights
	// (the q4_k_m dense down_proj / lm_head). q4kTime+q8Time+q6kTime == gemmTime by construction
	// (every timed projection lands in exactly one bucket), so the split is exhaustive and honest.
	var q4kTime, q8Time, q6kTime time.Duration
	// q4kGPUCompute is the on-GPU execute window (cb.GPUEndTime-cb.GPUStartTime, via
	// metalgemm.LastGEMMGPUMs) summed over every q4_k Metal dispatch that lands in q4kTime — the
	// grouped GEMMGroup path (q4kGemmGroupDispatch) and the per-weight GEMM path (q4kGemmDispatch).
	// LastGEMMGPUMs reflects only the MOST RECENT dispatch, so it is read immediately after EACH
	// dispatch and accumulated (never once at the end). q4kRoundtrip = q4kTime - q4kGPUCompute is
	// the wall-time remainder: CPU-side encode/commit/sync + the H2D activation upload. On a
	// non-fakmetal build (or when the q4_k upload declined and the dispatch fell back to CPU
	// q4kGemm) LastGEMMGPUMs returns 0, so q4kGPUCompute stays 0 and q4kRoundtrip == q4kTime — the
	// honest degenerate: no GPU window to attribute, all of q4kTime is wall time.
	var q4kGPUCompute time.Duration
	base := s.Cache.Len()
	eps := float32(cfg.RMSNormEps)

	// One reused Q8 activation-panel scratch for the qGemm8 minority projections (q/k always,
	// every linear_attn.* projection, any Q6_K weight). Each panel is fully consumed before
	// the next is built, so a single buffer is safe — same discipline as prefillBatchedQ4K.
	scratch := &q8Panel{}
	qz := func(X []float32, rows, width int) *q8Panel {
		t := s.phaseStart()
		quantizeBatchPanelInto(scratch, X, rows, width)
		s.phaseEnd("q8_panel_quantize", t)
		return scratch
	}
	proj := func(name string, Xf []float32, Xq *q8Panel) []float32 {
		if qt := m.q4kw[name]; qt != nil {
			// q4kGemmDispatch is the CPU q4kGemm by default; under -tags fakmetal with
			// s.MetalQ4K set it routes the q4_k-majority GEMM to the Metal q4_k dequant-GEMM.
			return s.q4kGemmDispatch(name, qt, Xf, P)
		}
		if qt := m.kqw[name]; qt != nil {
			// Resident Q5_K/Q6_K matmul weight (the q4_k_m dense down_proj / lm_head now load
			// Q6_K into kqw, not the Q8 store). Without this branch m.q8(name) below would panic
			// ("q8 tensor not built") — the prefill twin of the decode-path kqw consultation in
			// sessionQ4KKernel.mul. kQuantMatRowsIntoBatch dequantizes each weight super-block ONCE
			// and dots it against all P token columns (a GEMM), amortizing the expensive Q6_K/Q5_K
			// super-block dequant P-fold instead of the old per-token GEMV loop that re-dequantized
			// every block P times — the #2378 prefill-wall lever (~78% of prefill was this dequant).
			// It is ONE parForRange over qt.out rows (serial over tokens INSIDE the row body), so it
			// does not re-enter parDispatchMu (parFor is not re-entrant) and stays bit-identical to
			// the per-token kQuantMatRowsInto loop. Same Y layout (P×qt.out row-major); still lands
			// in the q6kTime profile bucket.
			Y := make([]float32, P*qt.out)
			kQuantMatRowsIntoBatch(qt, Xf, P, Y)
			return Y
		}
		return s.q8GemmDispatch(name, m.q8(name), Xq)
	}
	if profile {
		rawProj := proj
		proj = func(name string, Xf []float32, Xq *q8Panel) []float32 {
			t0 := time.Now()
			Y := rawProj(name, Xf, Xq)
			dt := time.Since(t0)
			gemmTime += dt
			// Attribute this projection to a sub-bucket by the SAME resident-map dispatch order
			// rawProj (the proj closure above) uses: q4kw first → q4k, else kqw → q6k, else Q8.
			switch {
			case m.q4kw[name] != nil:
				q4kTime += dt
				// The q4kw branch just ran q4kGemmDispatch → one Q4KWeight.GEMM = one Metal
				// command buffer, which freshly stored its on-GPU execute window. Read it right
				// here (most-recent-dispatch semantics) and add to the GPU-compute sub-total. On
				// CPU fallback / non-fakmetal it is 0, so roundtrip absorbs all of this dt.
				q4kGPUCompute += time.Duration(metalgemm.LastGEMMGPUMs() * float64(time.Millisecond))
			case m.kqw[name] != nil:
				q6kTime += dt
			default:
				q8Time += dt
			}
			return Y
		}
	}
	// pgroup runs a group of projections that SHARE one activation panel Xf (a layer's q/k/v,
	// gate/up, or the GDN in_proj quad) through the batched one-command-buffer q4_k GEMM group,
	// collapsing the per-weight submit/sync round-trips that dominate prefill. It returns results
	// in `names` order: the q4_k-resident majority is filled by metalgemm.GEMMGroup (via
	// q4kGemmGroupDispatch) and any nil member (Q8/Q6_K minority, or a declined upload / non-Metal
	// build) is filled by the same per-weight `proj`, so the result is identical to calling proj
	// per name — just fewer command buffers. Xq is the pre-quantized Q8 panel proj needs for the
	// minority fallback.
	pgroup := func(names []string, Xf []float32, Xq *q8Panel) [][]float32 {
		t0 := time.Now()
		out := s.q4kGemmGroupDispatch(names, Xf, P)
		if profile {
			dt := time.Since(t0) // the grouped GEMM+roundtrip, so the profile split stays honest
			gemmTime += dt
			// The grouped dispatch fills ONLY the q4_k-resident members in one Metal command
			// buffer (the q4_k dequant-GEMM + submit/sync roundtrip); every nil member is filled
			// below by the per-weight proj, which buckets itself. So the grouped time is q4_k-Metal
			// work and lands in q4kTime — keeping q4k_metal-vs-q8_cpu the honest split.
			q4kTime += dt
			// out != nil ⟺ GEMMGroup actually dispatched (one command buffer) and freshly stored
			// its on-GPU execute window; read it HERE, before the nil→make reassignment below, and
			// only when it dispatched — a declined group (out == nil) ran no GEMMGroup, so
			// LastGEMMGPUMs would be a stale prior value and the per-weight proj fallbacks below
			// bucket their own GPU time instead.
			if out != nil {
				q4kGPUCompute += time.Duration(metalgemm.LastGEMMGPUMs() * float64(time.Millisecond))
			}
		}
		if out == nil {
			out = make([][]float32, len(names))
		}
		for i, name := range names {
			if out[i] == nil {
				out[i] = proj(name, Xf, Xq) // per-weight fallback (already time-accounted by proj)
			}
		}
		return out
	}

	t := s.phaseStart()
	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg)
	}
	s.phaseEnd("embed", t)

	if s.MetalQ4K {
		// Bulk-upload every Q4_K projection to the GPU before the layer loop, exactly as the
		// full-attention batched path does (prefill_q4k.go). Without it the lazy per-weight
		// upload in metalQ4KWeight interleaves an H2D round-trip with the first use of each
		// projection, which caps warm hybrid prefill at ~7x under llama.cpp-Metal (#1113);
		// amortizing all the copies up front restores full prefill speed on the Metal hybrid
		// path the 27B Qwen3.6 takes (#71). No-op on the pure-Go build (stub returns nil).
		m.metalQ4KWeights()
		// Upload the Q8-minority projections too (full-attn q/k and linear_attn.*). Otherwise
		// #1087's Metal Q8 GEMM path would pay one upload inside the first timed projection call.
		m.metalQ8Weights()
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		Xn := make([]float32, P*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		t = s.phaseStart()
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		s.phaseEnd("input_norm", t)

		var o []float32
		if cfg.isLinearAttnLayer(l) {
			o = s.prefillQwen35LinearLayerQ4K(l, Xn, P, proj, pgroup, qz)
		} else {
			o = s.prefillQwen35FullAttnLayerQ4K(l, Xn, P, base, proj, pgroup, qz)
		}
		t = s.phaseStart()
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += o[i]
			}
		})
		s.phaseEnd("attn_residual", t)

		Xn2 := make([]float32, P*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		t = s.phaseStart()
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		s.phaseEnd("post_attn_norm", t)
		I := cfg.IntermediateSize
		Xn2q := qz(Xn2, P, H)
		t = s.phaseStart()
		gu := pgroup([]string{lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight")}, Xn2, Xn2q)
		G, U := gu[0], gu[1]
		s.phaseEnd("mlp_gate_up_proj", t)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		t = s.phaseStart()
		parFor(len(G), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = act(G[i], cfg) * U[i]
			}
		})
		s.phaseEnd("mlp_activation", t)
		t = s.phaseStart()
		Down := proj(lp("mlp.down_proj.weight"), G, qz(G, P, I))
		s.phaseEnd("mlp_down_proj", t)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		t = s.phaseStart()
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
		s.phaseEnd("mlp_residual", t)
	}

	t = s.phaseStart()
	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, base+t)
	}
	s.phaseEnd("cache_positions", t)
	t = s.phaseStart()
	xf := rmsnormCfg(X[(P-1)*H:P*H], m.tensor("model.norm.weight"), eps, cfg)
	s.phaseEnd("final_norm", t)
	if profile {
		total := time.Since(start)
		rest := total - gemmTime
		if rest < 0 {
			rest = 0
		}
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[metalprof-hybrid P=%d] total=%.1f  gemm+roundtrip=%.1f  rest(recurrence/attn/norm)=%.1f ms path=q4k\n",
			P, ms(total), ms(gemmTime), ms(rest))
		// Split the gemm+roundtrip bucket by which resident kernel served each projection. The
		// three buckets sum to gemm+roundtrip; the point is to tell the next session whether the
		// durable prefill lever is the Q8 CPU path (q8_cpu dominates → the OOM-gated qGemm8
		// minority is the wall) or the q4_k Metal kernel (q4k_metal dominates → kernel cleverness
		// like FAK_Q4K_MM is the lever). The mix on the 27B Mac is the whole question (#71, #977).
		fmt.Fprintf(os.Stderr, "[metalprof-split P=%d] q4k_metal=%.1f  q8_cpu=%.1f  q6k=%.1f ms  (sum=gemm+roundtrip=%.1f) path=q4k\n",
			P, ms(q4kTime), ms(q8Time), ms(q6kTime), ms(gemmTime))
		// Split q4k_metal itself into GPU-compute vs roundtrip. q4kGPUCompute is the summed
		// cb.GPUEndTime-cb.GPUStartTime of every q4_k Metal dispatch (grouped + per-weight);
		// q4kRoundtrip is the wall-time remainder (CPU encode/commit/sync + H2D upload). This is
		// the lever question this session answers: if roundtrip dominates, the next lever is
		// upload-caching / command-buffer batching (fewer submit/sync); if gpu_compute dominates,
		// it is fp16-staging / a GPU counter trace / kernel cleverness. Sum-check: gpu_compute +
		// roundtrip == q4k_metal by construction (roundtrip is defined as the remainder), though
		// gpu_compute may carry small slop vs the wall window since the GPU-execute window and the
		// wall window are not perfectly nested (the wall clock also brackets the cgo call boundary).
		q4kRoundtrip := q4kTime - q4kGPUCompute
		if q4kRoundtrip < 0 {
			q4kRoundtrip = 0
		}
		fmt.Fprintf(os.Stderr, "[metalprof-q4ksplit P=%d] q4k_gpu_compute=%.1f  q4k_roundtrip=%.1f ms  (sum=q4k_metal=%.1f) path=q4k\n",
			P, ms(q4kGPUCompute), ms(q4kRoundtrip), ms(q4kTime))
	}
	return xf
}

func (s *Session) prefillQwen35LinearLayerQ4K(l int, Xn []float32, P int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz func([]float32, int, int) *q8Panel) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	nK, nV, kHd, vHd, keyDim, valDim, convDim := cfg.linearAttnDims()
	K := cfg.LinearConvKernelDim
	eps := float32(cfg.RMSNormEps)
	p := func(str string) string { return layerName(l, str) }
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	lst := s.Cache.linear.layer(cfg, l)

	Xnq := qz(Xn, P, H)
	t := s.phaseStart()
	// The in_proj quad all reads the same post-norm panel Xn → one command buffer for whichever
	// members are q4_k-resident (in a q4_k_m Qwen3.6 the linear_attn.* projections are unpermuted
	// and resolve to Q8, so pgroup falls back to proj for them — harmless, no regression).
	ip := pgroup([]string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
	}, Xn, Xnq)
	mixed, zAll, bvec, avec := ip[0], ip[1], ip[2], ip[3]
	s.phaseEnd("qwen35_linear_in_proj", t)

	conv := m.tensor(p("linear_attn.conv1d.weight"))
	convOut := make([]float32, P*convDim)
	hist := lst.conv
	t = s.phaseStart()
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			outRow := convOut[t*convDim : (t+1)*convDim]
			for c := 0; c < convDim; c++ {
				var acc float32
				cb := c * K
				for j := 0; j < K; j++ {
					src := t + j - (K - 1)
					var row []float32
					switch {
					case src >= 0:
						row = mixed[src*convDim : (src+1)*convDim]
					default:
						idx := len(hist) + src
						if idx < 0 {
							continue
						}
						row = hist[idx]
					}
					acc += conv[cb+j] * row[c]
				}
				outRow[c] = silu(acc)
			}
		}
	})
	s.phaseEnd("qwen35_linear_conv", t)
	for t := 0; t < P; t++ {
		lst.pushConvRow(mixed[t*convDim:(t+1)*convDim], K-1)
	}

	aLog := m.tensor(p("linear_attn.A_log"))
	dtBias := m.tensor(p("linear_attn.dt_bias"))
	normW := m.tensor(p("linear_attn.norm.weight"))
	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	aExp := make([]float32, nV)
	for h := 0; h < nV; h++ {
		aExp[h] = float32(math.Exp(float64(aLog[h])))
	}
	core := make([]float32, P*valDim)
	qNormAll := make([]float32, P*keyDim)
	kNormAll := make([]float32, P*keyDim)
	t = s.phaseStart()
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			row := convOut[t*convDim : (t+1)*convDim]
			q := row[0:keyDim]
			k := row[keyDim : 2*keyDim]
			qNorm := qNormAll[t*keyDim : (t+1)*keyDim]
			kNorm := kNormAll[t*keyDim : (t+1)*keyDim]
			for h := 0; h < nK; h++ {
				l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
				l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
				for i := h * kHd; i < (h+1)*kHd; i++ {
					qNorm[i] *= scale
				}
			}
		}
	})
	s.phaseEnd("qwen35_linear_qk_norm", t)
	t = s.phaseStart()
	parFor(nV, numWorkers, func(lo, hi int) {
		for h := lo; h < hi; h++ {
			kh := h / repeat
			st := lst.recurrent[h]
			a := aExp[h]
			dtB := dtBias[h]
			kvmem := make([]float32, vHd)
			delta := make([]float32, vHd)
			for t := 0; t < P; t++ {
				row := convOut[t*convDim : (t+1)*convDim]
				qn := qNormAll[t*keyDim+kh*kHd : t*keyDim+(kh+1)*kHd]
				kn := kNormAll[t*keyDim+kh*kHd : t*keyDim+(kh+1)*kHd]
				vh := row[2*keyDim+h*vHd : 2*keyDim+(h+1)*vHd]
				bt := sigmoidf(bvec[t*nV+h])
				dt := softplus(avec[t*nV+h] + dtB)
				g := float32(math.Exp(float64(-a * dt)))
				for i := range st {
					st[i] *= g
				}
				for d := range kvmem {
					kvmem[d] = 0
				}
				for i := 0; i < kHd; i++ {
					ki := kn[i]
					base := i * vHd
					for d := 0; d < vHd; d++ {
						kvmem[d] += st[base+d] * ki
					}
				}
				for d := 0; d < vHd; d++ {
					delta[d] = (vh[d] - kvmem[d]) * bt
				}
				od := core[t*valDim+h*vHd : t*valDim+(h+1)*vHd]
				for i := 0; i < kHd; i++ {
					ki := kn[i]
					qi := qn[i]
					base := i * vHd
					for d := 0; d < vHd; d++ {
						st[base+d] += ki * delta[d]
						od[d] += st[base+d] * qi
					}
				}
			}
		}
	})
	s.phaseEnd("qwen35_linear_recurrent", t)
	t = s.phaseStart()
	parFor(P*nV, numWorkers, func(lo, hi int) {
		for idx := lo; idx < hi; idx++ {
			t := idx / nV
			h := idx - t*nV
			rmsNormGatedInPlace(
				core[t*valDim+h*vHd:t*valDim+(h+1)*vHd],
				normW,
				zAll[t*valDim+h*vHd:t*valDim+(h+1)*vHd],
				eps,
			)
		}
	})
	s.phaseEnd("qwen35_linear_gated_norm", t)
	t = s.phaseStart()
	O := proj(p("linear_attn.out_proj.weight"), core, qz(core, P, valDim))
	s.phaseEnd("qwen35_linear_out_proj", t)
	return O
}

func (s *Session) prefillQwen35FullAttnLayerQ4K(l int, Xn []float32, P, base int, proj hybridQ4KProj, pgroup hybridQ4KGroup, qz func([]float32, int, int) *q8Panel) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	qWidth := nH * hd
	w := nKV * hd
	grp := cfg.GroupSize()
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	p := func(str string) string { return layerName(l, str) }
	Xnq := qz(Xn, P, H)
	t := s.phaseStart()
	// q/k/v all read the same post-norm panel Xn → one command buffer for the group.
	qkv := pgroup([]string{p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")}, Xn, Xnq)
	qf, Kp, V := qkv[0], qkv[1], qkv[2]
	s.phaseEnd("qwen35_full_qkv_proj", t)
	Q := make([]float32, P*qWidth)
	gate := make([]float32, P*qWidth)
	t = s.phaseStart()
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			src := qf[t*2*qWidth : (t+1)*2*qWidth]
			for h := 0; h < nH; h++ {
				copy(Q[t*qWidth+h*hd:t*qWidth+(h+1)*hd], src[h*2*hd:h*2*hd+hd])
				copy(gate[t*qWidth+h*hd:t*qWidth+(h+1)*hd], src[h*2*hd+hd:h*2*hd+2*hd])
			}
		}
	})
	s.phaseEnd("qwen35_full_split_gate", t)
	t = s.phaseStart()
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			m.applyProjBias(l, Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w])
		}
	})
	s.Cache.Kraw[l] = append(s.Cache.Kraw[l], Kp...)
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			cos, sin := ropeRowForLayer(cfg, l, base+t)
			ropeRowQKInto(Q[t*qWidth:(t+1)*qWidth], Kp[t*w:(t+1)*w], cos, sin, hd, nH, nKV)
		}
	})
	s.Cache.K[l] = append(s.Cache.K[l], Kp...)
	s.Cache.V[l] = append(s.Cache.V[l], V...)
	s.phaseEnd("qwen35_full_qk_norm_rope", t)

	attnOut := make([]float32, P*qWidth)
	t = s.phaseStart()
	attnPrefillInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, s.M.attnObs)
	s.phaseEnd("qwen35_full_attn", t)
	t = s.phaseStart()
	for i := range attnOut {
		attnOut[i] *= sigmoidf(gate[i])
	}
	s.phaseEnd("qwen35_full_gate", t)
	t = s.phaseStart()
	O := proj(p("self_attn.o_proj.weight"), attnOut, qz(attnOut, P, qWidth))
	s.phaseEnd("qwen35_full_o_proj", t)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(O[t*H:(t+1)*H], p("self_attn.o_proj.bias"))
	}
	return O
}
