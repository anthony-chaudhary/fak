package model

import (
	"fmt"
	"math"
	"os"
	"time"
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
			// sessionQ4KKernel.mul. The P token rows are looped SERIALLY because kQuantMatRowsInto
			// already parallelizes across the qt.out output rows via parFor; wrapping it in an
			// outer parFor would re-enter parDispatchMu and DEADLOCK (parFor is not re-entrant).
			in := qt.in
			Y := make([]float32, P*qt.out)
			for t := 0; t < P; t++ {
				kQuantMatRowsInto(qt, Xf[t*in:(t+1)*in], Y[t*qt.out:(t+1)*qt.out])
			}
			return Y
		}
		return s.q8GemmDispatch(name, m.q8(name), Xq)
	}
	if profile {
		rawProj := proj
		proj = func(name string, Xf []float32, Xq *q8Panel) []float32 {
			t0 := time.Now()
			Y := rawProj(name, Xf, Xq)
			gemmTime += time.Since(t0)
			return Y
		}
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
			o = s.prefillQwen35LinearLayerQ4K(l, Xn, P, proj, qz)
		} else {
			o = s.prefillQwen35FullAttnLayerQ4K(l, Xn, P, base, proj, qz)
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
		G := proj(lp("mlp.gate_proj.weight"), Xn2, Xn2q)
		U := proj(lp("mlp.up_proj.weight"), Xn2, Xn2q)
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
	}
	return xf
}

func (s *Session) prefillQwen35LinearLayerQ4K(l int, Xn []float32, P int, proj hybridQ4KProj, qz func([]float32, int, int) *q8Panel) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	nK := cfg.LinearNumKeyHeads
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	keyDim := nK * kHd
	valDim := nV * vHd
	convDim := 2*keyDim + valDim
	K := cfg.LinearConvKernelDim
	eps := float32(cfg.RMSNormEps)
	p := func(str string) string { return layerName(l, str) }
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	lst := s.Cache.linear.layer(cfg, l)

	Xnq := qz(Xn, P, H)
	t := s.phaseStart()
	mixed := proj(p("linear_attn.in_proj_qkv.weight"), Xn, Xnq)
	zAll := proj(p("linear_attn.in_proj_z.weight"), Xn, Xnq)
	bvec := proj(p("linear_attn.in_proj_b.weight"), Xn, Xnq)
	avec := proj(p("linear_attn.in_proj_a.weight"), Xn, Xnq)
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

func (s *Session) prefillQwen35FullAttnLayerQ4K(l int, Xn []float32, P, base int, proj hybridQ4KProj, qz func([]float32, int, int) *q8Panel) []float32 {
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
	qf := proj(p("self_attn.q_proj.weight"), Xn, Xnq)
	Kp := proj(p("self_attn.k_proj.weight"), Xn, Xnq)
	V := proj(p("self_attn.v_proj.weight"), Xn, Xnq)
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
