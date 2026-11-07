package model

import (
	"math"
	"os"
)

// qwen35_prefill.go is the Q8 host-prefill path for Qwen3.5/Qwen3.6 hybrid
// Gated-DeltaNet models. The generic Q8 prefill batcher deliberately refuses this
// family because three of four layers are recurrent linear attention, not ordinary
// causal attention. This path preserves that recurrence but batches each layer's
// projection/MLP GEMMs over the prompt panel so prompt processing stops being just
// repeated decode.

const qwen35HybridQBatchMinPrompt = 16

// q8Qwen35HybridPrefillOK gates the batched Q8 hybrid prefill. It admits the
// Qwen3.5/3.6 MoE hybrid (qwen3_5_moe, e.g. Qwen3.6-35B-A3B): the attention and
// linear-attention sublayers stay batched and only the routed FFN runs per row
// (prefillQwen35RowFFNQ), through the token loop's exact ffnForLayer(l).apply call on
// the token loop's kernel. The resident-Q4_K lane and the Metal twin share this gate.
func q8Qwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return qwen35HybridPrefillArchOK(cfg, promptLen)
}

// qwen35HybridPrefillArchOK holds every non-FFN axis the batched hybrid panel models.
func qwen35HybridPrefillArchOK(cfg Config, promptLen int) bool {
	// Cross-quant diagnostic for #4273: force the established token-at-a-time
	// prefill so real-model runs can isolate the batched hybrid prefill from the
	// tokenizer, sampler, and decode path. Default performance stays unchanged.
	if os.Getenv("FAK_QWEN35_PREFILL_TOKEN_LOOP") != "" {
		return false
	}
	return cfg.IsQwen35Hybrid() &&
		promptLen >= qwen35HybridQBatchMinPrompt &&
		cfg.AttnOutputGate &&
		!cfg.DenseMLP &&
		!cfg.Alibi &&
		cfg.NormGain1p &&
		!cfg.LayerNorm &&
		cfg.BlockTopology == PreNorm &&
		!cfg.hasLayerSpecificRopeTheta()
}

// q8Qwen35HybridPrefillSessionOK admits complete float-KV prefixes, including
// restored recurrent state. Packed KV and scalar-only capture/steering/model
// transforms retain token-loop execution; the panel must not silently skip them.
func (s *Session) q8Qwen35HybridPrefillSessionOK(promptLen int) bool {
	return q8Qwen35HybridPrefillOK(s.M.Cfg, promptLen) && s.qwen35HybridPrefillObserversOK()
}

// qwen35HybridPrefillObserversOK is the session half of the batched hybrid admission:
// packed KV, target-hidden capture, residual hooks, PRISM transforms, hidden taps and
// attention sinks all run only inside the token loop, so any of them keeps it.
func (s *Session) qwen35HybridPrefillObserversOK() bool {
	if s.Cache.quantized() ||
		s.captureTargetHidden || s.M.Cfg.EnableResidualHook || s.M.prism != nil || s.activeTap() != nil {
		return false
	}
	for layer := 0; layer < s.M.Cfg.NumLayers; layer++ {
		if s.M.hasWeight(layerName(layer, "self_attn.sinks")) {
			return false
		}
	}
	return true
}

func (s *Session) prefillQwen35HybridQ(ids []int) []float32 {
	xf := s.prefillQwen35HybridQHidden(ids)
	return s.headQ(xf)
}

func (s *Session) prefillQwen35HybridQNoLogits(ids []int) {
	_ = s.prefillQwen35HybridQHidden(ids)
}

func (s *Session) prefillQwen35HybridQHidden(ids []int) []float32 {
	return s.prefillQwen35HybridQHiddenWithNormWorkspace(ids, s.newQwen35PrefillNormWorkspace(len(ids)))
}

func (s *Session) prefillQwen35HybridQHiddenWithNormWorkspace(ids []int, normWorkspace *qwen35PrefillNormWorkspace) []float32 {
	return s.prefillQwen35HybridQHiddenWithGDNMode(ids, normWorkspace, qwen35GDNHostBuildSelected)
}

func (s *Session) prefillQwen35HybridQHiddenWithGDNMode(ids []int, normWorkspace *qwen35PrefillNormWorkspace, gdnMode qwen35GDNHostMode) []float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	P := len(ids)
	if P == 0 {
		return nil
	}
	base := s.Cache.Len()
	eps := float32(cfg.RMSNormEps)
	scratch := &q8Panel{}
	qz := func(X []float32, rows, width int) *q8Panel {
		t := s.phaseStart()
		quantizeBatchPanelInto(scratch, X, rows, width)
		s.phaseEnd("q8_panel_quantize", t)
		return scratch
	}

	t := s.phaseStart()
	X := make([]float32, P*H)
	if m.Q2KEmbedding != nil {
		scale := cfg.embedScale()
		for t, id := range ids {
			if err := m.Q2KEmbedding.GatherRow(id, X[t*H:(t+1)*H], scale); err != nil {
				panic(err)
			}
		}
	} else {
		embed := m.embedRows()
		for t, id := range ids {
			copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
			scaleEmbedInPlace(X[t*H:(t+1)*H], cfg)
		}
	}
	s.phaseEnd("embed", t)
	ropeRows := s.newQwen35PrefillRopeRows(base, P)

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		Xn := normWorkspace.panel(P, H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		t = s.phaseStart()
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					qwen35PrefillNormIntoDisjoint(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps, cfg)
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		s.phaseEnd("input_norm", t)

		var o []float32
		if cfg.isLinearAttnLayer(l) {
			o = s.prefillQwen35LinearLayerQWithHostChunks(l, Xn, P, qz, qwen35GDNHostModeAllows(gdnMode))
		} else {
			o = s.prefillQwen35FullAttnLayerQWithRopeRows(l, Xn, P, base, qz, ropeRows)
		}
		t = s.phaseStart()
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += o[i]
			}
		})
		s.phaseEnd("attn_residual", t)

		if cfg.IsMoE() {
			// Routed FFN: each row picks its own top-k experts, so it runs per row
			// through the token loop's exact PreNorm sublayer and kernel.
			t = s.phaseStart()
			s.prefillQwen35RowFFNQ(l, X, P)
			s.phaseEnd("qwen35_moe_row_ffn", t)
			continue
		}

		Xn2 := normWorkspace.panel(P, H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		t = s.phaseStart()
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					qwen35PrefillNormIntoDisjoint(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps, cfg)
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		s.phaseEnd("post_attn_norm", t)
		I := cfg.IntermediateSize
		Xn2q := qz(Xn2, P, H)
		G := make([]float32, P*I)
		U := make([]float32, P*I)
		t = s.phaseStart()
		qGemm8IntoMany(Xn2q,
			qgemm8Target{qt: m.q8(lp("mlp.gate_proj.weight")), Y: G},
			qgemm8Target{qt: m.q8(lp("mlp.up_proj.weight")), Y: U},
		)
		s.phaseEnd("mlp_gate_up_proj", t)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		t = s.phaseStart()
		parFor(len(G), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = act(G[i], cfg) * U[i]
			}
		})
		s.phaseEnd("mlp_activation", t)
		Down := make([]float32, P*H)
		Gq := qz(G, P, I)
		t = s.phaseStart()
		qGemm8Into(m.q8(lp("mlp.down_proj.weight")), Gq, Down)
		s.phaseEnd("mlp_down_proj", t)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		t = s.phaseStart()
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
		s.phaseEnd("mlp_residual", t)
	}

	t = s.phaseStart()
	for t := 0; t < P; t++ {
		s.Cache.appendPosition(base+t, ids[t])
	}
	s.phaseEnd("cache_positions", t)
	t = s.phaseStart()
	xf := rmsnormCfg(X[(P-1)*H:P*H], m.tensor("model.norm.weight"), eps, cfg)
	s.phaseEnd("final_norm", t)
	return xf
}

// prefillQwen35RowFFNQ applies layer l's FFN sublayer to each of the P rows of X in
// place, in token order, through prefillRowFFN (blockStep's composeSublayer(PreNorm) +
// mlpNorms + ffnForLayer(l).apply) on tokenHiddenQKernel, the kernel the token loop hands
// blockStep: q8Kernel for a Q8 MoE session, sessionQ4KKernel with resident Q4_K weights.
// So each row's FFN delta (routed experts plus the gated qwen35SharedExpert) is the token
// loop's own arithmetic for the same residual input.
func (s *Session) prefillQwen35RowFFNQ(l int, X []float32, P int) {
	m, cfg := s.M, s.M.Cfg
	m.prefillRowFFN(l, X, P, cfg.HiddenSize, float32(cfg.RMSNormEps), cfg, s.tokenHiddenQKernel())
}

func (s *Session) prefillQwen35LinearLayerQ(l int, Xn []float32, P int, qz func([]float32, int, int) *q8Panel) []float32 {
	return s.prefillQwen35LinearLayerQWithHostChunks(l, Xn, P, qz, false)
}

// The ordinary Q8 panel owns this input and state; direct callback callers use
// the non-selecting wrapper above.
func (s *Session) prefillQwen35LinearLayerQWithHostChunks(l int, Xn []float32, P int, qz func([]float32, int, int) *q8Panel, ordinaryHost bool) []float32 {
	dispatchWorkers := currentWorkerCount()
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
	mixed := make([]float32, P*convDim)
	zAll := make([]float32, P*valDim)
	bvec := make([]float32, P*nV)
	avec := make([]float32, P*nV)
	t := s.phaseStart()
	qGemm8IntoMany(Xnq,
		qgemm8Target{qt: m.q8(p("linear_attn.in_proj_qkv.weight")), Y: mixed},
		qgemm8Target{qt: m.q8(p("linear_attn.in_proj_z.weight")), Y: zAll},
		qgemm8Target{qt: m.q8(p("linear_attn.in_proj_b.weight")), Y: bvec},
		qgemm8Target{qt: m.q8(p("linear_attn.in_proj_a.weight")), Y: avec},
	)
	s.phaseEnd("qwen35_linear_in_proj", t)

	conv := m.tensor(p("linear_attn.conv1d.weight"))
	convOut := make([]float32, P*convDim)
	hist := lst.conv
	t = s.phaseStart()
	if K != 4 || !qwen35PrefillConv4(convOut, mixed, conv, hist, P, convDim, dispatchWorkers) {
		parFor(P, dispatchWorkers, func(lo, hi int) {
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
	}
	s.phaseEnd("qwen35_linear_conv", t)
	if !lst.pushConvPrefillTail(mixed, P, convDim, K-1) {
		for t := 0; t < P; t++ {
			lst.pushConvRow(mixed[t*convDim:(t+1)*convDim], K-1)
		}
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
	parFor(P, dispatchWorkers, func(lo, hi int) {
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
	// Prefill reuse of the existing decode kernel is explicitly opt-in. Do
	// not promote the default from source reasoning: hardware float32 FMA
	// and the scalar float64-FMA-to-float32 path can differ even for finite
	// inputs. Existing numeric/physical qualification remains required.
	vectorized := os.Getenv("FAK_VECTORIZED_DELTANET") == "1" && HasVectorizedDeltaNetFor(kHd, vHd)
	recurrentPhase := "qwen35_linear_recurrent"
	chunked := false
	if s.qwen35GDNHostSelected(ordinaryHost, vectorized, P) {
		result, ok := qwen35GDNHostCompute(qwen35GDNHostInput{
			tokens: P, keyHeads: nK, valueHeads: nV, keyWidth: kHd, valueWidth: vHd,
			q: qNormAll, k: kNormAll, conv: convOut, a: avec, b: bvec,
			aExp: aExp, dtBias: dtBias, state: lst.recurrent,
		}, dispatchWorkers)
		if ok {
			core = result.core
			for h := range lst.recurrent {
				copy(lst.recurrent[h], result.state[h])
			}
			chunked = true
			recurrentPhase = "qwen35_linear_chunked_host"
		}
	}
	if !chunked {
		parFor(nV, dispatchWorkers, func(lo, hi int) {
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
					if vectorized && qwen35PrefillVectorizedStep(st, qn, kn, vh, core, kvmem, delta, bt, g, t, valDim, h) {
						continue
					}
					// The ordinary head state and scratch are disjoint exact-shape
					// rows. Decay each cell when retrieving it instead of making a
					// separate full-state pass. Refused geometry keeps the old mutation
					// and panic order, including extra state/scratch elements.
					if !qwen35PrefillDecayRead(st, kn, kvmem, g, kHd, vHd) {
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
								kvmem[d] = gdnFMA32(st[base+d], ki, kvmem[d])
							}
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
							st[base+d] = gdnFMA32(ki, delta[d], st[base+d])
							od[d] = gdnFMA32(st[base+d], qi, od[d])
						}
					}
				}
			}
		})
	}
	s.phaseEnd(recurrentPhase, t)
	t = s.phaseStart()
	parFor(P*nV, dispatchWorkers, func(lo, hi int) {
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
	O := make([]float32, P*H)
	coreQ := qz(core, P, valDim)
	t = s.phaseStart()
	qGemm8Into(m.q8(p("linear_attn.out_proj.weight")), coreQ, O)
	s.phaseEnd("qwen35_linear_out_proj", t)
	return O
}

func (s *Session) prefillQwen35FullAttnLayerQ(l int, Xn []float32, P, base int, qz func([]float32, int, int) *q8Panel) []float32 {
	return s.prefillQwen35FullAttnLayerQWithRopeRows(l, Xn, P, base, qz, nil)
}

func (s *Session) prefillQwen35FullAttnLayerQWithRopeRows(l int, Xn []float32, P, base int, qz func([]float32, int, int) *q8Panel, ropeRows *qwen35PrefillRopeRows) []float32 {
	dispatchWorkers := currentWorkerCount()
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
	qf := make([]float32, P*2*qWidth)
	K := make([]float32, P*w)
	V := make([]float32, P*w)
	t := s.phaseStart()
	qGemm8IntoMany(Xnq,
		qgemm8Target{qt: m.q8(p("self_attn.q_proj.weight")), Y: qf},
		qgemm8Target{qt: m.q8(p("self_attn.k_proj.weight")), Y: K},
		qgemm8Target{qt: m.q8(p("self_attn.v_proj.weight")), Y: V},
	)
	s.phaseEnd("qwen35_full_qkv_proj", t)
	Q := make([]float32, P*qWidth)
	gate := make([]float32, P*qWidth)
	t = s.phaseStart()
	parFor(P, dispatchWorkers, func(lo, hi int) {
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
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			m.applyProjBias(l, Q[t*qWidth:(t+1)*qWidth], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*qWidth:(t+1)*qWidth], K[t*w:(t+1)*w])
		}
	})
	s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
	reuseRope := ropeRows.prepare(cfg, l, base, P)
	parFor(P, dispatchWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			var cos, sin []float32
			if reuseRope {
				cos, sin = ropeRows.row(t)
			} else {
				cos, sin = ropeRowForLayer(cfg, l, base+t)
			}
			ropeRowQKInto(Q[t*qWidth:(t+1)*qWidth], K[t*w:(t+1)*w], cos, sin, hd, nH, nKV)
		}
	})
	s.Cache.K[l] = append(s.Cache.K[l], K...)
	s.Cache.V[l] = append(s.Cache.V[l], V...)
	s.phaseEnd("qwen35_full_qk_norm_rope", t)

	attnOut := make([]float32, P*qWidth)
	t = s.phaseStart()
	// qf is dead after the joined split: Q and gate own independent copies,
	// projection kernels retain no target, and phase profiling records only time.
	// Loan it as bounded score scratch only for unobserved builtin attention.
	W := cfg.windowForLayer(l)
	// Qualified host-only tiled KV reuse takes the same dead loan. Every
	// refusal precedes writes, so the original GQA3/score-row oracle is intact.
	tiled := qwen35PrefillTiledAttentionQualification && !s.Metal && !s.MetalQ4K && s.Backend == nil && m.prism == nil &&
		!cfg.Alibi && !m.hasWeight(p("self_attn.sinks")) &&
		qwen35PrefillTiledAttentionInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], qf, P, base, nH, hd, w, grp, W, scale, attnCap, s.M.attnObs)
	if !tiled && !attnPrefillGQA3FDotInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], qf, P, base, nH, hd, w, grp, W, scale, attnCap, s.M.attnObs) {
		attnPrefillInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], P, base, nH, hd, w, grp, W, l, scale, attnCap, fdot, s.M.attnObs)
	}
	s.phaseEnd("qwen35_full_attn", t)
	t = s.phaseStart()
	qwen35PrefillGateDisjoint(attnOut, gate, P, qWidth)
	s.phaseEnd("qwen35_full_gate", t)
	O := make([]float32, P*H)
	attnQ := qz(attnOut, P, qWidth)
	t = s.phaseStart()
	qGemm8Into(m.q8(p("self_attn.o_proj.weight")), attnQ, O)
	s.phaseEnd("qwen35_full_o_proj", t)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(O[t*H:(t+1)*H], p("self_attn.o_proj.bias"))
	}
	return O
}
