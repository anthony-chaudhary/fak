package model

import "math"

// qwen35_prefill.go is the Q8 fresh-prefill path for Qwen3.5/Qwen3.6 hybrid
// Gated-DeltaNet models. The generic Q8 prefill batcher deliberately refuses this
// family because three of four layers are recurrent linear attention, not ordinary
// causal attention. This path preserves that recurrence but batches each layer's
// projection/MLP GEMMs over the prompt panel so prompt processing stops being just
// repeated decode.

const qwen35HybridQBatchMinPrompt = 16

func q8Qwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return cfg.IsQwen35Hybrid() &&
		promptLen >= qwen35HybridQBatchMinPrompt &&
		cfg.AttnOutputGate &&
		!cfg.IsMoE() &&
		!cfg.DenseMLP &&
		!cfg.Alibi &&
		cfg.NormGain1p &&
		!cfg.LayerNorm &&
		cfg.BlockTopology == PreNorm &&
		!cfg.hasLayerSpecificRopeTheta()
}

func (s *Session) prefillQwen35HybridQ(ids []int) []float32 {
	xf := s.prefillQwen35HybridQHidden(ids)
	return s.headQ(xf)
}

func (s *Session) prefillQwen35HybridQNoLogits(ids []int) {
	_ = s.prefillQwen35HybridQHidden(ids)
}

func (s *Session) prefillQwen35HybridQHidden(ids []int) []float32 {
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
	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg)
	}
	s.phaseEnd("embed", t)

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
			o = s.prefillQwen35LinearLayerQ(l, Xn, P, qz)
		} else {
			o = s.prefillQwen35FullAttnLayerQ(l, Xn, P, base, qz)
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
		parFor(len(G), numWorkers, func(lo, hi int) {
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
	return xf
}

func (s *Session) prefillQwen35LinearLayerQ(l int, Xn []float32, P int, qz func([]float32, int, int) *q8Panel) []float32 {
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
	O := make([]float32, P*H)
	coreQ := qz(core, P, valDim)
	t = s.phaseStart()
	qGemm8Into(m.q8(p("linear_attn.out_proj.weight")), coreQ, O)
	s.phaseEnd("qwen35_linear_out_proj", t)
	return O
}

func (s *Session) prefillQwen35FullAttnLayerQ(l int, Xn []float32, P, base int, qz func([]float32, int, int) *q8Panel) []float32 {
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
			m.applyProjBias(l, Q[t*qWidth:(t+1)*qWidth], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*qWidth:(t+1)*qWidth], K[t*w:(t+1)*w])
		}
	})
	s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			cos, sin := ropeRowForLayer(cfg, l, base+t)
			ropeRowQKInto(Q[t*qWidth:(t+1)*qWidth], K[t*w:(t+1)*w], cos, sin, hd, nH, nKV)
		}
	})
	s.Cache.K[l] = append(s.Cache.K[l], K...)
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
