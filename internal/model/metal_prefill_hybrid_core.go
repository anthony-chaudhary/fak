package model

// metal_prefill_hybrid_core.go — the backend-agnostic core of the Qwen3.6 hybrid (Gated-DeltaNet)
// batched prefill. It is built in EVERY configuration (no build tag, no cgo): the projection /
// MLP GEMMs are abstracted behind an injected `hybridGemmFn`, so the same recurrence + attention +
// norm body runs against ANY matmul backend. The Metal twin (metal_prefill_hybrid.go on Apple
// Silicon+cgo) passes a GPU f16 GEMM; the parity test (metal_prefill_hybrid_core_test.go) passes a
// CPU Q8 GEMM that reproduces the proven prefillQwen35HybridQHidden path bit-for-bit (up to the
// documented grouped-vs-ungrouped float-order drift), which is what makes the twin's CPU-side logic
// witnessable host-independently — without a Mac Metal runtime (#71).
//
// This file is a structural copy of the Q8 CPU template prefillQwen35HybridQHidden /
// prefillQwen35LinearLayerQ / prefillQwen35FullAttnLayerQ (qwen35_prefill.go): every elementwise
// op — the embedding scale, both RMSNorms, the full-attention RoPE/GQA/output-gate, the conv1d+
// SiLU mixer, the q/k L2-norm, the delta-rule recurrent scan, the gated RMSNorm readout, and all
// the residuals — is the identical f32 CPU math, and ONLY the projection GEMMs are routed through
// `mm`. The split is the measured lever: the projection GEMMs are the bulk of the prefill wall
// while the GDN scan is ~0.5% (FAK_QPROFILE; #65, #977), so moving the projections to a FLOP-rich
// backend and keeping the recurrence on the CPU is the whole win (#71).

import (
	"fmt"
	"math"
	"os"
	"time"
)

// hybridGemmFn computes Y[P,out] = X[P,in] * W[name]^T into a fresh row-major buffer, where `in`
// is inferred from len(X)/P. It is the one substitution between the CPU template (Q8 qGemm8) and
// the Metal twin (GPU f16 MatMul); everything else in this file is identical f32 CPU math.
type hybridGemmFn = func(name string, X []float32, out int) []float32

// prefillQwen35HybridViaMM is the backend-agnostic Qwen3.6 hybrid batched prefill. It returns the
// last token's post-final-norm hidden (the caller applies the head); it assumes base == 0, the
// fresh-prefill precondition the routing gates on (s.Cache.Len() == 0).
func (s *Session) prefillQwen35HybridViaMM(ids []int, mm hybridGemmFn) []float32 {
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
	timedMM := mm
	if profile {
		timedMM = func(name string, X []float32, out int) []float32 {
			t0 := time.Now()
			Y := mm(name, X, out)
			gemmTime += time.Since(t0)
			return Y
		}
	}
	base := s.Cache.Len()
	eps := float32(cfg.RMSNormEps)

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		Xn := make([]float32, P*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})

		var o []float32
		if cfg.isLinearAttnLayer(l) {
			o = s.prefillQwen35LinearLayerMM(l, Xn, P, timedMM)
		} else {
			o = s.prefillQwen35FullAttnLayerMM(l, Xn, P, base, timedMM)
		}
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += o[i]
			}
		})

		Xn2 := make([]float32, P*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		G := timedMM(lp("mlp.gate_proj.weight"), Xn2, I)
		U := timedMM(lp("mlp.up_proj.weight"), Xn2, I)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		parFor(len(G), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = act(G[i], cfg) * U[i]
			}
		})
		Down := timedMM(lp("mlp.down_proj.weight"), G, H)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, base+t)
	}
	out := rmsnormCfg(X[(P-1)*H:P*H], m.tensor("model.norm.weight"), eps, cfg)
	if profile {
		total := time.Since(start)
		rest := total - gemmTime
		if rest < 0 {
			rest = 0
		}
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[metalprof-hybrid P=%d] total=%.1f  gemm+roundtrip=%.1f  rest(recurrence/attn/norm)=%.1f ms\n",
			P, ms(total), ms(gemmTime), ms(rest))
	}
	return out
}

// prefillQwen35LinearLayerMM is the backend-agnostic twin of prefillQwen35LinearLayerQ: the five
// linear_attn projection GEMMs (in_proj_qkv/z/b/a + out_proj) run through mm, the conv1d+SiLU
// mixer, the q/k L2-norm, the per-head delta-rule recurrent scan, and the gated RMSNorm readout
// are the identical f32 CPU recurrence, and the linearAttnCache it advances is the same one decode
// reads.
func (s *Session) prefillQwen35LinearLayerMM(l int, Xn []float32, P int, mm hybridGemmFn) []float32 {
	m, cfg := s.M, s.M.Cfg
	nK, nV, kHd, vHd, keyDim, valDim, convDim := cfg.linearAttnDims()
	K := cfg.LinearConvKernelDim
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	p := func(str string) string { return layerName(l, str) }
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	lst := s.Cache.linear.layer(cfg, l)

	mixed := mm(p("linear_attn.in_proj_qkv.weight"), Xn, convDim)
	zAll := mm(p("linear_attn.in_proj_z.weight"), Xn, valDim)
	bvec := mm(p("linear_attn.in_proj_b.weight"), Xn, nV)
	avec := mm(p("linear_attn.in_proj_a.weight"), Xn, nV)

	conv := m.tensor(p("linear_attn.conv1d.weight"))
	convOut := make([]float32, P*convDim)
	hist := lst.conv
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
	return mm(p("linear_attn.out_proj.weight"), core, H)
}

// prefillQwen35FullAttnLayerMM is the backend-agnostic twin of prefillQwen35FullAttnLayerQ: the
// q⧺gate, k, v and o projection GEMMs run through mm; the q/gate split, the projection bias and
// per-head QK-norm, RoPE, the causal GQA attention, the sigmoid output gate, and the Kraw/K/V
// cache appends are the identical f32 CPU math, so the KV cache it builds matches the proven path.
func (s *Session) prefillQwen35FullAttnLayerMM(l int, Xn []float32, P, base int, mm hybridGemmFn) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	qWidth := nH * hd
	w := nKV * hd
	grp := cfg.GroupSize()
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	p := func(str string) string { return layerName(l, str) }

	qf := mm(p("self_attn.q_proj.weight"), Xn, 2*qWidth)
	K := mm(p("self_attn.k_proj.weight"), Xn, w)
	V := mm(p("self_attn.v_proj.weight"), Xn, w)
	Q := make([]float32, P*qWidth)
	gate := make([]float32, P*qWidth)
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			src := qf[t*2*qWidth : (t+1)*2*qWidth]
			for h := 0; h < nH; h++ {
				copy(Q[t*qWidth+h*hd:t*qWidth+(h+1)*hd], src[h*2*hd:h*2*hd+hd])
				copy(gate[t*qWidth+h*hd:t*qWidth+(h+1)*hd], src[h*2*hd+hd:h*2*hd+2*hd])
			}
		}
	})
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

	attnOut := make([]float32, P*qWidth)
	attnPrefillInto(attnOut, Q, s.Cache.K[l], s.Cache.V[l], P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, s.M.attnObs)
	for i := range attnOut {
		attnOut[i] *= sigmoidf(gate[i])
	}
	O := mm(p("self_attn.o_proj.weight"), attnOut, H)
	for t := 0; t < P; t++ {
		m.addBiasIfPresent(O[t*H:(t+1)*H], p("self_attn.o_proj.bias"))
	}
	return O
}
