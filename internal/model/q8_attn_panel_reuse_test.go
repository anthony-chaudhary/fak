package model

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
	"time"
)

// Frozen native Q8 prefill reference: fresh attention output per layer,
// preserving request-local normalization panel reuse.
func (s *Session) prefillFreshAttnReference(ids []int) []float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	P := len(ids)
	base := s.Cache.Len()

	// legacy reconstructs the pre-optimization prefill (legacy per-element GEMM + serial
	// SwiGLU + naive single-accumulator attention dot) so FAK_QGEMM=legacy gives a clean
	// same-environment before/after A/B of the whole prefill, not just the GEMM kernel.
	legacy := qgemmMode == qgemmModeLegacy
	scoreDot := fdot
	if legacy {
		scoreDot = dot
	}

	var tQuant, tGemm, tAttn time.Duration
	t0 := time.Now()
	tic := func() time.Time {
		if qprofOn {
			return time.Now()
		}
		return time.Time{}
	}
	toc := func(d *time.Duration, t time.Time) {
		if qprofOn {
			*d += time.Since(t)
		}
	}
	gemm := func(qt *q8Tensor, qp *q8Panel) []float32 {
		t := tic()
		r := qGemm8(qt, qp)
		toc(&tGemm, t)
		return r
	}
	// One reused scratch panel for all 4×NumLayers activation quantizations: each panel is
	// fully consumed before the next is built (q/k/v → o → gate/up → down), so a single
	// buffer is safe and avoids ~120 large allocations per prefill.
	scratch := &q8Panel{}
	qz := func(X []float32, P, width int) *q8Panel {
		t := tic()
		quantizeBatchPanelInto(scratch, X, P, width)
		toc(&tQuant, t)
		return scratch
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	// Normalization consumers finish synchronously before the next norm stage.
	// Every row is overwritten, so one request-local panel serves both stages.
	normPanel := make([]float32, P*H)
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		ql := m.q8Layer(l)

		// q8PrefillNeedsTokenLoop (kv.go:825) has no LayerNorm term, so a quantized PreNorm
		// LayerNorm family prefills HERE while decoding through the bias-aware blockStep. The
		// learned input_layernorm.bias must therefore ride along; rmsnormCfg hard-passes nil.
		Xn := normPanel
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			bIn := m.tensorOptional(lp("input_layernorm.bias"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wIn, bIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		Xnq := qz(Xn, P, H)

		Q := gemm(ql.qProj, Xnq)
		K := gemm(ql.kProj, Xnq)
		V := gemm(ql.vProj, Xnq)
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Stash raw (pre-RoPE, post-qk-norm) K straight into the cache, THEN RoPE K in place — this is the
		// same bytes the old `Kraw := append(nil, K...)` temp captured, without the extra
		// 196KB alloc+copy per layer (~5.9MB/prefill of GC churn the "rest" phase paid for).
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		attnOut := make([]float32, P*nH*hd)
		tA := tic()
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, scoreDot, s.M.attnObs)
		toc(&tAttn, tA)

		O := gemm(ql.oProj, qz(attnOut, P, nH*hd))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := normPanel
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wPost, bPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		Xn2q := qz(Xn2, P, H)
		G := gemm(ql.gateProj, Xn2q)
		U := gemm(ql.upProj, Xn2q)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		if legacy {
			for i := range G {
				G[i] = act(G[i], cfg) * U[i]
			}
		} else {
			parFor(len(G), dispatchWorkers, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					G[i] = act(G[i], cfg) * U[i]
				}
			})
		}
		Down := gemm(ql.downProj, qz(G, P, I))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	for t := 0; t < P; t++ {
		s.Cache.appendPosition(base+t, ids[t])
	}
	if qprofOn {
		total := time.Since(t0)
		rest := total - tGemm - tAttn - tQuant
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[qprof P=%d] total=%.1f  gemm=%.1f  attn=%.1f  quant=%.1f  rest(norm/rope/resid)=%.1f ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(tQuant), ms(rest))
	}
	last := X[(P-1)*H : P*H]
	// finalNorm, not a hand-rolled normCfg: it is the ONE place the final-norm weight, its
	// optional bias, and eps are bound together, so this lane cannot drift from the per-token
	// path again the way the hard-coded nil bias here did.
	return m.finalNorm(last)
}

func TestQ8AttnPanelReuseExactAndAllocation(t *testing.T) {
	old := NumWorkers()
	defer SetWorkers(old)
	for _, workers := range []int{1, 2} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, mode := range []string{"rms", "bias", "gain"} {
			cfg := normBiasArchConfig()
			cfg.LayerNorm = mode == "bias"
			cfg.NormGain1p = mode == "gain"
			m := newSyntheticExtra(cfg, normBiasExtras(cfg))
			m.Quantize()
			if q8PrefillNeedsTokenLoop(cfg) {
				t.Fatal("test does not exercise batched Q8")
			}
			a, b := m.NewSession(), m.NewSession()
			a.Quant = true
			b.Quant = true
			for _, ids := range [][]int{{1, 3, 5, 7, 9, 11, 13, 15, 17}, {19, 21, 23}} {
				got, want := a.prefillBatchedQ(ids), b.prefillFreshAttnReference(ids)
				exact := func(x, y []float32) {
					t.Helper()
					if len(x) != len(y) {
						t.Fatalf("length mismatch: %d != %d", len(x), len(y))
					}
					for i := range x {
						if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
							t.Fatalf("mode=%s workers=%d index=%d", mode, workers, i)
						}
					}
				}
				exact(got, want)
				exact(a.headQ(got), b.headQ(want))
				assertCachesBitEqual(t, fmt.Sprintf("mode=%s workers=%d", mode, workers), cfg, b.Cache, a.Cache)
				if !reflect.DeepEqual(a.Cache, b.Cache) {
					t.Fatalf("KV differs mode=%s workers=%d", mode, workers)
				}
			}
			a.Close()
			b.Close()
			ids := []int{1, 3, 5, 7, 9, 11, 13, 15, 17}
			measure := func(reference bool) int64 {
				return testing.Benchmark(func(bb *testing.B) {
					for i := 0; i < bb.N; i++ {
						ss := m.NewSession()
						ss.Quant = true
						if reference {
							ss.prefillFreshAttnReference(ids)
						} else {
							ss.prefillBatchedQ(ids)
						}
						ss.Close()
					}
				}).AllocedBytesPerOp()
			}
			before, after := measure(true), measure(false)
			t.Logf("engine=fak-native CPU mode=%s workers=%d before=%d after=%d B/op exact=true", mode, workers, before, after)
			expectedSaving := int64((cfg.NumLayers - 1) * len(ids) * cfg.NumHeads * cfg.HeadDim * 4)
			// Benchmark byte totals include unrelated runtime allocations and truncate
			// independently averaged B/op. Require 90% of the removed panel bytes,
			// not an exact lower bound that can fail on a one-byte rounding difference.
			// The original fresh-panel path saves approximately zero and must fail.
			minimumSaving := expectedSaving * 9 / 10
			if before-after < minimumSaving {
				t.Errorf("missing attention panel savings: %d -> %d B/op; saved=%d minimum=%d", before, after, before-after, minimumSaving)
			}
		}
	}
}
