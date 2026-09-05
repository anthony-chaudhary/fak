package model

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
	"time"
)

// Frozen allocating native CPU prefill oracle for #11595. Keep Xn and Xn2
// independently allocated inside each layer; do not apply reuse optimizations here.
func (s *Session) prefillQ4KFreshNormReference(ids []int) []float32 {
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
	// One reused Q8 activation-panel scratch for the minority projections that need it
	// (q/k always; plus any Q6_K minority such as down_proj). Each panel is fully consumed
	// before the next is built, so a single buffer is safe — same discipline as prefillBatchedQ.
	scratch := &q8Panel{}
	qz := func(X []float32, P, width int) *q8Panel {
		t := tic()
		quantizeBatchPanelInto(scratch, X, P, width)
		toc(&tQuant, t)
		return scratch
	}
	// proj dispatches a batched projection [P,out] by resident format: q4kw-resident →
	// q4kGemmDispatch on the f32 activation Xf; kqw-resident → kQuantGemmDispatch;
	// otherwise → q8GemmDispatch on the Q8 panel Xq (with m.q8 quantizing on demand from
	// the f32 manifest if un-quantized). The width is inferred from the resident tensor's
	// .out, so the caller does not pass it. Xq may be nil when the caller knows the projection
	// is q4k-resident (it is only read on the q8 branch); passing the matching panel is the
	// caller's responsibility for minority names.
	//
	// q4kGemmDispatch is the CPU q4kGemm by default (pure-Go build, bit-identical to before);
	// under -tags fakmetal with s.MetalQ4K set it routes the q4_k-majority batched GEMM to the
	// Metal q4_k dequant-GEMM, the same GPU path decode's GEMV already takes — so the resident-
	// Q4_K prefill runs on the GPU instead of the slow CPU GEMM that timed out real prompts
	// (#1071), mirroring the already-dispatched qwen35-hybrid prefill (qwen35_prefill_q4k.go).
	proj := func(name string, Xf []float32, Xq *q8Panel) []float32 {
		t := tic()
		var r []float32
		if qt := m.q4kw[name]; qt != nil {
			r = s.q4kGemmDispatch(name, qt, Xf, P)
		} else if qt := m.kqw[name]; qt != nil {
			r = s.kQuantGemmDispatch(name, qt, Xf, P)
		} else {
			r = s.q8GemmDispatch(name, m.q8(name), Xq)
		}
		toc(&tGemm, t)
		return r
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama/Qwen
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	if s.MetalQ4K {
		m.metalQ4KWeights() // upload all Q4_K weights upfront — avoids per-call GPU round-trips (#1113)
		m.metalQ8Weights()  // upload Q8-minority projection weights upfront for Metal Q8 prefill (#1087)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }

		// kv.go:670 routes here on !q8PrefillNeedsTokenLoop, which has no LayerNorm term, so a
		// resident-Q4_K PreNorm LayerNorm family prefills HERE while decoding through the
		// bias-aware blockStep. The learned input_layernorm.bias must ride along; rmsnormCfg
		// hard-passes nil.
		Xn := make([]float32, P*H)
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
		// Xnq feeds the Q8-minority projections (q/k at minimum). The q4_k_m majority reads
		// raw f32 Xn directly, so the panel is built once and consumed by whichever of
		// q/k/v are on the Q8 path; a q4k-resident v just ignores it.
		Xnq := qz(Xn, P, H)

		Q := proj(lp("self_attn.q_proj.weight"), Xn, Xnq)
		K := proj(lp("self_attn.k_proj.weight"), Xn, Xnq)
		V := proj(lp("self_attn.v_proj.weight"), Xn, Xnq)
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Stash raw (pre-RoPE, post-qk-norm) K straight into the cache, THEN RoPE K in place —
		// same bytes the per-token path's Kraw captures, no extra alloc+copy per layer.
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
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, nil)
		toc(&tAttn, tA)

		O := proj(lp("self_attn.o_proj.weight"), attnOut, qz(attnOut, P, nH*hd))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		parFor(len(X), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := make([]float32, P*H)
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
		G := proj(lp("mlp.gate_proj.weight"), Xn2, Xn2q)
		U := proj(lp("mlp.up_proj.weight"), Xn2, Xn2q)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		parFor(len(G), dispatchWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = act(G[i], cfg) * U[i]
			}
		})
		Down := proj(lp("mlp.down_proj.weight"), G, qz(G, P, I))
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
		fmt.Fprintf(os.Stderr, "[q4kprof P=%d] total=%.1f  gemm=%.1f  attn=%.1f  quant=%.1f  rest(norm/rope/resid)=%.1f ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(tQuant), ms(rest))
	}
	last := X[(P-1)*H : P*H]
	// finalNorm, not a hand-rolled normCfg: it is the ONE place the final-norm weight, its
	// optional bias, and eps are bound together, so this lane cannot drift from the per-token
	// path again the way the hard-coded nil bias here did.
	return m.finalNorm(last)
}

func TestPrefillQ4KNormReuseExactAndAllocation(t *testing.T) {
	old := NumWorkers()
	defer SetWorkers(old)
	for _, workers := range []int{1, 2} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, mode := range []string{"rms", "bias", "gain"} {
			t.Run(fmt.Sprintf("%s/workers=%d", mode, workers), func(t *testing.T) {
				cfg := Config{HiddenSize: 256, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
					IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, AttnSoftcap: 50, RopeTheta: 10000}
				cfg.LayerNorm = mode == "bias"
				cfg.NormGain1p = mode == "gain"
				m := newSyntheticExtra(cfg, normBiasExtras(cfg))
				var projs [][2]any
				for l := 0; l < cfg.NumLayers; l++ {
					p := layerPrefix(l)
					// Q/K stay Q8; native Q4_K consumes both normalization panels directly.
					projs = append(projs,
						[2]any{p + "self_attn.v_proj.weight", cfg.NumKVHeads * cfg.HeadDim},
						[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
						[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
						[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
						[2]any{p + "mlp.down_proj.weight", cfg.HiddenSize})
				}
				fillQ4KW(t, m, projs, 11595)
				a, b := m.NewSession(), m.NewSession()
				defer a.Close()
				defer b.Close()
				a.Q4K, b.Q4K = true, true
				if a.MetalQ4K || b.MetalQ4K {
					t.Fatal("requires native CPU dispatch")
				}
				exact := func(label string, x, y []float32) {
					t.Helper()
					if len(x) != len(y) {
						t.Fatalf("%s length %d != %d", label, len(x), len(y))
					}
					for i := range x {
						if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) {
							t.Fatalf("%s[%d] nonfinite", label, i)
						}
						if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
							t.Fatalf("%s[%d] bits %08x != %08x", label, i, math.Float32bits(x[i]), math.Float32bits(y[i]))
						}
					}
				}
				total := 0
				// Unequal chunks exercise nonzero RoPE/cache bases and request-local resizing.
				chunks := [][]int{{1, 3, 5, 7, 9, 11, 13, 15, 17}, {19, 21, 23}, {25, 27, 29, 31, 33}}
				for chunk, ids := range chunks {
					got, want := a.prefillBatchedQ4K(ids), b.prefillQ4KFreshNormReference(ids)
					exact("hidden", got, want)
					exact("logits", a.headResident(got), b.headResident(want))
					total += len(ids)
					if a.Cache.Len() != total || b.Cache.Len() != total {
						t.Fatal("cache length")
					}
					for l := 0; l < cfg.NumLayers; l++ {
						for _, pair := range []struct {
							name string
							x, y []float32
						}{
							{"K", a.Cache.K[l], b.Cache.K[l]}, {"Kraw", a.Cache.Kraw[l], b.Cache.Kraw[l]}, {"V", a.Cache.V[l], b.Cache.V[l]},
						} {
							if len(pair.x) != total*cfg.NumKVHeads*cfg.HeadDim {
								t.Fatalf("%s layer %d cache shape", pair.name, l)
							}
							exact(fmt.Sprintf("chunk=%d layer=%d %s", chunk, l, pair.name), pair.x, pair.y)
						}
					}
					if !reflect.DeepEqual(a.Cache, b.Cache) {
						t.Fatalf("chunk %d full cache (positions/lineage included) differs", chunk)
					}
				}
				t.Log("engine=fak-native CPU exact hidden/logits/FULL KV parity; bases=0,9,12")
				// Both routes are warm above; model weights and lazy Q8 conversions are shared.
				// Each measured request owns a fresh session and executes the real native path.
				measure := func(reference bool) int64 {
					return testing.Benchmark(func(bb *testing.B) {
						for i := 0; i < bb.N; i++ {
							ss := m.NewSession()
							ss.Q4K = true
							if reference {
								ss.prefillQ4KFreshNormReference(chunks[0])
							} else {
								ss.prefillBatchedQ4K(chunks[0])
							}
							ss.Close()
						}
					}).AllocedBytesPerOp()
				}
				before, after := measure(true), measure(false)
				// Hoisting two panels removes 2*(layers-1) allocations; aliasing the panels
				// safely may save one more. Allow 10% for independently averaged runtime noise.
				expected := int64(2 * (cfg.NumLayers - 1) * len(chunks[0]) * cfg.HiddenSize * 4)
				minimum := expected * 9 / 10
				t.Logf("engine=fak-native CPU allocating=%d reused=%d B/op saving=%d required=%d", before, after, before-after, minimum)
				if before-after < minimum {
					t.Fatalf("missing request-local Xn/Xn2 allocation savings: got %d B/op, need >=%d", before-after, minimum)
				}
			})
		}
	}
}
