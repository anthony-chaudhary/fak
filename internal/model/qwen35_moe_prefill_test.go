package model

import (
	"math"
	"strings"
	"testing"
)

// qwen35MoEHybridTestCfg is the qwen35HybridTestCfg Gated-DeltaNet hybrid with a routed
// MoE FFN plus the gated shared expert (qwen3_5_moe, e.g. Qwen3.6-35B-A3B), shrunk.
func qwen35MoEHybridTestCfg() Config {
	cfg := qwen35HybridTestCfg()
	cfg.ModelType = "qwen3_5_moe_text"
	cfg.NumExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.NormTopKProb = true
	cfg.IntermediateSize = 32
	cfg.SharedIntermediateSize = 64
	return cfg
}

// newSyntheticQwen35MoEHybrid builds NewSynthetic's hybrid layout with every dense MLP
// replaced by a router, NumExperts routed SwiGLU experts and the singular qwen3_5_moe
// shared expert + sigmoid gate, so moeFFN's routed sum and qwen35SharedExpert both run.
func newSyntheticQwen35MoEHybrid(cfg Config) *Model {
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V, E, SI := cfg.HiddenSize, cfg.expertIntermediate(), cfg.VocabSize, cfg.NumExperts, cfg.SharedIntermediateSize
	type ts = synthTensor
	tensors := []ts{{"model.embed_tokens.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors, ts{p + "input_layernorm.weight", []int{H}})
		if cfg.isLinearAttnLayer(l) {
			_, nV, _, vHd, _, valDim, convDim := cfg.linearAttnDims()
			K := cfg.LinearConvKernelDim
			tensors = append(tensors,
				ts{p + "linear_attn.in_proj_qkv.weight", []int{convDim, H}},
				ts{p + "linear_attn.in_proj_z.weight", []int{valDim, H}},
				ts{p + "linear_attn.in_proj_b.weight", []int{nV, H}},
				ts{p + "linear_attn.in_proj_a.weight", []int{nV, H}},
				ts{p + "linear_attn.conv1d.weight", []int{convDim * K}},
				ts{p + "linear_attn.A_log", []int{nV}},
				ts{p + "linear_attn.dt_bias", []int{nV}},
				ts{p + "linear_attn.norm.weight", []int{vHd}},
				ts{p + "linear_attn.out_proj.weight", []int{H, valDim}},
			)
		} else {
			tensors = append(tensors,
				ts{p + "self_attn.q_proj.weight", []int{2 * nH * hd, H}},
				ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
				ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
				ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			)
		}
		tensors = append(tensors,
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{routerName(l), []int{E, H}},
			ts{qwen35SharedExpertName(l, "gate_proj.weight"), []int{SI, H}},
			ts{qwen35SharedExpertName(l, "up_proj.weight"), []int{SI, H}},
			ts{qwen35SharedExpertName(l, "down_proj.weight"), []int{H, SI}},
			ts{qwen35SharedExpertName(l, "gate.weight"), []int{1, H}},
		)
		for e := 0; e < E; e++ {
			tensors = append(tensors,
				ts{expertName(l, e, "gate_proj.weight"), []int{I, H}},
				ts{expertName(l, e, "up_proj.weight"), []int{I, H}},
				ts{expertName(l, e, "down_proj.weight"), []int{H, I}},
			)
		}
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})
	man, raw := synthBuildRawResolved(tensors, func(name string) synthElemFill {
		if strings.HasSuffix(name, "layernorm.weight") || strings.HasSuffix(name, "linear_attn.norm.weight") || name == "model.norm.weight" {
			return synthConstFill(1.0)
		}
		return synthMatmulFillFor(name)
	})
	cfg.TieWordEmbeddings = true
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

func newQuantQwen35MoEHybrid(t *testing.T) *Model {
	t.Helper()
	m := newSyntheticQwen35MoEHybrid(qwen35MoEHybridTestCfg())
	m.Quantize()
	if _, ok := m.ffnForLayer(0).(moeFFN); !ok {
		t.Fatalf("premise broken: layer 0 FFN is %T, want moeFFN", m.ffnForLayer(0))
	}
	if !m.has(qwen35SharedExpertName(0, "gate.weight")) {
		t.Fatal("premise broken: synthetic qwen3_5_moe has no shared expert gate")
	}
	return m
}

var qwen35MoEPromptA = []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71}
var qwen35MoEPromptB = []int{2, 13, 89, 41, 8, 77, 30, 64, 9, 15, 50, 21, 33, 44, 6, 90}

// TestQwen35MoEHybridQuantBatchedPrefillMatchesTokenLoop mirrors
// TestQwen35HybridQuantBatchedPrefillMatchesTokenLoop for qwen3_5_moe: the batched Q8
// hybrid lane (attention/linear-attention batched, routed+shared FFN per row) matches the
// per-token tokenHiddenQ loop across a fresh prefill and a second prefill appended at a
// non-zero base (restored conv/recurrent state). The tolerance is the established Q8
// panel-vs-GEMV drift of the dense hybrid's own gate; the FFN sublayer itself is pinned
// bit-for-bit by TestQwen35MoERowFFNQMatchesTokenLoopSublayer. [SW-VERIFIED, synthetic]
// fak-test:runtime fast est=200ms lane=default
func TestQwen35MoEHybridQuantBatchedPrefillMatchesTokenLoop(t *testing.T) {
	m := newQuantQwen35MoEHybrid(t)
	cfg := m.Cfg
	if !q8Qwen35HybridPrefillOK(cfg, len(qwen35MoEPromptA)) {
		t.Fatal("q8Qwen35HybridPrefillOK refused the qwen3_5_moe hybrid")
	}
	if !q4kQwen35HybridPrefillOK(cfg, len(qwen35MoEPromptA)) {
		t.Fatal("q4kQwen35HybridPrefillOK refused the qwen3_5_moe hybrid")
	}
	if !metalQwen35HybridPrefillOK(cfg, len(qwen35MoEPromptA)) {
		t.Fatal("metalQwen35HybridPrefillOK refused the qwen3_5_moe hybrid")
	}

	ref := m.NewSession()
	ref.Quant = true
	var refHidden []float32
	for _, id := range qwen35MoEPromptA {
		refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	wantA := ref.headQ(refHidden)
	for _, id := range qwen35MoEPromptB {
		refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	wantB := ref.headQ(refHidden)

	got := m.NewSession()
	got.Quant = true
	if !got.q8Qwen35HybridPrefillSessionOK(len(qwen35MoEPromptA)) {
		t.Fatal("q8Qwen35HybridPrefillSessionOK refused a plain qwen3_5_moe session")
	}
	gotA := got.Prefill(qwen35MoEPromptA)
	gotB := got.Prefill(qwen35MoEPromptB)

	assertQuantLogitsClose(t, "qwen3_5_moe q8 batched prefill logits", wantA, gotA)
	assertQuantLogitsClose(t, "qwen3_5_moe q8 batched appended prefill logits", wantB, gotB)
	assertKVCacheQuantClose(t, "qwen3_5_moe q8 batched prefill", ref.Cache, got.Cache)
	assertLinearAttnCacheQuantClose(t, "qwen3_5_moe q8 batched prefill", ref.Cache.linear, got.Cache.linear)
}

// TestQwen35MoERowFFNQMatchesTokenLoopSublayer pins the per-row FFN bit-for-bit: for P
// distinct residual rows, prefillQwen35RowFFNQ must equal an independently assembled token
// loop PreNorm sublayer (post_attention_layernorm normCfg, moeFFN routed sum + gated shared
// expert on the q8Kernel tokenHiddenQ selects for a Q8 MoE session). Rows route to
// different experts, so feeding one row's normed input to every row's FFN fails here.
// fak-test:runtime fast est=50ms lane=default
func TestQwen35MoERowFFNQMatchesTokenLoopSublayer(t *testing.T) {
	m := newQuantQwen35MoEHybrid(t)
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	embed := m.embedRows()
	P := len(qwen35MoEPromptA)
	mat := q8Kernel{m}
	for l := 0; l < cfg.NumLayers; l++ {
		X := make([]float32, P*H)
		for r, id := range qwen35MoEPromptA {
			copy(X[r*H:(r+1)*H], embed[id*H:(id+1)*H])
		}
		want := append([]float32(nil), X...)
		distinct := false
		var firstPick []int
		for r := 0; r < P; r++ {
			row := want[r*H : (r+1)*H]
			xn := normCfg(row, m.tensor(layerName(l, "post_attention_layernorm.weight")), nil, eps, cfg)
			picks := route(m, l, mat.prep(xn), mat)
			ex := make([]int, len(picks))
			for i, pk := range picks {
				ex[i] = pk.expert
			}
			if r == 0 {
				firstPick = ex
			} else if !intsEqual(ex, firstPick) {
				distinct = true
			}
			delta := moeFFN{}.apply(m, l, mat.prep(xn), mat)
			for i := range row {
				row[i] += delta[i]
			}
		}
		if !distinct {
			t.Fatalf("layer %d: every row routes to the same experts; the parity check is vacuous", l)
		}
		s := m.NewSession()
		s.Quant = true
		s.prefillQwen35RowFFNQ(l, X, P)
		if !float32BitsEqual(want, X) {
			t.Fatalf("layer %d: per-row MoE FFN differs from the token loop's sublayer", l)
		}
	}
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestQwen35MoEHybridQuantPrefillTakesBatchedLane proves Prefill routes qwen3_5_moe to the
// batched hybrid lane: the batched linear recurrence and per-row FFN phases are recorded,
// and blockStep's mlp_decode (once per token x layer on the token loop) never runs.
// fak-test:runtime fast est=100ms lane=default
func TestQwen35MoEHybridQuantPrefillTakesBatchedLane(t *testing.T) {
	m := newQuantQwen35MoEHybrid(t)
	s := m.NewSession()
	s.Quant = true
	pp := NewPhaseProfiler()
	s.PhaseProfiler = pp
	s.Prefill(qwen35MoEPromptA)
	prof := pp.Snapshot("prefill", len(qwen35MoEPromptA), 0, 1)
	if n := phaseCalls(prof, "mlp_decode"); n != 0 {
		t.Fatalf("qwen3_5_moe Prefill ran blockStep's FFN %d times, want the batched hybrid lane", n)
	}
	if n := phaseCalls(prof, "qwen35_moe_row_ffn"); n != m.Cfg.NumLayers {
		t.Fatalf("qwen35_moe_row_ffn calls = %d, want one per layer (%d)", n, m.Cfg.NumLayers)
	}
	for _, phase := range []string{"q8_panel_quantize", "qwen35_linear_recurrent", "qwen35_full_attn"} {
		if phaseCalls(prof, phase) == 0 {
			t.Fatalf("batched hybrid phase %q not recorded: %#v", phase, prof.Phases)
		}
	}
}

// TestQwen35MoEHybridObserversKeepTokenLoop: a hidden-state tap or an enabled residual hook
// is invoked only inside blockStep, so a qwen3_5_moe session carrying one keeps the token
// loop (mlp_decode once per token x layer) and the hook still fires per token x layer.
// fak-test:runtime fast est=100ms lane=default
func TestQwen35MoEHybridObserversKeepTokenLoop(t *testing.T) {
	prompt := qwen35MoEPromptA
	t.Run("tap", func(t *testing.T) {
		m := newQuantQwen35MoEHybrid(t)
		s := m.NewSession()
		s.Quant = true
		s.tap = &hiddenTap{dir: t.TempDir(), pos: 999} // armed, never reached
		if s.q8Qwen35HybridPrefillSessionOK(len(prompt)) {
			t.Fatal("tapped qwen3_5_moe session admitted to the batched lane")
		}
		pp := NewPhaseProfiler()
		s.PhaseProfiler = pp
		s.Prefill(prompt)
		prof := pp.Snapshot("prefill", len(prompt), 0, 1)
		if n, want := phaseCalls(prof, "mlp_decode"), len(prompt)*m.Cfg.NumLayers; n != want {
			t.Fatalf("tapped session mlp_decode = %d, want token loop %d", n, want)
		}
	})
	t.Run("residual-hook", func(t *testing.T) {
		cfg := qwen35MoEHybridTestCfg()
		cfg.EnableResidualHook = true
		m := newSyntheticQwen35MoEHybrid(cfg)
		m.Quantize()
		calls := 0
		m.SetResidualHook(func(int, []float32) { calls++ })
		s := m.NewSession()
		s.Quant = true
		if s.q8Qwen35HybridPrefillSessionOK(len(prompt)) {
			t.Fatal("residual-hook qwen3_5_moe session admitted to the batched lane")
		}
		s.Prefill(prompt)
		if want := len(prompt) * cfg.NumLayers; calls != want {
			t.Fatalf("residual hook fired %d times, want %d (session left the token loop)", calls, want)
		}
	})
}

// newQ4KQwen35MoEHybrid is the qwen3_5_moe hybrid at qwen35HybridQ4KTestCfg's Q4_K-legal
// geometry (every reduction dim a multiple of 256). Full-attention v/o and every routed
// and shared expert gate/up are resident Q4_K; q/k, linear_attn.*, the router and every
// down projection resolve through sessionQ4KKernel's Q8 minority, as on a q4_k_m GGUF.
func newQ4KQwen35MoEHybrid(t *testing.T, mutate func(*Config)) *Model {
	t.Helper()
	cfg := qwen35HybridQ4KTestCfg()
	cfg.ModelType = "qwen3_5_moe_text"
	cfg.NumExperts = 4
	cfg.NumExpertsPerTok = 2
	cfg.NormTopKProb = true
	cfg.SharedIntermediateSize = 256
	if mutate != nil {
		mutate(&cfg)
	}
	m := newSyntheticQwen35MoEHybrid(cfg)
	m.Quantize()
	var projs [][2]any
	nKVhd := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		if !cfg.isLinearAttnLayer(l) {
			projs = append(projs,
				[2]any{p + "self_attn.v_proj.weight", nKVhd},
				[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			)
		}
		projs = append(projs,
			[2]any{qwen35SharedExpertName(l, "gate_proj.weight"), cfg.SharedIntermediateSize},
			[2]any{qwen35SharedExpertName(l, "up_proj.weight"), cfg.SharedIntermediateSize},
		)
		for e := 0; e < cfg.NumExperts; e++ {
			projs = append(projs,
				[2]any{expertName(l, e, "gate_proj.weight"), cfg.expertIntermediate()},
				[2]any{expertName(l, e, "up_proj.weight"), cfg.expertIntermediate()},
			)
		}
	}
	fillQ4KW(t, m, projs, 11)
	return m
}

// TestQwen35MoEHybridQ4KBatchedPrefillMatchesTokenLoop checks the mixed resident-Q4_K
// hybrid across fresh and appended prefills. The strict leg matches the Q8 minority's
// scalar decode and legacy prefill reductions; legacy GEMM alone does not select that
// decode reduction. Native Q8 minority legs retain the existing 0.9999 cosine contract.
// Q4_K activation quantization is disabled in every leg to isolate Q8 reduction drift;
// these are not full configured-Q4_K decode qualification. [SW-VERIFIED, synthetic]
// fak-test:runtime fast est=3s lane=default
func TestQwen35MoEHybridQ4KBatchedPrefillMatchesTokenLoop(t *testing.T) {
	oldSDOT := q4kSDOTForce
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { q4kSDOTForce = oldSDOT })
	first := qwen35MoEPromptA
	second := []int{2, 13, 89, 41, 8, 77, 30, 64, 9, 15, 50, 21, 33, 44, 6, 90, 101, 255}

	runQ8MoEParityModes(t, func(t *testing.T, strict bool) {
		m := newQ4KQwen35MoEHybrid(t, nil)
		ref, got := m.NewSession(), m.NewSession()
		ref.Q4K, got.Q4K = true, true
		if _, ok := ref.tokenHiddenQKernel().(sessionQ4KKernel); !ok {
			t.Fatal("premise broken: the Q4_K MoE token loop does not run sessionQ4KKernel")
		}
		got.PhaseProfiler = NewPhaseProfiler()
		// Synthetic-fixture ceilings, not analytical roundoff or model-quality bounds.
		// Across native AVX-512 legacy and AVX2 tiled reductions, measured scaled
		// maxima are 0.03438 logits, 0.01918 K and 0.04095 V. Recurrent-state
		// max absolute error is 6.834e-5; conv stays below the original 1e-5 bar.
		// Cosine and greedy-token agreement remain separate gates.
		const nativeLogitScaledTol, nativeKScaledTol, nativeVScaledTol = 0.04, 0.02, 0.05
		const nativeRecurrentAbsTol = 1e-4
		check := func(label string, want, have []float32, nativeScaledTol, nativeAbsTol float64) {
			t.Helper()
			if len(want) != len(have) {
				t.Fatalf("%s length got %d want %d", label, len(have), len(want))
			}
			if len(want) == 0 {
				return
			}
			var maxAbs, maxScaled, sumError, sumRef float64
			for i, w := range want {
				g := have[i]
				if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) || math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
					t.Fatalf("%s[%d] is not finite: got %v want %v", label, i, g, w)
				}
				diff := math.Abs(float64(g) - float64(w))
				maxAbs = math.Max(maxAbs, diff)
				maxScaled = math.Max(maxScaled, diff/math.Max(1, math.Abs(float64(w))))
				sumError += diff * diff
				sumRef += float64(w) * float64(w)
			}
			c := cosine(want, have)
			t.Logf("%s max-abs=%.3e max-scaled=%.3e error-RMS=%.3e reference-RMS=%.3e cosine=%.9f", label, maxAbs, maxScaled,
				math.Sqrt(sumError/float64(len(want))), math.Sqrt(sumRef/float64(len(want))), c)
			if !strict && nativeScaledTol > 0 && maxScaled > nativeScaledTol {
				t.Errorf("%s scaled error %.3e > %.3e", label, maxScaled, nativeScaledTol)
			}
			if !strict && nativeAbsTol > 0 && maxAbs > nativeAbsTol {
				t.Errorf("%s absolute error %.3e > %.3e", label, maxAbs, nativeAbsTol)
			}
			if !strict && nativeScaledTol > 0 && maxAbs != 0 && !(c >= 0.9999) {
				t.Errorf("%s cosine %.9f < 0.9999", label, c)
			}
		}
		checkState := func(label string) {
			t.Helper()
			if got.Cache.Len() != ref.Cache.Len() {
				t.Fatalf("%s cache length got %d want %d", label, got.Cache.Len(), ref.Cache.Len())
			}
			if !intsEqual(ref.Cache.pos, got.Cache.pos) {
				t.Fatalf("%s cache positions differ", label)
			}
			for l := 0; l < m.Cfg.NumLayers; l++ {
				if m.Cfg.isLinearAttnLayer(l) {
					continue
				}
				for name, pair := range map[string][2][]float32{
					"K": {ref.Cache.K[l], got.Cache.K[l]}, "Kraw": {ref.Cache.Kraw[l], got.Cache.Kraw[l]}, "V": {ref.Cache.V[l], got.Cache.V[l]},
				} {
					tol := nativeKScaledTol
					if name == "V" {
						tol = nativeVScaledTol
					}
					check(label+" layer "+itoa(l)+" "+name, pair[0], pair[1], tol, 0)
				}
			}
			wl, gl := ref.Cache.linear, got.Cache.linear
			if wl == nil || gl == nil || len(wl.layers) != len(gl.layers) {
				t.Fatal("linear cache shape mismatch")
			}
			for l, w := range wl.layers {
				g := gl.layers[l]
				if len(w.recurrent) != len(g.recurrent) || len(w.conv) != len(g.conv) {
					t.Fatalf("linear layer %d state shape mismatch", l)
				}
				for h := range w.recurrent {
					check(label+" layer "+itoa(l)+" recurrent "+itoa(h), w.recurrent[h], g.recurrent[h], 0, nativeRecurrentAbsTol)
				}
				for row := range w.conv {
					check(label+" layer "+itoa(l)+" conv "+itoa(row), w.conv[row], g.conv[row], 0, 1e-5)
				}
			}
			if strict {
				assertKVCacheQuantCloseTol(t, label, ref.Cache, got.Cache, prefillQ4KKTol(), prefillQ4KVTol())
				assertLinearAttnCacheQuantClose(t, label, ref.Cache.linear, got.Cache.linear)
			}
		}
		for i, prompt := range [][]int{first, second} {
			var h []float32
			for _, id := range prompt {
				h = ref.tokenHiddenQ(id, ref.Cache.Len())
			}
			want, have := ref.headResident(h), got.Prefill(prompt)
			label := "prefill " + itoa(i)
			check(label+" logits", want, have, nativeLogitScaledTol, 0)
			if len(want) == 0 || argmax(want) != argmax(have) {
				t.Errorf("%s logits have no matching greedy token", label)
			}
			if strict {
				assertQuantLogitsClose(t, label, want, have)
			}
			checkState(label)
		}
		prof := got.PhaseProfiler.Snapshot("prefill", len(first)+len(second), 0, 1)
		if n := phaseCalls(prof, "qwen35_moe_row_ffn"); n != 2*m.Cfg.NumLayers {
			t.Fatalf("qwen35_moe_row_ffn calls = %d, want %d: the batched Q4_K hybrid lane was not taken", n, 2*m.Cfg.NumLayers)
		}
	})
}

// TestQwen35MoEHybridQ4KPrefillTakesBatchedLane proves resident-Q4_K Prefill and
// PrefillNoLogits route qwen3_5_moe to the batched hybrid lane (no blockStep FFN, one
// per-row FFN pass per layer), and that tap / residual-hook sessions keep the token loop.
// fak-test:runtime fast est=1s lane=default
func TestQwen35MoEHybridQ4KPrefillTakesBatchedLane(t *testing.T) {
	prompt := qwen35MoEPromptA
	m := newQ4KQwen35MoEHybrid(t, nil)
	for _, noLogits := range []bool{false, true} {
		s := m.NewSession()
		s.Q4K = true
		pp := NewPhaseProfiler()
		s.PhaseProfiler = pp
		if noLogits {
			s.PrefillNoLogits(prompt)
		} else {
			s.Prefill(prompt)
		}
		prof := pp.Snapshot("prefill", len(prompt), 0, 1)
		if n := phaseCalls(prof, "mlp_decode"); n != 0 {
			t.Fatalf("noLogits=%v: Q4_K qwen3_5_moe Prefill ran blockStep's FFN %d times", noLogits, n)
		}
		if n := phaseCalls(prof, "qwen35_moe_row_ffn"); n != m.Cfg.NumLayers {
			t.Fatalf("noLogits=%v: qwen35_moe_row_ffn calls = %d, want %d", noLogits, n, m.Cfg.NumLayers)
		}
	}

	t.Run("tap", func(t *testing.T) {
		s := m.NewSession()
		s.Q4K = true
		s.tap = &hiddenTap{dir: t.TempDir(), pos: 999} // armed, never reached
		pp := NewPhaseProfiler()
		s.PhaseProfiler = pp
		s.Prefill(prompt)
		prof := pp.Snapshot("prefill", len(prompt), 0, 1)
		if n, want := phaseCalls(prof, "mlp_decode"), len(prompt)*m.Cfg.NumLayers; n != want {
			t.Fatalf("tapped Q4_K session mlp_decode = %d, want token loop %d", n, want)
		}
	})
	t.Run("residual-hook", func(t *testing.T) {
		hm := newQ4KQwen35MoEHybrid(t, func(c *Config) { c.EnableResidualHook = true })
		calls := 0
		hm.SetResidualHook(func(int, []float32) { calls++ })
		s := hm.NewSession()
		s.Q4K = true
		s.Prefill(prompt)
		if want := len(prompt) * hm.Cfg.NumLayers; calls != want {
			t.Fatalf("residual hook fired %d times, want %d (Q4_K session left the token loop)", calls, want)
		}
	})
}

// TestQwen35MoEHybridViaMMMatchesTokenLoop is the host-independent witness for the Metal
// hybrid twin's qwen3_5_moe arm: drive the backend-agnostic core prefillQwen35HybridViaMM
// with the CPU Q8 mm TestQwen35HybridViaMMMatchesCPUTemplate uses, and require logits, KV and
// linear state to match both the per-token tokenHiddenQ loop and the proven batched Q8 hybrid
// (prefillQwen35HybridQHidden), with the routed FFN taken per row once per layer. What this
// cannot witness off-device is the GPU f16 GEMM and the darwin wiring around the core.
// [SW-VERIFIED, synthetic; Metal runtime not exercised]
// fak-test:runtime fast est=300ms lane=default
func TestQwen35MoEHybridViaMMMatchesTokenLoop(t *testing.T) {
	m := newQuantQwen35MoEHybrid(t)
	prompt := qwen35MoEPromptA
	P := len(prompt)

	loop := m.NewSession()
	loop.Quant = true
	var h []float32
	for _, id := range prompt {
		h = loop.tokenHiddenQ(id, loop.Cache.Len())
	}
	wantLoop := loop.headQ(h)

	tmpl := m.NewSession()
	tmpl.Quant = true
	wantTmpl := tmpl.headQ(tmpl.prefillQwen35HybridQHidden(prompt))

	got := m.NewSession()
	got.Quant = true
	if got.qwen35MetalHybridMoEKeepsTokenLoop() {
		t.Fatal("a plain qwen3_5_moe session must not be held on the token loop")
	}
	pp := NewPhaseProfiler()
	got.PhaseProfiler = pp
	cpuMM := func(name string, X []float32, out int) []float32 {
		var panel q8Panel
		quantizeBatchPanelInto(&panel, X, P, len(X)/P)
		Y := make([]float32, P*out)
		qGemm8Into(got.M.q8(name), &panel, Y)
		return Y
	}
	gotLogits := got.headQ(got.prefillQwen35HybridViaMM(prompt, cpuMM))
	prof := pp.Snapshot("prefill", P, 0, 1)
	if n := phaseCalls(prof, "qwen35_moe_row_ffn"); n != m.Cfg.NumLayers {
		t.Fatalf("via-mm core qwen35_moe_row_ffn calls = %d, want %d", n, m.Cfg.NumLayers)
	}
	if n := phaseCalls(prof, "mlp_decode"); n != 0 {
		t.Fatalf("via-mm core ran blockStep's FFN %d times", n)
	}

	assertQuantLogitsClose(t, "qwen3_5_moe via-mm core vs token loop logits", wantLoop, gotLogits)
	assertKVCacheQuantClose(t, "qwen3_5_moe via-mm core vs token loop", loop.Cache, got.Cache)
	assertLinearAttnCacheQuantClose(t, "qwen3_5_moe via-mm core vs token loop", loop.Cache.linear, got.Cache.linear)
	assertQuantLogitsClose(t, "qwen3_5_moe via-mm core vs Q8 template logits", wantTmpl, gotLogits)
	assertKVCacheQuantClose(t, "qwen3_5_moe via-mm core vs Q8 template", tmpl.Cache, got.Cache)
}

// TestQwen35MetalHybridMoEKeepsTokenLoop pins the Metal route's MoE session half: the
// cfg-only metalQwen35HybridPrefillOK admits qwen3_5_moe, so a session carrying a tap or a
// residual hook must be held on the token loop by qwen35MetalHybridMoEKeepsTokenLoop, while
// the dense hybrid's Metal routing is never affected by it.
// fak-test:runtime fast est=50ms lane=default
func TestQwen35MetalHybridMoEKeepsTokenLoop(t *testing.T) {
	m := newQuantQwen35MoEHybrid(t)
	tapped := m.NewSession()
	tapped.tap = &hiddenTap{dir: t.TempDir(), pos: 999}
	if !tapped.qwen35MetalHybridMoEKeepsTokenLoop() {
		t.Fatal("tapped qwen3_5_moe Metal session was not held on the token loop")
	}
	hookCfg := qwen35MoEHybridTestCfg()
	hookCfg.EnableResidualHook = true
	hm := newSyntheticQwen35MoEHybrid(hookCfg)
	if !hm.NewSession().qwen35MetalHybridMoEKeepsTokenLoop() {
		t.Fatal("residual-hook qwen3_5_moe Metal session was not held on the token loop")
	}
	dense := NewSynthetic(qwen35HybridTestCfg())
	ds := dense.NewSession()
	ds.tap = &hiddenTap{dir: t.TempDir(), pos: 999}
	if ds.qwen35MetalHybridMoEKeepsTokenLoop() {
		t.Fatal("the dense hybrid's Metal routing must not change")
	}
}
