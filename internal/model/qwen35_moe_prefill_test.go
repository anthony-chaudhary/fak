package model

import (
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
	if q4kQwen35HybridPrefillOK(cfg, len(qwen35MoEPromptA)) || metalQwen35HybridPrefillOK(cfg, len(qwen35MoEPromptA)) {
		t.Fatal("resident-Q4_K / Metal hybrid lanes have no per-row FFN seam and must still refuse MoE")
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
