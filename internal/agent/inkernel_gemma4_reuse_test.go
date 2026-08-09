package agent

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_gemma4_reuse_test.go — the #5548 witness that gemma4 prefix reuse on the
// InKernelPlanner is not merely INERT.
//
// internal/model/gemma4_session.go documents the recompute bridge as safe because it
// holds no K/V rows: "Cache.Len() stays 0, and eviction / prefix reuse are INERT rather
// than silently wrong." Eviction is inert. Prefix reuse is not — the planner admits that
// EMPTY cache into the radix tree keyed by the whole prompt, a later prompt sharing a
// prefix matches on TOKEN IDS (not on KV depth), and the rebuilt session prefills only
// the divergent suffix against a history that carries none of the prefix.
//
// These arms drive the real reuse wiring (NewInKernelPlanner + the radix tree +
// generateReused), because a unit test on model.Session alone cannot see the defect: the
// Session is correct in isolation, and it is the planner that tells it the prefix is
// already resident. The oracle is the cacheless dedicated forward (Model.Forward), the
// same reference internal/ggufload/gemma4_session_test.go pins the bridge against.

// tinyGemma4Cfg is the synthetic twin of internal/ggufload's tinyGemma4GGUF: 4 layers,
// 3 sliding + 1 global, per-layer head_dim 8/16 and kv-heads 2/1, q/k norms, a sliding
// window of 2, and a proportional-rope global layer. The two attention regimes must BOTH
// be live for the witness to be honest — the sliding layers drop most of the prefix while
// the global layer keeps all of it, so a silently-truncated history is visible in the
// logits rather than washed out by a short window.
func tinyGemma4Cfg() model.Config {
	const (
		H       = 32
		I       = 64
		V       = 41
		nH      = 4
		hdSWA   = 8
		hdFull  = 16
		kvSWA   = 2
		kvFull  = 1
		nLayers = 4
	)
	return model.Config{
		ModelType:          "gemma4",
		HiddenSize:         H,
		IntermediateSize:   I,
		VocabSize:          V,
		NumLayers:          nLayers,
		NumHeads:           nH,
		HeadDim:            hdFull,
		NumKVHeads:         kvFull,
		RMSNormEps:         1e-6,
		RopeTheta:          1e6,
		BlockTopology:      model.SandwichNorm,
		ActGeluTanh:        true,
		QKNorm:             true,
		EmbedScale:         math.Sqrt(H),
		LogitSoftcap:       30,
		EOSTokenID:         -1,
		LayerTypes:         []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
		HeadDimPerLayer:    []int{hdSWA, hdSWA, hdSWA, hdFull},
		NumKVHeadsPerLayer: []int{kvSWA, kvSWA, kvSWA, kvFull},
		RopeDimPerLayer:    []int{hdSWA, hdSWA, hdSWA, hdFull},
		RopeThetaPerLayer:  []float64{10000, 10000, 10000, 1e6},
		Window:             []int{2, 2, 2, -1},
	}
}

// gemma4ReusePlanner builds the planner the CPU serve route builds: backend nil, no
// Metal, no Q4_K — precisely the one configuration model.Session.gemma4SessionModeWired
// accepts, and precisely the one inKernelPlannerPrefixReuseSupported waves through.
//
// radixDefault selects the shipped DEFAULT (FAK_INKERNEL_RADIX unset — reuse requested)
// versus the shipped A/B disable (=off). Deliberately it does NOT assert whether the
// planner then builds a tree: whether a reuse-requesting gemma4 planner ends up reusing is
// the thing under test, not a precondition of it. The A/B arm's refusal IS asserted,
// because "off" must always mean off.
func gemma4ReusePlanner(t *testing.T, m *model.Model, radixDefault bool) *InKernelPlanner {
	t.Helper()
	if radixDefault {
		t.Setenv("FAK_INKERNEL_RADIX", "")
	} else {
		t.Setenv("FAK_INKERNEL_RADIX", "off")
	}
	p := NewInKernelPlanner(m, nil, "synthetic-gemma4", false, nil, false)
	p.quant = false
	if !radixDefault && p.tree != nil {
		t.Fatal("FAK_INKERNEL_RADIX=off must disable the radix tree")
	}
	return p
}

func argmaxLogit(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// TestInKernelGemma4ReuseMatchesCachelessForward is the #5548 correctness witness. A
// second turn sharing a substantial prefix with the first must decode exactly what a
// full prefill decodes, and its first sampled token must be the argmax of the cacheless
// dedicated forward over the WHOLE prompt.
func TestInKernelGemma4ReuseMatchesCachelessForward(t *testing.T) {
	cfg := tinyGemma4Cfg()
	m := model.NewSyntheticGemma4(cfg)

	sys := synthIDs(cfg.VocabSize, 24, 5548) // the shared system/tool-schema prefix
	turn1 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 6, 55481)...)
	turn2 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 7, 55482)...)
	const maxNew = 4

	// Premise: the geometry is genuinely heterogeneous, so both attention regimes run.
	if cfg.HeadDimPerLayer[0] == cfg.HeadDimPerLayer[cfg.NumLayers-1] {
		t.Fatalf("premise broken: per-layer head_dim is uniform %v", cfg.HeadDimPerLayer)
	}

	// Premise: on THIS model a plain Session already reproduces the cacheless dedicated
	// forward bit-exactly, so any divergence below is the planner's reuse wiring and not
	// the gemma4 bridge itself.
	want := m.Forward(turn2).Logits[len(turn2)-1]
	if got := m.NewSession().Prefill(turn2); !eqF32(got, want) {
		t.Fatalf("premise broken: Session.Prefill is not bit-exact with Model.Forward for gemma4")
	}
	wantFirst := argmaxLogit(want)

	// What the buggy wiring actually computes: the SUFFIX alone, because the session it
	// was handed carries none of the shared prefix's history.
	suffixOnly := m.Forward(turn2[len(sys):]).Logits[len(turn2)-len(sys)-1]
	suffixFirst := argmaxLogit(suffixOnly)

	poff := gemma4ReusePlanner(t, m, false)
	decode(poff, turn1, maxNew)
	gotOFF, matchedOFF := decode(poff, turn2, maxNew)
	if matchedOFF != 0 {
		t.Fatalf("reuse-disabled planner reused %d tokens", matchedOFF)
	}
	if gotOFF[0] != wantFirst {
		t.Fatalf("full-prefill first token = %d, want argmax(Forward(turn2)) = %d", gotOFF[0], wantFirst)
	}

	pon := gemma4ReusePlanner(t, m, true)
	decode(pon, turn1, maxNew) // prime the tree with turn1's (empty) gemma4 snapshot
	gotON, matched := decode(pon, turn2, maxNew)

	if !eqInts(gotON, gotOFF) {
		t.Fatalf("gemma4 prefix reuse changed the decode (reuse is NOT inert):\n"+
			"  reuse ON  = %v (reused %d/%d tokens)\n"+
			"  reuse OFF = %v\n"+
			"  argmax(Forward(whole prompt))   = %d, logit %.6f\n"+
			"  argmax(Forward(suffix only))    = %d, logit %.6f\n"+
			"  the ON arm's first token is %d — it matches the SUFFIX-ONLY forward, i.e. the\n"+
			"  session was told %d prefix tokens were resident when it held none.",
			gotON, matched, len(turn2), gotOFF,
			wantFirst, want[wantFirst], suffixFirst, suffixOnly[suffixFirst],
			gotON[0], matched)
	}
	t.Logf("gemma4 reuse parity: %d/%d tokens reported reused, decode identical to full prefill (%v)", matched, len(turn2), gotON)
}

// TestInKernelGemma4ExactReplayMatchesCachelessForward is the second half of the hazard.
// An exact-duplicate prompt hits the cached prompt-final logits, so prefill is skipped
// ENTIRELY (inkernel_decode.go step 2) — and every subsequent Step then recomputes over a
// history containing only the generated tokens. The first sampled token is right by
// accident; the rest are not.
func TestInKernelGemma4ExactReplayMatchesCachelessForward(t *testing.T) {
	cfg := tinyGemma4Cfg()
	m := model.NewSyntheticGemma4(cfg)
	ids := synthIDs(cfg.VocabSize, 20, 55483)
	const maxNew = 4

	poff := gemma4ReusePlanner(t, m, false)
	decode(poff, ids, maxNew)
	gotOFF, _ := decode(poff, ids, maxNew)

	pon := gemma4ReusePlanner(t, m, true)
	gotPrime, primeMatched := decode(pon, ids, maxNew)
	if primeMatched != 0 {
		t.Fatalf("first turn reused %d tokens", primeMatched)
	}
	gotReplay, matched := decode(pon, ids, maxNew)

	if !eqInts(gotReplay, gotOFF) || !eqInts(gotPrime, gotOFF) {
		t.Fatalf("gemma4 exact replay diverged from full prefill:\n"+
			"  first turn (reuse ON) = %v\n"+
			"  exact replay          = %v (reused %d/%d)\n"+
			"  full prefill          = %v",
			gotPrime, gotReplay, matched, len(ids), gotOFF)
	}
	t.Logf("gemma4 exact replay parity: %d/%d reported reused, decode identical to full prefill (%v)", matched, len(ids), gotReplay)
}

// TestInKernelGemma4PrefixReuseIsRefused is the fix's own contract: the capability gate
// must exclude gemma4 on the host path, and it must stay narrow — every other host model
// keeps reuse. This is what makes gemma4_session.go's "prefix reuse is INERT" sentence
// true rather than aspirational.
func TestInKernelGemma4PrefixReuseIsRefused(t *testing.T) {
	gemma4 := model.NewSyntheticGemma4(tinyGemma4Cfg())
	if inKernelPlannerPrefixReuseSupported(gemma4, nil) {
		t.Error("gemma4 must not report host prefix reuse as supported: the recompute bridge holds no K/V rows, so an admitted prefix is empty and a partial hit drops the prefix from the recompute")
	}
	if !inKernelPlannerPrefixReuseSupported(model.NewSynthetic(tinyCfg()), nil) {
		t.Error("the gemma4 refusal must not disable host prefix reuse for every other architecture")
	}

	t.Setenv("FAK_INKERNEL_RADIX", "")
	p := NewInKernelPlanner(gemma4, nil, "synthetic-gemma4", false, nil, false)
	if p.tree != nil {
		t.Error("a gemma4 planner must not build a radix tree it can never serve a correct hit from")
	}
	if got := p.kvPrefixEligiblePromptTokens(32); got != 0 {
		t.Errorf("gemma4 reports %d cacheable prompt tokens, want 0 — a recompute session realizes no prefill saving", got)
	}
}

func eqF32(a, b []float32) bool {
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
