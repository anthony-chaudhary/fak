package main

import (
	"flag"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default
func TestGemma4CachedDiagnosticSelection(t *testing.T) {
	fs := flag.NewFlagSet("diagtok", flag.ContinueOnError)
	f := bindGemma4CachedDiagnosticFlags(fs)
	if err := f.configure(nil, false, 0, 0); err != nil || f.enabled || f.stateBytes != 0 {
		t.Fatal("default-off route changed")
	}
	if err := fs.Parse([]string{"--gemma4-cached-qualification", "--gemma4-state-bytes=4096"}); err != nil {
		t.Fatal(err)
	}
	cfg := model.Config{ModelType: "gemma4", HiddenSize: 16, IntermediateSize: 32, VocabSize: 16,
		NumLayers: 2, NumHeads: 2, HeadDim: 8, NumKVHeads: 1, RMSNormEps: 1e-6, RopeTheta: 10000,
		BlockTopology: model.SandwichNorm, ActGeluTanh: true, QKNorm: true, EOSTokenID: -1,
		LayerTypes: []string{"sliding_attention", "full_attention"}, HeadDimPerLayer: []int{4, 8},
		NumKVHeadsPerLayer: []int{2, 1}, RopeDimPerLayer: []int{4, 8}, Window: []int{2, -1}}
	m := model.NewSyntheticGemma4(cfg)
	s := m.NewSession()
	defer s.Close()
	if err := f.configure(s, true, 3, 2); err == nil {
		t.Fatal("BOS-altering reference admitted")
	}
	if err := f.configure(s, false, 3, -1); err == nil {
		t.Fatal("negative decode bound admitted")
	}
	if err := f.configure(s, false, 3, 2); err != nil {
		t.Fatal(err)
	}
	s.Prefill([]int{1, 2, 3})
	s.Step(4)
	s.Step(5)
	ob := s.Gemma4CachedContinuationObservation()
	if !ob.Selected || ob.Tokens != 5 || ob.ComputedTokens != 5 || ob.Failed || ob.StateBytes > 4096 {
		t.Fatalf("real selected Session: %+v", ob)
	}
}

var gemma4CachedGGUF = flag.String("gemma4-cached-gguf", "", "existing exact Gemma4 GGUF for qualification; never downloaded")
var gemma4CachedGGUFLoad = flag.String("gemma4-cached-gguf-load", "q4k", "same-artifact residency for both sessions: f32 | q4k | q8")
var gemma4CachedGGUFRequired = flag.Bool("gemma4-cached-gguf-required", false, "fail rather than skip when the exact GGUF witness is not bound")

// This required-mode witness uses the REAL existing GGUF loader and the CLI's
// typed selection. Same model/store and fixed teacher-forced IDs isolate cached
// continuation from quantization, tokenization and differing BOS policies.
// It is not a whole-model quality, generated-text or performance acceptance.
// fak-test:runtime integration est=120s lane=optin
func TestGemma4CachedGGUFQualification(t *testing.T) {
	if *gemma4CachedGGUF == "" {
		if *gemma4CachedGGUFRequired {
			t.Fatal("required Gemma4 GGUF binding is absent")
		}
		t.Skip("opt-in: bind an existing exact Gemma4 GGUF")
	}
	var m *model.Model
	var err error
	switch *gemma4CachedGGUFLoad {
	case "f32":
		m, err = ggufload.LoadModel(*gemma4CachedGGUF)
	case "q4k":
		m, err = ggufload.LoadModelQ4K(*gemma4CachedGGUF)
	case "q8":
		m, err = ggufload.LoadModelQuant(*gemma4CachedGGUF)
	default:
		t.Fatal("unknown GGUF residency")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if m.Cfg.VocabSize < 9 {
		t.Fatal("qualification requires token IDs 1..8")
	}
	base, candidate := m.NewSession(), m.NewSession()
	defer base.Close()
	defer candidate.Close()
	f := &gemma4CachedDiagnosticFlags{enabled: true, stateBytes: 64 << 20}
	if err := f.configure(candidate, false, 5, 3); err != nil {
		t.Fatal(err)
	}
	suppressed := make(map[int]bool, len(m.Cfg.SuppressTokens))
	for _, id := range m.Cfg.SuppressTokens {
		suppressed[id] = true
	}
	check := func(got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatal("logit length mismatch")
		}
		for i := range want {
			// SuppressTokens deliberately publishes -Inf at these exact IDs.
			// Every other candidate AND reference value must be finite.
			if suppressed[i] {
				if !math.IsInf(float64(got[i]), -1) || !math.IsInf(float64(want[i]), -1) {
					t.Fatalf("declared suppressed logit %d must be -Inf in both routes", i)
				}
			} else if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
				t.Fatalf("undeclared non-finite logit %d", i)
			}
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("same-artifact logit %d: %08x, want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	ids := []int{1, 2, 3, 4, 5}
	check(candidate.Prefill(ids), base.Prefill(ids))
	for _, id := range []int{6, 7, 8} {
		check(candidate.Step(id), base.Step(id))
	}
	ob := candidate.Gemma4CachedContinuationObservation()
	if !ob.Selected || ob.ComputedTokens != 8 || ob.Tokens != 8 || ob.Failed || len(candidate.Prefill(nil)) != 0 {
		t.Fatalf("selected continuation not witnessed: %+v", ob)
	}
	if base.Gemma4CachedContinuationObservation().Selected {
		t.Fatal("reference did not stay full-history recompute")
	}
	t.Logf("engine=fak-native source-bound-required-witness residency=%s fixed_ids=1..8 selected=%+v; model/hash/build/resource pins belong in pickup receipt", *gemma4CachedGGUFLoad, ob)
}
