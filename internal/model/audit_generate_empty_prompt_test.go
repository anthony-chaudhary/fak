package model

import (
	"reflect"
	"testing"
)

func auditGenerateEmptyTinyConfig() Config {
	return Config{
		ModelType:         "llama",
		VocabSize:         16,
		HiddenSize:        32,
		IntermediateSize:  64,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        2,
		HeadDim:           16,
		RMSNormEps:        1e-6,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAuditGenerateRejectsEmptyPromptInsteadOfPanicking(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prompt []int
	}{
		{name: "nil", prompt: nil},
		{name: "empty", prompt: []int{}},
	} {
		for _, populated := range []bool{false, true} {
			state := "fresh"
			if populated {
				state = "populated"
			}
			t.Run(tc.name+"/"+state, func(t *testing.T) {
				m := NewSynthetic(auditGenerateEmptyTinyConfig())
				s := m.NewSession()
				defer s.Close()
				if populated {
					s.SetLastLogits(s.Prefill([]int{1, 2, 3}))
					if s.Cache.Len() != 3 || len(s.LastLogits()) != m.Cfg.VocabSize {
						t.Fatal("fixture did not populate the cache and logits")
					}
				}
				cache := s.Cache
				beforeCache := cache.Clone()
				beforeLogits := append([]float32(nil), s.LastLogits()...)
				defer func() {
					if p := recover(); p != nil {
						t.Errorf("Generate(%v, 3) panicked: %v", tc.prompt, p)
					}
				}()

				if got := s.Generate(tc.prompt, 3); len(got) != 0 {
					t.Errorf("Generate(%v, 3) = %v, want no tokens", tc.prompt, got)
				}
				if s.M != m || s.Cache != cache || !s.modelWeightsHeld || s.halClosed {
					t.Fatal("empty prompt changed session ownership or lifecycle state")
				}
				// Compare independently owned snapshots so in-place mutations cannot
				// change the expected value. Clone normalizes nil versus empty rows.
				if !reflect.DeepEqual(s.Cache.Clone(), beforeCache) {
					t.Error("empty prompt changed cached positions, lineage, or K/V rows")
				}
				if !reflect.DeepEqual(s.LastLogits(), beforeLogits) {
					t.Error("empty prompt changed saved logits")
				}
			})
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAuditGenerateNonEmptyPromptMatchesForward(t *testing.T) {
	m := NewSynthetic(auditGenerateEmptyTinyConfig())
	prompt := []int{1, 2, 3}
	const n = 3
	ids := append([]int(nil), prompt...)
	want := make([]int, 0, n)
	for i := 0; i < n; i++ {
		logits := m.Forward(ids).Logits
		next := argmaxF32(logits[len(ids)-1])
		want = append(want, next)
		ids = append(ids, next)
	}

	s := m.NewSession()
	defer s.Close()
	got := s.Generate(prompt, n)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("nonempty Generate = %v, want full-forward greedy tokens %v", got, want)
	}
	if s.Cache.Len() != len(prompt)+n {
		t.Errorf("cache length = %d, want %d after prefill and decode", s.Cache.Len(), len(prompt)+n)
	}

	// EOS is emitted once and stops before Step ingests it.
	m.Cfg.EOSTokenID = want[0]
	eos := m.NewSession()
	defer eos.Close()
	if got := eos.Generate(prompt, n); !reflect.DeepEqual(got, want[:1]) {
		t.Fatalf("EOS Generate = %v, want %v", got, want[:1])
	}
	if eos.Cache.Len() != len(prompt) {
		t.Errorf("EOS cache length = %d, want prompt length %d", eos.Cache.Len(), len(prompt))
	}
}
