package model

import (
	"slices"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestAuditGenerateRejectsEmptyPromptInsteadOfPanicking(t *testing.T) {
	m := NewSynthetic(Config{
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
	})
	for _, tc := range []struct {
		name   string
		prompt []int
	}{
		{name: "nil"},
		{name: "empty", prompt: []int{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := m.NewSession()
			defer s.Close()
			if got := s.Generate(tc.prompt, 1); got != nil {
				t.Fatalf("Generate(%v, 1) = %v, want nil", tc.prompt, got)
			}
			if got := s.Cache.Len(); got != 0 {
				t.Fatalf("empty prompt mutated cache length: got %d, want 0", got)
			}
		})
	}
	t.Run("non-empty-control", func(t *testing.T) {
		s := m.NewSession()
		defer s.Close()
		ref := m.NewSession()
		defer ref.Close()
		prompt := []int{1, 2}
		logits := ref.Prefill(prompt)
		var want []int
		for range 3 {
			next := argmaxF32(logits)
			want = append(want, next)
			logits = ref.Step(next)
		}
		if got := s.Generate(prompt, len(want)); !slices.Equal(got, want) {
			t.Fatalf("non-empty Generate = %v, want serial greedy control %v", got, want)
		}
	})
}
