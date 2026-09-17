package model

import "testing"

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

func captureGenerateEmptyPanic(fn func()) (any, bool) {
	var value any
	ok := false
	func() {
		defer func() {
			if value = recover(); value != nil {
				ok = true
			}
		}()
		fn()
	}()
	return value, ok
}

func TestAuditGenerateRejectsEmptyPromptInsteadOfPanicking(t *testing.T) {
	m := NewSynthetic(auditGenerateEmptyTinyConfig())
	for _, prompt := range [][]int{nil, []int{}} {
		s := m.NewSession()
		panicValue, panicked := captureGenerateEmptyPanic(func() {
			_ = s.Generate(prompt, 1)
		})
		s.Close()
		if panicked {
			t.Fatalf("Generate(%v, 1) panicked instead of handling an empty prompt: %v", prompt, panicValue)
		}
	}
}
