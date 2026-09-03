package gateway

import (
	"errors"
	"math"
	"testing"
)

func TestGrammarMaskMonotonicityWitness(t *testing.T) {
	// First witness requirements (#9925):
	// 1. Seed -inf outside tokenizer vocabulary (e.g. padded vocabulary tail).
	// 2. Apply grammar masks (including masks that mistakenly mark padded tokens as valid).
	// 3. Assert no previously impossible (-inf) token becomes finite under any grammar state.
	// 4. Verification that valid tokens are properly filtered.

	vocabSize := 1024
	tokenizerVocabLimit := 1000

	logits := make([]float32, vocabSize)
	for i := 0; i < tokenizerVocabLimit; i++ {
		logits[i] = float32(i%10) - 5.0
	}
	for i := tokenizerVocabLimit; i < vocabSize; i++ {
		logits[i] = float32(math.Inf(-1))
	}

	grammarAllowed := map[int]bool{
		10:   true,
		20:   true,
		30:   true,
		1005: true,
		1010: true,
	}

	receipt, err := ApplyGrammarMaskWithMonotonicity(logits, grammarAllowed)
	if err != nil {
		t.Fatalf("ApplyGrammarMaskWithMonotonicity failed: %v", err)
	}

	if !receipt.DomainMonotonicityOK {
		t.Fatal("expected DomainMonotonicityOK = true")
	}
	if receipt.WidenedTokens != 0 {
		t.Fatalf("expected 0 widened tokens, got %d", receipt.WidenedTokens)
	}

	for i := tokenizerVocabLimit; i < vocabSize; i++ {
		if !math.IsInf(float64(logits[i]), -1) {
			t.Fatalf("token %d in padded tail was widened to finite logit %v", i, logits[i])
		}
	}

	for i := 0; i < tokenizerVocabLimit; i++ {
		if !grammarAllowed[i] {
			if !math.IsInf(float64(logits[i]), -1) {
				t.Fatalf("unallowed token %d was not masked to -inf", i)
			}
		} else {
			if math.IsInf(float64(logits[i]), -1) {
				t.Fatalf("allowed valid token %d was wrongly masked", i)
			}
		}
	}
}

func TestGrammarMaskMonotonicityDefectDetection(t *testing.T) {
	naiveApply := func(logits []float32, allowed map[int]bool) error {
		origInf := make([]bool, len(logits))
		for i, v := range logits {
			origInf[i] = math.IsInf(float64(v), -1)
		}

		for i := range logits {
			if allowed[i] {
				logits[i] = 1.0
			} else {
				logits[i] = float32(math.Inf(-1))
			}
			if origInf[i] && !math.IsInf(float64(logits[i]), -1) {
				return ErrGrammarMaskWidenedDomain
			}
		}
		return nil
	}

	logits := []float32{1.0, 2.0, float32(math.Inf(-1))}
	allowed := map[int]bool{2: true}

	err := naiveApply(logits, allowed)
	if !errors.Is(err, ErrGrammarMaskWidenedDomain) {
		t.Fatalf("expected ErrGrammarMaskWidenedDomain, got: %v", err)
	}
}
