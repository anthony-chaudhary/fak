package ctxmmu

import (
	"errors"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestRulerTasksGenerationAndScoring(t *testing.T) {
	for _, fam := range []RulerTaskFamily{RulerNIAH, RulerVarTracking, RulerAggregation} {
		tc, err := GenerateRulerTask(RulerTaskSpec{
			Family:         fam,
			ContextTokens:  256,
			NeedleDepthPct: 50,
			NumNeedles:     3,
			Seed:           42,
		})
		if err != nil {
			t.Fatalf("%s: %v", fam, err)
		}
		if tc.ExpectedAnswer == "" || !strings.Contains(tc.Prompt, "Question:") {
			t.Fatalf("%s: malformed task case %+v", fam, tc)
		}
		if fam == RulerNIAH && !strings.Contains(tc.Prompt, tc.ExpectedAnswer) {
			t.Fatalf("niah prompt missing expected answer %q", tc.ExpectedAnswer)
		}
		score := ScoreRulerResidency(tc.ExpectedAnswer, "Answer: "+tc.ExpectedAnswer, 256, 256)
		if !score.Matched || score.ResidencyRatio != 1.0 {
			t.Fatalf("%s: unexpected score %+v", fam, score)
		}
	}

	if _, err := GenerateRulerTask(RulerTaskSpec{Family: "bad", ContextTokens: 256, NumNeedles: 1}); !errors.Is(err, ErrInvalidRulerTaskSpec) {
		t.Fatalf("expected ErrInvalidRulerTaskSpec, got %v", err)
	}
	if _, err := GenerateRulerTask(RulerTaskSpec{Family: RulerNIAH, ContextTokens: 16, NumNeedles: 1}); !errors.Is(err, ErrInvalidRulerTaskSpec) {
		t.Fatalf("expected ErrInvalidRulerTaskSpec for small context, got %v", err)
	}
}
