package reasoneval

import (
	"errors"
	"testing"
)

func TestScoreRuleAIMEUnicodeDecimalTail(t *testing.T) {
	t.Run("final unicode decimal span wins and normalizes", func(t *testing.T) {
		got, err := ScoreRule("aime", "reasoning ١٢ </think> final ١٢ ٣٤", "34")
		if err != nil {
			t.Fatal(err)
		}
		if got.Prediction != "34" || got.Source != "tail_number" || !got.Correct {
			t.Fatalf("ScoreRule = %+v, want normalized final Unicode-decimal answer", got)
		}
	})

	t.Run("python word boundary and four digit spans stay rejected", func(t *testing.T) {
		for _, text := range []string{"final x٣٤", "final ١٢٣٤"} {
			got, err := ScoreRule("aime", text, "34")
			if err != nil {
				t.Fatal(err)
			}
			if got.Prediction != "" || got.Source != "none" || got.Correct {
				t.Fatalf("ScoreRule(%q) = %+v, want no bounded numeric answer", text, got)
			}
		}
	})
}

func TestScoreRuleGPQAFinalStandaloneChoice(t *testing.T) {
	got, err := ScoreRule("gpqa", "analysis considers A, then final A C", "C")
	if err != nil {
		t.Fatal(err)
	}
	if got.Prediction != "C" || got.Source != "tail_letter" || !got.Correct {
		t.Fatalf("ScoreRule = %+v, want final standalone choice C", got)
	}
}

func TestScoreRuleHLERequiresJudge(t *testing.T) {
	got, err := ScoreRule("hle", "model answer", "reference answer")
	if !errors.Is(err, ErrJudgeRequired) {
		t.Fatalf("error = %v, want ErrJudgeRequired", err)
	}
	if got != (RuleResult{}) {
		t.Fatalf("result = %+v, want zero result on judge refusal", got)
	}
}
