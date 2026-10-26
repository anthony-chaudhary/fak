package reasoneval

import "testing"

// TestDeepInferAIMENormalization ports four source-derived compatibility cases
// from DeepInfer@553741557a57df44a46fb3ab72b6aafeaf945337, eval/aime/bench.py:
// 73-89 (last complete box), 95-110 (normalization), and 127-148 (scoring).
// The large-integer cases stress normalization, not the valid AIME 0-999 domain.
// Signed fallback and incomplete-box handling remain native policy controls.
//
// fak-test:runtime fast est=1ms lane=default
func TestDeepInferAIMENormalization(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		text string
		gold string
		want RuleResult
	}{
		{
			name: "empty last complete box uses tail",
			text: `\boxed{12} then \boxed{} final 34`,
			gold: "34",
			want: RuleResult{Prediction: "34", Source: "tail_number", Correct: true},
		},
		{
			name: "last complete box normalizing to empty uses tail",
			text: `\boxed{12} then \boxed{$...$} final 34`,
			gold: "34",
			want: RuleResult{Prediction: "34", Source: "tail_number", Correct: true},
		},
		{
			name: "outer whitespace precedes dollar removal",
			text: `\boxed{ $123$ }`,
			gold: " $123$ ",
			want: RuleResult{Prediction: "123", Source: "boxed", Correct: true},
		},
		{
			name: "all trailing periods are removed",
			text: `\boxed{123...}`,
			gold: "123",
			want: RuleResult{Prediction: "123", Source: "boxed", Correct: true},
		},
		{
			name: "integer beyond machine range loses leading zeros",
			text: `\boxed{0009223372036854775808}`,
			gold: "9223372036854775808",
			want: RuleResult{Prediction: "9223372036854775808", Source: "boxed", Correct: true},
		},
		{
			name: "large negative integer retains sign",
			text: `\boxed{-0009223372036854775809}`,
			gold: "-9223372036854775809",
			want: RuleResult{Prediction: "-9223372036854775809", Source: "boxed", Correct: true},
		},
		{
			name: "Unicode negative zero is canonical zero",
			text: `\boxed{-٠٠٠}`,
			gold: "0",
			want: RuleResult{Prediction: "0", Source: "boxed", Correct: true},
		},
		{
			name: "native signed fallback is not donor unsigned fallback",
			text: "Final -123",
			gold: "123",
			want: RuleResult{Prediction: "-123", Source: "tail_number", Correct: false},
		},
		{
			name: "incomplete final box still permits earlier complete box",
			text: `\boxed{12} then \boxed{`,
			gold: "12",
			want: RuleResult{Prediction: "12", Source: "boxed", Correct: true},
		},
		{
			name: "incomplete box without answer remains unscored",
			text: `\boxed{`,
			gold: "12",
			want: RuleResult{Source: "none"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScoreRule("aime", tc.text, tc.gold)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("ScoreRule = %+v, want %+v", got, tc.want)
			}
		})
	}
}
