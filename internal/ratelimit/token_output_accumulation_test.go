package ratelimit

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestTokenUsageOutputAccumulation(t *testing.T) {
	for _, tc := range []struct {
		name              string
		left, right, want int64
	}{
		{"ordinary", 3, 4, 7},
		{"zero", 0, 0, 0},
		{"max_with_zero", math.MaxInt64, 0, math.MaxInt64},
		{"exact_boundary", math.MaxInt64 - 2, 2, math.MaxInt64},
		{"one_over_boundary", math.MaxInt64 - 2, 3, math.MaxInt64},
		{"both_max", math.MaxInt64, math.MaxInt64, math.MaxInt64},
		{"negative_normalized", -1, 3, 3},
		{"both_negative", -1, -2, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u := TokenUsage{InputTokens: 5, CachedInputTokens: 2, UncachedInputTokens: 3, OutputTokens: tc.left}
			v := TokenUsage{InputTokens: 7, CachedInputTokens: 3, UncachedInputTokens: 4, OutputTokens: tc.right}
			originalU, originalV := u, v
			want := TokenUsage{InputTokens: 12, CachedInputTokens: 5, UncachedInputTokens: 7, OutputTokens: tc.want}
			if got := u.Add(v); got != want {
				t.Fatalf("output accumulation wrapped: got=%+v want=%+v", got, want)
			}
			if got := v.Add(u); got != want {
				t.Fatalf("output accumulation order changed: got=%+v want=%+v", got, want)
			}
			if u != originalU || v != originalV {
				t.Fatal("Add mutated caller usage")
			}
		})
	}
}
