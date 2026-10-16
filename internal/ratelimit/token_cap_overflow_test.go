package ratelimit

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestTokenCapsRefuseAdditionOverflow(t *testing.T) {
	const cap = int64(10)
	for _, tc := range []struct {
		name          string
		caps          TokenCaps
		used, request TokenUsage
	}{
		{TokenCapInputTokens, TokenCaps{MaxInputTokens: cap}, NewTokenUsage(math.MaxInt64, 0, 0), NewTokenUsage(1, 0, 0)},
		{TokenCapCachedInputTokens, TokenCaps{MaxCachedInputTokens: cap}, NewTokenUsage(math.MaxInt64, math.MaxInt64, 0), NewTokenUsage(1, 1, 0)},
		{TokenCapUncachedInputTokens, TokenCaps{MaxUncachedInputTokens: cap}, NewTokenUsage(math.MaxInt64, 0, 0), NewTokenUsage(1, 0, 0)},
		{TokenCapOutputTokens, TokenCaps{MaxOutputTokens: cap}, NewTokenUsage(0, 0, math.MaxInt64), NewTokenUsage(0, 0, 1)},
		// Each total is representable; only used + request exceeds int64 capacity.
		{TokenCapTotalTokens, TokenCaps{MaxTotalTokens: cap}, NewTokenUsage(math.MaxInt64, 0, 0), NewTokenUsage(1, 0, 0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.caps.Decide(TokenLoad{Tokens: tc.used}, tc.request)
			if got.Admit || got.Cap != tc.name || got.Limit != cap || got.Used != math.MaxInt64 || got.Requested != 1 {
				t.Fatalf("overflowed cap admitted normalized counts: %+v", got)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestTokenCapComparisonBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		limit, used, requested int64
		admit                  bool
	}{
		{"exact", 10, 4, 6, true},
		{"below", 10, 4, 5, true},
		{"above", 10, 4, 7, false},
		{"zero_at_limit", 10, 10, 0, true},
		{"zero_over_limit", 10, 11, 0, false},
		{"max_exact", math.MaxInt64, math.MaxInt64 - 1, 1, true},
		{"max_overflow", math.MaxInt64, math.MaxInt64, 1, false},
		{"request_overflow", 10, 1, math.MaxInt64, false},
		{"unlimited_zero", 0, math.MaxInt64, math.MaxInt64, true},
		{"unlimited_negative", -1, math.MaxInt64, math.MaxInt64, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := checkTokenCap(TokenCapInputTokens, tc.limit, tc.used, tc.requested)
			if got.Admit != tc.admit {
				t.Fatalf("boundary admission=%+v want admit=%t", got, tc.admit)
			}
			if !tc.admit && (got.Cap != TokenCapInputTokens || got.Limit != tc.limit || got.Used != tc.used || got.Requested != tc.requested || got.Headroom != tc.limit-tc.used) {
				t.Fatalf("refusal evidence changed: %+v", got)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestTokenCapsPreserveDerivedInvalidTotalPolicy(t *testing.T) {
	// TotalTokens' separate unchecked aggregation can still produce a negative
	// operand. This comparator repair deliberately does not change that policy.
	invalid := NewTokenUsage(math.MaxInt64, 0, math.MaxInt64)
	for _, tc := range []struct {
		name          string
		used, request TokenUsage
	}{
		{"negative_used", invalid, NewTokenUsage(1, 0, 0)},
		{"negative_request", NewTokenUsage(1, 0, 0), invalid},
		{"both_negative", invalid, invalid},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := (TokenCaps{MaxTotalTokens: math.MaxInt64}).Decide(TokenLoad{Tokens: tc.used}, tc.request)
			if !got.Admit {
				t.Fatalf("legacy invalid-total policy changed: %+v", got)
			}
		})
	}
}
