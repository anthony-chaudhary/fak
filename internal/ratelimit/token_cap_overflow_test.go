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
		// Unsupported negative comparison operands retain their existing policy.
		{"legacy_negative_used", math.MaxInt64, -2, 1, true},
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
func TestTokenCapsDerivedTotalOverflow(t *testing.T) {
	oversized := NewTokenUsage(math.MaxInt64, 0, math.MaxInt64)
	for _, tc := range []struct {
		name          string
		used, request TokenUsage
	}{
		{"overflowed_used", oversized, NewTokenUsage(1, 0, 0)},
		{"overflowed_request", NewTokenUsage(1, 0, 0), oversized},
		{"both_overflowed", oversized, oversized},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := (TokenCaps{MaxTotalTokens: math.MaxInt64}).Decide(TokenLoad{Tokens: tc.used}, tc.request)
			if got.Admit || got.Cap != TokenCapTotalTokens || got.Used != tc.used.TotalTokens() || got.Requested != tc.request.TotalTokens() {
				t.Fatalf("derived total overflow admitted usage: %+v", got)
			}
			if room := (TokenCaps{MaxTotalTokens: 1}).Headroom(TokenLoad{Tokens: tc.used}); room.TotalTokens != 0 {
				t.Fatalf("derived total exposed false headroom: %+v", room)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestTokenUsageTotalTokensBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage TokenUsage
		want  int64
	}{
		{"zero", TokenUsage{}, 0},
		{"ordinary", NewTokenUsage(3, 1, 2), 5},
		{"exact_boundary", NewTokenUsage(math.MaxInt64-1, 0, 1), math.MaxInt64},
		{"overflow", NewTokenUsage(math.MaxInt64, 0, 1), math.MaxInt64},
		{"existing_negative_normalization", TokenUsage{InputTokens: -1, OutputTokens: 3}, 3},
		// The separate invalid split-normalization policy is unchanged by derived-total saturation.
		{"existing_split_overflow", TokenUsage{CachedInputTokens: math.MaxInt64, UncachedInputTokens: math.MaxInt64}, -2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := tc.usage
			if got := tc.usage.TotalTokens(); got != tc.want || tc.usage != before {
				t.Fatalf("derived total=%d want=%d raw=%+v before=%+v", got, tc.want, tc.usage, before)
			}
		})
	}
}
