package ratelimit

import (
	"encoding/json"
	"testing"
)

func mustUsageFixture(t *testing.T, raw string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("fixture did not unmarshal: %v", err)
	}
	return out
}

func TestNormalizeProviderTokensOpenAICachedFixture(t *testing.T) {
	raw := mustUsageFixture(t, `{
		"usage": {
			"input_tokens": 1280,
			"input_tokens_details": {"cached_tokens": 960},
			"output_tokens": 144,
			"total_tokens": 1424
		}
	}`)

	got := NormalizeProviderTokens(raw)
	want := TokenUsage{
		InputTokens:         1280,
		CachedInputTokens:   960,
		UncachedInputTokens: 320,
		OutputTokens:        144,
	}
	if got != want {
		t.Fatalf("normalized OpenAI cached fixture = %+v, want %+v", got, want)
	}
	if got.TotalTokens() != 1424 {
		t.Fatalf("total tokens = %d, want 1424", got.TotalTokens())
	}
	if got.UncachedTotalTokens() != 464 {
		t.Fatalf("uncached total = %d, want 464", got.UncachedTotalTokens())
	}
}

func TestNormalizeProviderTokensCodexUncachedFixture(t *testing.T) {
	raw := mustUsageFixture(t, `{
		"usage": {
			"cached_input_tokens": 2048,
			"uncached_input_tokens": 512,
			"output_tokens": 256
		}
	}`)

	got := NormalizeProviderTokens(raw)
	want := TokenUsage{
		InputTokens:         2560,
		CachedInputTokens:   2048,
		UncachedInputTokens: 512,
		OutputTokens:        256,
	}
	if got != want {
		t.Fatalf("normalized Codex uncached fixture = %+v, want %+v", got, want)
	}
}

func TestNormalizeProviderTokensOpenAIChatFixture(t *testing.T) {
	raw := mustUsageFixture(t, `{
		"usage": {
			"prompt_tokens": 200,
			"prompt_tokens_details": {"cached_tokens": 75},
			"completion_tokens": 25,
			"total_tokens": 225
		}
	}`)

	got := NormalizeProviderTokens(raw)
	want := TokenUsage{
		InputTokens:         200,
		CachedInputTokens:   75,
		UncachedInputTokens: 125,
		OutputTokens:        25,
	}
	if got != want {
		t.Fatalf("normalized OpenAI chat fixture = %+v, want %+v", got, want)
	}
}

func TestTokenCapsHeadroomAndAdmit(t *testing.T) {
	caps := TokenCaps{
		MaxConcurrent:          2,
		MaxInputTokens:         100,
		MaxCachedInputTokens:   70,
		MaxUncachedInputTokens: 40,
		MaxOutputTokens:        50,
		MaxTotalTokens:         140,
	}
	load := TokenLoad{
		Concurrent: 1,
		Tokens:     NewTokenUsage(60, 40, 20),
	}
	headroom := caps.Headroom(load)
	if headroom.Concurrent != 1 ||
		headroom.InputTokens != 40 ||
		headroom.CachedInputTokens != 30 ||
		headroom.UncachedInputTokens != 20 ||
		headroom.OutputTokens != 30 ||
		headroom.TotalTokens != 60 {
		t.Fatalf("headroom = %+v, want concurrent=1 input=40 cached=30 uncached=20 output=30 total=60", headroom)
	}

	request := NewTokenUsage(30, 20, 10)
	decision := caps.Decide(load, request)
	if !decision.Admit {
		t.Fatalf("admission denied unexpectedly: %+v", decision)
	}
	after := load.AfterAdmit(request)
	if after.Concurrent != 2 {
		t.Fatalf("after admit concurrent = %d, want 2", after.Concurrent)
	}
	wantTokens := TokenUsage{InputTokens: 90, CachedInputTokens: 60, UncachedInputTokens: 30, OutputTokens: 30}
	if after.Tokens != wantTokens {
		t.Fatalf("after admit tokens = %+v, want %+v", after.Tokens, wantTokens)
	}
}

func TestTokenCapsDeniesConcurrencyBeforeTokenCaps(t *testing.T) {
	caps := TokenCaps{
		MaxConcurrent:  1,
		MaxInputTokens: 10,
	}
	load := TokenLoad{
		Concurrent: 1,
		Tokens:     NewTokenUsage(10, 0, 0),
	}
	decision := caps.Decide(load, NewTokenUsage(1, 0, 0))
	if decision.Admit {
		t.Fatal("admission allowed despite full concurrency cap")
	}
	if decision.Cap != TokenCapConcurrency || decision.Limit != 1 || decision.Used != 1 || decision.Requested != 1 || decision.Headroom != 0 {
		t.Fatalf("concurrency deny = %+v, want cap=%s limit=1 used=1 requested=1 headroom=0", decision, TokenCapConcurrency)
	}
}

func TestTokenCapsDeniesUncachedHeadroom(t *testing.T) {
	caps := TokenCaps{MaxUncachedInputTokens: 40}
	load := TokenLoad{
		Concurrent: 1,
		Tokens:     NewTokenUsage(90, 55, 0), // uncached = 35
	}
	request := NewTokenUsage(20, 10, 0) // uncached = 10, would take total uncached to 45

	decision := caps.Decide(load, request)
	if decision.Admit {
		t.Fatal("admission allowed despite uncached-input cap")
	}
	if decision.Cap != TokenCapUncachedInputTokens ||
		decision.Limit != 40 ||
		decision.Used != 35 ||
		decision.Requested != 10 ||
		decision.Headroom != 5 {
		t.Fatalf("uncached-input deny = %+v, want cap=%s limit=40 used=35 requested=10 headroom=5", decision, TokenCapUncachedInputTokens)
	}
}

func TestTokenCapsDeniesTotalHeadroom(t *testing.T) {
	caps := TokenCaps{MaxTotalTokens: 100}
	load := TokenLoad{Tokens: NewTokenUsage(70, 50, 20)}
	request := NewTokenUsage(5, 0, 6)

	decision := caps.Decide(load, request)
	if decision.Admit {
		t.Fatal("admission allowed despite total-token cap")
	}
	if decision.Cap != TokenCapTotalTokens ||
		decision.Limit != 100 ||
		decision.Used != 90 ||
		decision.Requested != 11 ||
		decision.Headroom != 10 {
		t.Fatalf("total-token deny = %+v, want cap=%s limit=100 used=90 requested=11 headroom=10", decision, TokenCapTotalTokens)
	}
}

func TestTokenCapsAppliesTargetUtilizationHeadroom(t *testing.T) {
	caps := TokenCaps{
		MaxConcurrent:     10,
		MaxTotalTokens:    100,
		TargetUtilization: 0.9,
	}
	load := TokenLoad{
		Concurrent: 8,
		Tokens:     NewTokenUsage(70, 0, 10), // total = 80
	}

	headroom := caps.Headroom(load)
	if headroom.Concurrent != 1 || headroom.TotalTokens != 10 {
		t.Fatalf("headroom = %+v, want concurrent=1 total=10", headroom)
	}

	if decision := caps.Decide(load, NewTokenUsage(8, 0, 2)); !decision.Admit {
		t.Fatalf("admission denied under target headroom: %+v", decision)
	}
	decision := caps.Decide(load, NewTokenUsage(9, 0, 2))
	if decision.Admit {
		t.Fatal("admission allowed past 90% token target")
	}
	if decision.Cap != TokenCapTotalTokens ||
		decision.Limit != 90 ||
		decision.Used != 80 ||
		decision.Requested != 11 ||
		decision.Headroom != 10 {
		t.Fatalf("target-token deny = %+v, want cap=%s limit=90 used=80 requested=11 headroom=10", decision, TokenCapTotalTokens)
	}

	decision = caps.Decide(TokenLoad{Concurrent: 9}, TokenUsage{})
	if decision.Admit {
		t.Fatal("admission allowed past 90% concurrency target")
	}
	if decision.Cap != TokenCapConcurrency || decision.Limit != 9 || decision.Used != 9 || decision.Requested != 1 || decision.Headroom != 0 {
		t.Fatalf("target-concurrency deny = %+v, want cap=%s limit=9 used=9 requested=1 headroom=0", decision, TokenCapConcurrency)
	}
}

// TestTokenHeadroomSaturatedIsZeroNotUnlimitedSentinel pins a headroom sentinel
// collision: Headroom reserves -1 (UnlimitedHeadroom / UnlimitedConcurrentHeadroom)
// to mean "this dimension has no configured cap". But a CONFIGURED dimension that
// is exactly one unit over its effective cap computed limit-used == -1, which is
// byte-identical to the unlimited sentinel — so a scheduler checking the sentinel
// would misread a saturated cap as unlimited and over-admit. Remaining capacity
// can never be negative, so a saturated/over limited dimension must report 0, and
// only an UNCONFIGURED dimension may report the sentinel. Reachable via the
// documented TargetUtilization knob (below) and via a raw one-over-cap load.
func TestTokenHeadroomSaturatedIsZeroNotUnlimitedSentinel(t *testing.T) {
	// TargetUtilization=0.9 lowers the effective input cap to floor(100*0.9)=90 and
	// concurrency to 90, so a load of 91 is one over each effective cap. Output is
	// left uncapped: a genuinely unlimited dimension that must still report -1.
	caps := TokenCaps{
		MaxConcurrent:     100,
		MaxInputTokens:    100,
		TargetUtilization: 0.9,
	}
	load := TokenLoad{Concurrent: 91, Tokens: NewTokenUsage(91, 0, 0)}
	h := caps.Headroom(load)

	if h.InputTokens == UnlimitedHeadroom {
		t.Fatalf("saturated input headroom = %d, collides with UnlimitedHeadroom sentinel", h.InputTokens)
	}
	if h.InputTokens != 0 {
		t.Fatalf("saturated input headroom = %d, want 0", h.InputTokens)
	}
	if h.Concurrent == UnlimitedConcurrentHeadroom {
		t.Fatalf("saturated concurrency headroom = %d, collides with UnlimitedConcurrentHeadroom sentinel", h.Concurrent)
	}
	if h.Concurrent != 0 {
		t.Fatalf("saturated concurrency headroom = %d, want 0", h.Concurrent)
	}
	// The unlimited contract still holds for a dimension with no configured cap.
	if h.OutputTokens != UnlimitedHeadroom {
		t.Fatalf("uncapped output headroom = %d, want UnlimitedHeadroom(%d)", h.OutputTokens, UnlimitedHeadroom)
	}

	// The collision is not utilization-specific: an exact one-over on a raw cap
	// (no TargetUtilization) hits the same -1.
	raw := TokenCaps{MaxCachedInputTokens: 50}
	rawLoad := TokenLoad{Tokens: NewTokenUsage(51, 51, 0)} // cached = 51, one over the 50 cap
	if got := raw.Headroom(rawLoad).CachedInputTokens; got != 0 {
		t.Fatalf("cached headroom one-over cap = %d, want 0 (not the -1 sentinel)", got)
	}
}
