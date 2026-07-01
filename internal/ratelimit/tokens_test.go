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
