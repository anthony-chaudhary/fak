package cachevaluereport

import (
	"math"
	"testing"
)

// TestModelMomentGolden pins the per-model rollup to hand-verified dollars:
// two sessions of IDENTICAL token shape on different models must yield
// per-model billed/saving numbers that scale exactly with each model's base
// input $/MTok, so the apex model's larger saving is visible instead of being
// folded under the cheaper model's rate.
//
// Hand math for the opus row (ipt=5/1e6, opt=25/1e6):
//
//	billed = 10000*ipt + 1000000*ipt*0.1 + 200000*ipt*1.25 + 20000*opt
//	       = 0.05 + 0.50 + 1.25 + 0.50 = 2.30
//	saving = 1000000*ipt*(1-0.1) - 200000*ipt*(1.25-1) = 4.50 - 0.25 = 4.25
//
// The fable row prices the same shape at 2x the list price, so both dollars
// must be exactly 2x opus.
func TestModelMomentGolden(t *testing.T) {
	sessions := []ModelSession{
		{Model: "claude-opus-4-8", InputTokens: 10000, CacheReadInputTokens: 1000000, CacheCreationInputTokens: 200000, OutputTokens: 20000},
		{Model: "claude-fable-5", InputTokens: 10000, CacheReadInputTokens: 1000000, CacheCreationInputTokens: 200000, OutputTokens: 20000},
	}
	prices := map[string]ModelPrice{
		"claude-opus-4-8": {InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
		"claude-fable-5":  {InputPerMTokUSD: 10, OutputPerMTokUSD: 50},
	}

	groups, unpriced := FableMoment(sessions, prices)
	if len(unpriced) != 0 {
		t.Fatalf("unpriced = %v, want empty", unpriced)
	}
	if len(groups) != 2 {
		t.Fatalf("len(groups) = %d, want 2", len(groups))
	}
	// Groups are sorted by Model: claude-fable-5 < claude-opus-4-8.
	fable, opus := groups[0], groups[1]
	if fable.Model != "claude-fable-5" || opus.Model != "claude-opus-4-8" {
		t.Fatalf("group order = [%q %q], want [claude-fable-5 claude-opus-4-8]", fable.Model, opus.Model)
	}

	const tol = 1e-9
	assertUSD := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > tol {
			t.Errorf("%s = %.12f, want %.12f (tol %g)", name, got, want, tol)
		}
	}
	if opus.Sessions != 1 || fable.Sessions != 1 {
		t.Errorf("Sessions = opus:%d fable:%d, want 1 each", opus.Sessions, fable.Sessions)
	}
	assertUSD("opus BilledUSD", opus.BilledUSD, 2.30)
	assertUSD("opus NetCacheSavingUSD", opus.NetCacheSavingUSD, 4.25)
	assertUSD("fable BilledUSD", fable.BilledUSD, 4.60)
	assertUSD("fable NetCacheSavingUSD", fable.NetCacheSavingUSD, 8.50)

	// The visibility claim itself: fable's dollars are exactly 2x opus's.
	assertUSD("fable/opus billed ratio", fable.BilledUSD, 2*opus.BilledUSD)
	assertUSD("fable/opus saving ratio", fable.NetCacheSavingUSD, 2*opus.NetCacheSavingUSD)
}

// TestModelMomentUnpriced pins the skip contract: a session whose model has no
// list price is excluded from groups but surfaced in unpriced, never silently
// dropped.
func TestModelMomentUnpriced(t *testing.T) {
	sessions := []ModelSession{
		{Model: "claude-opus-4-8", InputTokens: 1000, OutputTokens: 100},
		{Model: "mystery", InputTokens: 500, CacheReadInputTokens: 400, OutputTokens: 50},
		{Model: "mystery", InputTokens: 700, OutputTokens: 70},
	}
	prices := map[string]ModelPrice{
		"claude-opus-4-8": {InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}

	groups, unpriced := FableMoment(sessions, prices)
	if len(unpriced) != 1 || unpriced[0] != "mystery" {
		t.Fatalf("unpriced = %v, want [mystery] (deduped)", unpriced)
	}
	if len(groups) != 1 || groups[0].Model != "claude-opus-4-8" {
		t.Fatalf("groups = %v, want only claude-opus-4-8", groups)
	}
	if groups[0].Sessions != 1 {
		t.Errorf("Sessions = %d, want 1 (mystery sessions must not fold in)", groups[0].Sessions)
	}
}
