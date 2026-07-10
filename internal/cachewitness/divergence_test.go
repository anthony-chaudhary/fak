package cachewitness

import "testing"

// rec builds a witness Record with a WITNESSED kv_prefix axis (promptTokens/reused) and an
// OBSERVED provider axis (providerCacheRead). providerCacheRead == 0 is the pure in-kernel
// path (no OBSERVED axis to compare against).
func rec(url string, promptTokens, reused, providerCacheRead uint64) Record {
	return Record{
		GatewayURL:              url,
		KVPrefix:                KVPrefixWitness{PromptTokens: promptTokens, ReusedTokens: reused},
		ProviderCacheReadTokens: providerCacheRead,
	}
}

func TestFoldReuseDivergence_FlagsDisagreement(t *testing.T) {
	records := []Record{
		rec("gw-A", 1000, 900, 100), // witnessed 0.90 vs observed 0.10 → diverge 0.80
		rec("gw-B", 1000, 800, 750), // witnessed 0.80 vs observed 0.75 → agree (0.05)
	}
	rep := FoldReuseDivergence(records, 0.15)
	if rep.Verdict != "TRUST_CLASS_DIVERGENCE" || rep.OK {
		t.Fatalf("a 0.90-vs-0.10 record must flag divergence, got %s ok=%v: %s", rep.Verdict, rep.OK, rep.Finding)
	}
	if rep.Compared != 2 {
		t.Fatalf("both records have both axes, expected Compared=2, got %d", rep.Compared)
	}
	if len(rep.Diverged) != 1 || rep.Diverged[0].GatewayURL != "gw-A" {
		t.Fatalf("only gw-A should flag, got %+v", rep.Diverged)
	}
	if got := rep.Diverged[0].AbsDivergence; got < 0.79 || got > 0.81 {
		t.Fatalf("gw-A divergence should be ~0.80, got %v", got)
	}
}

func TestFoldReuseDivergence_AgreeingNotFlagged(t *testing.T) {
	rep := FoldReuseDivergence([]Record{
		rec("a", 1000, 600, 550),  // 0.60 vs 0.55 → 0.05
		rec("b", 2000, 1000, 900), // 0.50 vs 0.45 → 0.05
	}, 0.15)
	if rep.Verdict != "OK" || !rep.OK || len(rep.Diverged) != 0 {
		t.Fatalf("agreeing records must be OK with no flags, got %s / %d flagged", rep.Verdict, len(rep.Diverged))
	}
	if rep.Compared != 2 {
		t.Fatalf("expected Compared=2, got %d", rep.Compared)
	}
}

func TestFoldReuseDivergence_SingleClassSkipped(t *testing.T) {
	// Pure in-kernel records (provider cache_read == 0) have no OBSERVED axis to compare —
	// reported as single-class, never flagged.
	rep := FoldReuseDivergence([]Record{
		rec("in-kernel-1", 1000, 950, 0),
		rec("in-kernel-2", 500, 400, 0),
	}, 0.15)
	if rep.Verdict != "INSUFFICIENT" || !rep.OK {
		t.Fatalf("an all-single-class corpus must fall open INSUFFICIENT, got %s ok=%v", rep.Verdict, rep.OK)
	}
	if rep.Compared != 0 || rep.SingleClass != 2 {
		t.Fatalf("expected Compared=0 SingleClass=2, got %d / %d", rep.Compared, rep.SingleClass)
	}
}

func TestFoldReuseDivergence_WorstFirst(t *testing.T) {
	rep := FoldReuseDivergence([]Record{
		rec("mild", 1000, 500, 300), // 0.50 vs 0.30 → 0.20
		rec("worst", 1000, 950, 50), // 0.95 vs 0.05 → 0.90
		rec("mid", 1000, 700, 300),  // 0.70 vs 0.30 → 0.40
	}, 0.15)
	if len(rep.Diverged) != 3 {
		t.Fatalf("all three should flag, got %d", len(rep.Diverged))
	}
	if rep.Diverged[0].GatewayURL != "worst" || rep.Diverged[2].GatewayURL != "mild" {
		t.Fatalf("diverged must be worst-first, got %s..%s", rep.Diverged[0].GatewayURL, rep.Diverged[2].GatewayURL)
	}
}

func TestFoldReuseDivergence_ObservedRatioClamped(t *testing.T) {
	// A provider may relay cumulative cache_read exceeding this window's prompt denominator;
	// the OBSERVED ratio clamps to 1.0 rather than inflating past it.
	rep := FoldReuseDivergence([]Record{rec("gw", 1000, 990, 5000)}, 0.15)
	d := rep.Diverged
	// witnessed 0.99, observed clamp 1.0 → divergence 0.01 → agree, no flag.
	if len(d) != 0 {
		t.Fatalf("clamped observed 1.0 vs witnessed 0.99 should agree, got flag %+v", d)
	}
	if rep.Compared != 1 || rep.Verdict != "OK" {
		t.Fatalf("expected one compared OK record, got %d / %s", rep.Compared, rep.Verdict)
	}
}

func TestFoldReuseDivergence_DefaultTolerance(t *testing.T) {
	// tolerance <= 0 uses DefaultReuseDivergenceTolerance (0.15).
	rep := FoldReuseDivergence([]Record{rec("gw", 1000, 900, 700)}, 0) // 0.90 vs 0.70 → 0.20 > 0.15
	if rep.Tolerance != DefaultReuseDivergenceTolerance {
		t.Fatalf("expected default tolerance %.2f, got %.2f", DefaultReuseDivergenceTolerance, rep.Tolerance)
	}
	if rep.Verdict != "TRUST_CLASS_DIVERGENCE" {
		t.Fatalf("0.20 divergence must flag under the default 0.15 tolerance, got %s", rep.Verdict)
	}
}

func TestFoldReuseDivergence_ZeroPromptNotComparable(t *testing.T) {
	rep := FoldReuseDivergence([]Record{rec("gw", 0, 0, 100)}, 0.15)
	if rep.Compared != 0 || rep.SingleClass != 1 || rep.Verdict != "INSUFFICIENT" {
		t.Fatalf("a zero-prompt record is not comparable, got compared=%d single=%d %s", rep.Compared, rep.SingleClass, rep.Verdict)
	}
}
