package vcachegov

import "testing"

func TestProveStarSavingsCodexLikeAnchor(t *testing.T) {
	p := ProveStarSavings(StarSavingsInput{
		AnchorTokens:    4096,
		SuffixTokens:    10,
		Requests:        7,
		MinPrefixTokens: 1024,
		ReadMult:        0.1,
		WriteMult:       WriteMult5Minutes,
		Secret:          Cacheable,
	})
	if p.Status != ProofProven {
		t.Fatalf("status=%s reason=%s, want PROVEN", p.Status, p.Reason)
	}
	if p.SavedTokenEquiv <= 21000 || p.SavedPct <= 70 {
		t.Fatalf("savings too small: tokens=%v pct=%v", p.SavedTokenEquiv, p.SavedPct)
	}
	if p.BreakEvenRequests != 2 {
		t.Fatalf("BreakEvenRequests=%d, want 2 for Anthropic 5m natural cache", p.BreakEvenRequests)
	}
	if p.CorrectnessDependsOn {
		t.Fatal("vCache proof must never make correctness depend on a provider cache hit")
	}
}

func TestProveStarSavingsRefutesBelowMinimum(t *testing.T) {
	p := ProveStarSavings(StarSavingsInput{
		AnchorTokens:    512,
		SuffixTokens:    10,
		Requests:        100,
		MinPrefixTokens: 1024,
		ReadMult:        0.1,
		WriteMult:       WriteMult5Minutes,
	})
	if p.Status != ProofRefuted {
		t.Fatalf("status=%s, want REFUTED", p.Status)
	}
	if p.Reason != "anchor is below the provider minimum cacheable prefix" {
		t.Fatalf("reason=%q", p.Reason)
	}
}

func TestProveStarSavingsRefutesUnwarmableContent(t *testing.T) {
	p := ProveStarSavings(StarSavingsInput{
		AnchorTokens:    4096,
		SuffixTokens:    10,
		Requests:        100,
		MinPrefixTokens: 1024,
		ReadMult:        0.1,
		WriteMult:       WriteMult5Minutes,
		Secret:          Secret,
	})
	if p.Status != ProofRefuted {
		t.Fatalf("status=%s, want REFUTED", p.Status)
	}
	if p.Reason != "prefix content is not allowed in an implicit provider cache" {
		t.Fatalf("reason=%q", p.Reason)
	}
}

func TestProveStarSavingsRefutesBelowBreakEven(t *testing.T) {
	p := ProveStarSavings(StarSavingsInput{
		AnchorTokens:    4096,
		SuffixTokens:    10,
		Requests:        1,
		MinPrefixTokens: 1024,
		ReadMult:        0.1,
		WriteMult:       WriteMult5Minutes,
	})
	if p.Status != ProofRefuted {
		t.Fatalf("status=%s, want REFUTED", p.Status)
	}
	if p.SavedTokenEquiv >= 0 {
		t.Fatalf("single request should lose under a write multiplier, saved=%v", p.SavedTokenEquiv)
	}
}

func TestNaturalBreakEvenRequests(t *testing.T) {
	if got := NaturalBreakEvenRequests(WriteMult5Minutes, 0.1); got != 2 {
		t.Fatalf("5m break-even=%d, want 2", got)
	}
	if got := NaturalBreakEvenRequests(WriteMult1Hour, 0.1); got != 3 {
		t.Fatalf("1h break-even=%d, want 3", got)
	}
	if got := NaturalBreakEvenRequests(1.0, 0.5); got != 2 {
		t.Fatalf("auto-cache first-real break-even=%d, want 2", got)
	}
}

func TestProveTelemetrySavingsClaudeProbeNeedsFourthTurn(t *testing.T) {
	rows := []TelemetryRow{
		{InputTokens: 10098, CacheCreationInputTokens: 59400, Ephemeral1hInputTokens: 59400},
		{InputTokens: 10065, CacheCreationInputTokens: 15411, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15411},
		{InputTokens: 10065, CacheCreationInputTokens: 15410, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15410},
		{InputTokens: 10065, CacheCreationInputTokens: 15424, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15424},
	}
	p3 := ProveTelemetrySavings(TelemetrySavingsInput{
		Rows:        rows[:3],
		ReadMult:    0.1,
		Write5mMult: WriteMult5Minutes,
		Write1hMult: WriteMult1Hour,
	})
	if p3.Status != ProofRefuted {
		t.Fatalf("first three Claude turns status=%s, want REFUTED", p3.Status)
	}
	if p3.SavedTokenEquiv >= 0 {
		t.Fatalf("first three Claude turns should still be below break-even, saved=%v", p3.SavedTokenEquiv)
	}

	p4 := ProveTelemetrySavings(TelemetrySavingsInput{
		Rows:        rows,
		ReadMult:    0.1,
		Write5mMult: WriteMult5Minutes,
		Write1hMult: WriteMult1Hour,
	})
	if p4.Status != ProofProven {
		t.Fatalf("four Claude turns status=%s reason=%s, want PROVEN", p4.Status, p4.Reason)
	}
	if p4.FirstPositiveRequest != 4 {
		t.Fatalf("FirstPositiveRequest=%d, want 4", p4.FirstPositiveRequest)
	}
	if p4.BaselineTokenEquiv != 277923 {
		t.Fatalf("BaselineTokenEquiv=%v, want 277923", p4.BaselineTokenEquiv)
	}
	if p4.ActualTokenEquiv != 264781.5 || p4.SavedTokenEquiv != 13141.5 {
		t.Fatalf("actual/saved=%v/%v, want 264781.5/13141.5", p4.ActualTokenEquiv, p4.SavedTokenEquiv)
	}
	if p4.CorrectnessDependsOn {
		t.Fatal("telemetry proof must never make correctness depend on a provider cache hit")
	}
}

func TestProveTelemetrySavingsRefutesNoCacheReads(t *testing.T) {
	p := ProveTelemetrySavings(TelemetrySavingsInput{
		Rows: []TelemetryRow{{InputTokens: 100, CacheCreationInputTokens: 100, Ephemeral5mInputTokens: 100}},
	})
	if p.Status != ProofRefuted || p.Reason != "telemetry observed no cache reads" {
		t.Fatalf("proof=%+v, want no-cache-read refutation", p)
	}
}
