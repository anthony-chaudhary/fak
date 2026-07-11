package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

func savingsF64Ptr(v float64) *float64 { return &v }

// twoPricedSavingsRows is a minimal, drift-free (NetUSD == its component parts) fixture of
// two priced provider-prompt-cache days, so the fold yields a MEASURED reduction.
func twoPricedSavingsRows() []cachevaluereport.SavingsRow {
	return []cachevaluereport.SavingsRow{
		{
			Date: "2026-06-21", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			InputTokens: 1000, CacheReadTokens: 9000, OutputTokens: 500,
			InputPerMTokUSD: 3.0, OutputPerMTokUSD: 15.0,
			RebateUSD: 5.0, WritePremiumUSD: 0.1, SpendUSD: 0.5, NetUSD: 4.4,
			SavedTokenEquiv: 8100,
		},
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			InputTokens: 1000, CacheReadTokens: 8000, OutputTokens: 400,
			InputPerMTokUSD: 3.0, OutputPerMTokUSD: 15.0,
			RebateUSD: 4.0, WritePremiumUSD: 0.2, SpendUSD: 0.4, NetUSD: 3.4,
			SavedTokenEquiv: 7200,
		},
	}
}

func TestBuildTUIOverviewSavingsFoldsMeasured(t *testing.T) {
	at := time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)
	s := buildTUIOverviewSavings(twoPricedSavingsRows(), at)
	if s == nil {
		t.Fatal("expected a savings hero model from two priced rows, got nil")
	}
	if s.Verdict != "MEASURED" {
		t.Fatalf("verdict = %q, want MEASURED", s.Verdict)
	}
	if s.Dates != 2 || s.Rows != 2 {
		t.Fatalf("dates=%d rows=%d, want 2/2", s.Dates, s.Rows)
	}
	if s.ReductionPct == nil || *s.ReductionPct <= 0 {
		t.Fatalf("reduction pct = %v, want a positive priced reduction", s.ReductionPct)
	}
	if got := round2(s.NetUSD); got != 7.8 {
		t.Fatalf("net usd = %v, want 7.8 (4.4 + 3.4)", got)
	}
	if s.SavedTokenEquiv != 15300 {
		t.Fatalf("saved token equiv = %v, want 15300", s.SavedTokenEquiv)
	}
	if s.Fidelity != "lossless" {
		t.Fatalf("fidelity = %q, want lossless", s.Fidelity)
	}
	if len(s.NetTrend) != 2 || round2(s.NetTrend[0]) != 4.4 || round2(s.NetTrend[1]) != 7.8 {
		t.Fatalf("net trend = %v, want cumulative [4.4 7.8]", s.NetTrend)
	}
}

func TestBuildTUIOverviewSavingsNilWhenEmpty(t *testing.T) {
	at := time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)
	if s := buildTUIOverviewSavings(nil, at); s != nil {
		t.Fatalf("expected nil hero from no rows, got %+v", s)
	}
}

func TestSavingsHeroHeadlineModes(t *testing.T) {
	priced := &tuiOverviewSavings{ReductionPct: savingsF64Ptr(80.8)}
	frac, head := savingsHeroHeadline(priced)
	if round2(frac) != 0.81 || !strings.Contains(head, "80.8% net API $ cost reduction") {
		t.Fatalf("priced headline: frac=%v head=%q", frac, head)
	}

	blind := &tuiOverviewSavings{CacheReadFraction: savingsF64Ptr(0.93)}
	frac, head = savingsHeroHeadline(blind)
	if round2(frac) != 0.93 || !strings.Contains(head, "dollar-blind") {
		t.Fatalf("dollar-blind headline: frac=%v head=%q", frac, head)
	}

	tokenOnly := &tuiOverviewSavings{}
	frac, head = savingsHeroHeadline(tokenOnly)
	if frac != 0 || !strings.Contains(head, "token-only") {
		t.Fatalf("token-only headline: frac=%v head=%q", frac, head)
	}
}

func TestRenderTUIOverviewSavingsHero(t *testing.T) {
	if got := renderTUIOverviewSavingsHero(nil, 120); got != "" {
		t.Fatalf("nil model should render empty, got %q", got)
	}
	s := buildTUIOverviewSavings(twoPricedSavingsRows(), time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC))
	out := renderTUIOverviewSavingsHero(s, 120)
	for _, want := range []string{"Cache savings", "net API $ cost reduction", "█", "net $", "rows / 2 dates", "lossless"} {
		if !strings.Contains(out, want) {
			t.Fatalf("hero render missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "\n\n") {
		t.Fatalf("hero should be tight two lines with no blank line, got:\n%s", out)
	}
}

func TestFormatSavingsUSD(t *testing.T) {
	cases := map[float64]string{
		0:         "$0.00",
		12.5:      "$12.50",
		1234:      "$1.2k",
		2_500_000: "$2.5M",
		-1500:     "-$1.5k",
	}
	for in, want := range cases {
		if got := formatSavingsUSD(in); got != want {
			t.Fatalf("formatSavingsUSD(%v) = %q, want %q", in, got, want)
		}
	}
}

func round2(v float64) float64 {
	return float64(int64(v*100+0.5)) / 100
}
