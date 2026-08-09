package cachevaluereport

import (
	"math"
	"strings"
	"testing"
	"time"
)

func mustDay(t *testing.T, s string) time.Time {
	t.Helper()
	d, err := time.Parse("2006-01-02", s)
	if err != nil {
		t.Fatalf("bad test date %q: %v", s, err)
	}
	return d
}

// pricedOpus is a trusted base price so the provider rows carry dollars.
var pricedOpus = SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25, Source: "test:opus"}

// warmProviderObs is a cache-read-heavy (warm) turn: the read rebate dwarfs the write
// premium, so it is NOT a cold-write row.
func warmProviderObs() SavingsObservation {
	return SavingsObservation{
		SessionType: "guard", Provider: "anthropic", Context: "opus",
		InputTokens: 1000, CacheReadTokens: 50000, CacheCreationTokens: 2000, OutputTokens: 3000,
		Pricing: pricedOpus,
	}
}

// coldProviderObs is a write-heavy (cold) turn: cache creation dwarfs the tiny read,
// so the write premium exceeds the rebate.
func coldProviderObs() SavingsObservation {
	return SavingsObservation{
		SessionType: "guard", Provider: "anthropic", Context: "opus",
		InputTokens: 1000, CacheReadTokens: 100, CacheCreationTokens: 100000, OutputTokens: 500,
		Pricing: pricedOpus,
	}
}

// blindProviderObs has real provider token evidence but no trusted price.
func blindProviderObs() SavingsObservation {
	return SavingsObservation{
		SessionType: "guard", Provider: "openai", Context: "codex",
		InputTokens: 500, CacheReadTokens: 20000, CacheCreationTokens: 1000, OutputTokens: 800,
		Pricing: SavingsPricing{DollarBlind: true, Source: "unpriced:openai/codex"},
	}
}

// compactionObs is a bounded-lossy compaction shed (priced).
func compactionObs() SavingsObservation {
	return SavingsObservation{
		SessionType: "guard", Provider: "anthropic", Context: "opus",
		CompactionShedTokens: 4000, CompactionFired: 1, CompactionBudget: 100000,
		Pricing: pricedOpus,
	}
}

func TestFidelityDeriverIsTotalAndShared(t *testing.T) {
	cases := map[string]string{
		"provider_prompt_cache": FidelityLossless,
		"compaction_shed":       FidelityBounded,
		"":                      FidelityUnknown,
		"unknown_mechanism":     FidelityUnknown,
		"something_new":         FidelityUnknown,
	}
	for mech, want := range cases {
		if got := Fidelity(mech); got != want {
			t.Errorf("Fidelity(%q)=%q want %q", mech, got, want)
		}
	}
	// The write path stamps the same value the audit derives — no producer/consumer drift.
	rows := NewSavingsRows(warmProviderObs(), mustDay(t, "2026-07-01"))
	if len(rows) != 1 || rows[0].Fidelity != FidelityLossless {
		t.Fatalf("provider row fidelity = %+v, want one lossless row", rows)
	}
	rows = NewSavingsRows(compactionObs(), mustDay(t, "2026-07-01"))
	if len(rows) != 1 || rows[0].Fidelity != FidelityBounded {
		t.Fatalf("compaction row fidelity = %+v, want one bounded row", rows)
	}
}

// TestFoldAuditReconcilesAndExcludesBlind is the core #2780/#2782 check: per-date and
// cumulative reconciliation, with dollar-blind rows kept OUT of the reduction
// denominator but visible on the token axis.
func TestFoldAuditReconcilesAndExcludesBlind(t *testing.T) {
	var rows []SavingsRow
	rows = append(rows, NewSavingsRows(warmProviderObs(), mustDay(t, "2026-07-01"))...)
	rows = append(rows, NewSavingsRows(blindProviderObs(), mustDay(t, "2026-07-02"))...)
	rows = append(rows, NewSavingsRows(compactionObs(), mustDay(t, "2026-07-02"))...)
	rows = append(rows, NewSavingsRows(coldProviderObs(), mustDay(t, "2026-07-03"))...)

	rep := FoldAudit(rows, mustDay(t, "2026-07-10"))

	if rep.Verdict != "MEASURED" {
		t.Fatalf("verdict=%q want MEASURED", rep.Verdict)
	}
	if len(rep.Dates) != 3 {
		t.Fatalf("dates=%d want 3", len(rep.Dates))
	}
	// Dates sorted chronologically.
	if rep.Dates[0].Date != "2026-07-01" || rep.Dates[2].Date != "2026-07-03" {
		t.Fatalf("dates not sorted: %s..%s", rep.Dates[0].Date, rep.Dates[2].Date)
	}
	c := rep.Cumulative
	if c.Rows != 4 {
		t.Fatalf("cumulative rows=%d want 4", c.Rows)
	}
	if c.DollarBlindRows != 1 {
		t.Fatalf("dollar_blind_rows=%d want 1", c.DollarBlindRows)
	}
	if c.DollarBlindSavedTokenEquiv <= 0 {
		t.Fatalf("blind saved-token-equiv should be positive, got %v", c.DollarBlindSavedTokenEquiv)
	}
	if c.ColdWriteRows != 1 {
		t.Fatalf("cold_write_rows=%d want 1 (the 07-03 write-heavy turn)", c.ColdWriteRows)
	}
	if c.DollarReductionPct == nil {
		t.Fatalf("dollar_reduction_pct nil; priced counterfactual should exist")
	}
	// The blind row contributes NO counterfactual: the denominator is the two priced
	// provider rows only. Verify by re-folding without the blind row and comparing.
	if c.CounterfactualUSD <= 0 {
		t.Fatalf("counterfactual should be positive, got %v", c.CounterfactualUSD)
	}
	noBlind := FoldAudit([]SavingsRow{
		NewSavingsRows(warmProviderObs(), mustDay(t, "2026-07-01"))[0],
		NewSavingsRows(coldProviderObs(), mustDay(t, "2026-07-03"))[0],
	}, mustDay(t, "2026-07-10"))
	if math.Abs(noBlind.Cumulative.CounterfactualUSD-c.CounterfactualUSD) > 1e-9 {
		t.Fatalf("blind row leaked into counterfactual: with=%v without=%v", c.CounterfactualUSD, noBlind.Cumulative.CounterfactualUSD)
	}
	if noBlind.Cumulative.DollarReductionPct == nil || c.DollarReductionPct == nil ||
		math.Abs(*noBlind.Cumulative.DollarReductionPct-*c.DollarReductionPct) > 1e-9 {
		t.Fatalf("blind row shifted reduction pct: with=%v without=%v", derefPct(c.DollarReductionPct), derefPct(noBlind.Cumulative.DollarReductionPct))
	}

	// Reconciliation drift must be ~0: every stored NET equals its re-derived parts.
	if math.Abs(rep.NetReconciliationDriftUSD) > 1e-9 {
		t.Fatalf("net reconciliation drift=%v want ~0", rep.NetReconciliationDriftUSD)
	}
	// Cumulative NET equals the last date's running cumulative.
	if math.Abs(rep.Dates[2].CumulativeNetUSD-c.NetUSD) > 1e-9 {
		t.Fatalf("cumulative net line %v != cumulative account %v", rep.Dates[2].CumulativeNetUSD, c.NetUSD)
	}
	// Fidelity split: lossless (3 provider rows) + bounded (1 compaction).
	fid := map[string]AuditFidelityRow{}
	for _, f := range rep.FidelitySplit {
		fid[f.Fidelity] = f
	}
	if fid[FidelityLossless].Rows != 3 {
		t.Fatalf("lossless rows=%d want 3", fid[FidelityLossless].Rows)
	}
	if fid[FidelityBounded].Rows != 1 || fid[FidelityBounded].ShedTokens != 4000 {
		t.Fatalf("bounded split=%+v want 1 row / 4000 shed", fid[FidelityBounded])
	}
}

func derefPct(p *float64) float64 {
	if p == nil {
		return math.NaN()
	}
	return *p
}

func TestFoldAuditEmptyIsInsufficient(t *testing.T) {
	rep := FoldAudit(nil, mustDay(t, "2026-07-10"))
	if rep.Verdict != "INSUFFICIENT" || rep.OK != true || len(rep.Dates) != 0 {
		t.Fatalf("empty fold = %+v, want INSUFFICIENT/ok/no-dates", rep)
	}
	if rep.Cumulative.DollarReductionPct != nil {
		t.Fatalf("empty fold should not claim a reduction pct")
	}
}

// TestFoldAuditFullyBlindWindowHasNoReductionClaim: a window of only dollar-blind rows
// must render as "no priced counterfactual", never as a 0% reduction (#2782).
func TestFoldAuditFullyBlindWindowHasNoReductionClaim(t *testing.T) {
	rows := NewSavingsRows(blindProviderObs(), mustDay(t, "2026-07-01"))
	rep := FoldAudit(rows, mustDay(t, "2026-07-10"))
	if rep.Cumulative.DollarReductionPct != nil {
		t.Fatalf("fully-blind window claimed reduction %v, want nil", *rep.Cumulative.DollarReductionPct)
	}
	if rep.Cumulative.DollarBlindRows != 1 {
		t.Fatalf("blind rows=%d want 1", rep.Cumulative.DollarBlindRows)
	}
}

func TestLintSavingsFlagsDollarBlindRows(t *testing.T) {
	var rows []SavingsRow
	rows = append(rows, NewSavingsRows(warmProviderObs(), mustDay(t, "2026-07-01"))...)
	rows = append(rows, NewSavingsRows(blindProviderObs(), mustDay(t, "2026-07-01"))...)

	rep := LintSavings(rows, mustDay(t, "2026-07-10"))
	if rep.Verdict != "DOLLAR_BLIND" || rep.DollarBlindRows != 1 {
		t.Fatalf("lint=%+v want DOLLAR_BLIND/1", rep)
	}
	if len(rep.Findings) != 1 || rep.Findings[0].Provider != "openai" {
		t.Fatalf("finding=%+v want one openai row", rep.Findings)
	}
	if rep.BlindSavedTokenEquiv <= 0 {
		t.Fatalf("blind saved-token-equiv should be positive, got %v", rep.BlindSavedTokenEquiv)
	}

	clean := LintSavings(NewSavingsRows(warmProviderObs(), mustDay(t, "2026-07-01")), mustDay(t, "2026-07-10"))
	if clean.Verdict != "CLEAN" || clean.DollarBlindRows != 0 {
		t.Fatalf("all-priced lint=%+v want CLEAN/0", clean)
	}
}

func TestGateSavingsLosslessOrBetter(t *testing.T) {
	lossless := NewSavingsRows(warmProviderObs(), mustDay(t, "2026-07-01"))

	// All lossless → PASS.
	if g := GateSavings(lossless, SLOLosslessOrBetter, 0, mustDay(t, "2026-07-10")); !g.Pass {
		t.Fatalf("lossless-only window should PASS, got %+v", g)
	}

	// Add bounded compaction shedding; budget 0 → FAIL, budget above shed → PASS.
	withShed := append(append([]SavingsRow{}, lossless...), NewSavingsRows(compactionObs(), mustDay(t, "2026-07-01"))...)
	strict := GateSavings(withShed, SLOLosslessOrBetter, 0, mustDay(t, "2026-07-10"))
	if strict.Pass {
		t.Fatalf("shed 4000 vs budget 0 should FAIL")
	}
	if strict.BoundedShedTokens != 4000 {
		t.Fatalf("bounded shed=%d want 4000", strict.BoundedShedTokens)
	}
	if lenient := GateSavings(withShed, SLOLosslessOrBetter, 5000, mustDay(t, "2026-07-10")); !lenient.Pass {
		t.Fatalf("shed 4000 vs budget 5000 should PASS, got %+v", lenient)
	}

	// A row whose fidelity is worse than the floor (not lossless, not bounded) fails
	// regardless of budget.
	worse := []SavingsRow{{Schema: SavingsLedgerSchema, Date: "2026-07-01", Provider: "x", Mechanism: "provider_prompt_cache", Fidelity: FidelityRecoverable, CacheReadTokens: 10}}
	g := GateSavings(worse, SLOLosslessOrBetter, 1_000_000, mustDay(t, "2026-07-10"))
	if g.Pass || len(g.Violations) != 1 {
		t.Fatalf("recoverable fidelity should FAIL with one violation, got %+v", g)
	}
}

// healthyCacheRow is a warm provider-cache session: ~94% hit-rate, write-amp ~0.04.
func healthyCacheRow(genAt string) SavingsRow {
	return SavingsRow{
		Schema: SavingsLedgerSchema, Date: "2026-07-01", GeneratedAt: genAt,
		Provider: "anthropic", Mechanism: "provider_prompt_cache",
		InputTokens: 1000, CacheReadTokens: 50000, CacheCreationTokens: 2000,
	}
}

func TestFoldAuditFlagsPerSessionEfficiencyRegressions(t *testing.T) {
	// Three healthy sessions set the session-median norm (~94% hit). One churny session
	// re-writes the cache far faster than it reuses it (100 read vs 100k creation) — the
	// low-hit / high-write-amp outlier the issue says currently passes silently (#1992).
	churny := SavingsRow{
		Schema: SavingsLedgerSchema, Date: "2026-07-01", GeneratedAt: "2026-07-01T14:07:00Z",
		Provider: "anthropic", Mechanism: "provider_prompt_cache",
		InputTokens: 1000, CacheReadTokens: 100, CacheCreationTokens: 100000,
	}
	rows := []SavingsRow{
		healthyCacheRow("2026-07-01T12:57:00Z"),
		healthyCacheRow("2026-07-01T13:17:00Z"),
		healthyCacheRow("2026-07-01T13:37:00Z"),
		churny,
	}

	rep := FoldAudit(rows, mustDay(t, "2026-07-02"))
	if len(rep.EfficiencyOutliers) != 1 {
		t.Fatalf("want exactly 1 efficiency outlier (the churny session), got %d: %+v",
			len(rep.EfficiencyOutliers), rep.EfficiencyOutliers)
	}
	got := rep.EfficiencyOutliers[0]
	if got.GeneratedAt != churny.GeneratedAt {
		t.Errorf("flagged the wrong session: got %s want %s", got.GeneratedAt, churny.GeneratedAt)
	}
	if got.HitRatePct > 1.0 {
		t.Errorf("churny hit-rate should be ~0%%, got %.2f%%", got.HitRatePct)
	}
	if got.WriteAmp < sessionWriteAmpCeiling {
		t.Errorf("churny write-amp should exceed the ceiling %.2f, got %.2f", sessionWriteAmpCeiling, got.WriteAmp)
	}
	if !strings.Contains(got.Reason, "hit-rate") || !strings.Contains(got.Reason, "write-amp") {
		t.Errorf("reason should name both tripwires, got %q", got.Reason)
	}
	// The signal must actually surface, not stay silent: next-action and the rendered
	// table both carry the regression.
	if !strings.Contains(rep.NextAction, "cache-efficiency regression") {
		t.Errorf("NextAction should surface the regression, got %q", rep.NextAction)
	}
	if out := RenderAudit(rep); !strings.Contains(out, "cache-efficiency outliers") {
		t.Errorf("RenderAudit should include the outliers section:\n%s", out)
	}

	// Negative control: an all-healthy window flags nothing (no false positives) and the
	// rendered table omits the outliers section entirely.
	clean := FoldAudit(rows[:3], mustDay(t, "2026-07-02"))
	if len(clean.EfficiencyOutliers) != 0 {
		t.Errorf("all-healthy window should have no outliers, got %+v", clean.EfficiencyOutliers)
	}
	if strings.Contains(RenderAudit(clean), "cache-efficiency outliers") {
		t.Errorf("clean render should not mention outliers")
	}
}

func TestCheckTrack2QualityRejectsBelowSLOCorpus(t *testing.T) {
	rows := []QualityObservation{
		track2FidelityRow(true, true, true),
		track2FidelityRow(true, true, false),
		track2FidelityRow(true, true, false),
	}
	got := CheckTrack2Quality(rows, QualityFloor{Floor: 0.75, MinRows: 3})
	if got.Passed || got.Code != "fidelity_below_floor" {
		t.Fatalf("expected below-floor refusal, got %+v", got)
	}
	if got.EligibleRows != 3 || got.LosslessRows != 1 || got.Fidelity != 1.0/3.0 {
		t.Fatalf("unexpected fidelity fold: %+v", got)
	}
}

func TestCheckTrack2QualityPassesLosslessCorpusAndExcludesDollarBlindRows(t *testing.T) {
	rows := []QualityObservation{
		track2FidelityRow(true, true, true),
		track2FidelityRow(true, true, true),
		track2FidelityRow(false, false, false),
		track2FidelityRow(true, false, false),
	}
	got := CheckTrack2Quality(rows, QualityFloor{Floor: 1, MinRows: 2})
	if !got.Passed {
		t.Fatalf("expected lossless corpus to pass, got %+v", got)
	}
	if got.EligibleRows != 2 || got.LosslessRows != 2 || got.Fidelity != 1 {
		t.Fatalf("dollar-blind rows entered judgment: %+v", got)
	}
}

func TestCheckTrack2QualityRefusesThinCorpus(t *testing.T) {
	got := CheckTrack2Quality([]QualityObservation{track2FidelityRow(true, true, true)}, QualityFloor{Floor: 1, MinRows: 2})
	if got.Passed || got.Code != "fidelity_corpus_too_thin" {
		t.Fatalf("expected thick-corpus refusal, got %+v", got)
	}
}

func track2FidelityRow(gateDollarKnown, baselineDollarKnown, lossless bool) QualityObservation {
	return QualityObservation{ManagedDollarKnown: gateDollarKnown, BaselineDollarKnown: baselineDollarKnown, Lossless: lossless}
}
