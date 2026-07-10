package cachevaluereport

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

var cohortNow = time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC)

// srow builds a Track-2 savings row for the cohort fold. A non-zero rebate/writePrem keeps
// the row dollar-priced (not blind); non-zero cacheRead/cacheCreate makes managed cache
// ACTIVE. NetUSD is stored directly (the cohort trusts the stored net, like FoldAudit).
func srow(date, ctx, mech string, cacheRead, cacheCreate uint64, netUSD, rebate, writePrem float64) SavingsRow {
	return SavingsRow{
		Schema:              SavingsLedgerSchema,
		Date:                date,
		SessionType:         "guard",
		Context:             ctx,
		Provider:            "anthropic",
		Mechanism:           mech,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheCreate,
		NetUSD:              netUSD,
		RebateUSD:           rebate,
		WritePremiumUSD:     writePrem,
	}
}

func TestFoldCacheDidntHelp_RanksUnhelpedWorstFirst(t *testing.T) {
	rows := []SavingsRow{
		// helped: active, net +0.40
		srow("2026-07-01", "sess-helped", "provider_prompt_cache", 5000, 100, 0.40, 1.0, 0.1),
		// unhelped cold-write: active, net -0.60, rebate < write premium
		srow("2026-07-02", "sess-cold", "provider_prompt_cache", 100, 8000, -0.60, 0.1, 0.5),
		// unhelped (not cold-write): active, net -0.20, rebate > write premium (spend drove it)
		srow("2026-07-03", "sess-spend", "provider_prompt_cache", 4000, 100, -0.20, 0.2, 0.1),
		// inactive: no cache tokens → excluded from ActiveSessions even though net<0
		srow("2026-07-04", "sess-cold-start", "provider_prompt_cache", 0, 0, -1.0, 0.0, 0.0),
		// dollar-blind: active but no $ → excluded (no net-$ to judge)
		srow("2026-07-05", "sess-blind", "provider_prompt_cache", 9000, 0, 0.0, 0.0, 0.0),
	}
	rep := FoldCacheDidntHelp(rows, cohortNow)

	if rep.Verdict != "COHORT" {
		t.Fatalf("expected COHORT verdict, got %s: %s", rep.Verdict, rep.Finding)
	}
	if rep.ActiveSessions != 3 {
		t.Fatalf("expected 3 active+priced sessions (helped + 2 unhelped), got %d", rep.ActiveSessions)
	}
	if len(rep.Unhelped) != 2 {
		t.Fatalf("expected 2 unhelped sessions, got %d: %+v", len(rep.Unhelped), rep.Unhelped)
	}
	// worst-first: -0.60 before -0.20
	if rep.Unhelped[0].NetUSD != -0.60 || rep.Unhelped[1].NetUSD != -0.20 {
		t.Fatalf("cohort not sorted most-negative-first: %v, %v", rep.Unhelped[0].NetUSD, rep.Unhelped[1].NetUSD)
	}
	if !rep.Unhelped[0].ColdWrite {
		t.Fatalf("the -0.60 session should be flagged cold-write")
	}
	if rep.Unhelped[1].ColdWrite {
		t.Fatalf("the -0.20 session (rebate > write premium) should NOT be cold-write")
	}
}

func TestFoldCacheDidntHelp_NetZeroIsUnhelped(t *testing.T) {
	// A net EXACTLY $0 active session is in the <=0 cohort — the cache paid for itself but
	// did not help.
	rep := FoldCacheDidntHelp([]SavingsRow{
		srow("2026-07-01", "sess-zero", "provider_prompt_cache", 3000, 100, 0.0, 0.3, 0.3),
	}, cohortNow)
	if rep.Verdict != "COHORT" || len(rep.Unhelped) != 1 {
		t.Fatalf("net-$0 active session must be in the cohort, got %s / %d", rep.Verdict, len(rep.Unhelped))
	}
}

func TestFoldCacheDidntHelp_CleanWhenAllPositive(t *testing.T) {
	rep := FoldCacheDidntHelp([]SavingsRow{
		srow("2026-07-01", "a", "provider_prompt_cache", 5000, 100, 0.5, 1.0, 0.1),
		srow("2026-07-02", "b", "provider_prompt_cache", 6000, 100, 0.9, 1.2, 0.1),
	}, cohortNow)
	if rep.Verdict != "CLEAN" || len(rep.Unhelped) != 0 || rep.ActiveSessions != 2 {
		t.Fatalf("all-positive corpus must be CLEAN with 2 active sessions, got %s / %d unhelped / %d active",
			rep.Verdict, len(rep.Unhelped), rep.ActiveSessions)
	}
}

func TestFoldCacheDidntHelp_InsufficientWhenNoActivePriced(t *testing.T) {
	rep := FoldCacheDidntHelp([]SavingsRow{
		// inactive (no cache tokens) and dollar-blind — neither is a judgeable active session
		srow("2026-07-01", "cold", "provider_prompt_cache", 0, 0, -1.0, 0.0, 0.0),
		srow("2026-07-02", "blind", "provider_prompt_cache", 9000, 0, 0.0, 0.0, 0.0),
	}, cohortNow)
	if rep.Verdict != "INSUFFICIENT" || rep.ActiveSessions != 0 {
		t.Fatalf("no active+priced session must be INSUFFICIENT with 0 active, got %s / %d", rep.Verdict, rep.ActiveSessions)
	}
}

func TestFoldCacheDidntHelp_MultiRowSessionSummed(t *testing.T) {
	// One session split across two mechanism rows (provider + compaction) sharing
	// (date, session_type, context): the cohort judges the SESSION whole by summing net-$.
	rows := []SavingsRow{
		srow("2026-07-01", "sess-x", "provider_prompt_cache", 4000, 200, 0.30, 0.5, 0.2), // +0.30
		srow("2026-07-01", "sess-x", "compaction", 0, 0, -0.55, 0.0, 0.0),                // -0.55 (compaction)
	}
	// Make the compaction row priced+active so the session is judged: give it a shed +$ and a fire.
	rows[1].CompactionShedTokens = 1000
	rows[1].CompactionSavedUSD = 0.05
	rows[1].CompactionFired = 1
	rep := FoldCacheDidntHelp(rows, cohortNow)
	if rep.ActiveSessions != 1 {
		t.Fatalf("the two rows are ONE session, expected 1 active session, got %d", rep.ActiveSessions)
	}
	if len(rep.Unhelped) != 1 {
		t.Fatalf("summed net (0.30 + -0.55 = -0.25) <= 0 must be in the cohort, got %d", len(rep.Unhelped))
	}
	if got := rep.Unhelped[0]; got.Rows != 2 {
		t.Fatalf("session entry should report Rows=2, got %d", got.Rows)
	}
	if net := rep.Unhelped[0].NetUSD; net > -0.24 || net < -0.26 {
		t.Fatalf("summed session net should be ~-0.25, got %v", net)
	}
}

func TestFoldCacheDidntHelp_SkipsUnparseableDate(t *testing.T) {
	rep := FoldCacheDidntHelp([]SavingsRow{
		srow("not-a-date", "x", "provider_prompt_cache", 5000, 0, -0.5, 0.1, 0.4),
	}, cohortNow)
	if rep.Verdict != "INSUFFICIENT" || rep.ActiveSessions != 0 {
		t.Fatalf("an unparseable-date row must be skipped, got %s / %d active", rep.Verdict, rep.ActiveSessions)
	}
}

func TestScoreCacheDidntHelpFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "savings.jsonl")
	rows := []SavingsRow{
		srow("2026-07-01", "helped", "provider_prompt_cache", 5000, 100, 0.4, 1.0, 0.1),
		srow("2026-07-02", "cold", "provider_prompt_cache", 100, 8000, -0.7, 0.1, 0.6),
	}
	var lines []string
	for _, r := range rows {
		line, err := AppendSavingsLine(r)
		if err != nil {
			t.Fatalf("marshal savings row: %v", err)
		}
		lines = append(lines, line)
	}
	if err := os.WriteFile(path, []byte(joinLines(lines)), 0o644); err != nil {
		t.Fatalf("write seeded ledger: %v", err)
	}
	rep := ScoreCacheDidntHelpFile(path, cohortNow)
	if rep.Verdict != "COHORT" || len(rep.Unhelped) != 1 || rep.Unhelped[0].Context != "cold" {
		t.Fatalf("seeded ledger should surface the one cold session, got %s / %+v", rep.Verdict, rep.Unhelped)
	}

	missing := ScoreCacheDidntHelpFile(filepath.Join(dir, "nope.jsonl"), cohortNow)
	if missing.Verdict != "INSUFFICIENT" {
		t.Fatalf("missing ledger must fall open INSUFFICIENT, got %s", missing.Verdict)
	}
}

func joinLines(lines []string) string {
	out := ""
	for _, l := range lines {
		out += l + "\n"
	}
	return out
}
