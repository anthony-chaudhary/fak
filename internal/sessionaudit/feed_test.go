package sessionaudit

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func sampleCompactReport() CompactReport {
	since := 0.25
	return CompactReport{
		Schema: "fak.sessionaudit.compact.v1",
		Scope: CompactScope{
			NamespaceFilter: "", // all
			SinceDays:       &since,
			Discovered:      44,
			Audited:         44,
			Clipped:         false,
		},
		Totals: CompactTotals{
			OutputTokens:       3983773,
			CacheReadTokens:    233176632,
			TotalContextTokens: 302773056,
			CacheReadShare:     0.77,
			IORatio:            76.0,
			EstimatedCostUSD:   1950.64,
		},
		Tiers: []CompactTier{
			{Tier: "<synthetic>", EstimatedCostUSD: 0, CostShare: 0},
			{Tier: "opus", EstimatedCostUSD: 1950.64, CostShare: 1.0, OutputTokens: 3983773},
		},
		TopLongContext: []CompactLongContext{
			{Session: "1a58399b", Namespace: "C--work-fak", TotalContextTokens: 25828218},
			{Session: "24e6e23d", Namespace: "C--work-fak", TotalContextTokens: 15958284},
		},
		Recommendations: []CompactRecommendation{
			{Kind: "long_context_pressure", Severity: "high"},
			{Kind: "opus_cost_pressure", Severity: "medium"},
			{Kind: "process_issue_pressure", Severity: "medium"},
		},
		Behavior:  &CompactBehavior{StuckSessions: 0, TimeoutKills: 0, RecurringFailures: []RecurringFailureRow{{Tool: "Bash", Sig: "x"}}},
		Confusion: &CompactConfusion{ConfusedSessions: 2, SilentConfusedSessions: 1},
	}
}

func TestFoldFeedRow(t *testing.T) {
	now := time.Date(2026, 7, 9, 22, 16, 5, 0, time.UTC)
	row := FoldFeedRow(sampleCompactReport(), now)

	if row.Schema != FeedSchema {
		t.Fatalf("schema = %q, want %q", row.Schema, FeedSchema)
	}
	if row.TS != "2026-07-09T22:16:05Z" {
		t.Fatalf("ts = %q, want the passed clock in UTC", row.TS)
	}
	if row.WindowDays != 0.25 || row.NamespaceScope != "" {
		t.Fatalf("scope = window %g ns %q, want 0.25/all", row.WindowDays, row.NamespaceScope)
	}
	if row.SessionsAudited != 44 || row.SessionsDiscovered != 44 {
		t.Fatalf("sessions = %d/%d, want 44/44", row.SessionsAudited, row.SessionsDiscovered)
	}
	if row.EstCostUSD != 1950.64 || row.CacheReadShare != 0.77 {
		t.Fatalf("totals = $%.2f cache %.2f", row.EstCostUSD, row.CacheReadShare)
	}
	// The highest-cost tier wins over the zero-cost <synthetic> tier.
	if row.TopTier != "opus" || row.TopTierCostShare != 1.0 {
		t.Fatalf("top tier = %q %.2f, want opus 1.0", row.TopTier, row.TopTierCostShare)
	}
	// The worst long-context session is the headline, not the runner-up.
	if row.LongContextMax != 25828218 || row.LongContextSession != "1a58399b" {
		t.Fatalf("long-context = %d %q, want the worst session", row.LongContextMax, row.LongContextSession)
	}
	if row.RecHigh != 1 || row.RecMedium != 2 || row.RecLow != 0 {
		t.Fatalf("rec counts = h%d m%d l%d, want 1/2/0", row.RecHigh, row.RecMedium, row.RecLow)
	}
	if row.RecurringFailureClasses != 1 {
		t.Fatalf("recurring failure classes = %d, want 1", row.RecurringFailureClasses)
	}
	if row.ConfusedSessions != 2 || row.SilentConfusedSessions != 1 {
		t.Fatalf("confusion = %d/%d, want 2/1", row.ConfusedSessions, row.SilentConfusedSessions)
	}
}

func TestFoldFeedRow_NilOptionalFolds(t *testing.T) {
	// A minimal report (no tiers, long-context, recs, behavior, confusion) must fold
	// without panicking and leave the optional fields zero.
	row := FoldFeedRow(CompactReport{Scope: CompactScope{Audited: 3}}, time.Unix(0, 0).UTC())
	if row.SessionsAudited != 3 || row.TopTier != "" || row.LongContextMax != 0 || row.RecHigh != 0 || row.ConfusedSessions != 0 {
		t.Fatalf("minimal fold leaked state: %+v", row)
	}
}

func TestAppendFeedRow_RoundTrips(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-audit.jsonl")
	now := time.Date(2026, 7, 9, 22, 16, 5, 0, time.UTC)

	// Two appends (a fresh file then an existing one) must yield exactly two JSON lines.
	for i := 0; i < 2; i++ {
		if err := AppendFeedRow(path, FoldFeedRow(sampleCompactReport(), now)); err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var lines int
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		lines++
		var back FeedRow
		if err := json.Unmarshal([]byte(line), &back); err != nil {
			t.Fatalf("line %d not valid JSON: %v (%s)", lines, err, line)
		}
		if back.Schema != FeedSchema || back.SessionsAudited != 44 {
			t.Fatalf("line %d round-trip mismatch: %+v", lines, back)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if lines != 2 {
		t.Fatalf("ledger has %d lines, want 2 (append must not truncate)", lines)
	}
}
