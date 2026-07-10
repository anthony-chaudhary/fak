package turnbench

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestFanLedgerRowFromSweep: the cadence row is extracted at the HEADLINE (widest) cell
// of a real sweep, carries the measured reuse figures, and round-trips as one JSONL line.
func TestFanLedgerRowFromSweep(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	agents := []int{1, 4, 16}
	subTurns := []int{2, 4}
	prefixes := []int{1024, 8192}
	sw := RunFanoutPrefixSweep(ctx, FanoutResearch, agents, subTurns, prefixes, 4, 5, cm, nil)

	row, ok := FanLedgerRowFromSweep(sw, "2026-07-10T08:00:00Z")
	if !ok {
		t.Fatal("FanLedgerRowFromSweep reported no headline cell for a non-empty sweep")
	}
	// Headline = widest fan-out: max agents, then max prefix, then max sub-turns.
	if row.Agents != 16 || row.PrefixTokens != 8192 || row.SubTurns != 4 {
		t.Fatalf("headline cell = N=%d prefix=%d sub_turns=%d, want N=16 prefix=8192 sub_turns=4", row.Agents, row.PrefixTokens, row.SubTurns)
	}
	if row.Schema != FanLedgerSchema {
		t.Fatalf("row schema=%q, want %q", row.Schema, FanLedgerSchema)
	}
	if row.Profile != FanoutResearch.Name {
		t.Fatalf("row profile=%q, want %q", row.Profile, FanoutResearch.Name)
	}
	if row.AppVersion == "" {
		t.Fatal("row app_version is empty")
	}
	// The measured prefix-reuse geometry is exact: (N−1)·prefix.
	if want := (row.Agents - 1) * row.PrefixTokens; row.PrefixTokensSaved != want {
		t.Fatalf("prefix_tokens_saved=%d, want (N-1)*prefix=%d", row.PrefixTokensSaved, want)
	}

	// One JSONL line, no embedded newline, and it round-trips.
	line := row.JSONL()
	if bytes.Contains(line, []byte("\n")) {
		t.Fatal("JSONL row contains a newline")
	}
	var back FanLedgerRow
	if err := json.Unmarshal(line, &back); err != nil {
		t.Fatalf("JSONL row does not round-trip: %v", err)
	}
	if back != row {
		t.Fatalf("round-tripped row differs:\n got %+v\nwant %+v", back, row)
	}
}

// TestFoldFanLedgerTrend: folding a multi-run ledger orders rows by date, computes the
// first→last reuse delta, and classifies the direction — the visible trend fold.
func TestFoldFanLedgerTrend(t *testing.T) {
	// Deliberately out of date order to prove the fold sorts.
	rows := []FanLedgerRow{
		{Schema: FanLedgerSchema, Date: "2026-07-09T08:00:00Z", Profile: "research-goal", Agents: 1000, CrossUpliftP50: 40, PrefixTokensSaved: 2046000},
		{Schema: FanLedgerSchema, Date: "2026-07-07T08:00:00Z", Profile: "research-goal", Agents: 1000, CrossUpliftP50: 30, PrefixTokensSaved: 2046000},
		{Schema: FanLedgerSchema, Date: "2026-07-08T08:00:00Z", Profile: "research-goal", Agents: 1000, CrossUpliftP50: 35, PrefixTokensSaved: 2046000},
	}
	var buf bytes.Buffer
	for _, r := range rows {
		buf.Write(r.JSONL())
		buf.WriteByte('\n')
	}
	// Blank lines must be tolerated.
	buf.WriteString("\n")

	tr, err := FoldFanLedger(&buf)
	if err != nil {
		t.Fatalf("FoldFanLedger: %v", err)
	}
	if tr.Count != 3 {
		t.Fatalf("count=%d, want 3", tr.Count)
	}
	if tr.First.Date != "2026-07-07T08:00:00Z" || tr.Last.Date != "2026-07-09T08:00:00Z" {
		t.Fatalf("rows not date-ordered: first=%s last=%s", tr.First.Date, tr.Last.Date)
	}
	if tr.CrossUpliftDelta != 10 { // 40 - 30
		t.Fatalf("cross_uplift_delta=%d, want 10", tr.CrossUpliftDelta)
	}
	if tr.Direction != FanTrendRising {
		t.Fatalf("direction=%q, want %q", tr.Direction, FanTrendRising)
	}
	// The rendered card shows the dated figures and the delta line.
	card := tr.Render()
	if !strings.Contains(card, "2026-07-08T08:00:00Z") || !strings.Contains(card, "Δ first→last") {
		t.Fatalf("render missing a dated row or the delta line:\n%s", card)
	}
	// The JSON artifact is non-empty and re-parses.
	var reparse FanTrend
	if err := json.Unmarshal(tr.JSON(), &reparse); err != nil {
		t.Fatalf("trend JSON does not re-parse: %v", err)
	}
}

// TestFoldFanLedgerDirections: single/flat/regressing verdicts, and the empty ledger folds
// cleanly (a first-ever run must render, not error).
func TestFoldFanLedgerDirections(t *testing.T) {
	// Empty ledger.
	tr, err := FoldFanLedger(strings.NewReader(""))
	if err != nil {
		t.Fatalf("empty ledger errored: %v", err)
	}
	if tr.Count != 0 || tr.Direction != FanTrendSingle {
		t.Fatalf("empty ledger: count=%d direction=%q, want 0/single", tr.Count, tr.Direction)
	}
	if !strings.Contains(tr.Render(), "ledger empty") {
		t.Fatalf("empty ledger render missing baseline note:\n%s", tr.Render())
	}

	// Single row → single.
	one := FanLedgerRow{Schema: FanLedgerSchema, Date: "2026-07-07T08:00:00Z", CrossUpliftP50: 30}
	tr = mustFold(t, one)
	if tr.Direction != FanTrendSingle {
		t.Fatalf("single row direction=%q, want single", tr.Direction)
	}

	// Flat: equal endpoints.
	tr = mustFold(t,
		FanLedgerRow{Schema: FanLedgerSchema, Date: "2026-07-07T08:00:00Z", CrossUpliftP50: 30},
		FanLedgerRow{Schema: FanLedgerSchema, Date: "2026-07-08T08:00:00Z", CrossUpliftP50: 30},
	)
	if tr.Direction != FanTrendFlat || tr.CrossUpliftDelta != 0 {
		t.Fatalf("flat: direction=%q delta=%d, want flat/0", tr.Direction, tr.CrossUpliftDelta)
	}

	// Regressing: reuse dropped.
	tr = mustFold(t,
		FanLedgerRow{Schema: FanLedgerSchema, Date: "2026-07-07T08:00:00Z", CrossUpliftP50: 40},
		FanLedgerRow{Schema: FanLedgerSchema, Date: "2026-07-08T08:00:00Z", CrossUpliftP50: 25},
	)
	if tr.Direction != FanTrendRegressing || tr.CrossUpliftDelta != -15 {
		t.Fatalf("regressing: direction=%q delta=%d, want regressing/-15", tr.Direction, tr.CrossUpliftDelta)
	}

	// A malformed line is a hard error, not a silently dropped row.
	if _, err := FoldFanLedger(strings.NewReader("{not json}\n")); err == nil {
		t.Fatal("malformed ledger line did not error")
	}
}

func mustFold(t *testing.T, rows ...FanLedgerRow) FanTrend {
	t.Helper()
	var buf bytes.Buffer
	for _, r := range rows {
		buf.Write(r.JSONL())
		buf.WriteByte('\n')
	}
	tr, err := FoldFanLedger(&buf)
	if err != nil {
		t.Fatalf("FoldFanLedger: %v", err)
	}
	return tr
}
