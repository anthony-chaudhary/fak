package modver

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// moversLedger is an append-only module-versions ledger crafted so the two
// doctor sub-sections land on DISJOINT modules: internal/gateway (r5→r20, Δ+15)
// and cmd/fak (r30→r40, Δ+10) GREW over the window but were touched recently, so
// they are movers, not dormant; internal/idle and internal/quiet were stamped
// once (Δ0) with an ANCIENT last_date, so they are dormant, not movers. One line
// is a scar the fold must skip.
const moversLedger = `{"schema":"fak-module-versions/1","ts":"2026-05-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g1","last_date":"2026-05-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":20,"last_commit":"g2","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-05-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":30,"last_commit":"c1","last_date":"2026-05-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":40,"last_commit":"c2","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-15T00:00:00Z","module":"internal/idle","kind":"internal","rev":2,"last_commit":"i1","last_date":"2026-04-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-15T00:00:00Z","module":"internal/quiet","kind":"internal","rev":3,"last_commit":"q1","last_date":"2026-04-15T00:00:00Z"}
not json — a scar the fold must skip
`

func moversIssues() []OpenIssue {
	return []OpenIssue{
		{Number: 11, Title: "idle rot", Paths: []string{"internal/idle/x.go"}},
		{Number: 12, Title: "quiet bug", Paths: []string{"internal/quiet/q.go"}},
	}
}

// TestMoversRenderSection is the doctor render witness (#2472): it folds a
// fixture ledger + issue feed at a fixed `now` and asserts the rendered section
// carries both sub-sections — the fastest-moving modules (biggest delta first)
// and the dormant-with-open-issues candidates (most-dormant first) — on the
// disjoint module sets the fixture arranges.
func TestMoversRenderSection(t *testing.T) {
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	sec := Movers([]byte(moversLedger), moversIssues(), now, 0, DefaultMoversTop)

	// Top movers: gateway (Δ+15) then cmd/fak (Δ+10); the flat dormant modules
	// are excluded from the movers list.
	if len(sec.TopMovers) != 2 {
		t.Fatalf("top movers = %+v, want 2 (gateway, cmd/fak)", sec.TopMovers)
	}
	if sec.TopMovers[0].Module != "internal/gateway" || sec.TopMovers[0].RevDelta != 15 {
		t.Errorf("top mover[0] = %+v, want internal/gateway Δ15", sec.TopMovers[0])
	}
	if sec.TopMovers[1].Module != "cmd/fak" || sec.TopMovers[1].RevDelta != 10 {
		t.Errorf("top mover[1] = %+v, want cmd/fak Δ10", sec.TopMovers[1])
	}

	// Dormant: idle (105d) before quiet (91d), each with its one open issue.
	if len(sec.Dormant.Candidates) != 2 {
		t.Fatalf("dormant = %+v, want 2 (idle, quiet)", sec.Dormant.Candidates)
	}
	if sec.Dormant.Candidates[0].Module != "internal/idle" || sec.Dormant.Candidates[1].Module != "internal/quiet" {
		t.Errorf("dormant order = %s,%s, want internal/idle before internal/quiet",
			sec.Dormant.Candidates[0].Module, sec.Dormant.Candidates[1].Module)
	}

	out := sec.Render()
	for _, want := range []string{
		"== fak doctor: module movers ==",
		"top movers (rev delta over 2026-05-01..2026-07-01",
		"Δr+15  r5→r20  internal/gateway",
		"Δr+10  r30→r40  cmd/fak",
		"dormant modules with open issues (idle ≥ 30d, judged 2026-07-15):",
		"internal/idle",
		"issues: #11",
		"internal/quiet",
		"issues: #12",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered section missing %q\n---\n%s", want, out)
		}
	}
	// A mover must not leak into the dormant lines and vice versa.
	if strings.Contains(out, "internal/gateway   ") { // dormant rows pad the module name; movers do not
		t.Errorf("gateway (a mover) must not appear as a dormant row:\n%s", out)
	}

	// The section must round-trip as JSON for --json consumers.
	if _, err := json.Marshal(sec); err != nil {
		t.Fatalf("MoversSection must marshal to JSON: %v", err)
	}
}

// TestMoversTopTruncates witnesses that `top` caps BOTH sub-sections and that a
// module which did not grow never enters the movers list.
func TestMoversTopTruncates(t *testing.T) {
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	sec := Movers([]byte(moversLedger), moversIssues(), now, 0, 1)
	if len(sec.TopMovers) != 1 || sec.TopMovers[0].Module != "internal/gateway" {
		t.Errorf("top=1 movers = %+v, want only internal/gateway", sec.TopMovers)
	}
	if len(sec.Dormant.Candidates) != 1 || sec.Dormant.Candidates[0].Module != "internal/idle" {
		t.Errorf("top=1 dormant = %+v, want only internal/idle", sec.Dormant.Candidates)
	}
	// Scanned stays the full judged count even when the shown slice is capped.
	if sec.Dormant.Scanned != 2 {
		t.Errorf("Scanned = %d, want 2 (both referenced modules judged)", sec.Dormant.Scanned)
	}
}

// TestMoversEmptyLedgerHonest witnesses the zero-data degradation: an empty
// ledger and no issue feed render explicit "why empty" lines, never a bare
// blank that reads as "healthy".
func TestMoversEmptyLedgerHonest(t *testing.T) {
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	sec := Movers(nil, nil, now, 0, DefaultMoversTop)
	if len(sec.TopMovers) != 0 || len(sec.Dormant.Candidates) != 0 {
		t.Fatalf("empty ledger should yield no movers/dormant, got %+v", sec)
	}
	out := sec.Render()
	if !strings.Contains(out, "no module grew over the recorded window") {
		t.Errorf("empty movers must render an honest reason:\n%s", out)
	}
	if !strings.Contains(out, "no open-issue feed cross-referenced") {
		t.Errorf("no issue feed must render an honest reason:\n%s", out)
	}
}
