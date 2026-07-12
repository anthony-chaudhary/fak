package modver

import (
	"encoding/json"
	"testing"
	"time"
)

// dormantLedger is an append-only module-versions ledger with the scars a fleet
// leaves: internal/gateway is stamped twice OUT OF timestamp order (the newest
// stamp is written first, not last) to witness that the finder ages a module by
// its newest-TS last_date rather than trusting file order; one line is not JSON
// and one module (internal/badly) carries an unparseable last_date. Judged
// against now = 2026-07-01 with the default 30-day window: internal/idle (last
// touch 2026-04-01, 91d) and internal/gateway (2026-05-01, 61d) are dormant;
// cmd/fak (2026-06-25, 6d) is fresh; internal/alpha and internal/beta share
// gateway's 61d idle to exercise the tie-breaks.
const dormantLedger = `{"schema":"fak-module-versions/1","ts":"2026-06-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":8,"last_commit":"g2","last_date":"2026-05-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-05-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g1","last_date":"2026-04-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-15T00:00:00Z","module":"internal/idle","kind":"internal","rev":2,"last_commit":"i1","last_date":"2026-04-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-25T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":40,"last_commit":"c1","last_date":"2026-06-25T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-10T00:00:00Z","module":"internal/alpha","kind":"internal","rev":3,"last_commit":"a1","last_date":"2026-05-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-10T00:00:00Z","module":"internal/beta","kind":"internal","rev":3,"last_commit":"b1","last_date":"2026-05-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-06-10T00:00:00Z","module":"internal/badly","kind":"internal","rev":1,"last_commit":"x1","last_date":"garbage-date"}
not json — a scar the fold must skip
`

// dormantIssues pairs open issues with the paths they reference. #10 names two
// gateway paths (must dedupe to one ref); #80 names only an unknown-scope module
// and non-module paths (must contribute nothing); #40 names fresh cmd/fak; #70
// names the unparseable-date module.
func dormantIssues() []OpenIssue {
	return []OpenIssue{
		{Number: 10, Title: "gateway leak", Paths: []string{"internal/gateway/a.go", "internal/gateway/b.go"}},
		{Number: 20, Title: "gateway retry", Paths: []string{"internal/gateway/c.go"}},
		{Number: 30, Title: "idle rot", Paths: []string{"internal/idle/x.go"}},
		{Number: 40, Title: "fresh work", Paths: []string{"cmd/fak/main.go"}},
		{Number: 50, Title: "alpha bug", Paths: []string{"internal/alpha/a.go"}},
		{Number: 60, Title: "beta bug", Paths: []string{"internal/beta/b.go"}},
		{Number: 70, Title: "badly dated", Paths: []string{"internal/badly/z.go"}},
		{Number: 80, Title: "no home", Paths: []string{"internal/notracked/n.go", "README.md", "docs/generation.md"}},
	}
}

func TestDormantJoinsLedgerAndIssues(t *testing.T) {
	now := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	rep := Dormant([]byte(dormantLedger), dormantIssues(), now, 0)

	if rep.Now != "2026-07-01T00:00:00Z" {
		t.Errorf("Now = %q, want 2026-07-01T00:00:00Z", rep.Now)
	}
	if rep.Days != DefaultDormantDays {
		t.Errorf("Days = %d, want default %d for a non-positive threshold", rep.Days, DefaultDormantDays)
	}
	// Scanned counts every ledger-known module an open issue referenced and the
	// finder actually judged: gateway, idle, cmd/fak, alpha, beta, badly = 6.
	// internal/notracked (#80) has no ledger row, so it never enters the judged
	// set — dormancy is not judgeable without a last-touch date.
	if rep.Scanned != 6 {
		t.Errorf("Scanned = %d, want 6 ledger-known referenced modules", rep.Scanned)
	}

	// Candidates are dormant-only, most-dormant first; cmd/fak (fresh) and
	// internal/badly (unparseable date) are judged but excluded.
	got := make([]string, len(rep.Candidates))
	for i, c := range rep.Candidates {
		got[i] = c.Module
	}
	want := []string{"internal/idle", "internal/gateway", "internal/alpha", "internal/beta"}
	if len(got) != len(want) {
		t.Fatalf("candidates = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidate order = %v, want %v (days-idle desc, then issue-count desc, then name asc)", got, want)
		}
	}

	// internal/idle is the most dormant: 91 days idle, one issue.
	idle := rep.Candidates[0]
	if idle.DaysIdle != 91 || idle.IssueCount != 1 || idle.Rev != 2 || idle.Kind != "internal" {
		t.Errorf("idle = %+v, want 91d idle, 1 issue, rev 2, kind internal", idle)
	}
	if idle.LastDate != "2026-04-01T00:00:00Z" {
		t.Errorf("idle.LastDate = %q, want 2026-04-01T00:00:00Z", idle.LastDate)
	}

	// internal/gateway: aged by its NEWEST-TS stamp (last_date 2026-05-01 → 61d,
	// rev 8), NOT the out-of-order older stamp (2026-04-01, rev 5). Its two open
	// issues survive the per-module dedupe of #10's two gateway paths, sorted by
	// number.
	gw := rep.Candidates[1]
	if gw.DaysIdle != 61 || gw.Rev != 8 || gw.LastDate != "2026-05-01T00:00:00Z" {
		t.Errorf("gateway = %+v, want 61d idle, rev 8, last_date 2026-05-01 (newest stamp wins)", gw)
	}
	if gw.IssueCount != 2 || len(gw.Issues) != 2 || gw.Issues[0].Number != 10 || gw.Issues[1].Number != 20 {
		t.Errorf("gateway issues = %+v, want deduped [10,20]", gw.Issues)
	}

	// Tie at 61 days idle: gateway (2 issues) outranks alpha/beta (1 each) by
	// issue count; alpha before beta by name.
	if a, b := rep.Candidates[2], rep.Candidates[3]; a.Module != "internal/alpha" || b.Module != "internal/beta" {
		t.Errorf("61d tie order = [%s %s], want alpha before beta", a.Module, b.Module)
	}

	// The readout is the dispatcher's fuel feed: it must round-trip as JSON.
	blob, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("report must marshal to JSON for the dispatcher: %v", err)
	}
	var back DormantReport
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatalf("report JSON must round-trip: %v", err)
	}
	if len(back.Candidates) != len(rep.Candidates) {
		t.Errorf("round-trip lost candidates: got %d, want %d", len(back.Candidates), len(rep.Candidates))
	}
}

// TestDormantThresholdKnob witnesses that widening the window past a module's
// idle age drops it: at 90 days only internal/idle (91d) stays dormant; the 61d
// modules fall inside the window.
func TestDormantThresholdKnob(t *testing.T) {
	now := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	rep := Dormant([]byte(dormantLedger), dormantIssues(), now, 90)
	if rep.Days != 90 {
		t.Errorf("Days = %d, want the explicit 90", rep.Days)
	}
	if len(rep.Candidates) != 1 || rep.Candidates[0].Module != "internal/idle" {
		t.Fatalf("at a 90-day window, candidates = %+v, want only internal/idle", rep.Candidates)
	}
}

// TestDormantFreshCorpusEmpty witnesses the zero-behaviour case: when every
// referenced module was touched inside the window, the finder surfaces no fuel.
func TestDormantFreshCorpusEmpty(t *testing.T) {
	now := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	// One issue naming a module whose only stamp is one day old.
	ledger := `{"schema":"fak-module-versions/1","ts":"2026-06-30T00:00:00Z","module":"internal/hot","kind":"internal","rev":9,"last_commit":"h1","last_date":"2026-06-30T00:00:00Z"}`
	issues := []OpenIssue{{Number: 1, Paths: []string{"internal/hot/z.go"}}}
	rep := Dormant([]byte(ledger), issues, now, 0)
	if rep.Scanned != 1 {
		t.Errorf("Scanned = %d, want 1 (the fresh module was judged)", rep.Scanned)
	}
	if len(rep.Candidates) != 0 {
		t.Errorf("candidates = %+v, want none on a fresh corpus", rep.Candidates)
	}
}
