package qaprocessscore

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/brittleness"
	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// chronic builds the canonical over-budget fixture: one identity that flaked 3x (> the default
// budget of 2) across 2 distinct trees (>= the default soak), plus one identity that flaked twice
// on a single tree -- loud, but neither over budget nor soaked, so it must stay SOFT.
func chronic() []FlakeObservation {
	return []FlakeObservation{
		{Pkg: "github.com/anthony-chaudhary/fak/internal/flaky", Test: "TestRacy", Tree: "aaa1111"},
		{Pkg: "github.com/anthony-chaudhary/fak/internal/flaky", Test: "TestRacy", Tree: "aaa1111"},
		{Pkg: "github.com/anthony-chaudhary/fak/internal/flaky", Test: "TestRacy", Tree: "bbb2222"},
		{Pkg: "github.com/anthony-chaudhary/fak/internal/steady", Tree: "aaa1111"},
		{Pkg: "github.com/anthony-chaudhary/fak/internal/steady", Tree: "aaa1111"},
	}
}

// TestOverBudgetFlakeIsQuarantined is the first half of the issue's done condition: a flaky test
// over budget appears in the ledger, keyed by its per-test identity, with its soak evidence.
func TestOverBudgetFlakeIsQuarantined(t *testing.T) {
	l := FoldQuarantineLedger(chronic(), 0, 0) // zero-value dials -> documented defaults
	if l.Budget != DefaultRerunBudget || l.Soak != DefaultSoakTrees {
		t.Fatalf("zero dials must fall back to defaults, got budget=%d soak=%d", l.Budget, l.Soak)
	}
	if len(l.Entries) != 2 {
		t.Fatalf("got %d entries want 2 (%+v)", len(l.Entries), l.Entries)
	}
	// Worst-first: the 3-flake identity leads.
	got := l.Entries[0]
	if got.ID != "github.com/anthony-chaudhary/fak/internal/flaky.TestRacy" {
		t.Fatalf("ledger must key on per-test identity, got %q", got.ID)
	}
	if got.Flakes != 3 {
		t.Errorf("Flakes = %d want 3 (every observation counts)", got.Flakes)
	}
	if len(got.Trees) != 2 || got.Trees[0] != "aaa1111" || got.Trees[1] != "bbb2222" {
		t.Errorf("Trees must de-duplicate and sort the soak stamps, got %v", got.Trees)
	}
	if !got.Quarantined {
		t.Errorf("3 flakes over 2 trees must be quarantined at budget=2/soak=2")
	}

	over := l.OverBudget()
	if len(over) != 1 || over[0].ID != got.ID {
		t.Fatalf("OverBudget = %+v, want exactly the chronic identity", over)
	}
}

// TestBudgetAndSoakBothGate pins that NEITHER dial alone quarantines. Loud-on-one-tree is a
// transient (possibly one broken commit); once-per-tree across many trees is under budget. Only
// chronic non-determinism -- over budget AND soaked -- may become HARD debt.
func TestBudgetAndSoakBothGate(t *testing.T) {
	burst := []FlakeObservation{ // 4 flakes, ONE tree: over budget, not soaked
		{Pkg: "m/burst", Tree: "aaa1111"}, {Pkg: "m/burst", Tree: "aaa1111"},
		{Pkg: "m/burst", Tree: "aaa1111"}, {Pkg: "m/burst", Tree: "aaa1111"},
	}
	if q := FoldQuarantineLedger(burst, 0, 0).OverBudget(); len(q) != 0 {
		t.Errorf("a one-tree burst must not gate (no soak), got %+v", q)
	}

	spread := []FlakeObservation{ // 2 flakes, TWO trees: soaked, not over budget
		{Pkg: "m/spread", Tree: "aaa1111"}, {Pkg: "m/spread", Tree: "bbb2222"},
	}
	if q := FoldQuarantineLedger(spread, 0, 0).OverBudget(); len(q) != 0 {
		t.Errorf("2 flakes must sit within a budget of 2 (strictly-greater gate), got %+v", q)
	}

	// The dials are honored when supplied: budget 1 turns the same spread into debt.
	if q := FoldQuarantineLedger(spread, 1, 2).OverBudget(); len(q) != 1 {
		t.Errorf("budget=1 must quarantine the soaked 2-flake identity, got %+v", q)
	}
}

// TestQuarantineIsHardQAProcessDebt is the second half of the done condition: an over-budget flake
// surfaces as HARD flake_quarantine debt on the card, while a within-budget flake stays SOFT and
// cannot gate.
func TestQuarantineIsHardQAProcessDebt(t *testing.T) {
	kpi := FlakeQuarantine(FoldQuarantineLedger(chronic(), 0, 0))
	if kpi.Key != FlakeQuarantineKey {
		t.Fatalf("KPI key = %q want %q", kpi.Key, FlakeQuarantineKey)
	}
	if len(kpi.Defects) != 1 {
		t.Fatalf("want exactly 1 HARD defect, got %d (%v)", len(kpi.Defects), kpi.Defects)
	}
	if !strings.HasPrefix(kpi.Defects[0], flakeOverBudgetClass+" ") {
		t.Errorf("defect must lead with the closed-vocabulary token: %q", kpi.Defects[0])
	}
	if len(kpi.Soft) != 1 || !strings.Contains(kpi.Soft[0], "internal/steady") {
		t.Errorf("the within-budget identity must ride SOFT, got %v", kpi.Soft)
	}

	// The card gates on Σ len(kpi.Defects): the quarantined flake is one unit of qa_process_debt.
	debt := scorecard.IntValue(Compose([]scorecard.KPI{kpi}).Corpus[DebtKey])
	if debt != 1 {
		t.Fatalf("qa_process_debt = %d want 1 (the quarantined flake must gate)", debt)
	}

	// An empty ledger is an honest 100 with no debt -- the fold never manufactures unmeasured debt.
	cleanKPI := FlakeQuarantine(FoldQuarantineLedger(nil, 0, 0))
	if cleanKPI.Score != 100 || len(cleanKPI.Defects) != 0 {
		t.Errorf("empty ledger = score %v defects %v, want 100/none", cleanKPI.Score, cleanKPI.Defects)
	}
}

// TestHistoryNeverFeedsTheGate pins the issue's out-of-scope line: brittleness stays advisory for
// landed history. Only FLAKY_RETRY_PASS (a current, rerun-masked flake) becomes an observation;
// RECURRING_FIX and REVERTED_LANDING are history and must never reach this HARD gate.
func TestHistoryNeverFeedsTheGate(t *testing.T) {
	findings := []brittleness.Finding{
		{Class: brittleness.ClassFlakyRetryPass, Ref: "m/flaky", Weight: 2, Fresh: []string{"aaa1111", "bbb2222"}},
		{Class: brittleness.ClassRecurringFix, Ref: "internal/x/x.go", Weight: 9, Fresh: []string{"ccc3333"}},
		{Class: brittleness.ClassRevertedLanding, Ref: "ddd4444", Weight: 4},
	}
	obs := ObservationsFromFindings(findings)
	if len(obs) != 2 {
		t.Fatalf("only the FLAKY_RETRY_PASS weight may become observations, got %+v", obs)
	}
	for _, o := range obs {
		if o.Pkg != "m/flaky" {
			t.Fatalf("history leaked into the ledger: %+v", o)
		}
	}
	// The finding's two stamps spread across its weight, so the soak is preserved, not collapsed.
	l := FoldQuarantineLedger(obs, 0, 0)
	if len(l.Entries) != 1 || len(l.Entries[0].Trees) != 2 {
		t.Fatalf("weight must spread over the captured stamps, got %+v", l.Entries)
	}
}

// TestLedgerJSONLRoundTrip pins the durable wire format: rows written by one run decode back
// identically in the next, which is what makes the soak window accumulate across runs.
func TestLedgerJSONLRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := EncodeObservations(&buf, chronic()); err != nil {
		t.Fatalf("encode: %v", err)
	}
	// Append a second run's rows to the same stream -- the ledger is append-only.
	if err := EncodeObservations(&buf, []FlakeObservation{{Pkg: "m/late", Tree: "ccc3333"}}); err != nil {
		t.Fatalf("append: %v", err)
	}
	got, err := DecodeObservations(&buf)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(chronic())+1 {
		t.Fatalf("round trip lost rows: got %d want %d", len(got), len(chronic())+1)
	}
	if got[0] != chronic()[0] {
		t.Errorf("row 0 drifted: %+v vs %+v", got[0], chronic()[0])
	}

	// A malformed line must ERROR, never silently shrink the ledger (that would fail open).
	if _, err := DecodeObservations(strings.NewReader("{\"pkg\":\"m/a\"}\nnot json\n")); err == nil {
		t.Errorf("a malformed ledger line must be an error, not a silent drop")
	}
}

// TestDeflakeDispatchIsOneDedupedTicketPerTest is the third half of the done condition: a dry-run
// dispatch plans exactly one create per quarantined test, and a re-run whose flake counts have
// drifted updates that same ticket in place instead of opening a duplicate.
func TestDeflakeDispatchIsOneDedupedTicketPerTest(t *testing.T) {
	gaps := FlakeQuarantineGaps(FoldQuarantineLedger(chronic(), 0, 0))
	if len(gaps) != 1 {
		t.Fatalf("want exactly one gap per quarantined test, got %d (%+v)", len(gaps), gaps)
	}
	if gaps[0].Ref != "github.com/anthony-chaudhary/fak/internal/flaky.TestRacy" {
		t.Errorf("gap must anchor on the per-test identity, got %q", gaps[0].Ref)
	}
	// Routing trims the module prefix so the ticket lands in the lane that owns the files.
	if gaps[0].Paths[0] != "internal/flaky/**" {
		t.Errorf("gap must route to the owning package tree, got %v", gaps[0].Paths)
	}

	items := ActionItems(gaps, "fak score qa-process --json")
	opt := dogfoodissues.BuildOptions{DedupeChecked: true, DedupeCap: 10, ParentBaseline: 40, CompletionStandard: "development"}
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, nil, opt)
	if len(skipped) != 0 {
		t.Fatalf("the deflake item must pass the dispatchability review, skipped=%+v", skipped)
	}
	if len(plan) != 1 || plan[0].Action != "create" {
		t.Fatalf("dry run must plan exactly one create row, got %+v", plan)
	}

	// Re-run with MORE flakes on a third tree: the Detail drifts, the key must not.
	worse := append(chronic(), FlakeObservation{
		Pkg: "github.com/anthony-chaudhary/fak/internal/flaky", Test: "TestRacy", Tree: "ccc3333"})
	worseGaps := FlakeQuarantineGaps(FoldQuarantineLedger(worse, 0, 0))
	if worseGaps[0].Key() != gaps[0].Key() {
		t.Fatalf("dedup key must be stable across flake-count drift: %q != %q", worseGaps[0].Key(), gaps[0].Key())
	}
	if worseGaps[0].Detail == gaps[0].Detail {
		t.Fatalf("fixture bug: the second run should render a different Detail (%q)", gaps[0].Detail)
	}

	existing := []dogfoodissues.Issue{{
		Number: 4242, Title: items[0].Title, Body: dogfoodissues.IssueBody(items[0]), State: "open",
	}}
	plan2, _ := dogfoodissues.BuildPlanWithOptions(ActionItems(worseGaps, "fak score qa-process --json"), existing, opt)
	if len(plan2) != 1 || plan2[0].Action != "update" {
		t.Fatalf("a still-flaky rerun must UPDATE the same ticket, got %+v", plan2)
	}
}

// TestDeflakeItemTellsItsWorkerTheFix pins that the generated ticket carries flake-specific
// guidance (deflake, do not raise the budget) rather than the generic qa-process fallback.
func TestDeflakeItemTellsItsWorkerTheFix(t *testing.T) {
	gaps := FlakeQuarantineGaps(FoldQuarantineLedger(chronic(), 0, 0))
	it := gaps[0].ToActionItem("fak score qa-process --json")
	if !strings.Contains(it.NextAction, "deterministic") || !strings.Contains(it.NextAction, "rerun budget") {
		t.Errorf("next action must name the real fix and forbid raising the budget: %q", it.NextAction)
	}
	if !strings.Contains(it.WorkingSpine, "-race") {
		t.Errorf("working spine must point at reproducing the non-determinism: %q", it.WorkingSpine)
	}
	if !strings.Contains(it.Key, FlakeQuarantineKey) {
		t.Errorf("key must be namespaced by the KPI: %q", it.Key)
	}
}
