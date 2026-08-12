package main

// trajctl_depth_test.go — the behavioral floor for the depth gate (DEPTH_NOT_CARRIED).
// These run the real CLI entry points against a real ledger file, so they pin the
// wiring, not just the pure fold in internal/depthadmit.

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/depthadmit"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// declarePlannedObjective writes an objective with a three-phase plan straight to
// the ledger, bypassing the declare CLI so these tests pin the close/depth path
// only.
func declarePlannedObjective(t *testing.T, ledger, id string) {
	t.Helper()
	obj := trajctl.Objective{
		ID:        id,
		Statement: "carry the " + id + " line to its declared end",
		Plan: []trajctl.PlanPhase{
			{ID: "p1", Title: "declare the fold"},
			{ID: "p2", Title: "wire the gate"},
			{ID: "p3", Title: "register the reason"},
		},
		Status: trajctl.StatusActive,
	}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append objective: %v", err)
	}
}

// witnessPhases appends the W3 commit-progress row the depth read trusts: one
// evidence ref per phase whose commit resolved, exactly the shape
// trajctl.CommitProgressScorer writes.
func witnessPhases(t *testing.T, ledger, id string, phases ...string) {
	t.Helper()
	row := trajctl.ScoreRow{
		ObjectiveID: id,
		Value:       float64(len(phases)) / 3.0,
		Method:      trajctl.CommitScorerMethod,
		Version:     trajctl.CommitScorerVersion,
		Witness:     trajctl.W3,
	}
	for i, p := range phases {
		row.Evidence = append(row.Evidence, trajctl.EvidenceRef{
			Kind: "commit", Ref: strings.Repeat(string(rune('a'+i)), 40), Detail: p,
		})
	}
	if err := trajctl.Append(ledger, trajctl.ScoreRecord(row)); err != nil {
		t.Fatalf("append score: %v", err)
	}
}

// The hole this gate closes: before it, `close --status met` on a three-phase plan
// with one phase witnessed wrote `met` and nothing objected.
func TestTrajctlCloseMetRefusesAnUncarriedPlan(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "deep-line")
	witnessPhases(t, ledger, "deep-line", "p1")

	var out, errb bytes.Buffer
	code := runTrajctl(&out, &errb, []string{"close", "--id", "deep-line", "--ledger", ledger})
	if code == 0 {
		t.Fatalf("close met on a 1-of-3 plan succeeded; want refusal. stdout=%q", out.String())
	}
	stderr := errb.String()
	if !strings.Contains(stderr, depthadmit.RefusalReason) {
		t.Errorf("stderr %q does not carry the closed token %q", stderr, depthadmit.RefusalReason)
	}
	// The refusal must name the concrete next step, not just say no.
	if !strings.Contains(stderr, "p2") || !strings.Contains(stderr, "wire the gate") {
		t.Errorf("stderr %q does not name the frontier phase p2 / its title", stderr)
	}
	if !strings.Contains(stderr, "--force") || !strings.Contains(stderr, "abandoned") {
		t.Errorf("stderr %q does not offer the escapes (--force / --status abandoned)", stderr)
	}
	// A refused close must not have mutated the ledger.
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	if got := st.Objectives["deep-line"].Status; got != trajctl.StatusActive {
		t.Errorf("status = %q after a refused close, want it left %q", got, trajctl.StatusActive)
	}
}

// `met` with NO plan declared is refused too: it claims a depth nobody can check.
func TestTrajctlCloseMetRefusesAnUndeclaredPlan(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	obj := trajctl.Objective{ID: "no-plan", Statement: "vibes", Status: trajctl.StatusActive}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append: %v", err)
	}
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"close", "--id", "no-plan", "--ledger", ledger}); code == 0 {
		t.Fatalf("close met with no plan succeeded; want refusal. stdout=%q", out.String())
	}
	if !strings.Contains(errb.String(), "no plan is declared") {
		t.Errorf("stderr %q does not explain the missing plan", errb.String())
	}
}

// A fully witnessed plan closes clean — the gate is a depth check, not a wall.
func TestTrajctlCloseMetAdmitsACarriedPlan(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "deep-line")
	witnessPhases(t, ledger, "deep-line", "p1", "p2", "p3")

	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"close", "--id", "deep-line", "--ledger", ledger}); code != 0 {
		t.Fatalf("close = %d on a carried plan, want 0. stderr=%q", code, errb.String())
	}
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	if got := st.Objectives["deep-line"].Status; got != trajctl.StatusMet {
		t.Errorf("status = %q, want %q", got, trajctl.StatusMet)
	}
}

// --force closes shallow anyway, and says so on stderr: an escape, never a silent one.
func TestTrajctlCloseForceOverridesLoudly(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "deep-line")
	witnessPhases(t, ledger, "deep-line", "p1")

	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"close", "--id", "deep-line", "--ledger", ledger, "--force"}); code != 0 {
		t.Fatalf("forced close = %d, want 0. stderr=%q", code, errb.String())
	}
	if !strings.Contains(errb.String(), "--force overrode") ||
		!strings.Contains(errb.String(), depthadmit.RefusalReason) {
		t.Errorf("stderr %q does not announce the override with its token", errb.String())
	}
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	if got := st.Objectives["deep-line"].Status; got != trajctl.StatusMet {
		t.Errorf("status = %q after --force, want %q", got, trajctl.StatusMet)
	}
}

// Abandoning is never gated — it claims nothing — but the depth reached is still
// on the record.
func TestTrajctlCloseAbandonIsNeverGated(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "dead-line")

	var out, errb bytes.Buffer
	code := runTrajctl(&out, &errb, []string{"close", "--id", "dead-line", "--status", "abandoned", "--ledger", ledger})
	if code != 0 {
		t.Fatalf("abandon = %d, want 0. stderr=%q", code, errb.String())
	}
	if strings.Contains(errb.String(), depthadmit.RefusalReason) {
		t.Errorf("abandon emitted a depth refusal: %q", errb.String())
	}
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	if got := st.Objectives["dead-line"].Status; got != trajctl.StatusAbandoned {
		t.Errorf("status = %q, want %q", got, trajctl.StatusAbandoned)
	}
}

// `fak trajctl depth` is the handoff surface: it names the frontier so a successor
// resumes mid-line instead of re-planning from the top.
func TestTrajctlDepthReportsTheFrontier(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "deep-line")
	witnessPhases(t, ledger, "deep-line", "p1")

	var out, errb bytes.Buffer
	if code := runTrajctlDepth(&out, &errb, []string{"--id", "deep-line", "--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("depth = %d, stderr=%q", code, errb.String())
	}
	var got []struct {
		ObjectiveID string            `json:"objective_id"`
		Report      depthadmit.Report `json:"report"`
		Handoff     string            `json:"handoff"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal %q: %v", out.String(), err)
	}
	if len(got) != 1 {
		t.Fatalf("entries = %d, want 1", len(got))
	}
	e := got[0]
	if e.Report.Verdict != depthadmit.VerdictAdvancing {
		t.Errorf("verdict = %q, want %q", e.Report.Verdict, depthadmit.VerdictAdvancing)
	}
	if e.Report.Coverage.Carried != 1 || e.Report.Coverage.Declared != 3 {
		t.Errorf("coverage = %d/%d, want 1/3", e.Report.Coverage.Carried, e.Report.Coverage.Declared)
	}
	if e.Report.Frontier == nil || e.Report.Frontier.PhaseID != "p2" {
		t.Fatalf("frontier = %+v, want p2", e.Report.Frontier)
	}
	if !strings.Contains(e.Handoff, "p2") || !strings.Contains(e.Handoff, "2 remaining") {
		t.Errorf("handoff %q does not carry the next phase and what remains", e.Handoff)
	}
}

// Only the LATEST W3 row counts. A phase credited by an earlier scoring pass whose
// commit later went dangling must LOSE its credit, or the ledger would remember a
// witness that no longer exists.
func TestTrajctlDepthReadsTheLatestRowNotTheUnion(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "deep-line")
	witnessPhases(t, ledger, "deep-line", "p1", "p2") // earlier pass: two resolved
	witnessPhases(t, ledger, "deep-line", "p1")       // re-score: p2's commit went dangling

	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	rep := depthadmit.Fold(trajctlDepthInput(st.Objectives["deep-line"], st))
	if rep.Coverage.Carried != 1 {
		t.Fatalf("carried = %d, want 1 — a lost witness must not stay credited", rep.Coverage.Carried)
	}
	if rep.Frontier == nil || rep.Frontier.PhaseID != "p2" {
		t.Errorf("frontier = %+v, want it back at p2", rep.Frontier)
	}
	// And a met close is refused again, which is the point of losing the credit.
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"close", "--id", "deep-line", "--ledger", ledger}); code == 0 {
		t.Error("close met succeeded after a witness was lost; want refusal")
	}
}

// A non-W3 row must not buy depth: the gate credits verified commit progress only.
func TestTrajctlDepthIgnoresNonW3Rows(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	declarePlannedObjective(t, ledger, "deep-line")
	if err := trajctl.Append(ledger, trajctl.ScoreRecord(trajctl.ScoreRow{
		ObjectiveID: "deep-line",
		Value:       1.0,
		Method:      trajctl.JudgeScorerMethod,
		Version:     trajctl.JudgeScorerVersion,
		Witness:     trajctl.W1,
		Evidence: []trajctl.EvidenceRef{
			{Kind: "commit", Ref: strings.Repeat("a", 40), Detail: "p1"},
			{Kind: "commit", Ref: strings.Repeat("b", 40), Detail: "p2"},
			{Kind: "commit", Ref: strings.Repeat("c", 40), Detail: "p3"},
		},
	})); err != nil {
		t.Fatalf("append: %v", err)
	}
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	rep := depthadmit.Fold(trajctlDepthInput(st.Objectives["deep-line"], st))
	if rep.Verdict != depthadmit.VerdictShallow {
		t.Fatalf("verdict = %q, want %q — a W1 judge row must not witness phase depth",
			rep.Verdict, depthadmit.VerdictShallow)
	}
}
