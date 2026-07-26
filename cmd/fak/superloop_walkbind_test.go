package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// writeDispatchProgressLedger fabricates the dispatch-progress ledger the
// issue-progress gate folds (the same closed_now rows `fak dispatch progress`
// witnesses), so the walk-path binding is testable hermetically.
func writeDispatchProgressLedger(t *testing.T, root string, closedNow int) {
	t.Helper()
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	row, err := json.Marshal(map[string]any{"closed_now": closedNow})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, dispatchProgressLogName), append(row, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestSuperloopWalkBindsIssueProgress is the #4958 walk-bind witness: `fak superloop
// walk` folds a declared-target intent's LIVE dispatch-ledger progress into the
// walk's issue gate — the same issueProgressWalkOpts the drive path binds — so a
// walk can never read a target-owing intent cleaner than the drive would. With a
// ledger witnessing 3 closes against run-the-night's declared 200, the report must
// carry measured progress 3 and shortfall 197; with NO ledger the gate stays
// surface-only (declared target visible, nothing measured, no fabricated zero).
func TestSuperloopWalkBindsIssueProgress(t *testing.T) {
	root := t.TempDir()
	writeDispatchProgressLedger(t, root, 3)

	var out, errb bytes.Buffer
	code := runSuperloop(&out, &errb, []string{"walk", "run-the-night", "--workspace", root, "--json"})
	if code != 1 {
		t.Fatalf("walk of a shortfall-owing intent must exit 1, got %d: %s", code, errb.String())
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, out.String())
	}
	if !rep.IssueProgressMeasured {
		t.Fatal("a present dispatch ledger must make the walk's issue gate MEASURED (#4958: the walk binds the same opts as the drive)")
	}
	if rep.IssueTarget != 200 || rep.IssueProgressed != 3 || rep.IssueShortfall != 197 {
		t.Fatalf("issue gate = target %d progressed %d shortfall %d, want 200/3/197",
			rep.IssueTarget, rep.IssueProgressed, rep.IssueShortfall)
	}

	// No ledger: surface-only — the declared target shows, nothing is measured, and
	// no zero is fabricated (the nightIssueProgress fail-closed edge).
	bare := t.TempDir()
	out.Reset()
	errb.Reset()
	if code := runSuperloop(&out, &errb, []string{"walk", "run-the-night", "--workspace", bare, "--json"}); code != 1 {
		t.Fatalf("walk of an empty workspace exits 1, got %d: %s", code, errb.String())
	}
	rep = superloop.WalkReport{}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, out.String())
	}
	if rep.IssueProgressMeasured || rep.IssueShortfall != 0 || rep.IssueTarget != 200 {
		t.Fatalf("no ledger must stay surface-only: measured=%v shortfall=%d target=%d, want false/0/200",
			rep.IssueProgressMeasured, rep.IssueShortfall, rep.IssueTarget)
	}
}

// TestSuperloopDescendFoldsShortfallAtRoot is the #4958 descend-bind witness: the
// SAME issue gate folds inside the recursive DESCEND, so run-the-night's headline
// shortfall lands in its SubwalkStatus debt at the ROOT walk's altitude and a big
// headline miss out-ranks trivial member debt — through the walk path, not only the
// drive.
func TestSuperloopDescendFoldsShortfallAtRoot(t *testing.T) {
	root := t.TempDir()
	writeDispatchProgressLedger(t, root, 3)

	var out, errb bytes.Buffer
	if code := runSuperloop(&out, &errb, []string{"walk", "tend", "--workspace", root, "--json"}); code != 1 {
		t.Fatalf("tend walk exits 1 over an unsatisfied fleet, got %d: %s", code, errb.String())
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, out.String())
	}
	var night *superloop.MemberStatus
	for i := range rep.Statuses {
		if rep.Statuses[i].Member.Kind == superloop.KindSuperloop && rep.Statuses[i].Member.Ref == "run-the-night" {
			night = &rep.Statuses[i]
		}
	}
	if night == nil {
		t.Fatalf("tend walk lost the run-the-night member: %+v", rep.Statuses)
	}
	if !night.Measured {
		t.Fatal("run-the-night must arrive DESCENDED (measured), not a container pointer")
	}
	// SubwalkStatus folds TotalDebt + IssueShortfall; with 197 issues still owed the
	// descended debt must carry at least the shortfall, so the root's worst-first
	// ranks the headline miss ahead of any sibling's trivial debt.
	if night.Debt < 197 {
		t.Fatalf("descended run-the-night debt = %d, want >= 197 (the headline shortfall folded at the parent's altitude)", night.Debt)
	}
	if !strings.Contains(night.Detail, "shortfall 197") {
		t.Fatalf("descended detail %q should carry the sub-walk's shortfall fold", night.Detail)
	}
}

// TestSuperloopFleetManagerRendersAndDescends covers the manager surface that the
// JSON-identity witnesses in superloop_fleet_test.go deliberately do not reach: what
// an operator actually READS on a terminal, and what the delegated walk carries
// beneath it. Both matter for a supervision verb — a status whose human render omits
// the rank key leaves the operator unable to interpret the ordering, and a delegated
// walk that quietly loses a re-parented child would still be byte-identical to the
// standing walk it delegates to.
func TestSuperloopFleetManagerRendersAndDescends(t *testing.T) {
	root := t.TempDir()

	t.Run("status human render names the rank key", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runSuperloop(&out, &errb, []string{"fleet", "status", "--workspace", root}); code != 1 {
			t.Fatalf("exit %d, want 1 (honestly unsatisfied over an empty root): %s", code, errb.String())
		}
		s := out.String()
		for _, want := range []string{"tend-fleet", "liveness × progress × follow-on", "orphaned"} {
			if !strings.Contains(s, want) {
				t.Errorf("status render missing %q — the operator cannot interpret the ordering without it:\n%s", want, s)
			}
		}
	})

	t.Run("next selects without entering", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runSuperloop(&out, &errb, []string{"fleet", "next", "--workspace", root})
		if code != 0 {
			t.Fatalf("fleet next with a selectable member exits 0, got %d: %s", code, errb.String())
		}
		s := out.String()
		if !strings.Contains(s, "worst-first:") || !strings.Contains(s, "front door:") {
			t.Fatalf("next must print the SELECT and its front-door class:\n%s", s)
		}
		// It must POINT AT the gated run verb rather than enter anything: `next` is
		// the read-only preview, and a preview that spawns is not one.
		if !strings.Contains(s, "fleet run") {
			t.Fatalf("next must point at the gated run verb, not enter anything itself:\n%s", s)
		}
	})

	t.Run("the delegated walk keeps its re-parented reporting child", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runSuperloop(&out, &errb, []string{"fleet", "walk", "--workspace", root, "--json"})
		if code != 1 {
			t.Fatalf("fleet walk of an empty workspace exits 1, got %d: %s", code, errb.String())
		}
		var rep superloop.WalkReport
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("walk json: %v\n%s", err, out.String())
		}
		if rep.Schema != superloop.WalkSchema || rep.Name != "tend-fleet" {
			t.Fatalf("delegated walk report = %q/%q, want the tend-fleet walk", rep.Schema, rep.Name)
		}
		// The reporting family (#4862) rides beneath this intent as ONE descended
		// child — counted once, at this altitude, not re-walked at the root.
		var reporting bool
		for _, st := range rep.Statuses {
			if st.Member.Kind == superloop.KindSuperloop && st.Member.Ref == "tend-reporting" {
				reporting = true
				if !st.Measured {
					t.Errorf("tend-reporting must arrive DESCENDED (measured) through the identical fold, got %+v", st)
				}
			}
		}
		if !reporting {
			t.Error("the fleet walk lost its re-parented tend-reporting child (#4862)")
		}
	})
}
