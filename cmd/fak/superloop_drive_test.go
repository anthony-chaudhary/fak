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

// TestSuperloopDriveEntersOneMemberAndRefolds is DoD witness (#2224, 4a): a drive on a
// registered intent whose gate ADMITS enters EXACTLY ONE member — the worst-first — under
// a (here uncoordinated) lease, records the admission witness on the loop ledger with the
// standing vocabulary, and re-folds. In an empty workspace every member is unmeasured, so
// the re-fold is honestly unsatisfied (exit 1) — a driven-but-unwitnessed member cannot
// satisfy the intent. The default `superloopDriveAdmitGate` runs (no stub): with no
// --lane it is the uncoordinated admit, which touches no lease fabric.
func TestSuperloopDriveEntersOneMemberAndRefolds(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--json"})
	if code != 1 {
		t.Fatalf("an entered-but-unsatisfied drive of an empty workspace must exit 1, got %d: stderr=%s", code, errb.String())
	}

	var rep superloopDriveReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if rep.Outcome != "entered" {
		t.Fatalf("outcome = %q, want %q", rep.Outcome, "entered")
	}
	if !rep.Decision.Enter || rep.Decision.Rank != 1 {
		t.Errorf("must enter the single head member, got enter=%v rank=%d", rep.Decision.Enter, rep.Decision.Rank)
	}
	if rep.Decision.Member.Ref != "code" {
		t.Errorf("worst-first among unmeasured members is the first declared (code), got %q", rep.Decision.Member.Ref)
	}
	if rep.Admission == nil || !rep.Admission.Admitted || rep.Admission.Status != "UNCOORDINATED" {
		t.Errorf("no-lane admission should be an uncoordinated admit, got %+v", rep.Admission)
	}
	if rep.Refold == nil {
		t.Fatal("an entered drive must re-fold as the exit check")
	}
	if rep.Refold.Satisfied {
		t.Error("an empty workspace re-fold cannot be satisfied (every member unmeasured)")
	}

	// The admission witness lands on the loop ledger with the standing vocabulary.
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	row := string(data)
	if !strings.Contains(row, `"status":"admitted"`) {
		t.Errorf("ledger missing the admitted witness row:\n%s", row)
	}
	if !strings.Contains(row, "superloop-sweep-surfaces") {
		t.Errorf("ledger row not keyed on the super loop id:\n%s", row)
	}
}

// TestSuperloopDriveRefusalSurfacesTokenEntersNothing is DoD witness (#2224, 4b): when
// the member's admission gate REFUSES (here COLLISION_RISK on a lease overlap), the drive
// surfaces the token and enters NOTHING — it never bypasses the gate with a private spawn
// path. The refusal is recorded on the ledger as `refused`; no `admitted` row is written,
// and no re-fold runs (nothing was entered).
func TestSuperloopDriveRefusalSurfacesTokenEntersNothing(t *testing.T) {
	orig := superloopDriveAdmitGate
	t.Cleanup(func() { superloopDriveAdmitGate = orig })
	superloopDriveAdmitGate = func(root, lane string, tree []string, intent string) (superloopDriveAdmit, func()) {
		return superloopDriveAdmit{
			Status:   "COLLISION_RISK",
			Admitted: false,
			Lease:    "loop-superloop-" + intent,
			Detail:   "region lease overlaps a live lease held by a peer worker",
		}, func() {}
	}

	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--json"})
	if code != 3 {
		t.Fatalf("a gate-refused drive must exit 3, got %d: stderr=%s", code, errb.String())
	}

	var rep superloopDriveReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if rep.Outcome != "refused" {
		t.Fatalf("outcome = %q, want %q", rep.Outcome, "refused")
	}
	if rep.Refold != nil {
		t.Error("a refused drive enters nothing, so it must NOT re-fold — that would be a bypass")
	}
	if rep.Admission == nil || rep.Admission.Status != "COLLISION_RISK" {
		t.Errorf("the refusal token must be surfaced, got %+v", rep.Admission)
	}
	if !strings.Contains(errb.String(), "COLLISION_RISK") {
		t.Errorf("stderr must surface the refusal token, got %q", errb.String())
	}

	data, _ := os.ReadFile(ledger)
	row := string(data)
	if !strings.Contains(row, `"status":"refused"`) {
		t.Errorf("ledger missing the refused witness row:\n%s", row)
	}
	if strings.Contains(row, `"status":"admitted"`) {
		t.Errorf("a refused drive must NOT write an admitted row (no bypass):\n%s", row)
	}
}

// TestSuperloopDriveUnknownIntentIsUsageError pins that an unknown intent is a usage
// error (exit 2), not a panic or a false enter.
func TestSuperloopDriveUnknownIntentIsUsageError(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSuperloopDrive(&out, &errb, []string{"no-such-intent"}); code != 2 {
		t.Fatalf("unknown intent should be a usage error (2), got %d", code)
	}
	_ = superloop.DriveSchema // bind the schema symbol so the drive payload tag is witnessed here
}
