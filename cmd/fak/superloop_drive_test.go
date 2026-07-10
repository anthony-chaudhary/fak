package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
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

// TestSuperloopDriveEntersTrajectoryMember is DoD witness (#2224, 4a) over the NEW
// trajectory member kind (#2563): the drive rung enters a KindTrajectory member — an open
// trajctl objective — through the SAME admission gate as any other member kind, with no
// private path. A drifting objective (declining W3 progress) is the worst-first member; the
// drive admits it (uncoordinated, no --lane), surfaces the objective's OWN front door as the
// single action, and re-folds. The driven-but-unwitnessed objective still drifts, so the
// re-fold is honestly unsatisfied — the interior-node property holds across member kinds.
func TestSuperloopDriveEntersTrajectoryMember(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	// Seed a drifting trajectory objective so improve-trajectory has a worst-first member.
	traj := filepath.Join(root, filepath.FromSlash(trajctl.DefaultLedgerRel))
	if err := trajctl.Append(traj, trajctl.ObjectiveRecord(trajctl.Objective{
		ID: "drift-obj", Statement: "ship the widget", Status: trajctl.StatusActive,
	})); err != nil {
		t.Fatalf("seed objective: %v", err)
	}
	for i, v := range []float64{0.6, 0.2} { // declining → DRIFT
		if err := trajctl.Append(traj, trajctl.ScoreRecord(trajctl.ScoreRow{
			ObjectiveID: "drift-obj", Value: v, Method: trajctl.CommitScorerMethod,
			Version: "1", Witness: trajctl.W3, UnixMillis: int64(i + 1),
		})); err != nil {
			t.Fatalf("seed score %d: %v", i, err)
		}
	}

	var out, errb bytes.Buffer
	code := runSuperloopDrive(&out, &errb, []string{"improve-trajectory", "--workspace", root, "--ledger", ledger, "--json"})
	if code != 1 {
		t.Fatalf("a driven-but-unwitnessed trajectory member must exit 1, got %d: stderr=%s", code, errb.String())
	}

	var rep superloopDriveReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if rep.Outcome != "entered" {
		t.Fatalf("outcome = %q, want %q", rep.Outcome, "entered")
	}
	if rep.Decision.Member.Kind != superloop.KindTrajectory || rep.Decision.Member.Ref != "drift-obj" {
		t.Errorf("drive must enter the worst-first trajectory objective, got kind=%q ref=%q",
			rep.Decision.Member.Kind, rep.Decision.Member.Ref)
	}
	if !strings.Contains(rep.Decision.Action, "trajctl curve --objective drift-obj") {
		t.Errorf("drive action must surface the objective's own front door, got %q", rep.Decision.Action)
	}
	if rep.Admission == nil || !rep.Admission.Admitted || rep.Admission.Status != "UNCOORDINATED" {
		t.Errorf("no-lane admission should be an uncoordinated admit, got %+v", rep.Admission)
	}
	if rep.Refold == nil || rep.Refold.Satisfied {
		t.Error("an entered-but-still-drifting objective must re-fold unsatisfied (interior-node property)")
	}
}

// TestSuperloopDriveBatchEntersTopKConcurrently is the batch DoD witness: `--batch 2`
// offers the top-2 worst-first members and, when the SHARED gate admits both under
// DISTINCT member-scoped leases, enters both in ONE invocation and re-folds ONCE.
// This is the throughput widening — K interior-node entries per drive instead of one
// — with no private spawn path (every member still passes the same gate seam).
func TestSuperloopDriveBatchEntersTopKConcurrently(t *testing.T) {
	orig := superloopDriveAdmitGate
	t.Cleanup(func() { superloopDriveAdmitGate = orig })
	// Admit every member; echo the member-scoped intent into the lease so the test
	// can prove K distinct holds coexisted (the gate mints one lease id per member).
	var gateCalls []string
	superloopDriveAdmitGate = func(root, lane string, tree []string, intent string) (superloopDriveAdmit, func()) {
		gateCalls = append(gateCalls, intent)
		return superloopDriveAdmit{Status: "ADMITTED", Admitted: true, Lease: "lease-" + intent, Tree: tree}, func() {}
	}

	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--batch", "2", "--json"})
	if code != 1 {
		t.Fatalf("an entered-but-unsatisfied batch of an empty workspace must exit 1, got %d: stderr=%s", code, errb.String())
	}

	var rep superloopDriveBatchReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("batch json: %v\n%s", err, out.String())
	}
	if rep.Outcome != "entered" || rep.Admitted != 2 || rep.Refused != 0 || rep.Offered != 2 {
		t.Fatalf("want 2 admitted/0 refused/entered, got outcome=%q admitted=%d refused=%d offered=%d", rep.Outcome, rep.Admitted, rep.Refused, rep.Offered)
	}
	if len(rep.Entries) != 2 || !rep.Entries[0].Entered || !rep.Entries[1].Entered {
		t.Fatalf("both offered members must be entered, got %+v", rep.Entries)
	}
	if rep.Entries[0].Decision.Rank != 1 || rep.Entries[1].Decision.Rank != 2 {
		t.Errorf("entries must be worst-first (rank 1 then 2), got %d then %d", rep.Entries[0].Decision.Rank, rep.Entries[1].Decision.Rank)
	}
	if l0, l1 := rep.Entries[0].Admission.Lease, rep.Entries[1].Admission.Lease; l0 == l1 {
		t.Errorf("each batch member must hold a DISTINCT member-scoped lease, got both %q", l0)
	}
	if rep.Refold == nil || rep.Refold.Satisfied {
		t.Error("a batch of an empty workspace must re-fold ONCE and honestly unsatisfied")
	}
	if len(gateCalls) != 2 || gateCalls[0] == gateCalls[1] {
		t.Errorf("the gate must be consulted once per member with distinct scopes, got %v", gateCalls)
	}

	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if n := strings.Count(string(data), `"status":"admitted"`); n != 2 {
		t.Errorf("ledger must carry one admitted witness per entered member (2), got %d:\n%s", n, string(data))
	}
}

// TestSuperloopDriveBatchSurfacesRefusalAndEntersRest is the batch collision witness:
// when the gate ADMITS one member and REFUSES another (COLLISION_RISK), the batch
// enters the admitted one, SURFACES the refusal token (never bypassing the gate), and
// still re-folds — refused members are skipped, not silently dropped. This is how the
// batch stays honest while scaling: throughput follows the non-colliding members only.
func TestSuperloopDriveBatchSurfacesRefusalAndEntersRest(t *testing.T) {
	orig := superloopDriveAdmitGate
	t.Cleanup(func() { superloopDriveAdmitGate = orig })
	var n int
	superloopDriveAdmitGate = func(root, lane string, tree []string, intent string) (superloopDriveAdmit, func()) {
		n++
		if n == 1 {
			return superloopDriveAdmit{Status: "ADMITTED", Admitted: true, Lease: "lease-" + intent}, func() {}
		}
		return superloopDriveAdmit{Status: "COLLISION_RISK", Admitted: false, Lease: "lease-" + intent,
			Detail: "region overlaps a live lease held by a peer worker"}, func() {}
	}

	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--batch", "2", "--json"})
	if code != 1 {
		t.Fatalf("a batch that entered >=1 member (unsatisfied) must exit 1, got %d: stderr=%s", code, errb.String())
	}

	var rep superloopDriveBatchReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("batch json: %v\n%s", err, out.String())
	}
	if rep.Outcome != "entered" || rep.Admitted != 1 || rep.Refused != 1 {
		t.Fatalf("want 1 admitted/1 refused/entered, got outcome=%q admitted=%d refused=%d", rep.Outcome, rep.Admitted, rep.Refused)
	}
	if !rep.Entries[0].Entered || rep.Entries[1].Entered {
		t.Errorf("first must be entered, second (refused) must NOT be, got %+v", rep.Entries)
	}
	if rep.Entries[1].Admission.Status != "COLLISION_RISK" {
		t.Errorf("the refused member must carry the surfaced token, got %q", rep.Entries[1].Admission.Status)
	}
	if !strings.Contains(errb.String(), "COLLISION_RISK") {
		t.Errorf("stderr must surface the refusal token, got %q", errb.String())
	}
	if rep.Refold == nil {
		t.Error("a batch that entered at least one member must re-fold")
	}

	data, _ := os.ReadFile(ledger)
	row := string(data)
	if !strings.Contains(row, `"status":"admitted"`) || !strings.Contains(row, `"status":"refused"`) {
		t.Errorf("ledger must carry BOTH the admitted and the refused witness:\n%s", row)
	}
}

// TestSuperloopDriveBatchAllRefusedEntersNothing pins that when the gate refuses EVERY
// offered member, the batch enters nothing, exits 3 (the single drive's refusal code),
// and does NOT re-fold — a re-fold would falsely imply work happened. No admitted row
// is written (no bypass).
func TestSuperloopDriveBatchAllRefusedEntersNothing(t *testing.T) {
	orig := superloopDriveAdmitGate
	t.Cleanup(func() { superloopDriveAdmitGate = orig })
	superloopDriveAdmitGate = func(root, lane string, tree []string, intent string) (superloopDriveAdmit, func()) {
		return superloopDriveAdmit{Status: "COLLISION_RISK", Admitted: false, Lease: "lease-" + intent,
			Detail: "region overlaps a live lease"}, func() {}
	}

	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--batch", "3", "--json"})
	if code != 3 {
		t.Fatalf("an all-refused batch must exit 3, got %d: stderr=%s", code, errb.String())
	}
	var rep superloopDriveBatchReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("batch json: %v\n%s", err, out.String())
	}
	if rep.Outcome != "refused" || rep.Admitted != 0 || rep.Refused != 3 {
		t.Fatalf("want 0 admitted/3 refused/refused, got outcome=%q admitted=%d refused=%d", rep.Outcome, rep.Admitted, rep.Refused)
	}
	if rep.Refold != nil {
		t.Error("an all-refused batch entered nothing, so it must NOT re-fold")
	}
	data, _ := os.ReadFile(ledger)
	if strings.Contains(string(data), `"status":"admitted"`) {
		t.Errorf("an all-refused batch must NOT write an admitted row (no bypass):\n%s", string(data))
	}
}

// schemaTag is the minimal shape shared by both drive reports — enough to tell a
// batch drive (fak.superloop-drive-batch.v1) from a single drive (fak.superloop-drive.v1)
// without depending on either full report struct.
type schemaTag struct {
	Schema string `json:"schema"`
	Batch  int    `json:"batch"`
}

// TestSuperloopDriveEnvBatchRaisesDefaultThroughput is the LIVE-throughput witness for
// the FAK_SUPERLOOP_BATCH deployment lever: with the env set and NO --batch flag, a
// drive takes the fan-out path (batch schema, Batch=N) instead of the historical
// single-member drive — so a scheduled meta-loop raises super-loop throughput past one
// member per invocation with a single env knob and no per-caller edits.
func TestSuperloopDriveEnvBatchRaisesDefaultThroughput(t *testing.T) {
	t.Setenv(superloopBatchEnv, "2")
	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--json"})
	if code != 1 {
		t.Fatalf("an entered-but-unsatisfied env-batch drive of an empty workspace must exit 1, got %d: stderr=%s", code, errb.String())
	}
	var tag schemaTag
	if err := json.Unmarshal(out.Bytes(), &tag); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if tag.Schema != superloop.BatchDriveSchema {
		t.Fatalf("FAK_SUPERLOOP_BATCH=2 with no --batch must take the fan-out path; got schema %q", tag.Schema)
	}
	if tag.Batch != 2 {
		t.Errorf("env-driven batch size = %d, want 2", tag.Batch)
	}
}

// TestSuperloopDriveExplicitBatchFlagBeatsEnv pins that the command-line --batch is the
// stronger signal: an explicit --batch 1 keeps the historical single-member drive even
// when FAK_SUPERLOOP_BATCH would widen it, so an operator can always pin one invocation
// back to one member regardless of the deployment default.
func TestSuperloopDriveExplicitBatchFlagBeatsEnv(t *testing.T) {
	t.Setenv(superloopBatchEnv, "5")
	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--batch", "1", "--json"})
	if code != 1 {
		t.Fatalf("an entered-but-unsatisfied single drive of an empty workspace must exit 1, got %d: stderr=%s", code, errb.String())
	}
	var tag schemaTag
	if err := json.Unmarshal(out.Bytes(), &tag); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if tag.Schema != superloop.DriveSchema {
		t.Fatalf("explicit --batch 1 must win over FAK_SUPERLOOP_BATCH=5 and stay single-member; got schema %q", tag.Schema)
	}
}

// TestSuperloopDriveUnparseableEnvBatchFailsClosedToOne pins the fail-closed contract:
// a non-integer FAK_SUPERLOOP_BATCH is surfaced on stderr and IGNORED, falling back to
// the safe single-member default rather than erroring or silently mis-scaling the fleet.
func TestSuperloopDriveUnparseableEnvBatchFailsClosedToOne(t *testing.T) {
	t.Setenv(superloopBatchEnv, "not-a-number")
	root := t.TempDir()
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	var out, errb bytes.Buffer

	code := runSuperloopDrive(&out, &errb, []string{"sweep-surfaces", "--workspace", root, "--ledger", ledger, "--json"})
	if code != 1 {
		t.Fatalf("unparseable env must fall back to the single-member drive (exit 1), got %d: stderr=%s", code, errb.String())
	}
	var tag schemaTag
	if err := json.Unmarshal(out.Bytes(), &tag); err != nil {
		t.Fatalf("drive json: %v\n%s", err, out.String())
	}
	if tag.Schema != superloop.DriveSchema {
		t.Fatalf("unparseable env must NOT enter the fan-out path; got schema %q", tag.Schema)
	}
	if !strings.Contains(errb.String(), "ignoring unparseable") {
		t.Errorf("unparseable env must be surfaced on stderr, got: %q", errb.String())
	}
}

// TestSuperloopDriveNoEnterOutcome pins the operator honesty seam (#3147): a
// non-entering drive exits cleanly (0, "satisfied") ONLY when the walk is satisfied. An
// empty-worklist walk left UNSATISFIED by an unmet headline (an issue shortfall) enters
// nothing but is NOT done — it reports "shortfall" and exits non-zero, so an automated
// night loop cannot read an unmet ~200-issue headline as a finished night.
func TestSuperloopDriveNoEnterOutcome(t *testing.T) {
	clean := superloop.DriveDecision{Enter: false, Satisfied: true}
	if outcome, code := superloopDriveNoEnterOutcome(clean); outcome != "satisfied" || code != 0 {
		t.Errorf("a satisfied non-enter must be a clean exit; got outcome %q code %d", outcome, code)
	}

	shortfall := superloop.DriveDecision{Enter: false, Satisfied: false, IssueShortfall: 200}
	outcome, code := superloopDriveNoEnterOutcome(shortfall)
	if outcome != "shortfall" {
		t.Errorf("an unmet headline must surface as a shortfall, not satisfied; got %q", outcome)
	}
	if code == 0 {
		t.Error("an unmet headline must exit non-zero — a clean exit over a shortfall is the silent-defeat defect #3147 closes")
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
