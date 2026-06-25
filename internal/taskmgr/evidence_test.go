package taskmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPathWitnessReadsBackEvidence(t *testing.T) {
	dir := t.TempDir()
	present := filepath.Join(dir, "present.txt")
	if err := os.WriteFile(present, []byte("x"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	missing := filepath.Join(dir, "missing.txt")
	w := PathWitness{}

	if rec := w.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: "path", Ref: present}}}); rec.VerifiedState != VerifiedDone {
		t.Fatalf("present path: state = %q, want verified_done", rec.VerifiedState)
	}
	if rec := w.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: "path", Ref: present}, {Kind: "path", Ref: missing}}}); rec.VerifiedState != VerifiedRefused {
		t.Fatalf("missing path: state = %q, want verified_refused", rec.VerifiedState)
	}
	if rec := w.WitnessClaim(Claim{Refs: []EvidenceRef{{Kind: "commit", Ref: "HEAD"}}}); rec.VerifiedState != VerifiedUnavailable {
		t.Fatalf("no path evidence: state = %q, want verified_unavailable", rec.VerifiedState)
	}
}

// TestClaimedDoneStaysDistinctFromWitness is the core invariant: a process may
// claim done, but a refusing witness leaves the claim standing beside an explicit
// verified_refused rung rather than silently downgrading or overwriting it.
func TestClaimedDoneStaysDistinctFromWitness(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_w", Total: 1})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := task.Finish(); err != nil { // the process CLAIMS done
		t.Fatalf("finish: %v", err)
	}
	refuser := WitnessFunc(func(Claim) WitnessRecord {
		return WitnessRecord{VerifiedState: VerifiedRefused, Source: "test", Verdict: "no evidence"}
	})
	rec, err := m.WitnessTask("task_w", refuser, nil)
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if rec.VerifiedState != VerifiedRefused {
		t.Fatalf("witness rec = %q, want refused", rec.VerifiedState)
	}
	got := m.Snapshot().Tasks[0]
	if got.State != StateDone {
		t.Fatalf("claimed state = %q, want done (claim must survive a refusing witness)", got.State)
	}
	if got.Witness == nil || got.Witness.VerifiedState != VerifiedRefused {
		t.Fatalf("witness rung = %+v, want verified_refused beside the claim", got.Witness)
	}
	if got.Witness.CheckedUnixNano == 0 {
		t.Fatalf("witness timestamp not stamped")
	}
}

func TestWitnessTaskPathEvidenceEndToEnd(t *testing.T) {
	dir := t.TempDir()
	artifact := filepath.Join(dir, "artifact.bin")
	if err := os.WriteFile(artifact, []byte("ok"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_art"})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := task.Finish(); err != nil {
		t.Fatalf("finish: %v", err)
	}
	rec, err := m.WitnessTask("task_art", PathWitness{}, []EvidenceRef{{Kind: "path", Ref: artifact}})
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if rec.VerifiedState != VerifiedDone {
		t.Fatalf("end-to-end path witness = %q, want verified_done", rec.VerifiedState)
	}
	if err := ValidateSnapshot(m.Snapshot()); err != nil {
		t.Fatalf("witnessed snapshot failed validation: %v", err)
	}
}

func TestSetWitnessRejectsBadStateAndUnknownTarget(t *testing.T) {
	m := NewManager()
	if _, err := m.StartTask(TaskSpec{TaskID: "task_b"}); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := m.SetTaskWitness("task_b", WitnessRecord{VerifiedState: VerifiedState("maybe")}); err == nil {
		t.Fatalf("bad verified state accepted")
	}
	if err := m.SetTaskWitness("task_missing", WitnessRecord{VerifiedState: VerifiedDone}); err == nil {
		t.Fatalf("witness on unknown task accepted")
	}
	if _, err := m.WitnessTask("task_b", nil, nil); err == nil {
		t.Fatalf("nil witness accepted")
	}
}

func TestValidateSnapshotChecksWitnessVocabulary(t *testing.T) {
	s := validSnapshot()
	s.Tasks[0].Witness = &WitnessRecord{VerifiedState: VerifiedState("bogus")}
	if err := ValidateSnapshot(s); err == nil {
		t.Fatalf("snapshot with bogus verified_state passed validation")
	}
	s.Tasks[0].Witness = &WitnessRecord{VerifiedState: VerifiedDone, Source: "test"}
	if err := ValidateSnapshot(s); err != nil {
		t.Fatalf("snapshot with a valid witness rejected: %v", err)
	}
}

// TestSnapshotJSONShowsClaimedAndWitnessed is the golden sample: one claimed-only
// task whose JSON omits any witness, and one task carrying independent witness
// evidence, in the same snapshot.
func TestSnapshotJSONShowsClaimedAndWitnessed(t *testing.T) {
	m := NewManager()
	for _, id := range []string{"task_claimed", "task_witnessed"} {
		if _, err := m.StartTask(TaskSpec{TaskID: id, Total: 1}); err != nil {
			t.Fatalf("start %s: %v", id, err)
		}
		if err := m.FinishTask(id); err != nil {
			t.Fatalf("finish %s: %v", id, err)
		}
	}
	if err := m.SetTaskWitness("task_witnessed", WitnessRecord{
		VerifiedState: VerifiedDone, Source: "commit-audit", SHA: "deadbeef",
		EvidenceRefs: []EvidenceRef{{Kind: "commit", Ref: "deadbeef", Note: "diff-witnessed"}},
	}); err != nil {
		t.Fatalf("set witness: %v", err)
	}
	snap := m.Snapshot()
	if err := ValidateSnapshot(snap); err != nil {
		t.Fatalf("validate: %v", err)
	}

	var claimed, witnessed TaskSnapshot
	for _, ts := range snap.Tasks {
		switch ts.TaskID {
		case "task_claimed":
			claimed = ts
		case "task_witnessed":
			witnessed = ts
		}
	}
	if cb, _ := json.Marshal(claimed); strings.Contains(string(cb), "witness") {
		t.Fatalf("claimed-only task leaked a witness key: %s", cb)
	}
	if claimed.State != StateDone {
		t.Fatalf("claimed task state = %q, want done", claimed.State)
	}
	if wb, _ := json.Marshal(witnessed); !strings.Contains(string(wb), `"verified_state":"verified_done"`) {
		t.Fatalf("witnessed task JSON missing verified_done: %s", wb)
	}
	if witnessed.Witness == nil || witnessed.Witness.SHA != "deadbeef" {
		t.Fatalf("witnessed task evidence missing: %+v", witnessed.Witness)
	}
}
