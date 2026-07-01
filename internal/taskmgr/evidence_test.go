package taskmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

func TestOriginWitnessRunsWhenTaskStartsWithEvidence(t *testing.T) {
	now := time.Unix(1700000000, 0)
	var got Claim
	m := NewManager(
		WithClock(func() time.Time { return now }),
		WithOriginWitness(WitnessFunc(func(c Claim) WitnessRecord {
			got = c
			return WitnessRecord{
				VerifiedState: VerifiedDone,
				Source:        "origin-test",
				Detail:        "origin evidence checked",
				EvidenceRefs:  cloneEvidenceRefs(c.Refs),
			}
		})),
	)

	ref := EvidenceRef{Kind: "path", Ref: "artifact.txt", Note: "declared before dispatch"}
	if _, err := m.StartTask(TaskSpec{TaskID: "task_origin", EvidenceRefs: []EvidenceRef{ref}}); err != nil {
		t.Fatalf("start task: %v", err)
	}
	if got.TaskID != "task_origin" || got.StepID != "" || got.State != StateRunning {
		t.Fatalf("origin claim = %+v, want running task claim", got)
	}

	snap := m.Snapshot()
	if err := ValidateSnapshot(snap); err != nil {
		t.Fatalf("validate origin-witnessed snapshot: %v", err)
	}
	task := snap.Tasks[0]
	if len(task.EvidenceRefs) != 1 || task.EvidenceRefs[0] != ref {
		t.Fatalf("task evidence refs = %+v, want %+v", task.EvidenceRefs, []EvidenceRef{ref})
	}
	if task.Witness == nil || task.Witness.VerifiedState != VerifiedDone || task.Witness.Source != "origin-test" {
		t.Fatalf("origin witness = %+v, want verified_done from origin-test", task.Witness)
	}
	if task.Witness.CheckedUnixNano != now.UnixNano() {
		t.Fatalf("origin witness timestamp = %d, want %d", task.Witness.CheckedUnixNano, now.UnixNano())
	}
}

func TestOriginWitnessRunsWhenStepStartsWithEvidence(t *testing.T) {
	var got Claim
	m := NewManager(WithOriginWitness(WitnessFunc(func(c Claim) WitnessRecord {
		got = c
		return WitnessRecord{VerifiedState: VerifiedDone, Source: "origin-step", EvidenceRefs: cloneEvidenceRefs(c.Refs)}
	})))
	task, err := m.StartTask(TaskSpec{TaskID: "task_step_origin"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}

	ref := EvidenceRef{Kind: OutputRefKind, Ref: coherentOutput}
	if _, err := task.StartStep(StepSpec{StepID: "step_origin", EvidenceRefs: []EvidenceRef{ref}}); err != nil {
		t.Fatalf("start step: %v", err)
	}
	if got.TaskID != "task_step_origin" || got.StepID != "step_origin" || got.State != StateRunning {
		t.Fatalf("origin step claim = %+v, want running step claim", got)
	}
	step := m.Snapshot().Tasks[0].Steps[0]
	if len(step.EvidenceRefs) != 1 || step.EvidenceRefs[0] != ref {
		t.Fatalf("step evidence refs = %+v, want %+v", step.EvidenceRefs, []EvidenceRef{ref})
	}
	if step.Witness == nil || step.Witness.VerifiedState != VerifiedDone || step.Witness.Source != "origin-step" {
		t.Fatalf("step origin witness = %+v, want verified_done from origin-step", step.Witness)
	}
}

func TestOriginWitnessIsSkippedWithoutEvidenceRefs(t *testing.T) {
	called := false
	m := NewManager(WithOriginWitness(WitnessFunc(func(Claim) WitnessRecord {
		called = true
		return WitnessRecord{VerifiedState: VerifiedDone}
	})))
	task, err := m.StartTask(TaskSpec{TaskID: "task_no_refs"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	if _, err := task.StartStep(StepSpec{StepID: "step_no_refs"}); err != nil {
		t.Fatalf("start step: %v", err)
	}
	if called {
		t.Fatalf("origin witness ran for records with no evidence refs")
	}
	snap := m.Snapshot()
	if snap.Tasks[0].Witness != nil || snap.Tasks[0].Steps[0].Witness != nil {
		t.Fatalf("witnesses should stay nil without evidence refs: %+v / %+v", snap.Tasks[0].Witness, snap.Tasks[0].Steps[0].Witness)
	}
}

func TestOriginWitnessByKindConfirmsMixedPathAndOutput(t *testing.T) {
	dir := t.TempDir()
	artifact := filepath.Join(dir, "artifact.txt")
	if err := os.WriteFile(artifact, []byte("ok"), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	m := NewManager(WithDefaultOriginWitnesses())
	refs := []EvidenceRef{
		{Kind: PathRefKind, Ref: artifact},
		{Kind: OutputRefKind, Ref: coherentOutput},
	}
	if _, err := m.StartTask(TaskSpec{TaskID: "task_mixed_origin", EvidenceRefs: refs}); err != nil {
		t.Fatalf("start task: %v", err)
	}

	got := m.Snapshot().Tasks[0]
	if got.Witness == nil {
		t.Fatalf("missing origin witness")
	}
	if got.Witness.VerifiedState != VerifiedDone || got.Witness.Source != kindWitnessSource {
		t.Fatalf("origin witness = %+v, want registry verified_done", got.Witness)
	}
	if !strings.Contains(got.Witness.Detail, PathRefKind) || !strings.Contains(got.Witness.Detail, OutputRefKind) {
		t.Fatalf("registry detail = %q, want both verified kinds", got.Witness.Detail)
	}
	var sawPath, sawScrubbedOutput bool
	for _, ref := range got.Witness.EvidenceRefs {
		switch ref.Kind {
		case PathRefKind:
			if ref.Ref == artifact {
				sawPath = true
			}
		case OutputRefKind:
			if ref.Ref == "" && strings.Contains(ref.Note, "graded") {
				sawScrubbedOutput = true
			}
		}
	}
	if !sawPath || !sawScrubbedOutput {
		t.Fatalf("registry evidence refs = %+v, want path ref plus scrubbed output ref", got.Witness.EvidenceRefs)
	}
}

func TestOriginWitnessByKindRefusesFailingKind(t *testing.T) {
	m := NewManager(WithDefaultOriginWitnesses())
	refs := []EvidenceRef{
		{Kind: PathRefKind, Ref: filepath.Join(t.TempDir(), "missing.txt")},
		{Kind: OutputRefKind, Ref: coherentOutput},
	}
	if _, err := m.StartTask(TaskSpec{TaskID: "task_refused_origin", EvidenceRefs: refs}); err != nil {
		t.Fatalf("start task: %v", err)
	}
	got := m.Snapshot().Tasks[0].Witness
	if got == nil || got.VerifiedState != VerifiedRefused {
		t.Fatalf("origin witness = %+v, want verified_refused", got)
	}
	if !strings.Contains(got.Detail, PathRefKind) || !strings.Contains(got.Detail, "missing") {
		t.Fatalf("refusal detail = %q, want failing path kind", got.Detail)
	}
}

func TestOriginWitnessByKindUnavailableForUnregisteredKind(t *testing.T) {
	m := NewManager(WithDefaultOriginWitnesses())
	ref := EvidenceRef{Kind: "commit", Ref: "HEAD"}
	if _, err := m.StartTask(TaskSpec{TaskID: "task_unknown_origin", EvidenceRefs: []EvidenceRef{ref}}); err != nil {
		t.Fatalf("start task: %v", err)
	}
	got := m.Snapshot().Tasks[0].Witness
	if got == nil || got.VerifiedState != VerifiedUnavailable {
		t.Fatalf("origin witness = %+v, want verified_unavailable", got)
	}
	if !strings.Contains(got.Detail, "commit") || !strings.Contains(got.Detail, "no registered witness") {
		t.Fatalf("unavailable detail = %q, want unregistered commit kind", got.Detail)
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
