package sessionimage

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// TestSoundRestoreSkipsWitnessedEffectReplay is the ACRFence-style semantic rollback
// fixture: the control arm replays a regenerated irreversible effect after restore, while
// the fak arm reloads a session image and consults an independently read VerifiedDone rung
// before deciding whether the effect should fire again.
func TestSoundRestoreSkipsWitnessedEffectReplay(t *testing.T) {
	ctx := context.Background()
	const (
		sessionID        = "sess-sound-restore"
		taskID           = "task-refund-500"
		originalEffect   = "refund-ledger/refund-500/original"
		regeneratedAfter = "refund-ledger/refund-500/regenerated-after-restore"
	)

	control := newEffectLedger()
	acrFenceThreatControl(control, originalEffect, regeneratedAfter)
	if got := control.Count(); got != 2 {
		t.Fatalf("control effect count = %d, want 2 duplicate firings", got)
	}
	if control.Refs()[0] == control.Refs()[1] {
		t.Fatalf("control did not model semantic rollback: refs = %+v", control.Refs())
	}

	effect := newEffectLedger()
	effect.Fire(originalEffect)
	receiptPath := filepath.Join(t.TempDir(), "refund-receipt.txt")
	if err := os.WriteFile(receiptPath, []byte(originalEffect), 0o644); err != nil {
		t.Fatalf("write receipt: %v", err)
	}

	now := time.Unix(1_700_100_000, 0)
	mgr := taskmgr.NewManager(taskmgr.WithClock(func() time.Time { return now }))
	task, err := mgr.StartTask(taskmgr.TaskSpec{TaskID: taskID, Title: "refund payment"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	if err := task.Finish(); err != nil {
		t.Fatalf("finish task: %v", err)
	}
	refs := []taskmgr.EvidenceRef{{
		Kind: "path",
		Ref:  receiptPath,
		Note: "receipt proves the irreversible effect already happened",
	}}
	witness, err := mgr.WitnessTask(taskID, taskmgr.PathWitness{}, refs)
	if err != nil {
		t.Fatalf("witness task: %v", err)
	}
	if witness.VerifiedState != taskmgr.VerifiedDone {
		t.Fatalf("initial witness = %q, want verified_done", witness.VerifiedState)
	}
	if got := mgr.Snapshot().Tasks[0]; got.State != taskmgr.StateDone || got.Witness == nil || got.Witness.VerifiedState != taskmgr.VerifiedDone {
		t.Fatalf("task snapshot did not keep claimed-done plus witness rung: %+v", got)
	}

	rec := recall.NewRecorder(sessionID)
	rec.RecordWithWitness(ctx, "task_witness", []byte("verified refund receipt: "+receiptPath), "taskmgr:"+taskID+":verified_done")
	drive := session.State{
		TraceID: sessionID,
		Run:     session.Draining,
		Budget:  session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded},
		Reason:  session.ReasonDrained,
		Rev:     9,
	}
	imageDir := filepath.Join(t.TempDir(), "image")
	if _, err := DumpDir(imageDir, Input{
		SessionID: sessionID,
		Drive:     drive,
		Recorder:  rec,
		Model:     "model-before",
		Host:      "host-before",
		Labels: map[string]string{
			"effect_task":          taskID,
			"effect_receipt_path":  receiptPath,
			"effect_witness_state": string(witness.VerifiedState),
		},
		Now: now.Unix(),
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	img, err := LoadDir(imageDir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if img.Meta.Labels["effect_witness_state"] != string(taskmgr.VerifiedDone) {
		t.Fatalf("image did not carry the witness pointer/state labels: %+v", img.Meta.Labels)
	}

	table := session.NewTable()
	resumed, err := img.Rehydrate(ctx, RehydrateOptions{
		Table:   table,
		ToModel: "model-after",
		ToHost:  "host-after",
		Reason:  "sound-restore-fixture",
		Now:     now.Add(time.Minute).Unix(),
	})
	if err != nil {
		t.Fatalf("Rehydrate: %v", err)
	}
	if !resumed.Migrated || resumed.Meta.Model != "model-after" || resumed.Meta.Host != "host-after" {
		t.Fatalf("restore migration not recorded: migrated=%v meta=%+v", resumed.Migrated, resumed.Meta)
	}
	if page := resumed.Session.Pages()[0]; page.Witness != "taskmgr:"+taskID+":verified_done" {
		t.Fatalf("recall witness page not restored: %+v", page)
	}
	if body, err := resumed.Session.Resolve(ctx, 0); err != nil || string(body) != "verified refund receipt: "+receiptPath {
		t.Fatalf("restored witness page resolve = (%q, %v)", body, err)
	}

	if got := table.Get(sessionID); got.Run != session.Draining || got.Reason != session.ReasonDrained || got.Rev != 9 {
		t.Fatalf("drive not restored before boundary: %+v", got)
	}
	if verdict := table.Decide(sessionID); verdict.Proceed || !verdict.Stop || verdict.Reason != session.ReasonDrained || verdict.State.Run != session.Stopped {
		t.Fatalf("draining restore was not taken at the next boundary: %+v", verdict)
	}

	restoredWitness := taskmgr.PathWitness{}.WitnessClaim(taskmgr.Claim{
		TaskID: taskID,
		State:  taskmgr.StateDone,
		Refs:   refs,
	})
	if restoredWitness.VerifiedState != taskmgr.VerifiedDone {
		t.Fatalf("restored witness read-back = %q, want verified_done", restoredWitness.VerifiedState)
	}
	if fired := resumeEffectWithWitnessGate(effect, restoredWitness, regeneratedAfter); fired {
		t.Fatal("witness-gated restore re-fired an already verified effect")
	}
	if got := effect.Count(); got != 1 {
		t.Fatalf("fak restore effect count = %d, want 1", got)
	}
}

type effectLedger struct {
	refs []string
}

func newEffectLedger() *effectLedger { return &effectLedger{} }

func (l *effectLedger) Fire(ref string) {
	l.refs = append(l.refs, ref)
}

func (l *effectLedger) Count() int { return len(l.refs) }

func (l *effectLedger) Refs() []string {
	out := make([]string, len(l.refs))
	copy(out, l.refs)
	return out
}

func acrFenceThreatControl(l *effectLedger, refs ...string) {
	for _, ref := range refs {
		l.Fire(ref)
	}
}

func resumeEffectWithWitnessGate(l *effectLedger, witness taskmgr.WitnessRecord, regeneratedRef string) bool {
	if witness.VerifiedState == taskmgr.VerifiedDone {
		return false
	}
	l.Fire(regeneratedRef)
	return true
}
