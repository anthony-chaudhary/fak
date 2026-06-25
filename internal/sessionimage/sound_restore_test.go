package sessionimage

import (
	"context"
	"os"
	"path/filepath"
	"strings"
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
	// The keep-bit rides a FIRST-CLASS, integrity-indexed part (witness.json) — NOT a
	// descriptive Meta.Label — so a restore reads a non-forgeable VerifiedDone rung that
	// fails the image closed if tampered (TestWitnessPartGatesReplayAfterTamperFailsClosed).
	if _, err := DumpDir(imageDir, Input{
		SessionID: sessionID,
		Drive:     drive,
		Recorder:  rec,
		Witness:   []WitnessEntry{{EffectID: taskID, Record: witness}},
		Model:     "model-before",
		Host:      "host-before",
		Now:       now.Unix(),
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	img, err := LoadDir(imageDir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	// The keep-bit is read back from integrity-checked image bytes, not re-derived from a
	// receipt that merely happens to still be on disk: this is the persisted distinction.
	entries, err := img.Witness()
	if err != nil {
		t.Fatalf("Witness: %v", err)
	}
	if len(entries) != 1 || entries[0].EffectID != taskID || entries[0].Record.VerifiedState != taskmgr.VerifiedDone {
		t.Fatalf("persisted keep-bit not restored: %+v", entries)
	}
	if !img.VerifiedDone(taskID) {
		t.Fatalf("VerifiedDone(%q) = false, want true (the persisted keep-bit)", taskID)
	}
	if img.VerifiedDone("task-never-witnessed") {
		t.Fatalf("VerifiedDone reported a never-witnessed effect as done")
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
	// The live restored handle carries the keep-bit too — a resumed loop gates on it
	// directly without re-reading the image directory.
	if len(resumed.Witness) != 1 || resumed.Witness[0].Record.VerifiedState != taskmgr.VerifiedDone {
		t.Fatalf("resumed handle did not carry the keep-bit: %+v", resumed.Witness)
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

	// The fak arm gates re-execution on the PERSISTED keep-bit (resumed.Witness[0].Record),
	// read from the image — not on a live re-witness of an on-disk receipt. Already
	// VerifiedDone -> the effect is NOT re-fired.
	restoredWitness := resumed.Witness[0].Record
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

// TestWitnessPartGatesReplayAfterTamperFailsClosed proves the keep-bit is non-forgeable:
// a tampered witness.json fails the WHOLE image closed on the sha256 integrity check, so a
// forged "already done" can never wave a skipped-but-not-actually-completed effect through.
// (The ACRFence soundness depends on this — a keep-bit an attacker could edit would be no
// distinction at all.)
func TestWitnessPartGatesReplayAfterTamperFailsClosed(t *testing.T) {
	const (
		sessionID = "sess-tamper"
		taskID    = "task-charge-card"
	)
	now := time.Unix(1_700_200_000, 0)
	witness := taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedDone, Source: "path", Detail: "all referenced paths exist"}

	imageDir := filepath.Join(t.TempDir(), "image")
	if _, err := DumpDir(imageDir, Input{
		SessionID: sessionID,
		Drive:     session.State{TraceID: sessionID, Run: session.Stopped, Reason: session.ReasonStopped, Rev: 3},
		Witness:   []WitnessEntry{{EffectID: taskID, Record: witness}},
		Now:       now.Unix(),
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	// Sanity: the untampered image loads and the keep-bit reads VerifiedDone.
	if img, err := LoadDir(imageDir); err != nil || !img.VerifiedDone(taskID) {
		t.Fatalf("clean image did not load with the keep-bit: err=%v", err)
	}

	// Forge the keep-bit on disk: flip verified_done to an attacker's claim. The bytes no
	// longer match the digest recorded in image.json's Parts index.
	wpath := filepath.Join(imageDir, WitnessFile)
	raw, err := os.ReadFile(wpath)
	if err != nil {
		t.Fatalf("read witness: %v", err)
	}
	forged := strings.Replace(string(raw), "verified_done", "verified_done_FORGED", 1)
	if forged == string(raw) {
		t.Fatal("test setup: witness.json did not contain the expected verified_done token")
	}
	if err := os.WriteFile(wpath, []byte(forged), 0o644); err != nil {
		t.Fatalf("write forged witness: %v", err)
	}

	// LoadDir must fail closed on the digest mismatch — the forged keep-bit never reaches
	// a VerifiedDone read.
	if _, err := LoadDir(imageDir); err == nil {
		t.Fatal("LoadDir accepted a tampered witness.json (forged keep-bit) — integrity gate did not fail closed")
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
