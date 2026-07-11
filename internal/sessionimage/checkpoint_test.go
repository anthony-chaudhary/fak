package sessionimage

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestCheckpointRoundTripLiveSession is the #2760 named witness: checkpoint a LIVE session,
// restore, and assert the restored drive state is equivalent WHILE the source session keeps
// running (its live drive is untouched, it is still controllable, and the op mutated nothing
// it was handed). The live session is a real session.Table entry — its boundary-consistent
// state read via Table.Get is exactly what an operator snapshots.
func TestCheckpointRoundTripLiveSession(t *testing.T) {
	ctx := context.Background()
	const trace = "sess-ckpt-live"

	// A live, RUNNING session with a real drive: budget, pace, priority set on the table.
	tbl := session.NewTable()
	tbl.Decide(trace) // seed a live (Running) record
	if _, ok := tbl.SetBudget(trace, session.Budget{TurnsLeft: 5, TokensLeft: session.Unbounded, ContextTokensLeft: session.Unbounded}); !ok {
		t.Fatal("SetBudget refused on a live session")
	}
	if _, ok := tbl.SetPace(trace, session.Pace{MaxTokensPerTurn: 512, MinTurnGapMs: 100}); !ok {
		t.Fatal("SetPace refused on a live session")
	}
	if _, ok := tbl.SetPriority(trace, 3); !ok {
		t.Fatal("SetPriority refused on a live session")
	}
	// The boundary-consistent read the checkpoint captures — an atomic snapshot value.
	before := tbl.Get(trace)
	if before.Run != session.Running {
		t.Fatalf("pre-checkpoint run-state = %v, want Running (a live session)", before.Run)
	}

	// Its context image: a couple of recall pages, the content half of the snapshot.
	rec := recall.NewRecorder(trace)
	rec.Record(ctx, "get_user_details", []byte(`{"user":"mia"}`))
	rec.Record(ctx, "search_flights", []byte("UA123 $310"))

	origLabels := map[string]string{"role": "primary"}
	in := Input{
		SessionID: trace,
		Drive:     before,
		Recorder:  rec,
		Model:     "model-live",
		Host:      "server-1",
		Labels:    origLabels,
		Now:       1_700_000_000,
	}

	dir := filepath.Join(t.TempDir(), "ckpt")
	meta, err := Checkpoint{Dest: dir, Reason: "witness"}.Snapshot(in)
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if meta.SessionID != trace {
		t.Fatalf("snapshot id = %q, want %q", meta.SessionID, trace)
	}

	// --- Restorable + drive-state equivalence: load the snapshot back, assert the drive. ---
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir snapshot: %v", err)
	}
	if !reflect.DeepEqual(img.Drive, before) {
		t.Fatalf("restored drive = %+v, want the captured live drive %+v", img.Drive, before)
	}
	if !img.HasCoreImage() {
		t.Fatal("snapshot dropped the context image — a checkpoint must capture drive + content")
	}
	if sess, err := img.Recall(); err != nil || sess == nil {
		t.Fatalf("restored content image absent: sess=%v err=%v", sess, err)
	}

	// The operator reason is recorded on the snapshot, and the caller's labels rode along
	// WITHOUT the op mutating the caller's map.
	if meta.Labels[checkpointReasonLabel] != "witness" || meta.Labels["role"] != "primary" {
		t.Fatalf("snapshot labels = %v, want role=primary + %s=witness", meta.Labels, checkpointReasonLabel)
	}
	if _, stamped := origLabels[checkpointReasonLabel]; stamped {
		t.Fatal("Snapshot mutated the caller's Labels map — the source must be unaffected")
	}

	// --- Source unaffected / keeps running: the live drive is untouched and still controllable. ---
	after := tbl.Get(trace)
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("live drive changed by the checkpoint: %+v != %+v — checkpointing must not disturb the source", after, before)
	}
	// Still live: the session continues to accept control after being checkpointed.
	if _, ok := tbl.Transition(trace, session.Paused, "post-checkpoint"); !ok {
		t.Fatal("live session refused a control write after checkpoint — it should have kept running, not stopped")
	}
}

// TestCheckpointRefusals pins the closed refusal vocabulary: an empty destination is
// CHECKPOINT_MALFORMED, and a session with no identity is CHECKPOINT_NO_SESSION. Both
// surface as a structured *CheckpointRefusal, never a bare error.
func TestCheckpointRefusals(t *testing.T) {
	// Empty destination — nowhere to write.
	_, err := Checkpoint{Dest: ""}.Snapshot(Input{SessionID: "s1"})
	var ref *CheckpointRefusal
	if !errors.As(err, &ref) || ref.Reason != CheckpointMalformed {
		t.Fatalf("empty dest error = %v, want a *CheckpointRefusal with %s", err, CheckpointMalformed)
	}

	// No session identity — nothing to capture.
	_, err = Checkpoint{Dest: t.TempDir()}.Snapshot(Input{})
	if !errors.As(err, &ref) || ref.Reason != CheckpointNoSession {
		t.Fatalf("no-identity error = %v, want a *CheckpointRefusal with %s", err, CheckpointNoSession)
	}

	// Validate() alone is the shape check the producer edge calls.
	if r := (Checkpoint{Dest: "  "}).Validate(); r == nil || r.Reason != CheckpointMalformed {
		t.Fatalf("Validate(blank dest) = %v, want %s", r, CheckpointMalformed)
	}
	if r := (Checkpoint{Dest: "x"}).Validate(); r != nil {
		t.Fatalf("Validate(ok dest) = %v, want nil", r)
	}
}
