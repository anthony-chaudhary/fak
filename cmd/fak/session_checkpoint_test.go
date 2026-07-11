package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// dumpCheckpointSource writes a small, whole session bundle the checkpoint verb can snapshot.
func dumpCheckpointSource(t *testing.T, id string) string {
	t.Helper()
	dir := t.TempDir()
	rec := recall.NewRecorder(id)
	rec.Record(context.Background(), "get_user_details", []byte(`{"user":"mia"}`))
	rec.Record(context.Background(), "search_flights", []byte("UA123 $310"))
	in := sessionimage.Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Running, Budget: session.Budget{TurnsLeft: 6}, Priority: 1},
		Recorder:  rec,
		Model:     "model-A",
		Host:      "server-1",
		Now:       1_700_000_000,
	}
	if _, err := sessionimage.DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir source: %v", err)
	}
	return dir
}

// TestRunSessionCheckpoint drives the CLI verb end-to-end: it snapshots a session bundle into
// a fresh addressable image preserving the id, reports the copy-on-write lineage, and leaves
// the source untouched — a restorable snapshot whose drive equals the source's.
func TestRunSessionCheckpoint(t *testing.T) {
	srcDir := dumpCheckpointSource(t, "sess-live")
	snapDir := filepath.Join(t.TempDir(), "snap")

	srcImageBefore, err := os.ReadFile(filepath.Join(srcDir, sessionimage.ImageFile))
	if err != nil {
		t.Fatalf("read source image.json: %v", err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSessionCheckpoint(&stdout, &stderr, []string{srcDir, "--out", snapDir, "--reason", "pre-risk"})
	if rc != 0 {
		t.Fatalf("runSessionCheckpoint rc=%d stderr=%s", rc, stderr.String())
	}

	// The snapshot exists, is whole, preserves the id, and restores to the source drive.
	src, err := sessionimage.LoadDir(srcDir)
	if err != nil {
		t.Fatalf("LoadDir source: %v", err)
	}
	snap, err := sessionimage.LoadDir(snapDir)
	if err != nil {
		t.Fatalf("LoadDir snapshot: %v", err)
	}
	if snap.Meta.SessionID != "sess-live" || snap.Meta.ParentID != "" {
		t.Fatalf("snapshot identity = id %q parent %q, want sess-live with no parent", snap.Meta.SessionID, snap.Meta.ParentID)
	}
	if !reflect.DeepEqual(snap.Drive, src.Drive) {
		t.Fatalf("snapshot drive %+v != source drive %+v", snap.Drive, src.Drive)
	}
	if got := stdout.String(); !strings.Contains(got, "checkpointed sess-live ->") {
		t.Fatalf("summary missing checkpoint line: %q", got)
	}

	// Source unaffected: its image.json is byte-identical after the checkpoint.
	srcImageAfter, err := os.ReadFile(filepath.Join(srcDir, sessionimage.ImageFile))
	if err != nil {
		t.Fatalf("read source image.json (after): %v", err)
	}
	if string(srcImageBefore) != string(srcImageAfter) {
		t.Fatal("checkpoint wrote the source image.json; the source session must be unaffected")
	}
}

// TestRunSessionCheckpointUsageErrors covers the argument guardrails.
func TestRunSessionCheckpointUsageErrors(t *testing.T) {
	srcDir := dumpCheckpointSource(t, "sess-live")
	var out, errb bytes.Buffer
	// Missing --out is the closed CHECKPOINT_MALFORMED shape refusal (usage exit 2).
	if rc := runSessionCheckpoint(&out, &errb, []string{srcDir}); rc != 2 {
		t.Fatalf("missing --out rc=%d, want 2", rc)
	}
	if got := errb.String(); !strings.Contains(got, "CHECKPOINT_MALFORMED") {
		t.Fatalf("missing --out error = %q, want CHECKPOINT_MALFORMED", got)
	}
	// No source arg.
	if rc := runSessionCheckpoint(&out, &errb, []string{"--out", t.TempDir()}); rc != 2 {
		t.Fatalf("missing source rc=%d, want 2", rc)
	}
	// A non-existent source dir is a runtime error (fail closed on a missing bundle).
	if rc := runSessionCheckpoint(&out, &errb, []string{filepath.Join(os.TempDir(), "does-not-exist-2760"), "--out", t.TempDir()}); rc != 1 {
		t.Fatalf("missing source dir rc=%d, want 1", rc)
	}
}
