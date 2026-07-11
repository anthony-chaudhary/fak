package sessionimage

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// dumpSnapshotSource writes a small, whole session bundle SnapshotDir can capture from.
func dumpSnapshotSource(t *testing.T, id string) string {
	t.Helper()
	dir := t.TempDir()
	rec := recall.NewRecorder(id)
	rec.Record(context.Background(), "get_user_details", []byte(`{"user":"mia"}`))
	rec.Record(context.Background(), "search_flights", []byte("UA123 $310"))
	in := Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Throttled, Budget: session.Budget{TurnsLeft: 4}, Priority: 2},
		Recorder:  rec,
		Model:     "model-A",
		Host:      "laptop",
		Now:       1_700_000_000,
	}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir source: %v", err)
	}
	return dir
}

// TestSnapshotDir proves the on-disk checkpoint capture: a fresh, integrity-verified snapshot
// that PRESERVES the session id and drive, shares content copy-on-write, records the checkpoint
// lineage — and leaves the source bundle byte-for-byte untouched.
func TestSnapshotDir(t *testing.T) {
	srcDir := dumpSnapshotSource(t, "sess-live")
	destDir := filepath.Join(t.TempDir(), "snap")

	// Capture the source bytes up front so we can prove the source was only READ.
	srcImageBefore, err := os.ReadFile(filepath.Join(srcDir, ImageFile))
	if err != nil {
		t.Fatalf("read source image.json: %v", err)
	}
	srcBefore, err := LoadDir(srcDir)
	if err != nil {
		t.Fatalf("LoadDir source (before): %v", err)
	}

	meta, err := SnapshotDir(srcDir, destDir, SnapshotOptions{Reason: "pre-risk", Now: 1_700_000_500})
	if err != nil {
		t.Fatalf("SnapshotDir: %v", err)
	}

	// The snapshot is whole, preserves the id + drive, and is not re-keyed (no parent link).
	snap, err := LoadDir(destDir)
	if err != nil {
		t.Fatalf("LoadDir snapshot: %v", err)
	}
	if snap.Meta.SessionID != "sess-live" {
		t.Fatalf("snapshot id = %q, want sess-live (a checkpoint preserves the id)", snap.Meta.SessionID)
	}
	if snap.Meta.ParentID != "" {
		t.Fatalf("snapshot has parent_id %q — a checkpoint is the same session, not a fork", snap.Meta.ParentID)
	}
	if !reflect.DeepEqual(snap.Drive, srcBefore.Drive) {
		t.Fatalf("snapshot drive = %+v, want the source drive %+v (verbatim)", snap.Drive, srcBefore.Drive)
	}
	if !snap.HasCoreImage() {
		t.Fatal("snapshot dropped the recall core image — content must ride along")
	}
	// The capture is an audited fact: the last migration names the checkpoint + the note.
	if len(meta.Migrations) == 0 {
		t.Fatal("snapshot recorded no migration entry")
	}
	last := meta.Migrations[len(meta.Migrations)-1].Reason
	if !strings.Contains(last, "checkpoint of sess-live at ") || !strings.Contains(last, "pre-risk") {
		t.Fatalf("migration reason = %q, want the checkpoint lineage + note", last)
	}
	if meta.UpdatedUnix != 1_700_000_500 {
		t.Fatalf("snapshot UpdatedUnix = %d, want the capture stamp", meta.UpdatedUnix)
	}

	// Content is shared, not dropped: a page file is byte-identical in both bundles.
	sBytes, err := os.ReadFile(filepath.Join(srcDir, CASFile))
	if err != nil {
		t.Fatalf("read source %s: %v", CASFile, err)
	}
	dBytes, err := os.ReadFile(filepath.Join(destDir, CASFile))
	if err != nil {
		t.Fatalf("read snapshot %s: %v", CASFile, err)
	}
	if string(sBytes) != string(dBytes) {
		t.Fatal("snapshot cas.json diverged from the source — content should be shared verbatim")
	}

	// Source unaffected: image.json byte-identical, drive unchanged, no new migration entry.
	srcImageAfter, err := os.ReadFile(filepath.Join(srcDir, ImageFile))
	if err != nil {
		t.Fatalf("read source image.json (after): %v", err)
	}
	if string(srcImageBefore) != string(srcImageAfter) {
		t.Fatal("SnapshotDir wrote the source image.json — the source session must be unaffected")
	}
	srcAfter, err := LoadDir(srcDir)
	if err != nil {
		t.Fatalf("LoadDir source (after): %v", err)
	}
	if !reflect.DeepEqual(srcAfter.Drive, srcBefore.Drive) {
		t.Fatalf("source drive changed after checkpoint: %+v != %+v", srcAfter.Drive, srcBefore.Drive)
	}
	if len(srcAfter.Meta.Migrations) != len(srcBefore.Meta.Migrations) {
		t.Fatal("checkpoint appended a migration to the SOURCE — only the snapshot records it")
	}
}

// TestSnapshotDirGuards covers the argument + integrity guardrails.
func TestSnapshotDirGuards(t *testing.T) {
	srcDir := dumpSnapshotSource(t, "sess-live")

	// Same source and destination is refused.
	if _, err := SnapshotDir(srcDir, srcDir, SnapshotOptions{}); err == nil {
		t.Fatal("SnapshotDir(src, src) was accepted, want a refusal")
	}
	// A missing source fails closed (no snapshot of a bundle that does not exist).
	if _, err := SnapshotDir(filepath.Join(os.TempDir(), "does-not-exist-2760"), t.TempDir(), SnapshotOptions{}); err == nil {
		t.Fatal("SnapshotDir of a missing source was accepted, want a load error")
	}
	// A truncated source (integrity failure) fails closed rather than minting a torn snapshot.
	torn := t.TempDir()
	if err := os.WriteFile(filepath.Join(torn, ImageFile), []byte("{}"), 0o644); err != nil {
		t.Fatalf("seed torn image: %v", err)
	}
	if _, err := SnapshotDir(torn, t.TempDir(), SnapshotOptions{}); err == nil {
		t.Fatal("SnapshotDir of a torn source was accepted, want an integrity error")
	}
}
