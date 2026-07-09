package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// dumpParentImage writes a small, whole parent checkpoint the branch verb can fork from.
func dumpParentImage(t *testing.T, id string) string {
	t.Helper()
	dir := t.TempDir()
	rec := recall.NewRecorder(id)
	rec.Record(context.Background(), "get_user_details", []byte(`{"user":"mia"}`))
	rec.Record(context.Background(), "search_flights", []byte("UA123 $310"))
	in := sessionimage.Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Throttled, Budget: session.Budget{TurnsLeft: 4}},
		Recorder:  rec,
		Model:     "model-A",
		Host:      "laptop",
		Now:       1_700_000_000,
	}
	if _, err := sessionimage.DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	return dir
}

// TestRunSessionBranch drives the CLI verb end-to-end: it forks a parent checkpoint into a
// new durable id, registers the branch descriptor with a parent_id link, and reports the
// copy-on-write lineage — while leaving the parent bundle untouched.
func TestRunSessionBranch(t *testing.T) {
	parentDir := dumpParentImage(t, "sess-parent")
	branchDir := filepath.Join(t.TempDir(), "branch")
	registry := filepath.Join(t.TempDir(), "registry.json")

	var stdout, stderr bytes.Buffer
	rc := runSessionBranch(&stdout, &stderr, []string{parentDir, "--out", branchDir, "--id", "sess-child", "--registry", registry})
	if rc != 0 {
		t.Fatalf("runSessionBranch rc=%d stderr=%s", rc, stderr.String())
	}

	// The branch image exists, is whole, and links the parent.
	img, err := sessionimage.LoadDir(branchDir)
	if err != nil {
		t.Fatalf("LoadDir branch: %v", err)
	}
	if img.Meta.SessionID != "sess-child" || img.Meta.ParentID != "sess-parent" {
		t.Fatalf("branch identity = id %q parent %q, want sess-child/sess-parent", img.Meta.SessionID, img.Meta.ParentID)
	}
	if got := stdout.String(); !strings.Contains(got, "branched sess-parent -> sess-child") {
		t.Fatalf("summary missing lineage line: %q", got)
	}

	// The C1 descriptor was written with the parent_id link.
	reg := session.NewRegistry(session.NewFileStore(registry))
	d, ok, err := reg.Get("sess-child")
	if err != nil || !ok {
		t.Fatalf("branch descriptor absent: ok=%v err=%v", ok, err)
	}
	if d.ParentID != "sess-parent" {
		t.Fatalf("descriptor parent_id = %q, want sess-parent", d.ParentID)
	}

	// The parent descriptor was NOT created by the fork (parent unaffected in the registry).
	if _, ok, _ := reg.Get("sess-parent"); ok {
		t.Fatal("fork wrote a parent descriptor; the parent must be unaffected")
	}
}

// TestRunSessionBranchUsageErrors covers the argument guardrails.
func TestRunSessionBranchUsageErrors(t *testing.T) {
	parentDir := dumpParentImage(t, "sess-parent")
	var out, errb bytes.Buffer
	// Missing --out.
	if rc := runSessionBranch(&out, &errb, []string{parentDir}); rc != 2 {
		t.Fatalf("missing --out rc=%d, want 2", rc)
	}
	// No parent arg.
	if rc := runSessionBranch(&out, &errb, []string{"--out", t.TempDir()}); rc != 2 {
		t.Fatalf("missing parent rc=%d, want 2", rc)
	}
	// A non-existent parent dir is a runtime error (fail closed on a missing checkpoint).
	if rc := runSessionBranch(&out, &errb, []string{filepath.Join(os.TempDir(), "does-not-exist-1200"), "--out", t.TempDir()}); rc != 1 {
		t.Fatalf("missing parent dir rc=%d, want 1", rc)
	}
}

// TestRunSessionBranchDefaultID checks the derived id when --id is omitted.
func TestRunSessionBranchDefaultID(t *testing.T) {
	parentDir := dumpParentImage(t, "sess-parent")
	branchDir := filepath.Join(t.TempDir(), "b")
	var out, errb bytes.Buffer
	if rc := runSessionBranch(&out, &errb, []string{parentDir, "--out", branchDir}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	img, err := sessionimage.LoadDir(branchDir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if img.Meta.SessionID != "sess-parent-branch" {
		t.Fatalf("default branch id = %q, want sess-parent-branch", img.Meta.SessionID)
	}
}
