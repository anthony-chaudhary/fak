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
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// dumpForkParent writes a small, whole parent session (drive + recall content + a two-turn
// trajectory) that `fak session fork` can snapshot-and-branch, and returns its bundle dir.
func dumpForkParent(t *testing.T, id string) string {
	t.Helper()
	dir := t.TempDir()
	rec := recall.NewRecorder(id)
	rec.Record(context.Background(), "get_user_details", []byte(`{"user":"mia"}`))
	rec.Record(context.Background(), "search_flights", []byte("UA123 $310"))
	in := sessionimage.Input{
		SessionID: id,
		Drive:     session.State{TraceID: id, Run: session.Throttled, Budget: session.Budget{TurnsLeft: 4}, Priority: 2},
		Recorder:  rec,
		Trajectory: []trajectory.Turn{
			{TraceID: id, Seq: 1, Query: "what refund fee?", Tool: "get_user_details", Verdict: "ALLOW"},
			{TraceID: id, Seq: 2, Tool: "search_flights", Verdict: "ALLOW"},
		},
		Model: "model-A",
		Host:  "laptop",
		Now:   1_700_000_000,
	}
	if _, err := sessionimage.DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	return dir
}

// TestRunSessionFork is the #2761 acceptance witness driven through the CLI: an operator forks
// a running session; the fork shares history up to the branch point, then the original and the
// fork DRIVE independently and diverge after it — with the original bundle left untouched.
func TestRunSessionFork(t *testing.T) {
	parentDir := dumpForkParent(t, "sess-parent")
	parentBefore := readForkBundle(t, parentDir)
	forkDir := filepath.Join(t.TempDir(), "fork")
	cpDir := filepath.Join(t.TempDir(), "cp")
	registry := filepath.Join(t.TempDir(), "registry.json")

	var stdout, stderr bytes.Buffer
	rc := runSessionFork(&stdout, &stderr, []string{
		parentDir, "--out", forkDir, "--checkpoint", cpDir, "--id", "sess-child", "--registry", registry,
	})
	if rc != 0 {
		t.Fatalf("runSessionFork rc=%d stderr=%s", rc, stderr.String())
	}
	if got := stdout.String(); !strings.Contains(got, "forked sess-parent -> sess-child") {
		t.Fatalf("summary missing lineage line: %q", got)
	}

	// --- shared prefix: the fork shares history up to the branch point ---
	fork, err := sessionimage.LoadDir(forkDir)
	if err != nil {
		t.Fatalf("LoadDir fork: %v", err)
	}
	if fork.Meta.SessionID != "sess-child" || fork.Meta.ParentID != "sess-parent" {
		t.Fatalf("fork identity = id %q parent %q, want sess-child/sess-parent", fork.Meta.SessionID, fork.Meta.ParentID)
	}
	parent, err := sessionimage.LoadDir(parentDir)
	if err != nil {
		t.Fatalf("LoadDir parent: %v", err)
	}
	// The fork inherits the parent's drive verbatim except the re-keyed trace: same branch point.
	if fork.Drive.TraceID != "sess-child" {
		t.Fatalf("fork drive TraceID = %q, want the fresh trace sess-child", fork.Drive.TraceID)
	}
	rekeyed := fork.Drive
	rekeyed.TraceID = parent.Drive.TraceID
	if !reflect.DeepEqual(rekeyed, parent.Drive) {
		t.Fatalf("fork drive did not inherit the branch-point state (beyond the trace):\n fork=%+v\n parent=%+v", fork.Drive, parent.Drive)
	}
	forkTraj, err := fork.Trajectory()
	if err != nil {
		t.Fatalf("fork trajectory: %v", err)
	}
	parentTraj, err := parent.Trajectory()
	if err != nil {
		t.Fatalf("parent trajectory: %v", err)
	}
	if len(forkTraj) == 0 || !reflect.DeepEqual(forkTraj, parentTraj) {
		t.Fatalf("fork trajectory != parent trajectory at branch point:\n fork=%+v\n parent=%+v", forkTraj, parentTraj)
	}
	// The branch point is a checkpoint of the parent (same id), pinned durably.
	bp, err := sessionimage.LoadDir(cpDir)
	if err != nil {
		t.Fatalf("LoadDir branch point: %v", err)
	}
	if bp.Meta.SessionID != "sess-parent" {
		t.Fatalf("branch-point id = %q, want sess-parent (a checkpoint preserves the id)", bp.Meta.SessionID)
	}

	// --- independent post-fork drive state on BOTH ---
	// Advance each line differently. A running session writes a fresh generation (copy-on-write
	// divergence), so each generation goes to its own dir.
	parentNext := driveSessionForward(t, parent, session.Running,
		trajectory.Turn{TraceID: "sess-parent", Seq: 3, Tool: "parent_only_step", Verdict: "ALLOW"})
	forkNext := driveSessionForward(t, fork, session.Paused,
		trajectory.Turn{TraceID: "sess-child", Seq: 3, Tool: "fork_only_step", Verdict: "ALLOW"})

	if parentNext.Drive.Run == forkNext.Drive.Run {
		t.Fatalf("post-fork run-state did not diverge: both %v (want independent drive on each line)", parentNext.Drive.Run)
	}
	pTail, err := parentNext.Trajectory()
	if err != nil {
		t.Fatalf("parentNext trajectory: %v", err)
	}
	fTail, err := forkNext.Trajectory()
	if err != nil {
		t.Fatalf("forkNext trajectory: %v", err)
	}
	// Both retained the entire shared prefix...
	if !reflect.DeepEqual(pTail[:len(forkTraj)], forkTraj) || !reflect.DeepEqual(fTail[:len(forkTraj)], forkTraj) {
		t.Fatal("a divergent line dropped the shared history prefix")
	}
	// ...but their tails diverge at the post-branch step.
	if reflect.DeepEqual(pTail, fTail) || pTail[len(pTail)-1].Tool == fTail[len(fTail)-1].Tool {
		t.Fatalf("post-fork trajectories did not diverge: parent tail %q fork tail %q",
			pTail[len(pTail)-1].Tool, fTail[len(fTail)-1].Tool)
	}

	// --- the fork is a new C1 descriptor; the parent bundle is untouched ---
	reg := session.NewRegistry(session.NewFileStore(registry))
	d, ok, err := reg.Get("sess-child")
	if err != nil || !ok {
		t.Fatalf("fork descriptor absent: ok=%v err=%v", ok, err)
	}
	if d.ParentID != "sess-parent" {
		t.Fatalf("descriptor parent_id = %q, want sess-parent", d.ParentID)
	}
	if _, ok, _ := reg.Get("sess-parent"); ok {
		t.Fatal("fork wrote a parent descriptor; the parent must be unaffected")
	}
	parentAfter := readForkBundle(t, parentDir)
	if len(parentAfter) != len(parentBefore) {
		t.Fatalf("fork changed the parent bundle's file set (%d -> %d); the original must be untouched", len(parentBefore), len(parentAfter))
	}
	for name, before := range parentBefore {
		if string(parentAfter[name]) != string(before) {
			t.Fatalf("parent %s changed by the fork (the original must be untouched)", name)
		}
	}
}

// TestRunSessionForkDefaultID checks the derived fork id when --id is omitted.
func TestRunSessionForkDefaultID(t *testing.T) {
	parentDir := dumpForkParent(t, "sess-parent")
	var out, errb bytes.Buffer
	rc := runSessionFork(&out, &errb, []string{
		parentDir, "--out", filepath.Join(t.TempDir(), "f"), "--checkpoint", filepath.Join(t.TempDir(), "c"),
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if got := out.String(); !strings.Contains(got, "forked sess-parent -> sess-parent-fork") {
		t.Fatalf("default fork id summary = %q, want the derived sess-parent-fork", got)
	}
}

// TestRunSessionForkUsageErrors covers the argument guardrails.
func TestRunSessionForkUsageErrors(t *testing.T) {
	parentDir := dumpForkParent(t, "sess-parent")
	var out, errb bytes.Buffer
	// Missing --out.
	if rc := runSessionFork(&out, &errb, []string{parentDir, "--checkpoint", t.TempDir()}); rc != 2 {
		t.Fatalf("missing --out rc=%d, want 2", rc)
	}
	// Missing --checkpoint (fork must pin a branch point).
	if rc := runSessionFork(&out, &errb, []string{parentDir, "--out", t.TempDir()}); rc != 2 {
		t.Fatalf("missing --checkpoint rc=%d, want 2", rc)
	}
	// No parent arg.
	if rc := runSessionFork(&out, &errb, []string{"--out", t.TempDir(), "--checkpoint", t.TempDir()}); rc != 2 {
		t.Fatalf("missing parent rc=%d, want 2", rc)
	}
	// A non-existent parent dir fails closed (no fork of a checkpoint that does not exist).
	if rc := runSessionFork(&out, &errb, []string{
		filepath.Join(os.TempDir(), "does-not-exist-2761"), "--out", t.TempDir(), "--checkpoint", t.TempDir(),
	}); rc != 1 {
		t.Fatalf("missing parent dir rc=%d, want 1", rc)
	}
}

// --- helpers ---

// driveSessionForward simulates a loaded session advancing one step: it re-dumps the image
// with a new run-state and one appended trajectory turn into a FRESH generation dir, and
// returns the loaded next generation.
func driveSessionForward(t *testing.T, img *sessionimage.Image, run session.RunState, next trajectory.Turn) *sessionimage.Image {
	t.Helper()
	traj, err := img.Trajectory()
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	drive := img.Drive
	drive.Run = run
	genDir := t.TempDir()
	if _, err := sessionimage.DumpDir(genDir, sessionimage.Input{
		SessionID:  img.Meta.SessionID,
		Drive:      drive,
		Trajectory: append(append([]trajectory.Turn(nil), traj...), next),
		Now:        img.Meta.UpdatedUnix + 1,
	}); err != nil {
		t.Fatalf("drive forward: %v", err)
	}
	nextImg, err := sessionimage.LoadDir(genDir)
	if err != nil {
		t.Fatalf("load next gen: %v", err)
	}
	return nextImg
}

// readForkBundle snapshots every file in a bundle dir so a test can prove the parent was only
// read (byte-for-byte unchanged) by the fork op.
func readForkBundle(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir %s: %v", dir, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("ReadFile %s: %v", e.Name(), err)
		}
		out[e.Name()] = b
	}
	return out
}
