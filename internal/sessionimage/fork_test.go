package sessionimage

import (
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// TestForkDirSnapshotAndBranch is the load-bearing #2761 witness: forking a running session
// pins an immutable checkpoint (the branch point) and mints a SECOND session under a fresh
// trace that shares history up to that point; the two then DRIVE independently and diverge,
// and the original bundle is left untouched by the fork op.
func TestForkDirSnapshotAndBranch(t *testing.T) {
	parentDir := t.TempDir()
	if _, err := DumpDir(parentDir, buildInput(t, "sess-parent")); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	// Snapshot the parent's bytes up front to prove the fork op only READS it.
	parentBefore := readBundle(t, parentDir)

	cpDir := filepath.Join(t.TempDir(), "cp")
	forkDir := filepath.Join(t.TempDir(), "fork")
	res, err := ForkDir(parentDir, cpDir, forkDir, ForkOptions{ForkID: "sess-fork", Reason: "explore-redirect", Now: 1_700_000_100})
	if err != nil {
		t.Fatalf("ForkDir: %v", err)
	}

	// --- shared prefix: the fork shares history up to the branch point ---
	// The branch point is the parent captured (same id); the fork is a distinct second session.
	if res.BranchPoint.SessionID != "sess-parent" {
		t.Fatalf("branch point id = %q, want sess-parent (a checkpoint preserves the id)", res.BranchPoint.SessionID)
	}
	if res.Fork.SessionID != "sess-fork" {
		t.Fatalf("fork id = %q, want sess-fork", res.Fork.SessionID)
	}
	if res.Fork.ParentID != "sess-parent" {
		t.Fatalf("fork parent_id = %q, want sess-parent", res.Fork.ParentID)
	}

	fork, err := LoadDir(forkDir)
	if err != nil {
		t.Fatalf("LoadDir fork: %v", err)
	}
	parent, err := LoadDir(parentDir)
	if err != nil {
		t.Fatalf("LoadDir parent: %v", err)
	}
	// The fork inherits the parent's drive verbatim EXCEPT the re-keyed trace: same run-state,
	// same budget — the two stand at the same branch point.
	if fork.Drive.TraceID != "sess-fork" {
		t.Fatalf("fork drive TraceID = %q, want the fresh trace sess-fork", fork.Drive.TraceID)
	}
	forkDriveRekeyed := fork.Drive
	forkDriveRekeyed.TraceID = parent.Drive.TraceID
	if !reflect.DeepEqual(forkDriveRekeyed, parent.Drive) {
		t.Fatalf("fork drive did not inherit the branch-point state (beyond the trace):\n fork=%+v\n parent=%+v", fork.Drive, parent.Drive)
	}
	// Shared trajectory prefix: the fork carries the parent's history up to the branch point,
	// verbatim (the trajectory rides copy-on-write; only the drive is re-keyed).
	forkTraj, err := fork.Trajectory()
	if err != nil {
		t.Fatalf("fork trajectory: %v", err)
	}
	parentTraj, err := parent.Trajectory()
	if err != nil {
		t.Fatalf("parent trajectory: %v", err)
	}
	if len(forkTraj) == 0 {
		t.Fatal("expected a non-empty shared history prefix")
	}
	if !reflect.DeepEqual(forkTraj, parentTraj) {
		t.Fatalf("fork trajectory != parent trajectory at branch point:\n fork=%+v\n parent=%+v", forkTraj, parentTraj)
	}
	// Lineage is an audited fact: checkpoint then branch.
	last := res.Fork.Migrations[len(res.Fork.Migrations)-1].Reason
	if !strings.Contains(last, "branched from sess-parent at ") {
		t.Fatalf("fork lineage = %q, want a 'branched from sess-parent at' entry", last)
	}

	// --- independent post-fork drive state on BOTH ---
	// Drive each line forward differently. A running session advancing writes a FRESH
	// generation (copy-on-write divergence — it never truncates the shared branch-point bytes),
	// so each generation is dumped into its own dir.
	parentNext := driveForward(t, parent, session.Running,
		trajectory.Turn{TraceID: "sess-parent", Seq: 3, Tool: "parent_only_step", Verdict: "ALLOW"})
	forkNext := driveForward(t, fork, session.Paused,
		trajectory.Turn{TraceID: "sess-fork", Seq: 3, Tool: "fork_only_step", Verdict: "ALLOW"})

	// The drive state diverged: the two lines now hold different run-states.
	if parentNext.Drive.Run == forkNext.Drive.Run {
		t.Fatalf("post-fork run-state did not diverge: both %v (want the two lines driven independently)", parentNext.Drive.Run)
	}
	pTail, err := parentNext.Trajectory()
	if err != nil {
		t.Fatalf("parentNext trajectory: %v", err)
	}
	fTail, err := forkNext.Trajectory()
	if err != nil {
		t.Fatalf("forkNext trajectory: %v", err)
	}
	// Both retained the ENTIRE shared prefix...
	if !reflect.DeepEqual(pTail[:len(forkTraj)], forkTraj) {
		t.Fatal("the advanced parent dropped the shared history prefix")
	}
	if !reflect.DeepEqual(fTail[:len(forkTraj)], forkTraj) {
		t.Fatal("the advanced fork dropped the shared history prefix")
	}
	// ...but their tails diverge at the post-branch step.
	if reflect.DeepEqual(pTail, fTail) {
		t.Fatal("post-fork trajectories did not diverge — the two lines must have distinct tails")
	}
	if pTail[len(pTail)-1].Tool == fTail[len(fTail)-1].Tool {
		t.Fatalf("divergent tails share their last step %q — the lines did not actually diverge", pTail[len(pTail)-1].Tool)
	}

	// --- the branch point is immutable + the parent bundle untouched by the fork op ---
	bp, err := LoadDir(cpDir)
	if err != nil {
		t.Fatalf("LoadDir branch point: %v", err)
	}
	bpTraj, err := bp.Trajectory()
	if err != nil {
		t.Fatalf("branch-point trajectory: %v", err)
	}
	if !reflect.DeepEqual(bpTraj, forkTraj) {
		t.Fatal("the pinned branch point drifted from the shared prefix — a checkpoint must be immutable")
	}
	parentAfter := readBundle(t, parentDir)
	for name, before := range parentBefore {
		if string(parentAfter[name]) != string(before) {
			t.Fatalf("parent %s changed by the fork (the original must be untouched)", name)
		}
	}
	if len(parentAfter) != len(parentBefore) {
		t.Fatalf("fork wrote %d new file(s) into the parent bundle (must be untouched)", len(parentAfter)-len(parentBefore))
	}
}

// TestForkDirRejectsBadShape covers the closed-reason guardrails: a blank fork id, a
// parent-colliding fork id, and colliding directories are refused before anything is written.
func TestForkDirRejectsBadShape(t *testing.T) {
	parentDir := t.TempDir()
	if _, err := DumpDir(parentDir, buildInput(t, "sess-parent")); err != nil {
		t.Fatalf("DumpDir parent: %v", err)
	}
	cpDir := filepath.Join(t.TempDir(), "cp")
	forkDir := filepath.Join(t.TempDir(), "fork")

	assertForkRefusal := func(name string, cp, fk string, opts ForkOptions, want ForkRefuseReason) {
		_, err := ForkDir(parentDir, cp, fk, opts)
		var ref *ForkRefusal
		if !errors.As(err, &ref) {
			t.Fatalf("%s: err = %v, want a *ForkRefusal", name, err)
		}
		if ref.Reason != want {
			t.Fatalf("%s: refusal reason = %q, want %q", name, ref.Reason, want)
		}
	}

	assertForkRefusal("blank id", cpDir, forkDir, ForkOptions{ForkID: ""}, ForkMalformed)
	assertForkRefusal("parent-colliding id", cpDir, forkDir, ForkOptions{ForkID: "sess-parent"}, ForkSameID)
	// Colliding dirs (checkpoint == fork) is a shape refusal before any write.
	assertForkRefusal("colliding dirs", forkDir, forkDir, ForkOptions{ForkID: "sess-fork"}, ForkMalformed)
}

// --- helpers ---

// driveForward simulates a session advancing one step: it re-dumps the loaded image with a
// new run-state and one appended trajectory turn into a FRESH generation dir (copy-on-write
// divergence — the running line writes new content rather than truncating the shared bytes),
// and returns the loaded next generation.
func driveForward(t *testing.T, img *Image, run session.RunState, next trajectory.Turn) *Image {
	t.Helper()
	traj, err := img.Trajectory()
	if err != nil {
		t.Fatalf("read trajectory: %v", err)
	}
	drive := img.Drive
	drive.Run = run
	genDir := t.TempDir()
	if _, err := DumpDir(genDir, Input{
		SessionID:  img.Meta.SessionID,
		Drive:      drive,
		Trajectory: append(append([]trajectory.Turn(nil), traj...), next),
		Now:        img.Meta.UpdatedUnix + 1,
	}); err != nil {
		t.Fatalf("drive forward: %v", err)
	}
	nextImg, err := LoadDir(genDir)
	if err != nil {
		t.Fatalf("load next gen: %v", err)
	}
	return nextImg
}
