package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func TestWIPQueuePriorityDedupeAndUnknownSafety(t *testing.T) {
	now := time.Unix(2_000, 0).UTC()
	rows := []worktreeWorkerLifecycleRow{
		lifecycleFixture("/fleet/healthy", "h5", "h5", worktreeEvidenceLive, worktreeEvidenceLive, worktreeEvidenceDirty, worktreeLifecycleReady, false),
		lifecycleFixture("/fleet/cold", "h4", "h4", worktreeEvidenceDead, worktreeEvidenceReleased, worktreeEvidenceClean, worktreeLifecycleCold, true),
		lifecycleFixture("/fleet/unlanded", "h2", "base", worktreeEvidenceDead, worktreeEvidenceReleased, worktreeEvidenceClean, worktreeLifecycleRetained, false),
		lifecycleFixture("/fleet/dirty", "h1", "h1", worktreeEvidenceDead, worktreeEvidenceReleased, worktreeEvidenceDirty, worktreeLifecycleDirty, false),
		lifecycleFixture("/fleet/unknown", "hu", "hu", worktreeEvidenceUnknown, worktreeEvidenceUnknown, worktreeEvidenceUnknown, worktreeLifecycleUnknown, false),
	}
	checkpoints := []wipinventory.Checkpoint{
		{Ref: "refs/fak/wip/matched", SHA: "h2", Unix: 1_900, Known: true},
		{Ref: "refs/fak/wip/local", SHA: "h3", Unix: 1_800, Known: true},
	}
	out := buildWIPQueue("/repo", now, rows, checkpoints, nil, func(string) (os.FileInfo, error) { return nil, os.ErrNotExist })
	if out.Count != 6 {
		t.Fatalf("count=%d, want 6 (matched checkpoint deduped): %#v", out.Count, out.Rows)
	}
	wantPriorities := []int{1, 1, 2, 3, 4, 5}
	for i, want := range wantPriorities {
		if out.Rows[i].Priority != want {
			t.Fatalf("row %d priority=%d want %d: %#v", i, out.Rows[i].Priority, want, out.Rows[i])
		}
		if out.Rows[i].Reason == "" || out.Rows[i].Risk == "" || out.Rows[i].State == "" || out.Rows[i].NextCommand == "" || len(out.Rows[i].Provenance) == 0 {
			t.Fatalf("row %d missing required action evidence: %#v", i, out.Rows[i])
		}
	}
	if out.Rows[1].ID != "/fleet/unknown" || out.Rows[1].Risk != wipQueueRiskProtect {
		t.Fatalf("unknown evidence was not protected conservatively: %#v", out.Rows[1])
	}
	var matched *wipQueueRow
	for i := range out.Rows {
		if out.Rows[i].ID == "/fleet/unlanded" {
			matched = &out.Rows[i]
		}
	}
	if matched == nil || len(matched.Provenance) != 2 {
		t.Fatalf("concrete SHA checkpoint association missing: %#v", matched)
	}
	for _, row := range out.Rows {
		if row.Kind == "CHECKPOINT" {
			want := "fak wip restore -C /repo local"
			if row.NextCommand != want {
				t.Fatalf("checkpoint next command=%q want %q", row.NextCommand, want)
			}
		}
	}
}

func TestWIPQueueAmbiguousSHAIsNotDeduped(t *testing.T) {
	rows := []worktreeWorkerLifecycleRow{
		lifecycleFixture("/fleet/a", "same", "base", worktreeEvidenceDead, worktreeEvidenceReleased, worktreeEvidenceClean, worktreeLifecycleRetained, false),
		lifecycleFixture("/fleet/b", "same", "base", worktreeEvidenceDead, worktreeEvidenceReleased, worktreeEvidenceClean, worktreeLifecycleRetained, false),
	}
	out := buildWIPQueue("/repo", time.Unix(2_000, 0), rows, []wipinventory.Checkpoint{{Ref: "refs/fak/wip/c", SHA: "same", Known: true}}, nil, func(string) (os.FileInfo, error) { return nil, os.ErrNotExist })
	if out.Count != 3 || out.Rows[2].Kind != "CHECKPOINT" {
		t.Fatalf("ambiguous SHA must preserve checkpoint separately: %#v", out.Rows)
	}
}

func TestWIPQueueStableTieBreakAndAge(t *testing.T) {
	dir := t.TempDir()
	older := filepath.Join(dir, "B")
	newer := filepath.Join(dir, "a")
	for _, path := range []string{older, newer} {
		if err := os.Mkdir(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	stamp := time.Unix(1_000, 0)
	if err := os.Chtimes(older, stamp, stamp); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(newer, stamp, stamp); err != nil {
		t.Fatal(err)
	}
	rows := []worktreeWorkerLifecycleRow{
		lifecycleFixture(older, "b", "b", worktreeEvidenceLive, worktreeEvidenceLive, worktreeEvidenceClean, worktreeLifecycleReady, false),
		lifecycleFixture(newer, "a", "a", worktreeEvidenceLive, worktreeEvidenceLive, worktreeEvidenceClean, worktreeLifecycleReady, false),
	}
	out := buildWIPQueue("/repo", time.Unix(2_000, 0), rows, nil, nil, os.Stat)
	if out.Rows[0].ID != filepath.ToSlash(newer) || !out.Rows[0].Age.Known || out.Rows[0].Age.Seconds != 1_000 {
		t.Fatalf("unstable normalized-path order or age: %#v", out.Rows)
	}
}

func TestWIPQueueDispatchAndUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runWip(&out, &errOut, []string{"queue", "unexpected"}); code != 2 {
		t.Fatalf("dispatch code=%d want 2; stderr=%s", code, errOut.String())
	}
	out.Reset()
	wipUsage(&out)
	if !strings.Contains(out.String(), "fak wip queue [--json]") {
		t.Fatalf("usage missing queue: %s", out.String())
	}
}

func TestWIPQueueTextRendersProvenance(t *testing.T) {
	out := wipQueueOut{Count: 1, Rows: []wipQueueRow{{
		Priority: 1, Kind: "WORKTREE", ID: "/fleet/dirty", Reason: "DIRTY_DEAD_OR_UNKNOWN_PROTECTION",
		Risk: wipQueueRiskProtect, Owner: wipQueueOwner{State: "DEAD", PID: 42, LeaseState: "RELEASED", LeaseID: "issue", Lane: "cmd"}, Age: wipQueueAge{Basis: "worktree_path_mtime"},
		State: "DIRTY", NextCommand: "git -C /fleet/dirty status --short",
		Provenance: []wipQueueProvenance{{Source: "WORKTREE_LIFECYCLE", ID: "/fleet/dirty", SHA: "abc"}},
	}}}
	var rendered bytes.Buffer
	renderWIPQueue(&rendered, out)
	for _, want := range []string{"reason: DIRTY_DEAD_OR_UNKNOWN_PROTECTION", "owner: state=DEAD pid=42 lease-state=RELEASED lease-id=issue lane=cmd", "provenance: source=WORKTREE_LIFECYCLE id=/fleet/dirty sha=abc", "next: git -C /fleet/dirty status --short"} {
		if !strings.Contains(rendered.String(), want) {
			t.Fatalf("text output missing %q: %s", want, rendered.String())
		}
	}
}

func lifecycleFixture(path, head, base string, owner, lease, clean worktreeWorkerEvidenceState, state worktreeWorkerLifecycleState, reapable bool) worktreeWorkerLifecycleRow {
	return worktreeWorkerLifecycleRow{
		Path: path, HeadSHA: head, BaseSHA: base,
		Association:   worktreeWorkerAssociation{State: worktreeEvidenceAssociated, Lane: "cmd", OwnerPID: 42, LeaseID: "issue"},
		Liveness:      worktreeWorkerLiveness{Owner: owner, Lease: lease},
		Cleanliness:   worktreeWorkerCleanliness{State: clean},
		Lifecycle:     state,
		ReapReadiness: worktreeWorkerReapReadiness{Reapable: reapable, Verdict: map[bool]worktreeWorkerReapVerdict{true: worktreeReapable, false: worktreeKeep}[reapable]},
	}
}
