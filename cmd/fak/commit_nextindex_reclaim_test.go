package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// nextIndexResidueReport is a lane report with no index.lock at all, carrying only
// next-index residue: two stale dead-owner files (reapable), one young file and one
// whose named owner pid is still alive (both kept).
func nextIndexResidueReport() commitlane.Report {
	return commitlane.Report{
		Schema:       commitlane.Schema,
		ProcessProbe: "ok",
		IndexLock:    commitlane.IndexLock{Path: "repo/.git/index.lock"},
		NextIndexLocks: []commitlane.NextIndexLock{
			{Path: "repo/.git/next-index-11228.lock", PID: 11228, StaleHint: true},
			{Path: "repo/.git/next-index-12640.lock", PID: 12640, StaleHint: true},
			{Path: "repo/.git/next-index-30000.lock", PID: 30000, StaleHint: false},
			{Path: "repo/.git/next-index-40000.lock", PID: 40000, StaleHint: true, OwnerAlive: true},
		},
	}
}

// TestReclaimNextIndexApplyReapsOnlyStaleOrphans is DoD item 3 (#5338): N stale
// next-index files with no live writer are reaped, and a young / live-owner one is kept.
func TestReclaimNextIndexApplyReapsOnlyStaleOrphans(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return nextIndexResidueReport(), nil
	})
	var removed []string
	withRemoveIndexLockFn(t, func(p string) error { removed = append(removed, p); return nil })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	sort.Strings(removed)
	want := []string{"repo/.git/next-index-11228.lock", "repo/.git/next-index-12640.lock"}
	if strings.Join(removed, ",") != strings.Join(want, ",") {
		t.Fatalf("removed = %v, want only the two stale dead-owner files %v", removed, want)
	}
	if !strings.Contains(out.String(), "reclaimed 2 of 4 stale orphan(s)") {
		t.Fatalf("output missing the reap summary:\n%s", out.String())
	}
	// The keep reasons must be visible so an operator sees WHY two survived.
	if !strings.Contains(out.String(), "keep_fresh=1") || !strings.Contains(out.String(), "keep_live_owner=1") {
		t.Fatalf("output missing keep reasons:\n%s", out.String())
	}
}

func TestReclaimNextIndexDryRunRemovesNothing(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return nextIndexResidueReport(), nil
	})
	called := false
	withRemoveIndexLockFn(t, func(string) error { called = true; return nil })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if called {
		t.Fatal("dry-run must NOT remove next-index residue")
	}
	if !strings.Contains(out.String(), "WOULD reclaim 2") {
		t.Fatalf("dry-run output missing the WOULD line:\n%s", out.String())
	}
}

// TestReclaimNextIndexRefusesWhenLiveWriter proves a live writer keeps EVERY file, even
// the ones whose own named owner pid is dead: an unrelated live index writer means the
// index is in flight and the residue may still be claimed.
func TestReclaimNextIndexRefusesWhenLiveWriter(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		rep := nextIndexResidueReport()
		rep.LiveWriters = []commitlane.ProcessFact{{PID: 4242, Match: "git_writer"}}
		return rep, nil
	})
	called := false
	withRemoveIndexLockFn(t, func(string) error { called = true; return nil })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if called {
		t.Fatal("must NOT remove residue while a live writer is present")
	}
	if !strings.Contains(out.String(), "no reclaim (keep_live_writer=4)") {
		t.Fatalf("output missing the keep summary:\n%s", out.String())
	}
}

// TestReclaimNextIndexRemoveErrorFailsButContinues proves one unremovable file reports a
// failure exit WITHOUT abandoning the rest of the sweep.
func TestReclaimNextIndexRemoveErrorFailsButContinues(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return nextIndexResidueReport(), nil
	})
	withRemoveIndexLockFn(t, func(p string) error {
		if strings.Contains(p, "11228") {
			return errors.New("permission denied")
		}
		return nil
	})

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 1 {
		t.Fatalf("a real remove error should exit 1, got %d out=%q", code, out.String())
	}
	if !strings.Contains(errb.String(), "next-index reclaim failed") {
		t.Fatalf("stderr missing the failure:\n%s", errb.String())
	}
	if !strings.Contains(out.String(), "reclaimed 1 of 4") {
		t.Fatalf("the sweep should continue past one failure:\n%s", out.String())
	}
}

// TestReclaimNextIndexAlreadyClearedIsIdempotent: a peer session reclaiming the same
// residue concurrently is the expected case on a shared trunk, not a failure.
func TestReclaimNextIndexAlreadyClearedIsIdempotent(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return nextIndexResidueReport(), nil
	})
	withRemoveIndexLockFn(t, func(string) error { return os.ErrNotExist })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 0 {
		t.Fatalf("already-cleared should be success, got exit %d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "reclaimed 2 of 4") {
		t.Fatalf("output missing the reap summary:\n%s", out.String())
	}
}

// TestReclaimNextIndexSilentWhenNoResidue pins that a clean repo's reclaim output is
// unchanged — the next-index sweep prints nothing when there is nothing to sweep.
func TestReclaimNextIndexSilentWhenNoResidue(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return reapableReport(), nil
	})
	withRemoveIndexLockFn(t, func(string) error { return nil })

	var out, errb bytes.Buffer
	if code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"}); code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if strings.Contains(out.String(), "next-index") {
		t.Fatalf("no residue should print no next-index line:\n%s", out.String())
	}
}

// TestCommitReclaimAliasRunsRecoveryWithoutCommitting is DoD item 2 (#5338): the flag is
// reachable from `fak commit` itself, needs neither -m nor --path, and never reaches the
// commit machinery.
func TestCommitReclaimAliasRunsRecoveryWithoutCommitting(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return nextIndexResidueReport(), nil
	})
	var removed []string
	withRemoveIndexLockFn(t, func(p string) error { removed = append(removed, p); return nil })

	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if len(removed) != 2 {
		t.Fatalf("alias removed %v, want the 2 stale dead-owner files", removed)
	}
	if !strings.Contains(out.String(), "reclaimed 2 of 4 stale orphan(s)") {
		t.Fatalf("alias output missing the reap summary:\n%s", out.String())
	}
	// It must NOT have demanded a message or a path (the normal `fak commit` refusals).
	if strings.Contains(errb.String(), "message is required") || strings.Contains(errb.String(), "--path") {
		t.Fatalf("recovery mode must not require a message or paths:\n%s", errb.String())
	}
}

// TestCommitHelpAdvertisesReclaimFlag is the discoverability assertion behind DoD item 2
// (#5338): the whole point of the alias is that `fak commit --help` — the one place a
// wedged committer looks — names the reclaim. A flag that exists but is unlisted there
// is the exact gap this issue was filed for.
func TestCommitHelpAdvertisesReclaimFlag(t *testing.T) {
	var out, errb bytes.Buffer
	runCommitCommand(&out, &errb, []string{"--help"})
	help := out.String() + errb.String()
	if !strings.Contains(help, "reclaim-stale-index-lock") {
		t.Fatalf("`fak commit --help` must list the reclaim flag:\n%s", help)
	}
	if !strings.Contains(help, "next-index") {
		t.Fatalf("`fak commit --help` should name the next-index residue it sweeps:\n%s", help)
	}
}

// TestCommitLockBusyRefusalNamesTheReclaimPath is the other half of DoD item 2 (#5338):
// the refusal a wedged committer actually sees must name the way out, since that is the
// one moment they cannot go hunting for it. Non-lock refusals stay unchanged.
func TestCommitLockBusyRefusalNamesTheReclaimPath(t *testing.T) {
	var busy bytes.Buffer
	renderCommitResult(&busy, safecommit.Result{Reason: safecommit.ReasonLockBusy, Detail: "held by pid 4242"})
	if !strings.Contains(busy.String(), "--reclaim-stale-index-lock") {
		t.Fatalf("LOCK_BUSY refusal must name the reclaim path:\n%s", busy.String())
	}

	var other bytes.Buffer
	renderCommitResult(&other, safecommit.Result{Reason: safecommit.ReasonOffTrunk})
	if strings.Contains(other.String(), "--reclaim-stale-index-lock") {
		t.Fatalf("an unrelated refusal must not advertise the reclaim path:\n%s", other.String())
	}
}

// TestCommitReclaimAliasDryRunIsDefault pins that the alias inherits the dry-run default:
// discoverability must not make deletion the accident.
func TestCommitReclaimAliasDryRunIsDefault(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return nextIndexResidueReport(), nil
	})
	called := false
	withRemoveIndexLockFn(t, func(string) error { called = true; return nil })

	var out, errb bytes.Buffer
	if code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-index-lock"}); code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if called {
		t.Fatal("the alias must default to a dry run")
	}
	if !strings.Contains(out.String(), "re-run with --apply") {
		t.Fatalf("alias dry-run missing the --apply hint:\n%s", out.String())
	}
}
