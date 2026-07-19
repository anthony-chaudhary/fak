package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
)

// reapableReport is the orphaned-index.lock signature: present, probe ran clean, no
// live writer, stale past the grace window — the one shape DecideIndexLockReclaim reaps.
func reapableReport() commitlane.Report {
	return commitlane.Report{
		Schema:       commitlane.Schema,
		ProcessProbe: "ok",
		IndexLock:    commitlane.IndexLock{Path: "repo/.git/index.lock", Present: true, StaleHint: true},
	}
}

func withRemoveIndexLockFn(t *testing.T, fn func(string) error) {
	t.Helper()
	prev := removeIndexLockFn
	removeIndexLockFn = fn
	t.Cleanup(func() { removeIndexLockFn = prev })
}

func TestReclaimStaleIndexLockApplyRemoves(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return reapableReport(), nil
	})
	var removed string
	withRemoveIndexLockFn(t, func(p string) error { removed = p; return nil })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if removed != "repo/.git/index.lock" {
		t.Fatalf("removed = %q, want the index.lock path", removed)
	}
	if !strings.Contains(out.String(), "reclaimed stale orphan") {
		t.Fatalf("output missing reclaim confirmation:\n%s", out.String())
	}
}

func TestReclaimStaleIndexLockDryRunKeepsLock(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return reapableReport(), nil
	})
	called := false
	withRemoveIndexLockFn(t, func(string) error { called = true; return nil })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%q", code, errb.String())
	}
	if called {
		t.Fatal("dry-run must NOT remove the lock")
	}
	if !strings.Contains(out.String(), "WOULD reclaim") {
		t.Fatalf("dry-run output missing WOULD reclaim:\n%s", out.String())
	}
}

func TestReclaimStaleIndexLockRefusesWhenLiveWriter(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		rep := reapableReport()
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
		t.Fatal("must NOT remove a lock while a live writer is present")
	}
	if !strings.Contains(out.String(), "no reclaim (keep_live_writer)") {
		t.Fatalf("output missing keep reason:\n%s", out.String())
	}
}

func TestReclaimStaleIndexLockAlreadyClearedIsIdempotent(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return reapableReport(), nil
	})
	withRemoveIndexLockFn(t, func(string) error { return os.ErrNotExist })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 0 {
		t.Fatalf("already-cleared should be success, got exit %d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "already cleared") {
		t.Fatalf("output missing already-cleared:\n%s", out.String())
	}
}

func TestReclaimStaleIndexLockRealRemoveErrorFails(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return reapableReport(), nil
	})
	withRemoveIndexLockFn(t, func(string) error { return errors.New("permission denied") })

	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--reclaim-stale-index-lock", "--apply"})
	if code != 1 {
		t.Fatalf("a real remove error should exit 1, got %d out=%q", code, out.String())
	}
	if !strings.Contains(errb.String(), "reclaim failed") {
		t.Fatalf("stderr missing reclaim failure:\n%s", errb.String())
	}
}
