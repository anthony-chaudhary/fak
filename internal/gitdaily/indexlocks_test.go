package gitdaily

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
)

func TestSweepIndexLocksReusesCommitLaneDecisions(t *testing.T) {
	gitDir := filepath.Join(t.TempDir(), ".git")
	index := filepath.Join(gitDir, "index.lock")
	next := filepath.Join(gitDir, "next-index-42.lock")
	rep := commitlane.Report{
		GitDir:       gitDir,
		ProcessProbe: "ok",
		IndexLock: commitlane.IndexLock{
			Path: index, Present: true, StaleHint: true,
		},
		NextIndexLocks: []commitlane.NextIndexLock{
			{Path: next, PID: 42, StaleHint: true},
			{Path: filepath.Join(gitDir, "next-index-99.lock"), PID: 99, StaleHint: true, OwnerAlive: true},
		},
	}
	status := func(_ context.Context, opts commitlane.Options) (commitlane.Report, error) {
		if !opts.LocksOnly {
			t.Fatal("daily index sweep did not request commitlane's locks-only view")
		}
		return rep, nil
	}

	var removed []string
	remove := func(path string) error {
		removed = append(removed, path)
		return nil
	}
	now := time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC)

	preview := sweepIndexLocksWith(context.Background(), t.TempDir(), now, false, status, remove)
	if len(removed) != 0 {
		t.Fatalf("dry run removed %v", removed)
	}
	if want := []string{"index.lock", "next-index-42.lock"}; !reflect.DeepEqual(preview.Reaped, want) {
		t.Fatalf("dry-run candidates = %v, want %v", preview.Reaped, want)
	}

	applied := sweepIndexLocksWith(context.Background(), t.TempDir(), now, true, status, remove)
	if !reflect.DeepEqual(removed, []string{index, next}) {
		t.Fatalf("removed = %v, want only the two commitlane-approved locks", removed)
	}
	if !reflect.DeepEqual(applied.Reaped, preview.Reaped) {
		t.Fatalf("apply result %v drifted from preview %v", applied.Reaped, preview.Reaped)
	}
}

func TestSweepIndexLocksSurfacesRemovalFailure(t *testing.T) {
	gitDir := filepath.Join(t.TempDir(), ".git")
	index := filepath.Join(gitDir, "index.lock")
	status := func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return commitlane.Report{
			GitDir: gitDir, ProcessProbe: "ok",
			IndexLock: commitlane.IndexLock{Path: index, Present: true, StaleHint: true},
		}, nil
	}
	res := sweepIndexLocksWith(context.Background(), t.TempDir(), time.Now(), true, status,
		func(string) error { return errors.New("access denied") })
	if res.Err == "" || len(res.Reaped) != 0 {
		t.Fatalf("failed sweep = %+v, want an error and no claimed reap", res)
	}
}
