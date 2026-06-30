package commitlane

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func TestStatusClear(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	rep, err := Status(context.Background(), Options{
		Runner:      fakeRepoRunner(root, gitDir),
		ProbeLock:   func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Stat:        func(path string) FileFact { return FileFact{} },
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.OK || rep.Verdict != VerdictClear {
		t.Fatalf("clear report = %+v, want ok clear", rep)
	}
	if rep.CommitLock.Path != filepath.Join(gitDir, "fak-commit.lock") {
		t.Fatalf("commit lock path = %q", rep.CommitLock.Path)
	}
	if rep.IndexLock.Path != filepath.Join(gitDir, "index.lock") {
		t.Fatalf("index lock path = %q", rep.IndexLock.Path)
	}
}

func TestStatusReportsStaleCommitLock(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	rep, err := Status(context.Background(), Options{
		Runner: fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe {
			return safecommit.LockProbe{Path: path, Exists: true, HolderPID: 1234, Alive: false, Stale: true}
		},
		Stat:        func(path string) FileFact { return FileFact{} },
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.OK || rep.Verdict != VerdictStale {
		t.Fatalf("stale lock report = %+v, want not-ok stale", rep)
	}
	if !rep.CommitLock.Stale || rep.CommitLock.HolderPID != 1234 {
		t.Fatalf("stale lock fields = %+v", rep.CommitLock)
	}
	if !strings.Contains(rep.NextAction, "tree-doctor") {
		t.Fatalf("next action should point at stale-lock recovery, got %q", rep.NextAction)
	}
}

func TestStatusAttachesLiveOwnerAndQueueCandidates(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	procs := []Process{
		{PID: 42, ParentPID: 7, Name: "fak.exe", Command: root + `\fak.exe commit --path a.go -m msg --dir ` + root},
		{PID: 50, ParentPID: 8, Name: "fak.exe", Command: root + `\fak.exe commit --path b.go -m msg --dir ` + root},
		{PID: 60, ParentPID: 8, Name: "fak.exe", Command: root + `\fak.exe commit status --dir ` + root},
	}
	rep, err := Status(context.Background(), Options{
		Runner: fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe {
			return safecommit.LockProbe{Path: path, Exists: true, HolderPID: 42, Alive: true}
		},
		Stat:        func(path string) FileFact { return FileFact{} },
		ProcessList: func(context.Context) ([]Process, error) { return procs, nil },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.OK || rep.Verdict != VerdictBusy {
		t.Fatalf("live owner report = %+v, want ok busy", rep)
	}
	if rep.Owner == nil || rep.Owner.PID != 42 || rep.Owner.Role != "owner" {
		t.Fatalf("owner = %+v, want pid 42 owner", rep.Owner)
	}
	if len(rep.Queue) != 1 || rep.Queue[0].PID != 50 {
		t.Fatalf("queue = %+v, want only pid 50", rep.Queue)
	}
	if len(rep.LiveWriters) != 2 {
		t.Fatalf("live writers = %+v, want owner + queued candidate only", rep.LiveWriters)
	}
}

func TestStatusFlagsOldIndexLockWithoutWriter(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	rep, err := Status(context.Background(), Options{
		Runner:    fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Stat: func(path string) FileFact {
			return FileFact{Exists: true, ModTime: now.Add(-30 * time.Minute), Size: 12}
		},
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
		Now:         func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.OK || rep.Verdict != VerdictBlocked {
		t.Fatalf("old index lock report = %+v, want not-ok blocked", rep)
	}
	if !rep.IndexLock.StaleHint || rep.IndexLock.AgeSeconds != int64((30*time.Minute)/time.Second) {
		t.Fatalf("index lock fields = %+v", rep.IndexLock)
	}
}

func TestStatusKeepsIndexLockBusyWhenWriterIsLive(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	rep, err := Status(context.Background(), Options{
		Runner:    fakeRepoRunner(root, gitDir),
		ProbeLock: func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Stat: func(path string) FileFact {
			return FileFact{Exists: true, ModTime: now.Add(-30 * time.Minute), Size: 12}
		},
		ProcessList: func(context.Context) ([]Process, error) {
			return []Process{{PID: 77, Name: "git.exe", Command: `C:\Program Files\Git\cmd\git.exe commit -- ` + filepath.Join(root, "file.go")}}, nil
		},
		Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.OK || rep.Verdict != VerdictBusy {
		t.Fatalf("index lock with live writer = %+v, want ok busy", rep)
	}
	if len(rep.LiveWriters) != 1 || rep.LiveWriters[0].Match != "git_writer" {
		t.Fatalf("live writers = %+v, want git writer", rep.LiveWriters)
	}
}

func TestStatusSurfacesProcessProbeErrors(t *testing.T) {
	root, gitDir := testRepoPaths(t)
	rep, err := Status(context.Background(), Options{
		Runner:      fakeRepoRunner(root, gitDir),
		ProbeLock:   func(path string) safecommit.LockProbe { return safecommit.LockProbe{Path: path} },
		Stat:        func(path string) FileFact { return FileFact{} },
		ProcessList: func(context.Context) ([]Process, error) { return nil, errProcessProbe },
		Now:         fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ProcessProbe != "error" || len(rep.Errors) != 1 {
		t.Fatalf("process probe error not surfaced: %+v", rep)
	}
	if !rep.OK || rep.Verdict != VerdictClear {
		t.Fatalf("process probe errors should not fake a blockage: %+v", rep)
	}
}

func fakeRepoRunner(root, gitDir string) Runner {
	return func(_ context.Context, _ string, args ...string) RunResult {
		args = stripNoOptional(args)
		switch strings.Join(args, " ") {
		case "rev-parse --show-toplevel":
			return RunResult{Stdout: root, Code: 0}
		case "rev-parse --absolute-git-dir":
			return RunResult{Stdout: gitDir, Code: 0}
		default:
			return RunResult{Code: 1, Stderr: "unexpected git args: " + strings.Join(args, " ")}
		}
	}
}

func stripNoOptional(args []string) []string {
	if len(args) > 0 && args[0] == "--no-optional-locks" {
		return args[1:]
	}
	return args
}

func fixedNow() time.Time {
	return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
}

func testRepoPaths(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	return root, filepath.Join(root, ".git")
}

var errProcessProbe = processProbeErr{}

type processProbeErr struct{}

func (processProbeErr) Error() string { return "inventory unavailable" }
