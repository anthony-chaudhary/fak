package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func TestCommitStaleCommitLockReclaimDryRunThenApply(t *testing.T) {
	gitDir := t.TempDir()
	lockPath := filepath.Join(gitDir, "fak-commit.lock")
	writeCommitLockFixture(t, lockPath, fmt.Sprintf("%d\n", deadCommitLockPID(t)))
	withCommitLockGitDir(t, gitDir)

	var out, errb bytes.Buffer
	if code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-commit-lock"}); code != 0 {
		t.Fatalf("dry-run exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "WOULD reclaim") || !strings.Contains(out.String(), "--apply") {
		t.Fatalf("dry-run output does not identify the gated action:\n%s", out.String())
	}
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("dry-run changed the lock: %v", err)
	}

	out.Reset()
	errB := &errb
	errB.Reset()
	if code := runCommitCommand(&out, errB, []string{"--reclaim-stale-commit-lock", "--apply"}); code != 0 {
		t.Fatalf("apply exit=%d stderr=%s", code, errB.String())
	}
	if !strings.Contains(out.String(), "reclaimed") {
		t.Fatalf("apply output missing reclaim receipt:\n%s", out.String())
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("apply left the dead-owner lock behind: %v", err)
	}
}

func TestCommitStaleCommitLockReclaimRefusesLiveAbsentAndGarbage(t *testing.T) {
	t.Run("live holder", func(t *testing.T) {
		gitDir := t.TempDir()
		lockPath := filepath.Join(gitDir, "fak-commit.lock")
		want := []byte(fmt.Sprintf("%d\n", os.Getpid()))
		if err := os.WriteFile(lockPath, want, 0o600); err != nil {
			t.Fatal(err)
		}
		withCommitLockGitDir(t, gitDir)

		var out, errb bytes.Buffer
		if code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-commit-lock", "--apply"}); code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb.String())
		}
		if !strings.Contains(out.String(), "holder live") {
			t.Fatalf("live refusal missing from output:\n%s", out.String())
		}
		assertFileBytes(t, lockPath, want)
	})

	t.Run("absent", func(t *testing.T) {
		gitDir := t.TempDir()
		withCommitLockGitDir(t, gitDir)
		var out, errb bytes.Buffer
		if code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-commit-lock", "--apply"}); code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb.String())
		}
		if !strings.Contains(out.String(), "absent") {
			t.Fatalf("absent result missing from output:\n%s", out.String())
		}
	})

	t.Run("garbage owner", func(t *testing.T) {
		gitDir := t.TempDir()
		lockPath := filepath.Join(gitDir, "fak-commit.lock")
		want := []byte("not-a-pid\n")
		if err := os.WriteFile(lockPath, want, 0o600); err != nil {
			t.Fatal(err)
		}
		withCommitLockGitDir(t, gitDir)
		var out, errb bytes.Buffer
		if code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-commit-lock", "--apply"}); code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb.String())
		}
		if !strings.Contains(out.String(), "owner unknown") {
			t.Fatalf("garbage-owner refusal missing from output:\n%s", out.String())
		}
		assertFileBytes(t, lockPath, want)
	})
}

func TestCommitStaleCommitLockReclaimTouchesOnlyExactLock(t *testing.T) {
	gitDir := t.TempDir()
	lockPath := filepath.Join(gitDir, "fak-commit.lock")
	writeCommitLockFixture(t, lockPath, fmt.Sprintf("%d\n", deadCommitLockPID(t)))
	sentinels := map[string][]byte{
		filepath.Join(gitDir, "index.lock"):                           []byte("index-sentinel\x00\xff"),
		filepath.Join(gitDir, "refs", "heads", "peer"):                []byte("ref-sentinel\n"),
		filepath.Join(gitDir, "worktrees", "peer", "locked"):          []byte("worktree-sentinel\n"),
		filepath.Join(gitDir, "worktrees", "peer", "index.lock"):      []byte("peer-index-sentinel\n"),
		filepath.Join(gitDir, "worktrees", "peer", "fak-commit.lock"): []byte("peer-commit-sentinel\n"),
	}
	for path, data := range sentinels {
		writeCommitLockFixture(t, path, string(data))
	}
	withCommitLockGitDir(t, gitDir)

	var out, errb bytes.Buffer
	if code := runCommitCommand(&out, &errb, []string{"--reclaim-stale-commit-lock", "--apply"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("target lock still exists: %v", err)
	}
	for path, want := range sentinels {
		assertFileBytes(t, path, want)
	}
}

func TestCommitStaleCommitLockHelpAndRecoveryRoute(t *testing.T) {
	var out, errb bytes.Buffer
	runCommitCommand(&out, &errb, []string{"--help"})
	if help := out.String() + errb.String(); !strings.Contains(help, "reclaim-stale-commit-lock") {
		t.Fatalf("commit help omits the commit-lock actuator:\n%s", help)
	}

	out.Reset()
	errB := &errb
	errB.Reset()
	if code := runRecover(&out, errB, []string{"LOCK_BUSY", "--dry-run"}); code != 0 {
		t.Fatalf("recover exit=%d stderr=%s", code, errB.String())
	}
	got := out.String()
	if !strings.Contains(got, "fak commit --reclaim-stale-commit-lock") {
		t.Fatalf("LOCK_BUSY recovery does not route to the commit-lock actuator:\n%s", got)
	}
	if strings.Contains(got, "--reclaim-stale-index-lock") {
		t.Fatalf("LOCK_BUSY recovery still routes to the unrelated index lock:\n%s", got)
	}

	var busy bytes.Buffer
	renderCommitResult(&busy, safecommit.Result{Reason: safecommit.ReasonLockBusy})
	if !strings.Contains(busy.String(), "--reclaim-stale-commit-lock") {
		t.Fatalf("LOCK_BUSY result omits the commit-lock actuator:\n%s", busy.String())
	}
}

func withCommitLockGitDir(t *testing.T, gitDir string) {
	t.Helper()
	withCommitStatusFn(t, func(_ context.Context, _ commitlane.Options) (commitlane.Report, error) {
		return commitlane.Report{GitDir: gitDir}, nil
	})
}

func deadCommitLockPID(t *testing.T) int {
	t.Helper()
	for pid := int(^uint32(0) >> 1); pid > int(^uint32(0)>>1)-4096; pid-- {
		if !safecommit.ProcessAlive(pid) {
			return pid
		}
	}
	t.Fatal("could not find a definitely absent PID for stale-lock witness")
	return 0
}

func writeCommitLockFixture(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func assertFileBytes(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read sentinel %s: %v", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("sentinel %s changed: got %q want %q", path, got, want)
	}
}
