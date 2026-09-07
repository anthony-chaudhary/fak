package rsiloop

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestGitLockErrorClassification(t *testing.T) {
	cases := []struct {
		msg  string
		want bool
	}{
		{"fatal: Unable to create '.git/index.lock': File exists.", true},
		{"error: cannot lock ref 'refs/heads/main': is at abc but expected def", true},
		{"fatal: Another git process seems to be running in this repository", true},
		{"fatal: packed-refs.lock: File exists", true},
		{"error: lock_busy: another process holds lock", true},
		{"error: lock exists", true},
		{"could not reset index", true},
		{"unable to write new index file", true},
		{"some regular compilation error", false},
		{"404 not found", false},
	}

	for _, c := range cases {
		err := errors.New(c.msg)
		got := IsGitLockError(err)
		if got != c.want {
			t.Errorf("IsGitLockError(%q) = %v, want %v", c.msg, got, c.want)
		}
	}
}

func TestGitLockErrorSatisfiesTransientInterfaces(t *testing.T) {
	gle := &GitLockError{Err: errors.New("git lock busy"), Msg: "index.lock exists"}
	if !gle.IsTransient() {
		t.Error("gle.IsTransient() should be true")
	}
	if !gle.Transient() {
		t.Error("gle.Transient() should be true")
	}
	if !gle.TransientMeasure() {
		t.Error("gle.TransientMeasure() should be true")
	}

	wrapped := NewTransientMeasureError(gle)
	if !IsTransientMeasureError(wrapped) {
		t.Error("IsTransientMeasureError(wrapped) should be true")
	}
	if !IsGitLockError(wrapped) {
		t.Error("IsGitLockError(wrapped) should be true")
	}
}

// TestGitLockHelperProcess simulates git worktree add failing with lock contention.
func TestGitLockHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_GIT_LOCK_HELPER_PROCESS") != "1" {
		return
	}
	fmt.Fprintln(os.Stderr, "fatal: Unable to create '.git/index.lock': File exists.")
	os.Exit(1)
}

func TestWithWorktree_GitWorktreeAddLockContention(t *testing.T) {
	// Create a real minimal git repo in a temp dir
	tmpDir := t.TempDir()
	initCmd := windowgate.Command("git", "init", "-b", "main", tmpDir)
	windowgate.ConfigureBackgroundCommand(initCmd)
	if out, err := initCmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}

	configName := windowgate.Command("git", "-C", tmpDir, "config", "user.name", "Test")
	windowgate.ConfigureBackgroundCommand(configName)
	_ = configName.Run()
	configEmail := windowgate.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com")
	windowgate.ConfigureBackgroundCommand(configEmail)
	_ = configEmail.Run()

	readme := filepath.Join(tmpDir, "README.md")
	if err := os.WriteFile(readme, []byte("test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	addCmd := windowgate.Command("git", "-C", tmpDir, "add", "README.md")
	windowgate.ConfigureBackgroundCommand(addCmd)
	if out, err := addCmd.CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	commitCmd := windowgate.Command("git", "-C", tmpDir, "commit", "-m", "initial")
	windowgate.ConfigureBackgroundCommand(commitCmd)
	if out, err := commitCmd.CombinedOutput(); err != nil {
		t.Fatalf("git commit: %v: %s", err, out)
	}

	cfg := WorktreeConfig{
		Repo:        tmpDir,
		BaselineRef: "main",
		Command: func(name string, args ...string) *exec.Cmd {
			cmd := exec.Command(os.Args[0], "-test.run=TestGitLockHelperProcess")
			cmd.Env = append(os.Environ(), "GO_WANT_GIT_LOCK_HELPER_PROCESS=1")
			return cmd
		},
	}

	err := withWorktree(cfg, "main", func(wtPaths) error {
		return nil
	})
	if err == nil {
		t.Fatal("withWorktree should fail when git worktree add fails with lock")
	}

	if !IsTransient(err) {
		t.Fatalf("expected IsTransient(err) == true, got: %v", err)
	}
	if !IsTransientMeasureError(err) {
		t.Fatalf("expected transient measurement error, got: %v", err)
	}
	if !IsGitLockError(err) {
		t.Fatalf("expected git lock error, got: %v", err)
	}
}

func TestWithWorktree_FnGitLockContention(t *testing.T) {
	tmpDir := t.TempDir()
	initCmd := windowgate.Command("git", "init", "-b", "main", tmpDir)
	windowgate.ConfigureBackgroundCommand(initCmd)
	if out, err := initCmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}

	configName := windowgate.Command("git", "-C", tmpDir, "config", "user.name", "Test")
	windowgate.ConfigureBackgroundCommand(configName)
	_ = configName.Run()
	configEmail := windowgate.Command("git", "-C", tmpDir, "config", "user.email", "test@example.com")
	windowgate.ConfigureBackgroundCommand(configEmail)
	_ = configEmail.Run()

	readme := filepath.Join(tmpDir, "README.md")
	if err := os.WriteFile(readme, []byte("test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	addCmd := windowgate.Command("git", "-C", tmpDir, "add", "README.md")
	windowgate.ConfigureBackgroundCommand(addCmd)
	if out, err := addCmd.CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	commitCmd := windowgate.Command("git", "-C", tmpDir, "commit", "-m", "initial")
	windowgate.ConfigureBackgroundCommand(commitCmd)
	if out, err := commitCmd.CombinedOutput(); err != nil {
		t.Fatalf("git commit: %v: %s", err, out)
	}

	cfg := WorktreeConfig{
		Repo:        tmpDir,
		BaselineRef: "main",
	}

	err := withWorktree(cfg, "main", func(wtPaths) error {
		return errors.New("fatal: Unable to create '.git/index.lock': File exists.")
	})
	if err == nil {
		t.Fatal("expected error from fn")
	}
	if !IsTransient(err) {
		t.Fatalf("expected IsTransient(err) == true, got: %v", err)
	}
	if !IsTransientMeasureError(err) {
		t.Fatalf("expected transient measurement error, got: %v", err)
	}
	if !IsGitLockError(err) {
		t.Fatalf("expected git lock error, got: %v", err)
	}
}

func TestGitLockContention_IsTransient(t *testing.T) {
	lockErrors := []string{
		"fatal: Unable to create '.git/index.lock': File exists.",
		"error: cannot lock ref 'refs/heads/main': is at abc but expected def",
		"fatal: Another git process seems to be running in this repository",
		"error: lock_busy: another process holds lock",
		"fatal: packed-refs.lock: File exists",
		"could not reset index",
		"unable to write new index file",
	}

	for _, msg := range lockErrors {
		err := errors.New(msg)
		if !IsTransient(err) {
			t.Errorf("IsTransient(errors.New(%q)) = false, want true", msg)
		}
		if !IsTransientMeasureError(err) {
			t.Errorf("IsTransientMeasureError(errors.New(%q)) = false, want true", msg)
		}
		gle := &GitLockError{Err: err, Msg: msg}
		if !IsTransient(gle) {
			t.Errorf("IsTransient(&GitLockError{%q}) = false, want true", msg)
		}
		if !IsTransientMeasureError(gle) {
			t.Errorf("IsTransientMeasureError(&GitLockError{%q}) = false, want true", msg)
		}
		wrapped := NewTransientMeasureError(gle)
		if !IsTransient(wrapped) {
			t.Errorf("IsTransient(NewTransientMeasureError(&GitLockError{%q})) = false, want true", msg)
		}
		if !IsTransientMeasureError(wrapped) {
			t.Errorf("IsTransientMeasureError(NewTransientMeasureError(&GitLockError{%q})) = false, want true", msg)
		}
	}

	nonLock := errors.New("compilation failed: undefined symbol")
	if IsTransient(nonLock) {
		t.Errorf("IsTransient(nonLock) = true, want false")
	}
	if IsTransientMeasureError(nonLock) {
		t.Errorf("IsTransientMeasureError(nonLock) = true, want false")
	}
	if IsTransient(nil) {
		t.Errorf("IsTransient(nil) = true, want false")
	}
	if IsTransientMeasureError(nil) {
		t.Errorf("IsTransientMeasureError(nil) = true, want false")
	}
}
