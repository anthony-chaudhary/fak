package wipinventory

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// holdPipeShim makes a status call exit with git's own status while a background
// helper keeps the inherited stdout pipe open well past the runner's WaitDelay:
// the exec.ErrWaitDelay-with-a-successful-exit path. FAK_DR_HOLD=once holds only
// the first status call; FAK_DR_HOLD=always holds every one.
const holdPipeShim = "#!/bin/sh\n" +
	"echo \"$*\" >> \"$FAK_DR_COUNTER\"\n" +
	"hold=\n" +
	"if [ \"$1\" = status ]; then\n" +
	"  case \"$FAK_DR_HOLD\" in\n" +
	"    always) hold=1 ;;\n" +
	"    once) if [ ! -e \"$FAK_DR_MARKER\" ]; then : > \"$FAK_DR_MARKER\"; hold=1; fi ;;\n" +
	"  esac\n" +
	"fi\n" +
	"if [ -n \"$hold\" ]; then\n" +
	"  \"$FAK_DR_REAL_GIT\" \"$@\"\n" +
	"  rc=$?\n" +
	"  sleep 2 &\n" +
	"  exit $rc\n" +
	"fi\n" +
	"exec \"$FAK_DR_REAL_GIT\" \"$@\"\n"

// installGitShim puts a POSIX `git` script first on PATH for the rest of the test
// and returns the file each shim invocation appends one line to.
func installGitShim(t *testing.T, body string) string {
	t.Helper()
	realGit, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	shimDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(shimDir, "git"), []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	counter := filepath.Join(shimDir, "invocations")
	t.Setenv("FAK_DR_REAL_GIT", realGit)
	t.Setenv("FAK_DR_COUNTER", counter)
	t.Setenv("FAK_DR_MARKER", filepath.Join(shimDir, "held-once"))
	t.Setenv("PATH", shimDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return counter
}

func shimInvocations(t *testing.T, counter string) int {
	t.Helper()
	b, err := os.ReadFile(counter)
	if os.IsNotExist(err) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Count(string(b), "\n")
}

func TestDeadlineRunnerRetriesSuccessfulExitWhosePipeOutlivesIt(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake git shim is a POSIX script; the acceptance gate runs under WSL")
	}
	repo, _ := initTestRepo(t)
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("unlanded edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	statusArgs := []string{"status", "--porcelain=v1", "-z", "--untracked-files=all"}

	t.Run("one overrun is rerun", func(t *testing.T) {
		counter := installGitShim(t, holdPipeShim)
		t.Setenv("FAK_DR_HOLD", "once")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := DeadlineRunner{Ctx: ctx}.Run(repo, statusArgs...)
		if err != nil {
			t.Fatalf("git exited 0 but its outliving pipe was reported as a failure: %v", err)
		}
		if !strings.Contains(string(out), " M base.txt") {
			t.Fatalf("status output=%q, want the dirty porcelain", out)
		}
		if n := shimInvocations(t, counter); n != 2 {
			t.Fatalf("git invocations=%d, want 2 (the overrun rerun once)", n)
		}
	})

	t.Run("a repeated overrun is an error without stdout", func(t *testing.T) {
		counter := installGitShim(t, holdPipeShim)
		t.Setenv("FAK_DR_HOLD", "always")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := DeadlineRunner{Ctx: ctx}.Run(repo, statusArgs...)
		if err == nil {
			t.Fatalf("a repeated overrun returned no error; out=%q", out)
		}
		if !errors.Is(err, exec.ErrWaitDelay) {
			t.Fatalf("err=%v, want it to wrap exec.ErrWaitDelay", err)
		}
		if strings.Contains(err.Error(), "base.txt") {
			t.Fatalf("error text carries stdout data: %v", err)
		}
		if n := shimInvocations(t, counter); n != 2 {
			t.Fatalf("git invocations=%d, want 2 (one rerun, then the error)", n)
		}
	})
}

func TestDeadlineRunnerFailureNamesCauseNotStdout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake git shim is a POSIX script; the acceptance gate runs under WSL")
	}
	counter := installGitShim(t, "#!/bin/sh\n"+
		"echo \"$*\" >> \"$FAK_DR_COUNTER\"\n"+
		"if [ \"$1\" = status ]; then\n"+
		"  echo ' M leaked.go'\n"+
		"  echo boom >&2\n"+
		"  exit 1\n"+
		"fi\n"+
		"exec \"$FAK_DR_REAL_GIT\" \"$@\"\n")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := DeadlineRunner{Ctx: ctx}.Run(t.TempDir(), "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err == nil {
		t.Fatalf("a failing git returned no error; out=%q", out)
	}
	msg := err.Error()
	if !strings.Contains(msg, "exit status 1") || !strings.Contains(msg, "boom") {
		t.Fatalf("error=%q, want the exit cause and stderr", msg)
	}
	if strings.Contains(msg, "leaked.go") {
		t.Fatalf("error text carries stdout data: %q", msg)
	}
	if n := shimInvocations(t, counter); n != 1 {
		t.Fatalf("git invocations=%d, want 1 (a real failure is not rerun)", n)
	}
}

func TestDeadlineRunnerRefusesAfterDeadline(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  func() (context.Context, context.CancelFunc)
		want error
	}{
		{"expired", func() (context.Context, context.CancelFunc) {
			return context.WithTimeout(context.Background(), -1)
		}, context.DeadlineExceeded},
		{"cancelled", func() (context.Context, context.CancelFunc) {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx, cancel
		}, context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := tc.ctx()
			defer cancel()
			_, err := DeadlineRunner{Ctx: ctx}.Run(t.TempDir(), "status")
			if err == nil {
				t.Fatal("a call after the deadline returned no error")
			}
			if !errors.Is(err, tc.want) || !strings.HasPrefix(err.Error(), "git status") {
				t.Fatalf("err=%q, want a %q refusal naming the git command", err, tc.want)
			}
		})
	}

	t.Run("a live deadline reports git's own exit", func(t *testing.T) {
		if _, err := exec.LookPath("git"); err != nil {
			t.Skip("git is not on PATH")
		}
		dir := t.TempDir()
		t.Setenv("GIT_CEILING_DIRECTORIES", filepath.Dir(dir))
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err := DeadlineRunner{Ctx: ctx}.Run(dir, "status")
		if err == nil || !strings.Contains(err.Error(), "exit status 128") {
			t.Fatalf("err=%v, want git's not-a-repository exit status 128", err)
		}
	})
}
