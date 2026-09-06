package ghexec

import (
	"context"
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestDefaultTimeout(t *testing.T) {
	if DefaultTimeout != 60*time.Second {
		t.Fatalf("DefaultTimeout = %v, want 60s", DefaultTimeout)
	}
}

func TestCommandDisablesPromptingInEnv(t *testing.T) {
	cmd := Command(context.Background(), "issue", "list")
	for _, want := range []string{"GH_PROMPT_DISABLED=1", "GH_NO_UPDATE_NOTIFIER=1"} {
		found := false
		for _, e := range cmd.Env {
			if e == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Command env is missing %q", want)
		}
	}
	if len(cmd.Args) == 0 || cmd.Args[0] != "gh" {
		t.Errorf("Command args = %v, want gh argv[0]", cmd.Args)
	}
	if runtime.GOOS == "windows" {
		if cmd.SysProcAttr == nil || !cmd.SysProcAttr.HideWindow {
			t.Errorf("Command SysProcAttr missing HideWindow on windows: %+v", cmd.SysProcAttr)
		}
	}
}

func TestCommandPreservesCallerEnvironment(t *testing.T) {
	const key, val = "TEST_GHEXEC_INHERITED", "true"
	t.Setenv(key, val)
	cmd := Command(context.Background(), "issue", "list")
	want := key + "=" + val
	found := false
	for _, e := range cmd.Env {
		if e == want {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Command env missing inherited caller var %q", want)
	}
}

func TestCommandTimeoutDeadlineIsWired(t *testing.T) {
	cmd, cancel := CommandTimeout(context.Background(), -time.Second, "issue", "list")
	defer cancel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	cmd.Path = exe
	cmd.Err = nil
	if err := cmd.Run(); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Run with an expired deadline = %v, want context.DeadlineExceeded", err)
	}
}

func TestCommandTimeoutNilParentAndCancelWired(t *testing.T) {
	cmd, cancel := CommandTimeout(nil, time.Hour, "issue", "list")
	cancel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	cmd.Path = exe
	cmd.Err = nil
	if err := cmd.Run(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run after cancel = %v, want context.Canceled", err)
	}
}

func TestCommandTimeoutParentCancellation(t *testing.T) {
	parent, cancelParent := context.WithCancel(context.Background())
	cmd, cancel := CommandTimeout(parent, time.Hour, "issue", "list")
	defer cancel()
	cancelParent()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	cmd.Path = exe
	cmd.Err = nil
	if err := cmd.Run(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Run after parent cancel = %v, want context.Canceled", err)
	}
}

func TestCommandScrubsProtectedNamesFromGitHubBody(t *testing.T) {
	cpu, gpu := "da"+"33", "dgx"+"1"
	cmd := Command(context.Background(), "issue", "comment", "7", "--body", "compare "+cpu+" with "+gpu)
	joined := strings.Join(cmd.Args, "\n")
	if !strings.Contains(joined, "compare CPU server with GPU server") || strings.Contains(strings.ToLower(joined), cpu) || strings.Contains(strings.ToLower(joined), gpu) {
		t.Fatalf("Command args not scrubbed: %#v", cmd.Args)
	}
}
