package ghexec

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

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
}

// The deadline probes below repoint the command at the (always-present)
// running test binary so they need neither `gh` installed nor a portable
// sleep. The context wiring under test is unchanged, and because every probe
// uses an already-done context, exec.Cmd.Start returns the context error
// BEFORE spawning anything — the test binary is never actually re-executed.

func TestCommandTimeoutDeadlineIsWired(t *testing.T) {
	cmd, cancel := CommandTimeout(context.Background(), -time.Second, "issue", "list")
	defer cancel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	// The already-expired deadline must make Run refuse before spawning,
	// which proves the derived context is wired to the returned command.
	cmd.Path = exe
	cmd.Err = nil // clear any gh-not-installed lookup error; the deadline must win
	if err := cmd.Run(); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Run with an expired deadline = %v, want context.DeadlineExceeded", err)
	}
}

func TestCommandTimeoutNilParentAndCancelWired(t *testing.T) {
	cmd, cancel := CommandTimeout(nil, time.Hour, "issue", "list")
	cancel() // cancel must reach the same context the command watches
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

func TestCommandScrubsProtectedNamesFromGitHubBody(t *testing.T) {
	cpu, gpu := "da"+"33", "dgx"+"1"
	cmd := Command(context.Background(), "issue", "comment", "7", "--body", "compare "+cpu+" with "+gpu)
	joined := strings.Join(cmd.Args, "\n")
	if !strings.Contains(joined, "compare CPU server with GPU server") || strings.Contains(strings.ToLower(joined), cpu) || strings.Contains(strings.ToLower(joined), gpu) {
		t.Fatalf("Command args not scrubbed: %#v", cmd.Args)
	}
}
