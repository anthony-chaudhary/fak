package wipinventory

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// deadlineRunnerWaitDelay bounds how long an exited or killed git may keep its
// output pipes open through a handle some other process inherited.
const deadlineRunnerWaitDelay = 100 * time.Millisecond

// DeadlineRunner is GitRunner bound to one caller-owned deadline, for bounded
// callers such as the single-worktree reap's lifecycle bracket. A call is killed
// at the deadline, and one that starts after it fails without spawning git.
//
// It classifies a call by git's exit status, not by whether os/exec finished
// draining the pipes. exec.ErrWaitDelay is returned only when git exited 0 but
// its pipes did not reach EOF within the wait delay: the reader goroutine was
// starved on a loaded host, or a helper process inherited the pipe. Treating
// that as a git failure once recorded a dirty worktree's own porcelain as its
// capture error. The output of such a call may still be short (a starved reader
// can lose the unread tail when the pipe is closed), so the call is run once
// more; only a repeated overrun is reported, and a failure's text names the exit
// cause and stderr, never stdout data. Every command the inventory issues is a
// read, so the rerun is safe.
type DeadlineRunner struct {
	Ctx context.Context
}

func (r DeadlineRunner) Run(dir string, args ...string) ([]byte, error) {
	return r.RunWithStdin(dir, nil, args...)
}

func (r DeadlineRunner) RunWithStdin(dir string, in []byte, args ...string) ([]byte, error) {
	for attempt := 1; ; attempt++ {
		out, exitedOK, err := r.runOnce(dir, in, args)
		if err == nil {
			return out, nil
		}
		if attempt == 1 && exitedOK && errors.Is(err, exec.ErrWaitDelay) {
			continue
		}
		return nil, fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
}

func (r DeadlineRunner) runOnce(dir string, in []byte, args []string) (out []byte, exitedOK bool, err error) {
	ctx := r.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	cmd.WaitDelay = deadlineRunnerWaitDelay
	if in != nil {
		cmd.Stdin = bytes.NewReader(in)
	}
	configureDispatchHelperCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	runErr := cmd.Run()
	if runErr == nil {
		return stdout.Bytes(), true, nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, false, ctxErr
	}
	exitedOK = cmd.ProcessState != nil && cmd.ProcessState.Success()
	if detail := strings.TrimSpace(stderr.String()); detail != "" {
		return nil, exitedOK, fmt.Errorf("%w: %s", runErr, detail)
	}
	return nil, exitedOK, runErr
}
