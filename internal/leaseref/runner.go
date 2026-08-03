package leaseref

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// runnerWaitDelay is the portable backstop that makes a CALLER's context
// deadline actually end the call (#5564). Cancelling a context kills the DIRECT
// child only, and both runners capture stdout into a strings.Builder — not an
// *os.File — so Go wires an OS pipe and cmd.Wait blocks until every writer
// closes it. A `git push` over https spawns a credential helper that inherits
// that pipe; if the helper survives the kill (a stalled auth prompt is exactly
// the case a deadline is meant to escape) Wait waits on it forever, and the
// caller's deadline buys nothing at all. WaitDelay bounds that post-kill wait,
// after which Wait closes the pipe and returns exec.ErrWaitDelay — not an
// *exec.ExitError, so it reaches the caller as "git could not be executed",
// which is the honest shape for "no exit status was ever observed". Same
// grandchild-holds-the-pipe hazard the dispatch tick's helper spawns document
// (cmd/fak/dispatch_tick.go) and cmd/dispatchworker's launch backstop.
//
// SHORTER than those callers' 10s on purpose. They are their own outermost
// bound, so a generous delay costs them nothing. This runner sits UNDER a
// convergence budget that blocks a loop tick (ambientLeaseRefSyncBudget), and a
// backstop that approaches its parent's budget stops being a backstop and
// becomes most of the wait. What it has to cover is only a pipe closing after
// the process is already gone — microseconds — since a grandchild that intends
// to hold the pipe indefinitely is not going to release it in ten seconds
// either. A healthy git never reaches the delay at all: it exits, its pipes
// close, Wait returns.
const runnerWaitDelay = 2 * time.Second

// gitRunner is the default Runner: it runs the real git binary. It mirrors
// witness.gitRunner's contract — a non-zero git exit is returned in code (not err);
// err signals git could not be EXECUTED at all. stderr is discarded (the package
// surfaces its own typed errors), matching the witness resolver rather than safecommit.
func gitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.WaitDelay = runnerWaitDelay
	if dir != "" {
		cmd.Dir = dir
	}
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = nil
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return out.String(), ee.ExitCode(), nil // git ran, returned non-zero
	}
	return "", -1, err // git could not be executed
}

// gitStdinRunner is the stdin-carrying sibling of gitRunner, used ONLY by the batched
// delete (`git update-ref --stdin`, reap.go), which reads its command list from stdin.
// Same contract and background-window configuration as gitRunner — a non-zero git exit
// rides code (a lock-contended transaction is exit != 0, not an err), and err signals
// only that git could not be EXECUTED. stderr is discarded, exactly like gitRunner.
func gitStdinRunner(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.WaitDelay = runnerWaitDelay
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Stdin = strings.NewReader(stdin)
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = nil
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return out.String(), ee.ExitCode(), nil // git ran, returned non-zero
	}
	return "", -1, err // git could not be executed
}
