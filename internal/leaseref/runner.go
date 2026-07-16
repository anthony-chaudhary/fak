package leaseref

import (
	"context"
	"errors"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// gitRunner is the default Runner: it runs the real git binary. It mirrors
// witness.gitRunner's contract — a non-zero git exit is returned in code (not err);
// err signals git could not be EXECUTED at all. stderr is discarded (the package
// surfaces its own typed errors), matching the witness resolver rather than safecommit.
func gitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
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
