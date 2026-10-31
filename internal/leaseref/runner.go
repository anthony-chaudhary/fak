package leaseref

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/sessionref"
)

const runnerWaitDelay = sessionref.RunnerWaitDelay

func gitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	return sessionref.GitRunner(ctx, dir, args...)
}

func gitStdinRunner(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
	return sessionref.GitStdinRunner(ctx, dir, stdin, args...)
}
