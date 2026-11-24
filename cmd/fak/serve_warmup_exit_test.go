package main

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/servewarmup"
)

func serveWarmupPanicRun(context.Context) (time.Duration, error) {
	return time.Second, fmt.Errorf("%w: model: resident Q2_K embedding does not support GPU layer offload", gateway.ErrWarmupPanic)
}

func serveWarmupPanicFailure() (string, string, bool) {
	return gateway.WarmupFailurePanic, gateway.ErrWarmupPanic.Error(), true
}

// A warmup panic (the aegis2 split-embedding guard) must end the serve with a typed reason and a
// nonzero exit code, not leave it answering 503 warmup_failed forever.
func TestServeWarmupPanicExitsNonzeroWithTypedReason(t *testing.T) {
	ctx, failServe := context.WithCancelCause(context.Background())
	defer failServe(nil)
	var stderr bytes.Buffer
	if !runServeBackendWarmup(ctx, serveWarmupPanicRun, serveWarmupPanicFailure, func(string) string { return "" }, &stderr, failServe) {
		t.Fatal("warmup panic was not treated as fatal")
	}
	if ctx.Err() == nil {
		t.Fatal("serve context not cancelled; ListenAndServe would never drain")
	}
	if code := serveExitCode(ctx, &stderr); code != 1 {
		t.Fatalf("exit code = %d, want 1", code)
	}
	got := stderr.String()
	for _, want := range []string{"readiness held as warmup_failed", "WARMUP_FAILED: warmup_panic: ", servewarmup.FailureModeEnv + "=hold"} {
		if !strings.Contains(got, want) {
			t.Fatalf("stderr missing %q:\n%s", want, got)
		}
	}
}

// The hold opt-out and a shutdown already in progress keep today's behaviour: no cancel, exit 0.
func TestServeWarmupFailureNotFatalWhenHeldOrShuttingDown(t *testing.T) {
	hold := func(k string) string {
		if k == servewarmup.FailureModeEnv {
			return "hold"
		}
		return ""
	}
	ctx, failServe := context.WithCancelCause(context.Background())
	defer failServe(nil)
	var stderr bytes.Buffer
	if runServeBackendWarmup(ctx, serveWarmupPanicRun, serveWarmupPanicFailure, hold, &stderr, failServe) || ctx.Err() != nil {
		t.Fatal("hold must keep the serve up")
	}
	if code := serveExitCode(ctx, &stderr); code != 0 {
		t.Fatalf("held exit code = %d, want 0", code)
	}

	parent, cancel := context.WithCancel(context.Background())
	sctx, sfail := context.WithCancelCause(parent)
	defer sfail(nil)
	cancel()
	if runServeBackendWarmup(sctx, func(context.Context) (time.Duration, error) { return 0, context.Canceled }, nil, nil, &stderr, sfail) {
		t.Fatal("a shutdown in progress is not a warmup failure")
	}
	if code := serveExitCode(sctx, &stderr); code != 0 {
		t.Fatalf("signal shutdown exit code = %d, want 0", code)
	}
}
