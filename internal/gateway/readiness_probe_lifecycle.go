package gateway

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrWarmupTimeout identifies a bounded startup wait whose worker did not return.
var ErrWarmupTimeout = errors.New("backend startup warmup timed out")

// ErrWarmupInProgress rejects a concurrent attempt without changing the active
// attempt's readiness state or launching another backend turn.
var ErrWarmupInProgress = errors.New("backend startup warmup already running")

// warmupProbeLifecycle separates backend termination acknowledgment from a
// canceled wait. A timed-out worker may ignore cancellation; done stays installed
// until it actually returns, so a subsequent warmup cannot overlap it.
type warmupProbeLifecycle struct {
	runMu sync.Mutex
	mu    sync.Mutex
	done  <-chan struct{}
}

// begin reserves the entire warmup, including its initial one-token turn.
// Rejection must not overwrite the readiness state of an active attempt.
func (l *warmupProbeLifecycle) begin() error {
	if !l.runMu.TryLock() {
		return ErrWarmupInProgress
	}
	l.mu.Lock()
	pending := l.pendingLocked()
	l.mu.Unlock()
	if pending {
		l.runMu.Unlock()
		return fmt.Errorf("%w: previous decode confirm still running", ErrWarmupTimeout)
	}
	return nil
}

func (l *warmupProbeLifecycle) end() { l.runMu.Unlock() }

func (l *warmupProbeLifecycle) pendingLocked() bool {
	if l.done == nil {
		return false
	}
	select {
	case <-l.done:
		l.done = nil
		return false
	default:
		return true
	}
}

// run waits only for the probe's lifecycle, not its measurement result. The
// caller may ignore a completed measurement error, but must not publish readiness
// after a nonnil lifecycle error. The probe owns its result and panic handling.
func (l *warmupProbeLifecycle) run(ctx context.Context, ceiling time.Duration, probe func(context.Context)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	l.mu.Lock()
	if l.pendingLocked() {
		l.mu.Unlock()
		return fmt.Errorf("%w: previous decode confirm still running", ErrWarmupTimeout)
	}
	done := make(chan struct{})
	l.done = done
	l.mu.Unlock()
	probeCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		probe(probeCtx)
		close(done)
	}()
	var timeout <-chan time.Time
	if ceiling > 0 {
		timer := time.NewTimer(ceiling)
		defer timer.Stop()
		timeout = timer.C
	}
	select {
	case <-done:
		// Completion and cancellation can become ready together. Do not turn a
		// canceled startup into readiness by randomly selecting completion.
		return ctx.Err()
	case <-ctx.Done():
		return ctx.Err()
	case <-timeout:
		return fmt.Errorf("%w: decode confirm exceeded %s", ErrWarmupTimeout, ceiling)
	}
}
