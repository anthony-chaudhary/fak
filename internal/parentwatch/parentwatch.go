// Package parentwatch cancels a context when a watched parent process exits,
// so stdio servers exit instead of reparenting to launchd and leaking.
package parentwatch

import (
	"context"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// ParentAlive reports whether parentPID is a live process. It delegates to the
// shared, no-spawn, cross-platform probe in internal/processalive (POSIX signal
// 0 on unix; a process-handle query on Windows), preserving the exact
// semantics: positive live pid -> true; pid <= 0 -> false; reaped pid -> false.
func ParentAlive(parentPID int) bool {
	return processalive.Check(parentPID)
}

// Watch returns a context canceled when parentPID exits. parentPID <= 1 means
// "no parent to watch": return parent unchanged + no-op stop. If the parent is
// already dead, the returned ctx is already canceled. stop is idempotent and
// safe to call after cancel.
func Watch(parent context.Context, parentPID int) (context.Context, context.CancelFunc) {
	startTime, ok := processalive.StartTime(parentPID)
	return WatchIdent(parent, parentPID, startTime, ok)
}

// WatchIdent is Watch with the watched parent's creation time supplied by the
// caller. The poll loop treats the parent as dead when the liveness probe fails
// OR when startTime is known (ok) and the live creation time no longer matches:
// a reused PID then reads as dead instead of keeping the child alive forever.
// startTime/ok come from processalive.StartTime at Watch time; ok is false on
// platforms where the creation time is unavailable (non-Windows), which falls
// back to the PID-only probe.
func WatchIdent(parent context.Context, parentPID int, startTime time.Time, ok bool) (context.Context, context.CancelFunc) {
	if parentPID <= 1 {
		return parent, func() {}
	}
	if parent.Err() != nil {
		return parent, func() {}
	}

	ctx, cancel := context.WithCancel(parent)
	stop := watch(ctx, cancel, identity{pid: parentPID, start: startTime, startOK: ok})
	return ctx, func() {
		cancel()
		stop()
	}
}

// identity is the watched parent's PID paired with its creation time. startOK
// reports whether start is a real probe result rather than the zero-value
// fallback returned where the platform exposes no creation time.
type identity struct {
	pid     int
	start   time.Time
	startOK bool
}

// deadNow reports whether the watched parent is no longer the process that was
// watched: the probe says it is gone, or the PID now names a different process
// (same number, different creation time).
func (id identity) deadNow() bool {
	if !ParentAlive(id.pid) {
		return true
	}
	if !id.startOK {
		return false
	}
	start, ok := processalive.StartTime(id.pid)
	return !ok || !start.Equal(id.start)
}

func pollWatch(ctx context.Context, cancel context.CancelFunc, id identity) func() {
	if id.deadNow() {
		cancel()
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case <-ticker.C:
				if id.deadNow() {
					cancel()
					return
				}
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() { close(done) })
	}
}
