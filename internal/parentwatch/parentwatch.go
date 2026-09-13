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
	if parentPID <= 1 {
		return parent, func() {}
	}
	if parent.Err() != nil {
		return parent, func() {}
	}

	ctx, cancel := context.WithCancel(parent)
	stop := watch(ctx, cancel, parentPID)
	return ctx, func() {
		cancel()
		stop()
	}
}

func pollWatch(ctx context.Context, cancel context.CancelFunc, parentPID int) func() {
	if !ParentAlive(parentPID) {
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
				if !ParentAlive(parentPID) {
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
