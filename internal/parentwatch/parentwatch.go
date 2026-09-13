// Package parentwatch cancels a context when a watched parent process exits,
// so stdio servers exit instead of reparenting to launchd and leaking.
package parentwatch

import (
	"context"
	"sync"
	"syscall"
	"time"
)

// ParentAlive reports whether parentPID is a live process.
func ParentAlive(parentPID int) bool {
	if parentPID <= 0 {
		return false
	}
	err := syscall.Kill(parentPID, 0)
	return err == nil || err == syscall.EPERM
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
