//go:build darwin

package power

import (
	"context"
	"fmt"
	"os/exec"
	"sync"
)

var (
	// darwinForceCaffeinate is an internal hook for tests to exercise the caffeinate fallback.
	darwinForceCaffeinate = false
	darwinMu              sync.Mutex
)

// SetDarwinForceCaffeinateForTesting forces Darwin to use the caffeinate command fallback.
func SetDarwinForceCaffeinateForTesting(force bool) {
	darwinMu.Lock()
	defer darwinMu.Unlock()
	darwinForceCaffeinate = force
}

func platformAcquire(reason string, flags WakeFlags) (platformLock, error) {
	darwinMu.Lock()
	force := darwinForceCaffeinate
	darwinMu.Unlock()

	if !force {
		lock, err := acquireDarwinIOKit(reason, flags)
		if err == nil && lock != nil {
			return lock, nil
		}
	}
	return acquireDarwinCaffeinate(reason, flags)
}

type caffeinateLock struct {
	cmd    *exec.Cmd
	cancel context.CancelFunc
	done   chan struct{}
	mu     sync.Mutex
	closed bool
}

func acquireDarwinCaffeinate(reason string, flags WakeFlags) (platformLock, error) {
	ctx, cancel := context.WithCancel(context.Background())
	args := []string{"-i", "-s"}
	if flags&PreventDisplaySleep != 0 {
		args = append([]string{"-d"}, args...)
	}

	cmd := exec.CommandContext(ctx, "caffeinate", args...)
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("caffeinate start: %w", err)
	}

	cl := &caffeinateLock{
		cmd:    cmd,
		cancel: cancel,
		done:   make(chan struct{}),
	}
	go func() {
		_ = cmd.Wait()
		close(cl.done)
	}()
	return cl, nil
}

func (c *caffeinateLock) Release() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	if c.cancel != nil {
		c.cancel()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
	}
	if c.done != nil {
		<-c.done
	}
	return nil
}
