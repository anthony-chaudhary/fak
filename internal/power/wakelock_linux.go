//go:build linux

package power

import (
	"context"
	"fmt"
	"os/exec"
	"sync"
)

type linuxLock struct {
	cmd    *exec.Cmd
	cancel context.CancelFunc
	done   chan struct{}
	mu     sync.Mutex
	closed bool
}

func platformAcquire(reason string, flags WakeFlags) (platformLock, error) {
	if path, err := exec.LookPath("systemd-inhibit"); err == nil && path != "" {
		what := "idle:sleep"
		if flags&PreventDisplaySleep != 0 {
			what = "idle:sleep:handle-lid-switch"
		}
		ctx, cancel := context.WithCancel(context.Background())
		args := []string{
			"--what=" + what,
			"--who=fak",
			fmt.Sprintf("--why=%s", reason),
			"--mode=block",
			"sleep", "2147483647",
		}
		cmd := exec.CommandContext(ctx, path, args...)
		if err := cmd.Start(); err == nil {
			l := &linuxLock{
				cmd:    cmd,
				cancel: cancel,
				done:   make(chan struct{}),
			}
			go func() {
				_ = cmd.Wait()
				close(l.done)
			}()
			return l, nil
		}
		cancel()
	}

	// Fallback to no-op lock when systemd-inhibit is absent (e.g. containers, minimal hosts)
	return &noOpLock{}, nil
}

func (l *linuxLock) Release() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if l.cancel != nil {
		l.cancel()
	}
	if l.cmd != nil && l.cmd.Process != nil {
		_ = l.cmd.Process.Kill()
	}
	if l.done != nil {
		<-l.done
	}
	return nil
}
