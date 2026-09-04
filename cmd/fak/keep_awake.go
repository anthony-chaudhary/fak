package main

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/power"
	"github.com/anthony-chaudhary/fak/internal/session"
)

const (
	// KeepAwakeOff disables power assertions.
	KeepAwakeOff = "off"
	// KeepAwakeWhileActive holds power assertions while tasks, sessions, or requests are actively running.
	KeepAwakeWhileActive = "while-active"
	// KeepAwakeAlways holds power assertions for the entire lifetime of the process.
	KeepAwakeAlways = "always"
)

// validateKeepAwake normalizes and validates the --keep-awake flag value.
func validateKeepAwake(val string) (string, error) {
	norm := strings.ToLower(strings.TrimSpace(val))
	switch norm {
	case "", "off", "false", "0":
		return KeepAwakeOff, nil
	case "while-active", "active", "true", "1":
		return KeepAwakeWhileActive, nil
	case "always":
		return KeepAwakeAlways, nil
	default:
		return "", fmt.Errorf("invalid --keep-awake %q: want off, while-active, or always", val)
	}
}

// acquireProcessKeepAwake acquires a process-lifetime wake lock if mode is "always".
func acquireProcessKeepAwake(mode, reason string) (power.Releaser, error) {
	if mode == KeepAwakeAlways {
		return power.Acquire(reason, power.PreventSystemSleep)
	}
	return nil, nil
}

// acquireAgentRunKeepAwake acquires an active-run wake lock if mode is "while-active".
func acquireAgentRunKeepAwake(mode string) (power.Releaser, error) {
	if mode == KeepAwakeWhileActive {
		return power.Acquire("fak agent run", power.PreventSystemSleep)
	}
	return nil, nil
}

func hasActiveSessions(sessions *session.Table) bool {
	if sessions == nil {
		return false
	}
	for _, s := range sessions.Snapshot() {
		if s.Run == session.Running {
			return true
		}
	}
	return false
}

// startKeepAwakeActiveMonitor monitors session activity and acquires a wake lock while any
// session is in Running state. It returns a cleanup function to stop the monitor.
func startKeepAwakeActiveMonitor(ctx context.Context, sessions *session.Table) func() {
	var (
		mu       sync.Mutex
		releaser power.Releaser
	)
	ticker := time.NewTicker(500 * time.Millisecond)
	stop := make(chan struct{})

	check := func() {
		mu.Lock()
		defer mu.Unlock()
		active := hasActiveSessions(sessions)
		if active && releaser == nil {
			releaser, _ = power.Acquire("fak serve (active sessions)", power.PreventSystemSleep)
		} else if !active && releaser != nil {
			_ = releaser.Release()
			releaser = nil
		}
	}

	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				mu.Lock()
				if releaser != nil {
					_ = releaser.Release()
					releaser = nil
				}
				mu.Unlock()
				return
			case <-stop:
				mu.Lock()
				if releaser != nil {
					_ = releaser.Release()
					releaser = nil
				}
				mu.Unlock()
				return
			case <-ticker.C:
				check()
			}
		}
	}()

	return func() {
		close(stop)
	}
}
