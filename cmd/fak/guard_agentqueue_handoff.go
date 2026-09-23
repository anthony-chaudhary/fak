package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
)

const (
	guardAgentQueueStatePathEnv = "FAK_AGENTQUEUE_STATE_PATH"
	guardAgentQueueAttemptIDEnv = "FAK_AGENTQUEUE_ATTEMPT_ID"
	guardAgentQueueNonceEnv     = "FAK_AGENTQUEUE_NONCE"
	guardAgentQueueHoldParked   = "GOAL_PARKED"
	guardAgentQueueHoldBudget   = "TIME_BUDGET"
	guardAgentQueueHoldHeadroom = "SYSTEM_COMMIT_HEADROOM"
	guardAgentQueueHoldWitness  = "AWAITING_WORK_WITNESS"
)

type guardAgentQueueLifecycle struct {
	store      agentqueue.Store
	attemptID  string
	nonce      string
	pid        int
	startedAt  time.Time
	running    bool
	aborted    bool
	holdReason string
}

func guardAgentQueueLifecycleFromEnv() (*guardAgentQueueLifecycle, error) {
	path := strings.TrimSpace(os.Getenv(guardAgentQueueStatePathEnv))
	attemptID := strings.TrimSpace(os.Getenv(guardAgentQueueAttemptIDEnv))
	nonce := strings.TrimSpace(os.Getenv(guardAgentQueueNonceEnv))

	// Queue launch identity belongs to this wrapper only. Remove it before any
	// wrapped-agent environment is assembled, especially the fencing nonce.
	_ = os.Unsetenv(guardAgentQueueStatePathEnv)
	_ = os.Unsetenv(guardAgentQueueAttemptIDEnv)
	_ = os.Unsetenv(guardAgentQueueNonceEnv)

	present := 0
	for _, value := range []string{path, attemptID, nonce} {
		if value != "" {
			present++
		}
	}
	if present == 0 {
		return nil, nil
	}
	if present != 3 {
		return nil, errors.New("agentqueue handoff requires state path, attempt id, and nonce together")
	}

	pid := os.Getpid()
	// This timestamp is a portable launch identity token, captured once and reused
	// byte-for-byte for every fenced transition. The queue nonce supplies the
	// cross-process uniqueness; platforms need not expose an OS creation time.
	startedAt := time.Now().UTC()
	lifecycle := &guardAgentQueueLifecycle{
		store: agentqueue.FileStore(path), attemptID: attemptID, nonce: nonce,
		pid: pid, startedAt: startedAt,
	}
	if _, err := lifecycle.store.RegisterWrapper(context.Background(), attemptID, nonce, pid, lifecycle.startedAt); err != nil {
		return nil, fmt.Errorf("agentqueue register wrapper: %w", err)
	}
	return lifecycle, nil
}

func (l *guardAgentQueueLifecycle) markRunning() error {
	if l == nil || l.aborted || l.running {
		return nil
	}
	if _, err := l.store.MarkRunning(context.Background(), l.attemptID, l.nonce, l.pid, l.startedAt); err != nil {
		return fmt.Errorf("agentqueue mark running: %w", err)
	}
	l.running = true
	return nil
}

func (l *guardAgentQueueLifecycle) abortLaunching() error {
	if l == nil || l.running || l.aborted {
		return nil
	}
	if _, err := l.store.AbortLaunching(context.Background(), l.attemptID, l.nonce, l.pid, l.startedAt); err != nil {
		return fmt.Errorf("agentqueue abort launching: %w", err)
	}
	l.aborted = true
	return nil
}

func (l *guardAgentQueueLifecycle) hold(reason string) {
	if l != nil && l.holdReason == "" {
		l.holdReason = reason
	}
}

func (l *guardAgentQueueLifecycle) complete(success, childJoined bool) error {
	if l == nil || !l.running || l.aborted || !childJoined {
		return nil
	}
	if l.holdReason != "" {
		if _, err := l.store.HoldAttempt(context.Background(), l.attemptID, l.nonce, l.pid, l.startedAt, l.holdReason); err != nil {
			return fmt.Errorf("agentqueue hold attempt: %w", err)
		}
		return nil
	}
	if _, err := l.store.CompleteAttempt(context.Background(), l.attemptID, l.nonce, l.pid, l.startedAt, success); err != nil {
		// Keep the durable attempt Running on any write/fencing failure. A later
		// controller reconciliation can then judge the dead wrapper conservatively.
		return fmt.Errorf("agentqueue complete attempt: %w", err)
	}
	return nil
}
