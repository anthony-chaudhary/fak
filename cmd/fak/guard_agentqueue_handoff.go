package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
	"github.com/anthony-chaudhary/fak/internal/processalive"
)

const (
	guardAgentQueueStatePathEnv = "FAK_AGENTQUEUE_STATE_PATH"
	guardAgentQueueAttemptIDEnv = "FAK_AGENTQUEUE_ATTEMPT_ID"
	guardAgentQueueNonceEnv     = "FAK_AGENTQUEUE_NONCE"
	guardAgentQueueHoldParked   = "GOAL_PARKED"
	guardAgentQueueHoldBudget   = "TIME_BUDGET"
	guardAgentQueueHoldHeadroom = "SYSTEM_COMMIT_HEADROOM"
	guardAgentQueueHoldWitness  = agentqueue.HoldAwaitingWorkWitness
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

// validateDispatchAgentQueueHandoff binds an opaque CLI handoff to the durable
// attempt before the dispatcher can start a guarded worker. The wrapper still
// performs the final locked registration, which fences a concurrent contender.
func validateDispatchAgentQueueHandoff(path, attemptID, nonce string, issue int, lane string) error {
	return checkDispatchAgentQueueHandoff(path, attemptID, nonce, issue, lane, true)
}

// The spawner first proves it owns the still-unregistered nonce before setting
// up its no-start abort. That permits a deadline expiry during preparation to
// return the attempt to the queue without letting a stale nonce abort a winner.
func validateDispatchAgentQueueHandoffIdentity(path, attemptID, nonce string, issue int, lane string) error {
	return checkDispatchAgentQueueHandoff(path, attemptID, nonce, issue, lane, false)
}

func checkDispatchAgentQueueHandoff(path, attemptID, nonce string, issue int, lane string, requireFreshDeadline bool) error {
	if path == "" || !filepath.IsAbs(path) || attemptID == "" || nonce == "" || issue <= 0 || lane == "" {
		return errors.New("agentqueue handoff requires absolute state path, attempt, nonce, issue, and lane")
	}
	snapshot, err := agentqueue.FileStore(path).Load()
	if err != nil {
		return fmt.Errorf("agentqueue load handoff: %w", err)
	}
	for _, attempt := range snapshot.Attempts {
		if attempt.ID != attemptID {
			continue
		}
		if attempt.State != agentqueue.AttemptLaunching || attempt.Nonce != nonce || attempt.PID != 0 || !attempt.StartedAt.IsZero() ||
			attempt.LaunchDeadline.IsZero() || (requireFreshDeadline && !time.Now().UTC().Before(attempt.LaunchDeadline)) {
			return fmt.Errorf("agentqueue handoff attempt %q is not an unregistered live launch", attemptID)
		}
		for _, intent := range snapshot.Intents {
			if intent.ID == attempt.IntentID {
				if intent.State != agentqueue.IntentQueued || intent.Launch.Issue != issue || intent.Launch.Lane != lane {
					return fmt.Errorf("agentqueue handoff attempt %q does not match its routed intent", attemptID)
				}
				return nil
			}
		}
		return fmt.Errorf("agentqueue handoff attempt %q has no intent", attemptID)
	}
	return fmt.Errorf("agentqueue handoff attempt %q is missing", attemptID)
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
	// Use kernel process creation time where available so restart can reject a
	// later process that reused this PID. Other platforms keep an opaque token;
	// restart then holds the attempt because kernel identity cannot be proved.
	startedAt, ok := processalive.StartTime(pid)
	if !ok {
		startedAt = time.Now().UTC()
	}
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
