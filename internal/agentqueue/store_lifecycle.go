package agentqueue

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// ErrFenced means an attempt transition presented stale state, nonce, or
// wrapper identity and must not be retried as a fresh launch.
var ErrFenced = errors.New("agentqueue: attempt fenced")

// BeginLaunching durably advances one reserved attempt into its pre-start
// launching state. Repeating the same nonce returns the persisted attempt
// without another write; a competing nonce or any other state is fenced.
func (s Store) BeginLaunching(ctx context.Context, attemptID, nonce string, deadline time.Time) (Attempt, bool, error) {
	if attemptID == "" {
		return Attempt{}, false, errors.New("agentqueue: attempt id is required")
	}
	if nonce == "" {
		return Attempt{}, false, errors.New("agentqueue: launch nonce is required")
	}
	if deadline.IsZero() {
		return Attempt{}, false, errors.New("agentqueue: launch deadline is required")
	}
	deadline = deadline.UTC()
	if !time.Now().UTC().Before(deadline) {
		return Attempt{}, false, fmt.Errorf("%w: launch deadline must be in the future", ErrFenced)
	}

	var result Attempt
	started := false
	err := s.withLifecycleState(ctx, "begin launching", func(snapshot *Snapshot) (bool, error) {
		attempt, err := findLifecycleAttempt(snapshot, attemptID)
		if err != nil {
			return false, err
		}
		switch attempt.State {
		case AttemptReserved:
			if !time.Now().UTC().Before(deadline) {
				return false, fmt.Errorf("%w: attempt %q launch deadline elapsed", ErrFenced, attemptID)
			}
			if attempt.Nonce != "" || !attempt.LaunchDeadline.IsZero() || attempt.PID != 0 || !attempt.StartedAt.IsZero() {
				return false, fmt.Errorf("%w: reserved attempt %q carries launch identity", ErrFenced, attemptID)
			}
			attempt.State = AttemptLaunching
			attempt.Nonce = nonce
			attempt.LaunchDeadline = deadline
			started = true
			result = *attempt
			snapshot.Generation = nextLifecycleGeneration(snapshot.Generation, "launch", result)
			return true, nil
		case AttemptLaunching:
			if attempt.Nonce != nonce {
				return false, fmt.Errorf("%w: attempt %q has a different launch nonce", ErrFenced, attemptID)
			}
			result = *attempt
			return false, nil
		default:
			return false, fmt.Errorf("%w: attempt %q is %q", ErrFenced, attemptID, attempt.State)
		}
	})
	return result, started, err
}

// RegisterWrapper binds a still-live launching attempt to the wrapper process
// identity. The same PID and start instant are idempotent; stale or competing
// identities are fenced.
func (s Store) RegisterWrapper(ctx context.Context, attemptID, nonce string, pid int, startAt time.Time) (Attempt, error) {
	if attemptID == "" {
		return Attempt{}, errors.New("agentqueue: attempt id is required")
	}
	if nonce == "" {
		return Attempt{}, errors.New("agentqueue: launch nonce is required")
	}
	if pid <= 0 {
		return Attempt{}, errors.New("agentqueue: wrapper PID must be positive")
	}
	if startAt.IsZero() {
		return Attempt{}, errors.New("agentqueue: wrapper start time is required")
	}
	startAt = startAt.UTC()

	var result Attempt
	err := s.withLifecycleState(ctx, "register wrapper", func(snapshot *Snapshot) (bool, error) {
		attempt, err := findLifecycleAttempt(snapshot, attemptID)
		if err != nil {
			return false, err
		}
		if attempt.State != AttemptLaunching || attempt.Nonce != nonce {
			return false, fmt.Errorf("%w: attempt %q is not the matching launch", ErrFenced, attemptID)
		}
		if attempt.PID != 0 || !attempt.StartedAt.IsZero() {
			if attempt.PID == pid && attempt.StartedAt.Equal(startAt) {
				result = *attempt
				return false, nil
			}
			return false, fmt.Errorf("%w: attempt %q has a different wrapper identity", ErrFenced, attemptID)
		}
		if attempt.LaunchDeadline.IsZero() || !time.Now().UTC().Before(attempt.LaunchDeadline) {
			return false, fmt.Errorf("%w: attempt %q launch deadline elapsed", ErrFenced, attemptID)
		}
		attempt.PID = pid
		attempt.StartedAt = startAt
		result = *attempt
		snapshot.Generation = nextLifecycleGeneration(snapshot.Generation, "register", result)
		return true, nil
	})
	return result, err
}

// MarkRunning durably promotes a registered wrapper and its intent to running.
// Repeating the same identity after promotion is an idempotent readback.
func (s Store) MarkRunning(ctx context.Context, attemptID, nonce string, pid int, startAt time.Time) (Attempt, error) {
	startAt, err := validateLifecycleIdentity(attemptID, nonce, pid, startAt)
	if err != nil {
		return Attempt{}, err
	}
	var result Attempt
	err = s.withLifecycleState(ctx, "mark running", func(snapshot *Snapshot) (bool, error) {
		attempt, err := findLifecycleAttempt(snapshot, attemptID)
		if err != nil {
			return false, err
		}
		intent, err := findLifecycleIntent(snapshot, attempt.IntentID)
		if err != nil {
			return false, err
		}
		if attempt.Nonce != nonce || attempt.PID != pid || !attempt.StartedAt.Equal(startAt) {
			return false, fmt.Errorf("%w: attempt %q has a different wrapper identity", ErrFenced, attemptID)
		}
		switch attempt.State {
		case AttemptLaunching:
			if intent.State != IntentQueued {
				return false, fmt.Errorf("%w: intent %q is %q before running", ErrFenced, intent.ID, intent.State)
			}
			attempt.State = AttemptRunning
			intent.State = IntentRunning
			intent.PID = pid
			result = *attempt
			snapshot.Generation = nextLifecycleGeneration(snapshot.Generation, "running", result)
			return true, nil
		case AttemptRunning:
			if intent.State != IntentRunning || intent.PID != pid {
				return false, fmt.Errorf("%w: intent %q does not match running attempt", ErrFenced, intent.ID)
			}
			result = *attempt
			return false, nil
		default:
			return false, fmt.Errorf("%w: attempt %q is %q", ErrFenced, attemptID, attempt.State)
		}
	})
	return result, err
}

// CompleteAttempt durably records one running wrapper's terminal result on the
// attempt and intent. It never releases external resource grants.
func (s Store) CompleteAttempt(ctx context.Context, attemptID, nonce string, pid int, startAt time.Time, success bool) (Attempt, error) {
	startAt, err := validateLifecycleIdentity(attemptID, nonce, pid, startAt)
	if err != nil {
		return Attempt{}, err
	}
	targetAttempt := AttemptFailed
	targetIntent := IntentFailed
	transition := "failed"
	if success {
		targetAttempt = AttemptSucceeded
		targetIntent = IntentCompleted
		transition = "succeeded"
	}

	var result Attempt
	err = s.withLifecycleState(ctx, "complete attempt", func(snapshot *Snapshot) (bool, error) {
		attempt, err := findLifecycleAttempt(snapshot, attemptID)
		if err != nil {
			return false, err
		}
		intent, err := findLifecycleIntent(snapshot, attempt.IntentID)
		if err != nil {
			return false, err
		}
		if attempt.Nonce != nonce || attempt.PID != pid || !attempt.StartedAt.Equal(startAt) {
			return false, fmt.Errorf("%w: attempt %q has a different wrapper identity", ErrFenced, attemptID)
		}
		switch attempt.State {
		case AttemptRunning:
			if intent.State != IntentRunning || intent.PID != pid {
				return false, fmt.Errorf("%w: intent %q does not match running attempt", ErrFenced, intent.ID)
			}
			attempt.State = targetAttempt
			intent.State = targetIntent
			result = *attempt
			snapshot.Generation = nextLifecycleGeneration(snapshot.Generation, transition, result)
			return true, nil
		case targetAttempt:
			if intent.State != targetIntent || intent.PID != pid {
				return false, fmt.Errorf("%w: intent %q does not match terminal attempt", ErrFenced, intent.ID)
			}
			result = *attempt
			return false, nil
		default:
			return false, fmt.Errorf("%w: attempt %q is %q, not %q", ErrFenced, attemptID, attempt.State, targetAttempt)
		}
	})
	return result, err
}

func validateLifecycleIdentity(attemptID, nonce string, pid int, startAt time.Time) (time.Time, error) {
	if attemptID == "" {
		return time.Time{}, errors.New("agentqueue: attempt id is required")
	}
	if nonce == "" {
		return time.Time{}, errors.New("agentqueue: launch nonce is required")
	}
	if pid <= 0 {
		return time.Time{}, errors.New("agentqueue: wrapper PID must be positive")
	}
	if startAt.IsZero() {
		return time.Time{}, errors.New("agentqueue: wrapper start time is required")
	}
	return startAt.UTC(), nil
}

func (s Store) withLifecycleState(ctx context.Context, operation string, mutate func(*Snapshot) (bool, error)) error {
	if s.Path == "" {
		return errors.New("agentqueue: snapshot path is required")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	lock, err := os.OpenFile(s.Path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("agentqueue: open lifecycle lock: %w", err)
	}
	defer lock.Close()
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("agentqueue: lock lifecycle: %w", err)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("agentqueue: %s: %w", operation, ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
	defer flock.Unlock(lock)

	snapshot, err := s.Load()
	if err != nil {
		return err
	}
	changed, err := mutate(&snapshot)
	if err != nil || !changed {
		return err
	}
	return s.Save(snapshot)
}

func findLifecycleAttempt(snapshot *Snapshot, attemptID string) (*Attempt, error) {
	var found *Attempt
	for i := range snapshot.Attempts {
		if snapshot.Attempts[i].ID != attemptID {
			continue
		}
		if found != nil {
			return nil, fmt.Errorf("%w: duplicate attempt id %q", ErrFenced, attemptID)
		}
		found = &snapshot.Attempts[i]
	}
	if found == nil {
		return nil, fmt.Errorf("%w: attempt %q is absent", ErrFenced, attemptID)
	}
	return found, nil
}

func findLifecycleIntent(snapshot *Snapshot, intentID string) (*Intent, error) {
	var found *Intent
	for i := range snapshot.Intents {
		if snapshot.Intents[i].ID != intentID {
			continue
		}
		if found != nil {
			return nil, fmt.Errorf("%w: duplicate intent id %q", ErrFenced, intentID)
		}
		found = &snapshot.Intents[i]
	}
	if found == nil {
		return nil, fmt.Errorf("%w: intent %q is absent", ErrFenced, intentID)
	}
	return found, nil
}

func nextLifecycleGeneration(current, transition string, attempt Attempt) string {
	h := sha256.New()
	_, _ = h.Write([]byte(current))
	_, _ = h.Write([]byte("\x00lifecycle\x00" + transition + "\x00" + attempt.ID + "\x00" + attempt.Nonce))
	_, _ = h.Write([]byte("\x00" + strconv.Itoa(attempt.PID)))
	_, _ = h.Write([]byte("\x00" + attempt.LaunchDeadline.UTC().Format(time.RFC3339Nano)))
	_, _ = h.Write([]byte("\x00" + attempt.StartedAt.UTC().Format(time.RFC3339Nano)))
	return "gen:" + hex.EncodeToString(h.Sum(nil)[:16])
}
