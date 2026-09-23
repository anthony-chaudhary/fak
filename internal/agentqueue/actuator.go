package agentqueue

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// LaunchSpec maps durable intent to the existing guarded dispatch lifecycle.
// Dispatch tick remains the single owner of DOS admission, lane leases,
// detached worktrees, fak manage, landing, witness, and cleanup.
type LaunchSpec struct {
	Issue int    `json:"issue"`
	Lane  string `json:"lane"`
}

type LaunchReceipt struct {
	IntentID       string   `json:"intent_id"`
	IdempotencyKey string   `json:"idempotency_key"`
	Command        []string `json:"command"`
}

type CommandRunner interface {
	Run(context.Context, string, ...string) error
}

type ExecRunner struct{}

func (ExecRunner) Run(ctx context.Context, name string, args ...string) error {
	command := exec.CommandContext(ctx, name, args...)
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

// Actuate executes only starts accepted by Store.Reserve. It deliberately does
// not implement another worker executor; each accepted reservation is handed to
// fak dispatch tick, the repository's guarded end-to-end worker lifecycle.
func Actuate(ctx context.Context, fakPath string, snapshot Snapshot, starts []StartAction, runner CommandRunner) ([]LaunchReceipt, error) {
	return actuate(ctx, fakPath, nil, snapshot, starts, runner)
}

// ActuateReserved persists the launch identity before invoking dispatch. A
// crash after this transition is ambiguous and must not requeue the attempt.
func ActuateReserved(ctx context.Context, fakPath string, store Store, snapshot Snapshot, starts []StartAction, runner CommandRunner) ([]LaunchReceipt, error) {
	if strings.TrimSpace(store.Path) == "" {
		return nil, errors.New("agentqueue: queue state path is required")
	}
	absolutePath, err := filepath.Abs(store.Path)
	if err != nil {
		return nil, fmt.Errorf("agentqueue: resolve queue state path: %w", err)
	}
	store.Path = absolutePath
	return actuate(ctx, fakPath, &store, snapshot, starts, runner)
}

func actuate(ctx context.Context, fakPath string, store *Store, snapshot Snapshot, starts []StartAction, runner CommandRunner) ([]LaunchReceipt, error) {
	if strings.TrimSpace(fakPath) == "" {
		err := errors.New("agentqueue: fak executable is required")
		if store != nil {
			err = errors.Join(err, abortReservedStarts(ctx, *store, starts))
		}
		return nil, err
	}
	if runner == nil {
		err := errors.New("agentqueue: command runner is required")
		if store != nil {
			err = errors.Join(err, abortReservedStarts(ctx, *store, starts))
		}
		return nil, err
	}
	intents := make(map[string]Intent, len(snapshot.Intents))
	for _, intent := range snapshot.Intents {
		intents[intent.ID] = intent
	}
	if store != nil {
		for _, start := range starts {
			intent, ok := intents[start.IntentID]
			if !ok || intent.Launch.Issue <= 0 || strings.TrimSpace(intent.Launch.Lane) == "" {
				return nil, errors.Join(fmt.Errorf("agentqueue: start %q has no routable intent", start.IdempotencyKey), abortReservedStarts(ctx, *store, starts))
			}
			attempt, err := findLifecycleAttempt(&snapshot, start.IdempotencyKey)
			if err != nil || attempt.State != AttemptReserved || attempt.IntentID != start.IntentID {
				return nil, errors.Join(fmt.Errorf("agentqueue: start %q is not its reserved attempt: %w", start.IdempotencyKey, ErrFenced), abortReservedStarts(ctx, *store, starts))
			}
		}
	}
	receipts := make([]LaunchReceipt, 0, len(starts))
	for index, start := range starts {
		intent, ok := intents[start.IntentID]
		if !ok {
			return receipts, fmt.Errorf("agentqueue: start references unknown intent %q", start.IntentID)
		}
		if intent.Launch.Issue <= 0 || strings.TrimSpace(intent.Launch.Lane) == "" {
			return receipts, fmt.Errorf("agentqueue: intent %q requires launch issue and lane", intent.ID)
		}
		args := []string{
			"dispatch", "tick",
			"--target-issue", strconv.Itoa(intent.Launch.Issue),
			"--lane", intent.Launch.Lane,
			"--lease-id", start.IdempotencyKey,
		}
		if store != nil {
			nonceBytes := make([]byte, 16)
			if _, err := rand.Read(nonceBytes); err != nil {
				return receipts, errors.Join(fmt.Errorf("agentqueue: mint launch nonce: %w", err), abortReservedStarts(ctx, *store, starts[index:]))
			}
			nonce := hex.EncodeToString(nonceBytes)
			launchCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_, started, beginErr := store.BeginLaunching(launchCtx, start.IdempotencyKey, nonce, time.Now().UTC().Add(10*time.Minute))
			cancel()
			if beginErr != nil || !started {
				if beginErr == nil {
					beginErr = fmt.Errorf("%w: attempt %q already launching", ErrFenced, start.IdempotencyKey)
				}
				return receipts, errors.Join(fmt.Errorf("agentqueue: begin launch intent %q: %w", intent.ID, beginErr), abortReservedStarts(ctx, *store, starts[index+1:]))
			}
			args = append(args, "--agentqueue-state", store.Path, "--agentqueue-attempt", start.IdempotencyKey, "--agentqueue-nonce", nonce)
		}
		args = append(args, "--live", "--json")
		runErr := runner.Run(ctx, fakPath, args...)
		if store != nil {
			attempt, intentState, stateErr := currentAttemptAndIntent(*store, start.IdempotencyKey)
			if stateErr != nil {
				return receipts, errors.Join(fmt.Errorf("agentqueue: inspect launch intent %q: %w", intent.ID, stateErr), runErr, abortReservedStarts(ctx, *store, starts[index+1:]))
			}
			if attempt.State == AttemptFailed && intentState == IntentQueued {
				// Dispatch proved non-start and returned this attempt to the queue.
				if ctxErr := ctx.Err(); ctxErr != nil {
					return receipts, errors.Join(ctxErr, abortReservedStarts(ctx, *store, starts[index+1:]))
				}
				continue
			}
			if runErr != nil {
				return receipts, errors.Join(fmt.Errorf("agentqueue: launch intent %q: %w", intent.ID, runErr), abortReservedStarts(ctx, *store, starts[index+1:]))
			}
		} else if runErr != nil {
			return receipts, fmt.Errorf("agentqueue: launch intent %q: %w", intent.ID, runErr)
		}
		receipts = append(receipts, LaunchReceipt{IntentID: intent.ID, IdempotencyKey: start.IdempotencyKey, Command: append([]string{fakPath}, args...)})
	}
	return receipts, nil
}

func currentAttemptAndIntent(store Store, id string) (Attempt, IntentState, error) {
	snapshot, err := store.Load()
	if err != nil {
		return Attempt{}, "", err
	}
	attempt, err := findLifecycleAttempt(&snapshot, id)
	if err != nil {
		return Attempt{}, "", err
	}
	intent, err := findLifecycleIntent(&snapshot, attempt.IntentID)
	if err != nil {
		return Attempt{}, "", err
	}
	return *attempt, intent.State, nil
}

func abortReservedStarts(_ context.Context, store Store, starts []StartAction) error {
	var errs []error
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, start := range starts {
		if _, err := store.AbortReserved(ctx, start.IdempotencyKey); err != nil {
			errs = append(errs, fmt.Errorf("agentqueue: return unstarted attempt %q: %w", start.IdempotencyKey, err))
		}
	}
	return errors.Join(errs...)
}
