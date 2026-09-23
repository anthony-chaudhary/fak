package agentqueue

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
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
	AttemptID      string   `json:"attempt_id"`
	Nonce          string   `json:"nonce"`
	Command        []string `json:"command"`
}

// LaunchHandoff is the durable launch identity the parent fenced before process
// creation and hands to the guarded dispatch wrapper. AttemptID and Nonce are
// the values persisted by Store.BeginLaunching; StatePath is the absolute
// durable snapshot path so the wrapper (or a later reconciliation) can read the
// exact attempt back without guessing after a crash.
type LaunchHandoff struct {
	AttemptID string `json:"attempt_id"`
	Nonce     string `json:"nonce"`
	StatePath string `json:"queue_state"`
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
//
// Every start must carry the handoff fenced by Store.BeginLaunching, so no
// process is created before its launch identity is durable. Actuate never
// registers the wrapper or marks the attempt running/completed: dispatch tick
// stays the single owner of the downstream lifecycle.
func Actuate(ctx context.Context, fakPath string, snapshot Snapshot, starts []StartAction, handoffs map[string]LaunchHandoff, runner CommandRunner) ([]LaunchReceipt, error) {
	if strings.TrimSpace(fakPath) == "" {
		return nil, errors.New("agentqueue: fak executable is required")
	}
	if runner == nil {
		return nil, errors.New("agentqueue: command runner is required")
	}
	intents := make(map[string]Intent, len(snapshot.Intents))
	for _, intent := range snapshot.Intents {
		intents[intent.ID] = intent
	}
	receipts := make([]LaunchReceipt, 0, len(starts))
	for _, start := range starts {
		intent, ok := intents[start.IntentID]
		if !ok {
			return receipts, fmt.Errorf("agentqueue: start references unknown intent %q", start.IntentID)
		}
		if intent.Launch.Issue <= 0 || strings.TrimSpace(intent.Launch.Lane) == "" {
			return receipts, fmt.Errorf("agentqueue: intent %q requires launch issue and lane", intent.ID)
		}
		handoff, ok := handoffs[start.IdempotencyKey]
		if !ok {
			return receipts, fmt.Errorf("agentqueue: start %q has no fenced launch handoff", start.IdempotencyKey)
		}
		if handoff.AttemptID != start.IdempotencyKey || handoff.Nonce == "" || handoff.StatePath == "" {
			return receipts, fmt.Errorf("agentqueue: start %q has an incomplete launch handoff", start.IdempotencyKey)
		}
		args := []string{
			"dispatch", "tick",
			"--target-issue", strconv.Itoa(intent.Launch.Issue),
			"--lane", intent.Launch.Lane,
			"--lease-id", start.IdempotencyKey,
			"--attempt-id", handoff.AttemptID,
			"--launch-nonce", handoff.Nonce,
			"--queue-state", handoff.StatePath,
			"--live", "--json",
		}
		if err := runner.Run(ctx, fakPath, args...); err != nil {
			return receipts, fmt.Errorf("agentqueue: launch intent %q: %w", intent.ID, err)
		}
		receipts = append(receipts, LaunchReceipt{
			IntentID:       intent.ID,
			IdempotencyKey: start.IdempotencyKey,
			AttemptID:      handoff.AttemptID,
			Nonce:          handoff.Nonce,
			Command:        append([]string{fakPath}, args...),
		})
	}
	return receipts, nil
}
