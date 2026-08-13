package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const retryAblationFailure = "fixture transient: upstream reset"

type microRetryAblation struct {
	WithoutRetryCompleted bool     `json:"without_retry_completed"`
	WithRetryCompleted    bool     `json:"with_retry_completed"`
	WithoutRetryAttempts  int      `json:"without_retry_attempts"`
	WithRetryAttempts     int      `json:"with_retry_attempts"`
	Evidence              []string `json:"evidence"`
}

// retryAblationAgent is a deterministic fault-injection fixture around the real
// native Host retry seam. The second attempt can succeed only after the Host
// passes the exact first error through RetryFeedback, so this is an ablation of
// grounded retry rather than a synthetic provider-quality comparison.
type retryAblationAgent struct {
	attempts int
	evidence []string
}

func (a *retryAblationAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	a.attempts++
	if len(a.evidence) == 0 {
		return false, errors.New(retryAblationFailure)
	}
	return true, nil
}

func (a *retryAblationAgent) RetryFeedback(_ context.Context, evidence error) error {
	if evidence == nil {
		return errors.New("retry fixture received nil evidence")
	}
	a.evidence = append(a.evidence, evidence.Error())
	return nil
}

func runMicroRetryAblation(ctx context.Context) (microRetryAblation, error) {
	offAgent := &retryAblationAgent{}
	off, err := runRetryAblationArm(ctx, 0, offAgent)
	if err != nil {
		return microRetryAblation{}, fmt.Errorf("retry-off arm: %w", err)
	}
	onAgent := &retryAblationAgent{}
	on, err := runRetryAblationArm(ctx, 1, onAgent)
	if err != nil {
		return microRetryAblation{}, fmt.Errorf("retry-on arm: %w", err)
	}
	return microRetryAblation{
		WithoutRetryCompleted: off.Done && off.Err == nil,
		WithRetryCompleted:    on.Done && on.Err == nil,
		WithoutRetryAttempts:  offAgent.attempts,
		WithRetryAttempts:     onAgent.attempts,
		Evidence:              append([]string(nil), onAgent.evidence...),
	}, nil
}

func runRetryAblationArm(parent context.Context, maxRetries int, fixture *retryAblationAgent) (microagent.Result, error) {
	h, err := microagent.NewHost(agent.NewMockPlanner("retry-ablation"), microagent.Config{Workers: 1, MaxRetries: maxRetries})
	if err != nil {
		return microagent.Result{}, err
	}
	defer h.Close()
	if err := h.Spawn("retry-ablation", fixture); err != nil {
		return microagent.Result{}, err
	}
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		return microagent.Result{}, err
	}
	results := h.Reap()
	if len(results) != 1 {
		return microagent.Result{}, fmt.Errorf("results=%d, want 1", len(results))
	}
	return results[0], nil
}

func retryAblationPassed(r microRetryAblation) bool {
	return !r.WithoutRetryCompleted && r.WithRetryCompleted &&
		r.WithoutRetryAttempts == 1 && r.WithRetryAttempts == 2 &&
		len(r.Evidence) == 1 && r.Evidence[0] == retryAblationFailure
}

const verifierAblationClaim = "claimed-effect.txt"

type microVerifierAblation struct {
	WithoutVerifierCompleted bool   `json:"without_verifier_completed"`
	WithVerifierCompleted    bool   `json:"with_verifier_completed"`
	WithVerifierCaught       bool   `json:"with_verifier_caught"`
	Readback                 string `json:"readback"`
	Evidence                 string `json:"evidence"`
}

type claimedDoneAgent struct{}

func (*claimedDoneAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	return true, nil
}

func runMicroVerifierAblation(ctx context.Context) (microVerifierAblation, error) {
	without, err := runVerifierAblationArm(ctx, nil)
	if err != nil {
		return microVerifierAblation{}, fmt.Errorf("verifier-off arm: %w", err)
	}
	artifact := filepath.Join(os.TempDir(), fmt.Sprintf("fak-micro-verify-%d", time.Now().UnixNano()), verifierAblationClaim)
	readback := "artifact-absent"
	verifier := microagent.VerifierFunc(func(_ context.Context, _ microagent.VerificationInput) error {
		_, readErr := os.ReadFile(artifact)
		if errors.Is(readErr, os.ErrNotExist) {
			return fmt.Errorf("%s: independent readback found no claimed artifact", readback)
		}
		if readErr != nil {
			return fmt.Errorf("artifact readback: %w", readErr)
		}
		return nil
	})
	with, err := runVerifierAblationArm(ctx, verifier)
	if err != nil {
		return microVerifierAblation{}, fmt.Errorf("verifier-on arm: %w", err)
	}
	evidence := ""
	var verification *microagent.VerificationError
	caught := errors.As(with.Err, &verification)
	if caught && verification.Evidence != nil {
		evidence = verification.Evidence.Error()
	}
	return microVerifierAblation{
		WithoutVerifierCompleted: without.Done && without.Err == nil,
		WithVerifierCompleted:    with.Done && with.Err == nil,
		WithVerifierCaught:       caught,
		Readback:                 readback,
		Evidence:                 evidence,
	}, nil
}

func runVerifierAblationArm(parent context.Context, verifier microagent.Verifier) (microagent.Result, error) {
	h, err := microagent.NewHost(agent.NewMockPlanner("verifier-ablation"), microagent.Config{Workers: 1, Verifier: verifier})
	if err != nil {
		return microagent.Result{}, err
	}
	defer h.Close()
	if err := h.Spawn("verifier-ablation", &claimedDoneAgent{}); err != nil {
		return microagent.Result{}, err
	}
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()
	if err := h.Drain(ctx); err != nil {
		return microagent.Result{}, err
	}
	results := h.Reap()
	if len(results) != 1 {
		return microagent.Result{}, fmt.Errorf("results=%d, want 1", len(results))
	}
	return results[0], nil
}

func verifierAblationPassed(r microVerifierAblation) bool {
	return r.WithoutVerifierCompleted && !r.WithVerifierCompleted && r.WithVerifierCaught &&
		r.Readback == "artifact-absent" && r.Evidence == "artifact-absent: independent readback found no claimed artifact"
}
