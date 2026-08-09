package issuefanout

import (
	"errors"
	"fmt"
	"testing"
)

// The reason code is only worth having if a real refusal from the planner carries it. These
// drive Build's actual contract-refusal path rather than constructing a *Refusal directly, so a
// refusal that stopped being typed would fail here instead of passing on a hand-built fixture
// (#5608).

func TestContractRefusalCarriesItsDeclaredReason(t *testing.T) {
	// An input with no spine witness is the planner's canonical contract refusal.
	_, err := Build(Input{})
	if err == nil {
		t.Fatal("Build(Input{}) planned a fan-out from an empty input; expected a contract refusal")
	}
	if got := ClassifyOutcome(err); got != OutcomeRefused {
		t.Fatalf("ClassifyOutcome = %q, want %q — the refusal is no longer typed, so it can carry no reason", got, OutcomeRefused)
	}
	reason, ok := RefusalReason(err)
	if !ok {
		t.Fatalf("RefusalReason(%v) reported no reason: a contract refusal that names no code cannot be attributed by a consumer routing on the closed vocabulary", err)
	}
	if reason != ContractRefusedReason {
		t.Fatalf("RefusalReason = %q, want %q (the code dos.toml and AGENTS.md declare for this floor)", reason, ContractRefusedReason)
	}
}

// A non-refusal must NOT claim the code. A reason that attaches to every error would let a
// genuine internal failure be routed as a deliberate contract refusal.
func TestRefusalReasonDoesNotClaimUnrelatedErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"nil", nil},
		{"a plain error", errors.New("disk gone")},
		{"a wrapped plain error", fmt.Errorf("outer: %w", errors.New("inner"))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if reason, ok := RefusalReason(tc.err); ok {
				t.Fatalf("RefusalReason(%v) claimed %q; only a contract refusal may carry this code", tc.err, reason)
			}
		})
	}
}

// The code travels through wrapping, or a caller that adds context to a refusal loses the
// attribution the vocabulary exists to provide.
func TestRefusalReasonSurvivesWrapping(t *testing.T) {
	_, err := Build(Input{})
	if err == nil {
		t.Fatal("Build(Input{}) did not refuse")
	}
	wrapped := fmt.Errorf("planning wave 3: %w", err)
	reason, ok := RefusalReason(wrapped)
	if !ok || reason != ContractRefusedReason {
		t.Fatalf("RefusalReason(wrapped) = (%q,%v), want (%q,true)", reason, ok, ContractRefusedReason)
	}
}
