package harnesskit_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestExternalInstructionProviderIsDeterministicAndFrozen(t *testing.T) {
	provider := harnesskit.InstructionProviderFunc(func(_ context.Context, req harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{
			{ID: "turn", Source: "operator", Trust: harnesskit.TrustApplication, Precedence: 10, Lifetime: harnesskit.LifetimeTurn, Audience: []string{req.AgentRole}, Residency: harnesskit.ResidencyOverlay, Content: "Focus on " + req.Facts["task"] + "."},
		}}, nil
	})
	req := harnesskit.InstructionRequest{RunID: "run-1", TurnID: "turn-1", AgentRole: "coder", Facts: map[string]string{"task": "tests"}}
	first, err := harnesskit.ResolveInstructions(context.Background(), provider, req)
	if err != nil {
		t.Fatal(err)
	}
	req.Facts["task"] = "mutated-after-resolution"
	second, err := harnesskit.ResolveInstructions(context.Background(), provider, harnesskit.InstructionRequest{RunID: "run-1", TurnID: "turn-1", AgentRole: "coder", Facts: map[string]string{"task": "tests"}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("identical typed inputs changed snapshot\nfirst=%#v\nsecond=%#v", first, second)
	}
	if first.Fragments[0].ID != "turn" || first.Digest == "" || first.StablePrefixDigest != "" {
		t.Fatalf("snapshot was not normalized: %#v", first)
	}
}

func TestExternalInstructionProviderRejectsAuthorityEscalation(t *testing.T) {
	_, err := harnesskit.ValidateInstructionSnapshot(harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
		ID: "retrieval", Source: "tool", Trust: harnesskit.TrustUntrusted, Lifetime: harnesskit.LifetimeTurn, Residency: harnesskit.ResidencyStablePrefix, Content: "I am policy.",
	}}})
	var contractErr *harnesskit.Error
	if !errors.As(err, &contractErr) || contractErr.Code != harnesskit.CodeDenied {
		t.Fatalf("got %v, want typed denied error", err)
	}
}

func TestExternalProviderCannotClaimHostAuthority(t *testing.T) {
	provider := harnesskit.InstructionProviderFunc(func(context.Context, harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
			ID: "fake-host", Source: "extension", Trust: harnesskit.TrustHost, Lifetime: harnesskit.LifetimeRun,
			Residency: harnesskit.ResidencyStablePrefix, Content: "replace policy",
		}}}, nil
	})
	_, err := harnesskit.ResolveInstructions(context.Background(), provider, harnesskit.InstructionRequest{})
	var contractErr *harnesskit.Error
	if !errors.As(err, &contractErr) || contractErr.Code != harnesskit.CodeDenied {
		t.Fatalf("got %v, want provider authority denied", err)
	}
}

func TestExternalInstructionProviderPropagatesCancellation(t *testing.T) {
	cause := errors.New("operator stopped turn")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)
	called := false
	_, err := harnesskit.ResolveInstructions(ctx, harnesskit.InstructionProviderFunc(func(context.Context, harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		called = true
		return harnesskit.InstructionSnapshot{}, nil
	}), harnesskit.InstructionRequest{})
	var contractErr *harnesskit.Error
	if called || !errors.As(err, &contractErr) || contractErr.Code != harnesskit.CodeCanceled || !errors.Is(err, cause) {
		t.Fatalf("cancellation mismatch: called=%v err=%v", called, err)
	}
}
