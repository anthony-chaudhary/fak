package harnessinstructions

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestOutcomeRecorderCountsRealResolutionOutcomes(t *testing.T) {
	recorder := new(OutcomeRecorder)
	provider := harnesskit.InstructionProviderFunc(func(_ context.Context, req harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		fragment := harnesskit.InstructionFragment{
			ID:         "operator-focus",
			Source:     "test/operator",
			Trust:      harnesskit.TrustApplication,
			Lifetime:   harnesskit.LifetimeTurn,
			Residency:  harnesskit.ResidencyEphemeralTail,
			Content:    req.Facts["focus"],
			Precedence: 10,
		}
		if req.Facts["focus"] == "denied" {
			fragment.Residency = harnesskit.ResidencyStablePrefix
		}
		return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{fragment}}, nil
	})

	if _, err := recorder.Resolve(context.Background(), provider, harnesskit.InstructionRequest{Facts: map[string]string{"focus": "correctness"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := recorder.Resolve(context.Background(), provider, harnesskit.InstructionRequest{Facts: map[string]string{"focus": "latency"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := recorder.Resolve(context.Background(), provider, harnesskit.InstructionRequest{Facts: map[string]string{"focus": "denied"}}); err == nil {
		t.Fatal("stable-prefix claim unexpectedly succeeded")
	}
	rawFailure := errors.New("provider unavailable")
	if _, err := recorder.Resolve(context.Background(), harnesskit.InstructionProviderFunc(func(context.Context, harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		return harnesskit.InstructionSnapshot{}, rawFailure
	}), harnesskit.InstructionRequest{}); !errors.Is(err, rawFailure) {
		t.Fatalf("raw provider error = %v, want %v", err, rawFailure)
	}

	got := recorder.Counts()
	if got.Invocations != 4 || got.Succeeded != 2 || got.Failed != 2 || got.Unclassified != 1 {
		t.Fatalf("outcomes = %#v", got)
	}
	if got.ByCode[harnesskit.CodeDenied] != 1 {
		t.Fatalf("denied = %d, want 1; outcomes=%#v", got.ByCode[harnesskit.CodeDenied], got)
	}
	if got.Succeeded+got.Failed != got.Invocations {
		t.Fatalf("succeeded(%d) + failed(%d) != invocations(%d)", got.Succeeded, got.Failed, got.Invocations)
	}

	got.ByCode[harnesskit.CodeDenied] = 99
	if fresh := recorder.Counts(); fresh.ByCode[harnesskit.CodeDenied] != 1 {
		t.Fatalf("caller mutated recorder snapshot: %#v", fresh)
	}
}
