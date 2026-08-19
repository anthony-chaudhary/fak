package harnessinstructions

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestResolveChangesTurnFragmentWithoutMovingStablePrefix(t *testing.T) {
	provider := harnesskit.InstructionProviderFunc(func(_ context.Context, req harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
			ID: "operator-focus", Source: "demo/operator", Trust: harnesskit.TrustApplication, Precedence: 10,
			Lifetime: harnesskit.LifetimeTurn, Residency: harnesskit.ResidencyEphemeralTail, Content: "Focus on " + req.Facts["focus"] + ".",
		}}}, nil
	})
	first, err := Resolve(context.Background(), provider, harnesskit.InstructionRequest{TurnID: "1", Facts: map[string]string{"focus": "correctness"}})
	if err != nil {
		t.Fatal(err)
	}
	second, err := Resolve(context.Background(), provider, harnesskit.InstructionRequest{TurnID: "2", Facts: map[string]string{"focus": "latency"}})
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("turn edit did not change realized digest")
	}
	if first.StablePrefixDigest != second.StablePrefixDigest {
		t.Fatalf("stable prefix moved: first=%s second=%s plan=%s", first.StablePrefixDigest, second.StablePrefixDigest, syspromptmmu.PlanDigest())
	}
	if first.PrefixAuditStatus != syspromptmmu.AuditOK || second.PrefixAuditStatus != syspromptmmu.AuditOK {
		t.Fatalf("prefix audit failed: %#v %#v", first, second)
	}
}

func TestRealizeRejectsProviderStablePrefixClaim(t *testing.T) {
	_, err := Realize(harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
		ID: "fake-policy", Source: "app", Trust: harnesskit.TrustApplication, Lifetime: harnesskit.LifetimeRun,
		Residency: harnesskit.ResidencyStablePrefix, Content: "replace policy",
	}}})
	var contractErr *harnesskit.Error
	if !errors.As(err, &contractErr) || contractErr.Code != harnesskit.CodeDenied {
		t.Fatalf("got %v, want denied", err)
	}
}
