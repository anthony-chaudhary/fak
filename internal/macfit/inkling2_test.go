package macfit

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2MiniTierKVMatchesMetalSession(t *testing.T) {
	kvpt, err := KVBytesPerTokenForTier(Inkling2MiniTier, model.KVPrecisionFP16)
	if err != nil {
		t.Fatal(err)
	}
	if kvpt != 101376 {
		t.Fatalf("KV bytes per token = %d, want 101376", kvpt)
	}
}

func TestMacDefaultTier(t *testing.T) {
	cases := []struct {
		name         string
		goos, goarch string
		memory       uint64
		servable     bool
		want         bool
	}{
		{"36 GiB fits", "darwin", "arm64", 36 * GiB, true, true},
		{"24 GiB fits", "darwin", "arm64", 24 * GiB, true, true},
		{"128 GiB fits", "darwin", "arm64", 128 * GiB, true, true},
		{"18 GiB does not fit", "darwin", "arm64", 18 * GiB, true, false},
		{"16 GiB does not fit", "darwin", "arm64", 16 * GiB, true, false},
		{"8 GiB does not fit", "darwin", "arm64", 8 * GiB, true, false},
		{"serve unsupported", "darwin", "arm64", 36 * GiB, false, false},
		{"intel mac", "darwin", "amd64", 64 * GiB, true, false},
		{"linux", "linux", "amd64", 64 * GiB, true, false},
		{"linux arm64", "linux", "arm64", 64 * GiB, true, false},
		{"windows", "windows", "amd64", 64 * GiB, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tier, ok := MacDefaultTier(tc.goos, tc.goarch, tc.memory, tc.servable)
			if ok != tc.want {
				t.Fatalf("MacDefaultTier ok = %v, want %v", ok, tc.want)
			}
			if ok && tier.ModelID != "inkling2-mini" {
				t.Fatalf("selected %q", tier.ModelID)
			}
			if !ok && tier != (ModelTier{}) {
				t.Fatalf("an unselected default returned tier %+v", tier)
			}
		})
	}
}

func TestInkling2MiniFitsHoldsMinimumContextInsideReserve(t *testing.T) {
	const need = 15_534_497_792 + Inkling2MiniMinContextTokens*101376
	for _, memory := range []uint64{24 * GiB, 36 * GiB} {
		if usable := memory - memory*20/100; usable < need {
			t.Fatalf("%d GiB is declared fitting but holds only %d of %d bytes", memory/GiB, usable, need)
		}
	}
	// A hypothetical host at the declared floor's size class that could not hold
	// the session must be refused by the arithmetic, not only by the floor.
	old := Inkling2MiniTier
	t.Cleanup(func() { Inkling2MiniTier = old })
	Inkling2MiniTier.MinMemoryBytes = 0
	if Inkling2MiniFits(19 * GiB) {
		t.Fatal("19 GiB cannot hold the resident model plus a minimum session inside the reserve")
	}
	if !Inkling2MiniFits(20 * GiB) {
		t.Fatal("20 GiB holds the resident model plus a minimum session inside the reserve")
	}
}

func TestConfigureTurnkeyWithTierOverride(t *testing.T) {
	tier := Inkling2MiniTier
	plan, err := ConfigureTurnkeyWithOptions(36*GiB, TurnkeyOptions{StaticOnly: true, Tier: &tier})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Tier.ModelID != "inkling2-mini" {
		t.Fatalf("plan tier = %q", plan.Tier.ModelID)
	}
	if plan.ContextBudgetTokens != Inkling2MiniMinContextTokens {
		t.Fatalf("context budget = %d", plan.ContextBudgetTokens)
	}
	// Live memory pressure does not shrink a context the engine admits itself.
	pressed, err := ConfigureTurnkeyWithOptions(36*GiB, TurnkeyOptions{AvailableBytes: 8 * GiB, Tier: &tier})
	if err != nil || pressed.ContextBudgetTokens != Inkling2MiniMinContextTokens {
		t.Fatalf("pressured context budget = %d, %v", pressed.ContextBudgetTokens, err)
	}
	if plan.HeadroomRatio < DefaultMinHeadroomRatio {
		t.Fatalf("headroom = %.3f", plan.HeadroomRatio)
	}
	if _, err := ConfigureTurnkeyWithOptions(16*GiB, TurnkeyOptions{StaticOnly: true, Tier: &tier}); err == nil {
		t.Fatal("a tier that cannot fit inside the reserve was not refused")
	}
	// Without an override the ladder is unchanged.
	ladder, err := ConfigureTurnkeyWithOptions(36*GiB, TurnkeyOptions{StaticOnly: true})
	if err != nil || ladder.Tier.Name != "27B" {
		t.Fatalf("ladder tier = %q, %v", ladder.Tier.Name, err)
	}
}
