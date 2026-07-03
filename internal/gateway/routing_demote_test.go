package gateway

import (
	"testing"
	"time"
)

// A persistent entitlement-403 for a model must route AROUND that model via the existing fallback
// chain — the auto-model-switch response the 2026-07-03 gem8 storm's permanent-403 population wants
// — WITHOUT a code change to Route: DemoteModel is the routing-side counterpart of the agent's
// transient-403 recovery arm. Here the small tier is entitlement-denied, so a small request that
// would normally win must fall through to the next permitted rung.
func TestRouter_DemoteModel_RoutesAroundDeniedTier(t *testing.T) {
	r := threeTier(StrategyHybrid)
	base := time.Unix(1_700_000_000, 0)
	r.now = func() time.Time { return base }

	// A tiny request normally wins "small".
	d, err := r.Route(Classify(100, LatencyInteractive, ComplexityLow))
	if err != nil || d.Tier.Name != "small" {
		t.Fatalf("baseline: a 100-token request should win small, got %q err=%v", d.Tier.Name, err)
	}

	// Small's model is entitlement-denied for 30s: the same request must now route to medium.
	r.DemoteModel("small", 30*time.Second)
	if r.Healthy("small") {
		t.Fatal("small should read unhealthy while its entitlement cooldown is active")
	}
	d, err = r.Route(Classify(100, LatencyInteractive, ComplexityLow))
	if err != nil {
		t.Fatalf("a denied small tier should fall through, not fail: %v", err)
	}
	if d.Tier.Name != "medium" {
		t.Fatalf("a demoted small tier should route to medium, got %q", d.Tier.Name)
	}
}

// The demotion SELF-EXPIRES: an entitlement that comes back (plan change, lifted abuse gate) is
// picked up automatically with no operator un-demote — the distinction from SetHealth's hard flag,
// and what keeps a transient-but-slow 403 from stranding a tier forever.
func TestRouter_DemoteModel_SelfRecoversAfterCooldown(t *testing.T) {
	r := threeTier(StrategyHybrid)
	now := time.Unix(1_700_000_000, 0)
	r.now = func() time.Time { return now }

	r.DemoteModel("small", 30*time.Second)
	if r.Healthy("small") {
		t.Fatal("small should be demoted immediately after DemoteModel")
	}

	// Just before expiry: still demoted.
	now = now.Add(29 * time.Second)
	if r.Healthy("small") {
		t.Fatal("small should still be demoted 1s before the cooldown expires")
	}

	// Past expiry: healthy again, and it wins its natural request with no un-demote call.
	now = now.Add(2 * time.Second)
	if !r.Healthy("small") {
		t.Fatal("small should self-recover once the cooldown elapses")
	}
	d, err := r.Route(Classify(100, LatencyInteractive, ComplexityLow))
	if err != nil || d.Tier.Name != "small" {
		t.Fatalf("recovered small should win its natural request again, got %q err=%v", d.Tier.Name, err)
	}
}

// A fresh 403 during an active cooldown EXTENDS the demotion (never shortens it), so a sustained
// denial keeps the tier out rather than flapping back mid-storm.
func TestRouter_DemoteModel_ExtendsNeverShortens(t *testing.T) {
	r := threeTier(StrategyHybrid)
	now := time.Unix(1_700_000_000, 0)
	r.now = func() time.Time { return now }

	r.DemoteModel("small", 60*time.Second)
	// A shorter re-demote must NOT pull the expiry in.
	r.DemoteModel("small", 5*time.Second)
	now = now.Add(10 * time.Second)
	if r.Healthy("small") {
		t.Fatal("a shorter re-demote must not shorten an active longer cooldown")
	}
	// A longer re-demote DOES push it out.
	r.DemoteModel("small", 120*time.Second)
	now = now.Add(70 * time.Second) // past the original 60s window, inside the extended one
	if r.Healthy("small") {
		t.Fatal("a longer re-demote should extend the cooldown")
	}
}

// Guard the no-op contract: a non-positive cooldown or an unknown model changes nothing, so a
// caller cannot accidentally demote every tier or wedge one with a zero window.
func TestRouter_DemoteModel_NoOpGuards(t *testing.T) {
	r := threeTier(StrategyHybrid)
	r.DemoteModel("small", 0)            // zero window: no-op
	r.DemoteModel("", time.Second)       // empty model: no-op
	r.DemoteModel("nonesuch", time.Hour) // unknown model: matches no tier
	for _, name := range []string{"small", "medium", "large"} {
		if !r.Healthy(name) {
			t.Fatalf("no-op DemoteModel must leave %q healthy", name)
		}
	}
}
