package gateway

import (
	"errors"
	"testing"
)

// threeTier is the standard fixture: small (4k, cheap, interactive), medium (32k,
// mid, interactive), large (unbounded, premium, batch-only).
func threeTier(strategy RoutingStrategy) *Router {
	cfg := DefaultRouterConfig()
	cfg.Strategy = strategy
	r, err := NewRouter(cfg)
	if err != nil {
		panic(err)
	}
	return r
}

func TestRouter_SizeBased_SmallRequestGetsSmallTier(t *testing.T) {
	r := threeTier(StrategySizeBased)
	d, err := r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "small" {
		t.Fatalf("a 100-token request should route to small, got %q (%s)", d.Tier.Name, d.Reason)
	}
	// Fallbacks present and ascending by capacity (medium before large/unbounded).
	if len(d.Fallbacks) != 2 || d.Fallbacks[0].Name != "medium" || d.Fallbacks[1].Name != "large" {
		t.Fatalf("fallback chain should be [medium large], got %+v", d.Fallbacks)
	}
}

func TestRouter_SizeBased_LargePromptSkipsSmallTiers(t *testing.T) {
	r := threeTier(StrategySizeBased)
	d, err := r.Route(Classify(50000, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "large" {
		t.Fatalf("a 50k-token prompt exceeds small+medium, expected large, got %q", d.Tier.Name)
	}
	if len(d.Fallbacks) != 0 {
		t.Fatalf("no smaller tier has capacity, fallbacks should be empty, got %+v", d.Fallbacks)
	}
}

func TestRouter_Hybrid_MediumPromptGetsMediumTier(t *testing.T) {
	r := threeTier(StrategyHybrid)
	d, err := r.Route(Classify(8000, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "medium" {
		t.Fatalf("8k tokens overflow small(4k), expected medium, got %q", d.Tier.Name)
	}
}

func TestRouter_Complexity_HighFloorsToLargeTier(t *testing.T) {
	r := threeTier(StrategyHybrid)
	// Tiny prompt but high complexity: the floor index (2) forces the large tier.
	d, err := r.Route(Classify(10, LatencyUnknown, ComplexityHigh))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "large" {
		t.Fatalf("high complexity should floor to the 3rd tier, got %q", d.Tier.Name)
	}
}

func TestRouter_Complexity_MediumFloorsAboveSmall(t *testing.T) {
	r := threeTier(StrategyHybrid)
	d, err := r.Route(Classify(10, LatencyUnknown, ComplexityMedium))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "medium" {
		t.Fatalf("medium complexity should floor above small, got %q", d.Tier.Name)
	}
}

func TestRouter_Latency_InteractiveAvoidsBatchOnlyTier(t *testing.T) {
	r := threeTier(StrategyHybrid)
	// A large prompt would want "large", but large is batch-only and the request is
	// interactive -> no candidate -> ErrNoTier.
	_, err := r.Route(Classify(50000, LatencyInteractive, ComplexityLow))
	if !errors.Is(err, ErrNoTier) {
		t.Fatalf("interactive request that only fits a batch-only tier should be ErrNoTier, got %v", err)
	}
}

func TestRouter_Latency_BatchPrefersLargestTier(t *testing.T) {
	r := threeTier(StrategyLatencyBased)
	d, err := r.Route(Classify(100, LatencyBatch, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "large" {
		t.Fatalf("batch latency should prefer the largest tier, got %q", d.Tier.Name)
	}
}

func TestRouter_Latency_InteractivePrefersSmallestTier(t *testing.T) {
	r := threeTier(StrategyLatencyBased)
	d, err := r.Route(Classify(100, LatencyInteractive, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "small" {
		t.Fatalf("interactive latency should prefer the smallest tier, got %q", d.Tier.Name)
	}
}

func TestRouter_Cost_PicksCheapestThatFits(t *testing.T) {
	// Custom config where the cheapest tier is NOT the smallest, to prove cost wins.
	cfg := RouterConfig{
		Strategy: StrategyCostBased,
		Tiers: []Tier{
			{Name: "a", Model: "a", MaxPromptTokens: 10000, CostPerMTok: 8, Interactive: true},
			{Name: "b", Model: "b", MaxPromptTokens: 10000, CostPerMTok: 3, Interactive: true},
			{Name: "c", Model: "c", MaxPromptTokens: 10000, CostPerMTok: 5, Interactive: true},
		},
	}
	r, err := NewRouter(cfg)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	d, err := r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "b" {
		t.Fatalf("cost strategy should pick the cheapest fitting tier (b), got %q", d.Tier.Name)
	}
}

func TestRouter_Health_FallsBackToNextTier(t *testing.T) {
	r := threeTier(StrategySizeBased)
	if !r.Healthy("small") {
		t.Fatal("small should start healthy")
	}
	r.SetHealth("small", false)
	if r.Healthy("small") {
		t.Fatal("small should be marked unhealthy")
	}
	// Small is down, so a small request now lands on medium.
	d, err := r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "medium" {
		t.Fatalf("with small down, a small request should fall back to medium, got %q", d.Tier.Name)
	}
	// Recover small and confirm it is chosen again.
	r.SetHealth("small", true)
	d, err = r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "small" {
		t.Fatalf("recovered small should be chosen again, got %q", d.Tier.Name)
	}
}

func TestRouter_Health_AllDownIsErrNoTier(t *testing.T) {
	r := threeTier(StrategySizeBased)
	for _, n := range []string{"small", "medium", "large"} {
		r.SetHealth(n, false)
	}
	_, err := r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if !errors.Is(err, ErrNoTier) {
		t.Fatalf("all tiers down should be ErrNoTier, got %v", err)
	}
}

func TestRouter_SetHealth_UnknownTierIsNoop(t *testing.T) {
	r := threeTier(StrategySizeBased)
	r.SetHealth("does-not-exist", false) // must not panic or affect real tiers
	d, err := r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "small" {
		t.Fatalf("unknown tier health change should not affect routing, got %q", d.Tier.Name)
	}
}

func TestRouterConfig_Validate(t *testing.T) {
	if err := DefaultRouterConfig().Validate(); err != nil {
		t.Fatalf("default config should validate, got %v", err)
	}
	cases := []struct {
		name string
		cfg  RouterConfig
	}{
		{"no tiers", RouterConfig{Strategy: StrategySizeBased}},
		{"bad strategy", RouterConfig{Strategy: "bogus", Tiers: []Tier{{Name: "x"}}}},
		{"empty name", RouterConfig{Tiers: []Tier{{Name: ""}}}},
		{"dup name", RouterConfig{Tiers: []Tier{{Name: "x"}, {Name: "x"}}}},
		{"negative capacity", RouterConfig{Tiers: []Tier{{Name: "x", MaxPromptTokens: -1}}}},
	}
	for _, c := range cases {
		if err := c.cfg.Validate(); err == nil {
			t.Errorf("%s: expected validation error, got nil", c.name)
		}
	}
}

func TestNewRouter_RejectsInvalidConfig(t *testing.T) {
	if _, err := NewRouter(RouterConfig{}); err == nil {
		t.Fatal("NewRouter should reject an empty config")
	}
}

func TestNewRouter_DefaultsEmptyStrategyToSize(t *testing.T) {
	cfg := DefaultRouterConfig()
	cfg.Strategy = ""
	r, err := NewRouter(cfg)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	d, err := r.Route(Classify(100, LatencyUnknown, ComplexityLow))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Strategy != StrategySizeBased {
		t.Fatalf("empty strategy should default to size-based, got %q", d.Strategy)
	}
}

func TestClassify_ClampsNegativeTokens(t *testing.T) {
	rc := Classify(-5, LatencyBatch, ComplexityHigh)
	if rc.PromptTokens != 0 {
		t.Fatalf("negative tokens should clamp to 0, got %d", rc.PromptTokens)
	}
	if rc.Latency != LatencyBatch || rc.Complexity != ComplexityHigh {
		t.Fatalf("Classify should preserve latency/complexity, got %+v", rc)
	}
}

func TestComplexity_FloorIndexCapsAtLastTier(t *testing.T) {
	// High complexity (index 2) against a 2-tier config caps at the last index (1).
	if got := ComplexityHigh.floorIndex(2); got != 1 {
		t.Fatalf("floorIndex should cap at nTiers-1, got %d", got)
	}
	if got := ComplexityLow.floorIndex(3); got != 0 {
		t.Fatalf("low complexity floor should be 0, got %d", got)
	}
}
