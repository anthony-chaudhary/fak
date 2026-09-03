package gateway

import (
	"errors"
	"net/http"
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

func TestRouter_MaxCost_PicksCheapestWithinBudget(t *testing.T) {
	// small(1) overflows at 8000 tokens; medium(5) fits and is under a $10 ceiling;
	// large(20) is over the ceiling -> medium wins and large is excluded entirely.
	r := threeTier(StrategyHybrid)
	rc := Classify(8000, LatencyUnknown, ComplexityLow)
	rc.MaxCostPerMTok = 10
	d, err := r.Route(rc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "medium" {
		t.Fatalf("an $10 ceiling should admit medium(5) but not large(20), got %q (%s)", d.Tier.Name, d.Reason)
	}
	// large(20) is over the ceiling, so it must not appear in the fallback chain either.
	for _, f := range d.Fallbacks {
		if f.Name == "large" {
			t.Fatalf("large(20) is over the $10 ceiling and must be excluded from fallbacks, got %+v", d.Fallbacks)
		}
	}
}

func TestRouter_MaxCost_AllOverCeilingIsErrNoTier(t *testing.T) {
	// A ceiling below the cheapest tier's cost refuses fail-closed rather than
	// silently serving from a pricier tier — fak's max_price analogue.
	r := threeTier(StrategyHybrid)
	rc := Classify(100, LatencyUnknown, ComplexityLow)
	rc.MaxCostPerMTok = 0.5 // below small(1)
	if _, err := r.Route(rc); !errors.Is(err, ErrNoTier) {
		t.Fatalf("a ceiling under every tier's cost should be ErrNoTier, got %v", err)
	}
}

func TestRouter_MaxCost_CapacityAndCeilingCompound(t *testing.T) {
	// 8000 tokens overflows small(4k); a $4 ceiling excludes medium(5) and large(20).
	// Capacity and ceiling compound to leave no candidate -> ErrNoTier.
	r := threeTier(StrategyHybrid)
	rc := Classify(8000, LatencyUnknown, ComplexityLow)
	rc.MaxCostPerMTok = 4
	if _, err := r.Route(rc); !errors.Is(err, ErrNoTier) {
		t.Fatalf("small over-capacity + medium/large over-ceiling should be ErrNoTier, got %v", err)
	}
}

func TestRouter_MaxCost_ZeroIsUnbounded(t *testing.T) {
	// The default (0) imposes no ceiling: routing is identical to no max_price.
	r := threeTier(StrategyHybrid)
	rc := Classify(100, LatencyUnknown, ComplexityLow)
	rc.MaxCostPerMTok = 0
	d, err := r.Route(rc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "small" {
		t.Fatalf("a 0 ceiling should be unbounded and route to small, got %q", d.Tier.Name)
	}
}

func TestRouter_MaxCost_CostStrategyRespectsCeiling(t *testing.T) {
	// Cost strategy would pick the cheapest tier (b=3); a $4 ceiling that still admits
	// b confirms the ceiling and the strategy agree, and a $2 ceiling that excludes ALL
	// refuses rather than picking the next-cheapest above budget.
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
	rc := Classify(100, LatencyUnknown, ComplexityLow)
	rc.MaxCostPerMTok = 4
	d, err := r.Route(rc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Tier.Name != "b" {
		t.Fatalf("a $4 ceiling should still pick the cheapest fitting tier b(3), got %q", d.Tier.Name)
	}
	rc.MaxCostPerMTok = 2 // below every tier
	if _, err := r.Route(rc); !errors.Is(err, ErrNoTier) {
		t.Fatalf("a $2 ceiling below every tier should be ErrNoTier, got %v", err)
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

func TestRequestClass_SessionID(t *testing.T) {
	rc := RequestClass{
		PromptTokens: 100,
		Latency:      LatencyInteractive,
		Complexity:   ComplexityLow,
		AffinityKey:  "session-42",
	}
	if rc.AffinityKey != "session-42" {
		t.Fatalf("AffinityKey = %q, want session-42", rc.AffinityKey)
	}
}

func TestExtractAffinityKey(t *testing.T) {
	r, _ := http.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set("X-Session-ID", "header-sess")

	if got := extractAffinityKey(r, ""); got != "header-sess" {
		t.Fatalf("extractAffinityKey from header = %q, want header-sess", got)
	}
	if got := extractAffinityKey(r, "explicit-sess"); got != "explicit-sess" {
		t.Fatalf("extractAffinityKey with explicit = %q, want explicit-sess", got)
	}
	if got := extractAffinityKey(nil, "explicit-only"); got != "explicit-only" {
		t.Fatalf("extractAffinityKey with nil request = %q, want explicit-only", got)
	}
	if got := extractAffinityKey(nil, ""); got != "" {
		t.Fatalf("extractAffinityKey with empty = %q, want empty", got)
	}
}
