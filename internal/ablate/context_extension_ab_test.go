package ablate

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// opusPricing is shared with compaction_ab_test.go / anchor_ab_test.go (same package): the
// default guarded-Claude base price the context-extension classifier is exercised at.

// perTokInput is the base input dollars-per-token the value math prices a cold shed token at.
var perTokInput = gateway.ClaudeOpus48InputPerMTokUSD / 1_000_000

// The window test: a fire whose pre-fire resident met or exceeded the hard window is
// limit-avoiding (it kept the session under the cap); one below the window is cache-shaving.
func TestContextExtensionFire_ClassifiesByWindow(t *testing.T) {
	over := ContextExtensionFire{ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 205_000, ShedTokens: 10_000}
	at := ContextExtensionFire{ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 200_000, ShedTokens: 10_000}
	under := ContextExtensionFire{ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 120_000, ShedTokens: 10_000}
	if !over.LimitAvoiding() {
		t.Errorf("resident over window should be limit-avoiding")
	}
	if !at.LimitAvoiding() {
		t.Errorf("resident exactly at window should be limit-avoiding (next token overflows)")
	}
	if under.LimitAvoiding() {
		t.Errorf("resident under window should be cache-shaving, not limit-avoiding")
	}
}

// A fully-warm shed is near-zero net (priced at the 0.1x read marginal); a fully-cold shed carries
// full input value. This is the SAME warm/cold blend the gateway split values a shed token at.
func TestContextExtensionFire_CacheShavingIsNearZeroNet(t *testing.T) {
	cold := ContextExtensionFire{ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 205_000, ShedTokens: 10_000, WarmShedTokens: 0}
	warm := ContextExtensionFire{ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 120_000, ShedTokens: 10_000, WarmShedTokens: 10_000}

	wantCold := 10_000 * perTokInput
	if got := cold.ShedValueUSD(opusPricing); math.Abs(got-wantCold) > 1e-12 {
		t.Errorf("cold shed value = %v, want %v", got, wantCold)
	}
	// Fully warm: value collapses to the 0.1x read marginal — near-zero relative to the cold shed.
	wantWarm := 10_000 * gateway.CacheReadMultiplier * perTokInput
	if got := warm.ShedValueUSD(opusPricing); math.Abs(got-wantWarm) > 1e-12 {
		t.Errorf("warm shed value = %v, want %v", got, wantWarm)
	}
	if warm.ShedValueUSD(opusPricing) >= cold.ShedValueUSD(opusPricing) {
		t.Errorf("a fully-warm shed must be worth strictly less than a fully-cold shed of the same size")
	}
}

// The acceptance: the report splits fires into the two populations and attributes the real value to
// the limit-avoiding fires, kept separate from the near-zero-net cache-shaving fires.
func TestClassifyContextExtensionFires_SeparatesPopulationsAndAttributesValue(t *testing.T) {
	fires := []ContextExtensionFire{
		// Limit-avoiding: fired at/over the window, shedding cold (uncached) tokens — real value.
		{SessionID: "a", ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 205_000, ShedTokens: 10_000, WarmShedTokens: 0},
		{SessionID: "b", ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 200_000, ShedTokens: 6_000, WarmShedTokens: 0},
		// Cache-shaving: fired below the window, shedding tokens already served warm — near-zero net.
		{SessionID: "c", ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 120_000, ShedTokens: 8_000, WarmShedTokens: 8_000},
		{SessionID: "d", ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 90_000, ShedTokens: 4_000, WarmShedTokens: 4_000},
	}

	rep, err := ClassifyContextExtensionFires(fires, opusPricing, "test:opus")
	if err != nil {
		t.Fatalf("ClassifyContextExtensionFires: %v", err)
	}

	// Populations are split by the window test.
	if rep.LimitAvoiding.N != 2 || rep.CacheShaving.N != 2 {
		t.Fatalf("population split = %d limit-avoiding / %d cache-shaving, want 2/2", rep.LimitAvoiding.N, rep.CacheShaving.N)
	}
	if rep.LimitAvoiding.Label != FireLimitAvoiding || rep.CacheShaving.Label != FireCacheShaving {
		t.Errorf("population labels = %q/%q, want %q/%q", rep.LimitAvoiding.Label, rep.CacheShaving.Label, FireLimitAvoiding, FireCacheShaving)
	}
	if rep.LimitAvoiding.ShedTokens != 16_000 || rep.CacheShaving.ShedTokens != 12_000 {
		t.Errorf("shed totals = %d/%d, want 16000/12000", rep.LimitAvoiding.ShedTokens, rep.CacheShaving.ShedTokens)
	}
	if got, want := rep.LimitAvoiding.Sessions, []string{"a", "b"}; strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("limit-avoiding sessions = %v, want %v", got, want)
	}

	// Value: limit-avoiding = (10000+6000) cold tokens at full input; cache-shaving = (8000+4000)
	// warm tokens at the 0.1x read marginal — near-zero relative to the real value.
	wantReal := 16_000 * perTokInput
	wantNearZero := 12_000 * gateway.CacheReadMultiplier * perTokInput
	if math.Abs(rep.RealValueUSD()-wantReal) > 1e-12 {
		t.Errorf("RealValueUSD = %v, want %v", rep.RealValueUSD(), wantReal)
	}
	if math.Abs(rep.NearZeroNetUSD()-wantNearZero) > 1e-12 {
		t.Errorf("NearZeroNetUSD = %v, want %v", rep.NearZeroNetUSD(), wantNearZero)
	}
	// The whole point: the real value is attributed to the limit-avoiding fires.
	if !rep.AttributesRealValueToLimitAvoiding() {
		t.Errorf("expected real value to concentrate in limit-avoiding fires: real=%v near-zero=%v", rep.RealValueUSD(), rep.NearZeroNetUSD())
	}
	if rep.RealValueUSD() <= rep.NearZeroNetUSD() {
		t.Errorf("limit-avoiding value (%v) should dominate cache-shaving value (%v)", rep.RealValueUSD(), rep.NearZeroNetUSD())
	}

	// The rendered one-liner names both populations, the real value, and the fire count.
	line := rep.SweepRow()
	for _, want := range []string{"limit-avoiding", "cache-shaving", "real value", "near-zero net", "N=4"} {
		if !strings.Contains(line, want) {
			t.Errorf("SweepRow() = %q, missing %q", line, want)
		}
	}
	// The JSON artifact carries both populations separately.
	js := string(rep.JSON())
	for _, want := range []string{`"limit_avoiding"`, `"cache_shaving"`, `"value_usd"`} {
		if !strings.Contains(js, want) {
			t.Errorf("JSON() missing %q; got %s", want, js)
		}
	}
}

// Fail closed: no fires is an error (never a fabricated $0 split); a fire with no window cannot be
// classified; a fire that shed nothing is not a fire.
func TestClassifyContextExtensionFires_FailsClosed(t *testing.T) {
	if _, err := ClassifyContextExtensionFires(nil, opusPricing, ""); err == nil {
		t.Errorf("empty fire set should error, got nil")
	}
	noWindow := []ContextExtensionFire{{SessionID: "w", ResidentTokensBeforeFire: 10, ShedTokens: 5}}
	if _, err := ClassifyContextExtensionFires(noWindow, opusPricing, ""); err == nil {
		t.Errorf("fire with no context window should error, got nil")
	}
	noShed := []ContextExtensionFire{{SessionID: "s", ContextWindowTokens: 200_000, ResidentTokensBeforeFire: 205_000, ShedTokens: 0}}
	if _, err := ClassifyContextExtensionFires(noShed, opusPricing, ""); err == nil {
		t.Errorf("fire that shed no tokens should error, got nil")
	}
}
