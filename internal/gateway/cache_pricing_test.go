package gateway

import (
	"math"
	"testing"
)

const eps = 1e-12

func approx(a, b float64) bool { return math.Abs(a-b) <= eps }

// TestCacheTTLWriteMultiplier pins the published write multipliers and the
// conservative 5m fallback for an unset/unknown TTL.
func TestCacheTTLWriteMultiplier(t *testing.T) {
	cases := []struct {
		ttl  CacheTTL
		want float64
	}{
		{CacheTTL5m, CacheWrite5mMultiplier},
		{CacheTTL1h, CacheWrite1hMultiplier},
		{"", CacheWrite5mMultiplier},    // unset → default (cheaper) tier
		{"30m", CacheWrite5mMultiplier}, // unknown → default, never free
	}
	for _, c := range cases {
		if got := c.ttl.WriteMultiplier(); !approx(got, c.want) {
			t.Errorf("CacheTTL(%q).WriteMultiplier() = %v, want %v", c.ttl, got, c.want)
		}
	}
	// Guard the constants themselves so a stray edit to the economics is caught.
	if CacheReadMultiplier != 0.1 || CacheWrite5mMultiplier != 1.25 || CacheWrite1hMultiplier != 2.0 {
		t.Fatalf("cache multipliers drifted: read=%v write5m=%v write1h=%v",
			CacheReadMultiplier, CacheWrite5mMultiplier, CacheWrite1hMultiplier)
	}
}

// TestCostUSD prices each axis at its correct multiplier against a base input/output
// price. Opus-4.8-shaped pricing ($5/$25 per MTok) keeps the numbers legible.
func TestCostUSD(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	u := CacheUsage{
		InputTokens:         1_000_000, // 1.0× → $5
		CacheReadTokens:     1_000_000, // 0.1× → $0.50
		CacheCreationTokens: 1_000_000, // 1.25× (5m default) → $6.25
		OutputTokens:        1_000_000, // output → $25
		WriteTTL:            CacheTTL5m,
	}
	// $5 + $0.50 + $6.25 + $25 = $36.75
	if got := p.CostUSD(u); !approx(got, 36.75) {
		t.Errorf("CostUSD = %v, want 36.75", got)
	}
	// Same turn, 1h write tier: creation now 2.0× → $10, total $40.50.
	u.WriteTTL = CacheTTL1h
	if got := p.CostUSD(u); !approx(got, 40.50) {
		t.Errorf("CostUSD(1h) = %v, want 40.50", got)
	}
}

// TestUncachedCostUSD checks the counterfactual: every prompt token at full input
// price, output unchanged.
func TestUncachedCostUSD(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	u := CacheUsage{InputTokens: 1_000_000, CacheReadTokens: 1_000_000, CacheCreationTokens: 1_000_000, OutputTokens: 1_000_000}
	// 3M prompt tokens × $5/MTok = $15, + $25 output = $40.
	if got := p.UncachedCostUSD(u); !approx(got, 40) {
		t.Errorf("UncachedCostUSD = %v, want 40", got)
	}
}

// TestSavingsReadOnlyIsPositive: a turn that only READS the cache saves 0.9× of base
// input on every cached token — the steady-state win.
func TestSavingsReadOnlyIsPositive(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	u := CacheUsage{InputTokens: 0, CacheReadTokens: 1_000_000, OutputTokens: 100}
	// 1M read × $5/MTok × (1 − 0.1) = $4.50 saved.
	if got := p.SavingsUSD(u); !approx(got, 4.50) {
		t.Errorf("SavingsUSD(read-only) = %v, want 4.50", got)
	}
}

// TestSavingsColdWriteIsNegative: a cold miss that only WRITES the cache costs MORE
// than uncached (1.25× vs 1.0×), so the model must report a negative saving rather
// than pretend the write was free.
func TestSavingsColdWriteIsNegative(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	u := CacheUsage{CacheCreationTokens: 1_000_000, WriteTTL: CacheTTL5m, OutputTokens: 100}
	// 1M write × $5/MTok × (1 − 1.25) = −$1.25.
	if got := p.SavingsUSD(u); !approx(got, -1.25) {
		t.Errorf("SavingsUSD(cold write) = %v, want -1.25", got)
	}
	// 1h tier is a steeper premium: × (1 − 2.0) = −$5.
	u.WriteTTL = CacheTTL1h
	if got := p.SavingsUSD(u); !approx(got, -5) {
		t.Errorf("SavingsUSD(cold write, 1h) = %v, want -5", got)
	}
}

// TestSavingsEqualsDifference ties the three together: SavingsUSD must be exactly
// UncachedCostUSD − CostUSD, and output tokens must cancel out of the difference.
func TestSavingsEqualsDifference(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15} // Sonnet-4.6-shaped
	u := CacheUsage{InputTokens: 400, CacheReadTokens: 9_000, CacheCreationTokens: 600, OutputTokens: 1_234, WriteTTL: CacheTTL1h}
	if got, want := p.SavingsUSD(u), p.UncachedCostUSD(u)-p.CostUSD(u); !approx(got, want) {
		t.Errorf("SavingsUSD = %v, want UncachedCostUSD−CostUSD = %v", got, want)
	}
	// Doubling output tokens changes neither cost difference (saving is prompt-only).
	u2 := u
	u2.OutputTokens *= 2
	if !approx(p.SavingsUSD(u), p.SavingsUSD(u2)) {
		t.Errorf("SavingsUSD changed with output tokens: %v vs %v", p.SavingsUSD(u), p.SavingsUSD(u2))
	}
}

// TestProviderCacheSavingsUSD checks the summary integration: the observed
// cumulative cache_read tokens valued at 0.9× of base input price.
func TestProviderCacheSavingsUSD(t *testing.T) {
	s := AdjudicationSummary{CachedPromptTokens: 2_000_000}
	// 2M × $5/MTok × 0.9 = $9.
	if got := s.ProviderCacheSavingsUSD(5); !approx(got, 9) {
		t.Errorf("ProviderCacheSavingsUSD = %v, want 9", got)
	}
	// No observed reuse → no saving (and no panic on the zero summary).
	if got := (AdjudicationSummary{}).ProviderCacheSavingsUSD(5); got != 0 {
		t.Errorf("ProviderCacheSavingsUSD(zero) = %v, want 0", got)
	}
}

// TestCostUSDZeroUsageIsZeroCost pins the zero-token edge: no activity on any
// axis must price to exactly zero, regardless of the base per-MTok price.
func TestCostUSDZeroUsageIsZeroCost(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	var u CacheUsage
	if got := p.CostUSD(u); got != 0 {
		t.Errorf("CostUSD(zero usage) = %v, want 0", got)
	}
	if got := p.UncachedCostUSD(u); got != 0 {
		t.Errorf("UncachedCostUSD(zero usage) = %v, want 0", got)
	}
	if got := p.SavingsUSD(u); got != 0 {
		t.Errorf("SavingsUSD(zero usage) = %v, want 0", got)
	}
}

// TestCostUSDNegativeTokensMirrorsPositive pins the CURRENT defined behavior for
// a garbage/negative token count: CacheUsage's fields are plain int (not uint),
// so a negative count is constructible. The model applies no clamp or rejection
// -- it is linear in each axis, so a negative count contributes the exact
// negation of what the same positive magnitude would. This is "defined" (no
// panic, no NaN) rather than "meaningful" for a real token count; pinning it
// here means a future decision to reject/clamp negative input is a deliberate
// change, not a silent one.
func TestCostUSDNegativeTokensMirrorsPositive(t *testing.T) {
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	cases := []CacheUsage{
		{InputTokens: 1_000_000},
		{CacheReadTokens: 1_000_000},
		{CacheCreationTokens: 1_000_000, WriteTTL: CacheTTL1h},
	}
	for _, pos := range cases {
		neg := pos
		neg.InputTokens, neg.CacheReadTokens, neg.CacheCreationTokens =
			-neg.InputTokens, -neg.CacheReadTokens, -neg.CacheCreationTokens
		if got, want := p.CostUSD(neg), -p.CostUSD(pos); !approx(got, want) {
			t.Errorf("CostUSD(%+v) = %v, want %v (negation of the positive case)", neg, got, want)
		}
	}
}

// TestCostUSDLargeCountsStayWithinRelativeError guards against a naive change
// (e.g. switching to a lower-precision accumulator) silently blowing up the
// economics at scale. A trillion-token count is still exactly representable as
// a float64 (well under the 2^53 exact-integer ceiling), so the computed cost
// must match the hand-derived value to a tight relative tolerance.
func TestCostUSDLargeCountsStayWithinRelativeError(t *testing.T) {
	const tokens = 1_000_000_000_000 // 1e12, three orders below the float64 exact-integer ceiling
	p := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	u := CacheUsage{InputTokens: tokens, CacheReadTokens: tokens, CacheCreationTokens: tokens, OutputTokens: tokens, WriteTTL: CacheTTL5m}
	want := float64(tokens) * (5.0/1e6*1 + 5.0/1e6*CacheReadMultiplier + 5.0/1e6*CacheWrite5mMultiplier + 25.0/1e6)
	got := p.CostUSD(u)
	if relErr := math.Abs(got-want) / want; relErr > 1e-9 {
		t.Errorf("CostUSD(1e12 tokens) = %v, want %v (relative error %.3e > 1e-9)", got, want, relErr)
	}
}

// TestProviderCacheSavingsUSDNearUint64MaxStaysFinite guards the uint64->float64
// conversion path (AdjudicationSummary's counters are unsigned, so they can't go
// negative, but they can get close to the type's max on a very long-lived
// process): the result must stay a finite, non-NaN number rather than
// overflowing or wrapping silently.
func TestProviderCacheSavingsUSDNearUint64MaxStaysFinite(t *testing.T) {
	s := AdjudicationSummary{CachedPromptTokens: ^uint64(0) - 1}
	got := s.ProviderCacheSavingsUSD(5)
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("ProviderCacheSavingsUSD(near-max uint64) = %v, want a finite number", got)
	}
	if got <= 0 {
		t.Fatalf("ProviderCacheSavingsUSD(near-max uint64) = %v, want > 0", got)
	}
}

// TestDefaultCachePricingResolvesKnownModels pins the built-in price table: the
// guarded-Claude Opus 4.8 default ($5/$25), the Fable 5 frontier tier ($10/$50,
// Anthropic's published rate — the entry that makes a fable worker's Track-2 row
// dollar-real instead of dollar-blind, task #3), and the fail-closed miss for a
// provider/context the table does not know (which the caller stamps dollar-blind).
// Context normalization (last path segment, lowercased) is exercised so the model
// id resolves whether it arrives bare or as a session-path tail.
func TestDefaultCachePricingResolvesKnownModels(t *testing.T) {
	cases := []struct {
		name            string
		provider, ctx   string
		wantOK          bool
		wantIn, wantOut float64
		wantSource      string
	}{
		{"opus bare", "anthropic", "claude-opus-4-8", true, ClaudeOpus48InputPerMTokUSD, ClaudeOpus48OutputPerMTokUSD, CachePricingSourceAnthropicClaudeOpus48},
		{"opus via claude binary", "anthropic", "claude", true, ClaudeOpus48InputPerMTokUSD, ClaudeOpus48OutputPerMTokUSD, CachePricingSourceAnthropicClaudeOpus48},
		{"fable bare", "anthropic", "claude-fable-5", true, ClaudeFable5InputPerMTokUSD, ClaudeFable5OutputPerMTokUSD, CachePricingSourceAnthropicClaudeFable5},
		{"fable via session path tail", "anthropic", "/home/run/sessions/claude-fable-5", true, ClaudeFable5InputPerMTokUSD, ClaudeFable5OutputPerMTokUSD, CachePricingSourceAnthropicClaudeFable5},
		{"fable via claude provider alias", "claude", "claude-fable-5", true, ClaudeFable5InputPerMTokUSD, ClaudeFable5OutputPerMTokUSD, CachePricingSourceAnthropicClaudeFable5},
		{"unknown model is a fail-closed miss", "anthropic", "some-unlisted-model", false, 0, 0, ""},
		{"unknown provider is a fail-closed miss", "openai", "codex", false, 0, 0, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p, source, ok := DefaultCachePricing(c.provider, c.ctx)
			if ok != c.wantOK {
				t.Fatalf("DefaultCachePricing(%q,%q) ok = %v, want %v", c.provider, c.ctx, ok, c.wantOK)
			}
			if !c.wantOK {
				return
			}
			if !approx(p.InputPerMTokUSD, c.wantIn) || !approx(p.OutputPerMTokUSD, c.wantOut) {
				t.Errorf("DefaultCachePricing(%q,%q) = {in:%v out:%v}, want {in:%v out:%v}",
					c.provider, c.ctx, p.InputPerMTokUSD, p.OutputPerMTokUSD, c.wantIn, c.wantOut)
			}
			if source != c.wantSource {
				t.Errorf("DefaultCachePricing(%q,%q) source = %q, want %q", c.provider, c.ctx, source, c.wantSource)
			}
		})
	}
	// Fable is a frontier tier: pricing it must exceed the Opus default on both axes,
	// so a fable worker's savings are never silently understated as Opus-priced.
	if ClaudeFable5InputPerMTokUSD <= ClaudeOpus48InputPerMTokUSD || ClaudeFable5OutputPerMTokUSD <= ClaudeOpus48OutputPerMTokUSD {
		t.Fatalf("fable pricing (%v/%v) must exceed opus (%v/%v)",
			ClaudeFable5InputPerMTokUSD, ClaudeFable5OutputPerMTokUSD, ClaudeOpus48InputPerMTokUSD, ClaudeOpus48OutputPerMTokUSD)
	}
}

func TestMechanismSavingsSumsOwnersAndMechanisms(t *testing.T) {
	s := AdjudicationSummary{
		CachedPromptTokens:   1000, // provider read rebate = 900 token-equiv
		CacheCreationTokens:  200,  // provider write premium = -50 token-equiv
		CompactionShedTokens: 300,
		KVPrefixReusedTokens: 400,
	}
	m := s.MechanismSavings()
	m.FakVDSOAvoidedCalls = 7

	if !approx(m.ProviderPromptCacheReadTokenEquiv, 900) {
		t.Fatalf("provider read rebate = %v, want 900", m.ProviderPromptCacheReadTokenEquiv)
	}
	if !approx(m.ProviderPromptCacheWritePremiumTokenEquiv, -50) {
		t.Fatalf("provider write premium = %v, want -50", m.ProviderPromptCacheWritePremiumTokenEquiv)
	}
	if !approx(m.ProviderTokenEquiv(), 850) {
		t.Fatalf("provider net = %v, want 850", m.ProviderTokenEquiv())
	}
	if !approx(m.FakTokenEquiv(), 700) {
		t.Fatalf("fak token-equiv = %v, want compaction+kv = 700", m.FakTokenEquiv())
	}
	if !approx(m.TotalTokenEquiv(), 1550) {
		t.Fatalf("total token-equiv = %v, want provider+fak = 1550", m.TotalTokenEquiv())
	}
	if m.FakVDSOAvoidedCalls != 7 {
		t.Fatalf("vdso avoided calls = %d, want 7", m.FakVDSOAvoidedCalls)
	}
}

// TestFakTokenEquivWarmShedPricedAtMarginal locks the #2794/#2798 live-path fix: when a
// session's compaction fires observed a provider cache_read (warm), FakTokenEquiv values the
// shed at the 0.1x cache-read marginal (CacheReadMultiplier), not full input — the same
// marginal the Track-2 report and the fire gate price a shed token at. Prefix reuse (local,
// fak-realized) keeps its full 1.0x marginal. Booking warm shed at 1.0x was the ~10x
// over-credit this fix removes.
func TestFakTokenEquivWarmShedPricedAtMarginal(t *testing.T) {
	warm := AdjudicationSummary{
		CompactionShedTokens:      1000,
		CompactionCacheReadTokens: 5000, // >0 => at least one warm fire this session
		KVPrefixReusedTokens:      400,
	}.MechanismSavings()
	if want := 1000*CacheReadMultiplier + 400; !approx(warm.FakTokenEquiv(), want) {
		t.Fatalf("warm fak token-equiv = %v, want shed*0.1 + kv = %v", warm.FakTokenEquiv(), want)
	}
	// The SAME shed with no observed cache_read (cold) keeps the full 1.0x input basis.
	cold := AdjudicationSummary{
		CompactionShedTokens: 1000,
		KVPrefixReusedTokens: 400,
	}.MechanismSavings()
	if !approx(cold.FakTokenEquiv(), 1400) {
		t.Fatalf("cold fak token-equiv = %v, want shed*1.0 + kv = 1400", cold.FakTokenEquiv())
	}
	// The warm discount only ever REMOVES phantom value — it can never make fak look larger.
	if warm.FakTokenEquiv() >= cold.FakTokenEquiv() {
		t.Fatalf("warm %v must be < cold %v", warm.FakTokenEquiv(), cold.FakTokenEquiv())
	}
}

// TestFakTokenEquivBlendsWarmCold pins the durable fix for the SECOND defect the #2794 binary
// warm rule introduced: a cold-dominant session with a warm SLIVER (one warm cache_read among
// mostly-cold fires) had its ENTIRE shed discounted 10x, under-crediting fak as badly as the
// original all-1.0x rule over-credited it. The proportional blend (cacheprice.ShedTokenEquiv)
// values only the witnessed-warm portion — min(shed, cache_read) — at 0.1x and the cold
// remainder at 1.0x, so the valuation is continuous in cache_read with no cliff at 1.
func TestFakTokenEquivBlendsWarmCold(t *testing.T) {
	// Cold-dominant: 30_000 shed, a single warm cache_read token witnessed. Only that 1 token
	// prices at 0.1x; the other 29_999 keep full input. The binary rule booked all 30_000 at
	// 0.1x (=3_000), a ~10x under-credit this asserts against.
	sliver := AdjudicationSummary{
		CompactionShedTokens:      30_000,
		CompactionCacheReadTokens: 1,
	}.MechanismSavings()
	if want := 1*CacheReadMultiplier + 29_999.0; !approx(sliver.FakTokenEquiv(), want) {
		t.Fatalf("cold-sliver fak token-equiv = %v, want warm1*0.1 + cold29999 = %v", sliver.FakTokenEquiv(), want)
	}
	if sliver.FakTokenEquiv() < 29_000 {
		t.Fatalf("cold-sliver value %v collapsed toward the binary 0.1x floor (~3000) — the undercount is back", sliver.FakTokenEquiv())
	}

	// Blended midpoint: half the shed witnessed warm. warm 400*0.1 + cold 600*1.0 = 640.
	mid := AdjudicationSummary{
		CompactionShedTokens:      1000,
		CompactionCacheReadTokens: 400,
		KVPrefixReusedTokens:      100,
	}.MechanismSavings()
	if want := 400*CacheReadMultiplier + 600.0 + 100.0; !approx(mid.FakTokenEquiv(), want) {
		t.Fatalf("blended fak token-equiv = %v, want warm400*0.1 + cold600 + kv100 = %v", mid.FakTokenEquiv(), want)
	}

	// Monotone: as more of the shed is witnessed warm, fak's value only ever falls — never a
	// step. This is the property the binary cliff violated.
	var prev float64 = 1e18
	for _, cr := range []uint64{0, 250, 500, 750, 1000, 5000} {
		v := AdjudicationSummary{CompactionShedTokens: 1000, CompactionCacheReadTokens: cr}.MechanismSavings().FakTokenEquiv()
		if v > prev {
			t.Fatalf("FakTokenEquiv rose from %v to %v as cache_read grew to %d — must be monotone non-increasing", prev, v, cr)
		}
		prev = v
	}
}

// TestFakShareCollapsesUnderWarmShedDiscount is the #2794 headline acceptance: on warm
// traffic, booking compaction shed at full input (1.0x) over-credits fak's SHARE of the
// period's cache value — the adversarial audit found ~7.7x-10x overstatement. Pricing the
// warm shed at the 0.1x cache-read marginal (what those tokens actually cost as provider
// cache_reads) collapses that phantom share back to the honest few-percent the corrected
// report shows (~3.4% vs ~26% gross on the live last-2h window). This pins the CORRECTION
// end to end through MechanismSavings: fak_share = FakTokenEquiv / TotalTokenEquiv.
func TestFakShareCollapsesUnderWarmShedDiscount(t *testing.T) {
	const (
		cachedRead = 100_000 // provider cache_reads this session -> 90_000 read-rebate token-equiv
		shed       = 30_000  // compaction shed on WARM fires
	)
	warm := AdjudicationSummary{
		CachedPromptTokens:   cachedRead,
		CompactionShedTokens: shed,
		// Fully warm: the fires served a prefix at least as large as the shed from cache, so the
		// whole shed was already provider cache_reads and prices at the 0.1x read marginal. (A
		// bare cache_read=1 would NOT witness that — it would be a cold-dominant fire, which the
		// proportional blend correctly values near full input; see TestFakTokenEquivBlendsWarmCold.)
		CompactionCacheReadTokens: shed,
	}.MechanismSavings()

	// Corrected: warm shed priced at the 0.1x cache-read marginal.
	correctedShare := 100 * warm.FakTokenEquiv() / warm.TotalTokenEquiv()

	// Gross: the pre-#2794 booking that valued the same shed at full input (1.0x).
	grossFakTeq := float64(shed) * 1.0
	grossShare := 100 * grossFakTeq / (warm.ProviderTokenEquiv() + grossFakTeq)

	if grossShare <= 20 {
		t.Fatalf("gross (full-input) fak share = %.2f%%, want the >20%% over-credit the audit found", grossShare)
	}
	if correctedShare >= 5 {
		t.Fatalf("corrected (marginal) fak share = %.2f%%, want the honest few-percent (<5%%)", correctedShare)
	}
	// The overstatement factor must land in the ~7.7x-10x band the adversarial audit reported.
	if factor := grossShare / correctedShare; factor < 7.0 {
		t.Fatalf("warm-traffic overstatement factor = %.2fx, want >= 7x (audit found 7.7x-10x)", factor)
	}
}

// TestMechanismSavingsWritePremiumAttributesUpgradedTokensAt1h is the #2179 repro: a
// session whose cache-creation tokens were ALL written while the managed-cache 1h
// TTL-upgrade rung was active must price that write at CacheWrite1hMultiplier
// (2.0x), not the flat CacheWrite5mMultiplier (1.25x) the unattributed convention
// applies. Before the fix, CacheCreationTokensUpgraded did not exist and every
// write was priced at 1.25x regardless of the upgrade — under-counting the
// one-time write premium on every managed-cache session.
func TestMechanismSavingsWritePremiumAttributesUpgradedTokensAt1h(t *testing.T) {
	// All 1,000 creation tokens were upgraded: premium = 1000 * (1 - 2.0) = -1000.
	allUpgraded := AdjudicationSummary{CacheCreationTokens: 1000, CacheCreationTokensUpgraded: 1000}
	if got := allUpgraded.MechanismSavings().ProviderPromptCacheWritePremiumTokenEquiv; !approx(got, -1000) {
		t.Fatalf("all-upgraded write premium = %v, want -1000 (2.0x tier)", got)
	}

	// Half upgraded: 500 at 2.0x (-500) + 500 at 1.25x (-125) = -625.
	halfUpgraded := AdjudicationSummary{CacheCreationTokens: 1000, CacheCreationTokensUpgraded: 500}
	if got := halfUpgraded.MechanismSavings().ProviderPromptCacheWritePremiumTokenEquiv; !approx(got, -625) {
		t.Fatalf("half-upgraded write premium = %v, want -625 (blended 2.0x/1.25x)", got)
	}

	// Zero upgraded must stay BYTE-IDENTICAL to the pre-split, unattributed 5m-only
	// convention: 1000 * (1 - 1.25) = -250.
	unattributed := AdjudicationSummary{CacheCreationTokens: 1000}
	if got := unattributed.MechanismSavings().ProviderPromptCacheWritePremiumTokenEquiv; !approx(got, -250) {
		t.Fatalf("unattributed write premium = %v, want -250 (flat 5m convention unchanged)", got)
	}
}

// TestProviderCacheNetSavingsPricesUpgradedCreationAt1h proves ProviderCacheNetSavings
// (the vcachegov-backed proof engine) folds CacheCreationTokensUpgraded into the
// Ephemeral1hInputTokens axis rather than pricing every write at the 5m tier.
func TestProviderCacheNetSavingsPricesUpgradedCreationAt1h(t *testing.T) {
	s := AdjudicationSummary{CacheCreationTokens: 1000, CacheCreationTokensUpgraded: 1000, CachedPromptTokens: 10000}
	proof := s.ProviderCacheNetSavings()
	if !approx(proof.Ephemeral1hInputTokens, 1000) {
		t.Fatalf("Ephemeral1hInputTokens = %v, want 1000 (all creation tokens attributed to 1h)", proof.Ephemeral1hInputTokens)
	}
	if !approx(proof.Ephemeral5mInputTokens, 0) {
		t.Fatalf("Ephemeral5mInputTokens = %v, want 0", proof.Ephemeral5mInputTokens)
	}
}
