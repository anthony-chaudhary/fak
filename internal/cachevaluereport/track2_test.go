package cachevaluereport

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// now is a fixed clock so GeneratedAt is deterministic across the recompute runs.
var twoTrackNow = time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

// track1Fixture is a two-week WITNESSED kernel ledger trending up (60% -> 80%
// realized reuse), the same shape the CLI dry-run test uses.
func track1Fixture() []cachevalueledger.Row {
	return []cachevalueledger.Row{
		{Date: "2026-06-15", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600},
		{Date: "2026-06-22", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
}

// track2Fixture is a two-week OBSERVED-$ ledger that starts NEGATIVE (a cold
// write-heavy first week: write premium + spend exceed the rebate) and crosses
// break-even in the second week once reads repay the writes. The honest sign is
// preserved per #1303 (no floor at zero).
func track2Fixture() []SavingsRow {
	return []SavingsRow{
		// Week 1: cold writes dominate -> NET negative.
		{
			Date: "2026-06-15", SessionType: "guard",
			InputTokens: 2000, CacheCreationTokens: 8000, OutputTokens: 500,
			SavedTokenEquiv: 1000, NetSavedTokenEquiv: 1000,
			RebateUSD: 0.50, WritePremiumUSD: 2.00, SpendUSD: 1.00, CompactionSavedUSD: 0.25,
		},
		// Week 2: reads repay -> NET positive, cumulative crosses break-even.
		{
			Date: "2026-06-22", SessionType: "guard",
			InputTokens: 1000, CacheReadTokens: 9000, OutputTokens: 500,
			SavedTokenEquiv: 8100, NetSavedTokenEquiv: 8100,
			RebateUSD: 5.00, WritePremiumUSD: 0.10, SpendUSD: 0.50, CompactionSavedUSD: 0.40,
		},
	}
}

// TestFoldTwoTrackRecomputeIsByteForByte is the #1304 witness: the report
// reproduces byte-for-byte from folding the two ledgers. Re-folding the SAME
// rows must yield JSON-identical reports (the fold is pure: no clock beyond the
// supplied `now`, no map-iteration nondeterminism leaking into output order).
func TestFoldTwoTrackRecomputeIsByteForByte(t *testing.T) {
	t1, t2 := track1Fixture(), track2Fixture()

	a := FoldTwoTrack(t1, t2, twoTrackNow)
	b := FoldTwoTrack(t1, t2, twoTrackNow)

	ja, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	jb, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	if string(ja) != string(jb) {
		t.Fatalf("two-track fold is not deterministic:\n a=%s\n b=%s", ja, jb)
	}

	// The rendered table must likewise reproduce from the same fold.
	if RenderTwoTrack(a) != RenderTwoTrack(b) {
		t.Fatal("RenderTwoTrack is not reproducible from an identical fold")
	}
}

// TestFoldTwoTrackNetReDerivesFromComponents asserts every NET is exactly its
// component accounts (rebate + compaction − write premium − spend), per row and
// per folded bucket — the P&L identity the honesty fence rests on.
func TestFoldTwoTrackNetReDerivesFromComponents(t *testing.T) {
	rows := track2Fixture()
	for i, r := range rows {
		// A row with stored NetUSD must equal its computed parts; the fixture leaves
		// NetUSD zero, so NetUSDComputed is the source of truth the bucket folds.
		want := r.RebateUSD + r.CompactionSavedUSD - r.WritePremiumUSD - r.SpendUSD
		if got := r.NetUSDComputed(); math.Abs(got-want) > 1e-9 {
			t.Fatalf("row %d NetUSDComputed=%.6f want %.6f", i, got, want)
		}
	}

	rep := FoldTwoTrack(track1Fixture(), rows, twoTrackNow)
	if len(rep.Track2) != 2 {
		t.Fatalf("want 2 Track-2 buckets, got %d", len(rep.Track2))
	}

	var cumulative float64
	for i, b := range rep.Track2 {
		wantNet := b.RebateUSD + b.CompactionSavedUSD - b.WritePremiumUSD - b.SpendUSD
		if math.Abs(b.NetUSD-wantNet) > 1e-9 {
			t.Fatalf("bucket %d NetUSD=%.6f want %.6f (re-derived from components)", i, b.NetUSD, wantNet)
		}
		cumulative += b.NetUSD
		if math.Abs(b.CumulativeNetUSD-cumulative) > 1e-9 {
			t.Fatalf("bucket %d CumulativeNetUSD=%.6f want %.6f", i, b.CumulativeNetUSD, cumulative)
		}
	}
}

// TestFoldTwoTrackBreakEvenCrossing asserts the running cumulative starts below
// break-even (week 1 net is negative) and crosses it in week 2 — and that the
// crossing is shown explicitly, not hidden behind a gross headline.
func TestFoldTwoTrackBreakEvenCrossing(t *testing.T) {
	rep := FoldTwoTrack(track1Fixture(), track2Fixture(), twoTrackNow)

	w1 := rep.Track2[0]
	if w1.Provider != "unknown_provider" || w1.Mechanism != "provider_prompt_cache" {
		t.Fatalf("legacy Track-2 dimensions = %s/%s, want unknown_provider/provider_prompt_cache", w1.Provider, w1.Mechanism)
	}
	if w1.NetUSD >= 0 {
		t.Fatalf("week 1 should be NET negative (cold writes), got %.4f", w1.NetUSD)
	}
	if w1.BrokeEven {
		t.Fatalf("week 1 must not have broken even, cumulative=%.4f", w1.CumulativeNetUSD)
	}
	w2 := rep.Track2[1]
	if !w2.BrokeEven {
		t.Fatalf("week 2 should cross break-even, cumulative=%.4f", w2.CumulativeNetUSD)
	}
	if !rep.BrokeEven {
		t.Fatal("report headline should report broke-even once the running total crosses zero")
	}

	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "break-even") {
		t.Fatalf("rendered P&L must show the break-even crossing explicitly:\n%s", out)
	}
	if !strings.Contains(out, "net$") {
		t.Fatalf("rendered P&L must carry an explicit NET line:\n%s", out)
	}
	if !strings.Contains(out, "provider") || !strings.Contains(out, "mechanism") {
		t.Fatalf("rendered P&L must carry provider/mechanism columns:\n%s", out)
	}
	if !strings.Contains(out, "fak_teq") {
		t.Fatalf("rendered P&L must expose fak-authored token-equivalent attribution:\n%s", out)
	}
}

func TestFoldSavingsSplitsByProviderAndMechanism(t *testing.T) {
	rows := []SavingsRow{
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache", CacheReadTokens: 100, RebateUSD: 1.00},
		{Date: "2026-06-22", Provider: "openai", Mechanism: "provider_prompt_cache", CacheReadTokens: 200, RebateUSD: 2.00},
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "compaction_shed", CompactionShedTokens: 300, CompactionSavedUSD: 3.00},
	}
	buckets := foldSavings(rows)
	if len(buckets) != 3 {
		t.Fatalf("want 3 provider/mechanism buckets, got %d: %+v", len(buckets), buckets)
	}
	got := map[string]SavingsBucket{}
	for _, b := range buckets {
		got[b.Provider+"/"+b.Mechanism] = b
	}
	for _, key := range []string{"anthropic/provider_prompt_cache", "openai/provider_prompt_cache", "anthropic/compaction_shed"} {
		if _, ok := got[key]; !ok {
			t.Fatalf("missing bucket %s in %+v", key, buckets)
		}
	}
	if got["anthropic/compaction_shed"].CompactionShedTokens != 300 {
		t.Fatalf("compaction bucket did not retain shed tokens: %+v", got["anthropic/compaction_shed"])
	}
}

func TestFoldTwoTrackOwnerAttributionSeparatesProviderAndFakTokens(t *testing.T) {
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 1000, SavedTokenEquiv: 900, NetSavedTokenEquiv: 900, RebateUSD: 1,
		},
		{
			Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 300, SavedTokenEquiv: 300, NetSavedTokenEquiv: 300, CompactionSavedUSD: 0.3,
		},
	}
	rep := FoldTwoTrack(track1, track2, twoTrackNow)
	if len(rep.OwnerAttribution) != 1 {
		t.Fatalf("want one owner-attribution bucket, got %d: %+v", len(rep.OwnerAttribution), rep.OwnerAttribution)
	}
	got := rep.OwnerAttribution[0]
	if got.ProviderPromptCacheTokenEquiv != 900 {
		t.Fatalf("provider prompt-cache token-equiv = %.1f, want 900", got.ProviderPromptCacheTokenEquiv)
	}
	if got.FakKVPrefixReusedTokens != 800 || got.FakCompactionShedTokens != 300 {
		t.Fatalf("fak mechanism tokens not decomposed: %+v", got)
	}
	if got.FakAuthoredTokenEquiv != 1100 {
		t.Fatalf("fak-authored token-equiv = %.1f, want 1100", got.FakAuthoredTokenEquiv)
	}
	out := RenderTwoTrack(rep)
	for _, want := range []string{"Owner attribution", "provider_teq", "fak_teq", "fak_share", "kv_tok", "compact_tok", "900", "1100"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

func TestFoldTwoTrackComponentHealthNamesPlanesAndFidelity(t *testing.T) {
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 1000, SavedTokenEquiv: 900, NetSavedTokenEquiv: 900, RebateUSD: 1,
		},
		{
			Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 300, SavedTokenEquiv: 300, NetSavedTokenEquiv: 300,
		},
	}
	rep := FoldTwoTrack(track1, track2, twoTrackNow)
	byPlane := map[string]ComponentHealth{}
	for _, h := range rep.ComponentHealth {
		byPlane[h.Plane] = h
	}
	for _, plane := range []string{"local_kv", "provider_prompt_cache", "context_compression", "gateway_usage"} {
		if _, ok := byPlane[plane]; !ok {
			t.Fatalf("component_health missing plane %q: %+v", plane, rep.ComponentHealth)
		}
	}
	if h := byPlane["local_kv"]; h.Owner != "fak" || h.Fidelity != "lossless" || h.Evidence != "WITNESSED" || h.Status != "measured" {
		t.Fatalf("local_kv health = %+v, want fak/lossless/WITNESSED/measured", h)
	}
	if h := byPlane["provider_prompt_cache"]; h.Owner != "provider" || h.Fidelity != "lossless" || h.Evidence != "OBSERVED" || h.Status != "measured" {
		t.Fatalf("provider health = %+v, want provider/lossless/OBSERVED/measured", h)
	}
	if h := byPlane["context_compression"]; h.Owner != "fak" || h.Fidelity != "lossy" || h.Status != "measured" {
		t.Fatalf("compaction health = %+v, want fak/lossy/measured", h)
	}
	if h := byPlane["gateway_usage"]; h.Status != "missing" || h.Fidelity != "passive" {
		t.Fatalf("usage health = %+v, want missing/passive without usage ledger rows", h)
	}
	out := RenderTwoTrack(rep)
	for _, want := range []string{"Component health", "local_kv", "provider_prompt_cache", "context_compression", "lossy"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing component-health field %q:\n%s", want, out)
		}
	}
}

// TestFoldTwoTrackOwnerAttributionFakSharePct pins the pre-divided headline: the
// bucket carries fak's share of the period total so "what % of the cache value is
// fak's?" is answered by the report, not left as a division for the reader — and
// the share is refused (nil / "-") when the period total is not positive, so an
// upside-down period can never masquerade as a measured 0%.
func TestFoldTwoTrackOwnerAttributionFakSharePct(t *testing.T) {
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "run", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 1000, SavedTokenEquiv: 900, NetSavedTokenEquiv: 900,
		},
		{
			Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 300, SavedTokenEquiv: 300, NetSavedTokenEquiv: 300,
		},
	}
	rep := FoldTwoTrack(track1, track2, twoTrackNow)
	if len(rep.OwnerAttribution) != 1 {
		t.Fatalf("want one owner-attribution bucket, got %d", len(rep.OwnerAttribution))
	}
	got := rep.OwnerAttribution[0]
	pct, ok := got.FakShareOfTotalPct()
	if !ok || !approxTrack2(pct, 55) { // fak 1100 of total 2000
		t.Fatalf("fak share = %.4f ok=%v, want 55%% of the period total", pct, ok)
	}
	if got.FakSharePct == nil || !approxTrack2(*got.FakSharePct, 55) {
		t.Fatalf("folded bucket must carry the share for the JSON surface: %+v", got.FakSharePct)
	}
	if out := RenderTwoTrack(rep); !strings.Contains(out, "55.0000%") {
		t.Fatalf("render missing the fak share:\n%s", out)
	}

	// A period whose only row is a provider write premium (negative net) has no
	// positive total: the share must be refused, not reported as a number.
	neg := FoldTwoTrack(nil, []SavingsRow{{
		Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
		CacheCreationTokens: 1000, SavedTokenEquiv: -250, NetSavedTokenEquiv: -250,
	}}, twoTrackNow)
	if len(neg.OwnerAttribution) != 1 {
		t.Fatalf("want one owner-attribution bucket, got %d", len(neg.OwnerAttribution))
	}
	if _, ok := neg.OwnerAttribution[0].FakShareOfTotalPct(); ok {
		t.Fatal("share must be undefined when the period total is not positive")
	}
	if neg.OwnerAttribution[0].FakSharePct != nil {
		t.Fatal("folded bucket must not carry a share when the total is not positive")
	}
	if out := RenderTwoTrack(neg); !strings.Contains(out, "-") {
		t.Fatalf("render must show \"-\" for an undefined share:\n%s", out)
	}
}

func TestNewSavingsRowsSplitsProviderAndCompaction(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:          "guard",
		Provider:             "anthropic",
		Context:              "claude",
		InputTokens:          1000,
		CacheReadTokens:      10_000,
		CacheCreationTokens:  2000,
		OutputTokens:         500,
		CompactionShedTokens: 3000,
		Pricing:              SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}, twoTrackNow)

	if len(rows) != 2 {
		t.Fatalf("want provider + compaction rows, got %d: %+v", len(rows), rows)
	}
	provider, compaction := rows[0], rows[1]
	if provider.Provider != "anthropic" || provider.Mechanism != "provider_prompt_cache" {
		t.Fatalf("provider row dimensions = %s/%s", provider.Provider, provider.Mechanism)
	}
	if compaction.Provider != "fak" || compaction.Mechanism != "compaction_shed" {
		t.Fatalf("compaction row dimensions = %s/%s", compaction.Provider, compaction.Mechanism)
	}
	if !approxTrack2(provider.SavedTokenEquiv, 8500) {
		t.Fatalf("provider saved token-equiv = %.4f, want 8500", provider.SavedTokenEquiv)
	}
	if !approxTrack2(provider.RebateUSD, 0.045) || !approxTrack2(provider.WritePremiumUSD, 0.0025) {
		t.Fatalf("provider dollars not priced from observed axes: %+v", provider)
	}
	if !approxTrack2(provider.SpendUSD, 0.035) {
		t.Fatalf("provider spend = %.6f, want 0.035", provider.SpendUSD)
	}
	if compaction.CompactionShedTokens != 3000 || !approxTrack2(compaction.CompactionSavedUSD, 0.015) {
		t.Fatalf("compaction row did not price shed tokens: %+v", compaction)
	}
	// No cache_read witnessed at the fire -> observed-COLD -> shed keeps the full-input
	// basis (1.0x), the pre-#2794 number, and the row is labeled FULL_INPUT (#2796).
	if compaction.ValuationBasis != ValuationBasisFullInput {
		t.Fatalf("observed-cold compaction basis = %q, want %s", compaction.ValuationBasis, ValuationBasisFullInput)
	}
	if provider.ValuationBasis != ValuationBasisObservedNet {
		t.Fatalf("provider basis = %q, want %s", provider.ValuationBasis, ValuationBasisObservedNet)
	}
}

// TestWarmCompactionShedPricedAtMarginal is the #2794 witness: a compaction fire that
// landed on a WARM prefix (the provider reported cache_read at the fire) drops tokens the
// provider would have billed as cache_reads at 0.1x, so the avoided cost is shed*0.1x, not
// the full-input shed*1.0x the pre-fix report booked. The row is labeled CACHE_READ_MARGINAL.
func TestWarmCompactionShedPricedAtMarginal(t *testing.T) {
	warm := NewSavingsRows(SavingsObservation{
		SessionType:               "guard",
		Provider:                  "anthropic",
		Context:                   "claude",
		CompactionShedTokens:      3000,
		CompactionCacheReadTokens: 50_000, // a warm fire: provider served the prefix from cache
		CompactionFired:           1,
		Pricing:                   SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}, twoTrackNow)
	if len(warm) != 1 {
		t.Fatalf("want one compaction row, got %d: %+v", len(warm), warm)
	}
	row := warm[0]
	if row.ValuationBasis != ValuationBasisCacheReadMarginal {
		t.Fatalf("warm fire basis = %q, want %s", row.ValuationBasis, ValuationBasisCacheReadMarginal)
	}
	// 3000 shed * 0.1x = 300 token-equiv; priced at $5/MTok = $0.0015 (a tenth of the
	// $0.015 an observed-cold fire of the same shed would book).
	if !approxTrack2(row.SavedTokenEquiv, 300) {
		t.Fatalf("warm shed token-equiv = %.4f, want 300 (shed*0.1x)", row.SavedTokenEquiv)
	}
	if !approxTrack2(row.CompactionSavedUSD, 0.0015) {
		t.Fatalf("warm shed $ = %.6f, want 0.0015 (marginal, not full-input 0.015)", row.CompactionSavedUSD)
	}
}

// TestShedValueAgreesWithFireGate is the #2798 acceptance: the value the report books for
// a shed token on a warm fire must equal the value the fire gate
// (agent.CacheBurstBreakEvenTurns) prices that same token at. Both consult a single 0.1x
// read multiplier; this asserts they have not drifted.
//
// The canonical source of that 0.1 is gateway.CacheReadMultiplier (internal/gateway/
// cache_pricing.go), pinned by gateway's own cache_pricing_test.go. It is copied down —
// NOT imported — into agent.defaultCacheReadMult (agent must not import gateway: cycle) and
// into this package's two constants (cachevaluereport is architest tier 2 and must not
// import the tier-4 gateway/agent packages). So a real cross-package symbol pin is
// architecturally impossible here; the honest guard is (1) lock this file's two marginals to
// ONE value (providerCacheReadMultiplier), so no same-file edit can split the compaction and
// provider bases, and (2) assert that shared value still equals the documented 0.1 mirror,
// with a literal that names what it mirrors so a reviewer changing the canonical constant
// knows to sweep every copy. If gateway.CacheReadMultiplier ever moves, gateway's pin test
// fails first; this test's literal is the reminder that this copy must move with it.
func TestShedValueAgreesWithFireGate(t *testing.T) {
	const shed = 10_000
	// canonicalReadMarginal mirrors gateway.CacheReadMultiplier / agent.defaultCacheReadMult
	// (both unexported or tier-4, hence uninmportable here). Named, not a bare 0.1, so the
	// drift it guards is legible.
	const canonicalReadMarginal = 0.1
	// (1) Lock this file's two marginals together: the compaction shed and the provider
	// read rebate must price a cached token identically — the single-source-of-truth the
	// pre-#2798 split (compaction 1.0x vs provider 0.1x) violated.
	if compactionShedMarginalMultiplier != providerCacheReadMultiplier {
		t.Fatalf("in-file marginals split: compaction %.3f vs provider %.3f — one source of truth broken (#2798)",
			compactionShedMarginalMultiplier, providerCacheReadMultiplier)
	}
	// (2) That shared value must still equal the canonical 0.1 the fire gate uses.
	if compactionShedMarginalMultiplier != canonicalReadMarginal {
		t.Fatalf("report marginal %.3f drifted from canonical fire-gate readMult %.3f (gateway.CacheReadMultiplier) — sweep all copies (#2798)",
			compactionShedMarginalMultiplier, canonicalReadMarginal)
	}
	rows := NewSavingsRows(SavingsObservation{
		Provider:                  "anthropic",
		Context:                   "claude",
		CompactionShedTokens:      shed,
		CompactionCacheReadTokens: 1, // any positive read => warm
		CompactionFired:           1,
		Pricing:                   SavingsPricing{InputPerMTokUSD: 5},
	}, twoTrackNow)
	if len(rows) != 1 {
		t.Fatalf("want one compaction row, got %d", len(rows))
	}
	// The fire gate values the shed token at shed*marginal per turn; the report must
	// book the same shed*marginal token-equivalent, not shed*1.0x.
	wantTokenEquiv := float64(shed) * canonicalReadMarginal
	if !approxTrack2(rows[0].SavedTokenEquiv, wantTokenEquiv) {
		t.Fatalf("reported shed value = %.1f, fire-gate value = %.1f — they must agree (#2798)",
			rows[0].SavedTokenEquiv, wantTokenEquiv)
	}
}

// TestFakAuthoredTokenEquivRollupsAgree locks the invariant that the two roll-ups which
// credit fak's authored token-equivalent — the fleet-benefit report (fakTokenEqFromRow) and
// the owner-attribution fold (foldOwnerAttribution) — derive it the SAME way, because both
// now route through the shared fakAuthoredTokenEquiv (#2798). The regression it guards is
// real: the fold re-priced a legacy (unpriced) warm-fire row at the 0.1x cache-read marginal
// while the fleet report booked the raw shed at 1.0x, so the same historical row inflated
// fak's fleet token-equiv 10x on one page and not the other.
func TestFakAuthoredTokenEquivRollupsAgree(t *testing.T) {
	cases := []struct {
		name            string
		netSaved        float64
		shed, cacheRead uint64
		want            float64
	}{
		{"priced row uses its net token-equiv", 250, 3000, 1500, 250},
		{"legacy warm re-priced at 0.1x marginal", 0, 3000, 1500, 300},
		{"legacy cold keeps full input 1.0x", 0, 3000, 0, 3000},
		{"nothing shed credits nothing", 0, 0, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := fakAuthoredTokenEquiv(tc.netSaved, tc.shed, tc.cacheRead); !approxTrack2(got, tc.want) {
				t.Fatalf("fakAuthoredTokenEquiv = %.4f, want %.4f", got, tc.want)
			}
			// The fleet-benefit roll-up must credit exactly the shared value, row for row.
			row := SavingsRow{
				Provider: "fak", Mechanism: "compaction_shed",
				NetSavedTokenEquiv: tc.netSaved, CompactionShedTokens: tc.shed,
				CompactionCacheReadTokens: tc.cacheRead,
			}
			if fr := fakTokenEqFromRow(row); !approxTrack2(fr, tc.want) {
				t.Fatalf("fakTokenEqFromRow = %.4f, want %.4f (must match shared helper)", fr, tc.want)
			}
			// The owner-attribution fold must credit exactly the shared value, bucket for bucket.
			owner := foldOwnerAttribution(nil, []SavingsBucket{{
				Period: "2026-W26", Provider: "fak", Mechanism: "compaction_shed",
				NetSavedTokenEquiv: tc.netSaved, CompactionShedTokens: tc.shed,
				CompactionCacheReadTokens: tc.cacheRead,
			}})
			if len(owner) != 1 {
				t.Fatalf("foldOwnerAttribution returned %d buckets, want 1", len(owner))
			}
			if !approxTrack2(owner[0].FakAuthoredTokenEquiv, tc.want) {
				t.Fatalf("fold FakAuthoredTokenEquiv = %.4f, want %.4f (must match shared helper)",
					owner[0].FakAuthoredTokenEquiv, tc.want)
			}
		})
	}
}

// TestInferValuationBasisDefersToShedValuation locks InferValuationBasis to the SAME
// warm/cold decision compactionShedValuation makes, so the basis a legacy row infers can
// never drift from the basis its shed was priced at (#2798). Re-introducing a second inline
// CompactionCacheReadTokens>0 check in InferValuationBasis is exactly the drift this guards.
func TestInferValuationBasisDefersToShedValuation(t *testing.T) {
	for _, cacheRead := range []uint64{0, 1, 5000} {
		_, want := compactionShedValuation(cacheRead)
		row := SavingsRow{Mechanism: "compaction_shed", CompactionCacheReadTokens: cacheRead}
		if got := row.InferValuationBasis(); got != want {
			t.Fatalf("InferValuationBasis(cache_read=%d) = %q, compactionShedValuation basis = %q — must agree",
				cacheRead, got, want)
		}
	}
	// An explicit basis short-circuits inference.
	explicit := SavingsRow{Mechanism: "compaction_shed", ValuationBasis: ValuationBasisFullInput, CompactionCacheReadTokens: 9000}
	if got := explicit.InferValuationBasis(); got != ValuationBasisFullInput {
		t.Fatalf("explicit ValuationBasis not honored: got %q", got)
	}
	// Provider prompt-cache rows are observed-net; unknown mechanisms have no fak $ to label.
	if got := (SavingsRow{Mechanism: "provider_prompt_cache"}).InferValuationBasis(); got != ValuationBasisObservedNet {
		t.Fatalf("provider row basis = %q, want OBSERVED_NET", got)
	}
	if got := (SavingsRow{Mechanism: "something_else"}).InferValuationBasis(); got != "" {
		t.Fatalf("unknown mechanism basis = %q, want empty", got)
	}
}

// TestRenderRefusesUnlabeledFakDollar is the #2796 render acceptance: the P&L table must
// stamp the price basis onto every fak dollar figure, and must REFUSE (render a marker, not
// a bare number) a nonzero fak $ that carries no inferable basis. A warm fak bucket renders
// its CACHE_READ_MARGINAL basis; a cold one renders FULL_INPUT.
func TestRenderRefusesUnlabeledFakDollar(t *testing.T) {
	// A warm compaction bucket: cache_read witnessed at the fires, so the $ is marginal.
	warm := []SavingsRow{{
		Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
		CompactionShedTokens: 3000, CompactionCacheReadTokens: 50_000,
		SavedTokenEquiv: 300, NetSavedTokenEquiv: 300, CompactionSavedUSD: 0.0015,
		ValuationBasis: ValuationBasisCacheReadMarginal,
	}}
	out := RenderTwoTrack(FoldTwoTrack(nil, warm, twoTrackNow))
	if !strings.Contains(out, string(ValuationBasisCacheReadMarginal)) {
		t.Fatalf("warm fak $ must render its CACHE_READ_MARGINAL basis:\n%s", out)
	}

	// A cold compaction bucket keeps the full-input basis label.
	cold := []SavingsRow{{
		Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
		CompactionShedTokens: 3000, SavedTokenEquiv: 3000, NetSavedTokenEquiv: 3000,
		CompactionSavedUSD: 0.015, ValuationBasis: ValuationBasisFullInput,
	}}
	out = RenderTwoTrack(FoldTwoTrack(nil, cold, twoTrackNow))
	if !strings.Contains(out, string(ValuationBasisFullInput)) {
		t.Fatalf("cold fak $ must render its FULL_INPUT basis:\n%s", out)
	}

	// A fak bucket with a nonzero $ but an unknown mechanism has no inferable basis: the
	// renderer must refuse it, not print a bare dollar figure.
	unlabeled := []SavingsRow{{
		Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_mystery",
		CompactionSavedUSD: 9.99,
	}}
	out = RenderTwoTrack(FoldTwoTrack(nil, unlabeled, twoTrackNow))
	if !strings.Contains(out, "REFUSED:no-basis") {
		t.Fatalf("unlabeled fak $ must be REFUSED, not printed bare:\n%s", out)
	}
}

// TestNewSavingsRowsPricesUpgradedCreationAt1hTier is the #2179 fix witness: a
// session whose cache-creation tokens were partly written while the managed-cache
// 1h TTL-upgrade rung was active must price that slice at the 2.0x tier instead of
// the flat 1.25x the pre-split convention applied to every write.
func TestNewSavingsRowsPricesUpgradedCreationAt1hTier(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:                 "guard",
		Provider:                    "anthropic",
		Context:                     "claude",
		InputTokens:                 1000,
		CacheReadTokens:             10_000,
		CacheCreationTokens:         2000,
		CacheCreationTokensUpgraded: 1200, // 1200 tokens at 2.0x, 800 at 1.25x
		OutputTokens:                500,
		Pricing:                     SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}, twoTrackNow)
	if len(rows) != 1 {
		t.Fatalf("want one provider row, got %d: %+v", len(rows), rows)
	}
	row := rows[0]
	if row.CacheCreationTokensUpgraded != 1200 {
		t.Fatalf("CacheCreationTokensUpgraded = %d, want 1200", row.CacheCreationTokensUpgraded)
	}
	if row.CacheCreationTierProvenance != CacheCreationTierProvenanceGatewayAttributed {
		t.Fatalf("CacheCreationTierProvenance = %q, want %q", row.CacheCreationTierProvenance, CacheCreationTierProvenanceGatewayAttributed)
	}
	// write premium = 1200*(2.0-1) + 800*(1.25-1) = 1200 + 200 = 1400 token-equiv,
	// priced at $5/MTok = 0.007. The pre-split flat-1.25x convention would have
	// given 2000*0.25 = 500 token-equiv (0.0025) instead — under-counting the
	// premium, exactly the #2179 bug.
	if !approxTrack2(row.WritePremiumUSD, 0.007) {
		t.Fatalf("WritePremiumUSD = %.6f, want 0.007 (blended 2.0x/1.25x)", row.WritePremiumUSD)
	}
	if !approxTrack2(row.NetUSD, row.NetUSDComputed()) {
		t.Fatalf("NetUSD = %.6f, want recomputed %.6f", row.NetUSD, row.NetUSDComputed())
	}
}

// TestNewSavingsRowsZeroUpgradedIsByteIdenticalToUnsplit locks in acceptance
// bullet 3 of #2179: an observation with no upgrade attribution must price
// byte-identically to the pre-split convention.
func TestNewSavingsRowsZeroUpgradedIsByteIdenticalToUnsplit(t *testing.T) {
	obsBase := SavingsObservation{
		SessionType:         "guard",
		Provider:            "anthropic",
		Context:             "claude",
		InputTokens:         1000,
		CacheReadTokens:     10_000,
		CacheCreationTokens: 2000,
		OutputTokens:        500,
		Pricing:             SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}
	unsplit := NewSavingsRows(obsBase, twoTrackNow)[0]
	obsExplicitZero := obsBase
	obsExplicitZero.CacheCreationTokensUpgraded = 0
	explicitZero := NewSavingsRows(obsExplicitZero, twoTrackNow)[0]
	if explicitZero != unsplit {
		t.Fatalf("zero-upgraded row diverged from the unsplit row:\n  unsplit:      %+v\n  explicitZero: %+v", unsplit, explicitZero)
	}
	if unsplit.CacheCreationTierProvenance != "" {
		t.Fatalf("CacheCreationTierProvenance = %q, want empty when no upgrade was attributed", unsplit.CacheCreationTierProvenance)
	}
}

func TestNewSavingsRowsMarksDollarBlindWithoutPricing(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:         "guard",
		Provider:            "openai",
		Context:             "codex",
		CacheReadTokens:     10_000,
		CacheCreationTokens: 1000,
		Pricing:             SavingsPricing{DollarBlind: true, Source: "none"},
	}, twoTrackNow)
	if len(rows) != 1 {
		t.Fatalf("want one provider row, got %d: %+v", len(rows), rows)
	}
	if rows[0].DollarStatus != SavingsDollarStatusBlind {
		t.Fatalf("missing dollar-blind marker on unpriced row: %+v", rows[0])
	}
	if rows[0].PricingSource != "none" {
		t.Fatalf("pricing source = %q, want none", rows[0].PricingSource)
	}
	if rows[0].RebateUSD != 0 || rows[0].WritePremiumUSD != 0 || rows[0].SpendUSD != 0 || rows[0].NetUSD != 0 {
		t.Fatalf("unpriced row dollar fields should stay placeholders at zero: %+v", rows[0])
	}

	line, err := AppendSavingsLine(rows[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, `"dollar_status":"dollar_blind"`) {
		t.Fatalf("ledger line must carry the dollar-blind marker: %s", line)
	}

	rep := FoldTwoTrack(nil, rows, twoTrackNow)
	if rep.BrokeEven {
		t.Fatalf("dollar-blind zero dollars must not be reported as break-even: %+v", rep)
	}
	if rep.DollarBlindRows != 1 || len(rep.Track2) != 1 || rep.Track2[0].DollarStatus != SavingsDollarStatusBlind {
		t.Fatalf("fold did not carry dollar-blind status: %+v", rep)
	}
	var providerHealth *ComponentHealth
	for i := range rep.ComponentHealth {
		if rep.ComponentHealth[i].Plane == "provider_prompt_cache" {
			providerHealth = &rep.ComponentHealth[i]
			break
		}
	}
	if providerHealth == nil || providerHealth.Status != "dollar_blind" {
		t.Fatalf("provider component health = %+v, want dollar_blind", providerHealth)
	}
	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "dollar-blind") || !strings.Contains(out, "zero dollar fields are placeholders") {
		t.Fatalf("render should make dollar-blind rows explicit:\n%s", out)
	}
}

func TestNewSavingsRowsSkipsProviderWithoutCacheCounters(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:  "serve",
		Provider:     "openai",
		Context:      "http",
		InputTokens:  1000,
		OutputTokens: 200,
		Pricing:      SavingsPricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15},
	}, twoTrackNow)
	if len(rows) != 0 {
		t.Fatalf("pure input/output spend is not provider-cache evidence; got rows: %+v", rows)
	}
}

func TestAppendSavingsRoundTripsRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	rows := NewSavingsRows(SavingsObservation{
		SessionType:     "serve",
		Provider:        "openai",
		Context:         "http",
		CacheReadTokens: 100,
		Pricing:         SavingsPricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15},
	}, twoTrackNow)
	if len(rows) != 1 {
		t.Fatalf("want one provider row, got %d", len(rows))
	}
	if err := AppendSavings(path, rows[0]); err != nil {
		t.Fatalf("AppendSavings: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read append file: %v", err)
	}
	got := ParseSavingsLedger(string(raw))
	if len(got) != 1 {
		t.Fatalf("want one parsed row, got %d from %q", len(got), string(raw))
	}
	if got[0].Provider != "openai" || got[0].Mechanism != "provider_prompt_cache" || got[0].CacheReadTokens != 100 {
		t.Fatalf("parsed row lost dimensions/tokens: %+v", got[0])
	}
}

func approxTrack2(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

// TestFoldTwoTrackProvenanceNeverBlended is the honesty-fence witness: Track 1
// keeps its WITNESSED self-labels and Track 2 carries the OBSERVED projection
// fence; the two are separate fields, never a blended number.
func TestFoldTwoTrackProvenanceNeverBlended(t *testing.T) {
	rep := FoldTwoTrack(track1Fixture(), track2Fixture(), twoTrackNow)

	if !rep.Track1.VsNaiveMultipleExcluded {
		t.Fatal("Track 1 must keep the #1066 vs-naive-excluded fence")
	}
	if rep.Track1.PublishableValueFamily != PublishableValueFamily {
		t.Fatalf("Track 1 lost its WITNESSED value family: %q", rep.Track1.PublishableValueFamily)
	}
	if !strings.Contains(rep.ProjectionFence, "cost projection over labelled sources") ||
		!strings.Contains(rep.ProjectionFence, "never blended") {
		t.Fatalf("report missing the OBSERVED projection / never-blended fence: %q", rep.ProjectionFence)
	}
	// The verdict is MEASURED only when both tracks carry evidence.
	if rep.Verdict != "MEASURED" {
		t.Fatalf("both tracks have evidence; verdict should be MEASURED, got %q", rep.Verdict)
	}
}

// TestFoldTwoTrackEmptyTrack2IsHonest checks that with no OBSERVED-$ rows the
// report still folds Track 1 and says Track 2 is empty (rung B not appending yet),
// rather than fabricating a $ number.
func TestFoldTwoTrackEmptyTrack2IsHonest(t *testing.T) {
	rep := FoldTwoTrack(track1Fixture(), nil, twoTrackNow)
	if len(rep.Track2) != 0 {
		t.Fatalf("empty Track-2 ledger should fold to no buckets, got %d", len(rep.Track2))
	}
	if rep.BrokeEven {
		t.Fatal("no OBSERVED-$ rows cannot have broken even")
	}
	if !strings.Contains(rep.NextAction, "#1303") {
		t.Fatalf("empty Track 2 should point at rung B (#1303): %q", rep.NextAction)
	}
	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "no OBSERVED-$ rows yet") {
		t.Fatalf("render should say Track 2 is empty:\n%s", out)
	}
}

// TestParseSavingsLedgerRoundTrips checks the durable JSONL shape: a row appended
// with AppendSavingsLine parses back via ParseSavingsLedger to the same values.
func TestParseSavingsLedgerRoundTrips(t *testing.T) {
	row := SavingsRow{
		Date: "2026-06-22", SessionType: "guard",
		InputTokens: 1000, CacheReadTokens: 9000, CacheCreationTokens: 100, OutputTokens: 500,
		CompactionShedTokens: 1200,
		RebateUSD:            5.0, WritePremiumUSD: 0.1, SpendUSD: 0.5, CompactionSavedUSD: 0.4,
		NetUSD: 4.8,
	}
	line, err := AppendSavingsLine(row)
	if err != nil {
		t.Fatal(err)
	}
	got := ParseSavingsLedger(line + "\n")
	if len(got) != 1 {
		t.Fatalf("want 1 parsed row, got %d", len(got))
	}
	if got[0].Schema != SavingsLedgerSchema {
		t.Fatalf("AppendSavingsLine should stamp the schema, got %q", got[0].Schema)
	}
	if got[0].CacheReadTokens != 9000 || got[0].CompactionShedTokens != 1200 {
		t.Fatalf("round-trip dropped token axes: %+v", got[0])
	}
	if got[0].Provider != "unknown_provider" || got[0].Mechanism != "provider_prompt_cache" {
		t.Fatalf("round-trip dimensions = %s/%s, want unknown_provider/provider_prompt_cache", got[0].Provider, got[0].Mechanism)
	}
	if math.Abs(got[0].NetUSDComputed()-(5.0+0.4-0.1-0.5)) > 1e-9 {
		t.Fatalf("round-trip NET mismatch: %.6f", got[0].NetUSDComputed())
	}
}

// TestNewSavingsRowsEmitsCompactionHealthOnFiredButZeroShed is the #2039 witness: a
// session where the lever FIRED but shed NOTHING (the anchor-starved case #1407
// documents) must still emit a compaction_shed row carrying the health fields — not
// silence. Before #2039 the durable record was indistinguishable from an idle lever.
func TestNewSavingsRowsEmitsCompactionHealthOnFiredButZeroShed(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:             "guard",
		Provider:                "anthropic",
		Context:                 "claude",
		CompactionShedTokens:    0,
		CompactionFired:         3,
		CompactionBailed:        2,
		CompactionAnchorStarved: 2,
		CompactionBudget:        48000,
		Pricing:                 SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}, twoTrackNow)

	if len(rows) != 1 {
		t.Fatalf("want 1 compaction_health row (fired>0, shed=0), got %d: %+v", len(rows), rows)
	}
	row := rows[0]
	if row.Provider != "fak" || row.Mechanism != "compaction_shed" {
		t.Fatalf("row dimensions = %s/%s, want fak/compaction_shed", row.Provider, row.Mechanism)
	}
	if row.CompactionFired != 3 || row.CompactionBailed != 2 || row.CompactionAnchorStarved != 2 {
		t.Fatalf("health fields not persisted: fired=%d bailed=%d starved=%d", row.CompactionFired, row.CompactionBailed, row.CompactionAnchorStarved)
	}
	if row.CompactionBudget != 48000 {
		t.Fatalf("budget not persisted: %d", row.CompactionBudget)
	}
	if row.CompactionShedTokens != 0 {
		t.Fatalf("shed must stay 0 when nothing was shed: %d", row.CompactionShedTokens)
	}
	if row.SavedTokenEquiv != 0 || row.CompactionSavedUSD != 0 {
		t.Fatalf("zero-shed row must carry $0 saving: saved_teq=%.1f saved_usd=%.6f", row.SavedTokenEquiv, row.CompactionSavedUSD)
	}

	line, err := AppendSavingsLine(row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, `"compaction_fired":3`) || !strings.Contains(line, `"compaction_anchor_starved":2`) {
		t.Fatalf("durable line must carry health fields: %s", line)
	}
}

// TestNewSavingsRowsOmitsCompactionRowWhenLeverIdle verifies that a session with no
// compaction activity (fired=0, bailed=0, shed=0) produces NO compaction row — the
// lever was idle or disabled, which is the case that should remain silent.
func TestNewSavingsRowsOmitsCompactionRowWhenLeverIdle(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:  "guard",
		Provider:     "anthropic",
		Context:      "claude",
		InputTokens:  100,
		OutputTokens: 50,
		Pricing:      SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}, twoTrackNow)
	if len(rows) != 0 {
		t.Fatalf("idle compaction lever must produce no row, got %d: %+v", len(rows), rows)
	}
}

// TestFoldSavingsAggregatesCompactionHealth verifies the per-period fold sums the
// health fields so the report can chart the fire/starve/shed trend across sessions.
func TestFoldSavingsAggregatesCompactionHealth(t *testing.T) {
	rows := []SavingsRow{
		{
			Date: "2026-06-15", Provider: "fak", Mechanism: "compaction_shed",
			CompactionFired: 3, CompactionAnchorStarved: 3, CompactionShedTokens: 0,
			CompactionBudget: 48000,
		},
		{
			Date: "2026-06-17", Provider: "fak", Mechanism: "compaction_shed",
			CompactionFired: 5, CompactionBailed: 1, CompactionShedTokens: 8000,
			CompactionBudget: 48000,
		},
	}
	buckets := foldSavings(rows)
	if len(buckets) != 1 {
		t.Fatalf("want 1 folded bucket, got %d", len(buckets))
	}
	b := buckets[0]
	if b.CompactionFired != 8 {
		t.Fatalf("folded fired = %d, want 8", b.CompactionFired)
	}
	if b.CompactionBailed != 1 || b.CompactionAnchorStarved != 3 {
		t.Fatalf("folded bailed/starved = %d/%d, want 1/3", b.CompactionBailed, b.CompactionAnchorStarved)
	}
	if b.CompactionShedTokens != 8000 {
		t.Fatalf("folded shed = %d, want 8000", b.CompactionShedTokens)
	}
	if b.CompactionBudget != 48000 {
		t.Fatalf("folded budget = %d, want 48000", b.CompactionBudget)
	}

	rep := FoldTwoTrack(nil, rows, twoTrackNow)
	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "Compaction lever health") {
		t.Fatalf("render must show the compaction health section:\n%s", out)
	}
	if !strings.Contains(out, "fired") || !strings.Contains(out, "starved") {
		t.Fatalf("render must show fired/starved columns:\n%s", out)
	}
}
