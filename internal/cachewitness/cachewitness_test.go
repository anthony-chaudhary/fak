package cachewitness

import (
	"encoding/json"
	"strings"
	"testing"
)

// a representative slice of a fak gateway /metrics body: the in-kernel KV-prefix
// family (WITNESSED) + the provider cache_read counter (OBSERVED), with HELP/TYPE
// lines and an unrelated series interleaved to prove the scraper ignores them.
const sampleMetrics = `# HELP fak_gateway_kv_prefix_turns_total In-kernel model turns.
# TYPE fak_gateway_kv_prefix_turns_total counter
fak_gateway_kv_prefix_turns_total 7
# HELP fak_gateway_kv_prefix_prompt_tokens_total Prefill tokens.
fak_gateway_kv_prefix_prompt_tokens_total 16384
fak_gateway_kv_prefix_reused_tokens_total 15000
fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} 5
fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} 1
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 1
fak_blob_resident_bytes 4096
fak_gateway_inference_cached_prompt_tokens_total 2048
`

func TestParseFoldsCacheFamily(t *testing.T) {
	r, err := Parse("http://box:8080/metrics", sampleMetrics)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if r.KVPrefix.Turns != 7 {
		t.Errorf("turns = %d, want 7", r.KVPrefix.Turns)
	}
	if r.KVPrefix.PromptTokens != 16384 {
		t.Errorf("prompt tokens = %d, want 16384", r.KVPrefix.PromptTokens)
	}
	if r.KVPrefix.ReusedTokens != 15000 {
		t.Errorf("reused tokens = %d, want 15000", r.KVPrefix.ReusedTokens)
	}
	if r.KVPrefix.FrozenTurns != 5 || r.KVPrefix.PartialTurns != 1 || r.KVPrefix.ColdTurns != 1 {
		t.Errorf("regime turns = (%d,%d,%d), want (5,1,1)", r.KVPrefix.FrozenTurns, r.KVPrefix.PartialTurns, r.KVPrefix.ColdTurns)
	}
	if r.ProviderCacheReadTokens != 2048 {
		t.Errorf("provider cache_read = %d, want 2048", r.ProviderCacheReadTokens)
	}
	if r.GatewayURL != "http://box:8080/metrics" {
		t.Errorf("gateway url not recorded: %q", r.GatewayURL)
	}
	if r.GatewayUptimeTurns != 7 {
		t.Errorf("gateway uptime turns = %d, want cumulative 7", r.GatewayUptimeTurns)
	}
}

func TestRecordSubSubtractsCumulativeGatewayCounters(t *testing.T) {
	const before = `fak_gateway_kv_prefix_turns_total 10
fak_gateway_kv_prefix_prompt_tokens_total 10000
fak_gateway_kv_prefix_reused_tokens_total 4000
fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} 2
fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} 3
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 5
fak_gateway_inference_cached_prompt_tokens_total 100
`
	const after = `fak_gateway_kv_prefix_turns_total 13
fak_gateway_kv_prefix_prompt_tokens_total 19000
fak_gateway_kv_prefix_reused_tokens_total 8500
fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} 3
fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} 5
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 5
fak_gateway_inference_cached_prompt_tokens_total 150
`
	base, err := Parse("file://before.prom", before)
	if err != nil {
		t.Fatalf("Parse before: %v", err)
	}
	end, err := Parse("file://after.prom", after)
	if err != nil {
		t.Fatalf("Parse after: %v", err)
	}
	delta := end.Sub(base)
	if delta.KVPrefix.Turns != 3 || delta.KVPrefix.PromptTokens != 9000 || delta.KVPrefix.ReusedTokens != 4500 {
		t.Fatalf("delta kv = %+v, want 3 turns / 9000 prompt / 4500 reused", delta.KVPrefix)
	}
	if delta.KVPrefix.FrozenTurns != 1 || delta.KVPrefix.PartialTurns != 2 || delta.KVPrefix.ColdTurns != 0 {
		t.Fatalf("delta regimes = %+v, want frozen=1 partial=2 cold=0", delta.KVPrefix)
	}
	if delta.ProviderCacheReadTokens != 50 {
		t.Fatalf("delta provider cache_read = %d, want 50", delta.ProviderCacheReadTokens)
	}
	if delta.GatewayUptimeTurns != 13 {
		t.Fatalf("gateway uptime turns = %d, want end-scrape cumulative 13", delta.GatewayUptimeTurns)
	}
	if delta.WitnessWindow == nil || delta.WitnessWindow.StartScrape != "file://before.prom" || delta.WitnessWindow.EndScrape != "file://after.prom" {
		t.Fatalf("witness window = %+v, want before->after scrape labels", delta.WitnessWindow)
	}
	if delta.CacheValue.ReusedTokens != 4500 || delta.CacheValue.PromptTokens != 9000 {
		t.Fatalf("cache value = %+v, want recomputed from delta", delta.CacheValue)
	}
}

func TestRecordSubTreatsCounterResetAsFreshWindow(t *testing.T) {
	end := Record{KVPrefix: KVPrefixWitness{Turns: 2, PromptTokens: 700, ReusedTokens: 300}, ProviderCacheReadTokens: 9}
	base := Record{KVPrefix: KVPrefixWitness{Turns: 10, PromptTokens: 10000, ReusedTokens: 4000}, ProviderCacheReadTokens: 100}
	delta := end.Sub(base)
	if delta.KVPrefix.Turns != 2 || delta.KVPrefix.PromptTokens != 700 || delta.KVPrefix.ReusedTokens != 300 || delta.ProviderCacheReadTokens != 9 {
		t.Fatalf("reset delta = %+v provider=%d, want end counters as fresh run", delta.KVPrefix, delta.ProviderCacheReadTokens)
	}
}

func TestReuseRatioAndCacheBit(t *testing.T) {
	r, err := Parse("u", sampleMetrics)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got := r.KVPrefix.ReuseRatio(); got < 0.91 || got > 0.92 {
		t.Errorf("reuse ratio = %v, want ~0.9155 (15000/16384)", got)
	}
	if !r.CacheBit() {
		t.Error("CacheBit() = false, want true (reused 15000 > 0)")
	}
}

func TestReuseRatioZeroWhenNoTurns(t *testing.T) {
	var k KVPrefixWitness // all zero
	if got := k.ReuseRatio(); got != 0 {
		t.Errorf("ReuseRatio with 0 prompt tokens = %v, want 0 (no divide-by-zero)", got)
	}
}

func TestCacheDidNotBiteIsHonest(t *testing.T) {
	// An all-cold run: turns observed, but nothing reused. The record must report
	// CacheBit()==false rather than pretending the cache engaged.
	const allCold = `fak_gateway_kv_prefix_turns_total 3
fak_gateway_kv_prefix_prompt_tokens_total 9000
fak_gateway_kv_prefix_reused_tokens_total 0
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 3
`
	r, err := Parse("u", allCold)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if r.CacheBit() {
		t.Error("CacheBit() = true on an all-cold run, want false")
	}
}

func TestProvenanceLabelsSplitTrustClasses(t *testing.T) {
	r, err := Parse("u", sampleMetrics)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if r.Provenance["kv_prefix"] != Witnessed {
		t.Errorf("kv_prefix provenance = %q, want WITNESSED (fak's own RadixAttention)", r.Provenance["kv_prefix"])
	}
	if r.Provenance["provider_cache_read_tokens"] != Observed {
		t.Errorf("provider cache_read provenance = %q, want OBSERVED (provider-relayed)", r.Provenance["provider_cache_read_tokens"])
	}
	// The two must never be the same field/number: witnessed reuse is the kernel's,
	// observed cache_read is the provider's, and they are distinct in the record.
	if r.KVPrefix.ReusedTokens == r.ProviderCacheReadTokens {
		t.Error("witnessed reused == observed provider cache_read — the trust classes collapsed")
	}
}

func TestRecordRoundTripsJSON(t *testing.T) {
	r, err := Parse("http://box:8080/metrics", sampleMetrics)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Record
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.KVPrefix.ReusedTokens != r.KVPrefix.ReusedTokens || back.Provenance["kv_prefix"] != Witnessed {
		t.Errorf("round-trip lost the witnessed cache value or its provenance: %+v", back)
	}
}

func TestParseRejectsNonGatewayBody(t *testing.T) {
	if _, err := Parse("u", "# just comments\nsome_other_metric 5\n"); err == nil {
		t.Error("Parse accepted a body with no fak cache series, want error")
	}
}

// The #1066 honesty fence: the publishable cache-value view must frame the number
// as marginal-over-tuned-warm-KV and never surface the vs-naive multiple — even
// though this sample's 91.55% reuse tempts a ~11.8x (1/(1-0.9155)) re-prefill
// number.
func TestCacheValueIsWarmKVMarginalAndFencesVsNaive(t *testing.T) {
	r, err := Parse("u", sampleMetrics)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	cv := r.CacheValue
	if cv.PublishableValueFamily != WarmKVMarginalFamily {
		t.Errorf("publishable family = %q, want the marginal-over-warm-KV family", cv.PublishableValueFamily)
	}
	if cv.SingleSessionMarginalX != 1.0 {
		t.Errorf("single-session marginal = %vx, want 1.0 (a tuned warm-KV server earns the same single-trajectory turn reuse)", cv.SingleSessionMarginalX)
	}
	if !cv.VsNaiveMultipleExcluded {
		t.Error("VsNaiveMultipleExcluded = false, want true (the ~17.9-23.4x re-prefill band must not be published as a cache value)")
	}
	// The realized reuse is echoed as WITNESSED data, not dropped.
	if cv.ReusedTokens != 15000 || cv.ReuseRatio < 0.91 || cv.ReuseRatio > 0.92 {
		t.Errorf("cache value reuse = %d tok / %.4f ratio, want 15000 / ~0.9155", cv.ReusedTokens, cv.ReuseRatio)
	}
	if r.Provenance["cache_value"] != Witnessed {
		t.Errorf("cache_value provenance = %q, want WITNESSED", r.Provenance["cache_value"])
	}
	// The forbidden vs-naive multiple is the COMPUTED 1/(1-reuse) value — for this
	// sample 1/(1-0.9155) ≈ 11.8. It must never be surfaced as a number; the only
	// permitted mentions of the band are in the fence's own prose (the `note` and
	// the `vs_naive_multiple_excluded` flag), which name it precisely to forbid it.
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "11.8") {
		t.Errorf("cache-witness JSON leaked the computed vs-naive multiple (~11.8x): %s", b)
	}
}

// Even on an all-cold run (cache did not bite) the publishable view stays honest:
// the family and fence hold, and the realized reuse echoes 0 rather than inventing
// a speedup.
func TestCacheValueHonestWhenColdNoBite(t *testing.T) {
	const allCold = `fak_gateway_kv_prefix_turns_total 3
fak_gateway_kv_prefix_prompt_tokens_total 9000
fak_gateway_kv_prefix_reused_tokens_total 0
fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} 3
`
	r, err := Parse("u", allCold)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	cv := r.CacheValue
	if cv.ReusedTokens != 0 || cv.ReuseRatio != 0 {
		t.Errorf("cold-run cache value = %d tok / %v ratio, want 0 / 0", cv.ReusedTokens, cv.ReuseRatio)
	}
	if !cv.VsNaiveMultipleExcluded || cv.PublishableValueFamily != WarmKVMarginalFamily {
		t.Error("fence/family must hold on a cold run too")
	}
}
