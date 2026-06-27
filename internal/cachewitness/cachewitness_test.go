package cachewitness

import (
	"encoding/json"
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
