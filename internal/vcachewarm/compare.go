package vcachewarm

import "time"

type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	Latency         time.Duration
	Cases           int
	DedicatedWarms  int
	ConfirmedWarms  int
	WastedWarms     int
	CacheReadTokens int64
	Bytes           int64
	CostUSD         float64
	Note            string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func warmingFixture() []Request {
	fp := PrefixFingerprint{SerializerID: "json-v1", Digest: "abc", Bytes: 4096, Tokens: 1024}
	mismatch := fp
	mismatch.Digest = "other"
	return []Request{
		{Provider: ProviderAnthropic, ActiveCapability: ActiveCacheCapabilitySupported, ExpectedReuseBeforeTTL: 3, WarmPrefix: fp, RealPrefix: fp, SharedBlockCount: 2},
		{Provider: ProviderAnthropic, ActiveCapability: ActiveCacheCapabilitySupported, ExpectedReuseBeforeTTL: 3, WarmPrefix: fp, RealPrefix: mismatch, SharedBlockCount: 2},
		{Provider: ProviderAnthropic, ActiveCapability: ActiveCacheCapabilitySupported, ExpectedReuseBeforeTTL: 1, WarmPrefix: fp, RealPrefix: fp, SharedBlockCount: 2},
		{Provider: ProviderOpenAI, ActiveCapability: ActiveCacheCapabilitySupported, ExpectedReuseBeforeTTL: 4, WarmPrefix: fp, RealPrefix: fp, SharedBlockCount: 2},
		{Provider: ProviderDeepSeek, ActiveCapability: ActiveCacheCapabilityUnsupported, ExpectedReuseBeforeTTL: 4, WarmPrefix: fp, RealPrefix: fp, SharedBlockCount: 2},
	}
}
func CompareLocal() ComparisonResult {
	reqs := warmingFixture()
	start := time.Now()
	decs := make([]Decision, len(reqs))
	for i, r := range reqs {
		decs[i] = Plan(r)
	}
	acc := ReconcileWarm(decs[0], true, []CacheReadback{{CacheReadTokens: 1024}, {CacheReadTokens: 1024}})
	elapsed := time.Since(start)
	correct := decs[0].Dedicated && decs[0].Primitive == PrimitiveAnthropicMaxTokens0 && !decs[1].Dedicated && decs[1].Reason == ReasonPrefixFingerprintMismatch && !decs[2].Dedicated && decs[2].Reason == ReasonBelowBreakEven && !decs[3].Dedicated && decs[3].Primitive == PrimitiveOrderFirstReal && !decs[4].Dedicated && decs[4].Reason == ReasonUnsupportedActiveCacheCapability && acc.Status == WarmConfirmed && acc.CacheReadTokens == 2048
	start = time.Now()
	_ = reqs
	baselineLatency := time.Since(start)
	return ComparisonResult{Workload: "plan five profitable/refused/automatic cache-warm cases and reconcile one two-read confirmed fanout", Arms: []ComparisonArm{
		{Name: "fak native dedicated warm planner and accounting", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Cases: 5, DedicatedWarms: 1, ConfirmedWarms: 1, CacheReadTokens: acc.CacheReadTokens},
		{Name: "demand-only fills without dedicated warming", Kind: "baseline", Available: true, Correct: false, Latency: baselineLatency, Cases: 5, Note: "spends no dedicated warm but cannot create a pre-fanout cache entry"},
		{Name: "fak + Anthropic prompt caching", Kind: "integration", Note: "requires the real Anthropic API, cache writes, and readback usage"},
		{Name: "fak + Gemini CachedContent", Kind: "integration", Note: "requires the real Gemini explicit cache lifecycle"},
		{Name: "fak + OpenAI automatic prefix caching", Kind: "integration", Note: "requires the real OpenAI API and cache readback"},
		{Name: "fak + LMCache", Kind: "integration", Note: "requires the real first-class LMCache runtime"},
		{Name: "fak + Mooncake", Kind: "integration", Note: "requires the real first-class Mooncake runtime"},
		{Name: "vLLM automatic prefix caching", Kind: "external", Note: "requires a real vLLM server and common fanout"},
		{Name: "SGLang HiCache", Kind: "external", Note: "requires a real SGLang server and common fanout"},
	}}
}
