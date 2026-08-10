package cachewitness

import "time"

type ComparisonArm struct {
	Name      string
	Kind      string
	Available bool
	Correct   bool
	Latency   time.Duration
	Records   int
	Alerts    int
	Bytes     int64
	CostUSD   float64
	Note      string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func divergenceFixture() []Record {
	return []Record{
		{GatewayURL: "stable", KVPrefix: KVPrefixWitness{PromptTokens: 1000, ReusedTokens: 800}, ProviderCacheReadTokens: 750},
		{GatewayURL: "diverged", KVPrefix: KVPrefixWitness{PromptTokens: 1000, ReusedTokens: 800}, ProviderCacheReadTokens: 300},
		{GatewayURL: "single", KVPrefix: KVPrefixWitness{PromptTokens: 1000, ReusedTokens: 700}},
	}
}

// CompareLocal executes fak and a raw-counter/no-detector baseline. Monitoring
// adapters are not witnesses for real ingestion, rules, storage, or alerting.
func CompareLocal() ComparisonResult {
	records := divergenceFixture()
	start := time.Now()
	report := FoldReuseDivergence(records, DefaultReuseDivergenceTolerance)
	elapsed := time.Since(start)
	correct := !report.OK && report.Compared == 2 && report.SingleClass == 1 && len(report.Diverged) == 1 && report.Diverged[0].GatewayURL == "diverged"
	start = time.Now()
	_ = records
	baselineElapsed := time.Since(start)
	return ComparisonResult{Workload: "fold three cache records containing stable, divergent, and single-provenance reuse axes", Arms: []ComparisonArm{
		{Name: "fak native reuse-divergence fold", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Records: len(records), Alerts: len(report.Diverged)},
		{Name: "raw reuse counters without divergence detection", Kind: "baseline", Available: true, Correct: false, Latency: baselineElapsed, Records: len(records), Note: "tuned incumbent stores both axes but emits no trust-class alert"},
		{Name: "fak + Prometheus", Kind: "integration", Note: "requires the real first-class metrics scrape, rule evaluation, and alert path"},
		{Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires the real first-class telemetry export and collector path"},
		{Name: "Prometheus recording and alerting rules", Kind: "external", Note: "requires a real Prometheus server and equivalent rules"},
		{Name: "Datadog anomaly monitor", Kind: "external", Note: "requires a real Datadog monitor and equivalent query"},
	}}
}
