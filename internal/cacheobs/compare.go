package cacheobs

import (
	"time"
)

// ComparisonArm is one independently reportable cache-observability arm.
type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Events    int           `json:"events"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

// ComparisonResult observes one fixed cache-event trace.
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes fak and a no-telemetry baseline. Prometheus,
// OpenTelemetry, and Datadog remain unavailable until their real collectors run.
func CompareLocal() ComparisonResult {
	const events = 1_000
	obs := New()
	start := time.Now()
	for i := 0; i < events; i++ {
		obs.Observe(128, 96)
	}
	elapsed := time.Since(start)
	snapshot := obs.Snapshot()
	return ComparisonResult{
		Workload: "record 1,000 prompt-prefix observations and aggregate prompt, reused, cacheable, eligible, and miss counters",
		Arms: []ComparisonArm{
			{Name: "fak native cache observer", Kind: "native", Available: true, Correct: snapshot.Turns == events && snapshot.PromptTokens == events*128 && snapshot.ReusedTokens == events*96, Latency: elapsed, Events: events},
			{Name: "no telemetry", Kind: "baseline", Available: true, Correct: false, Note: "zero-work tuned baseline records no cache evidence"},
			{Name: "Prometheus client", Kind: "external", Note: "requires a real scrape and equivalent label/cardinality policy"},
			{Name: "OpenTelemetry metrics", Kind: "external", Note: "requires a real SDK/exporter/collector pipeline"},
			{Name: "Datadog DogStatsD", Kind: "external", Note: "requires a real agent intake and matching aggregation policy"},
		},
	}
}
