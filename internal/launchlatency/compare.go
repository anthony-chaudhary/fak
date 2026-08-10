package launchlatency

import "time"

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Samples   int           `json:"samples"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes fak and a raw-event baseline only. Telemetry backends
// remain unavailable until they ingest and query this exact ledger; adapters
// and mocks do not count as external witnesses.
func CompareLocal() ComparisonResult {
	launches := []Launch{{DispatchSec: 10, HeartbeatSec: 11}, {DispatchSec: 10, HeartbeatSec: 12}, {DispatchSec: 10, HeartbeatSec: 15}, {DispatchSec: 10, HeartbeatSec: 20}, {DispatchSec: 10, HeartbeatSec: 40}, {DispatchSec: 20, HeartbeatSec: 19}}
	buckets := []float64{1, 2, 5, 10, 30}
	start := time.Now()
	hist := Histogram(launches, buckets)
	p50, p95 := P50P95(launches)
	elapsed := time.Since(start)
	correct := len(hist) == 6 && hist[0].Count == 1 && hist[4].Count == 1 && hist[5].Count == 1 && p50 == 2 && p95 == 30 && Negatives(launches) == 1
	return ComparisonResult{Workload: "fold six dispatch-to-heartbeat samples across fixed buckets, including one negative clock-skew sample", Arms: []ComparisonArm{
		{Name: "fak native launch-latency summary", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Samples: len(launches)},
		{Name: "raw launch events without summary", Kind: "baseline", Available: true, Correct: false, Samples: len(launches), Note: "tuned no-feature baseline retains events but produces no buckets or percentiles"},
		{Name: "Prometheus histogram", Kind: "external", Note: "requires a real Prometheus scrape, storage, and query"},
		{Name: "OpenTelemetry metrics", Kind: "external", Note: "requires a real SDK, collector, backend, and query"},
		{Name: "Datadog distribution metric", Kind: "external", Note: "requires a real agent/backend ingestion and query"},
	}}
}
