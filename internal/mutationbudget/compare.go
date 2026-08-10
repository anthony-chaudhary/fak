package mutationbudget

import "time"

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Calls     int           `json:"calls"`
	Held      bool          `json:"held"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes only fak and the direct-call baseline. Real clients and
// gateways remain unavailable until they consume the same captured rate-limit
// state; package presence and mocks are not external witnesses.
func CompareLocal() ComparisonResult {
	budget := Budget{Remaining: 12, Limit: 5000, ResetAtUnix: 1_700_003_600}
	plan := HourlyPlan{Closes: 8, Comments: 5, Fetches: 2}
	start := time.Now()
	estimate := EstimateHour(budget, plan, 5, 1_700_000_000)
	elapsed := time.Since(start)
	correct := !estimate.Allow && estimate.TotalCalls == 15 && estimate.MutationCalls == 13 && estimate.FetchCalls == 2
	return ComparisonResult{Workload: "price eight closes, five comments, and two fetches against twelve remaining calls and a five-call reserve", Arms: []ComparisonArm{
		{Name: "fak native mutation reserve", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Calls: estimate.TotalCalls, Held: !estimate.Allow},
		{Name: "direct API calls without reserve", Kind: "baseline", Available: true, Correct: false, Calls: estimate.TotalCalls, Note: "tuned no-feature baseline sends the burst and can consume the operator reserve"},
		{Name: "GitHub Octokit rate-limit handling", Kind: "external", Note: "requires a real Octokit client and independently captured rate-limit response"},
		{Name: "gh api rate-limit handling", Kind: "external", Note: "requires the real gh client and independently captured response headers"},
		{Name: "Envoy global rate limit", Kind: "external", Note: "requires a real Envoy rate-limit service configured for the same budget"},
	}}
}
