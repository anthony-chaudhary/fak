package cachesweep

import "time"

type ComparisonArm struct {
	Name        string        `json:"name"`
	Kind        string        `json:"kind"`
	Available   bool          `json:"available"`
	Correct     bool          `json:"correct"`
	Latency     time.Duration `json:"latency"`
	Accesses    int           `json:"accesses"`
	SavedTokens int64         `json:"saved_tokens"`
	Bytes       int64         `json:"bytes"`
	CostUSD     float64       `json:"cost_usd"`
	Note        string        `json:"note,omitempty"`
}
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func comparisonTrace() Trace {
	return Trace{Accesses: []Access{{Tokens: []int{1, 2, 3, 4}, TimeNs: 1}, {Tokens: []int{1, 2, 3, 5}, TimeNs: 2}, {Tokens: []int{1, 2, 3, 4}, TimeNs: 3}, {Tokens: []int{8, 9}, TimeNs: 4}, {Tokens: []int{1, 2, 3, 5}, TimeNs: 5}}}
}

// CompareLocal executes fak and a no-cache baseline only. Real simulators and
// caches remain unavailable until they consume this exact trace and cost model;
// adapters and in-memory mocks are not external cache witnesses.
func CompareLocal() ComparisonResult {
	trace := comparisonTrace()
	start := time.Now()
	got := Sweep(trace, Options{Budgets: []int{1, 2, 4, 8}, WriteDelayNs: 1, KneeFraction: DefaultKneeFraction})
	elapsed := time.Since(start)
	correct := len(got.Curve) == 4 && got.Accesses == 5 && got.Ceiling.ReusedTokens > 0 && got.KneeReached
	return ComparisonResult{Workload: "sweep four cache budgets over five exact-prefix/divergent-child accesses with fixed timestamps and write delay", Arms: []ComparisonArm{
		{Name: "fak native prefix-cache sweep", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Accesses: got.Accesses, SavedTokens: got.Ceiling.ReusedTokens},
		{Name: "no prefix cache", Kind: "baseline", Available: true, Correct: false, Accesses: got.Accesses, Note: "zero-cache baseline saves no tokens and cannot identify an ROI knee"},
		{Name: "libCacheSim", Kind: "external", Note: "requires the real simulator with an equivalent variable-size prefix trace"},
		{Name: "Caffeine simulator", Kind: "external", Note: "requires the real simulator and equivalent weighted entries"},
		{Name: "Redis or Valkey maxmemory policies", Kind: "external", Note: "requires a real server with equivalent key sizes and eviction policy"},
	}}
}
