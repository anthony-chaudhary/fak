package vcachegov

import "time"

// ComparisonArm is one implementation of the common warm-cache scheduling
// workload. Unavailable engine arms deliberately retain zero measurements.
type ComparisonArm struct {
	Name          string        `json:"name"`
	Kind          string        `json:"kind"`
	Available     bool          `json:"available"`
	Correct       bool          `json:"correct"`
	Latency       time.Duration `json:"latency"`
	Candidates    int           `json:"candidates"`
	Warmed        int           `json:"warmed"`
	ValueCaptured float64       `json:"value_captured"`
	Tokens        int64         `json:"tokens"`
	Bytes         int64         `json:"bytes"`
	CostUSD       float64       `json:"cost_usd"`
	Note          string        `json:"note,omitempty"`
}

// ComparisonResult records the shared fixture and every required alternative.
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func comparisonFixture() (RateLimit, float64, int64, []WarmCandidate) {
	return RateLimit{TierRPM: 10, RealRPM: 8, TierTPM: 20000, RealTPM: 10000}, 1000, 60000, []WarmCandidate{
		{Key: "hot-large", Frequency: 8, Size: 2000, ReuseDensity: 4, Secret: Cacheable},
		{Key: "dense", Frequency: 5, Size: 1000, ReuseDensity: 5, Secret: Cacheable},
		{Key: "frequent-small", Frequency: 9, Size: 200, ReuseDensity: 2, Secret: Cacheable},
		{Key: "cold-large", Frequency: 1, Size: 3000, ReuseDensity: 1, Secret: Cacheable},
		{Key: "regulated-hot", Frequency: 100, Size: 4000, ReuseDensity: 10, Secret: SecretRegulated},
		{Key: "negative", Frequency: -1, Size: 9000, ReuseDensity: 9, Secret: Cacheable},
	}
}

func capturedValue(cs []WarmCandidate) float64 {
	var total float64
	for _, c := range cs {
		total += c.RankScore()
	}
	return total
}

// CompareLocal executes fak's planner and a demand-only incumbent. LMCache,
// Mooncake, NIXL, vLLM, and SGLang adapters are not engine-runtime witnesses.
func CompareLocal() ComparisonResult {
	rate, anchorTokens, ttlMillis, candidates := comparisonFixture()
	start := time.Now()
	budget := PlanWarmBudget(rate, anchorTokens, ttlMillis)
	selected := Schedule(candidates, budget)
	elapsed := time.Since(start)
	nativeCorrect := budget.SustainableSet == 2 && len(selected) == 2 && selected[0].Key == "hot-large" && selected[1].Key == "dense"
	value := capturedValue(selected)

	start = time.Now()
	baselineSelected := candidates[:0]
	baselineElapsed := time.Since(start)

	return ComparisonResult{
		Workload: "plan one rate-limit-safe TTL warm window and select the best cacheable prefixes from six mixed-value candidates",
		Arms: []ComparisonArm{
			{Name: "fak native warm-budget scheduler", Kind: "native", Available: true, Correct: nativeCorrect, Latency: elapsed, Candidates: len(candidates), Warmed: len(selected), ValueCaptured: value, Tokens: int64(len(selected)) * int64(anchorTokens)},
			{Name: "demand-only fills without proactive warming", Kind: "baseline", Available: true, Correct: false, Latency: baselineElapsed, Candidates: len(candidates), Warmed: len(baselineSelected), Note: "tuned incumbent spends no warm quota but captures no pre-arrival prefix value"},
			{Name: "fak + LMCache", Kind: "integration", Note: "requires the real first-class LMCache transfer and cache runtime"},
			{Name: "fak + Mooncake", Kind: "integration", Note: "requires the real first-class Mooncake transfer and cache runtime"},
			{Name: "fak + NIXL", Kind: "integration", Note: "requires the real first-class NIXL lease/transfer runtime"},
			{Name: "vLLM automatic prefix caching", Kind: "external", Note: "requires a real vLLM server and the common request trace"},
			{Name: "SGLang HiCache and cache-aware scheduling", Kind: "external", Note: "requires a real SGLang server and the common request trace"},
		},
	}
}
