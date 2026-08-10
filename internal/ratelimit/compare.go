package ratelimit

import (
	"context"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ComparisonArm is one independently reportable rate-limiter arm. Unavailable
// alternatives deliberately carry no measurements.
type ComparisonArm struct {
	Name        string        `json:"name"`
	Kind        string        `json:"kind"`
	Integration bool          `json:"integration"`
	Available   bool          `json:"available"`
	Correct     bool          `json:"correct"`
	Latency     time.Duration `json:"latency"`
	Admitted    int           `json:"admitted"`
	Denied      int           `json:"denied"`
	Bytes       int64         `json:"bytes"`
	CostUSD     float64       `json:"cost_usd"`
	Note        string        `json:"note,omitempty"`
}

// ComparisonResult reports one fixed-window-equivalent request trace.
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal runs only fak and the no-limiter baseline. Envoy, Kong, and
// Redis-cell require their real processes and remain explicitly unavailable.
func CompareLocal() ComparisonResult {
	const calls = 1_000
	const limit = 750
	result := ComparisonResult{
		Workload: "1,000 sequential calls for one trace with a 750-call cap",
		Arms: []ComparisonArm{
			{Name: "no limiter", Kind: "baseline", Available: true, Admitted: calls, Correct: false, Note: "zero-work tuned baseline admits 250 over-budget calls"},
			{Name: "Envoy local rate limit", Kind: "external", Note: "requires a real Envoy local-rate-limit deployment"},
			{Name: "Kong rate limiting", Kind: "external", Note: "requires a real Kong gateway and backing policy"},
			{Name: "Redis-cell", Kind: "external", Note: "requires a real Redis server with redis-cell"},
		},
	}
	limiter := New()
	limiter.SetLimit(Limit{MaxCalls: limit}, KeyPerTrace)
	call := &abi.ToolCall{TraceID: "comparison-trace", Tool: "search"}
	var admitted, denied int
	start := time.Now()
	for i := 0; i < calls; i++ {
		verdict := limiter.Adjudicate(context.Background(), call)
		switch verdict.Kind {
		case abi.VerdictDefer:
			admitted++
		case abi.VerdictDeny:
			if verdict.Reason == abi.ReasonRateLimited {
				denied++
			}
		}
	}
	elapsed := time.Since(start)
	result.Arms = append([]ComparisonArm{{
		Name: "fak native rate limiter", Kind: "native", Available: true,
		Correct: admitted == limit && denied == calls-limit, Latency: elapsed,
		Admitted: admitted, Denied: denied,
		Note: "in-process call-count fixture; not a distributed or wall-clock-window witness",
	}}, result.Arms...)
	return result
}
