package attemptbudget

import "time"

// ComparisonArm is one independently reportable retry-budget arm.
type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Attempts  int           `json:"attempts"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes fak and an unlimited-retry baseline. Envoy, gRPC, and
// AWS SDK alternatives remain unavailable until their real runtimes execute.
func CompareLocal() ComparisonResult {
	attempts := []Attempt{{FailureClass: "schema_mismatch", AtUnix: 1}, {FailureClass: "schema_mismatch", AtUnix: 2}, {FailureClass: "schema_mismatch", AtUnix: 3}}
	start := time.Now()
	decision := Decide(Input{IssueID: "same-call", Budget: 3, Attempts: attempts})
	elapsed := time.Since(start)
	return ComparisonResult{
		Workload: "classify three same-operation timeout failures against a three-attempt ceiling",
		Arms: []ComparisonArm{
			{Name: "fak native attempt budget", Kind: "native", Available: true, Correct: decision.Verdict == VerdictStructuralBlock && decision.Status == StatusHeld && decision.AttemptCount == 3, Latency: elapsed, Attempts: decision.AttemptCount},
			{Name: "unlimited retries", Kind: "baseline", Available: true, Correct: false, Attempts: 4, Note: "zero-policy baseline schedules another retry after the ceiling"},
			{Name: "Envoy retry budget", Kind: "external", Note: "requires real Envoy retry policy and upstream failures"},
			{Name: "gRPC retry policy", Kind: "external", Note: "requires a real gRPC client/service and service config"},
			{Name: "AWS SDK adaptive retry", Kind: "external", Note: "requires a real SDK client and throttling/failure trace"},
		},
	}
}
