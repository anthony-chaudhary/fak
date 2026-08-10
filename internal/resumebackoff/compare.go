package resumebackoff

import "time"

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Delay     time.Duration `json:"delay"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes fak and an immediate-resume baseline. Kubernetes,
// systemd, and AWS Step Functions remain unavailable until their schedulers run.
func CompareLocal() ComparisonResult {
	now := time.Unix(10_000, 0)
	history := []Event{{Session: "session", Signature: "same-failure", At: now.Add(-90 * time.Second)}, {Session: "session", Signature: "same-failure", At: now.Add(-30 * time.Second)}}
	start := time.Now()
	decision := Decide(Input{Session: "session", Signature: "same-failure", Now: now, History: history, Base: time.Minute, Ceiling: time.Hour})
	elapsed := time.Since(start)
	return ComparisonResult{
		Workload: "schedule a third resume for one session after two repeated same-signature failures",
		Arms: []ComparisonArm{
			{Name: "fak native resume backoff", Kind: "native", Available: true, Correct: !decision.Eligible && decision.Reason == ReasonBackoff && decision.Delay == 2*time.Minute, Latency: elapsed, Delay: decision.Delay},
			{Name: "immediate resume", Kind: "baseline", Available: true, Correct: false, Note: "zero-delay baseline re-enters the same failure immediately"},
			{Name: "Kubernetes CrashLoopBackOff", Kind: "external", Note: "requires a real pod restart trace"},
			{Name: "systemd RestartSec", Kind: "external", Note: "requires a real service restart trace"},
			{Name: "AWS Step Functions retry", Kind: "external", Note: "requires a real state-machine execution"},
		},
	}
}
