package computeadmit

import (
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"time"
)

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Decisions int           `json:"decisions"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func comparisonFixture() (Taxonomy, []Lease, []Request) {
	tax := Taxonomy{Space: map[string]Space{ClassDevice: {Lo: 0, Hi: 7}}}
	live := []Lease{{ID: "gpu0", Holder: "worker-a", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0-1", Mode: "exclusive"}}}
	requests := []Request{
		{Actor: "collision", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "1-2", Mode: "exclusive"}},
		{Actor: "disjoint", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "2-3", Mode: "exclusive"}},
		{Actor: "outside", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "8", Mode: "exclusive"}},
		{Actor: "other-class", Claim: dispatchorder.ComputeClaim{Class: ClassKVTier, Range: "0", Mode: "exclusive"}},
	}
	return tax, live, requests
}

// CompareLocal executes fak and an unbounded-dispatch baseline only. Real
// schedulers stay unavailable until they enforce this exact region inventory
// and taxonomy; configuration adapters and mocks are not scheduler witnesses.
func CompareLocal() ComparisonResult {
	tax, live, requests := comparisonFixture()
	start := time.Now()
	got := make([]Decision, len(requests))
	for i, req := range requests {
		got[i] = Decide(req, live, tax)
	}
	elapsed := time.Since(start)
	correct := !got[0].Admit && got[0].Reason == ReasonCollisionRisk && got[1].Admit && !got[2].Admit && got[2].Reason == ReasonPolicyBlock && got[3].Admit
	return ComparisonResult{Workload: "adjudicate overlapping, disjoint, out-of-taxonomy, and different-class compute-region claims against one live exclusive lease", Arms: []ComparisonArm{
		{Name: "fak native compute-region admission", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Decisions: len(got)},
		{Name: "dispatch without region admission", Kind: "baseline", Available: true, Correct: false, Decisions: len(requests), Note: "no-feature baseline admits collision and out-of-taxonomy claims"},
		{Name: "Kubernetes scheduler", Kind: "external", Note: "requires a real cluster with equivalent device regions and exclusivity"},
		{Name: "Slurm scheduler", Kind: "external", Note: "requires a real controller with equivalent generic-resource regions"},
		{Name: "Ray scheduler", Kind: "external", Note: "requires a real cluster with equivalent custom resource regions"},
		{Name: "AWS Batch", Kind: "external", Note: "requires real queues and environments with equivalent constraints"},
	}}
}
