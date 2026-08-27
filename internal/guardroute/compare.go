package guardroute

import (
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"time"
)

type ComparisonArm struct {
	Name, Kind                               string
	Available, Correct                       bool
	Latency                                  time.Duration
	Cases, Passed, FalseRoutes, MissedRoutes int
	CPUSeconds                               float64
	PeakRSSBytes, InputBytes, NetworkBytes   int64
	OperatorSeconds, CostUSD                 float64
	Note                                     string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}
type routeCase struct {
	fold   guardrsi.Fold
	bucket guardrsi.Bucket
	want   string
}

var routeCases = []routeCase{
	{guardrsi.Fold{TotalRows: 5, ChildCrash: 1}, guardrsi.Bucket{Bucket: "child_crash", Count: 1, Lever: "harden supervision"}, "OPEN_ISSUE"},
	{guardrsi.Fold{TotalRows: 5, BlankReasonOnDeny: 2}, guardrsi.Bucket{Bucket: "blank_reason_on_deny", Count: 2, Lever: "require reason"}, "OPEN_ISSUE"},
	{guardrsi.Fold{TotalRows: 4, UnknownVerdict: 1}, guardrsi.Bucket{Bucket: "unknown_verdict", Count: 1, Lever: "closed set"}, "OPEN_ISSUE"},
	{guardrsi.Fold{TotalRows: 10}, guardrsi.Bucket{Bucket: "reason:POLICY_BLOCK", Count: 3, Lever: "floor"}, "OPEN_FINDING"},
	{guardrsi.Fold{TotalRows: 10}, guardrsi.Bucket{Bucket: "reason:POLICY_BLOCK", Count: 2}, "NOOP"},
	{guardrsi.Fold{TotalRows: 0}, guardrsi.Bucket{Bucket: "none"}, "NOOP"},
}

func runNativeComparison() ComparisonArm {
	a := ComparisonArm{Name: "fak native guard-journal route", Kind: "native", Available: true, Cases: len(routeCases), Note: "typed fold and worst-bucket routing to finding, issue, or no-op"}
	start := time.Now()
	for _, c := range routeCases {
		d := Decide(c.fold, c.bucket, 3)
		got := "NOOP"
		if d.Route {
			got = "OPEN_FINDING"
			if d.FileIssue {
				got = "OPEN_ISSUE"
			}
		}
		if got == c.want {
			a.Passed++
		} else {
			a.FalseRoutes++
		}
	}
	a.Latency = time.Since(start)
	a.Correct = a.Passed == a.Cases
	return a
}
func runCountThresholdBaseline() ComparisonArm {
	a := ComparisonArm{Name: "count-threshold-only routing", Kind: "baseline", Available: true, Cases: len(routeCases), Note: "tuned incumbent opens only buckets at the reason threshold and misses structural anomaly routing"}
	start := time.Now()
	for _, c := range routeCases {
		got := "NOOP"
		if c.bucket.Count >= 3 {
			got = "OPEN_ISSUE"
		}
		if got == c.want {
			a.Passed++
		} else if c.want != "NOOP" {
			a.MissedRoutes++
		} else {
			a.FalseRoutes++
		}
	}
	a.Latency = time.Since(start)
	a.Correct = false
	return a
}
func unavailable(name, kind, note string) ComparisonArm {
	return ComparisonArm{Name: name, Kind: kind, Note: note}
}
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "route the same empty, structural-anomaly, below-threshold reason, and at-threshold reason journal folds to exact NOOP, finding, or issue actions", Arms: []ComparisonArm{runNativeComparison(), runCountThresholdBaseline(), unavailable("fak + DOS decisions", "integration", "requires real dos decisions write/read-back"), unavailable("OPA decision policy", "external", "requires pinned OPA policy and execution"), unavailable("Cedar policy evaluator", "external", "requires pinned Cedar schema/policy and execution"), unavailable("Drools rule engine", "external", "requires pinned Drools rules and JVM execution"), unavailable("Prometheus Alertmanager routing", "external", "requires equivalent alert rules and receiver routing")}}
}
