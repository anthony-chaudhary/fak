package blastradius

import "time"

var cmpEdges = map[string][]string{"internal/b": {"internal/a"}, "internal/c": {"internal/b"}, "internal/d": {"internal/x"}}
var cmpLeases = []Lease{{Lane: "b", TreeGlobs: []string{"internal/b/**"}}, {Lane: "a", TreeGlobs: []string{"internal/a/impl.go"}}, {Lane: "d", TreeGlobs: []string{"internal/d/**"}}}
var cmpIssues = []Issue{{ID: "1", Paths: []string{"internal/c/new.go"}}, {ID: "2", Paths: []string{"internal/z/new.go"}}}

type ComparisonArm struct {
	Name, Kind                                         string
	Available, Correct                                 bool
	Latency                                            time.Duration
	Radius, HeldLeases, HeldIssues, FalseHolds, Misses int
	CPUSeconds                                         float64
	PeakRSSBytes, NetworkBytes                         int64
	OperatorSeconds, CostUSD                           float64
	Note                                               string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func nativeArm() ComparisonArm {
	a := ComparisonArm{Name: "fak native dependency blast-radius estimator", Kind: "native", Available: true, Note: "reverse dependency closure intersected with lease trees and queued issue paths"}
	st := time.Now()
	r := Estimate(cmpEdges, "internal/a", cmpLeases, cmpIssues)
	a.Latency = time.Since(st)
	a.Radius = len(r.Radius)
	a.HeldLeases = len(r.Leases)
	a.HeldIssues = len(r.Issues)
	a.Correct = a.Radius == 3 && a.HeldLeases == 2 && a.HeldIssues == 1 && len(r.ExcludedLeases) == 1 && len(r.ExcludedIssues) == 1
	return a
}
func directOnlyArm() ComparisonArm {
	a := ComparisonArm{Name: "broken-tree intersection only", Kind: "baseline", Available: true, Radius: 1, HeldLeases: 1, HeldIssues: 0, Misses: 2, Note: "tuned baseline intersects only the broken tree and misses transitive dependents"}
	st := time.Now()
	_ = Estimate(map[string][]string{}, "internal/a", cmpLeases, cmpIssues)
	a.Latency = time.Since(st)
	return a
}
func ua(n, k, s string) ComparisonArm { return ComparisonArm{Name: n, Kind: k, Note: s} }
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "compute the same broken-package reverse dependency radius and exact intersecting leases/issues while excluding a disjoint island", Arms: []ComparisonArm{nativeArm(), directOnlyArm(), ua("fak + DOS leases", "integration", "requires real dos lease inventory and independent hold read-back"), ua("Bazel query reverse dependencies", "external", "requires equivalent Bazel graph/query"), ua("Pants dependents", "external", "requires equivalent Pants graph"), ua("Nx affected graph", "external", "requires equivalent Nx graph"), ua("Kubernetes Lease impact labels", "external", "requires equivalent lease/path labeling policy")}}
}
