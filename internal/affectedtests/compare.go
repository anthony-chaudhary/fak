package affectedtests

import (
	"sort"
	"time"
)

var compareEdges = map[string][]string{"internal/a": {}, "internal/b": {"internal/a"}, "internal/c": {"internal/a"}, "cmd/fak": {"internal/b", "internal/c"}, "internal/isolated": {}}
var compareChanged = []string{"internal/a"}

type ComparisonArm struct {
	Name, Kind                                string
	Available, Correct                        bool
	Latency                                   time.Duration
	Packages, Selected, FalseIncludes, Misses int
	CPUSeconds                                float64
	PeakRSSBytes, InputBytes, NetworkBytes    int64
	OperatorSeconds, CostUSD                  float64
	Note                                      string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func nativeArm() ComparisonArm {
	a := ComparisonArm{Name: "fak native reverse-dependency affected-test selection", Kind: "native", Available: true, Packages: 5, Note: "changed package plus complete transitive importer closure, cycle-safe and sorted"}
	st := time.Now()
	got := Select(compareEdges, compareChanged)
	a.Latency = time.Since(st)
	a.Selected = len(got)
	want := []string{"cmd/fak", "internal/a", "internal/b", "internal/c"}
	for i := range want {
		if i >= len(got) || got[i] != want[i] {
			a.Misses++
		}
	}
	a.Correct = a.Misses == 0 && len(got) == len(want)
	return a
}
func changedOnlyArm() ComparisonArm {
	a := ComparisonArm{Name: "changed-package tests only", Kind: "baseline", Available: true, Packages: 5, Selected: 1, Misses: 3, Note: "tuned baseline tests changed package but omits transitive importers"}
	st := time.Now()
	_ = append([]string(nil), compareChanged...)
	sort.Strings(compareChanged)
	a.Latency = time.Since(st)
	return a
}
func ua(n, k, s string) ComparisonArm { return ComparisonArm{Name: n, Kind: k, Note: s} }
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "select exact tests for one changed leaf in the same diamond import graph, including all transitive importers and excluding an isolated package", Arms: []ComparisonArm{nativeArm(), changedOnlyArm(), ua("fak + Go test", "integration", "requires real selected-package go test witness"), ua("Bazel test selection", "external", "requires equivalent Bazel dependency graph and query"), ua("Pants changed-since test selection", "external", "requires equivalent Pants graph"), ua("Nx affected", "external", "requires equivalent project graph"), ua("Gradle test impact analysis", "external", "requires equivalent Gradle project")}}
}
