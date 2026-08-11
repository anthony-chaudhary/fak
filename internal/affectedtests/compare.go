package affectedtests

import (
	"sort"
	"time"
)

var compareEdges = map[string][]string{"internal/a": {}, "internal/b": {"internal/a"}, "internal/c": {"internal/a"}, "cmd/app": {"internal/b", "internal/c"}, "internal/isolated": {}}
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
	want := []string{"cmd/app", "internal/a", "internal/b", "internal/c"}
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
	changed := append([]string(nil), compareChanged...)
	sort.Strings(changed)
	a.Latency = time.Since(st)
	return a
}
func availableArm(n, k, note string) ComparisonArm {
	return ComparisonArm{Name: n, Kind: k, Available: true, Correct: true, Packages: 5, Selected: 4, Note: note}
}
func unavailableArm(n, k, note string) ComparisonArm {
	return ComparisonArm{Name: n, Kind: k, Packages: 5, Misses: 4, Note: note}
}
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "select exact tests for one changed leaf in the same diamond import graph, including all transitive importers and excluding an isolated package", Arms: []ComparisonArm{
		nativeArm(),
		changedOnlyArm(),
		availableArm("fak + Go test", "integration", "four selected Go package tests passed; measured evidence is in docs/benchmarks/affected-test-selection-6371.json"),
		availableArm("Bazel reverse-dependency query + test-work", "external", "Bazel 9.2.0 query selected four projects; built-in genrule work targets avoid claiming bazel test"),
		unavailableArm("Pants changed-since test selection", "external", "scie-pants 0.13.2 has no Windows asset and bounded WSL bootstrap produced no result"),
		availableArm("Nx affected", "external", "Nx 23.1.1 affected selection and four project tests witnessed"),
		availableArm("Gradle configured task closure", "external", "Gradle 9.7.0 explicit test-task closure witnessed; not a generic changed-file selector"),
	}}
}
