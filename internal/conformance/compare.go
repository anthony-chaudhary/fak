package conformance

import (
	"encoding/json"
	"time"
)

type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	Latency         time.Duration
	Checks          int
	PassedChecks    int
	MissedChecks    int
	FalseFailures   int
	MutationCases   int
	MutationsCaught int
	ReasonErrors    int
	CPUSeconds      float64
	PeakRSSBytes    int64
	InputBytes      int64
	NetworkBytes    int64
	OperatorSeconds float64
	CostUSD         float64
	Note            string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func runNativeComparison() ComparisonArm {
	a := ComparisonArm{Name: "fak native compiled conformance suite", Kind: "native", Available: true, Checks: 2, Note: "compiled ABI matrix plus real adjudicator verdict matrix against embedded contracts"}
	start := time.Now()
	r := Run()
	a.Latency = time.Since(start)
	for _, c := range r.Checks {
		if c.Pass {
			a.PassedChecks++
		} else {
			a.FalseFailures++
		}
		a.InputBytes += int64(len(c.Detail))
		for _, f := range c.Failures {
			a.InputBytes += int64(len(f))
		}
	}
	a.Correct = r.Pass && a.PassedChecks == 2 && a.FalseFailures == 0
	return a
}
func runSchemaOnlyComparison() ComparisonArm {
	a := ComparisonArm{Name: "embedded JSON and schema equality only", Kind: "baseline", Available: true, Checks: 2, Note: "tuned no-execution baseline validates embedded JSON syntax and ABI golden presence but never calls the adjudicator"}
	start := time.Now()
	var abiDoc, policyDoc any
	abiOK := json.Unmarshal(abiGolden, &abiDoc) == nil
	policyOK := json.Unmarshal(dogfoodPolicy, &policyDoc) == nil
	a.Latency = time.Since(start)
	if abiOK {
		a.PassedChecks++
	} else {
		a.FalseFailures++
	}
	if policyOK {
		a.MissedChecks = 1
	} else {
		a.FalseFailures++
	}
	a.InputBytes = int64(len(abiGolden) + len(dogfoodPolicy))
	a.Correct = a.PassedChecks == 2 && a.MissedChecks == 0 && a.FalseFailures == 0
	return a
}
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "verify the compiled ABI freeze and execute the embedded adjudication verdict matrix with exact check coverage", Arms: []ComparisonArm{runNativeComparison(), runSchemaOnlyComparison(), {Name: "OPA test", Kind: "external", Note: "requires pinned OPA tests over equivalent policy"}, {Name: "Conftest", Kind: "external", Note: "requires pinned Conftest execution"}, {Name: "OpenAPI and JSON Schema contract tests", Kind: "external", Note: "requires real schema validation plus executable verdict oracle"}, {Name: "Pact", Kind: "external", Note: "requires real provider/consumer contract execution"}, {Name: "Cedar policy validator and tests", Kind: "external", Note: "requires real Cedar schema/policy tests"}}}
}
