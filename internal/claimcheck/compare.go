package claimcheck

import (
	"strings"
	"time"
)

type ComparisonArm struct {
	Name             string
	Kind             string
	Available        bool
	Correct          bool
	Latency          time.Duration
	Cases            int
	ExactVerdicts    int
	WrongNetTrue     int
	WrongStrawman    int
	WrongNotYet      int
	ReasonMismatches int
	CPUSeconds       float64
	PeakRSSBytes     int64
	InputBytes       int64
	ModelTokens      int64
	NetworkBytes     int64
	StorageBytes     int64
	OperatorSeconds  float64
	CostUSD          float64
	Note             string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func runNativeComparison(cases []FixtureCase) ComparisonArm {
	a := ComparisonArm{Name: "fak native net-true claim grader", Kind: "native", Available: true, Cases: len(cases)}
	start := time.Now()
	for _, tc := range cases {
		got := Grade(tc.Claim)
		if got.Verdict == tc.Expect {
			a.ExactVerdicts++
		} else {
			countWrong(&a, got.Verdict)
		}
		if got.Verdict != NetTrue && len(got.Missing) == 0 {
			a.ReasonMismatches++
		}
	}
	a.Latency = time.Since(start)
	a.Correct = a.ExactVerdicts == len(cases) && a.ReasonMismatches == 0
	a.Note = "six-question deterministic baseline/net/scope/provenance/witness/realization gate"
	return a
}

func runWitnessOnlyComparison(cases []FixtureCase) ComparisonArm {
	a := ComparisonArm{Name: "accept claim when any witness exists", Kind: "baseline", Available: true, Cases: len(cases)}
	start := time.Now()
	for _, tc := range cases {
		got := NotYet
		if strings.TrimSpace(tc.Claim.Witness) != "" {
			got = NetTrue
		}
		if got == tc.Expect {
			a.ExactVerdicts++
		} else {
			countWrong(&a, got)
		}
		if got != NetTrue {
			a.ReasonMismatches++
		}
	}
	a.Latency = time.Since(start)
	a.Correct = a.ExactVerdicts == len(cases) && a.ReasonMismatches == 0
	a.Note = "tuned no-grader baseline checks only whether a witness string is present"
	return a
}

func countWrong(a *ComparisonArm, got Verdict) {
	switch got {
	case NetTrue:
		a.WrongNetTrue++
	case Strawman:
		a.WrongStrawman++
	default:
		a.WrongNotYet++
	}
}

func CompareLocal() ComparisonResult {
	cases := Fixture()
	return ComparisonResult{Workload: "grade the same nine labeled net-true, strawman, and not-yet claims with exact verdict and failing-question oracle", Arms: []ComparisonArm{
		runNativeComparison(cases), runWitnessOnlyComparison(cases),
		{Name: "fak + Prometheus", Kind: "integration", Note: "requires real metric evidence ingestion and policy read-back"},
		{Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires real telemetry evidence and evaluator read-back"},
		{Name: "OPA/Rego", Kind: "external", Note: "requires pinned OPA and real Rego evaluation"},
		{Name: "OpenAI Evals graders", Kind: "external", Note: "requires real grader execution and token accounting"},
		{Name: "LangSmith evaluators", Kind: "external", Note: "requires real evaluation boundary"},
		{Name: "Braintrust scorers", Kind: "external", Note: "requires real scorer execution"},
		{Name: "DeepEval metrics", Kind: "external", Note: "requires pinned DeepEval execution"},
	}}
}
