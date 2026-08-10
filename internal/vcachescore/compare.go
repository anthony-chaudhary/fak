package vcachescore

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachewarm"
)

type ReadinessComparisonArm struct {
	Name             string
	Kind             string
	Available        bool
	Correct          bool
	Latency          time.Duration
	Cases            int
	TrueReady        int
	TrueBlocked      int
	FalseReady       int
	FalseBlocked     int
	ReasonMismatches int
	CPUSeconds       float64
	PeakRSSBytes     int64
	InputBytes       int64
	NetworkBytes     int64
	StorageBytes     int64
	OperatorSeconds  float64
	CostUSD          float64
	Note             string
}

type ReadinessComparisonResult struct {
	Workload string
	Arms     []ReadinessComparisonArm
}

type readinessCase struct {
	name       string
	report     Report
	wantReady  bool
	wantReason string
}

func readinessComparisonInput() Input {
	in := DefaultInput()
	in.TelemetryRows = []vcachegov.TelemetryRow{
		{InputTokens: 10098, CacheCreationInputTokens: 59400, Ephemeral1hInputTokens: 59400},
		{InputTokens: 10065, CacheCreationInputTokens: 15411, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15411},
		{InputTokens: 10065, CacheCreationInputTokens: 15410, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15410},
		{InputTokens: 10065, CacheCreationInputTokens: 15424, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15424},
	}
	in.Prediction = vcachecal.PredictionError{Total: 10, TrueWarm: 8, TrueCold: 2}
	in.AgenticActivation = AgenticActivationInput{KernelKVEvents: 1, ContextEvents: 1, ExternalEngineEvents: 1}
	in.KernelKV = PlaneEvidenceInput{Available: true, Provenance: "WITNESSED", BaselineTokenEquiv: 1000, SavedTokenEquiv: 800, CostTokenEquiv: 200, Reason: "kernel KV witness"}
	in.Context = PlaneEvidenceInput{Available: true, Provenance: "WITNESSED", BaselineTokenEquiv: 1000, SavedTokenEquiv: 750, CostTokenEquiv: 250, Reason: "context witness"}
	in.ExternalEngine = PlaneEvidenceInput{Available: true, Provenance: "OBSERVED", BaselineTokenEquiv: 1000, SavedTokenEquiv: 600, CostTokenEquiv: 400, HitRate: .6, Reason: "external observation"}
	return in
}

func readinessComparisonCases() []readinessCase {
	readyInput := readinessComparisonInput()
	ready := Score(readyInput)

	providerInput := DefaultInput()
	providerInput.TelemetryRows = readyInput.TelemetryRows
	providerOnly := Score(providerInput)

	provenanceInput := readinessComparisonInput()
	provenanceInput.KernelKV.Provenance = "OBSERVED"
	badProvenance := Score(provenanceInput)

	unsupportedInput := readinessComparisonInput()
	unsupportedInput.ExternalEngine.Reason = string(vcachewarm.ReasonUnsupportedActiveCacheCapability)
	unsupported := Score(unsupportedInput)

	coldFailure := ready
	coldFailure.ColdPathCorrect = false
	coldFailure.DefaultUsefulness.ColdPathCorrect = false

	return []readinessCase{
		{name: "fully witnessed ready", report: ready, wantReady: true},
		{name: "provider only", report: providerOnly, wantReason: "default_usefulness verdict"},
		{name: "wrong plane provenance", report: badProvenance, wantReason: "kernel_witnessed provenance"},
		{name: "unsupported active path", report: unsupported, wantReason: "unsupported active-cache"},
		{name: "cold path failure", report: coldFailure, wantReason: "cold-path correctness"},
	}
}

func scoreThresholdReady(rep Report) bool {
	return rep.DefaultUsefulness.Score >= 70
}

func reasonContains(reasons []string, needle string) bool {
	if needle == "" {
		return true
	}
	for _, reason := range reasons {
		if strings.Contains(reason, needle) {
			return true
		}
	}
	return false
}

func runNativeReadiness(cases []readinessCase) ReadinessComparisonArm {
	arm := ReadinessComparisonArm{Name: "fak native default-cache readiness gate", Kind: "native", Available: true, Cases: len(cases)}
	start := time.Now()
	for _, tc := range cases {
		got := DefaultReadiness(tc.report)
		if got.OK == tc.wantReady {
			if got.OK {
				arm.TrueReady++
			} else {
				arm.TrueBlocked++
			}
		} else if got.OK {
			arm.FalseReady++
		} else {
			arm.FalseBlocked++
		}
		if !reasonContains(got.Reasons, tc.wantReason) {
			arm.ReasonMismatches++
		}
	}
	arm.Latency = time.Since(start)
	arm.Correct = arm.TrueReady == 1 && arm.TrueBlocked == 4 && arm.FalseReady == 0 && arm.FalseBlocked == 0 && arm.ReasonMismatches == 0
	arm.Note = "versioned usefulness, cold-path, provenance-plane, and unsupported-capability checks"
	return arm
}

func runThresholdReadiness(cases []readinessCase) ReadinessComparisonArm {
	arm := ReadinessComparisonArm{Name: "usefulness-score threshold only", Kind: "baseline", Available: true, Cases: len(cases)}
	start := time.Now()
	for _, tc := range cases {
		got := scoreThresholdReady(tc.report)
		if got == tc.wantReady {
			if got {
				arm.TrueReady++
			} else {
				arm.TrueBlocked++
			}
		} else if got {
			arm.FalseReady++
		} else {
			arm.FalseBlocked++
		}
		if !got && tc.wantReason != "" {
			arm.ReasonMismatches++
		}
		if got && !tc.wantReady {
			arm.ReasonMismatches++
		}
	}
	arm.Latency = time.Since(start)
	arm.Correct = arm.TrueReady == 1 && arm.TrueBlocked == 4 && arm.FalseReady == 0 && arm.FalseBlocked == 0 && arm.ReasonMismatches == 0
	arm.Note = "tuned no-policy baseline uses the committed default-ready score threshold but ignores provenance and cold-path invariants"
	return arm
}

func CompareDefaultReadinessLocal() ReadinessComparisonResult {
	cases := readinessComparisonCases()
	return ReadinessComparisonResult{Workload: "classify five ordered cache-readiness reports with exact ready/blocked and reason-class oracle", Arms: []ReadinessComparisonArm{
		runNativeReadiness(cases),
		runThresholdReadiness(cases),
		{Name: "fak + Prometheus", Kind: "integration", Note: "requires real exported metrics, rules, and read-back"},
		{Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires real collector/exporter policy and read-back"},
		{Name: "OPA/Rego", Kind: "external", Note: "requires pinned OPA and real Rego policy evaluation"},
		{Name: "Prometheus rules", Kind: "external", Note: "requires pinned Prometheus rule evaluation"},
		{Name: "Datadog monitors", Kind: "external", Note: "requires real monitor ingestion and evaluation"},
		{Name: "LangSmith evaluations", Kind: "external", Note: "requires real trace/evaluation boundary"},
	}}
}
