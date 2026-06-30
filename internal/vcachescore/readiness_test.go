package vcachescore

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachewarm"
)

func readyScoreInput() Input {
	in := DefaultInput()
	in.TelemetryRows = []vcachegov.TelemetryRow{
		{InputTokens: 10098, CacheCreationInputTokens: 59400, Ephemeral1hInputTokens: 59400},
		{InputTokens: 10065, CacheCreationInputTokens: 15411, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15411},
		{InputTokens: 10065, CacheCreationInputTokens: 15410, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15410},
		{InputTokens: 10065, CacheCreationInputTokens: 15424, CacheReadInputTokens: 43995, Ephemeral1hInputTokens: 15424},
	}
	in.AgenticActivation = AgenticActivationInput{
		KernelKVEvents:       1,
		ContextEvents:        1,
		ExternalEngineEvents: 1,
	}
	in.KernelKV = PlaneEvidenceInput{
		Available:          true,
		Provenance:         "WITNESSED",
		BaselineTokenEquiv: 1000,
		SavedTokenEquiv:    800,
		CostTokenEquiv:     200,
		Reason:             "kernel KV reuse witnessed by local counters",
	}
	in.Context = PlaneEvidenceInput{
		Available:          true,
		Provenance:         "WITNESSED",
		BaselineTokenEquiv: 1000,
		SavedTokenEquiv:    750,
		CostTokenEquiv:     250,
		Reason:             "ctxplan resident-view witness supplied",
	}
	in.ExternalEngine = PlaneEvidenceInput{
		Available:          true,
		Provenance:         "OBSERVED",
		BaselineTokenEquiv: 1000,
		SavedTokenEquiv:    600,
		CostTokenEquiv:     400,
		HitRate:            0.6,
		Reason:             "external engine prefix-cache observation supplied",
	}
	return in
}

func TestDefaultReadinessPassesWithWitnessedPlanes(t *testing.T) {
	in := readyScoreInput()
	in.Prediction = predictionWithLowFalseWarm()
	rep := Score(in)
	if rep.DefaultUsefulness.Verdict != "default_ready" {
		t.Fatalf("setup default_usefulness=%+v, want default_ready", rep.DefaultUsefulness)
	}
	gate := DefaultReadiness(rep)
	if !gate.OK {
		t.Fatalf("readiness gate failed: %+v", gate)
	}
	if gate.PlaneProvenance["provider_observed"] != "OBSERVED" ||
		gate.PlaneProvenance["kernel_witnessed"] != "WITNESSED" {
		t.Fatalf("plane provenance collapsed: %+v", gate.PlaneProvenance)
	}
}

func TestDefaultReadinessRejectsProviderOnlyEvidence(t *testing.T) {
	in := DefaultInput()
	in.TelemetryRows = readyScoreInput().TelemetryRows
	rep := Score(in)
	gate := DefaultReadiness(rep)
	if gate.OK {
		t.Fatalf("provider-only cache evidence must not enable default readiness: %+v", gate)
	}
	if !containsReason(gate.Reasons, "default_usefulness verdict") {
		t.Fatalf("missing default-usefulness reason: %+v", gate.Reasons)
	}
}

func TestDefaultReadinessRejectsPlaneProvenanceCollapse(t *testing.T) {
	in := readyScoreInput()
	in.Prediction = predictionWithLowFalseWarm()
	in.KernelKV.Provenance = "OBSERVED"
	rep := Score(in)
	gate := DefaultReadiness(rep)
	if gate.OK {
		t.Fatalf("kernel plane with OBSERVED provenance must fail readiness: %+v", gate)
	}
	if !containsReason(gate.Reasons, "kernel_witnessed provenance") {
		t.Fatalf("missing provenance reason: %+v", gate.Reasons)
	}
}

func TestDefaultReadinessRejectsUnsupportedActivePath(t *testing.T) {
	in := readyScoreInput()
	in.Prediction = predictionWithLowFalseWarm()
	in.ExternalEngine.Reason = string(vcachewarm.ReasonUnsupportedActiveCacheCapability)
	rep := Score(in)
	gate := DefaultReadiness(rep)
	if gate.OK {
		t.Fatalf("unsupported active cache path must fail readiness: %+v", gate)
	}
	if !containsReason(gate.Reasons, "unsupported active-cache") {
		t.Fatalf("missing unsupported reason: %+v", gate.Reasons)
	}
}

func predictionWithLowFalseWarm() vcachecal.PredictionError {
	return vcachecal.PredictionError{Total: 10, TrueWarm: 8, TrueCold: 2}
}

func containsReason(reasons []string, needle string) bool {
	for _, reason := range reasons {
		if strings.Contains(reason, needle) {
			return true
		}
	}
	return false
}
