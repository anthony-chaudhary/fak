package coordination

import (
	"math"
	"strconv"
	"strings"
)

type HarnessKind string

const (
	HarnessKindFakNative HarnessKind = "fak_native"
	HarnessKindExternal  HarnessKind = "external"
)

type HarnessIntent struct {
	Kind         HarnessKind `json:"kind"`
	Task         string      `json:"task"`
	Outcome      string      `json:"outcome"`
	Capabilities []string    `json:"capabilities"`
}

type ContextState struct {
	Managed             bool    `json:"managed"`
	ReusablePrefixBytes int     `json:"reusablePrefixBytes"`
	Pressure            float64 `json:"pressure"`
}

type ComputeState struct {
	Engine     string `json:"engine"`
	Available  bool   `json:"available"`
	QueueDepth int    `json:"queueDepth"`
}

type ServeState struct {
	Admitted     bool    `json:"admitted"`
	Backpressure float64 `json:"backpressure"`
}

type EvidenceRequirements struct {
	RequireOutcome bool `json:"requireOutcome"`
	RequireEffects bool `json:"requireEffects"`
}

type PlanDisposition string

const (
	PlanDispositionExecute PlanDisposition = "execute"
	PlanDispositionDefer   PlanDisposition = "defer"
	PlanDispositionRefuse  PlanDisposition = "refuse"
)

type Plan struct {
	Disposition                PlanDisposition `json:"disposition"`
	FusedValue                 string          `json:"fusedValueExplanation"`
	SelectedHarness            HarnessKind     `json:"selectedHarness"`
	SelectedEngine             string          `json:"selectedEngine"`
	ContextReuse               bool            `json:"contextReuse"`
	RequiredEvidence           []string        `json:"requiredEvidence"`
	RawModelEvidenceSufficient bool            `json:"rawModelEvidenceSufficient"`
}

type Input struct {
	HarnessIntent        HarnessIntent        `json:"harnessIntent"`
	ContextState         ContextState         `json:"contextState"`
	ComputeState         ComputeState         `json:"computeState"`
	ServeState           ServeState           `json:"serveState"`
	EvidenceRequirements EvidenceRequirements `json:"evidenceRequirements"`
}

// Build validates the complete input before deciding whether work can execute.
// Invalid inputs refuse rather than falling through to a defer or execute plan.
func Build(input Input) Plan {
	if reason := validationFailure(input); reason != "" {
		return newPlan(PlanDispositionRefuse, "refused: "+reason, "", input.ComputeState.Engine, false)
	}

	contextReuse := input.ContextState.Managed && input.ContextState.ReusablePrefixBytes > 0
	if !input.ComputeState.Available || !input.ServeState.Admitted ||
		input.ContextState.Pressure > 0.8 || input.ServeState.Backpressure > 0.8 {
		return newPlan(
			PlanDispositionDefer,
			deferExplanation(input, contextReuse),
			input.HarnessIntent.Kind,
			input.ComputeState.Engine,
			contextReuse,
		)
	}

	return newPlan(
		PlanDispositionExecute,
		fusedValueExplanation(input, contextReuse),
		input.HarnessIntent.Kind,
		input.ComputeState.Engine,
		contextReuse,
	)
}

func validationFailure(input Input) string {
	switch input.HarnessIntent.Kind {
	case HarnessKindFakNative, HarnessKindExternal:
	default:
		return "harness kind must be fak_native or external"
	}
	if strings.TrimSpace(input.HarnessIntent.Task) == "" {
		return "task must not be empty"
	}
	if strings.TrimSpace(input.HarnessIntent.Outcome) == "" {
		return "outcome must not be empty"
	}
	if !validPressure(input.ContextState.Pressure) {
		return "context pressure must be between 0 and 1"
	}
	if !validPressure(input.ServeState.Backpressure) {
		return "serve backpressure must be between 0 and 1"
	}
	if input.ComputeState.Engine != "fak_native" {
		return "engine must be fak_native"
	}
	return ""
}

func validPressure(value float64) bool {
	return !math.IsNaN(value) && value >= 0 && value <= 1
}

func newPlan(
	disposition PlanDisposition,
	explanation string,
	harness HarnessKind,
	engine string,
	contextReuse bool,
) Plan {
	return Plan{
		Disposition:                disposition,
		FusedValue:                 explanation,
		SelectedHarness:            harness,
		SelectedEngine:             engine,
		ContextReuse:               contextReuse,
		RequiredEvidence:           []string{"agent_outcome", "effects"},
		RawModelEvidenceSufficient: false,
	}
}

func deferExplanation(input Input, contextReuse bool) string {
	reasons := make([]string, 0, 4)
	if !input.ComputeState.Available {
		reasons = append(reasons, "compute unavailable")
	}
	if !input.ServeState.Admitted {
		reasons = append(reasons, "serve not admitted")
	}
	if input.ContextState.Pressure > 0.8 {
		reasons = append(reasons, "high cache pressure")
	}
	if input.ServeState.Backpressure > 0.8 {
		reasons = append(reasons, "high serve backpressure")
	}
	return "deferred: " + strings.Join(reasons, "; ") + contextExplanation(input, contextReuse)
}

func fusedValueExplanation(input Input, contextReuse bool) string {
	base := "execute: "
	if input.HarnessIntent.Kind == HarnessKindFakNative {
		base += "fak-native harness"
	} else {
		base += "external harness"
	}
	base += contextExplanation(input, contextReuse)
	if input.ContextState.Pressure > 0.5 {
		base += "; elevated cache pressure"
	}
	if input.ServeState.Backpressure > 0.5 {
		base += "; elevated serve backpressure"
	}
	return base
}

func contextExplanation(input Input, contextReuse bool) string {
	if contextReuse {
		return " with context reuse (" + strconv.Itoa(input.ContextState.ReusablePrefixBytes) + " bytes)"
	}
	return " without context reuse"
}
