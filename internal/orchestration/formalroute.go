package orchestration

import (
	"path/filepath"
	"strings"
)

const (
	FormalPacketSchemaVersion = "fak-formal-packet/1"
	AstraWorkerModel          = "gpt-6-astra"
	AstraWorkerEffort         = "xhigh"
)

// FormalTaskKind is the closed set of work shapes for which Astra's formal
// reasoning strength is the intended automatic route. General implementation,
// exploration, review, and test execution are deliberately outside this set.
type FormalTaskKind string

const (
	FormalProof                    FormalTaskKind = "formal_proof"
	InvariantAudit                 FormalTaskKind = "invariant_audit"
	StateMachineAudit              FormalTaskKind = "state_machine_audit"
	NumericalCorrectnessDerivation FormalTaskKind = "numerical_correctness_derivation"
)

var allowedFormalTaskKinds = map[FormalTaskKind]struct{}{
	FormalProof:                    {},
	InvariantAudit:                 {},
	StateMachineAudit:              {},
	NumericalCorrectnessDerivation: {},
}

var excludedFormalTaskKinds = map[FormalTaskKind]struct{}{
	"documentation":  {},
	"exploration":    {},
	"general_review": {},
	"implementation": {},
	"issue_triage":   {},
	"test_execution": {},
}

// FormalPacket is the versioned, explicit contract required for automatic
// Astra selection. It is intentionally not derivable from task prose, labels,
// work tiers, or generic rigor keywords.
type FormalPacket struct {
	Schema                    string           `json:"schema"`
	TaskKinds                 []FormalTaskKind `json:"task_kinds"`
	DefinitionsAndAssumptions string           `json:"definitions_and_assumptions"`
	ExactProposition          string           `json:"exact_proposition"`
	RequiredOutputForm        string           `json:"required_output_form"`
	DeterministicWitness      string           `json:"deterministic_witness"`
	Surfaces                  []string         `json:"surfaces"`
}

type AstraRouteReason string

const (
	AstraReasonSchemaInvalid             AstraRouteReason = "ASTRA_FORMAL_PACKET_SCHEMA_INVALID"
	AstraReasonWorkClassRequired         AstraRouteReason = "ASTRA_FORMAL_PACKET_WORK_CLASS_REQUIRED"
	AstraReasonTaskKindRequired          AstraRouteReason = "ASTRA_FORMAL_PACKET_TASK_KIND_REQUIRED"
	AstraReasonTaskKindDuplicate         AstraRouteReason = "ASTRA_FORMAL_PACKET_TASK_KIND_DUPLICATE"
	AstraReasonTaskKindExcluded          AstraRouteReason = "ASTRA_FORMAL_PACKET_TASK_KIND_EXCLUDED"
	AstraReasonTaskKindUnknown           AstraRouteReason = "ASTRA_FORMAL_PACKET_TASK_KIND_UNKNOWN"
	AstraReasonDefinitionsAndAssumptions AstraRouteReason = "ASTRA_FORMAL_PACKET_DEFINITIONS_AND_ASSUMPTIONS_REQUIRED"
	AstraReasonExactProposition          AstraRouteReason = "ASTRA_FORMAL_PACKET_EXACT_PROPOSITION_REQUIRED"
	AstraReasonRequiredOutputForm        AstraRouteReason = "ASTRA_FORMAL_PACKET_REQUIRED_OUTPUT_FORM_REQUIRED"
	AstraReasonDeterministicWitness      AstraRouteReason = "ASTRA_FORMAL_PACKET_DETERMINISTIC_WITNESS_REQUIRED"
	AstraReasonSurfaceCount              AstraRouteReason = "ASTRA_FORMAL_PACKET_SURFACE_COUNT_INVALID"
	AstraReasonSurfaceUnbounded          AstraRouteReason = "ASTRA_FORMAL_PACKET_SURFACE_UNBOUNDED"
	AstraReasonSurfaceDuplicate          AstraRouteReason = "ASTRA_FORMAL_PACKET_SURFACE_DUPLICATE"
)

const (
	AstraRouteSourceFormalPacket = "formal-packet"
	AstraRouteSourceTaskPin      = "task.pin"
)

// AstraRoute is present only when a task supplied formal_packet. Eligible says
// the packet meets the automatic-routing contract; Selected says the effective
// worker route uses Astra after explicit task pins have been applied. Source
// distinguishes an automatic choice from an explicit Astra override.
type AstraRoute struct {
	Eligible              bool               `json:"eligible"`
	Selected              bool               `json:"selected"`
	Reasons               []AstraRouteReason `json:"reasons,omitempty"`
	Source                string             `json:"source"`
	Model                 string             `json:"model,omitempty"`
	ReasoningEffort       string             `json:"reasoning_effort,omitempty"`
	ReasoningEffortSource string             `json:"reasoning_effort_source,omitempty"`
}

// AssessAstraRoute returns nil when no formal packet was declared, preserving
// the pre-feature JSON byte shape. A declared but incomplete packet always
// returns a fail-closed receipt with deterministic, ordered reason codes.
func AssessAstraRoute(task TaskSpec) *AstraRoute {
	p := task.FormalPacket
	if p == nil {
		return nil
	}
	route := &AstraRoute{Source: AstraRouteSourceFormalPacket}
	add := func(reason AstraRouteReason, failed bool) {
		if failed {
			route.Reasons = append(route.Reasons, reason)
		}
	}

	add(AstraReasonSchemaInvalid, strings.TrimSpace(p.Schema) != FormalPacketSchemaVersion)
	add(AstraReasonWorkClassRequired, task.WorkClass != WorkRigor)

	missingKind := len(p.TaskKinds) == 0
	duplicateKind, excludedKind, unknownKind := false, false, false
	seenKinds := make(map[FormalTaskKind]struct{}, len(p.TaskKinds))
	for _, raw := range p.TaskKinds {
		kind := FormalTaskKind(strings.ToLower(strings.TrimSpace(string(raw))))
		if kind == "" {
			missingKind = true
			continue
		}
		if _, exists := seenKinds[kind]; exists {
			duplicateKind = true
			continue
		}
		seenKinds[kind] = struct{}{}
		if _, excluded := excludedFormalTaskKinds[kind]; excluded {
			excludedKind = true
			continue
		}
		if _, allowed := allowedFormalTaskKinds[kind]; !allowed {
			unknownKind = true
		}
	}
	add(AstraReasonTaskKindRequired, missingKind)
	add(AstraReasonTaskKindDuplicate, duplicateKind)
	add(AstraReasonTaskKindExcluded, excludedKind)
	add(AstraReasonTaskKindUnknown, unknownKind)

	add(AstraReasonDefinitionsAndAssumptions, placeholderFormalValue(p.DefinitionsAndAssumptions))
	add(AstraReasonExactProposition, placeholderFormalValue(p.ExactProposition))
	add(AstraReasonRequiredOutputForm, placeholderFormalValue(p.RequiredOutputForm))
	add(AstraReasonDeterministicWitness, placeholderFormalValue(p.DeterministicWitness))

	add(AstraReasonSurfaceCount, len(p.Surfaces) < 1 || len(p.Surfaces) > 3)
	unboundedSurface, duplicateSurface := false, false
	seenSurfaces := make(map[string]struct{}, len(p.Surfaces))
	for _, raw := range p.Surfaces {
		surface := filepathCleanSlash(raw)
		if !boundedAccessRegion(raw) {
			unboundedSurface = true
		}
		if _, exists := seenSurfaces[surface]; exists {
			duplicateSurface = true
		}
		seenSurfaces[surface] = struct{}{}
	}
	add(AstraReasonSurfaceUnbounded, unboundedSurface)
	add(AstraReasonSurfaceDuplicate, duplicateSurface)

	route.Eligible = len(route.Reasons) == 0
	if route.Eligible {
		route.Selected = true
		route.Model = AstraWorkerModel
		route.ReasoningEffort = AstraWorkerEffort
		route.ReasoningEffortSource = AstraRouteSourceFormalPacket
	}
	return route
}

func filepathCleanSlash(raw string) string {
	clean := filepath.Clean(strings.ReplaceAll(strings.TrimSpace(raw), `\`, string(filepath.Separator)))
	return strings.ToLower(filepath.ToSlash(clean))
}

func placeholderFormalValue(raw string) bool {
	v := strings.ToLower(strings.TrimSpace(raw))
	if v == "" {
		return true
	}
	switch v {
	case "...", "?", "n/a", "na", "none", "placeholder", "tbd", "todo", "unknown":
		return true
	}
	return strings.Contains(v, "<todo>") || strings.Contains(v, "<tbd>") ||
		strings.Contains(v, "fill this") || strings.Contains(v, "fill in")
}
