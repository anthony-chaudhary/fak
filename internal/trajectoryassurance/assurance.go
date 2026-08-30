// Package trajectoryassurance builds read-only, privacy-safe trajectory health receipts.
package trajectoryassurance

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const SchemaVersion = "fak.trajectory-assurance/v1"

type State string

const (
	Pass                         State = "PASS"
	Warn                         State = "WARN"
	Fail                         State = "FAIL"
	Unknown                      State = "UNKNOWN"
	RecommendationContinue             = "continue"
	RecommendationDeepenAudit          = "checkpoint/deepen-audit"
	RecommendationOperatorReview       = "hold-for-operator-review"
)

type Evidence struct {
	Source      string `json:"source"`
	Provenance  string `json:"provenance"`
	Authority   string `json:"authority"`
	Freshness   string `json:"freshness"`
	Reason      string `json:"reason"`
	ReasonToken string `json:"reason_token"`
}

type Observation struct {
	State    State    `json:"state"`
	Evidence Evidence `json:"evidence"`
}

type DeterministicCheck struct {
	Name     string   `json:"name"`
	Passed   *bool    `json:"passed"`
	Evidence Evidence `json:"evidence"`
}

type Usage struct {
	ParentUnits int64 `json:"parent_units"`
	ChildUnits  int64 `json:"child_units"`
	TotalUnits  int64 `json:"total_units"`
}

type EfficiencyInput struct {
	Outcome              *bool    `json:"outcome"`
	ConstraintsSatisfied *bool    `json:"constraints_satisfied"`
	ParentUnits          *int64   `json:"parent_units"`
	ChildUnits           []int64  `json:"child_units"`
	AccountingComplete   bool     `json:"accounting_complete"`
	Evidence             Evidence `json:"evidence"`
}

type Input struct {
	ObjectiveID         string               `json:"objective_id"`
	TrajectoryID        string               `json:"trajectory_id"`
	SessionID           string               `json:"session_id,omitempty"`
	RunID               string               `json:"run_id,omitempty"`
	ObservationWindow   string               `json:"observation_window"`
	DeterministicFloor  []DeterministicCheck `json:"deterministic_floor"`
	ObjectiveProgress   Observation          `json:"objective_progress"`
	Efficiency          EfficiencyInput      `json:"efficiency_with_quality"`
	DelegationIntegrity Observation          `json:"delegation_integrity"`
	SemanticReview      Observation          `json:"semantic_review"`
}

type Layer struct {
	Name        string `json:"name"`
	State       State  `json:"state"`
	Source      string `json:"source"`
	Provenance  string `json:"provenance"`
	Authority   string `json:"authority"`
	Freshness   string `json:"freshness"`
	Reason      string `json:"reason"`
	ReasonToken string `json:"reason_token"`
	Usage       *Usage `json:"usage,omitempty"`
}

type Conflict struct {
	Layers string `json:"layers"`
	Reason string `json:"reason"`
}

type Receipt struct {
	ObjectiveID       string     `json:"objective_id"`
	TrajectoryID      string     `json:"trajectory_id"`
	SessionID         string     `json:"session_id,omitempty"`
	RunID             string     `json:"run_id,omitempty"`
	ObservationWindow string     `json:"observation_window"`
	SchemaVersion     string     `json:"schema_version"`
	Shadow            bool       `json:"shadow"`
	State             State      `json:"state"`
	Layers            []Layer    `json:"layers"`
	Conflicts         []Conflict `json:"conflicts"`
	MissingEvidence   []string   `json:"missing_evidence"`
	Recommendation    string     `json:"recommendation"`
}

func Assess(in Input) Receipt {
	floor := assessFloor(in.DeterministicFloor)
	objective := assessObservation("objective_progress", in.ObjectiveProgress)
	efficiency := assessEfficiency(in.Efficiency)
	delegation := assessObservation("delegation_integrity", in.DelegationIntegrity)
	semantic := assessObservation("semantic_review", in.SemanticReview)
	layers := []Layer{floor, objective, efficiency, delegation, semantic}
	for i := range layers {
		if layers[i].ReasonToken == "" {
			layers[i].ReasonToken = defaultReasonToken(layers[i].Name, layers[i].State)
		}
	}

	var missing []string
	for _, layer := range layers {
		if layer.State == Unknown {
			missing = append(missing, layer.Name)
		}
	}
	sort.Strings(missing)

	var conflicts []Conflict
	if floor.State == Fail && semantic.State == Pass {
		conflicts = append(conflicts, Conflict{
			Layers: "deterministic_floor,semantic_review",
			Reason: "semantic PASS cannot override a deterministic FAIL",
		})
	}

	state := aggregate(layers)
	return Receipt{
		ObjectiveID:       in.ObjectiveID,
		TrajectoryID:      in.TrajectoryID,
		SessionID:         in.SessionID,
		RunID:             in.RunID,
		ObservationWindow: in.ObservationWindow,
		SchemaVersion:     SchemaVersion,
		Shadow:            true,
		State:             state,
		Layers:            layers,
		Conflicts:         conflicts,
		MissingEvidence:   missing,
		Recommendation:    recommendation(layers, state),
	}
}

func Marshal(receipt Receipt) ([]byte, error) {
	if !receipt.Shadow {
		return nil, fmt.Errorf("trajectory assurance: shadow must be true")
	}
	if strings.TrimSpace(receipt.Recommendation) == "" {
		return nil, fmt.Errorf("trajectory assurance: exactly one non-empty recommendation is required")
	}
	return json.Marshal(receipt)
}

func assessFloor(checks []DeterministicCheck) Layer {
	layer := Layer{Name: "deterministic_floor", State: Pass, Reason: "all deterministic checks passed"}
	if len(checks) == 0 {
		layer.State = Unknown
		layer.Reason = "deterministic checks are missing"
		return layer
	}
	checks = append([]DeterministicCheck(nil), checks...)
	sort.Slice(checks, func(i, j int) bool { return checks[i].Name < checks[j].Name })
	for _, check := range checks {
		if check.Passed == nil {
			if layer.State != Fail {
				layer.State = Unknown
			}
			layer.Reason = "one or more deterministic results are missing"
		} else if !*check.Passed {
			layer.State = Fail
			layer.Reason = "one or more deterministic checks failed"
		}
		layer = mergeEvidence(layer, check.Evidence)
	}
	return layer
}

func assessObservation(name string, observation Observation) Layer {
	state := observation.State
	if !validState(state) || missingMetadata(observation.Evidence) {
		state = Unknown
	}
	reason := observation.Evidence.Reason
	if reason == "" {
		reason = "required evidence is missing"
	}
	return Layer{Name: name, State: state, Source: observation.Evidence.Source, Provenance: observation.Evidence.Provenance, Authority: observation.Evidence.Authority, Freshness: observation.Evidence.Freshness, Reason: reason, ReasonToken: observation.Evidence.ReasonToken}
}

func assessEfficiency(in EfficiencyInput) Layer {
	layer := Layer{Name: "efficiency_with_quality", Source: in.Evidence.Source, Provenance: in.Evidence.Provenance, Authority: in.Evidence.Authority, Freshness: in.Evidence.Freshness, Reason: in.Evidence.Reason, ReasonToken: in.Evidence.ReasonToken}
	if in.Outcome == nil || in.ConstraintsSatisfied == nil || in.ParentUnits == nil || !in.AccountingComplete || missingMetadata(in.Evidence) {
		layer.State = Unknown
		layer.Reason = "outcome, constraints, and complete parent+child accounting are required"
		return layer
	}
	usage := Usage{ParentUnits: *in.ParentUnits}
	for _, units := range in.ChildUnits {
		usage.ChildUnits += units
	}
	usage.TotalUnits = usage.ParentUnits + usage.ChildUnits
	layer.Usage = &usage
	switch {
	case !*in.Outcome || !*in.ConstraintsSatisfied:
		layer.State = Fail
		layer.Reason = "quality outcome or constraints failed"
	case usage.ParentUnits < 0 || usage.ChildUnits < 0:
		layer.State = Fail
		layer.Reason = "usage accounting contains negative units"
	default:
		layer.State = Pass
		if layer.Reason == "" {
			layer.Reason = "quality held with complete parent+child accounting"
		}
	}
	return layer
}

func mergeEvidence(layer Layer, evidence Evidence) Layer {
	if layer.Source == "" {
		layer.Source = evidence.Source
	}
	if layer.Provenance == "" {
		layer.Provenance = evidence.Provenance
	}
	if layer.Authority == "" {
		layer.Authority = evidence.Authority
	}
	if layer.Freshness == "" {
		layer.Freshness = evidence.Freshness
	}
	if layer.ReasonToken == "" {
		layer.ReasonToken = evidence.ReasonToken
	}
	if missingMetadata(evidence) && layer.State != Fail {
		layer.State = Unknown
		layer.Reason = "deterministic result metadata is missing"
	}
	return layer
}

func aggregate(layers []Layer) State {
	seenWarn, seenUnknown := false, false
	for _, layer := range layers {
		switch layer.State {
		case Fail:
			return Fail
		case Warn:
			seenWarn = true
		case Unknown:
			seenUnknown = true
		case Pass:
			// Passing layers are neutral while stronger states are folded.
		}
	}
	if seenUnknown {
		return Unknown
	}
	if seenWarn {
		return Warn
	}
	return Pass
}

func recommendation(layers []Layer, state State) string {
	for _, layer := range layers {
		if layer.State != Fail {
			continue
		}
		if layer.Name == "deterministic_floor" || layer.Name == "delegation_integrity" {
			return RecommendationOperatorReview
		}
	}
	if state == Pass {
		return RecommendationContinue
	}
	return RecommendationDeepenAudit
}

func validState(state State) bool {
	return state == Pass || state == Warn || state == Fail || state == Unknown
}

func missingMetadata(e Evidence) bool {
	return strings.TrimSpace(e.Source) == "" || strings.TrimSpace(e.Provenance) == "" || strings.TrimSpace(e.Authority) == "" || strings.TrimSpace(e.Freshness) == "" || strings.TrimSpace(e.Reason) == ""
}

func defaultReasonToken(layer string, state State) string {
	normalized := strings.ToUpper(strings.ReplaceAll(layer, "-", "_"))
	return normalized + "_" + string(state)
}
