package trajectory

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

const (
	DerivedSchemaV1 = "fak-trajectory-derived/1alpha1"
	DeriverV1       = "fak-rule-deriver/1"
)

type DerivedKind string

const (
	DerivedPhase         DerivedKind = "phase"
	DerivedRetry         DerivedKind = "retry"
	DerivedStall         DerivedKind = "stall"
	DerivedOutcome       DerivedKind = "outcome"
	DerivedGoal          DerivedKind = "goal"
	DerivedControlAction DerivedKind = "decision"
	DerivedArtifact      DerivedKind = "artifact"
	DerivedCost          DerivedKind = "cost"
	DerivedCausalLink    DerivedKind = "causal_link"
)

// DerivedSignal is an interpretation of canonical evidence, never a replacement for it.
type DerivedSignal struct {
	ID              string          `json:"id"`
	Kind            DerivedKind     `json:"kind"`
	Label           string          `json:"label"`
	Start           time.Time       `json:"start"`
	End             *time.Time      `json:"end,omitempty"`
	SourceEventIDs  []string        `json:"source_event_ids"`
	Confidence      string          `json:"confidence"`
	ConfidenceBasis string          `json:"confidence_basis"`
	Rule            string          `json:"rule"`
	Payload         json.RawMessage `json:"payload,omitempty"`
}

type DeriveOptions struct {
	StallThreshold time.Duration
}

type DerivationReceipt struct {
	Schema         string            `json:"schema"`
	Deriver        string            `json:"deriver"`
	InputDigest    string            `json:"input_digest"`
	OutputDigest   string            `json:"output_digest"`
	Signals        []DerivedSignal   `json:"signals"`
	UnmatchedKinds map[EventKind]int `json:"unmatched_event_kinds,omitempty"`
}

type phaseStart struct {
	event Event
	label string
}

// DeriveSignals deterministically interprets an event stream while retaining the exact evidence for each claim.
func DeriveSignals(events []Event, opts DeriveOptions) (DerivationReceipt, error) {
	ordered := append([]Event(nil), events...)
	for i := range ordered {
		if err := ordered[i].Validate(); err != nil {
			return DerivationReceipt{}, fmt.Errorf("event %d: %w", i, err)
		}
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Timestamp.Equal(ordered[j].Timestamp) {
			return ordered[i].Sequence < ordered[j].Sequence
		}
		return ordered[i].Timestamp.Before(ordered[j].Timestamp)
	})
	encoded, err := EncodeEvents(ordered)
	if err != nil {
		return DerivationReceipt{}, err
	}

	var signals []DerivedSignal
	unsupported := map[EventKind]int{}
	phases := map[string]phaseStart{}
	failedTools := map[string]Event{}
	for i, event := range ordered {
		if opts.StallThreshold > 0 && i > 0 {
			prior := ordered[i-1]
			if gap := event.Timestamp.Sub(prior.Timestamp); gap >= opts.StallThreshold {
				payload, _ := json.Marshal(map[string]any{"duration_ms": gap.Milliseconds(), "threshold_ms": opts.StallThreshold.Milliseconds()})
				signals = append(signals, newSignal(DerivedStall, "event gap", prior.Timestamp, &event.Timestamp, []string{prior.ID, event.ID}, "high", "adjacent canonical timestamps exceed configured threshold", "stall-gap/v1", payload))
			}
		}
		for _, parentID := range event.ParentIDs {
			payload, _ := json.Marshal(map[string]string{"parent_event_id": parentID, "child_event_id": event.ID})
			signals = append(signals, newSignal(DerivedCausalLink, "parent", event.Timestamp, nil, []string{parentID, event.ID}, "high", "explicit canonical parent edge", "parent-edge/v1", payload))
		}

		switch event.Kind {
		case EventRunLifecycle:
			label := payloadString(event.Payload, "phase", "name", "state")
			if label == "" {
				label = event.Action
			}
			switch event.Action {
			case "started", "start", "entered":
				phases[label] = phaseStart{event: event, label: label}
			case "completed", "complete", "failed", "ended", "exit":
				start, ok := phases[label]
				if !ok {
					signals = append(signals, newSignal(DerivedPhase, label, event.Timestamp, &event.Timestamp, []string{event.ID}, "low", "terminal lifecycle event has no matching start", "lifecycle-phase/v1", event.Payload))
					break
				}
				end := event.Timestamp
				signals = append(signals, newSignal(DerivedPhase, label, start.event.Timestamp, &end, []string{start.event.ID, event.ID}, "high", "matching lifecycle start and terminal events", "lifecycle-phase/v1", event.Payload))
				delete(phases, label)
			}
		case EventTool:
			tool := payloadString(event.Payload, "tool", "name")
			if tool == "" {
				tool = event.Action
			}
			if event.Action == "failed" || event.Action == "error" {
				failedTools[tool] = event
			} else if event.Action == "started" || event.Action == "called" {
				if failed, ok := failedTools[tool]; ok {
					payload, _ := json.Marshal(map[string]any{"tool": tool})
					signals = append(signals, newSignal(DerivedRetry, tool, failed.Timestamp, &event.Timestamp, []string{failed.ID, event.ID}, "high", "same tool starts after its most recent canonical failure", "tool-retry/v1", payload))
					delete(failedTools, tool)
				}
			}
		case EventOutcome:
			label := payloadString(event.Payload, "outcome", "status", "result")
			if label == "" {
				label = event.Action
			}
			signals = append(signals, explicitSignal(DerivedOutcome, label, event, "explicit canonical outcome event", "outcome-event/v1"))
		case EventCheckpoint:
			if label := payloadString(event.Payload, "goal", "objective", "status", "progress"); label != "" {
				signals = append(signals, explicitSignal(DerivedGoal, label, event, "checkpoint explicitly names goal or progress", "goal-checkpoint/v1"))
			} else {
				unsupported[event.Kind]++
			}
		case EventIntervention, EventApproval:
			signals = append(signals, explicitSignal(DerivedControlAction, event.Action, event, "explicit canonical human control event", "control-decision/v1"))
		case EventArtifact:
			label := payloadString(event.Payload, "name", "path", "uri", "type")
			if label == "" {
				label = event.Action
			}
			signals = append(signals, explicitSignal(DerivedArtifact, label, event, "explicit canonical artifact event", "artifact-event/v1"))
		case EventObservation:
			if hasPayloadKey(event.Payload, "cost", "cost_usd", "tokens", "input_tokens", "output_tokens") {
				signals = append(signals, explicitSignal(DerivedCost, event.Action, event, "observation explicitly reports cost or token fields", "cost-observation/v1"))
			} else {
				unsupported[event.Kind]++
			}
		default:
			unsupported[event.Kind]++
		}
	}
	for _, start := range phases {
		signals = append(signals, newSignal(DerivedPhase, start.label, start.event.Timestamp, nil, []string{start.event.ID}, "medium", "lifecycle start has no terminal event in the observed window", "lifecycle-phase/v1", start.event.Payload))
	}
	sort.SliceStable(signals, func(i, j int) bool {
		if signals[i].Start.Equal(signals[j].Start) {
			return signals[i].ID < signals[j].ID
		}
		return signals[i].Start.Before(signals[j].Start)
	})
	outputBytes, err := json.Marshal(signals)
	if err != nil {
		return DerivationReceipt{}, err
	}
	return DerivationReceipt{Schema: DerivedSchemaV1, Deriver: DeriverV1, InputDigest: digestBytes(encoded), OutputDigest: digestBytes(outputBytes), Signals: signals, UnmatchedKinds: unsupported}, nil
}

func explicitSignal(kind DerivedKind, label string, event Event, basis, rule string) DerivedSignal {
	return newSignal(kind, label, event.Timestamp, nil, []string{event.ID}, "high", basis, rule, event.Payload)
}

func newSignal(kind DerivedKind, label string, start time.Time, end *time.Time, sources []string, confidence, basis, rule string, payload json.RawMessage) DerivedSignal {
	if len(payload) == 0 {
		payload = nil
	}
	seed, _ := json.Marshal([]any{kind, label, start.UTC(), end, sources, rule, payload})
	return DerivedSignal{ID: "derived:" + strings.TrimPrefix(digestBytes(seed), "sha256:")[:20], Kind: kind, Label: label, Start: start.UTC(), End: end, SourceEventIDs: append([]string(nil), sources...), Confidence: confidence, ConfidenceBasis: basis, Rule: rule, Payload: append(json.RawMessage(nil), payload...)}
}

func payloadObject(raw json.RawMessage) map[string]any {
	var object map[string]any
	if json.Unmarshal(raw, &object) != nil {
		return nil
	}
	return object
}

func payloadString(raw json.RawMessage, keys ...string) string {
	object := payloadObject(raw)
	for _, key := range keys {
		if value, ok := object[key].(string); ok && strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func hasPayloadKey(raw json.RawMessage, keys ...string) bool {
	object := payloadObject(raw)
	for _, key := range keys {
		if _, ok := object[key]; ok {
			return true
		}
	}
	return false
}

func (r DerivationReceipt) Validate() error {
	if r.Schema != DerivedSchemaV1 || r.Deriver == "" {
		return errors.New("invalid derivation receipt identity")
	}
	if !strings.HasPrefix(r.InputDigest, "sha256:") || !strings.HasPrefix(r.OutputDigest, "sha256:") {
		return errors.New("derivation receipt requires input and output digests")
	}
	outputBytes, err := json.Marshal(r.Signals)
	if err != nil {
		return err
	}
	if digestBytes(outputBytes) != r.OutputDigest {
		return errors.New("derivation receipt output digest does not bind signals")
	}
	for i, signal := range r.Signals {
		if signal.ID == "" || signal.Kind == "" || signal.Rule == "" || signal.Confidence == "" || signal.ConfidenceBasis == "" || len(signal.SourceEventIDs) == 0 {
			return fmt.Errorf("signal %d lacks provenance or interpretation metadata", i)
		}
	}
	return nil
}
