package trajectory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
)

const (
	ViewSpecSchema    = "fak-trajectory-view/1alpha1"
	ViewReceiptSchema = "fak-trajectory-view-receipt/1alpha1"
)

type ViewSpec struct {
	Schema           string   `json:"schema"`
	Audience         string   `json:"audience"`
	Include          []string `json:"include"`
	Exclude          []string `json:"exclude,omitempty"`
	RedactFields     []string `json:"redact_fields,omitempty"`
	RedactionProfile string   `json:"redaction_profile"`
	Density          string   `json:"density,omitempty"`
	LiveControls     []string `json:"live_controls,omitempty"`
}

func (s ViewSpec) Validate() error {
	if s.Schema != ViewSpecSchema || strings.TrimSpace(s.Audience) == "" || strings.TrimSpace(s.RedactionProfile) == "" {
		return errors.New("trajectory view requires schema, audience, and redaction_profile")
	}
	if len(s.Include) == 0 {
		return errors.New("trajectory view requires at least one include pattern")
	}
	for _, pattern := range append(append([]string(nil), s.Include...), s.Exclude...) {
		if _, err := path.Match(pattern, "message.completed"); err != nil {
			return fmt.Errorf("invalid trajectory view pattern %q: %w", pattern, err)
		}
	}
	return nil
}

type ProjectedEvent struct {
	Event
	RedactedFields []string `json:"redacted_fields,omitempty"`
}

type ViewReceipt struct {
	Schema           string         `json:"schema"`
	Audience         string         `json:"audience"`
	TrajectoryDigest string         `json:"trajectory_digest"`
	ViewSpecDigest   string         `json:"view_spec_digest"`
	RedactionProfile string         `json:"redaction_profile"`
	OrderingRule     string         `json:"ordering_rule"`
	InputEvents      int            `json:"input_events"`
	OutputEvents     int            `json:"output_events"`
	Omitted          map[string]int `json:"omitted"`
	RedactedFields   int            `json:"redacted_fields"`
	ProjectionDigest string         `json:"projection_digest"`
}

func (r ViewReceipt) Validate() error {
	if r.Schema != ViewReceiptSchema || r.Audience == "" || r.TrajectoryDigest == "" || r.ViewSpecDigest == "" || r.RedactionProfile == "" || r.OrderingRule == "" || r.ProjectionDigest == "" {
		return errors.New("incomplete trajectory view receipt")
	}
	if r.OutputEvents > r.InputEvents || r.RedactedFields < 0 {
		return errors.New("invalid trajectory view receipt counts")
	}
	return nil
}

// CompileView selects and redacts canonical events before a renderer receives them.
func CompileView(events []Event, spec ViewSpec) ([]ProjectedEvent, ViewReceipt, error) {
	if err := spec.Validate(); err != nil {
		return nil, ViewReceipt{}, err
	}
	canonical, err := EncodeEvents(events)
	if err != nil {
		return nil, ViewReceipt{}, err
	}
	specBytes, err := json.Marshal(spec)
	if err != nil {
		return nil, ViewReceipt{}, err
	}
	receipt := ViewReceipt{Schema: ViewReceiptSchema, Audience: spec.Audience, TrajectoryDigest: viewDigest(canonical), ViewSpecDigest: viewDigest(specBytes), RedactionProfile: spec.RedactionProfile, OrderingRule: "timestamp-then-sequence-stable", InputEvents: len(events), Omitted: make(map[string]int)}

	ordered := append([]Event(nil), events...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Timestamp.Equal(ordered[j].Timestamp) {
			return ordered[i].Sequence < ordered[j].Sequence
		}
		return ordered[i].Timestamp.Before(ordered[j].Timestamp)
	})
	projected := make([]ProjectedEvent, 0, len(ordered))
	for _, event := range ordered {
		key := string(event.Kind) + "." + event.Action
		if !matchesAny(spec.Include, key) {
			receipt.Omitted["not-included"]++
			continue
		}
		if matchesAny(spec.Exclude, key) {
			receipt.Omitted["excluded"]++
			continue
		}
		copyEvent := event
		copyEvent.ParentIDs = append([]string(nil), event.ParentIDs...)
		copyEvent.Payload = append(json.RawMessage(nil), event.Payload...)
		redactedPayload, redacted, err := redactPayload(copyEvent.Payload, spec.RedactFields)
		if err != nil {
			return nil, receipt, fmt.Errorf("event %s redaction: %w", event.ID, err)
		}
		copyEvent.Payload = redactedPayload
		receipt.RedactedFields += len(redacted)
		projected = append(projected, ProjectedEvent{Event: copyEvent, RedactedFields: redacted})
	}
	receipt.OutputEvents = len(projected)
	projectionBytes, err := json.Marshal(projected)
	if err != nil {
		return nil, receipt, err
	}
	receipt.ProjectionDigest = viewDigest(projectionBytes)
	if err := receipt.Validate(); err != nil {
		return nil, receipt, err
	}
	return projected, receipt, nil
}

func matchesAny(patterns []string, key string) bool {
	for _, pattern := range patterns {
		if matched, _ := path.Match(pattern, key); matched {
			return true
		}
	}
	return false
}

func redactPayload(payload json.RawMessage, fields []string) (json.RawMessage, []string, error) {
	if len(fields) == 0 {
		return append(json.RawMessage(nil), payload...), nil, nil
	}
	var value any
	if err := json.Unmarshal(payload, &value); err != nil {
		return nil, nil, err
	}
	wanted := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		wanted[strings.ToLower(field)] = struct{}{}
	}
	var redacted []string
	redactValue(value, "", wanted, &redacted)
	sort.Strings(redacted)
	encoded, err := json.Marshal(value)
	return encoded, redacted, err
}

func redactValue(value any, prefix string, wanted map[string]struct{}, redacted *[]string) {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			location := key
			if prefix != "" {
				location = prefix + "." + key
			}
			_, byName := wanted[strings.ToLower(key)]
			_, byPath := wanted[strings.ToLower(location)]
			if byName || byPath {
				typed[key] = "[REDACTED]"
				*redacted = append(*redacted, location)
				continue
			}
			redactValue(typed[key], location, wanted, redacted)
		}
	case []any:
		for index := range typed {
			redactValue(typed[index], fmt.Sprintf("%s[%d]", prefix, index), wanted, redacted)
		}
	}
}

func viewDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func EndUserView() ViewSpec {
	return ViewSpec{Schema: ViewSpecSchema, Audience: "end-user", Include: []string{"message.*", "approval.*", "artifact.*", "error.*", "intervention.*"}, Exclude: []string{"message.delta"}, RedactFields: []string{"secret", "token", "api_key", "private_reasoning"}, RedactionProfile: "end-user-default/v1", Density: "comfortable", LiveControls: []string{"steer"}}
}

func OperatorView() ViewSpec {
	return ViewSpec{Schema: ViewSpecSchema, Audience: "operator", Include: []string{"run-lifecycle.*", "message.completed", "tool.*", "approval.*", "artifact.*", "observation.*", "outcome.*", "error.*", "intervention.*"}, Exclude: []string{"tool.delta"}, RedactFields: []string{"secret", "token", "api_key"}, RedactionProfile: "operator-default/v1", Density: "compact", LiveControls: []string{"approve", "deny", "steer", "pause"}}
}
