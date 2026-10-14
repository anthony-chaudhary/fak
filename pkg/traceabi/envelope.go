// Package traceabi defines data-only causal identity shared by trace producers.
// It generates no IDs, records no events, and has no exporter or storage policy.
//
// W3C marks an ID encoding, never an inference from an opaque value's shape.
// Its widths follow https://www.w3.org/TR/trace-context/#trace-id and #parent-id.
// A span has at most one parent; fan-in uses separate links, which may cross
// traces. Unlike an OpenTelemetry recording-order API, links here are an
// unordered identity set with canonical serialization order.
package traceabi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const Schema = "fak-causal-envelope/1"
const DescriptorSchema = "fak-causal-envelope-schema/1"

type TraceFormat string

const (
	TraceOpaque TraceFormat = "opaque"
	TraceW3C    TraceFormat = "w3c"
)

// MappingStatus describes the producer's evidence for its causal mapping.
// Exact preserves supplied causal identity; derived is an explicitly known
// mapping; unknown makes missing causal information visible. Validation cannot
// establish the truth of a producer's claim or infer missing information.
type MappingStatus string

const (
	MappingExact   MappingStatus = "exact"
	MappingDerived MappingStatus = "derived"
	MappingUnknown MappingStatus = "unknown"
)

// ParentStatus distinguishes an explicitly known root from a missing parent.
type ParentStatus string

const (
	ParentKnown   ParentStatus = "known"
	ParentRoot    ParentStatus = "root"
	ParentUnknown ParentStatus = "unknown"
)

type CorrelationKind string

const (
	CorrelationSession        CorrelationKind = "session"
	CorrelationTurn           CorrelationKind = "turn"
	CorrelationRequest        CorrelationKind = "request"
	CorrelationRun            CorrelationKind = "run"
	CorrelationToolCall       CorrelationKind = "tool_call"
	CorrelationArtifact       CorrelationKind = "artifact"
	CorrelationProducerNative CorrelationKind = "producer_native"
)

// Correlation preserves one opaque producer-native identifier. The pair
// (kind, value) is its identity; multiple distinct values of one kind are valid.
type Correlation struct {
	Kind  CorrelationKind `json:"kind"`
	Value string          `json:"value"`
}

// SpanLink names an additional causal span, never an event parent or a second
// parent span. Identity is (trace_format, trace_id, span_id), including format.
type SpanLink struct {
	TraceID     string      `json:"trace_id"`
	SpanID      string      `json:"span_id"`
	TraceFormat TraceFormat `json:"trace_format"`
}

// CausalEnvelope preserves native identity beside explicit causal context.
// Strings must be nonempty valid UTF-8 so JSON cannot silently replace bytes;
// opaque values are never trimmed, parsed, case-folded or generated. Operation
// and Phase are source-defined labels, not a global operation vocabulary.
// Sequence is producer-local: validating one row proves only that it is nonzero,
// not monotonic ordering across producers or records.
//
// An absent span requires MappingUnknown and ParentUnknown. An absent parent
// never implies a root. ParentSpanID uses this envelope's trace and format;
// cross-trace and fan-in relationships belong in Links. Event-parent IDs are
// outside this contract. SourceSchema and SourceVersion identify the input
// contract; Schema identifies this envelope's wire version.
type CausalEnvelope struct {
	Schema        string        `json:"schema"`
	TraceID       string        `json:"trace_id"`
	TraceFormat   TraceFormat   `json:"trace_format"`
	SpanID        string        `json:"span_id,omitempty"`
	ParentSpanID  string        `json:"parent_span_id,omitempty"`
	ParentStatus  ParentStatus  `json:"parent_status"`
	Sequence      uint64        `json:"sequence"`
	LogicalID     string        `json:"logical_id"`
	Operation     string        `json:"operation"`
	Phase         string        `json:"phase"`
	Correlations  []Correlation `json:"correlations,omitempty"`
	Links         []SpanLink    `json:"links,omitempty"`
	SourceSchema  string        `json:"source_schema"`
	SourceVersion string        `json:"source_version"`
	MappingStatus MappingStatus `json:"mapping_status"`
}

// Descriptor enumerates the versioned fields and all closed vocabularies.
// Conditional field constraints are enforced by CausalEnvelope.Validate.
type Descriptor struct {
	Schema           string            `json:"schema"`
	EnvelopeSchema   string            `json:"envelope_schema"`
	Required         []string          `json:"required"`
	Optional         []string          `json:"optional"`
	TraceFormats     []TraceFormat     `json:"trace_formats"`
	MappingStatuses  []MappingStatus   `json:"mapping_statuses"`
	ParentStatuses   []ParentStatus    `json:"parent_statuses"`
	CorrelationKinds []CorrelationKind `json:"correlation_kinds"`
}

func CausalEnvelopeSchemaDescriptor() Descriptor {
	return Descriptor{
		Schema: DescriptorSchema, EnvelopeSchema: Schema,
		Required:         []string{"schema", "trace_id", "trace_format", "parent_status", "sequence", "logical_id", "operation", "phase", "source_schema", "source_version", "mapping_status"},
		Optional:         []string{"span_id", "parent_span_id", "correlations", "links"},
		TraceFormats:     []TraceFormat{TraceOpaque, TraceW3C},
		MappingStatuses:  []MappingStatus{MappingExact, MappingDerived, MappingUnknown},
		ParentStatuses:   []ParentStatus{ParentKnown, ParentRoot, ParentUnknown},
		CorrelationKinds: []CorrelationKind{CorrelationSession, CorrelationTurn, CorrelationRequest, CorrelationRun, CorrelationToolCall, CorrelationArtifact, CorrelationProducerNative},
	}
}

func (e CausalEnvelope) Validate() error {
	if e.Schema != Schema {
		return fmt.Errorf("causal envelope schema %q, want %q", e.Schema, Schema)
	}
	// Fixed traversal order keeps validation diagnostics deterministic.
	for _, field := range []struct{ name, value string }{
		{"trace_id", e.TraceID}, {"logical_id", e.LogicalID}, {"operation", e.Operation},
		{"phase", e.Phase}, {"source_schema", e.SourceSchema}, {"source_version", e.SourceVersion},
	} {
		if err := validString(field.name, field.value); err != nil {
			return err
		}
	}
	if e.Sequence == 0 {
		return fmt.Errorf("causal envelope sequence must be nonzero")
	}
	switch e.MappingStatus {
	case MappingExact, MappingDerived, MappingUnknown:
	default:
		return fmt.Errorf("unknown mapping_status %q", e.MappingStatus)
	}
	if err := validateIDs(e.TraceFormat, e.TraceID, e.SpanID, false); err != nil {
		return err
	}
	if e.SpanID == "" && (e.MappingStatus != MappingUnknown || e.ParentStatus != ParentUnknown) {
		return fmt.Errorf("absent span_id requires unknown mapping_status and parent_status")
	}
	switch e.ParentStatus {
	case ParentKnown:
		if e.SpanID == "" || e.ParentSpanID == "" {
			return fmt.Errorf("known parent requires span_id and parent_span_id")
		}
		if err := validateIDs(e.TraceFormat, e.TraceID, e.ParentSpanID, true); err != nil {
			return fmt.Errorf("parent_span_id: %w", err)
		}
		if e.SpanID == e.ParentSpanID {
			return fmt.Errorf("span cannot parent itself")
		}
	case ParentRoot:
		if e.SpanID == "" || e.ParentSpanID != "" {
			return fmt.Errorf("root requires span_id and absent parent_span_id")
		}
	case ParentUnknown:
		if e.ParentSpanID != "" {
			return fmt.Errorf("unknown parent requires absent parent_span_id")
		}
	default:
		return fmt.Errorf("unknown parent_status %q", e.ParentStatus)
	}
	correlations := make(map[Correlation]struct{}, len(e.Correlations))
	for i, correlation := range e.Correlations {
		if err := correlation.validate(); err != nil {
			return fmt.Errorf("correlation %d: %w", i, err)
		}
		if _, exists := correlations[correlation]; exists {
			return fmt.Errorf("duplicate correlation at %d", i)
		}
		correlations[correlation] = struct{}{}
	}
	links := make(map[SpanLink]struct{}, len(e.Links))
	for i, link := range e.Links {
		if err := validateIDs(link.TraceFormat, link.TraceID, link.SpanID, true); err != nil {
			return fmt.Errorf("link %d: %w", i, err)
		}
		if _, exists := links[link]; exists {
			return fmt.Errorf("duplicate link at %d", i)
		}
		links[link] = struct{}{}
	}
	return nil
}

func (c Correlation) validate() error {
	switch c.Kind {
	case CorrelationSession, CorrelationTurn, CorrelationRequest, CorrelationRun, CorrelationToolCall, CorrelationArtifact, CorrelationProducerNative:
	default:
		return fmt.Errorf("unknown correlation kind %q", c.Kind)
	}
	return validString("correlation value", c.Value)
}

func validString(name, value string) error {
	if value == "" || !utf8.ValidString(value) {
		return fmt.Errorf("%s must be nonempty UTF-8", name)
	}
	return nil
}

func validateIDs(format TraceFormat, trace, span string, requireSpan bool) error {
	if err := validString("trace_id", trace); err != nil {
		return err
	}
	if span != "" || requireSpan {
		if err := validString("span_id", span); err != nil {
			return err
		}
	}
	switch format {
	case TraceOpaque:
		return nil
	case TraceW3C:
		if !validW3C(trace, 32) || (span != "" && !validW3C(span, 16)) {
			return fmt.Errorf("W3C trace/span IDs require 32/16 lowercase nonzero hex digits")
		}
		return nil
	default:
		return fmt.Errorf("unknown trace_format %q", format)
	}
}

func validW3C(value string, size int) bool {
	if len(value) != size || strings.Trim(value, "0") == "" {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// Canonical returns a validated deep copy with deterministic set ordering.
// Empty and nil optional sets have the same omitted JSON representation.
func (e CausalEnvelope) Canonical() (CausalEnvelope, error) {
	if err := e.Validate(); err != nil {
		return CausalEnvelope{}, err
	}
	e.Correlations = slices.Clone(e.Correlations)
	e.Links = slices.Clone(e.Links)
	sort.Slice(e.Correlations, func(i, j int) bool {
		a, b := e.Correlations[i], e.Correlations[j]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		return a.Value < b.Value
	})
	sort.Slice(e.Links, func(i, j int) bool {
		a, b := e.Links[i], e.Links[j]
		if a.TraceFormat != b.TraceFormat {
			return a.TraceFormat < b.TraceFormat
		}
		if a.TraceID != b.TraceID {
			return a.TraceID < b.TraceID
		}
		return a.SpanID < b.SpanID
	})
	return e, nil
}

// MarshalJSON validates and canonicalizes without modifying the receiver.
// This is the schema's deterministic encoding, not RFC 8785 canonical JSON.
func (e CausalEnvelope) MarshalJSON() ([]byte, error) {
	canonical, err := e.Canonical()
	if err != nil {
		return nil, err
	}
	type wire CausalEnvelope
	return json.Marshal(wire(canonical))
}

// UnmarshalJSON rejects ambiguous/unknown fields and validates before replacing
// the receiver. Re-encoding the result always uses canonical ordering.
func (e *CausalEnvelope) UnmarshalJSON(data []byte) error {
	type wire CausalEnvelope
	var decoded wire
	descriptor := CausalEnvelopeSchemaDescriptor()
	if err := decodeObject(data, &decoded, append(descriptor.Required, descriptor.Optional...)); err != nil {
		return err
	}
	canonical, err := CausalEnvelope(decoded).Canonical()
	if err != nil {
		return err
	}
	*e = canonical
	return nil
}

func (c *Correlation) UnmarshalJSON(data []byte) error {
	type wire Correlation
	var decoded wire
	if err := decodeObject(data, &decoded, []string{"kind", "value"}); err != nil {
		return err
	}
	if err := Correlation(decoded).validate(); err != nil {
		return err
	}
	*c = Correlation(decoded)
	return nil
}

func (l *SpanLink) UnmarshalJSON(data []byte) error {
	type wire SpanLink
	var decoded wire
	if err := decodeObject(data, &decoded, []string{"trace_id", "span_id", "trace_format"}); err != nil {
		return err
	}
	if err := validateIDs(decoded.TraceFormat, decoded.TraceID, decoded.SpanID, true); err != nil {
		return err
	}
	*l = SpanLink(decoded)
	return nil
}

// encoding/json alone accepts duplicate keys and case-insensitive field aliases.
// Check the exact closed object vocabulary before decoding typed field values.
func decodeObject(data []byte, target any, allowed []string) error {
	if !utf8.Valid(data) {
		return fmt.Errorf("causal JSON must be UTF-8")
	}
	if err := validateSurrogates(data); err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	first, err := dec.Token()
	if err != nil {
		return err
	}
	if first != json.Delim('{') {
		return fmt.Errorf("causal JSON requires an object")
	}
	seen := make(map[string]bool, len(allowed))
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		if !ok || !slices.Contains(allowed, key) {
			return fmt.Errorf("unknown causal field %q", token)
		}
		if seen[key] {
			return fmt.Errorf("duplicate causal field %q", key)
		}
		seen[key] = true
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return err
		}
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	var extra json.RawMessage
	if err := dec.Decode(&extra); err != io.EOF {
		if err != nil {
			return err
		}
		return fmt.Errorf("trailing causal JSON value")
	}
	return json.Unmarshal(data, target)
}

// encoding/json replaces unpaired UTF-16 escapes with U+FFFD. Refuse them
// instead: an opaque identity must never acquire a replacement character.
func validateSurrogates(data []byte) error {
	for i := 0; i < len(data); i++ {
		if data[i] != '\\' {
			continue
		}
		i++ // Skip an escaped backslash/quote as a unit, not as a new escape.
		if i >= len(data) || data[i] != 'u' {
			continue
		}
		if i+4 >= len(data) {
			return fmt.Errorf("incomplete JSON Unicode escape")
		}
		value, err := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
		if err != nil {
			return fmt.Errorf("invalid JSON Unicode escape")
		}
		i += 4
		if value >= 0xd800 && value <= 0xdbff {
			if i+6 >= len(data) || string(data[i+1:i+3]) != `\u` {
				return fmt.Errorf("unpaired JSON surrogate escape")
			}
			low, err := strconv.ParseUint(string(data[i+3:i+7]), 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return fmt.Errorf("unpaired JSON surrogate escape")
			}
			i += 6
		} else if value >= 0xdc00 && value <= 0xdfff {
			return fmt.Errorf("unpaired JSON surrogate escape")
		}
	}
	return nil
}
