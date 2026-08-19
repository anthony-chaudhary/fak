package privacy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

const PolicySchema = "fak-privacy-policy/1"
const ReceiptSchema = "fak-privacy-receipt/1"

type Sink string

const (
	SinkLog       Sink = "log"
	SinkRetain    Sink = "retain"
	SinkExport    Sink = "export"
	SinkTelemetry Sink = "telemetry"
)

type Action string

const (
	ActionAllow  Action = "allow"
	ActionRedact Action = "redact"
	ActionDeny   Action = "deny"
	ActionExpire Action = "expire"
)

type SinkPolicy struct {
	Enabled      bool     `json:"enabled"`
	RedactFields []string `json:"redact_fields,omitempty"`
}
type RetentionPolicy struct {
	Enabled       bool     `json:"enabled"`
	MaxCount      int      `json:"max_count"`
	MaxAgeSeconds int64    `json:"max_age_seconds"`
	RedactFields  []string `json:"redact_fields,omitempty"`
}
type Policy struct {
	Schema    string          `json:"schema"`
	LocalOnly bool            `json:"local_only"`
	Log       SinkPolicy      `json:"log"`
	Retain    RetentionPolicy `json:"retain"`
	Export    SinkPolicy      `json:"export"`
	Telemetry SinkPolicy      `json:"telemetry"`
}
type Receipt struct {
	Schema         string   `json:"schema"`
	Sink           Sink     `json:"sink"`
	Action         Action   `json:"action"`
	Reason         string   `json:"reason"`
	ObservedAt     string   `json:"observed_at"`
	InputBytes     int      `json:"input_bytes"`
	OutputBytes    int      `json:"output_bytes"`
	RedactedFields []string `json:"redacted_fields"`
}
type Decision struct {
	Payload json.RawMessage
	Receipt Receipt
}

func DefaultPolicy() Policy {
	return Policy{Schema: PolicySchema, LocalOnly: true, Log: SinkPolicy{Enabled: true}, Retain: RetentionPolicy{Enabled: true, MaxCount: 1000, MaxAgeSeconds: 86400}, Export: SinkPolicy{Enabled: false}, Telemetry: SinkPolicy{Enabled: false}}
}
func ParsePolicy(raw []byte) (Policy, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var p Policy
	if err := dec.Decode(&p); err != nil {
		return Policy{}, err
	}
	if dec.More() {
		return Policy{}, fmt.Errorf("multiple privacy values")
	}
	if err := p.Validate(); err != nil {
		return Policy{}, err
	}
	return p, nil
}
func (p Policy) Validate() error {
	if p.Schema != PolicySchema {
		return fmt.Errorf("unsupported privacy schema")
	}
	if p.Retain.Enabled && (p.Retain.MaxCount < 1 || p.Retain.MaxCount > 1_000_000 || p.Retain.MaxAgeSeconds < 1 || p.Retain.MaxAgeSeconds > 31536000) {
		return fmt.Errorf("retention bounds invalid")
	}
	for _, fields := range [][]string{p.Log.RedactFields, p.Retain.RedactFields, p.Export.RedactFields, p.Telemetry.RedactFields} {
		for _, f := range fields {
			if strings.TrimSpace(f) == "" || strings.ContainsAny(f, "./[]") {
				return fmt.Errorf("invalid redact field")
			}
		}
	}
	return nil
}
func (p Policy) Evaluate(sink Sink, payload json.RawMessage, observed time.Time) (Decision, error) {
	if err := p.Validate(); err != nil {
		return Decision{}, err
	}
	if !json.Valid(payload) {
		return Decision{}, fmt.Errorf("privacy payload must be JSON")
	}
	enabled := false
	fields := []string{}
	switch sink {
	case SinkLog:
		enabled = p.Log.Enabled
		fields = p.Log.RedactFields
	case SinkRetain:
		enabled = p.Retain.Enabled
		fields = p.Retain.RedactFields
	case SinkExport:
		enabled = p.Export.Enabled
		fields = p.Export.RedactFields
	case SinkTelemetry:
		enabled = p.Telemetry.Enabled
		fields = p.Telemetry.RedactFields
	default:
		return Decision{}, fmt.Errorf("unknown sink")
	}
	receipt := Receipt{Schema: ReceiptSchema, Sink: sink, ObservedAt: observed.UTC().Format(time.RFC3339Nano), InputBytes: len(payload), RedactedFields: []string{}}
	if (p.LocalOnly && (sink == SinkExport || sink == SinkTelemetry)) || !enabled {
		receipt.Action = ActionDeny
		receipt.Reason = "sink_disabled"
		return Decision{Receipt: receipt}, nil
	}
	out := append(json.RawMessage(nil), payload...)
	if len(fields) > 0 {
		var value any
		if err := json.Unmarshal(payload, &value); err != nil {
			return Decision{}, err
		}
		redacted := map[string]bool{}
		redactValue(value, fields, redacted)
		out, _ = json.Marshal(value)
		for _, f := range fields {
			if redacted[strings.ToLower(f)] {
				receipt.RedactedFields = append(receipt.RedactedFields, f)
			}
		}
	}
	receipt.OutputBytes = len(out)
	if len(receipt.RedactedFields) > 0 {
		receipt.Action = ActionRedact
		receipt.Reason = "fields_redacted"
	} else {
		receipt.Action = ActionAllow
		receipt.Reason = "sink_allowed"
	}
	return Decision{Payload: out, Receipt: receipt}, nil
}
func redactValue(v any, fields []string, redacted map[string]bool) {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			match := false
			for _, f := range fields {
				if strings.EqualFold(k, f) {
					match = true
					redacted[strings.ToLower(f)] = true
					break
				}
			}
			if match {
				x[k] = "[REDACTED]"
			} else {
				redactValue(val, fields, redacted)
			}
		}
	case []any:
		for _, val := range x {
			redactValue(val, fields, redacted)
		}
	}
}

type Retained struct {
	At      time.Time
	Payload json.RawMessage
	Receipt Receipt
}
type Store struct {
	mu     sync.Mutex
	policy Policy
	rows   []Retained
}

func NewStore(p Policy) (*Store, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return &Store{policy: p}, nil
}
func (s *Store) Append(payload json.RawMessage, now time.Time) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, err := s.policy.Evaluate(SinkRetain, payload, now)
	if err != nil {
		return Receipt{}, err
	}
	s.expire(now)
	if d.Receipt.Action == ActionDeny {
		return d.Receipt, nil
	}
	s.rows = append(s.rows, Retained{now, d.Payload, d.Receipt})
	for len(s.rows) > s.policy.Retain.MaxCount {
		s.rows = s.rows[1:]
	}
	return d.Receipt, nil
}
func (s *Store) expire(now time.Time) {
	cut := now.Add(-time.Duration(s.policy.Retain.MaxAgeSeconds) * time.Second)
	i := 0
	for i < len(s.rows) && s.rows[i].At.Before(cut) {
		i++
	}
	s.rows = append([]Retained(nil), s.rows[i:]...)
}
func (s *Store) Rows(now time.Time) []Retained {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expire(now)
	return append([]Retained(nil), s.rows...)
}
