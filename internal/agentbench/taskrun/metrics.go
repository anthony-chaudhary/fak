package taskrun

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// identicalRunThreshold is the run of identical consecutive tool calls (same name
// and canonical arguments) that marks a trial as looping.
const identicalRunThreshold = 3

// ToolCallMetrics reduces one trial's emitted tool calls. Validity is nil when
// the trial emitted no tool calls, because 1 - 0/0 is not a measurement.
type ToolCallMetrics struct {
	Measured        bool     `json:"measured"`
	Total           int      `json:"tool_calls_total"`
	Invalid         int      `json:"tool_calls_invalid"`
	InvalidJSON     int      `json:"tool_calls_invalid_json"`
	UnknownTool     int      `json:"tool_calls_unknown_tool"`
	SchemaInvalid   int      `json:"tool_calls_schema_invalid"`
	Validity        *float64 `json:"tool_call_validity"`
	MaxIdenticalRun int      `json:"max_identical_consecutive_calls"`
	Repeating       bool     `json:"repeat_loop"`
	TurnCapHit      bool     `json:"turn_cap_hit"`
	Loop            bool     `json:"loop"`
}

// TrialSummary folds per-trial outcomes. Rates are nil when their denominator is
// zero: AcceptRate over trials, StuckRate over trials whose tool calls were measured.
type TrialSummary struct {
	Trials           int      `json:"trials"`
	Accepted         int      `json:"accepted"`
	AcceptRate       *float64 `json:"accept_rate"`
	MeasuredTrials   int      `json:"measured_trials"`
	ToolCallsTotal   int      `json:"tool_calls_total"`
	ToolCallsInvalid int      `json:"tool_calls_invalid"`
	ToolCallValidity *float64 `json:"tool_call_validity"`
	Loops            int      `json:"loops"`
	StuckRate        *float64 `json:"loop_rate"`
}

const (
	toolCallValid = iota
	toolCallInvalidJSON
	toolCallUnknownTool
	toolCallSchemaInvalid
)

func reduceToolCalls(turns []PlannerTurn, catalog []agent.ToolDef, turnCap int) ToolCallMetrics {
	schemas := make(map[string]json.RawMessage, len(catalog))
	for _, def := range catalog {
		schemas[def.Function.Name] = def.Function.Parameters
	}
	m := ToolCallMetrics{Measured: true}
	run, last := 0, ""
	for _, turn := range turns {
		for _, call := range turn.ToolCalls {
			m.Total++
			switch classifyToolCall(call.Function.Name, call.Function.Arguments, schemas) {
			case toolCallInvalidJSON:
				m.InvalidJSON++
			case toolCallUnknownTool:
				m.UnknownTool++
			case toolCallSchemaInvalid:
				m.SchemaInvalid++
			}
			key := call.Function.Name + "\x00" + canonicalArgs(call.Function.Arguments)
			if run > 0 && key == last {
				run++
			} else {
				run, last = 1, key
			}
			if run > m.MaxIdenticalRun {
				m.MaxIdenticalRun = run
			}
		}
	}
	m.Invalid = m.InvalidJSON + m.UnknownTool + m.SchemaInvalid
	if m.Total > 0 {
		m.Validity = ratio(m.Total-m.Invalid, m.Total)
	}
	m.Repeating = m.MaxIdenticalRun >= identicalRunThreshold
	m.TurnCapHit = turnCap > 0 && len(turns) >= turnCap && len(turns[len(turns)-1].ToolCalls) > 0
	m.Loop = m.Repeating || m.TurnCapHit
	return m
}

func SummarizeTrials(tasks []TaskReceipt) TrialSummary {
	a := TrialSummary{Trials: len(tasks)}
	for _, task := range tasks {
		if task.Accepted {
			a.Accepted++
		}
		if !task.ToolCalls.Measured {
			continue
		}
		a.MeasuredTrials++
		a.ToolCallsTotal += task.ToolCalls.Total
		a.ToolCallsInvalid += task.ToolCalls.Invalid
		if task.ToolCalls.Loop {
			a.Loops++
		}
	}
	a.AcceptRate = ratio(a.Accepted, a.Trials)
	a.ToolCallValidity = ratio(a.ToolCallsTotal-a.ToolCallsInvalid, a.ToolCallsTotal)
	a.StuckRate = ratio(a.Loops, a.MeasuredTrials)
	return a
}

func ratio(num, den int) *float64 {
	if den <= 0 {
		return nil
	}
	v := math.Round(float64(num)/float64(den)*1e6) / 1e6
	return &v
}

func classifyToolCall(name, rawArgs string, schemas map[string]json.RawMessage) int {
	args, ok := decodeArgs(rawArgs)
	if !ok {
		return toolCallInvalidJSON
	}
	schema, known := schemas[name]
	if !known {
		return toolCallUnknownTool
	}
	if !argsMatchSchema(args, schema) {
		return toolCallSchemaInvalid
	}
	return toolCallValid
}

// decodeArgs accepts a JSON object; blank arguments mean an empty object, which
// is how OpenAI-compatible servers render a call with no parameters.
func decodeArgs(raw string) (map[string]json.RawMessage, bool) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]json.RawMessage{}, true
	}
	var args map[string]json.RawMessage
	if err := json.Unmarshal([]byte(trimmed), &args); err != nil || args == nil {
		return nil, false
	}
	return args, true
}

func argsMatchSchema(args map[string]json.RawMessage, rawSchema json.RawMessage) bool {
	if len(bytes.TrimSpace(rawSchema)) == 0 {
		return true
	}
	var schema struct {
		Properties           map[string]json.RawMessage `json:"properties"`
		Required             []string                   `json:"required"`
		AdditionalProperties *bool                      `json:"additionalProperties"`
	}
	if err := json.Unmarshal(rawSchema, &schema); err != nil {
		return true
	}
	for _, name := range schema.Required {
		if _, ok := args[name]; !ok {
			return false
		}
	}
	for name, value := range args {
		property, declared := schema.Properties[name]
		if !declared {
			if schema.AdditionalProperties != nil && !*schema.AdditionalProperties {
				return false
			}
			continue
		}
		if !valueMatchesTypes(value, schemaTypes(property)) {
			return false
		}
	}
	return true
}

func schemaTypes(property json.RawMessage) []string {
	var p struct {
		Type json.RawMessage `json:"type"`
	}
	if json.Unmarshal(property, &p) != nil || len(p.Type) == 0 {
		return nil
	}
	var one string
	if json.Unmarshal(p.Type, &one) == nil {
		return []string{one}
	}
	var many []string
	_ = json.Unmarshal(p.Type, &many)
	return many
}

func valueMatchesTypes(value json.RawMessage, types []string) bool {
	if len(types) == 0 {
		return true
	}
	var decoded any
	if json.Unmarshal(value, &decoded) != nil {
		return false
	}
	for _, typ := range types {
		switch v := decoded.(type) {
		case nil:
			if typ == "null" {
				return true
			}
		case bool:
			if typ == "boolean" {
				return true
			}
		case float64:
			if typ == "number" || (typ == "integer" && v == math.Trunc(v)) {
				return true
			}
		case string:
			if typ == "string" {
				return true
			}
		case []any:
			if typ == "array" {
				return true
			}
		case map[string]any:
			if typ == "object" {
				return true
			}
		}
	}
	return false
}

// canonicalArgs re-encodes valid JSON with sorted object keys so that calls that
// differ only in key order or whitespace compare equal.
func canonicalArgs(raw string) string {
	var decoded any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return strings.TrimSpace(raw)
	}
	b, err := json.Marshal(decoded)
	if err != nil {
		return strings.TrimSpace(raw)
	}
	return string(b)
}
