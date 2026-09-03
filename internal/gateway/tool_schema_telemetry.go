package gateway

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// ToolSchemaConformanceOutcome is the closed outcome vocabulary for tool-call validation telemetry.
type ToolSchemaConformanceOutcome string

const (
	ToolConformanceValid          ToolSchemaConformanceOutcome = "valid"
	ToolConformanceInvalidJSON    ToolSchemaConformanceOutcome = "invalid_json"
	ToolConformanceUnknownTool    ToolSchemaConformanceOutcome = "unknown_tool"
	ToolConformanceSchemaMismatch ToolSchemaConformanceOutcome = "schema_mismatch"
)

// ToolSchemaDeclaration specifies expected parameter names and types for a tool.
type ToolSchemaDeclaration struct {
	Name       string            `json:"name"`
	ParamTypes map[string]string `json:"param_types"` // paramName -> "string" | "number" | "bool" | "object" | "array"
	Required   []string          `json:"required,omitempty"`
}

// ToolSchemaTelemetryEvent captures privacy-bounded telemetry without recording parameter values or validator prose.
type ToolSchemaTelemetryEvent struct {
	ToolName  string                       `json:"tool_name"`
	Outcome   ToolSchemaConformanceOutcome `json:"outcome"`
	ParamKeys []string                     `json:"param_keys,omitempty"` // only keys/names, NEVER values
	Violation string                       `json:"violation,omitempty"`  // structural code e.g. "missing:query" or "type:count"
}

// ToolSchemaTelemetryTracker counts conformance outcomes and maintains privacy-sanitized telemetry logs.
type ToolSchemaTelemetryTracker struct {
	mu           sync.Mutex
	declarations map[string]ToolSchemaDeclaration
	counts       map[ToolSchemaConformanceOutcome]int64
	events       []ToolSchemaTelemetryEvent
}

// NewToolSchemaTelemetryTracker constructs a telemetry tracker with declared tool schemas.
func NewToolSchemaTelemetryTracker(schemas []ToolSchemaDeclaration) *ToolSchemaTelemetryTracker {
	decls := make(map[string]ToolSchemaDeclaration, len(schemas))
	for _, s := range schemas {
		decls[s.Name] = s
	}
	return &ToolSchemaTelemetryTracker{
		declarations: decls,
		counts:       make(map[ToolSchemaConformanceOutcome]int64),
	}
}

// AuditToolCall validates a proposed tool call and emits privacy-bounded telemetry.
// The rawArgs payload is never copied into telemetry events or logs.
func (t *ToolSchemaTelemetryTracker) AuditToolCall(toolName string, rawArgs string) (ToolSchemaConformanceOutcome, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// 1. Invalid JSON check
	var parsed map[string]any
	if err := json.Unmarshal([]byte(rawArgs), &parsed); err != nil {
		t.counts[ToolConformanceInvalidJSON]++
		t.events = append(t.events, ToolSchemaTelemetryEvent{
			ToolName:  toolName,
			Outcome:   ToolConformanceInvalidJSON,
			Violation: "syntax_error",
		})
		return ToolConformanceInvalidJSON, fmt.Errorf("tool call invalid json")
	}

	// 2. Unknown tool check
	decl, known := t.declarations[toolName]
	if !known {
		t.counts[ToolConformanceUnknownTool]++
		t.events = append(t.events, ToolSchemaTelemetryEvent{
			ToolName:  toolName,
			Outcome:   ToolConformanceUnknownTool,
			Violation: "unregistered_tool",
		})
		return ToolConformanceUnknownTool, fmt.Errorf("unknown tool %q", toolName)
	}

	// Extract keys only (never values)
	keys := make([]string, 0, len(parsed))
	for k := range parsed {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// 3. Schema conformance check
	for _, reqKey := range decl.Required {
		if _, exists := parsed[reqKey]; !exists {
			t.counts[ToolConformanceSchemaMismatch]++
			t.events = append(t.events, ToolSchemaTelemetryEvent{
				ToolName:  toolName,
				Outcome:   ToolConformanceSchemaMismatch,
				ParamKeys: keys,
				Violation: "missing_required:" + reqKey,
			})
			return ToolConformanceSchemaMismatch, fmt.Errorf("missing required parameter %q", reqKey)
		}
	}

	for k, val := range parsed {
		wantType, specified := decl.ParamTypes[k]
		if specified && !matchesType(val, wantType) {
			t.counts[ToolConformanceSchemaMismatch]++
			t.events = append(t.events, ToolSchemaTelemetryEvent{
				ToolName:  toolName,
				Outcome:   ToolConformanceSchemaMismatch,
				ParamKeys: keys,
				Violation: "type_mismatch:" + k,
			})
			return ToolConformanceSchemaMismatch, fmt.Errorf("type mismatch on parameter %q", k)
		}
	}

	t.counts[ToolConformanceValid]++
	t.events = append(t.events, ToolSchemaTelemetryEvent{
		ToolName:  toolName,
		Outcome:   ToolConformanceValid,
		ParamKeys: keys,
	})
	return ToolConformanceValid, nil
}

func matchesType(val any, want string) bool {
	if val == nil {
		return true
	}
	switch strings.ToLower(want) {
	case "string":
		_, ok := val.(string)
		return ok
	case "number":
		_, ok := val.(float64)
		return ok
	case "bool":
		_, ok := val.(bool)
		return ok
	case "array":
		_, ok := val.([]any)
		return ok
	case "object":
		_, ok := val.(map[string]any)
		return ok
	default:
		return true
	}
}

// Counts returns a copy of outcome counters.
func (t *ToolSchemaTelemetryTracker) Counts() map[ToolSchemaConformanceOutcome]int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	res := make(map[ToolSchemaConformanceOutcome]int64, len(t.counts))
	for k, v := range t.counts {
		res[k] = v
	}
	return res
}

// Events returns the recorded privacy-bounded telemetry events.
func (t *ToolSchemaTelemetryTracker) Events() []ToolSchemaTelemetryEvent {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]ToolSchemaTelemetryEvent(nil), t.events...)
}
