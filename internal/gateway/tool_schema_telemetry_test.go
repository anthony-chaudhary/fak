package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolSchemaTelemetryWitness(t *testing.T) {
	// First witness requirements (#9911):
	// 1. Send invalid JSON -> outcome = invalid_json.
	// 2. Send unknown tool -> outcome = unknown_tool.
	// 3. Send schema mismatch -> outcome = schema_mismatch.
	// 4. Send valid call -> outcome = valid.
	// 5. Prove counters distinguish each outcome cleanly.
	// 6. Prove telemetry events and logs contain NO supplied argument values (privacy-bounded).

	schemas := []ToolSchemaDeclaration{
		{
			Name: "search_kb",
			ParamTypes: map[string]string{
				"query": "string",
				"limit": "number",
			},
			Required: []string{"query"},
		},
	}

	tracker := NewToolSchemaTelemetryTracker(schemas)

	secretQuery := "SECRET_PROMPT_PAYLOAD_ACCOUNT_NUMBER_98765"
	secretMalformed := "SECRET_BROKEN_SYNTAX_KEY_4321"

	// 1. Invalid JSON
	_, _ = tracker.AuditToolCall("search_kb", "{query: '"+secretMalformed+"'}") // unquoted JSON key -> invalid JSON

	// 2. Unknown tool
	_, _ = tracker.AuditToolCall("super_secret_tool", `{"action":"execute"}`)

	// 3. Schema mismatch: missing required "query"
	_, _ = tracker.AuditToolCall("search_kb", `{"limit": 10}`)

	// 3b. Schema mismatch: wrong type for "limit" (string instead of number)
	_, _ = tracker.AuditToolCall("search_kb", `{"query":"`+secretQuery+`", "limit": "not_a_number"}`)

	// 4. Valid call
	_, _ = tracker.AuditToolCall("search_kb", `{"query":"`+secretQuery+`", "limit": 5}`)

	// 5. Verify counters distinguish outcomes
	counts := tracker.Counts()
	if counts[ToolConformanceInvalidJSON] != 1 {
		t.Fatalf("expected 1 invalid_json, got %d", counts[ToolConformanceInvalidJSON])
	}
	if counts[ToolConformanceUnknownTool] != 1 {
		t.Fatalf("expected 1 unknown_tool, got %d", counts[ToolConformanceUnknownTool])
	}
	if counts[ToolConformanceSchemaMismatch] != 2 {
		t.Fatalf("expected 2 schema_mismatch, got %d", counts[ToolConformanceSchemaMismatch])
	}
	if counts[ToolConformanceValid] != 1 {
		t.Fatalf("expected 1 valid, got %d", counts[ToolConformanceValid])
	}

	// 6. Prove telemetry logs contain NO supplied values
	events := tracker.Events()
	eventsJSON, err := json.Marshal(events)
	if err != nil {
		t.Fatal(err)
	}

	logStr := string(eventsJSON)
	if strings.Contains(logStr, secretQuery) {
		t.Fatalf("privacy violation: telemetry logs contain secret query value %q", secretQuery)
	}
	if strings.Contains(logStr, secretMalformed) {
		t.Fatalf("privacy violation: telemetry logs contain malformed secret %q", secretMalformed)
	}
	if strings.Contains(logStr, "not_a_number") {
		t.Fatalf("privacy violation: telemetry logs contain supplied parameter value %q", "not_a_number")
	}

	// Logs should contain key names, tool names, and structural violation tags only
	if !strings.Contains(logStr, "search_kb") {
		t.Fatal("expected log to contain tool name search_kb")
	}
	if !strings.Contains(logStr, "type_mismatch:limit") {
		t.Fatal("expected log to contain structural violation code type_mismatch:limit")
	}
}
