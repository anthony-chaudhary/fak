package gateway

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestResponseFormatCapsUseExactWireKeys(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, raw string
	}{
		{"type", `{"type":"json_schema","json_schema":{"schema":{"type":"object"}},"Type":"json_object"}`},
		{"carrier", `{"type":"json_schema","json_schema":{"schema":{"type":"object"}},"JSON_SCHEMA":{"schema":true}}`},
		{"schema", `{"type":"json_schema","json_schema":{"schema":{"type":"object"},"Schema":true}}`},
		{"duplicate_type", `{"type":"json_schema","json_schema":{"schema":{"type":"object"}},"type":"json_object"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := json.RawMessage(tc.raw)
			before := string(raw)
			refusal := validateResponseFormatCarrier(raw, 8, defaultJSONSchemaMaxDepth)
			if refusal == nil || refusal.code != "invalid_response_format" {
				t.Fatalf("ambiguous carrier can hide an over-cap schema, got refusal %+v", refusal)
			}
			if string(raw) != before {
				t.Fatal("validation rewrote the forwarded carrier")
			}
		})
	}
	// Existing valid boolean and type-less object schemas remain provider-owned.
	for _, tc := range []struct {
		schema   string
		maxBytes int
	}{
		{`true`, 8},
		{`{"properties":{}}`, 32},
	} {
		raw := json.RawMessage(`{"type":"json_schema","json_schema":{"schema":` + tc.schema + `,"name":"answer"}}`)
		if refusal := validateResponseFormatCarrier(raw, tc.maxBytes, defaultJSONSchemaMaxDepth); refusal != nil {
			t.Fatalf("bounded exact wire schema refused: %+v", refusal)
		}
	}
}
