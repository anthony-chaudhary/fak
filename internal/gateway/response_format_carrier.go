package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// rejectInvalidResponseFormatCarrier checks only the structured-output envelope.
// Schema keywords remain the upstream's contract: their meaning differs across
// JSON Schema drafts. Object and boolean schemas pass through without rewriting.
func rejectInvalidResponseFormatCarrier(w http.ResponseWriter, raw json.RawMessage) bool {
	reject := func(code, message, path string) bool {
		var fields map[string]any
		if path != "" {
			fields = map[string]any{"schema_path": path}
		}
		writeErrCodeFields(w, http.StatusUnprocessableEntity, code, message, fields)
		return true
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return false
	}
	var format struct {
		Type       string          `json:"type"`
		JSONSchema json.RawMessage `json:"json_schema"`
	}
	if json.Unmarshal(trimmed, &format) != nil {
		return reject("invalid_response_format", "response_format must be a JSON object", "")
	}
	if format.Type != "json_schema" {
		return false
	}
	inner := bytes.TrimSpace(format.JSONSchema)
	if len(inner) == 0 || bytes.Equal(inner, []byte("null")) {
		return reject("json_schema_missing", "response_format.json_schema is required when type is json_schema", "")
	}
	var carrier struct {
		Schema json.RawMessage `json:"schema"`
	}
	if inner[0] != '{' || json.Unmarshal(inner, &carrier) != nil {
		return reject("json_schema_not_object", "response_format.json_schema must be a JSON object", "")
	}
	schema := bytes.TrimSpace(carrier.Schema)
	if len(schema) == 0 || bytes.Equal(schema, []byte("null")) {
		return reject("json_schema_missing", "response_format.json_schema.schema is required when type is json_schema", "")
	}
	if schema[0] != '{' && !bytes.Equal(schema, []byte("true")) && !bytes.Equal(schema, []byte("false")) {
		return reject("json_schema_not_object", "schema must be a JSON object or boolean", "#")
	}
	return false
}
