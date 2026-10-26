package gateway

import (
	"encoding/json"
	"net/http"
	"os"
)

// rejectInvalidResponseFormatCarrier applies the shared ingress validator before
// dispatch, preserving the closed 422 envelope and deployment cap overrides.
func rejectInvalidResponseFormatCarrier(w http.ResponseWriter, raw json.RawMessage) bool {
	maxBytes := jsonSchemaCap(os.Getenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES"), defaultJSONSchemaMaxBytes)
	maxDepth := jsonSchemaCap(os.Getenv("FAK_GATEWAY_JSON_SCHEMA_MAX_DEPTH"), defaultJSONSchemaMaxDepth)
	refusal := validateResponseFormatCarrier(raw, maxBytes, maxDepth)
	if refusal == nil {
		return false
	}
	var fields map[string]any
	if refusal.path != "" {
		fields = map[string]any{"schema_path": refusal.path}
	}
	writeErrCodeFields(w, http.StatusUnprocessableEntity, refusal.code, refusal.message, fields)
	return true
}
