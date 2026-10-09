package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// rejectInvalidResponseFormatCarrier checks the envelope and schema resource caps.
// Schema keywords remain the upstream's contract: their meaning differs across
// JSON Schema drafts. Admitted schemas pass through without rewriting.
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
	if maxBytes := jsonSchemaCap(os.Getenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES"), defaultJSONSchemaMaxBytes); len(schema) > maxBytes {
		return reject("json_schema_too_large", "response_format.json_schema.schema is "+strconv.Itoa(len(schema))+" bytes; the cap is "+strconv.Itoa(maxBytes), "")
	}
	if maxDepth := jsonSchemaCap(os.Getenv("FAK_GATEWAY_JSON_SCHEMA_MAX_DEPTH"), defaultJSONSchemaMaxDepth); jsonNestingDepth(schema, maxDepth) > maxDepth {
		return reject("json_schema_too_deep", "response_format.json_schema.schema nests deeper than the cap of "+strconv.Itoa(maxDepth), "")
	}
	if schema[0] != '{' && !bytes.Equal(schema, []byte("true")) && !bytes.Equal(schema, []byte("false")) {
		return reject("json_schema_not_object", "schema must be a JSON object or boolean", "#")
	}
	return false
}

// Structured-output resource caps adapted from PR #13718's TGI-style grammar-input
// guard (huggingface/text-generation-inference router/src/validation.rs:341-406
// @b4adbf2, Apache-2.0). Only response_format.json_schema.schema is bounded: its
// whitespace-trimmed bytes and all object/array nesting, including data keywords.
// Tool definitions and other fields keep their existing whole-request limit.
const (
	defaultJSONSchemaMaxBytes = 64 << 10
	defaultJSONSchemaMaxDepth = 32
)

// jsonSchemaCap parses a positive-integer cap override; anything else keeps def.
func jsonSchemaCap(raw string, def int) int {
	if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil && n > 0 {
		return n
	}
	return def
}

// jsonNestingDepth scans already-decoded JSON without interpreting schema keywords
// or allocating a recursive tree. An empty object is depth 1, a boolean is depth 0,
// and brackets inside strings do not count. Stop as soon as the cap is exceeded.
func jsonNestingDepth(raw []byte, limit int) int {
	depth, maxDepth := 0, 0
	inString, escaped := false, false
	for _, c := range raw {
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{', '[':
			depth++
			if depth > maxDepth {
				maxDepth = depth
				if maxDepth > limit {
					return maxDepth
				}
			}
		case '}', ']':
			depth--
		}
	}
	return maxDepth
}
