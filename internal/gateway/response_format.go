package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Closed error-code vocabulary for front-door structured-output validation
// (oss-port-gateway-json-schema-validate). ADAPTed from TGI's validation half
// (huggingface/text-generation-inference router/src/validation.rs:341-406@b4adbf2,
// Apache-2.0): a malformed json_schema response_format is refused with HTTP 422
// before any upstream call. fak does NOT compile the schema to a grammar or
// meta-validate it against a draft; it refuses only shapes no JSON Schema draft
// allows, so every schema a client can legitimately send (no "type", open objects,
// root anyOf/oneOf/allOf, root $ref + $defs, boolean schemas) is forwarded verbatim.
const (
	errCodeInvalidResponseFormat     = "invalid_response_format"
	errCodeJSONSchemaMissing         = "json_schema_missing"
	errCodeJSONSchemaNotObject       = "json_schema_not_object"
	errCodeJSONSchemaInvalidType     = "json_schema_invalid_type"
	errCodeJSONSchemaInvalidProps    = "json_schema_invalid_properties"
	errCodeJSONSchemaInvalidRequired = "json_schema_invalid_required"
	errCodeJSONSchemaTooLarge        = "json_schema_too_large"
	errCodeJSONSchemaTooDeep         = "json_schema_too_deep"
)

// Structured-output schema caps (oss-port-gateway-json-schema-size-cap), ADAPTed
// from TGI's size guard on grammar inputs (router/src/validation.rs:341-406@b4adbf2,
// Apache-2.0). A ride engine compiles response_format.json_schema.schema into a
// decoding FSM/grammar; an unbounded schema is a cheap way to burn upstream compile
// time. The cap applies ONLY to the json_schema response_format's schema bytes
// (whitespace-trimmed, as sent) and its object/array nesting depth ({} is depth 1),
// never to tool definitions or other request fields.
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

// jsonNestingDepth returns the maximum object/array nesting depth of raw, scanning
// bytes (string-aware) and stopping early once the depth exceeds limit. Malformed
// JSON is left to the shape checks that follow.
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

// responseFormatFault is a closed refusal: code is the error code, schemaPath the
// JSON Pointer fragment ("#", "#/properties/a/type") of the offending schema node,
// empty when the fault is in the carrier rather than the schema.
type responseFormatFault struct {
	code, msg, schemaPath string
}

// rejectInvalidResponseFormat writes the 422 for a malformed response_format and
// reports whether it did.
func rejectInvalidResponseFormat(w http.ResponseWriter, raw json.RawMessage) bool {
	f := validateResponseFormat(raw)
	if f == nil {
		return false
	}
	var fields map[string]any
	if f.schemaPath != "" {
		fields = map[string]any{"schema_path": f.schemaPath}
	}
	writeErrCodeFields(w, http.StatusUnprocessableEntity, f.code, f.msg, fields)
	return true
}

func isJSONNull(b []byte) bool { return len(b) == 0 || bytes.Equal(b, []byte("null")) }

// validateResponseFormat checks an OpenAI `response_format` carrier. It returns nil
// when the request may proceed (absent, json_object, text, any other type the
// upstream owns, or a well-formed json_schema). The raw bytes are never rewritten.
func validateResponseFormat(raw json.RawMessage) *responseFormatFault {
	trimmed := bytes.TrimSpace(raw)
	if isJSONNull(trimmed) {
		return nil
	}
	var rf struct {
		Type       string          `json:"type"`
		JSONSchema json.RawMessage `json:"json_schema"`
	}
	if err := json.Unmarshal(trimmed, &rf); err != nil {
		return &responseFormatFault{code: errCodeInvalidResponseFormat, msg: "response_format must be a JSON object"}
	}
	if rf.Type != "json_schema" {
		return nil
	}
	inner := bytes.TrimSpace(rf.JSONSchema)
	if isJSONNull(inner) {
		return &responseFormatFault{code: errCodeJSONSchemaMissing, msg: "response_format.json_schema is required when type is json_schema"}
	}
	var js struct {
		Schema json.RawMessage `json:"schema"`
	}
	if inner[0] != '{' || json.Unmarshal(inner, &js) != nil {
		return &responseFormatFault{code: errCodeJSONSchemaNotObject, msg: "response_format.json_schema must be a JSON object"}
	}
	schema := bytes.TrimSpace(js.Schema)
	if isJSONNull(schema) {
		return &responseFormatFault{code: errCodeJSONSchemaMissing, msg: "response_format.json_schema.schema is required when type is json_schema"}
	}
	if maxBytes := jsonSchemaCap(os.Getenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES"), defaultJSONSchemaMaxBytes); len(schema) > maxBytes {
		return &responseFormatFault{code: errCodeJSONSchemaTooLarge, msg: "response_format.json_schema.schema is " + strconv.Itoa(len(schema)) + " bytes; the cap is " + strconv.Itoa(maxBytes)}
	}
	if maxDepth := jsonSchemaCap(os.Getenv("FAK_GATEWAY_JSON_SCHEMA_MAX_DEPTH"), defaultJSONSchemaMaxDepth); jsonNestingDepth(schema, maxDepth) > maxDepth {
		return &responseFormatFault{code: errCodeJSONSchemaTooDeep, msg: "response_format.json_schema.schema nests deeper than the cap of " + strconv.Itoa(maxDepth)}
	}
	if string(schema) == "true" || string(schema) == "false" {
		return nil
	}
	return checkSchemaNode(schema, "#")
}

var jsonSchemaTypes = map[string]bool{
	"string": true, "number": true, "integer": true, "boolean": true,
	"object": true, "array": true, "null": true,
}

// Keywords whose value is a subschema, an array of subschemas, or a map of name to
// subschema. Only these are walked: const/enum/default/examples hold data, not schemas.
var (
	subschemaKeywords = []string{
		"additionalProperties", "items", "additionalItems", "contains", "propertyNames",
		"not", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems",
	}
	subschemaArrayKeywords = []string{"anyOf", "oneOf", "allOf", "prefixItems", "items"}
	subschemaMapKeywords   = []string{"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)

func pointerToken(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~", "~0"), "/", "~1")
}

// checkSchemaNode refuses only what no JSON Schema draft allows at a schema node:
// a non-object/non-boolean schema, a "type" that is not a known type name or a
// non-empty array of them, a non-object "properties", or a "required" that is not
// an array of strings. Recursion is bounded by the depth cap checked beforehand.
func checkSchemaNode(raw []byte, path string) *responseFormatFault {
	var node map[string]json.RawMessage
	if len(raw) == 0 || raw[0] != '{' || json.Unmarshal(raw, &node) != nil {
		return &responseFormatFault{code: errCodeJSONSchemaNotObject, msg: "schema must be a JSON object or boolean", schemaPath: path}
	}
	if t, ok := node["type"]; ok && !validSchemaType(t) {
		return &responseFormatFault{code: errCodeJSONSchemaInvalidType, msg: "\"type\" must be a JSON type name or a non-empty array of them", schemaPath: path + "/type"}
	}
	if p, ok := node["properties"]; ok {
		var props map[string]json.RawMessage
		if t := bytes.TrimSpace(p); len(t) == 0 || t[0] != '{' || json.Unmarshal(t, &props) != nil {
			return &responseFormatFault{code: errCodeJSONSchemaInvalidProps, msg: "\"properties\" must be an object", schemaPath: path + "/properties"}
		}
	}
	if r, ok := node["required"]; ok {
		var req []string
		if t := bytes.TrimSpace(r); len(t) == 0 || t[0] != '[' || json.Unmarshal(t, &req) != nil {
			return &responseFormatFault{code: errCodeJSONSchemaInvalidRequired, msg: "\"required\" must be an array of strings", schemaPath: path + "/required"}
		}
	}
	for _, kw := range subschemaKeywords {
		if v, ok := node[kw]; ok {
			if f := checkSubschema(v, path+"/"+pointerToken(kw)); f != nil {
				return f
			}
		}
	}
	for _, kw := range subschemaArrayKeywords {
		var arr []json.RawMessage
		if v, ok := node[kw]; ok && json.Unmarshal(v, &arr) == nil {
			for i, el := range arr {
				if f := checkSubschema(el, path+"/"+pointerToken(kw)+"/"+strconv.Itoa(i)); f != nil {
					return f
				}
			}
		}
	}
	for _, kw := range subschemaMapKeywords {
		var m map[string]json.RawMessage
		if v, ok := node[kw]; ok && json.Unmarshal(v, &m) == nil {
			names := make([]string, 0, len(m))
			for name := range m {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				if f := checkSubschema(m[name], path+"/"+pointerToken(kw)+"/"+pointerToken(name)); f != nil {
					return f
				}
			}
		}
	}
	return nil
}

// checkSubschema walks a nested position only when it holds an object schema;
// boolean subschemas are valid and other shapes are left to the ride engine.
func checkSubschema(raw json.RawMessage, path string) *responseFormatFault {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || t[0] != '{' {
		return nil
	}
	return checkSchemaNode(t, path)
}

func validSchemaType(raw json.RawMessage) bool {
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return jsonSchemaTypes[one]
	}
	var many []string
	if json.Unmarshal(raw, &many) != nil || len(many) == 0 {
		return false
	}
	for _, s := range many {
		if !jsonSchemaTypes[s] {
			return false
		}
	}
	return true
}
