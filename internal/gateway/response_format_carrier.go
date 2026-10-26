package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"strconv"
	"strings"
)

type responseFormatError struct {
	code, message, path string
}

// validateResponseFormatCarrier checks the envelope and schema resource caps.
// Schema keywords remain the upstream's contract: their meaning differs across
// JSON Schema drafts. Admitted schemas pass through without rewriting.
func validateResponseFormatCarrier(raw json.RawMessage, maxBytes, maxDepth int) *responseFormatError {
	reject := func(code, message, path string) *responseFormatError {
		return &responseFormatError{code: code, message: message, path: path}
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	// The raw object is forwarded unchanged. Refuse ambiguous recognized keys so
	// case-folding or duplicate-key choices cannot change which value is checked.
	format, ok := responseFormatMembers(trimmed, "type", "json_schema")
	if !ok {
		return reject("invalid_response_format", "response_format must be a JSON object", "")
	}
	var typ string
	if value, ok := format["type"]; ok && json.Unmarshal(value, &typ) != nil {
		return reject("invalid_response_format", "response_format must be a JSON object", "")
	}
	if typ != "json_schema" {
		return nil
	}
	inner := bytes.TrimSpace(format["json_schema"])
	if len(inner) == 0 || bytes.Equal(inner, []byte("null")) {
		return reject("json_schema_missing", "response_format.json_schema is required when type is json_schema", "")
	}
	if inner[0] != '{' {
		return reject("json_schema_not_object", "response_format.json_schema must be a JSON object", "")
	}
	carrier, ok := responseFormatMembers(inner, "schema")
	if !ok {
		return reject("invalid_response_format", "response_format.json_schema has invalid or ambiguous members", "")
	}
	schema := bytes.TrimSpace(carrier["schema"])
	if len(schema) == 0 || bytes.Equal(schema, []byte("null")) {
		return reject("json_schema_missing", "response_format.json_schema.schema is required when type is json_schema", "")
	}
	if len(schema) > maxBytes {
		return reject("json_schema_too_large", "response_format.json_schema.schema is "+strconv.Itoa(len(schema))+" bytes; the cap is "+strconv.Itoa(maxBytes), "")
	}
	if jsonNestingDepth(schema, maxDepth) > maxDepth {
		return reject("json_schema_too_deep", "response_format.json_schema.schema nests deeper than the cap of "+strconv.Itoa(maxDepth), "")
	}
	if schema[0] != '{' && !bytes.Equal(schema, []byte("true")) && !bytes.Equal(schema, []byte("false")) {
		return reject("json_schema_not_object", "schema must be a JSON object or boolean", "#")
	}
	return nil
}

// responseFormatMembers selects exact carrier keys and refuses their duplicates
// or case aliases. Unknown fields remain upstream-owned; schema keyword objects
// are decoded as raw values and are not constrained by this envelope check.
func responseFormatMembers(raw []byte, keys ...string) (map[string]json.RawMessage, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil, false
	}
	members := make(map[string]json.RawMessage, len(keys))
	for decoder.More() {
		token, err := decoder.Token()
		key, ok := token.(string)
		if err != nil || !ok {
			return nil, false
		}
		recognized := false
		for _, canonical := range keys {
			if !strings.EqualFold(key, canonical) {
				continue
			}
			if key != canonical {
				return nil, false
			}
			if _, duplicate := members[key]; duplicate {
				return nil, false
			}
			recognized = true
			break
		}
		var value json.RawMessage
		if decoder.Decode(&value) != nil {
			return nil, false
		}
		if recognized {
			members[key] = value
		}
	}
	if token, err := decoder.Token(); err != nil || token != json.Delim('}') {
		return nil, false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, false
	}
	return members, true
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
