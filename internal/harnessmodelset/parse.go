package harnessmodelset

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
)

const maxIntentBytes = 1 << 20

var simpleJSONField = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// ParseJSON strictly decodes and validates one v1 intent. Unknown fields,
// duplicate object keys, missing required wire fields, trailing documents, and
// invalid hard constraints all return *ValidationError and no partial intent.
func ParseJSON(raw []byte) (Intent, error) {
	if len(raw) > maxIntentBytes {
		return Intent{}, validationError(diagnostic(
			CodeJSONInvalid, "$", "intent exceeds the 1 MiB admission limit", "reduce the role declaration before parsing",
		))
	}
	value, err := decodeDocument(raw)
	if err != nil {
		return Intent{}, err
	}

	diagnostics := duplicateFieldDiagnostics(raw)
	inspectIntentShape(value, &diagnostics)
	if err := validationError(diagnostics...); err != nil {
		return Intent{}, err
	}

	var intent Intent
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&intent); err != nil {
		path := "$"
		var typeErr *json.UnmarshalTypeError
		if errors.As(err, &typeErr) && typeErr.Field != "" {
			path = "$." + typeErr.Field
		}
		return Intent{}, validationError(diagnostic(
			CodeValueInvalid, path, "field does not use the JSON type required by the schema", "replace the value with the declared scalar, object, or array type",
		))
	}
	if err := intent.Validate(); err != nil {
		return Intent{}, err
	}
	return canonicalIntent(intent), nil
}

func decodeDocument(raw []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, validationError(diagnostic(
			CodeJSONInvalid, "$", "intent must be one complete JSON document", "repair the JSON syntax and retry",
		))
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, validationError(diagnostic(
			CodeJSONTrailing, "$", "intent must contain exactly one JSON value", "remove the trailing JSON value or bytes",
		))
	}
	return value, nil
}

func duplicateFieldDiagnostics(raw []byte) []Diagnostic {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var diagnostics []Diagnostic
	var walk func(string)
	walk = func(path string) {
		token, err := decoder.Token()
		if err != nil {
			return
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return
		}
		switch delim {
		case '{':
			seen := map[string]struct{}{}
			for decoder.More() {
				keyToken, err := decoder.Token()
				if err != nil {
					return
				}
				key, _ := keyToken.(string)
				childPath := jsonPathField(path, key)
				if _, exists := seen[key]; exists {
					diagnostics = append(diagnostics, diagnostic(
						CodeFieldDuplicate, childPath, "object field is declared more than once", "keep exactly one value for the field",
					))
				}
				seen[key] = struct{}{}
				walk(childPath)
			}
			_, _ = decoder.Token()
		case '[':
			for index := 0; decoder.More(); index++ {
				walk(fmt.Sprintf("%s[%d]", path, index))
			}
			_, _ = decoder.Token()
		}
	}
	walk("$")
	return diagnostics
}

type objectShape struct {
	allowed  []string
	required []string
}

var (
	intentShape = objectShape{allowed: []string{"roles", "schema"}, required: []string{"roles", "schema"}}
	roleShape   = objectShape{
		allowed:  []string{"alternatives", "evidence", "id", "preference", "required"},
		required: []string{"alternatives", "evidence", "id", "required"},
	}
	alternativeShape      = objectShape{allowed: []string{"capabilities", "id", "operational"}, required: []string{"id"}}
	modelRequirementShape = objectShape{allowed: []string{"family", "minimum_input_tokens", "modalities", "quantization", "structured_output", "tool_calling", "tool_protocol"}}
	operationalShape      = objectShape{allowed: []string{"accelerators", "license_allowlist", "locality", "max_memory_bytes", "platforms", "privacy", "runtime", "serving_protocol"}}
	preferenceShape       = objectShape{allowed: []string{"mode"}, required: []string{"mode"}}
	evidenceShape         = objectShape{allowed: []string{"max_age_hours", "required_kinds"}, required: []string{"max_age_hours"}}
)

func inspectIntentShape(value any, diagnostics *[]Diagnostic) {
	root := inspectObject(value, "$", intentShape, diagnostics)
	rolesValue, exists := root["roles"]
	if !exists {
		return
	}
	roles, ok := rolesValue.([]any)
	if !ok {
		*diagnostics = append(*diagnostics, invalidContainer("$.roles", "array"))
		return
	}
	for roleIndex, roleValue := range roles {
		rolePath := fmt.Sprintf("$.roles[%d]", roleIndex)
		role := inspectObject(roleValue, rolePath, roleShape, diagnostics)
		if preference, exists := role["preference"]; exists {
			inspectObject(preference, rolePath+".preference", preferenceShape, diagnostics)
		}
		if evidence, exists := role["evidence"]; exists {
			inspectObject(evidence, rolePath+".evidence", evidenceShape, diagnostics)
		}
		alternativesValue, exists := role["alternatives"]
		if !exists {
			continue
		}
		alternatives, ok := alternativesValue.([]any)
		if !ok {
			*diagnostics = append(*diagnostics, invalidContainer(rolePath+".alternatives", "array"))
			continue
		}
		for alternativeIndex, alternativeValue := range alternatives {
			alternativePath := fmt.Sprintf("%s.alternatives[%d]", rolePath, alternativeIndex)
			alternative := inspectObject(alternativeValue, alternativePath, alternativeShape, diagnostics)
			if capabilities, exists := alternative["capabilities"]; exists {
				inspectObject(capabilities, alternativePath+".capabilities", modelRequirementShape, diagnostics)
			}
			if operational, exists := alternative["operational"]; exists {
				inspectObject(operational, alternativePath+".operational", operationalShape, diagnostics)
			}
		}
	}
}

func inspectObject(value any, path string, shape objectShape, diagnostics *[]Diagnostic) map[string]any {
	object, ok := value.(map[string]any)
	if !ok {
		*diagnostics = append(*diagnostics, invalidContainer(path, "object"))
		return nil
	}
	allowed := make(map[string]struct{}, len(shape.allowed))
	for _, field := range shape.allowed {
		allowed[field] = struct{}{}
	}
	keys := make([]string, 0, len(object))
	for field := range object {
		keys = append(keys, field)
	}
	sort.Strings(keys)
	for _, field := range keys {
		if _, exists := allowed[field]; !exists {
			*diagnostics = append(*diagnostics, diagnostic(
				CodeFieldUnknown, jsonPathField(path, field), "field is not defined by "+SchemaV1, "remove the field or use a schema version that defines it",
			))
		}
	}
	for _, field := range shape.required {
		if _, exists := object[field]; !exists {
			*diagnostics = append(*diagnostics, diagnostic(
				CodeFieldRequired, jsonPathField(path, field), "field is required on the JSON wire", "declare the field explicitly",
			))
		}
	}
	return object
}

func invalidContainer(path, expected string) Diagnostic {
	return diagnostic(CodeValueInvalid, path, "field must be a JSON "+expected, "replace the field with the schema-required "+expected)
}

func jsonPathField(path, field string) string {
	if simpleJSONField.MatchString(field) {
		return path + "." + field
	}
	raw, _ := json.Marshal(field)
	return path + "[" + string(raw) + "]"
}
