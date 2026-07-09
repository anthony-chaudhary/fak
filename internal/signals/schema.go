package signals

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// This is a deliberately small JSON Schema profile — enough to make a behavioral
// verdict a checkable contract without pulling a full draft-07 validator. It supports:
//
//   - top level: {"type":"object","properties":{...},"required":[...]}
//   - property types: "string", "boolean", "number", "integer", "object", "array"
//   - "enum": [...] on a string property (verdict must be one of the listed values)
//   - nested "object" properties (recursively, same subset)
//
// Anything outside this profile in a signal's schema is rejected at Validate time, so a
// signal can never carry a schema Run cannot actually enforce.

type schemaNode struct {
	Type       string                 `json:"type"`
	Properties map[string]*schemaNode `json:"properties"`
	Required   []string               `json:"required"`
	Enum       []json.RawMessage      `json:"enum"`
	Items      *schemaNode            `json:"items"`
}

var allowedTypes = map[string]bool{
	"string": true, "boolean": true, "number": true, "integer": true,
	"object": true, "array": true,
}

// validateSchemaDoc checks a schema document is itself within the supported profile.
func validateSchemaDoc(raw json.RawMessage) error {
	var node schemaNode
	if err := json.Unmarshal(raw, &node); err != nil {
		return fmt.Errorf("schema is not valid JSON: %w", err)
	}
	if node.Type == "" {
		return fmt.Errorf("schema: top-level \"type\" is required (want \"object\")")
	}
	return validateSchemaNode(&node, "schema")
}

func validateSchemaNode(n *schemaNode, path string) error {
	if !allowedTypes[n.Type] {
		return fmt.Errorf("%s: unsupported type %q", path, n.Type)
	}
	if n.Type == "object" {
		for name, prop := range n.Properties {
			if prop == nil {
				return fmt.Errorf("%s.%s: empty property schema", path, name)
			}
			if err := validateSchemaNode(prop, path+"."+name); err != nil {
				return err
			}
		}
		for _, req := range n.Required {
			if _, ok := n.Properties[req]; !ok {
				return fmt.Errorf("%s: required key %q is not among properties", path, req)
			}
		}
	}
	if n.Type == "array" && n.Items != nil {
		if err := validateSchemaNode(n.Items, path+"[]"); err != nil {
			return err
		}
	}
	return nil
}

// ValidateAgainstSchema checks a judge's verdict conforms to the signal's schema
// (within the supported profile). It is the gate that keeps an off-schema behavioral
// answer from entering the results ledger as if it were valid.
func ValidateAgainstSchema(schema, verdict json.RawMessage) error {
	var node schemaNode
	if err := json.Unmarshal(schema, &node); err != nil {
		return fmt.Errorf("schema unparseable: %w", err)
	}
	var value any
	if err := json.Unmarshal(verdict, &value); err != nil {
		return fmt.Errorf("verdict is not valid JSON: %w", err)
	}
	return checkValue(&node, value, "verdict")
}

func checkValue(n *schemaNode, value any, path string) error {
	switch n.Type {
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("%s: expected object, got %s", path, jsonKind(value))
		}
		for _, req := range n.Required {
			if _, present := obj[req]; !present {
				return fmt.Errorf("%s: missing required key %q", path, req)
			}
		}
		for name, prop := range n.Properties {
			v, present := obj[name]
			if !present {
				continue // only required keys must be present
			}
			if err := checkValue(prop, v, path+"."+name); err != nil {
				return err
			}
		}
	case "array":
		arr, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%s: expected array, got %s", path, jsonKind(value))
		}
		if n.Items != nil {
			for i, el := range arr {
				if err := checkValue(n.Items, el, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	case "string":
		s, ok := value.(string)
		if !ok {
			return fmt.Errorf("%s: expected string, got %s", path, jsonKind(value))
		}
		if len(n.Enum) > 0 && !enumContains(n.Enum, s) {
			return fmt.Errorf("%s: %q not in enum", path, s)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%s: expected boolean, got %s", path, jsonKind(value))
		}
	case "number":
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("%s: expected number, got %s", path, jsonKind(value))
		}
	case "integer":
		f, ok := value.(float64)
		if !ok {
			return fmt.Errorf("%s: expected integer, got %s", path, jsonKind(value))
		}
		if f != math.Trunc(f) {
			return fmt.Errorf("%s: expected integer, got fractional %g", path, f)
		}
	}
	return nil
}

func enumContains(enum []json.RawMessage, s string) bool {
	for _, e := range enum {
		var es string
		if err := json.Unmarshal(e, &es); err == nil && es == s {
			return true
		}
	}
	return false
}

func jsonKind(v any) string {
	switch v.(type) {
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64:
		return "number"
	case nil:
		return "null"
	default:
		return strings.TrimPrefix(fmt.Sprintf("%T", v), "interface ")
	}
}
