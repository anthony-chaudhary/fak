package agentopt

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// ParameterType defines the primitive data type expected by a property schema.
type ParameterType string

const (
	TypeString  ParameterType = "string"
	TypeNumber  ParameterType = "number"
	TypeInteger ParameterType = "integer"
	TypeBoolean ParameterType = "boolean"
	TypeArray   ParameterType = "array"
	TypeObject  ParameterType = "object"
)

// PropertySchema defines validation rules for a single tool argument.
type PropertySchema struct {
	Type       ParameterType             `json:"type"`
	Enum       []string                  `json:"enum,omitempty"`
	Items      *PropertySchema           `json:"items,omitempty"`
	Properties map[string]PropertySchema `json:"properties,omitempty"`
	Required   []string                  `json:"required,omitempty"`
}

// ToolSchema defines the structural contract for a callable agent tool.
type ToolSchema struct {
	Name                 string                    `json:"name"`
	Description          string                    `json:"description,omitempty"`
	Properties           map[string]PropertySchema `json:"properties"`
	Required             []string                  `json:"required,omitempty"`
	AdditionalProperties bool                      `json:"additional_properties"`
}

// SchemaValidationResult records the outcome and structural violations of an argument validation.
type SchemaValidationResult struct {
	Valid       bool     `json:"valid"`
	Conforming  bool     `json:"conforming"`
	Violations  []string `json:"violations,omitempty"`
	Attestation string   `json:"attestation,omitempty"`
}

// SchemaValidator enforces strict grammar and type constraints on tool call arguments.
type SchemaValidator struct {
	mu      sync.RWMutex
	schemas map[string]ToolSchema
}

// NewSchemaValidator constructs a validator pre-seeded with given tool schemas.
func NewSchemaValidator(schemas ...ToolSchema) *SchemaValidator {
	v := &SchemaValidator{
		schemas: make(map[string]ToolSchema),
	}
	for _, s := range schemas {
		v.schemas[s.Name] = s
	}
	return v
}

// RegisterSchema adds or replaces a tool schema in the validator.
func (v *SchemaValidator) RegisterSchema(s ToolSchema) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.schemas[s.Name] = s
}

// ValidateToolCall parses JSON arguments and checks them against the tool's registered schema.
func (v *SchemaValidator) ValidateToolCall(toolName string, rawArgs []byte) SchemaValidationResult {
	var args map[string]any
	if len(rawArgs) == 0 {
		args = make(map[string]any)
	} else if err := json.Unmarshal(rawArgs, &args); err != nil {
		return SchemaValidationResult{
			Valid:      false,
			Conforming: false,
			Violations: []string{fmt.Sprintf("malformed JSON arguments: %v", err)},
		}
	}
	return v.ValidateToolCallMap(toolName, args)
}

// ValidateToolCallMap checks argument values directly against the tool's registered schema.
func (v *SchemaValidator) ValidateToolCallMap(toolName string, args map[string]any) SchemaValidationResult {
	v.mu.RLock()
	schema, ok := v.schemas[toolName]
	v.mu.RUnlock()

	if !ok {
		return SchemaValidationResult{
			Valid:      false,
			Conforming: false,
			Violations: []string{fmt.Sprintf("unknown tool %q: no schema registered", toolName)},
		}
	}

	var violations []string

	// 1. Check required properties
	for _, req := range schema.Required {
		if _, present := args[req]; !present {
			violations = append(violations, fmt.Sprintf("missing required property %q", req))
		}
	}

	// 2. Check each provided argument
	for key, val := range args {
		propSchema, recognized := schema.Properties[key]
		if !recognized {
			if !schema.AdditionalProperties {
				violations = append(violations, fmt.Sprintf("unrecognized property %q not allowed by schema", key))
			}
			continue
		}
		violations = append(violations, validateProperty(key, val, propSchema)...)
	}

	valid := len(violations) == 0
	res := SchemaValidationResult{
		Valid:      valid,
		Conforming: valid,
		Violations: violations,
	}

	if valid {
		res.Attestation = computeSchemaAttestation(toolName, args)
	}

	return res
}

func validateProperty(name string, val any, schema PropertySchema) []string {
	var violations []string
	if val == nil {
		return nil
	}

	switch schema.Type {
	case TypeString:
		str, ok := val.(string)
		if !ok {
			violations = append(violations, fmt.Sprintf("property %q expected string, got %T", name, val))
		} else if len(schema.Enum) > 0 {
			match := false
			for _, allowed := range schema.Enum {
				if str == allowed {
					match = true
					break
				}
			}
			if !match {
				violations = append(violations, fmt.Sprintf("property %q value %q not in allowed enum [%s]", name, str, strings.Join(schema.Enum, ", ")))
			}
		}
	case TypeNumber:
		if _, ok := val.(float64); !ok {
			violations = append(violations, fmt.Sprintf("property %q expected number, got %T", name, val))
		}
	case TypeInteger:
		f, ok := val.(float64)
		if !ok || f != float64(int64(f)) {
			violations = append(violations, fmt.Sprintf("property %q expected integer, got %T (%v)", name, val, val))
		}
	case TypeBoolean:
		if _, ok := val.(bool); !ok {
			violations = append(violations, fmt.Sprintf("property %q expected boolean, got %T", name, val))
		}
	case TypeArray:
		arr, ok := val.([]any)
		if !ok {
			violations = append(violations, fmt.Sprintf("property %q expected array, got %T", name, val))
		} else if schema.Items != nil {
			for idx, item := range arr {
				itemName := fmt.Sprintf("%s[%d]", name, idx)
				violations = append(violations, validateProperty(itemName, item, *schema.Items)...)
			}
		}
	case TypeObject:
		obj, ok := val.(map[string]any)
		if !ok {
			violations = append(violations, fmt.Sprintf("property %q expected object, got %T", name, val))
		} else {
			for _, req := range schema.Required {
				if _, present := obj[req]; !present {
					violations = append(violations, fmt.Sprintf("property %q missing required field %q", name, req))
				}
			}
			for k, v := range obj {
				if childSchema, exists := schema.Properties[k]; exists {
					violations = append(violations, validateProperty(fmt.Sprintf("%s.%s", name, k), v, childSchema)...)
				}
			}
		}
	}

	return violations
}

func computeSchemaAttestation(toolName string, args map[string]any) string {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	h := sha256.New()
	fmt.Fprintf(h, "tool=%s\n", toolName)
	for _, k := range keys {
		fmt.Fprintf(h, "%s=%v\n", k, args[k])
	}
	return "attest:sha256:" + hex.EncodeToString(h.Sum(nil))
}
