package zaitask

import (
	"encoding/json"
	"fmt"
)

// ValidateToolCall checks that call corresponds to an allowed ToolDefinition and that
// its arguments conform to the definition's parameter schema (required properties and basic types).
func ValidateToolCall(call ToolCall, defs []ToolDefinition) error {
	name := call.Function.Name
	if name == "" {
		return fmt.Errorf("zaitask: tool call missing function name")
	}

	var matchedDef *ToolDefinition
	for i := range defs {
		if defs[i].Type == "function" && defs[i].Function.Name == name {
			matchedDef = &defs[i]
			break
		}
	}
	if matchedDef == nil {
		return fmt.Errorf("zaitask: unexpected tool call function %q not found in tool definitions", name)
	}

	rawArgs, err := call.Function.ArgumentsJSON()
	if err != nil {
		return fmt.Errorf("zaitask: invalid tool arguments for %q: %w", name, err)
	}

	var argsObj map[string]any
	if err := json.Unmarshal(rawArgs, &argsObj); err != nil {
		return fmt.Errorf("zaitask: tool arguments for %q must be a JSON object: %w", name, err)
	}

	if len(matchedDef.Function.Parameters) > 0 {
		var schema struct {
			Required   []string `json:"required"`
			Properties map[string]struct {
				Type string `json:"type"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(matchedDef.Function.Parameters, &schema); err == nil {
			for _, req := range schema.Required {
				if _, ok := argsObj[req]; !ok {
					return fmt.Errorf("zaitask: missing required argument %q for tool %q", req, name)
				}
			}
			for propName, propSpec := range schema.Properties {
				val, ok := argsObj[propName]
				if !ok || val == nil {
					continue
				}
				switch propSpec.Type {
				case "string":
					if _, isStr := val.(string); !isStr {
						return fmt.Errorf("zaitask: argument %q for tool %q must be a string", propName, name)
					}
				case "number", "integer":
					if _, isNum := val.(float64); !isNum {
						return fmt.Errorf("zaitask: argument %q for tool %q must be a number", propName, name)
					}
				case "boolean":
					if _, isBool := val.(bool); !isBool {
						return fmt.Errorf("zaitask: argument %q for tool %q must be a boolean", propName, name)
					}
				case "array":
					if _, isArr := val.([]any); !isArr {
						return fmt.Errorf("zaitask: argument %q for tool %q must be an array", propName, name)
					}
				case "object":
					if _, isObj := val.(map[string]any); !isObj {
						return fmt.Errorf("zaitask: argument %q for tool %q must be an object", propName, name)
					}
				}
			}
		}
	}

	return nil
}
