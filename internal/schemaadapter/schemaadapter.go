package schemaadapter

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Dialect identifies a target model provider schema dialect.
type Dialect string

const (
	DialectGemini       Dialect = "gemini"
	DialectOpenAI       Dialect = "openai"
	DialectOpenAIStrict Dialect = "openai_strict"
	DialectAnthropic    Dialect = "anthropic"
)

// Ready reports that the leaf is wired.
func Ready() bool { return true }

// strictAllowedFormats defines the set of JSON Schema string formats permitted
// under OpenAI strict mode.
var strictAllowedFormats = map[string]bool{
	"date-time": true,
	"time":      true,
	"date":      true,
	"email":     true,
	"uuid":      true,
}

// strictOmittedKeys lists keywords that OpenAI strict mode disallows.
var strictOmittedKeys = []string{
	"minLength", "maxLength",
	"minimum", "maximum",
	"exclusiveMinimum", "exclusiveMaximum",
	"minItems", "maxItems", "uniqueItems",
	"minProperties", "maxProperties",
	"pattern", "default", "multipleOf",
	"title", "$schema", "$id", "$comment",
	"not", "patternProperties",
}

// ToGemini converts a canonical JSON Schema to Google Gemini format:
//   - Type names become uppercase ("OBJECT", "STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY").
//   - required is permitted only when type is OBJECT.
//   - Every property in required must exist in properties of that object.
//   - In anyOf / oneOf / allOf, branches with bare required (missing properties or non-OBJECT type)
//     are stripped to prevent rejection.
//   - Recursively walks properties, items, anyOf, oneOf, allOf, and additionalProperties.
func ToGemini(raw json.RawMessage) (json.RawMessage, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, errors.New("schemaadapter: empty input schema")
	}
	var root any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("schemaadapter: invalid json schema: %w", err)
	}
	rootMap, ok := root.(map[string]any)
	if !ok {
		return nil, errors.New("schemaadapter: schema must be a JSON object")
	}

	if rootMap["type"] == nil && (rootMap["properties"] != nil || rootMap["required"] != nil) {
		rootMap["type"] = "OBJECT"
	}

	sanitizeGemini(rootMap)

	out, err := json.Marshal(rootMap)
	if err != nil {
		return nil, fmt.Errorf("schemaadapter: failed to marshal gemini schema: %w", err)
	}
	return json.RawMessage(out), nil
}

func sanitizeGemini(node map[string]any) {
	if node == nil {
		return
	}

	if t, ok := node["type"].(string); ok && t != "" {
		node["type"] = strings.ToUpper(t)
	} else if arr, ok := node["type"].([]any); ok {
		newArr := make([]any, 0, len(arr))
		for _, item := range arr {
			if s, ok := item.(string); ok {
				newArr = append(newArr, strings.ToUpper(s))
			} else {
				newArr = append(newArr, item)
			}
		}
		node["type"] = newArr
	} else if node["properties"] != nil {
		node["type"] = "OBJECT"
	} else if node["items"] != nil {
		node["type"] = "ARRAY"
	}

	if props, ok := node["properties"].(map[string]any); ok {
		for _, child := range props {
			if childMap, ok := child.(map[string]any); ok {
				sanitizeGemini(childMap)
			}
		}
	}

	if itemsMap, ok := node["items"].(map[string]any); ok {
		sanitizeGemini(itemsMap)
	} else if itemsList, ok := node["items"].([]any); ok {
		for _, it := range itemsList {
			if itMap, ok := it.(map[string]any); ok {
				sanitizeGemini(itMap)
			}
		}
	}

	if addMap, ok := node["additionalProperties"].(map[string]any); ok {
		sanitizeGemini(addMap)
	}

	for _, combKey := range []string{"anyOf", "any_of", "oneOf", "one_of", "allOf", "all_of"} {
		branches, ok := node[combKey].([]any)
		if !ok {
			continue
		}
		var validBranches []any
		for _, br := range branches {
			brMap, ok := br.(map[string]any)
			if !ok {
				validBranches = append(validBranches, br)
				continue
			}
			if isBareRequired(brMap) {
				continue
			}
			sanitizeGemini(brMap)
			validBranches = append(validBranches, brMap)
		}
		if len(validBranches) == 0 {
			delete(node, combKey)
		} else {
			node[combKey] = validBranches
		}
	}

	if reqVal, hasReq := node["required"]; hasReq {
		tStr, _ := node["type"].(string)
		if !strings.EqualFold(tStr, "OBJECT") {
			delete(node, "required")
		} else {
			props, _ := node["properties"].(map[string]any)
			reqList := toStringList(reqVal)
			var cleanReq []any
			for _, r := range reqList {
				if props != nil {
					if _, exists := props[r]; exists {
						cleanReq = append(cleanReq, r)
					}
				}
			}
			if len(cleanReq) == 0 {
				delete(node, "required")
			} else {
				node["required"] = cleanReq
			}
		}
	}
}

func isBareRequired(m map[string]any) bool {
	reqVal, hasReq := m["required"]
	if !hasReq {
		return false
	}
	reqList := toStringList(reqVal)
	if len(reqList) == 0 {
		return false
	}
	tStr, _ := m["type"].(string)
	if !strings.EqualFold(tStr, "object") && !strings.EqualFold(tStr, "OBJECT") {
		return true
	}
	props, hasProps := m["properties"].(map[string]any)
	if !hasProps || len(props) == 0 {
		return true
	}
	hasAnyInProps := false
	for _, r := range reqList {
		if _, exists := props[r]; exists {
			hasAnyInProps = true
			break
		}
	}
	return !hasAnyInProps
}

// ToOpenAI converts canonical JSON Schema to OpenAI format.
// If strict is false:
// - Lowercases type names.
// - Ensures root has type: "object" if properties or required exist.
// If strict is true:
// - Injects additionalProperties: false on root and every nested object schema.
// - Every property in properties is listed in required.
// - Properties not originally in required become nullable anyOf schemas.
// - Disallowed keywords and non-whitelisted formats are removed.
func ToOpenAI(raw json.RawMessage, strict bool) (json.RawMessage, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, errors.New("schemaadapter: empty input schema")
	}
	var root any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("schemaadapter: invalid json schema: %w", err)
	}
	rootMap, ok := root.(map[string]any)
	if !ok {
		return nil, errors.New("schemaadapter: schema must be a JSON object")
	}

	if strict {
		if rootMap["type"] == nil || rootMap["type"] == "" {
			rootMap["type"] = "object"
		}
		sanitizeOpenAIStrict(rootMap)
	} else {
		if rootMap["properties"] != nil || rootMap["required"] != nil {
			if rootMap["type"] == nil || rootMap["type"] == "" {
				rootMap["type"] = "object"
			}
		}
		lowercaseNodeTypes(rootMap)
	}

	out, err := json.Marshal(rootMap)
	if err != nil {
		return nil, fmt.Errorf("schemaadapter: failed to marshal openai schema: %w", err)
	}
	return json.RawMessage(out), nil
}

func sanitizeOpenAIStrict(node map[string]any) {
	if node == nil {
		return
	}

	if t, ok := node["type"].(string); ok && t != "" {
		node["type"] = strings.ToLower(t)
	} else if arr, ok := node["type"].([]any); ok {
		var newArr []any
		for _, item := range arr {
			if s, ok := item.(string); ok {
				newArr = append(newArr, strings.ToLower(s))
			} else {
				newArr = append(newArr, item)
			}
		}
		node["type"] = newArr
	}

	for _, k := range strictOmittedKeys {
		delete(node, k)
	}

	if fmtVal, ok := node["format"].(string); ok {
		if !strictAllowedFormats[fmtVal] {
			delete(node, "format")
		}
	} else if node["format"] != nil {
		delete(node, "format")
	}

	tStr, _ := node["type"].(string)
	isObj := strings.EqualFold(tStr, "object") || node["properties"] != nil
	if isObj {
		node["type"] = "object"
		node["additionalProperties"] = false

		propsVal, hasProps := node["properties"]
		if !hasProps || propsVal == nil {
			node["properties"] = map[string]any{}
			node["required"] = []any{}
		} else if props, ok := propsVal.(map[string]any); ok {
			origReq := make(map[string]bool)
			for _, r := range toStringList(node["required"]) {
				origReq[r] = true
			}

			propNames := sortedKeys(props)
			finalReq := make([]any, len(propNames))
			for i, name := range propNames {
				finalReq[i] = name
			}
			node["required"] = finalReq

			for _, name := range propNames {
				child := props[name]
				if childMap, isMap := child.(map[string]any); isMap {
					if origReq[name] {
						props[name] = childMap
					} else {
						props[name] = strictMakeNullable(childMap)
					}
				} else {
					if origReq[name] {
						props[name] = child
					} else {
						props[name] = map[string]any{
							"anyOf": []any{
								child,
								map[string]any{"type": "null"},
							},
						}
					}
				}
			}
			node["properties"] = props
		}
	}

	if oneOfVal, hasOneOf := node["oneOf"]; hasOneOf {
		if node["anyOf"] == nil {
			node["anyOf"] = oneOfVal
		}
		delete(node, "oneOf")
	}

	if props, ok := node["properties"].(map[string]any); ok {
		for _, child := range props {
			if childMap, ok := child.(map[string]any); ok {
				sanitizeOpenAIStrict(childMap)
			}
		}
	}

	if itemsMap, ok := node["items"].(map[string]any); ok {
		sanitizeOpenAIStrict(itemsMap)
	} else if itemsList, ok := node["items"].([]any); ok {
		for _, it := range itemsList {
			if itMap, ok := it.(map[string]any); ok {
				sanitizeOpenAIStrict(itMap)
			}
		}
	}

	if anyOfList, ok := node["anyOf"].([]any); ok {
		for _, b := range anyOfList {
			if bMap, ok := b.(map[string]any); ok {
				sanitizeOpenAIStrict(bMap)
			}
		}
	}

	if allOfList, ok := node["allOf"].([]any); ok {
		for _, b := range allOfList {
			if bMap, ok := b.(map[string]any); ok {
				sanitizeOpenAIStrict(bMap)
			}
		}
	}

	if defsMap, ok := node["$defs"].(map[string]any); ok {
		for _, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				sanitizeOpenAIStrict(dvMap)
			}
		}
	}

	if defsMap, ok := node["definitions"].(map[string]any); ok {
		for _, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				sanitizeOpenAIStrict(dvMap)
			}
		}
	}
}

func strictMakeNullable(p map[string]any) map[string]any {
	if t, _ := p["type"].(string); strings.EqualFold(t, "null") {
		return p
	}

	if anyOfVal, hasAnyOf := p["anyOf"]; hasAnyOf {
		if anyOfList, isList := anyOfVal.([]any); isList {
			hasNull := false
			for _, item := range anyOfList {
				if itemMap, ok := item.(map[string]any); ok {
					if t, _ := itemMap["type"].(string); strings.EqualFold(t, "null") {
						hasNull = true
						break
					}
				}
			}
			if !hasNull {
				p["anyOf"] = append(anyOfList, map[string]any{"type": "null"})
			}
			return p
		}
	}

	delete(p, "nullable")

	nonNullBranch := make(map[string]any, len(p))
	for k, v := range p {
		if k != "description" {
			nonNullBranch[k] = v
		}
	}

	res := map[string]any{
		"anyOf": []any{
			nonNullBranch,
			map[string]any{"type": "null"},
		},
	}
	if desc, hasDesc := p["description"]; hasDesc {
		res["description"] = desc
	}
	return res
}

// ToAnthropic converts canonical JSON Schema to Anthropic format:
// - Valid standard JSON Schema draft-07: lowercase types, top-level type: "object".
// - Preserves standard schema properties and constraints (minItems, pattern, etc.).
func ToAnthropic(raw json.RawMessage) (json.RawMessage, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, errors.New("schemaadapter: empty input schema")
	}
	var root any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("schemaadapter: invalid json schema: %w", err)
	}
	rootMap, ok := root.(map[string]any)
	if !ok {
		return nil, errors.New("schemaadapter: schema must be a JSON object")
	}

	rootMap["type"] = "object"
	if rootMap["properties"] == nil {
		rootMap["properties"] = map[string]any{}
	}

	lowercaseNodeTypes(rootMap)

	out, err := json.Marshal(rootMap)
	if err != nil {
		return nil, fmt.Errorf("schemaadapter: failed to marshal anthropic schema: %w", err)
	}
	return json.RawMessage(out), nil
}

func lowercaseNodeTypes(node map[string]any) {
	if node == nil {
		return
	}

	if t, ok := node["type"].(string); ok && t != "" {
		node["type"] = strings.ToLower(t)
	} else if arr, ok := node["type"].([]any); ok {
		newArr := make([]any, 0, len(arr))
		for _, item := range arr {
			if s, ok := item.(string); ok {
				newArr = append(newArr, strings.ToLower(s))
			} else {
				newArr = append(newArr, item)
			}
		}
		node["type"] = newArr
	}

	if props, ok := node["properties"].(map[string]any); ok {
		for _, child := range props {
			if childMap, ok := child.(map[string]any); ok {
				lowercaseNodeTypes(childMap)
			}
		}
	}

	if itemsMap, ok := node["items"].(map[string]any); ok {
		lowercaseNodeTypes(itemsMap)
	} else if itemsList, ok := node["items"].([]any); ok {
		for _, it := range itemsList {
			if itMap, ok := it.(map[string]any); ok {
				lowercaseNodeTypes(itMap)
			}
		}
	}

	if anyOfList, ok := node["anyOf"].([]any); ok {
		for _, b := range anyOfList {
			if bMap, ok := b.(map[string]any); ok {
				lowercaseNodeTypes(bMap)
			}
		}
	}

	if oneOfList, ok := node["oneOf"].([]any); ok {
		for _, b := range oneOfList {
			if bMap, ok := b.(map[string]any); ok {
				lowercaseNodeTypes(bMap)
			}
		}
	}

	if allOfList, ok := node["allOf"].([]any); ok {
		for _, b := range allOfList {
			if bMap, ok := b.(map[string]any); ok {
				lowercaseNodeTypes(bMap)
			}
		}
	}

	if addMap, ok := node["additionalProperties"].(map[string]any); ok {
		lowercaseNodeTypes(addMap)
	}

	if defsMap, ok := node["$defs"].(map[string]any); ok {
		for _, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				lowercaseNodeTypes(dvMap)
			}
		}
	}

	if defsMap, ok := node["definitions"].(map[string]any); ok {
		for _, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				lowercaseNodeTypes(dvMap)
			}
		}
	}
}

// Normalize adapts a canonical JSON Schema to the specified dialect.
func Normalize(raw json.RawMessage, d Dialect) (json.RawMessage, error) {
	switch d {
	case DialectGemini:
		return ToGemini(raw)
	case DialectOpenAI:
		return ToOpenAI(raw, false)
	case DialectOpenAIStrict:
		return ToOpenAI(raw, true)
	case DialectAnthropic:
		return ToAnthropic(raw)
	default:
		return nil, fmt.Errorf("schemaadapter: unknown dialect %q", d)
	}
}

func toStringList(v any) []string {
	switch val := v.(type) {
	case []any:
		res := make([]string, 0, len(val))
		for _, item := range val {
			if s, ok := item.(string); ok {
				res = append(res, s)
			}
		}
		return res
	case []string:
		return val
	default:
		return nil
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
