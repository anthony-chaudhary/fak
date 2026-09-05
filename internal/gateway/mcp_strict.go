package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// strictAllowedFormats defines the subset of JSON Schema formats permitted
// under OpenAI Structured Outputs strict mode.
var strictAllowedFormats = map[string]bool{
	"date-time": true,
	"time":      true,
	"date":      true,
	"email":     true,
	"uuid":      true,
}

// ToOpenAIStrictSchema transforms a canonical JSON Schema into an OpenAI
// Strict Mode (strict: true) compliant schema.
func ToOpenAIStrictSchema(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return nil, errors.New("empty schema")
	}
	var root any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, fmt.Errorf("invalid json schema: %w", err)
	}
	rootMap, ok := root.(map[string]any)
	if !ok {
		return nil, errors.New("schema must be a JSON object")
	}

	transformed := strictTransformObject(rootMap)
	out, err := json.Marshal(transformed)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal strict schema: %w", err)
	}
	return json.RawMessage(out), nil
}

// StrictToolDescriptors returns a copy of toolDescriptors() where every
// tool's inputSchema has been transformed via ToOpenAIStrictSchema and
// "strict": true is added.
func StrictToolDescriptors() []map[string]any {
	return ToStrictToolDescriptors(toolDescriptors())
}

// ToStrictToolDescriptors transforms a slice of tool descriptors so that every
// tool's inputSchema has been transformed via ToOpenAIStrictSchema and
// "strict": true is added.
func ToStrictToolDescriptors(rawList []map[string]any) []map[string]any {
	res := make([]map[string]any, len(rawList))
	for i, td := range rawList {
		cp := make(map[string]any, len(td)+1)
		for k, v := range td {
			cp[k] = v
		}
		if schemaRaw, ok := td["inputSchema"]; ok {
			var rawBytes []byte
			switch s := schemaRaw.(type) {
			case json.RawMessage:
				rawBytes = s
			case []byte:
				rawBytes = s
			case string:
				rawBytes = []byte(s)
			default:
				b, err := json.Marshal(s)
				if err == nil {
					rawBytes = b
				}
			}
			if len(rawBytes) > 0 {
				strictSchema, err := ToOpenAIStrictSchema(rawBytes)
				if err == nil {
					cp["inputSchema"] = strictSchema
				}
			}
		}
		cp["strict"] = true
		res[i] = cp
	}
	return res
}

// ValidateOpenAIStrictMode validates that a schema strictly adheres to OpenAI
// strict mode constraints:
// - Root and every nested object has additionalProperties: false.
// - Every property in properties is included in required.
// Returns a slice of validation error strings (empty if compliant).
func ValidateOpenAIStrictMode(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return []string{"empty schema"}
	}
	var root any
	if err := json.Unmarshal(raw, &root); err != nil {
		return []string{fmt.Sprintf("invalid json: %v", err)}
	}
	rootMap, ok := root.(map[string]any)
	if !ok {
		return []string{"schema must be a JSON object"}
	}
	var errs []string
	validateStrictNode(rootMap, "$", true, &errs)
	return errs
}

func validateStrictNode(node any, path string, isRoot bool, errs *[]string) {
	nodeMap, ok := node.(map[string]any)
	if !ok {
		return
	}

	schemaType, _ := nodeMap["type"].(string)
	hasProperties := nodeMap["properties"] != nil
	isObjectSchema := isRoot || schemaType == "object" || hasProperties

	if isObjectSchema {
		apVal, hasAP := nodeMap["additionalProperties"]
		if !hasAP {
			*errs = append(*errs, fmt.Sprintf("%s: missing additionalProperties: false", path))
		} else if apBool, isBool := apVal.(bool); !isBool || apBool {
			*errs = append(*errs, fmt.Sprintf("%s: additionalProperties must be false", path))
		}

		if hasProperties {
			if props, isMap := nodeMap["properties"].(map[string]any); isMap {
				reqSet := make(map[string]bool)
				if reqVal, hasReq := nodeMap["required"]; hasReq {
					if reqList, isList := reqVal.([]any); isList {
						for _, item := range reqList {
							if str, ok := item.(string); ok {
								reqSet[str] = true
							}
						}
					} else if reqListStr, isListStr := reqVal.([]string); isListStr {
						for _, str := range reqListStr {
							reqSet[str] = true
						}
					}
				}
				var propKeys []string
				for k := range props {
					propKeys = append(propKeys, k)
				}
				sort.Strings(propKeys)
				for _, k := range propKeys {
					if !reqSet[k] {
						*errs = append(*errs, fmt.Sprintf("%s: property %q in properties is not listed in required", path, k))
					}
				}
			}
		}
	}

	if props, ok := nodeMap["properties"].(map[string]any); ok {
		var propKeys []string
		for k := range props {
			propKeys = append(propKeys, k)
		}
		sort.Strings(propKeys)
		for _, k := range propKeys {
			validateStrictNode(props[k], path+".properties."+k, false, errs)
		}
	}

	if items, ok := nodeMap["items"].(map[string]any); ok {
		validateStrictNode(items, path+".items", false, errs)
	} else if itemsList, ok := nodeMap["items"].([]any); ok {
		for i, item := range itemsList {
			validateStrictNode(item, fmt.Sprintf("%s.items[%d]", path, i), false, errs)
		}
	}

	if prefixItems, ok := nodeMap["prefixItems"].([]any); ok {
		for i, item := range prefixItems {
			validateStrictNode(item, fmt.Sprintf("%s.prefixItems[%d]", path, i), false, errs)
		}
	}

	if anyOf, ok := nodeMap["anyOf"].([]any); ok {
		for i, item := range anyOf {
			validateStrictNode(item, fmt.Sprintf("%s.anyOf[%d]", path, i), false, errs)
		}
	}

	if allOf, ok := nodeMap["allOf"].([]any); ok {
		for i, item := range allOf {
			validateStrictNode(item, fmt.Sprintf("%s.allOf[%d]", path, i), false, errs)
		}
	}

	if oneOf, ok := nodeMap["oneOf"].([]any); ok {
		for i, item := range oneOf {
			validateStrictNode(item, fmt.Sprintf("%s.oneOf[%d]", path, i), false, errs)
		}
	}

	if defs, ok := nodeMap["$defs"].(map[string]any); ok {
		var defKeys []string
		for k := range defs {
			defKeys = append(defKeys, k)
		}
		sort.Strings(defKeys)
		for _, k := range defKeys {
			validateStrictNode(defs[k], path+".$defs."+k, false, errs)
		}
	}

	if defs, ok := nodeMap["definitions"].(map[string]any); ok {
		var defKeys []string
		for k := range defs {
			defKeys = append(defKeys, k)
		}
		sort.Strings(defKeys)
		for _, k := range defKeys {
			validateStrictNode(defs[k], path+".definitions."+k, false, errs)
		}
	}
}

func strictSanitize(s map[string]any) map[string]any {
	delete(s, "$schema")
	delete(s, "default")
	delete(s, "minItems")
	delete(s, "maxItems")
	delete(s, "uniqueItems")
	delete(s, "minLength")
	delete(s, "maxLength")
	delete(s, "pattern")
	delete(s, "minimum")
	delete(s, "maximum")
	delete(s, "exclusiveMinimum")
	delete(s, "exclusiveMaximum")
	delete(s, "multipleOf")
	delete(s, "minProperties")
	delete(s, "maxProperties")
	delete(s, "patternProperties")
	delete(s, "dependentRequired")

	if fmtVal, hasFmt := s["format"]; hasFmt {
		if fmtStr, ok := fmtVal.(string); ok {
			if !strictAllowedFormats[fmtStr] {
				delete(s, "format")
			}
		} else {
			delete(s, "format")
		}
	}

	if props, ok := s["properties"].(map[string]any); ok {
		for k, v := range props {
			if vMap, ok := v.(map[string]any); ok {
				props[k] = strictSanitize(vMap)
			}
		}
	}

	if apMap, ok := s["additionalProperties"].(map[string]any); ok {
		s["additionalProperties"] = strictSanitize(apMap)
	}

	if itemsMap, ok := s["items"].(map[string]any); ok {
		s["items"] = strictSanitize(itemsMap)
	} else if itemsSlice, ok := s["items"].([]any); ok {
		for i, it := range itemsSlice {
			if itMap, ok := it.(map[string]any); ok {
				itemsSlice[i] = strictSanitize(itMap)
			}
		}
	}

	if prefixSlice, ok := s["prefixItems"].([]any); ok {
		for i, it := range prefixSlice {
			if itMap, ok := it.(map[string]any); ok {
				prefixSlice[i] = strictSanitize(itMap)
			}
		}
	}

	if anyOfSlice, ok := s["anyOf"].([]any); ok {
		for i, it := range anyOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				anyOfSlice[i] = strictSanitize(itMap)
			}
		}
	}

	if allOfSlice, ok := s["allOf"].([]any); ok {
		for i, it := range allOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				allOfSlice[i] = strictSanitize(itMap)
			}
		}
	}

	if oneOfSlice, ok := s["oneOf"].([]any); ok {
		for i, it := range oneOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				oneOfSlice[i] = strictSanitize(itMap)
			}
		}
	}

	if notMap, ok := s["not"].(map[string]any); ok {
		s["not"] = strictSanitize(notMap)
	}

	if defsMap, ok := s["$defs"].(map[string]any); ok {
		for k, v := range defsMap {
			if vMap, ok := v.(map[string]any); ok {
				defsMap[k] = strictSanitize(vMap)
			}
		}
	}

	if defsMap, ok := s["definitions"].(map[string]any); ok {
		for k, v := range defsMap {
			if vMap, ok := v.(map[string]any); ok {
				defsMap[k] = strictSanitize(vMap)
			}
		}
	}

	return s
}

func strictTransformObject(obj map[string]any) map[string]any {
	obj = strictSanitize(obj)
	obj["type"] = "object"
	obj["additionalProperties"] = false

	if defsMap, isMap := obj["$defs"].(map[string]any); isMap {
		for dk, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				defsMap[dk] = strictTransformSchema(dvMap)
			}
		}
		obj["$defs"] = defsMap
	}

	if defsMap, isMap := obj["definitions"].(map[string]any); isMap {
		for dk, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				defsMap[dk] = strictTransformSchema(dvMap)
			}
		}
		obj["definitions"] = defsMap
	}

	if anyOfSlice, isSlice := obj["anyOf"].([]any); isSlice {
		for i, it := range anyOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				anyOfSlice[i] = strictTransformSchema(itMap)
			}
		}
		obj["anyOf"] = anyOfSlice
	}

	if allOfSlice, isSlice := obj["allOf"].([]any); isSlice {
		for i, it := range allOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				allOfSlice[i] = strictTransformSchema(itMap)
			}
		}
		obj["allOf"] = allOfSlice
	}

	if oneOfSlice, isSlice := obj["oneOf"].([]any); isSlice {
		for i, it := range oneOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				oneOfSlice[i] = strictTransformSchema(itMap)
			}
		}
		obj["oneOf"] = oneOfSlice
	}

	propsVal, hasProps := obj["properties"]
	if !hasProps || propsVal == nil {
		obj["properties"] = map[string]any{}
		obj["required"] = []any{}
		return obj
	}

	props, ok := propsVal.(map[string]any)
	if !ok {
		obj["properties"] = map[string]any{}
		obj["required"] = []any{}
		return obj
	}

	origReq := make(map[string]bool)
	if reqVal, hasReq := obj["required"]; hasReq {
		if reqList, isList := reqVal.([]any); isList {
			for _, item := range reqList {
				if str, ok := item.(string); ok {
					origReq[str] = true
				}
			}
		} else if reqListStr, isListStr := reqVal.([]string); isListStr {
			for _, str := range reqListStr {
				origReq[str] = true
			}
		}
	}

	propNames := make([]string, 0, len(props))
	for k := range props {
		propNames = append(propNames, k)
	}
	sort.Strings(propNames)

	reqList := make([]any, len(propNames))
	for i, name := range propNames {
		reqList[i] = name
	}
	obj["required"] = reqList

	for _, k := range propNames {
		val := props[k]
		if pMap, isMap := val.(map[string]any); isMap {
			pMap = strictTransformSchema(pMap)
			if origReq[k] {
				props[k] = pMap
			} else {
				props[k] = strictMakeNullable(pMap)
			}
		} else {
			if origReq[k] {
				props[k] = val
			} else {
				props[k] = map[string]any{
					"anyOf": []any{
						val,
						map[string]any{"type": "null"},
					},
				}
			}
		}
	}

	obj["properties"] = props
	return obj
}

func strictTransformSchema(s map[string]any) map[string]any {
	s = strictSanitize(s)

	if tList, isList := s["type"].([]any); isList {
		var anyOfList []any
		for _, t := range tList {
			if tStr, ok := t.(string); ok {
				anyOfList = append(anyOfList, map[string]any{"type": tStr})
			}
		}
		delete(s, "type")
		s["anyOf"] = anyOfList
	}

	sType, _ := s["type"].(string)
	hasProps := s["properties"] != nil
	if sType == "object" || hasProps {
		return strictTransformObject(s)
	}

	if itemsMap, isMap := s["items"].(map[string]any); isMap {
		s["items"] = strictTransformSchema(itemsMap)
	} else if itemsSlice, isSlice := s["items"].([]any); isSlice {
		for i, it := range itemsSlice {
			if itMap, ok := it.(map[string]any); ok {
				itemsSlice[i] = strictTransformSchema(itMap)
			}
		}
		s["items"] = itemsSlice
	}

	if prefixSlice, isSlice := s["prefixItems"].([]any); isSlice {
		for i, it := range prefixSlice {
			if itMap, ok := it.(map[string]any); ok {
				prefixSlice[i] = strictTransformSchema(itMap)
			}
		}
		s["prefixItems"] = prefixSlice
	}

	if anyOfSlice, isSlice := s["anyOf"].([]any); isSlice {
		for i, it := range anyOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				anyOfSlice[i] = strictTransformSchema(itMap)
			}
		}
		s["anyOf"] = anyOfSlice
	}

	if allOfSlice, isSlice := s["allOf"].([]any); isSlice {
		for i, it := range allOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				allOfSlice[i] = strictTransformSchema(itMap)
			}
		}
		s["allOf"] = allOfSlice
	}

	if oneOfSlice, isSlice := s["oneOf"].([]any); isSlice {
		for i, it := range oneOfSlice {
			if itMap, ok := it.(map[string]any); ok {
				oneOfSlice[i] = strictTransformSchema(itMap)
			}
		}
		s["oneOf"] = oneOfSlice
	}

	if defsMap, isMap := s["$defs"].(map[string]any); isMap {
		for dk, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				defsMap[dk] = strictTransformSchema(dvMap)
			}
		}
		s["$defs"] = defsMap
	}

	if defsMap, isMap := s["definitions"].(map[string]any); isMap {
		for dk, dv := range defsMap {
			if dvMap, ok := dv.(map[string]any); ok {
				defsMap[dk] = strictTransformSchema(dvMap)
			}
		}
		s["definitions"] = defsMap
	}

	return s
}

func isNullBranch(item any) bool {
	if item == nil {
		return true
	}
	itemMap, ok := item.(map[string]any)
	if !ok {
		return false
	}
	if t, ok := itemMap["type"].(string); ok && t == "null" {
		return true
	}
	if tList, ok := itemMap["type"].([]any); ok && len(tList) == 1 {
		if t, ok := tList[0].(string); ok && t == "null" {
			return true
		}
	}
	return false
}

func flattenAnyOf(items []any) []any {
	var result []any
	for _, item := range items {
		if item == nil {
			result = append(result, item)
			continue
		}
		itemMap, ok := item.(map[string]any)
		if ok {
			if tList, isList := itemMap["type"].([]any); isList {
				var innerList []any
				for _, t := range tList {
					if tStr, ok := t.(string); ok {
						innerList = append(innerList, map[string]any{"type": tStr})
					}
				}
				flatInner := flattenAnyOf(innerList)
				result = append(result, flatInner...)
				continue
			}
			if anyOfVal, hasAnyOf := itemMap["anyOf"]; hasAnyOf {
				if anyOfList, isList := anyOfVal.([]any); isList {
					flatInner := flattenAnyOf(anyOfList)
					result = append(result, flatInner...)
					continue
				}
			}
		}
		result = append(result, item)
	}
	return result
}

func strictMakeNullable(p map[string]any) map[string]any {
	if isNullBranch(p) {
		return p
	}

	if tList, isList := p["type"].([]any); isList {
		var anyOfList []any
		for _, t := range tList {
			if tStr, ok := t.(string); ok {
				anyOfList = append(anyOfList, map[string]any{"type": tStr})
			}
		}
		delete(p, "type")
		p["anyOf"] = anyOfList
	}

	if anyOfVal, hasAnyOf := p["anyOf"]; hasAnyOf {
		if anyOfList, isList := anyOfVal.([]any); isList {
			flat := flattenAnyOf(anyOfList)
			var nonNull []any
			for _, item := range flat {
				if isNullBranch(item) {
					continue
				}
				nonNull = append(nonNull, item)
			}
			nonNull = append(nonNull, map[string]any{"type": "null"})
			p["anyOf"] = nonNull
			return p
		}
	}

	nonNullBranch := make(map[string]any, len(p))
	for k, v := range p {
		if k != "description" && k != "title" {
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
	if title, hasTitle := p["title"]; hasTitle {
		res["title"] = title
	}
	return res
}
