package toolcallcontrol

import (
	"bytes"
	"encoding/json"
	"io"
)

// StructuredProjectionSchema identifies the bounded structured-result envelope.
const StructuredProjectionSchema = "fak-structured-tool-result/1"

// ProjectionCompleteness states whether projection retained every source item.
type ProjectionCompleteness string

const (
	ProjectionComplete ProjectionCompleteness = "complete"
	ProjectionPartial  ProjectionCompleteness = "partial"
)

// StructuredProjectionContract declares the only response shape and fields that
// may be projected. Item pointers are relative to each object in the item array.
type StructuredProjectionContract struct {
	ItemsPointer        string   `json:"items_pointer"`
	CompletenessPointer string   `json:"completeness_pointer"`
	FieldPointers       []string `json:"field_pointers"`
	IdentifierPointers  []string `json:"identifier_pointers"`
	CitationPointers    []string `json:"citation_pointers"`
	MaximumItems        int64    `json:"maximum_items"`
}

// StructuredProjectionReceipt records whether and how a response was reduced.
type StructuredProjectionReceipt struct {
	Disposition    ResponseDisposition    `json:"disposition"`
	Reason         string                 `json:"reason"`
	Completeness   ProjectionCompleteness `json:"completeness,omitempty"`
	OriginalItems  int64                  `json:"original_items"`
	ProjectedItems int64                  `json:"projected_items"`
	OriginalBytes  int64                  `json:"original_bytes"`
	ProjectedBytes int64                  `json:"projected_bytes"`
}

// StructuredProjectionResult contains either a labeled projected envelope or
// the byte-for-byte original payload when projection cannot be proven safe.
type StructuredProjectionResult struct {
	Payload json.RawMessage             `json:"payload"`
	Receipt StructuredProjectionReceipt `json:"receipt"`
}

type structuredProjectionEnvelope struct {
	Schema             string                 `json:"schema"`
	Completeness       ProjectionCompleteness `json:"completeness"`
	SourceCompleteness json.RawMessage        `json:"source_completeness"`
	OriginalItems      int64                  `json:"original_items"`
	ReturnedItems      int64                  `json:"returned_items"`
	Items              []any                  `json:"items"`
}

// ProjectStructuredResponse applies schema-contracted item and field filtering.
// Any ambiguity returns the original bytes; the function never emits unlabeled
// truncation or silently drops a required identifier or citation.
func ProjectStructuredResponse(payload []byte, contract StructuredProjectionContract) StructuredProjectionResult {
	fallback := func(reason string, originalItems int64) StructuredProjectionResult {
		return StructuredProjectionResult{
			Payload: append(json.RawMessage(nil), payload...),
			Receipt: StructuredProjectionReceipt{
				Disposition: ResponsePass, Reason: reason, OriginalItems: originalItems,
				ProjectedItems: originalItems, OriginalBytes: int64(len(payload)), ProjectedBytes: int64(len(payload)),
			},
		}
	}
	if reason := validateProjectionContract(contract); reason != "" {
		return fallback(reason, 0)
	}

	var root any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := decoder.Decode(&root); err != nil {
		return fallback("invalid_json", 0)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fallback("invalid_json", 0)
	}

	itemsValue, found := jsonPointerValue(root, contract.ItemsPointer)
	if !found {
		return fallback("items_missing", 0)
	}
	items, ok := itemsValue.([]any)
	if !ok {
		return fallback("items_not_array", 0)
	}
	originalItems := int64(len(items))
	completeness, found := jsonPointerValue(root, contract.CompletenessPointer)
	if !found {
		return fallback("completeness_missing", originalItems)
	}
	completenessJSON, err := json.Marshal(completeness)
	if err != nil {
		return fallback("completeness_invalid", originalItems)
	}

	returned := originalItems
	if returned > contract.MaximumItems {
		returned = contract.MaximumItems
	}
	projected := make([]any, 0, returned)
	for index := int64(0); index < returned; index++ {
		item, ok := items[index].(map[string]any)
		if !ok {
			return fallback("item_not_object", originalItems)
		}
		projectedItem := make(map[string]any)
		for _, pointer := range contract.FieldPointers {
			value, exists := jsonPointerValue(item, pointer)
			if !exists {
				if containsPointer(contract.IdentifierPointers, pointer) || containsPointer(contract.CitationPointers, pointer) {
					return fallback("required_field_missing", originalItems)
				}
				continue
			}
			if !setJSONPointerValue(projectedItem, pointer, value) {
				return fallback("field_shape_conflict", originalItems)
			}
		}
		projected = append(projected, projectedItem)
	}

	projectionCompleteness := ProjectionComplete
	if returned < originalItems {
		projectionCompleteness = ProjectionPartial
	}
	envelope := structuredProjectionEnvelope{
		Schema: StructuredProjectionSchema, Completeness: projectionCompleteness,
		SourceCompleteness: completenessJSON, OriginalItems: originalItems,
		ReturnedItems: returned, Items: projected,
	}
	encoded, err := json.Marshal(envelope)
	if err != nil || len(encoded) >= len(payload) {
		return fallback("projection_not_smaller", originalItems)
	}
	return StructuredProjectionResult{
		Payload: encoded,
		Receipt: StructuredProjectionReceipt{
			Disposition: ResponseProject, Reason: "structured_projection", Completeness: projectionCompleteness,
			OriginalItems: originalItems, ProjectedItems: returned,
			OriginalBytes: int64(len(payload)), ProjectedBytes: int64(len(encoded)),
		},
	}
}

func validateProjectionContract(contract StructuredProjectionContract) string {
	if contract.MaximumItems < 0 || contract.ItemsPointer == "" || contract.CompletenessPointer == "" || len(contract.FieldPointers) == 0 {
		return "invalid_contract"
	}
	selected := make(map[string]struct{}, len(contract.FieldPointers))
	for _, pointer := range contract.FieldPointers {
		if !validObjectPointer(pointer) {
			return "invalid_contract"
		}
		if _, duplicate := selected[pointer]; duplicate {
			return "invalid_contract"
		}
		selected[pointer] = struct{}{}
	}
	if len(contract.IdentifierPointers) == 0 || len(contract.CitationPointers) == 0 {
		return "invalid_contract"
	}
	for _, required := range append(append([]string(nil), contract.IdentifierPointers...), contract.CitationPointers...) {
		if _, ok := selected[required]; !ok {
			return "invalid_contract"
		}
	}
	return ""
}

func validObjectPointer(pointer string) bool {
	if pointer == "" || pointer == "/" || pointer[0] != '/' {
		return false
	}
	for _, token := range splitJSONPointer(pointer) {
		if _, ok := decodePointerToken(token); !ok {
			return false
		}
	}
	return true
}

func jsonPointerValue(root any, pointer string) (any, bool) {
	if !validObjectPointer(pointer) {
		return nil, false
	}
	current := root
	for _, encoded := range splitJSONPointer(pointer) {
		key, _ := decodePointerToken(encoded)
		object, ok := current.(map[string]any)
		if !ok {
			return nil, false
		}
		current, ok = object[key]
		if !ok {
			return nil, false
		}
	}
	return current, true
}

func setJSONPointerValue(root map[string]any, pointer string, value any) bool {
	parts := splitJSONPointer(pointer)
	current := root
	for _, encoded := range parts[:len(parts)-1] {
		key, _ := decodePointerToken(encoded)
		next, exists := current[key]
		if !exists {
			child := make(map[string]any)
			current[key] = child
			current = child
			continue
		}
		child, ok := next.(map[string]any)
		if !ok {
			return false
		}
		current = child
	}
	key, _ := decodePointerToken(parts[len(parts)-1])
	if _, exists := current[key]; exists {
		return false
	}
	current[key] = value
	return true
}

func splitJSONPointer(pointer string) []string {
	return bytesToStrings(bytes.Split([]byte(pointer[1:]), []byte{'/'}))
}

func bytesToStrings(values [][]byte) []string {
	result := make([]string, len(values))
	for index := range values {
		result[index] = string(values[index])
	}
	return result
}

func containsPointer(pointers []string, target string) bool {
	for _, pointer := range pointers {
		if pointer == target {
			return true
		}
	}
	return false
}
