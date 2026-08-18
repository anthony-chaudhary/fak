package strictjson

import (
	"encoding/json"
	"strings"
)

// Rows decodes either a JSON array or one JSON object into a slice. Empty or
// malformed input returns nil.
func Rows[T any](text string) []T {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	var rows []T
	if json.Unmarshal([]byte(text), &rows) == nil {
		return rows
	}
	var one T
	if json.Unmarshal([]byte(text), &one) == nil {
		return []T{one}
	}
	return nil
}
