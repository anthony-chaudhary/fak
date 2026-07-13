package fleetpane

import (
	"encoding/json"
	"fmt"
	"strconv"
)

func stringValueFirst(m map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringValue(m[key]); value != "" {
			return value
		}
	}
	return ""
}

func stringValue(v any) string {
	switch val := v.(type) {
	case string:
		return val
	case fmt.Stringer:
		return val.String()
	case json.Number:
		return val.String()
	case nil:
		return ""
	default:
		return fmt.Sprint(val)
	}
}

func stringValueDefault(v any, fallback string) string {
	if s := stringValue(v); s != "" {
		return s
	}
	return fallback
}

func boolValue(v any) bool {
	return boolValueDefault(v, false)
}

func boolValueDefault(v any, fallback bool) bool {
	switch val := v.(type) {
	case bool:
		return val
	case string:
		b, err := strconv.ParseBool(val)
		if err == nil {
			return b
		}
	case float64:
		return val != 0
	case int:
		return val != 0
	}
	return fallback
}

func intValueDefault(v any, fallback int) int {
	switch val := v.(type) {
	case int:
		return val
	case int64:
		return int(val)
	case float64:
		return int(val)
	case json.Number:
		i, err := val.Int64()
		if err == nil {
			return int(i)
		}
	case string:
		i, err := strconv.Atoi(val)
		if err == nil {
			return i
		}
	}
	return fallback
}

func floatValueDefault(v any, fallback float64) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case int:
		return float64(val)
	case json.Number:
		f, err := val.Float64()
		if err == nil {
			return f
		}
	case string:
		f, err := strconv.ParseFloat(val, 64)
		if err == nil {
			return f
		}
	}
	return fallback
}

func stringSlice(v any) []string {
	items := sliceAny(v)
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, stringValue(item))
	}
	return out
}

func intSlice(v any) []int {
	items := sliceAny(v)
	out := make([]int, 0, len(items))
	for _, item := range items {
		out = append(out, intValueDefault(item, 0))
	}
	return out
}

func sliceAny(v any) []any {
	switch val := v.(type) {
	case []any:
		return val
	case []string:
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = item
		}
		return out
	case []LoopCheck:
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = item
		}
		return out
	case nil:
		return nil
	default:
		return nil
	}
}

func firstAny(items []any, n int) []any {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

func mapValue(v any) map[string]any {
	switch val := v.(type) {
	case map[string]any:
		return val
	case map[string]int:
		out := map[string]any{}
		for k, v := range val {
			out[k] = v
		}
		return out
	default:
		return map[string]any{}
	}
}

func nestedMap(v any) map[string]map[string]any {
	out := map[string]map[string]any{}
	for key, val := range mapValue(v) {
		out[key] = mapValue(val)
	}
	return out
}

func mapInt(v any) map[string]int {
	out := map[string]int{}
	for key, val := range mapValue(v) {
		out[key] = intValueDefault(val, 0)
	}
	return out
}

func intMapAny(in map[string]int) map[string]any {
	out := map[string]any{}
	for key, val := range in {
		out[key] = val
	}
	return out
}

func firstStrings(values []string, limit int) []string {
	if limit <= 0 || len(values) == 0 {
		return nil
	}
	if len(values) < limit {
		limit = len(values)
	}
	return append([]string(nil), values[:limit]...)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
