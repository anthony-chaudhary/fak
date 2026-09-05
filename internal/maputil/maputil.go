// Package maputil provides generic helper functions for map operations.
package maputil

import "sort"

// SortedKeys returns the keys of m sorted in ascending order.
func SortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Str returns key k from m as a string, or empty if missing or non-string.
func Str(m map[string]any, k string) string {
	if v, ok := m[k].(string); ok {
		return v
	}
	return ""
}
