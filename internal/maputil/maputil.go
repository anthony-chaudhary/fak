// Package maputil holds small, generic map helpers that were previously
// copy-pasted across packages. Hoisting them here keeps the behaviour in one
// place so every caller deterministically iterates a map in the same order.
package maputil

import "sort"

// SortedKeys returns the keys of m sorted in ascending (lexicographic) order.
// It is the generic form of the several identical, per-package sortedKeys
// helpers that returned the alphabetically-sorted keys of a string-keyed map.
func SortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Str reads key k from a decoded-JSON object as a string, returning "" when the key is
// absent OR present with any non-string type. It is the generic form of the several
// identical per-package `func str(m map[string]any, k string) string` helpers that every
// `json.Unmarshal` into `map[string]any` grows.
//
// NOTE the deliberate lack of a "present but wrong type" signal: every caller hoisted here
// treated absent and wrong-typed alike, so a missing field and a numeric one both read as
// "". A caller that must tell them apart should index the map directly rather than widen
// this.
func Str(m map[string]any, k string) string {
	if v, ok := m[k].(string); ok {
		return v
	}
	return ""
}
