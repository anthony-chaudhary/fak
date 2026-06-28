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
