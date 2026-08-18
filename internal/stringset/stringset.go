// Package stringset provides deterministic views of string sets.
package stringset

import "sort"

// Sorted returns set members in lexical order.
func Sorted[V any](set map[string]V) []string {
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
