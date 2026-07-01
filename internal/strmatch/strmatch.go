// Package strmatch is the one shared any-substring matcher the per-package
// containsAny copies converged on (slop de-dup #776): six identical bool
// bodies and one first-match variant were duplicated across
// internal/{attemptbudget,benchlineagegate,headroom,windowgate,
// readmevisualaudit,terminalbench,vcacheqa}. Stdlib-only, off the hot path.
package strmatch

import "strings"

// ContainsAny reports whether s contains any of subs as a substring. Callers
// that need case-insensitivity lowercase s (and subs) beforehand.
func ContainsAny(s string, subs ...string) bool {
	_, ok := FirstContained(s, subs)
	return ok
}

// FirstContained returns the first needle contained in haystack, and whether
// any matched — the witness-carrying variant for a caller that reports WHICH
// phrase hit, not just that one did.
func FirstContained(haystack string, needles []string) (string, bool) {
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			return n, true
		}
	}
	return "", false
}
