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

// FirstNonBlank returns the first value containing non-whitespace text. It
// preserves the caller's original value rather than returning a trimmed copy.
func FirstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

// FirstNonEmpty returns the first value that is not exactly empty. Unlike
// FirstNonBlank, whitespace is a value and is returned unchanged.
func FirstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

// FirstTrimmed returns the first value containing non-whitespace text after
// trimming its surrounding whitespace.
func FirstTrimmed(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

// CommonPrefixLen returns the BYTE length of the longest common prefix of a and b —
// the shared "how far do these two strings agree?" primitive behind both fuzzy
// name-matching (rank a candidate by how much of the query it shares) and decode
// divergence reporting (name the first byte at which two decodings differ).
//
// The unit is bytes, not runes: a divergence inside a multi-byte rune yields a length
// that lands mid-rune. That is what the decode auditors WANT (they report the exact
// byte offset), and it is preserved verbatim from the per-package copies this replaced.
func CommonPrefixLen(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// Tail returns the last n BYTES of value's trimmed form, or the whole trimmed
// value when it is already at most n bytes. It is the shared "keep the end of a
// long command's output" clamp: a failing subprocess puts its diagnosis last, so
// a detail field quotes the tail, never the head.
//
// The cut is by byte, not by rune: a tail that lands mid-rune yields a leading
// partial UTF-8 sequence. That is preserved verbatim from the per-package copies
// this replaced (they clamp git/gh stderr, which is ASCII in practice), and it is
// what keeps n an exact bound on the returned size. A caller that needs a
// rune-safe cut must not use this.
func Tail(value string, n int) string {
	value = strings.TrimSpace(value)
	if len(value) <= n {
		return value
	}
	return value[len(value)-n:]
}

// DashIfBlank returns "-" for empty or whitespace-only text and otherwise
// preserves the original value. It is the compact placeholder used by reports.
func DashIfBlank(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return value
}
