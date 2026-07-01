package scorecard

import (
	"os"
	"strings"
)

// The tiny shared helpers the Python cards copy-paste into every file. Centralizing them
// here is the point of the kernel: a card calls these instead of re-deriving them.

// HasAny reports whether s contains any needle, case-insensitively -- the Python _has_any
// (conflation_scorecard.py:138). Matching is substring, not word-boundary, exactly as the
// Python lowercases both sides and tests `n.lower() in s.lower()`.
func HasAny(s string, needles []string) bool {
	low := strings.ToLower(s)
	for _, n := range needles {
		if strings.Contains(low, strings.ToLower(n)) {
			return true
		}
	}
	return false
}

// Clip collapses runs of whitespace and truncates to n with a trailing "..." -- the Python
// _clip (conflation_scorecard.py:208): `" ".join(s.split())`, then `s[:n-1]+"..."` when long.
func Clip(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= n {
		return s
	}
	if n < 1 {
		return "..."
	}
	return s[:n-1] + "..."
}

// SafeRead returns a file's contents or "" on any error -- the Python _read/_safe_read that
// every tree-reading card uses so a missing surface degrades to empty, not a crash.
func SafeRead(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

// joinSemicolon joins with "; " (the Python "; ".join used in the reason line).
func joinSemicolon(parts []string) string {
	return strings.Join(parts, "; ")
}

// anyInt coerces a JSON-decoded number (or int) to int, tolerating the float64 that
// encoding/json yields for a number. Ported from guardrsi.go:702 so the kernel's Compare can
// read a debt integer out of a prior --json payload regardless of its decoded numeric type.
func anyInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

// anyFloat coerces a JSON-decoded number (or int) to float64. It is intentionally
// narrow: booleans and strings are not numeric compatibility values.
func anyFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case float64:
		return n, true
	default:
		return 0, false
	}
}
