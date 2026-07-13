package scorecard

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
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
func IntValue(v any) int {
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

// PassMark renders the compact yes/no token used by human and Markdown
// scorecard rows. Keeping it here prevents each card from carrying an identical
// local branch while preserving the established lowercase wire text.
func PassMark(ok bool) string {
	if ok {
		return "yes"
	}
	return "no"
}

// ValueText renders the scalar and JSON values carried by scorecard payload maps.
// Strings stay unquoted, ints use decimal notation, and every other value follows
// encoding/json. Marshal failures preserve the historical empty-string fallback.
func ValueText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int:
		return strconv.Itoa(x)
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

// MetricText renders the numeric/string values used in scorecard reports.
// Float values retain their shortest decimal form; unsupported values preserve
// the cards' historical zero fallback through IntValue.
func MetricText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int:
		return strconv.Itoa(x)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	default:
		return strconv.Itoa(IntValue(v))
	}
}

// ScoreValueText renders the normalized 0..1 value corresponding to score.
func ScoreValueText(score int) string {
	return MetricText(Round3(ValueFromScore(float64(score))))
}

// CountNoun renders the compact scorecard count phrase used by debt reports.
// The established wire form appends literal "(s)" for every count except one.
func CountNoun(n int, noun string) string {
	s := fmt.Sprintf("%d %s", n, noun)
	if n != 1 {
		s += "(s)"
	}
	return s
}

// True reports whether v is the boolean true. Non-boolean payload values fail
// closed to false, matching JSON scorecard readers.
func True(v any) bool {
	b, ok := v.(bool)
	return ok && b
}
