package agentreadinessscore

import (
	"sort"
	"strconv"
	"strings"
)

// This file holds the small, dependency-free formatting and collection helpers used across
// agentreadinessscore.go. They were split out of the main file to keep it under the god-file
// ceiling; the seam is "pure stdlib utility" with no coupling to the scoring logic.

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// ftoa renders a score float the way Python str(float) does (shortest round-trip, but a
// whole number keeps its ".0": str(100.0) == "100.0"). Used everywhere Python interpolates a
// bare float into a human string.
func ftoa(x float64) string {
	s := strconv.FormatFloat(x, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

// fmt0 renders a float with Python's :.0f (round-half-to-even) formatting.
func fmt0(x float64) string { return strconv.FormatFloat(x, 'f', 0, 64) }

// pctString renders a rate the way Python's :.0% does (round-half-to-even, trailing %).
func pctString(rate float64) string { return strconv.FormatFloat(rate*100, 'f', 0, 64) + "%" }

func anyPresent(present func(string) bool, paths []string) bool {
	for _, p := range paths {
		if present(p) {
			return true
		}
	}
	return false
}

func dedup(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, b := range in {
		if !seen[b] {
			seen[b] = true
			out = append(out, b)
		}
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func anyMapToStr(m map[string]any) map[string]any { return m }

func anyToStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
