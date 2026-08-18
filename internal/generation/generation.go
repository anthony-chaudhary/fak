// Package generation normalizes the project board's delivery horizon vocabulary.
package generation

import "strings"

var order = [...]string{"now", "next", "second-next", "future"}

// Order returns the canonical ramp order.
func Order() []string { return append([]string(nil), order[:]...) }

// Normalize returns a canonical bare horizon or "unclassified".
func Normalize(raw string) string {
	s := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(raw)), "gen/")
	for _, want := range order {
		if s == want {
			return s
		}
	}
	return "unclassified"
}

// Label returns a canonical gen/ label or an empty string for unknown input.
func Label(raw string) string {
	s := Normalize(raw)
	if s == "unclassified" {
		return ""
	}
	return "gen/" + s
}
