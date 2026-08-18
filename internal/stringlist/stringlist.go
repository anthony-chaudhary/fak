// Package stringlist parses compact operator-facing string lists.
package stringlist

import "strings"

// SplitCSV returns trimmed, non-empty comma-separated values in input order.
func SplitCSV(value string) []string {
	var out []string
	for _, item := range strings.Split(value, ",") {
		if item = strings.TrimSpace(item); item != "" {
			out = append(out, item)
		}
	}
	return out
}

// NonEmptyLines returns trimmed, non-empty lines in input order.
func NonEmptyLines(value string) []string {
	var out []string
	for _, line := range strings.Split(value, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	return out
}
