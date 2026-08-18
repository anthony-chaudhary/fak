package strmatch

import "strings"

// StripUnquotedComment removes a trailing marker comment outside double quotes.
func StripUnquotedComment(s string, marker byte) string {
	quoted := false
	for i := range len(s) {
		switch s[i] {
		case '"':
			quoted = !quoted
		case marker:
			if !quoted {
				return s[:i]
			}
		}
	}
	return s
}

// SplitQuoted splits on separators outside double quotes.
func SplitQuoted(s string, separator byte) []string {
	var parts []string
	var current strings.Builder
	quoted := false
	for i := range len(s) {
		c := s[i]
		switch {
		case c == '"':
			quoted = !quoted
			current.WriteByte(c)
		case c == separator && !quoted:
			parts = append(parts, current.String())
			current.Reset()
		default:
			current.WriteByte(c)
		}
	}
	return append(parts, current.String())
}
