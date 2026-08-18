package strmatch

import (
	"fmt"
	"strings"
)

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

// ParseQuotedScalar decodes the parsers' strict double-quoted scalar subset.
func ParseQuotedScalar(value string) (string, error) {
	if len(value) < 2 || value[0] != '"' || value[len(value)-1] != '"' {
		return "", fmt.Errorf("expected a double-quoted string, got %q", value)
	}
	inner := value[1 : len(value)-1]
	if strings.Contains(inner, `"`) {
		return "", fmt.Errorf("unexpected quote inside string %q", value)
	}
	return inner, nil
}
