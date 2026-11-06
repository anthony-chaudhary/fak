package strmatch

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestRelativeIndentContractedReplacementPreservesBoundary(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, content, old, replacement, want string
	}{
		{
			name:    "unterminated EOF",
			content: "start\n    a\n    b", old: "  a\n  b",
			replacement: "  x", want: "start\n    x",
		},
		{
			name:    "final CRLF overrides earlier LF",
			content: "start\n    a\n    b\r\nafter\n", old: "  a\n  b",
			replacement: "  x", want: "start\n    x\r\nafter\n",
		},
		{
			name:    "final LF overrides earlier CRLF",
			content: "start\r\n    a\r\n    b\nafter\r\n", old: "  a\n  b",
			replacement: "  x", want: "start\r\n    x\nafter\r\n",
		},
		{
			name:    "keeps replacement inner LF and relative indentation",
			content: "start\n    a\n    b\n    c\r\nafter\n", old: "  a\n  b\n  c",
			replacement: "  x\n    y", want: "start\n    x\n      y\r\nafter\n",
		},
		{
			name:    "explicit final terminator wins",
			content: "start\n    a\n    b\r\nafter\n", old: "  a\n  b",
			replacement: "  x\n", want: "start\n    x\nafter\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if strings.Contains(tc.content, tc.old) {
				t.Fatal("fixture must exercise the tolerant path")
			}
			got, ok := MatchAndReplaceRelativeIndent(tc.content, tc.old, tc.replacement)
			if !ok || got != tc.want {
				t.Fatalf("replacement ok=%t, got %q, want %q", ok, got, tc.want)
			}
		})
	}
}
