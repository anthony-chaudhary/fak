package main

import "testing"

func TestTruncateTableFieldBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name string
		s    string
		n    int
		want string
	}{
		{"unchanged", "abcd", 4, "abcd"},
		{"shorter", "abc", 8, "abc"},
		{"zero", "abc", 0, ""},
		{"one", "abc", 1, "a"},
		{"marked", "abcdef", 4, "abc…"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := truncateTableField(tc.s, tc.n); got != tc.want {
				t.Fatalf("truncateTableField(%q,%d) = %q, want %q", tc.s, tc.n, got, tc.want)
			}
		})
	}
}
