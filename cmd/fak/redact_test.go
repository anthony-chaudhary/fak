package main

import "testing"

// TestRedactSecret pins the shared redaction contract (#1419): empty values
// read "(unset)", anything 4 bytes or shorter is fully masked, and a longer
// secret leaks only its last 4 bytes behind a "****" prefix.
func TestRedactSecret(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", "(unset)"},
		{"a", "****"},
		{"abcd", "****"},
		{"abcde", "****bcde"},
		{"xoxb-1234-secret-TAIL", "****TAIL"},
	}
	for _, c := range cases {
		if got := redactSecret(c.in); got != c.want {
			t.Errorf("redactSecret(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestRedactWrappersShareBehavior proves the chatrelay and slack entry points
// stayed byte-identical after being folded onto the shared helper.
func TestRedactWrappersShareBehavior(t *testing.T) {
	for _, in := range []string{"", "abc", "abcd", "abcdefgh"} {
		want := redactSecret(in)
		if got := redact(in); got != want {
			t.Errorf("redact(%q) = %q, want %q", in, got, want)
		}
		if got := redactToken(in); got != want {
			t.Errorf("redactToken(%q) = %q, want %q", in, got, want)
		}
	}
}
