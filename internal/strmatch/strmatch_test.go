package strmatch

import "testing"

func TestContainsAny(t *testing.T) {
	cases := []struct {
		s    string
		subs []string
		want bool
	}{
		{"merge conflict in rebase", []string{"auth", "conflict"}, true},
		{"clean run", []string{"auth", "conflict"}, false},
		{"anything", nil, false},
		{"", []string{""}, true}, // strings.Contains(s, "") is true — pinned, matching the folded copies
		{"ERROR: boom", []string{"WARN", "ERROR"}, true},
	}
	for _, tc := range cases {
		if got := ContainsAny(tc.s, tc.subs...); got != tc.want {
			t.Errorf("ContainsAny(%q, %v) = %v, want %v", tc.s, tc.subs, got, tc.want)
		}
	}
}

func TestFirstContained(t *testing.T) {
	phrase, ok := FirstContained("the provider probably has it", []string{"skip re-send", "probably has"})
	if !ok || phrase != "probably has" {
		t.Fatalf("FirstContained = (%q, %v), want (\"probably has\", true)", phrase, ok)
	}
	if p, ok := FirstContained("clean", []string{"dirty"}); ok || p != "" {
		t.Fatalf("FirstContained miss = (%q, %v), want (\"\", false)", p, ok)
	}
	// First match wins in needle order — a caller's witness must be deterministic.
	if p, _ := FirstContained("ab", []string{"b", "a"}); p != "b" {
		t.Fatalf("needle order not honored: got %q, want \"b\"", p)
	}
}

func TestFirstNonBlankPreservesOriginalValue(t *testing.T) {
	if got := FirstNonBlank("", " \t ", "  ready  ", "later"); got != "  ready  " {
		t.Fatalf("FirstNonBlank = %q, want original non-blank value", got)
	}
	if got := FirstNonBlank("", "\n"); got != "" {
		t.Fatalf("FirstNonBlank(all blank) = %q, want empty", got)
	}
}

func TestFirstNonEmptyTreatsWhitespaceAsAValue(t *testing.T) {
	if got := FirstNonEmpty("", "  ", "later"); got != "  " {
		t.Fatalf("FirstNonEmpty = %q, want whitespace value", got)
	}
	if got := FirstNonEmpty("", ""); got != "" {
		t.Fatalf("FirstNonEmpty(all empty) = %q, want empty", got)
	}
}

func TestFirstTrimmedReturnsTrimmedValue(t *testing.T) {
	if got := FirstTrimmed("", " \t ", "  ready  ", "later"); got != "ready" {
		t.Fatalf("FirstTrimmed = %q, want trimmed value", got)
	}
	if got := FirstTrimmed("", "\n"); got != "" {
		t.Fatalf("FirstTrimmed(all blank) = %q, want empty", got)
	}
}

func TestDashIfBlankPreservesNonblankValue(t *testing.T) {
	for _, blank := range []string{"", " \t ", "\n"} {
		if got := DashIfBlank(blank); got != "-" {
			t.Fatalf("DashIfBlank(%q) = %q, want dash", blank, got)
		}
	}
	if got := DashIfBlank("  ready  "); got != "  ready  " {
		t.Fatalf("DashIfBlank(nonblank) = %q, want original value", got)
	}
}

func TestTailKeepsTrimmedEndAndWholeShortValue(t *testing.T) {
	if got := Tail("  abcdefgh  ", 3); got != "fgh" {
		t.Fatalf("Tail(long) = %q, want the last 3 bytes of the trimmed value", got)
	}
	// "at most n" is inclusive, and the value comes back TRIMMED even when it is
	// short enough to be returned whole -- both copies this replaced trimmed first
	// and callers quote the result straight into a report line.
	if got := Tail("  abc  ", 3); got != "abc" {
		t.Fatalf("Tail(exactly n) = %q, want the whole trimmed value", got)
	}
	if got := Tail("  ab  ", 3); got != "ab" {
		t.Fatalf("Tail(shorter than n) = %q, want the whole trimmed value", got)
	}
	if got := Tail("  ab  ", 0); got != "" {
		t.Fatalf("Tail(n=0) = %q, want empty", got)
	}
}

func TestCommonPrefixLenCountsBytesAndBothBounds(t *testing.T) {
	if got := CommonPrefixLen("abcdef", "abcxyz"); got != 3 {
		t.Fatalf("CommonPrefixLen(diverging) = %d, want 3", got)
	}
	// Equal strings and a proper prefix each run the loop to a LENGTH bound rather
	// than a byte mismatch; an off-by-one in either bound only shows up here.
	if got := CommonPrefixLen("abc", "abc"); got != 3 {
		t.Fatalf("CommonPrefixLen(equal) = %d, want 3", got)
	}
	if got := CommonPrefixLen("ab", "abcd"); got != 2 {
		t.Fatalf("CommonPrefixLen(prefix) = %d, want 2", got)
	}
	if got := CommonPrefixLen("", "abc"); got != 0 {
		t.Fatalf("CommonPrefixLen(empty) = %d, want 0", got)
	}
	// BYTE semantics, deliberately: two strings whose first rune differs only in its
	// trailing byte still agree on the leading byte, so the answer is 1, not 0.
	if got := CommonPrefixLen("\xc3\xa9", "\xc3\xa8"); got != 1 {
		t.Fatalf("CommonPrefixLen(shared lead byte) = %d, want 1 byte", got)
	}
}
