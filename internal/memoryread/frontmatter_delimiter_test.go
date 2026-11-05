package memoryread

import "testing"

// Seven short strings with no I/O; the runtime estimate is unmeasured.
// fak-test:runtime fast est=1ms lane=default
func TestStripFrontmatterRequiresCompleteDelimiterLines(t *testing.T) {
	t.Parallel()
	const openingPrefix = "---heading\nkeep me\n---\nbody\n"
	const unclosed = "---\nname: fact\n---not-closing\nkeep me\n"
	for _, tc := range []struct {
		name, text, want string
	}{
		{"metadata key prefix", "---\nname: fact\n---note: metadata\nowner: metadata\n---\nbody\n", "body\n"},
		{"opening prefix is body", openingPrefix, openingPrefix},
		{"unclosed block is preserved", unclosed, unclosed},
		{"CRLF delimiters", "---\r\nname: fact\r\n---\r\nbody\r\n", "body\r\n"},
		{"closing delimiter at EOF", "---\nname: fact\n---", ""},
		{"delimiter horizontal whitespace", "--- \t\nname: fact\n---\t \nbody\n", "body\n"},
		{"delimiter comments", "--- # opening\nname: fact\n---\t# closing\nbody\n", "body\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripFrontmatter(tc.text); got != tc.want {
				t.Errorf("StripFrontmatter(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}
