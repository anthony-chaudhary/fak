package cacheheadlines

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// fak-test:runtime fast est=1ms lane=default
func TestFindingExcerptKeepsUTF8AndByteBudget(t *testing.T) {
	prefix := func(n int) string { return "cache win " + strings.Repeat("x", n-len("cache win ")) }
	for _, tc := range []struct{ name, line, want string }{
		{"short trimmed Unicode", "  cache win café  ", "cache win café"},
		{"ASCII clipped", prefix(170), prefix(160)},
		{"two byte rune split", prefix(159) + "étail", prefix(159)},
		{"three byte rune split", prefix(158) + "界tail", prefix(158)},
		{"four byte rune split", prefix(157) + "😀tail", prefix(157)},
		{"complete rune at cut", prefix(158) + "étail", prefix(158) + "é"},
		{"exact Unicode byte budget", prefix(156) + "😀", prefix(156) + "😀"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ScanText("\n"+tc.line+"\n", "notes.md")
			if len(got) != 1 {
				t.Fatalf("findings = %d, want 1", len(got))
			}
			if got[0].Text != tc.want || !utf8.ValidString(got[0].Text) || len(got[0].Text) > 160 {
				t.Fatalf("excerpt = %q (%d bytes), want %q", got[0].Text, len(got[0].Text), tc.want)
			}
			if got[0].File != "notes.md" || got[0].Line != 2 || got[0].Fix != DefaultFix {
				t.Fatalf("finding metadata changed: %+v", got[0])
			}
		})
	}
}
