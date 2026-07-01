package rehome

import "strings"

// ProjectSlug returns the on-disk session-store slug Claude derives from a working
// directory: EVERY non-alphanumeric character becomes '-' (C:\work\fak ->
// C--work-fak). This mirrors the Claude Code harness rule exactly — the same
// re.sub(r"[^A-Za-z0-9]", "-", ...) that resume_resolver.project_slug applies.
//
// The substitution is per-code-point (ranging a Go string yields runes, matching
// Python's per-character re.sub): a non-ASCII-alphanumeric rune maps to a single
// '-', not to one '-' per UTF-8 byte, so a punctuated or unicode path slugs
// one-to-one exactly as the harness does. Landing a re-home copy under any other
// slug is a silent 404 from a folder claude --resume never looks in.
func ProjectSlug(path string) string {
	var b strings.Builder
	b.Grow(len(path))
	for _, r := range path {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return b.String()
}
