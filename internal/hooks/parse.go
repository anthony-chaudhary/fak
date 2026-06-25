package hooks

import (
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// hunkRE matches a unified-diff hunk header and captures the new-file start line. It is the Go
// twin of the Python checkers' `^@@ -\d+(?:,\d+)? \+(\d+)...` — the new-side line number is what
// every added line is numbered from.
var hunkRE = regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

// parseUnifiedAddedLines folds a `git diff --cached --unified=0` blob into per-file added lines
// with their new-file line numbers — the exact walk the Python checkers do: a `+++ b/<path>`
// sets the current file, a hunk header resets the new-line counter, a `+` line (not `+++`)
// records (newLine, text) and advances the counter, a context/`-` line advances/holds per the
// unified-diff rules. With --unified=0 there are no context lines, but we handle them anyway so
// the parser is correct for any context width.
func parseUnifiedAddedLines(diff string) map[string][]AddedLine {
	out := map[string][]AddedLine{}
	var file string
	newLine := 0
	haveHunk := false
	for _, raw := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(raw, "+++ b/"):
			file = raw[len("+++ b/"):]
			haveHunk = false
		case strings.HasPrefix(raw, "+++ "):
			// "+++ /dev/null" (a deletion) — no new-side file.
			file = ""
			haveHunk = false
		case strings.HasPrefix(raw, "@@"):
			if m := hunkRE.FindStringSubmatch(raw); m != nil {
				if n, err := strconv.Atoi(m[1]); err == nil {
					newLine = n
					haveHunk = true
				}
			}
		case strings.HasPrefix(raw, "+"):
			if file == "" || !haveHunk {
				continue
			}
			out[file] = append(out[file], AddedLine{File: file, New: newLine, Text: raw[1:]})
			newLine++
		case strings.HasPrefix(raw, "-"):
			// removed line: no new-side advance.
		case strings.HasPrefix(raw, " "):
			// context line: advances the new-side counter.
			if haveHunk {
				newLine++
			}
		}
	}
	return out
}

// sortStrings sorts in place (small wrapper so callers don't import sort directly everywhere).
func sortStrings(s []string) { sort.Strings(s) }

// itoa renders an int64 in base 10.
func itoa(n int64) string { return strconv.FormatInt(n, 10) }

// cleanSlash normalizes a forward-slash path (os.path.normpath twin for relative refs).
func cleanSlash(p string) string { return path.Clean(p) }

// dedupePreserveOrder returns xs with later duplicates removed, first-seen order kept.
func dedupePreserveOrder(xs []string) []string {
	seen := map[string]bool{}
	out := xs[:0:0]
	for _, x := range xs {
		if seen[x] {
			continue
		}
		seen[x] = true
		out = append(out, x)
	}
	return out
}
