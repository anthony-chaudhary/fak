package witness

import (
	"context"
	"sort"
	"strings"
)

// buildTagsFromTestSources derives the build-constraint tag set from the bodies of the changed
// test files (#13243). It reads the leading `//go:build` expression (and the legacy `// +build`
// form), extracting each tag token and dropping the boolean operators. Unknown tokens (the
// `go1.21` release tags, `cgo`) are kept as-is: `go test -tags` accepts them, and passing an
// irrelevant tag is harmless, whereas DROPPING a real one silently excludes the witness again.
// A file with no constraint contributes nothing.
func buildTagsFromTestSources(bodies []string) []string {
	seen := map[string]bool{}
	var tags []string
	for _, body := range bodies {
		for _, tag := range parseBuildConstraints(body) {
			if seen[tag] {
				continue
			}
			seen[tag] = true
			tags = append(tags, tag)
		}
	}
	sort.Strings(tags)
	return tags
}

// parseBuildConstraints extracts the tag tokens from a source file's leading build constraints,
// stopping at the first non-comment, non-blank line (constraints must precede the package clause).
func parseBuildConstraints(body string) []string {
	var tags []string
	for _, line := range strings.Split(body, "\n") {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		if !strings.HasPrefix(s, "//") {
			break // the package clause (or any code) ends the build-constraint block
		}
		expr := ""
		switch {
		case strings.HasPrefix(s, "//go:build"):
			expr = strings.TrimSpace(strings.TrimPrefix(s, "//go:build"))
		case strings.HasPrefix(s, "// +build"):
			// Legacy form: space-separated tags on the line, OPTIONS on extra lines.
			// Legacy `,` is AND and a space is OR, but for -tags purposes either way each
			// token is a tag name, so the naive split is sufficient.
			expr = strings.TrimSpace(strings.TrimPrefix(s, "// +build"))
			expr = strings.ReplaceAll(expr, ",", " ")
		default:
			continue
		}
		tags = append(tags, tokenizeConstraint(expr)...)
	}
	return tags
}

// tokenizeConstraint splits a build expression into its tag tokens, dropping the boolean
// operators/punctuation (`&&`, `||`, `!`, `(`, `)`). `!windows` keeps its `!` so `go test`
// applies the negation the file declared.
func tokenizeConstraint(expr string) []string {
	fields := strings.FieldsFunc(expr, func(r rune) bool {
		return r == '(' || r == ')' || r == '&' || r == '|'
	})
	var tags []string
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		tags = append(tags, f)
	}
	return tags
}

// resolveGoBuildTags reads each changed test file's content AT ref through the injected git
// runner and returns the derived tag set. Any read failure yields nil (the untagged default),
// never a guessed tag set: an unreadable constraint means we cannot know the tags, and the
// caller's fail-closed ABSTAIN path handles the untagged run's outcome.
func resolveGoBuildTags(ctx context.Context, git Runner, repoDir, ref string, tests []string) []string {
	if git == nil {
		git = gitRunner
	}
	var bodies []string
	for _, rel := range tests {
		if !strings.HasSuffix(rel, "_test.go") {
			continue
		}
		content, code, err := git(ctx, repoDir, "show", ref+":"+rel)
		if err != nil || code != 0 {
			continue
		}
		bodies = append(bodies, content)
	}
	return buildTagsFromTestSources(bodies)
}
