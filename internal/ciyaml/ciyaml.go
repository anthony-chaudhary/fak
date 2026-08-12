// Package ciyaml is a stdlib-only STRUCTURAL checker for the GitHub Actions
// workflow YAML committed under .github/workflows/ (#5944, epic #5949).
//
// It is deliberately NOT a YAML parser and NOT a schema validator. fak's entire
// external dependency set is two golang.org/x extended-standard-library modules
// (go.mod: golang.org/x/term plus golang.org/x/sys indirectly), and that set is a
// standing contract — no YAML library is vendored anywhere and none may be added
// for a lint. `actionlint` does schema; this does STRUCTURE and REFERENCES:
//
//   - structure — tab indentation, a `key:value` with no space after the colon,
//     unbalanced quotes/brackets, and a dedent onto a column no enclosing block
//     ever opened (Check);
//   - references — the DAG checks a schema validator does not do: every job id
//     under `jobs:` is unique, and every `needs:` names a job id that exists in
//     the SAME workflow, with no cycle (see dag.go).
//
// # Why this is not internal/workflowlint
//
// internal/workflowlint is a DIFFERENT ARTEFACT: it grades ultracode *Workflow
// scripts* (JavaScript emitted by an ultracode session) for fak-nativeness —
// self-index, memory algebra, shared-path leasing. It never reads a byte of
// .github/. Extending it would have fused two unrelated input languages behind
// one name; this is an additive sibling instead.
//
// # Two structural lessons this package exists to encode
//
// 1. WALK the input set, never ENUMERATE it. In the tree this mechanism is
// borrowed from, the check named its inputs as a literal; .github/workflows/ grew
// two new files while the literal still named one, so two of the three workflows
// in the repo were checked by nothing at all — and the green result looked
// identical either way. Discover walks the directory. Nothing in this package or
// its tests writes a workflow filename down as coverage.
//
// 2. ONE line source, MANY scanners. "Comment-aware and block-scalar-aware" is a
// claim about EVERY scanner, and the way it silently becomes false is that each
// scanner grows its own half-right idea of what a comment is. eachStructuralLine
// is the single line source: it skips blank lines, skips the bodies of literal /
// folded block scalars (`|`, `>`) entirely, and hands every scanner a line whose
// inline comment has ALREADY been removed with quote-state tracking. A `run: |`
// body — a multi-line shell script full of colons, brackets and `#` — is opaque
// text, so it cannot declare a `needs:` edge or trip a structural check; and a
// trailing `# comment` cannot erase a real job id or a real edge, because no
// scanner ever sees the comment to be confused by it.
//
// # Scope, stated rather than implied
//
// A block-scalar body can never sit at the job-id column (its lines are strictly
// deeper than the `run: |` that opened it, which is itself strictly deeper than
// the job id), so the job-id scanner is immune to block scalars by construction
// rather than by the line source. The line source is what makes the CLAIM hold
// for the `needs:` scanner and the structural scanner, which read at every
// indent, and it is what makes the comment half hold for all of them.
package ciyaml

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// WorkflowDir is the repo-relative directory this package walks. It is the ONLY
// path literal here: the FILES inside it are always discovered (see Discover).
const WorkflowDir = ".github/workflows"

// Issues is the list of violations a checker found. Empty/nil means clean.
type Issues []string

// String joins every issue onto its own line, for readable output and test failures.
func (is Issues) String() string { return strings.Join(is, "\n") }

var (
	// blockScalarRe matches a key whose value OPENS a literal/folded block scalar:
	// `run: |`, `script: >-`, `body: |2`, `text: >+`. The indicator may carry an
	// explicit indentation digit and/or a chomping sign in either order.
	blockScalarRe = regexp.MustCompile(`:\s*[|>](?:[0-9][+\-]?|[+\-][0-9]?)?\s*$`)
	// seqBlockScalarRe is the same opener written as a bare sequence item (`- |`).
	seqBlockScalarRe = regexp.MustCompile(`^-\s*[|>](?:[0-9][+\-]?|[+\-][0-9]?)?\s*$`)
	// firstColonRe splits a plain `key:rest` so the colon-space rule can be applied
	// to the key's own colon and to nothing else.
	firstColonRe = regexp.MustCompile(`^([A-Za-z0-9_."'\-]+):(.*)$`)
	// topLevelKeyRe matches a mapping key at column 0.
	topLevelKeyRe = regexp.MustCompile(`^([A-Za-z0-9_."'\-]+):`)
)

// line is one STRUCTURAL line: never blank, never inside a block-scalar body, and
// with any inline comment already removed. Every scanner in this package consumes
// this type and nothing else — that is what keeps comment- and block-scalar
// awareness a property of one function instead of four.
type line struct {
	No     int    // 1-based line number in the file
	Indent int    // count of leading spaces AND tabs
	Lead   string // the raw leading whitespace, so the tab check has its evidence
	Raw    string // the raw line, trailing \r removed, nothing else done to it
	Text   string // Raw left-trimmed and inline-comment-stripped; "" for a pure comment
}

// eachStructuralLine is THE line source. It calls fn for every line of data that
// is structural YAML: blank lines are skipped, the body of a literal/folded block
// scalar is skipped wholesale (it is opaque text — a shell script, a heredoc, a
// prose blob — not YAML), and Text arrives with the inline comment already cut.
//
// It is unexported on purpose: a scanner cannot opt out of it by accident, and a
// caller cannot re-tokenise the file a fifth way. Check, HasTopLevelKey, JobIDs
// and NeedsRefs all go through here.
func eachStructuralLine(data []byte, fn func(line)) {
	blockUntilIndent := -1 // >= 0 while inside a literal/folded block-scalar body
	for i, raw := range strings.Split(string(data), "\n") {
		l := strings.TrimRight(raw, "\r")
		if strings.TrimSpace(l) == "" {
			continue
		}
		indent := len(l) - len(strings.TrimLeft(l, " \t"))

		if blockUntilIndent >= 0 {
			if indent > blockUntilIndent {
				continue // block-scalar content: opaque, not structural YAML
			}
			blockUntilIndent = -1
		}

		content := strings.TrimLeft(l, " \t")
		text := ""
		if !strings.HasPrefix(content, "#") {
			text = stripInlineComment(content)
			if blockScalarRe.MatchString(text) || seqBlockScalarRe.MatchString(text) {
				blockUntilIndent = indent
			}
		}
		fn(line{No: i + 1, Indent: indent, Lead: l[:indent], Raw: l, Text: text})
	}
}

// stripInlineComment cuts s at the first unquoted '#' that starts a comment
// (preceded by start-of-line or whitespace, per the YAML spec), tracking single
// and double quote state so a literal '#' inside a quoted scalar — a colour, a
// `${{ }}` expression, an issue reference — is never mistaken for one.
func stripInlineComment(s string) string {
	inSingle, inDouble := false, false
	for i, r := range s {
		switch {
		case r == '\'' && !inDouble:
			inSingle = !inSingle
		case r == '"' && !inSingle:
			inDouble = !inDouble
		case r == '#' && !inSingle && !inDouble:
			if i == 0 || s[i-1] == ' ' || s[i-1] == '\t' {
				return strings.TrimRight(s[:i], " \t")
			}
		}
	}
	return strings.TrimRight(s, " \t")
}

// Check returns every structural violation in data. name only prefixes messages
// (use the repo-relative path so a finding is clickable).
func Check(name string, data []byte) Issues {
	var issues Issues
	// open is the stack of indent columns an enclosing block has opened. A dedent
	// must land on one of them; landing between two is the "inconsistent sibling
	// indent" YAML rejects but a human eye slides straight over.
	open := []int{}

	eachStructuralLine(data, func(l line) {
		if strings.Contains(l.Lead, "\t") {
			issues = append(issues, fmt.Sprintf("%s:%d: tab used for indentation", name, l.No))
			return
		}
		if l.Text == "" || l.Text == "---" || l.Text == "..." {
			return // whole-line comment or a document marker: no structure to check
		}

		// Sibling-indent consistency. A sequence item written at its parent key's
		// own column is valid YAML and extremely common (`branches:` then `- main`
		// at the same indent), so a column that merely REPEATS an open one is fine;
		// only a dedent onto a column nothing opened is a finding.
		switch {
		case len(open) == 0 || l.Indent > open[len(open)-1]:
			open = append(open, l.Indent)
		case l.Indent < open[len(open)-1]:
			for len(open) > 0 && open[len(open)-1] > l.Indent {
				open = open[:len(open)-1]
			}
			if len(open) == 0 || open[len(open)-1] != l.Indent {
				issues = append(issues, fmt.Sprintf(
					"%s:%d: inconsistent sibling indent: dedent to column %d, which no enclosing block opened (open columns: %v)",
					name, l.No, l.Indent, open))
				open = append(open, l.Indent)
			}
		}

		item := l.Text
		switch {
		case strings.HasPrefix(item, "- "):
			item = strings.TrimPrefix(item, "- ")
		case item == "-":
			item = ""
		}
		if item != "" {
			if m := firstColonRe.FindStringSubmatch(item); m != nil {
				key, val := m[1], m[2]
				if val != "" && !strings.HasPrefix(val, " ") {
					issues = append(issues, fmt.Sprintf("%s:%d: %q missing space after ':' (key %q)", name, l.No, item, key))
				} else if iss := plainValueIssue(key, val); iss != "" {
					issues = append(issues, fmt.Sprintf("%s:%d: %s", name, l.No, iss))
				}
			}
		}

		if iss := balanceIssue(l.Text); iss != "" {
			issues = append(issues, fmt.Sprintf("%s:%d: %s", name, l.No, iss))
		}
	})
	return issues
}

// plainValueIssue reports the one YAML error a human eye slides straight over and
// every other scanner here misses: a SECOND mapping indicator inside a plain
// scalar value. `required: true  - type: dropdown` is two mapping keys welded onto
// one line by a botched edit; the colons all have their space, the quotes and
// brackets balance, nothing dedents, so Check passed it and the file shipped
// unparseable. A plain scalar may not contain ": " or end in ":" — that is exactly
// the "mapping values are not allowed here" the parser raises — so the rule is
// precise rather than heuristic, and a colon with no space after it (`foo:bar`, a
// URL, a Windows path) is legal and stays legal.
//
// Only a `key: value` line is examined. A continuation line of a multi-line plain
// scalar carries no key, never matches firstColonRe, and is therefore never judged.
func plainValueIssue(key, val string) string {
	v := strings.TrimSpace(val)
	if v == "" {
		return "" // `key:` opening a nested block — no plain value to judge
	}
	masked := strings.TrimRight(maskInert(v), " ")
	switch {
	case strings.Contains(masked, ": "):
		return fmt.Sprintf("mapping values are not allowed here: plain value of key %q contains \": \" "+
			"(two mapping keys welded onto one line, or a value that needed quoting) — %q", key, v)
	case strings.HasSuffix(masked, ":"):
		return fmt.Sprintf("mapping values are not allowed here: plain value of key %q ends in \":\" "+
			"(quote the value, or move the nested key to its own line) — %q", key, v)
	}
	return ""
}

// maskInert blanks the spans of a value in which a colon carries no structural
// meaning — quoted scalars ("Today: …"), and flow collections, which covers
// `{a: 1}`, `[x]` and the `${{ }}` expressions workflows are full of.
//
// It is deliberately BIASED TOWARD MASKING: an apostrophe in a plain scalar
// (`name: don't cancel`) opens a quote span that never closes, so the rest of the
// line is blanked. That can only DROP a finding, never invent one — the same
// reason balanceIssue declines to count single quotes at all.
func maskInert(s string) string {
	out := []rune(s)
	var inSingle, inDouble bool
	depth := 0
	for i, r := range out {
		switch {
		case inSingle:
			inSingle = r != '\''
		case inDouble:
			inDouble = r != '"'
		case r == '\'':
			inSingle = true
		case r == '"':
			inDouble = true
		case r == '{' || r == '[':
			depth++
		case r == '}' || r == ']':
			if depth > 0 {
				depth--
			}
		case depth == 0:
			continue // structural text: leave it for the scanner to read
		}
		out[i] = ' '
	}
	return string(out)
}

// balanceIssue reports an unbalanced bracket/brace or an odd double-quote count on
// a single structural line. Single quotes are NOT counted: an apostrophe in a plain
// scalar (`name: don't cancel`) is legal YAML and would be nothing but noise.
func balanceIssue(s string) string {
	if c := strings.Count(s, `"`); c%2 != 0 {
		return fmt.Sprintf("odd number of double quotes (%d)", c)
	}
	if strings.Count(s, "[") != strings.Count(s, "]") {
		return "unbalanced [ ]"
	}
	if strings.Count(s, "{") != strings.Count(s, "}") {
		return "unbalanced { }"
	}
	return ""
}

// HasTopLevelKey reports whether data has a mapping key at column 0 named key —
// "name", "on", "jobs" for a GitHub Actions workflow. A workflow with no `on:` is
// accepted by GitHub and simply never triggers, so the gate it was written to be
// is absent with no error anywhere.
func HasTopLevelKey(data []byte, key string) bool {
	found := false
	eachStructuralLine(data, func(l line) {
		if l.Indent != 0 || l.Text == "" {
			return
		}
		if m := topLevelKeyRe.FindStringSubmatch(l.Text); m != nil && strings.Trim(m[1], `"'`) == key {
			found = true
		}
	})
	return found
}

// Discover WALKS root/.github/workflows and returns every *.yml / *.yaml file it
// finds, as repo-relative slash paths, sorted.
//
// It is a walk and not a list because a list goes stale INVISIBLY: the checker
// keeps passing while the workflows added after it was written are covered by
// nothing, and the green result is identical either way. A missing directory is
// not an error — it returns nothing, and the caller's non-vacuity check is what
// turns "found nothing" into a loud result rather than a quiet pass.
func Discover(root string) ([]string, error) {
	dir := filepath.Join(root, filepath.FromSlash(WorkflowDir))
	var out []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		switch strings.ToLower(filepath.Ext(path)) {
		case ".yml", ".yaml":
			rel, relErr := filepath.Rel(root, path)
			if relErr != nil {
				return relErr
			}
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}
