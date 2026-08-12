package architest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIssueTemplatesAreWellFormed is the always-on gate for .github/ISSUE_TEMPLATE.
//
// It exists because the directory had NO gate at all and shipped broken. The
// don't-make-me-think problem-map rollout (45fafabb35) added a "Primary fak
// problem" dropdown to feature-request.yml and welded its first line onto the
// previous field's `required: true`, producing `required: true  - type: dropdown`.
// That is unparseable YAML, and GitHub's response to an unparseable issue form is
// to DROP it from the chooser — no error, no red build, nothing in the repo that
// looks wrong. The feature-request form was simply absent for everyone who tried
// to file one, and the only signal was the absence itself. It was the only
// unparseable YAML file anywhere under .github/.
//
// Two failure modes, neither caught by anything else in `go test ./...`:
//
//  1. STRUCTURE — the file does not parse, so the form vanishes.
//  2. SHAPE — it parses but lacks a key GitHub requires of an issue FORM
//     (`name`, `description`, `body`). GitHub ignores it just as silently.
//
// The check is stdlib-only and deliberately NOT a YAML parser: fak's entire
// dependency set is two golang.org/x modules and that floor is a standing
// contract, so no YAML library may be vendored for a lint. It scans for the one
// error that actually shipped, which the parser calls "mapping values are not
// allowed here": a plain scalar may not contain ": " or end in ":".
//
// The input set is WALKED, never enumerated: a template added tomorrow is covered
// the moment it lands. config.yml is a different schema in the same directory (the
// chooser's contact links, not a form), so it gets its own floor rather than a skip.
func TestIssueTemplatesAreWellFormed(t *testing.T) {
	root := filepath.Dir(internalDir(t)) // repo root = parent of internal/
	dir := filepath.Join(root, ".github", "ISSUE_TEMPLATE")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read issue-template dir %s: %v", dir, err)
	}

	checked := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if ext := strings.ToLower(filepath.Ext(name)); ext != ".yml" && ext != ".yaml" {
			continue
		}
		rel := ".github/ISSUE_TEMPLATE/" + name
		data, readErr := os.ReadFile(filepath.Join(dir, name))
		if readErr != nil {
			t.Errorf("read %s: %v", rel, readErr)
			continue
		}
		checked++

		for _, iss := range weldedMappingKeys(string(data)) {
			t.Errorf("%s:%d is structurally invalid YAML — GitHub drops an unparseable issue "+
				"template from the chooser silently, so the form is simply gone with no error "+
				"anywhere. %s", rel, iss.line, iss.msg)
		}

		if name == "config.yml" || name == "config.yaml" {
			// The issue-chooser config, not a form: a different required shape.
			if !hasTopLevelKey(string(data), "contact_links") {
				t.Errorf("%s is the issue-chooser config but has no top-level \"contact_links\" key", rel)
			}
			continue
		}
		for _, key := range []string{"name", "description", "body"} {
			if !hasTopLevelKey(string(data), key) {
				t.Errorf("%s is missing required top-level key %q — GitHub ignores an issue form "+
					"that lacks it, with no error shown anywhere", rel, key)
			}
		}
	}

	// Fail closed: zero files checked means the directory moved or the extension
	// filter drifted, and a gate that examined nothing passes green either way.
	if checked == 0 {
		t.Fatal("no issue templates were checked — this gate would be silently inert; " +
			"the .github/ISSUE_TEMPLATE layout changed")
	}
	t.Logf("issue templates checked under .github/ISSUE_TEMPLATE: %d", checked)
}

// TestWeldedMappingKeys pins the exact line that shipped feature-request.yml
// unparseable in 45fafabb35, quoted from the commit rather than paraphrased, and
// separates it from the values YAML accepts. The accept half is the load-bearing
// half: a colon is ordinary inside a quoted scalar, a flow collection, a `${{ }}`
// expression and a URL, and a scanner that cannot tell those from a real second
// mapping key is one nobody can keep switched on. Every case was confirmed against
// a real YAML parser before it was written down.
func TestWeldedMappingKeys(t *testing.T) {
	for _, tc := range []struct {
		name string
		yaml string
		want bool // true = must be reported
	}{
		{"the line that shipped", "    validations:\n      required: true  - type: dropdown\n", true},
		{"trailing colon", "a: see the following:", true},
		{"unquoted colon-space in prose", "about: note: this breaks", true},
		{"sequence item welded", "- a: b: c", true},

		{"quoted colon-space", `placeholder: "Today: …"`, false},
		{"flow mapping", "env: {A: 1, B: 2}", false},
		{"github expression", "if: ${{ github.ref == 'refs/heads/main' }}", false},
		{"url", "url: https://github.com/anthony-chaudhary/fak", false},
		{"colon with no space", "path: C:/work/fak", false},
		{"nested block opener", "attributes:", false},
		{"comment carrying a colon", "a: 1 # note: ignored", false},
		{"block scalar body", "value: |\n  Today: this is prose, not YAML\n", false},
		{"apostrophe in plain scalar", "name: don't cancel the run", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := weldedMappingKeys(tc.yaml)
			if (len(got) > 0) != tc.want {
				t.Fatalf("weldedMappingKeys(%q): reported=%v want=%v (%v)", tc.yaml, len(got) > 0, tc.want, got)
			}
		})
	}
}

// finding is one structural violation: the 1-based line and why it is wrong.
type finding struct {
	line int
	msg  string
}

// weldedMappingKeys reports every line whose plain scalar value carries a SECOND
// mapping indicator. `required: true  - type: dropdown` is two mapping keys welded
// onto one line by a botched edit: the colons all have their space, the quotes and
// brackets balance, and one line cannot dedent wrongly, so only a rule about the
// VALUE catches it. A plain scalar may not contain ": " or end in ":" — precisely
// the parser's "mapping values are not allowed here" — so the rule is exact rather
// than heuristic, and a colon with no space after it (a URL, `C:/work`) stays legal.
//
// Only a `key: value` line is judged. A continuation line of a multi-line plain
// scalar carries no key of its own and is therefore never examined.
func weldedMappingKeys(doc string) []finding {
	var out []finding
	blockUntilIndent := -1 // >= 0 while inside a literal/folded block-scalar body
	for i, raw := range strings.Split(doc, "\n") {
		l := strings.TrimRight(raw, "\r")
		if strings.TrimSpace(l) == "" {
			continue
		}
		indent := len(l) - len(strings.TrimLeft(l, " \t"))
		if blockUntilIndent >= 0 {
			if indent > blockUntilIndent {
				continue // block-scalar content: opaque text, not YAML
			}
			blockUntilIndent = -1
		}
		text := strings.TrimLeft(l, " \t")
		if strings.HasPrefix(text, "#") {
			continue
		}
		text = stripComment(text)
		if isBlockScalarOpener(text) {
			blockUntilIndent = indent
			continue
		}
		text = strings.TrimPrefix(text, "- ")

		colon := strings.Index(text, ": ")
		if colon < 0 {
			continue // no `key: value` on this line
		}
		key, val := text[:colon], strings.TrimSpace(text[colon+2:])
		if val == "" || strings.ContainsAny(key, `:#`) {
			continue
		}
		masked := strings.TrimRight(maskInert(val), " ")
		switch {
		case strings.Contains(masked, ": "):
			out = append(out, finding{i + 1, "the plain value of key \"" + key + "\" contains \": \" — " +
				"two mapping keys welded onto one line, or a value that needed quoting: " + val})
		case strings.HasSuffix(masked, ":"):
			out = append(out, finding{i + 1, "the plain value of key \"" + key + "\" ends in \":\" — " +
				"quote the value, or move the nested key to its own line: " + val})
		}
	}
	return out
}

// maskInert blanks the spans of a value where a colon carries no structural
// meaning: quoted scalars ("Today: …") and flow collections, which covers
// `{a: 1}`, `[x]` and the `${{ }}` expressions GitHub YAML is full of.
//
// It is deliberately BIASED TOWARD MASKING. An apostrophe in a plain scalar
// (`name: don't cancel`) opens a quote span that never closes, so the rest of the
// line is blanked. That can only DROP a finding, never invent one — the safe
// direction for a gate that reds the shared trunk.
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

// stripComment cuts s at the first unquoted '#' that starts a comment (preceded by
// start-of-line or whitespace, per the YAML spec), tracking quote state so a '#'
// inside a quoted scalar is never mistaken for one.
func stripComment(s string) string {
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

// isBlockScalarOpener reports whether a line's value opens a literal/folded block
// scalar (`value: |`, `description: >-`), whose body is opaque prose rather than
// YAML — an issue template's markdown blocks are full of colons.
func isBlockScalarOpener(text string) bool {
	i := strings.LastIndex(text, ":")
	if i < 0 {
		return false
	}
	rest := strings.TrimSpace(text[i+1:])
	if rest == "" {
		return false
	}
	if rest[0] != '|' && rest[0] != '>' {
		return false
	}
	return strings.Trim(rest[1:], "0123456789+-") == ""
}

// hasTopLevelKey reports whether doc has a mapping key at column 0 named key.
func hasTopLevelKey(doc, key string) bool {
	for _, raw := range strings.Split(doc, "\n") {
		l := strings.TrimRight(raw, "\r")
		if l == "" || l[0] == ' ' || l[0] == '\t' || l[0] == '#' {
			continue
		}
		if name, _, ok := strings.Cut(l, ":"); ok && strings.Trim(name, `"'`) == key {
			return true
		}
	}
	return false
}
