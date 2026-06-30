// Package corelocks parses and validates a DECLARATIVE core-lock taxonomy:
// lock classes and reason tokens carried as DATA, not a hand-coded table.
//
// The motivation (issue #1681): a core-lock checker that hard-codes its
// classes becomes a rival taxonomy that drifts from the workspace contract.
// Instead, the classes and the path globs that map onto them are declared in a
// small TOML-shaped fixture; this package reads, validates, and classifies
// against that declaration. Unknown class names, unknown reason tokens, or
// malformed declarations FAIL the local check (a parse/validation error) — but
// nothing here raises a spontaneous runtime refusal. This is data + validation
// only: the reason tokens are returned as classification metadata, never
// emitted as an enforcement action.
//
// Declaration shape (array-of-tables):
//
//	[[class]]
//	name   = "hard-self"
//	reason = "CORE_SELF_MODIFY"
//	globs  = ["internal/adjudicator/**", "internal/abi/**"]
//
// Classification: the most-specific (longest) matching glob wins; when no
// declared glob matches, a path falls through to the open-leaf class (which
// raises no reason), and a class with no usable mapping yields the
// unclassified reason.
package corelocks

import (
	"fmt"
	"path"
	"strings"
)

// The closed vocabulary. These two sets are the validation boundary: a
// declaration naming a class or reason outside them is malformed and fails
// Parse. They are intentionally hard-coded HERE (the validator's own grammar);
// the DATA that maps paths onto them lives in the fixture.

// Known lock classes, weakest-binding to strongest-binding is not implied by
// order — order here is only the membership set.
var knownClasses = map[string]bool{
	"hard-self":     true,
	"serial-core":   true,
	"soft-contract": true,
	"shadow-learn":  true,
	"open-leaf":     true,
}

// Known reason tokens (data-only; never auto-emitted as a refusal).
var knownReasons = map[string]bool{
	"CORE_SELF_MODIFY":              true,
	"CORE_SERIAL_REQUIRED":          true,
	"CORE_CONTRACT_WITNESS_MISSING": true,
	"CORE_LOCK_UNCLASSIFIED":        true,
	// The empty reason is permitted for the open-leaf class: a leaf raises no
	// lock and therefore carries no reason token.
	"": true,
}

// ClassOpenLeaf is the fall-through class for a path no declared glob claims.
const ClassOpenLeaf = "open-leaf"

// ReasonUnclassified is returned for a path that matches a declared class which
// itself carries no reason wiring — the data names a lock but cannot say why.
const ReasonUnclassified = "CORE_LOCK_UNCLASSIFIED"

// Class is one declared core-lock class: a name, the reason token it raises,
// and the path globs that map onto it.
type Class struct {
	// Name is the lock-class name; it must be a member of the known set.
	Name string
	// Reason is the data-only reason token this class raises when a path maps
	// to it. It must be a member of the known reason set. The open-leaf class
	// carries the empty reason.
	Reason string
	// Globs are repo-relative path globs. A "<dir>/**" glob means containment
	// (the dir itself or anything under it); any other glob is a single
	// path.Match (single-segment '*'/'?' wildcards).
	Globs []string
}

// Taxonomy is the parsed, validated set of core-lock classes.
type Taxonomy struct {
	Classes []Class
}

// Parse reads a core-lock declaration from TOML-shaped bytes and validates it.
// It returns an error when a declaration is malformed, names an unknown class,
// or names an unknown reason token. A successful Parse guarantees every class
// name and reason token is a member of the closed vocabulary.
func Parse(data []byte) (*Taxonomy, error) {
	tables, err := parseClassTables(data)
	if err != nil {
		return nil, err
	}
	if len(tables) == 0 {
		return nil, fmt.Errorf("corelocks: no [[class]] declarations found")
	}

	t := &Taxonomy{}
	seen := map[string]bool{}
	for i, tbl := range tables {
		name := tbl.values["name"]
		if name == "" {
			return nil, fmt.Errorf("corelocks: class #%d: missing required field %q", i+1, "name")
		}
		if !knownClasses[name] {
			return nil, fmt.Errorf("corelocks: class %q: unknown lock class (not in the declared vocabulary)", name)
		}
		if seen[name] {
			return nil, fmt.Errorf("corelocks: class %q: declared more than once", name)
		}
		seen[name] = true

		reason := tbl.values["reason"]
		if !knownReasons[reason] {
			return nil, fmt.Errorf("corelocks: class %q: unknown reason token %q", name, reason)
		}

		globs := tbl.globs
		// A non-open-leaf class with no globs cannot map any path; that is a
		// malformed declaration (it would silently never bind).
		if name != ClassOpenLeaf && len(globs) == 0 {
			return nil, fmt.Errorf("corelocks: class %q: at least one glob is required", name)
		}
		for _, g := range globs {
			if strings.TrimSpace(g) == "" {
				return nil, fmt.Errorf("corelocks: class %q: empty glob is not allowed", name)
			}
		}

		t.Classes = append(t.Classes, Class{Name: name, Reason: reason, Globs: globs})
	}
	return t, nil
}

// Classify returns the lock class and reason token for a repo-relative path.
// The most-specific (longest) matching declared glob wins. A path that no
// declared glob claims falls through to open-leaf with an empty reason. A path
// that maps to a declared class carrying no reason wiring returns that class
// with the unclassified reason.
func (t *Taxonomy) Classify(p string) (class string, reason string) {
	type hit struct {
		class  string
		reason string
		glob   string
	}
	var best *hit
	for _, c := range t.Classes {
		for _, g := range c.Globs {
			if !pathUnderGlob(g, p) {
				continue
			}
			// Most-specific wins: a longer glob is more specific. On a tie,
			// the first-declared class holds (deterministic).
			if best == nil || globSpecificity(g) > globSpecificity(best.glob) {
				h := hit{class: c.Name, reason: c.Reason, glob: g}
				best = &h
			}
		}
	}
	if best == nil {
		return ClassOpenLeaf, ""
	}
	if best.class != ClassOpenLeaf && best.reason == "" {
		return best.class, ReasonUnclassified
	}
	return best.class, best.reason
}

// Classes returns the declared class names in declaration order. A small helper
// so callers can introspect the taxonomy without reaching into the slice.
func (t *Taxonomy) ClassNames() []string {
	out := make([]string, 0, len(t.Classes))
	for _, c := range t.Classes {
		out = append(out, c.Name)
	}
	return out
}

// globSpecificity scores a glob so the longest/most-specific match wins. A
// containment "<dir>/**" glob scores by its directory depth; a literal or
// single-segment glob scores by its full segment count plus a bonus so an exact
// deeper path out-specifies a shallow "<dir>/**".
func globSpecificity(glob string) int {
	g := path.Clean(strings.ReplaceAll(glob, `\`, "/"))
	if strings.HasSuffix(glob, "/**") || glob == "**" {
		dir := path.Clean(strings.TrimSuffix(strings.ReplaceAll(glob, `\`, "/"), "**"))
		if dir == "." || dir == "/" {
			return 0
		}
		return strings.Count(dir, "/") + 1
	}
	// A non-containment glob targets a specific file/segment set; give it the
	// depth of its path plus one so it beats an equally-deep "<dir>/**".
	return strings.Count(g, "/") + 2
}

// pathUnderGlob reports whether value is a path admitted by glob. Mirrors the
// proven containment semantics used by the adjudicator: a "<dir>/**" glob is
// CONTAINMENT (the dir itself or anything inside it, with "../" escapes folded
// out by path.Clean); any other glob is a single path.Match (single-segment
// '*'/'?' wildcards).
func pathUnderGlob(glob, value string) bool {
	norm := func(s string) string { return path.Clean(strings.ReplaceAll(s, `\`, "/")) }
	v := norm(value)
	if strings.HasSuffix(glob, "/**") || glob == "**" {
		dir := norm(strings.TrimSuffix(strings.ReplaceAll(glob, `\`, "/"), "**"))
		if dir == "." || dir == "/" {
			return v != ".." && !strings.HasPrefix(v, "../") && !strings.HasPrefix(v, "/")
		}
		return v == dir || strings.HasPrefix(v, dir+"/")
	}
	ok, err := path.Match(norm(glob), v)
	return err == nil && ok
}

// --- minimal, self-contained parser for the array-of-tables fixture shape ---
//
// The fixture is intentionally a tiny, fixed subset of TOML: a sequence of
// [[class]] tables, each with bare key = "string" and key = ["str", ...] array
// lines, plus '#' comments and blank lines. Parsing it here keeps the package
// dependency-free (no TOML library is in the module). Anything outside this
// grammar is reported as a parse error so a malformed declaration fails.

type classTable struct {
	values map[string]string
	globs  []string
}

func parseClassTables(data []byte) ([]classTable, error) {
	var tables []classTable
	var cur *classTable

	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	for n, raw := range lines {
		line := stripComment(raw)
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "[[class]]" {
			tables = append(tables, classTable{values: map[string]string{}})
			cur = &tables[len(tables)-1]
			continue
		}
		if strings.HasPrefix(line, "[") {
			return nil, fmt.Errorf("corelocks: line %d: only [[class]] tables are allowed, got %q", n+1, line)
		}
		if cur == nil {
			return nil, fmt.Errorf("corelocks: line %d: key %q outside any [[class]] table", n+1, line)
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("corelocks: line %d: not a key = value declaration: %q", n+1, line)
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if key == "" {
			return nil, fmt.Errorf("corelocks: line %d: empty key", n+1)
		}
		switch key {
		case "globs":
			arr, err := parseStringArray(val)
			if err != nil {
				return nil, fmt.Errorf("corelocks: line %d: %v", n+1, err)
			}
			cur.globs = arr
		case "name", "reason":
			s, err := parseString(val)
			if err != nil {
				return nil, fmt.Errorf("corelocks: line %d: %v", n+1, err)
			}
			if _, dup := cur.values[key]; dup {
				return nil, fmt.Errorf("corelocks: line %d: duplicate key %q in table", n+1, key)
			}
			cur.values[key] = s
		default:
			return nil, fmt.Errorf("corelocks: line %d: unknown key %q", n+1, key)
		}
	}
	return tables, nil
}

// stripComment removes a trailing '#' comment that is not inside a string. The
// fixture grammar has no '#' inside its string literals, so a first-unquoted-#
// scan is sufficient.
func stripComment(s string) string {
	inStr := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"':
			inStr = !inStr
		case '#':
			if !inStr {
				return s[:i]
			}
		}
	}
	return s
}

func parseString(v string) (string, error) {
	if len(v) < 2 || v[0] != '"' || v[len(v)-1] != '"' {
		return "", fmt.Errorf("expected a double-quoted string, got %q", v)
	}
	inner := v[1 : len(v)-1]
	if strings.Contains(inner, `"`) {
		return "", fmt.Errorf("unexpected quote inside string %q", v)
	}
	return inner, nil
}

func parseStringArray(v string) ([]string, error) {
	if len(v) < 2 || v[0] != '[' || v[len(v)-1] != ']' {
		return nil, fmt.Errorf("expected a [\"...\"] array, got %q", v)
	}
	body := strings.TrimSpace(v[1 : len(v)-1])
	if body == "" {
		return []string{}, nil
	}
	var out []string
	for _, part := range splitTopLevel(body) {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		s, err := parseString(part)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// splitTopLevel splits an array body on commas. The grammar has no nested
// arrays, so a plain comma split (respecting quotes) is enough.
func splitTopLevel(body string) []string {
	var parts []string
	var b strings.Builder
	inStr := false
	for i := 0; i < len(body); i++ {
		c := body[i]
		switch c {
		case '"':
			inStr = !inStr
			b.WriteByte(c)
		case ',':
			if inStr {
				b.WriteByte(c)
			} else {
				parts = append(parts, b.String())
				b.Reset()
			}
		default:
			b.WriteByte(c)
		}
	}
	parts = append(parts, b.String())
	return parts
}
