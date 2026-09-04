// Package ctxplans is the CONTEXT-PLAN-REQUIRED advisory lint (R4, #2202, epic
// #2198). It is the code form of doctrine law L7 — "every surface declares its
// context plan" (docs/notes/CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md): a new
// cmd/fak verb (or in-repo skill) states at design time what enters the window,
// what pages out, and what warms, as structured data co-located with the verb —
// a `//fak:ctxplan verb=<name> enters="…" pages="…" warms="…"` comment directive
// (skills use the `<!-- fak:ctxplan skill=<name> … -->` HTML-comment twin in
// SKILL.md). The lint enumerates the context-touching surfaces and lists the
// UNDECLARED ones as debt. This rung is ADVISORY: the debt count is emitted, never
// gated (the maturity-ladder style — immaturity is not a defect). The witness is a
// unit-level fact: adding a context-touching verb with no declaration raises the
// debt count by exactly one.
//
// It is a sibling of internal/ctxknobs (R1, #2199, the manual-overlay counter):
// same pure-filesystem-walk shape, stdlib-only, imports nothing internal, off the
// hot path.
package ctxplans

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// SurfaceKind is what sort of surface declares (or should declare) a context plan.
type SurfaceKind string

const (
	KindVerb  SurfaceKind = "verb"  // a cmd/fak dispatch verb
	KindSkill SurfaceKind = "skill" // an in-repo .claude/skills skill
)

// Surface is one context-touching surface with its declaration status. A declared
// surface carries the plan fields the author wrote; an undeclared one is debt.
type Surface struct {
	Kind     SurfaceKind `json:"kind"`
	Name     string      `json:"name"`
	Declared bool        `json:"declared"`
	// File:Line is the declaration site (the directive) when Declared, else the
	// surface's own source (verb handler file / SKILL.md) as the place a plan is owed.
	File   string `json:"file,omitempty"`
	Line   int    `json:"line,omitempty"`
	Enters string `json:"enters,omitempty"`
	Pages  string `json:"pages,omitempty"`
	Warms  string `json:"warms,omitempty"`
}

// Report is the sorted lint result over a tree.
type Report struct {
	Surfaces []Surface `json:"surfaces"`
	// DeclaredVerbs is the count of context verbs carrying a real declaration —
	// the done-condition floor (#2202 wants >= 10 from the spine's survey).
	DeclaredVerbs int `json:"declared_verbs"`
	// Debt is the count of context-touching surfaces (verbs + skills) with no
	// real declaration — the undeclared-surface count this lint emits, advisory.
	Debt int `json:"debt"`
}

// Scan walks root and returns the sorted context-plan lint report. It is
// deterministic: the same tree yields byte-identical output. A missing scan root
// (no cmd/fak/main.go, no .claude/skills) is not an error — that source simply
// contributes nothing, and Scan returns an empty report.
//
// Contract: Scan preserves the ordering invariant that returned Surfaces are
// sorted deterministically by Kind then Name.
func Scan(root string) (Report, error) {
	directives, err := scanDirectives(filepath.Join(root, "cmd", "fak"))
	if err != nil {
		return Report{}, err
	}

	var surfaces []Surface

	tokens, err := dispatchVerbs(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		return Report{}, err
	}
	for _, tok := range tokens {
		d, declared := directives["verb:"+tok]
		// A surface is context-touching if its NAME is a context verb, OR the
		// author declared a plan for it (declaring a plan asserts it is one).
		if !isContextVerbName(tok) && !declared {
			continue
		}
		s := Surface{Kind: KindVerb, Name: tok, Declared: declared}
		if declared {
			s.File, s.Line = d.file, d.line
			s.Enters, s.Pages, s.Warms = d.enters, d.pages, d.warms
		}
		surfaces = append(surfaces, s)
	}

	skills, err := scanSkills(filepath.Join(root, ".claude", "skills"))
	if err != nil {
		return Report{}, err
	}
	surfaces = append(surfaces, skills...)

	sort.Slice(surfaces, func(i, j int) bool {
		a, b := surfaces[i], surfaces[j]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		return a.Name < b.Name
	})

	rep := Report{Surfaces: surfaces}
	for _, s := range surfaces {
		switch {
		case !s.Declared:
			rep.Debt++
		case s.Kind == KindVerb:
			rep.DeclaredVerbs++
		}
	}
	return rep, nil
}

// declaration is one parsed //fak:ctxplan directive.
type declaration struct {
	file   string
	line   int
	enters string
	pages  string
	warms  string
}

// complete is true when all three plan fields are present — a bare marker with a
// missing enters/pages/warms is NOT a real declaration (the L7 teeth: the author
// must actually state what enters, pages, and warms).
func (d declaration) complete() bool {
	return d.enters != "" && d.pages != "" && d.warms != ""
}

var (
	reCtxplanVerb  = regexp.MustCompile(`//fak:ctxplan\b`)
	reCtxplanSkill = regexp.MustCompile(`<!--\s*fak:ctxplan\b`)
	reSurfaceName  = regexp.MustCompile(`\b(verb|skill)=(?:"([^"]+)"|([A-Za-z0-9._-]+))`)
	reEnters       = regexp.MustCompile(`\benters="([^"]*)"`)
	rePages        = regexp.MustCompile(`\bpages="([^"]*)"`)
	reWarms        = regexp.MustCompile(`\bwarms="([^"]*)"`)
)

// parseDirective parses one directive line into (selector "verb:name"/"skill:name",
// declaration). ok is false when the line carries no well-formed, COMPLETE plan.
func parseDirective(text, file string, line int) (selector string, d declaration, ok bool) {
	nm := reSurfaceName.FindStringSubmatch(text)
	if nm == nil {
		return "", declaration{}, false
	}
	name := nm[2]
	if name == "" {
		name = nm[3]
	}
	d = declaration{file: file, line: line}
	if m := reEnters.FindStringSubmatch(text); m != nil {
		d.enters = strings.TrimSpace(m[1])
	}
	if m := rePages.FindStringSubmatch(text); m != nil {
		d.pages = strings.TrimSpace(m[1])
	}
	if m := reWarms.FindStringSubmatch(text); m != nil {
		d.warms = strings.TrimSpace(m[1])
	}
	if !d.complete() {
		return "", declaration{}, false
	}
	return nm[1] + ":" + name, d, true
}

// scanDirectives walks cmd/fak/*.go (excluding _test.go) collecting every complete
// //fak:ctxplan verb= directive, keyed "verb:<name>". The first complete directive
// for a name wins (deterministic by sorted filename then line).
func scanDirectives(dir string) (map[string]declaration, error) {
	out := map[string]declaration{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	for _, name := range names {
		rel := "cmd/fak/" + name
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		lineNo := 0
		for sc.Scan() {
			lineNo++
			text := sc.Text()
			if !reCtxplanVerb.MatchString(text) {
				continue
			}
			if sel, d, ok := parseDirective(text, rel, lineNo); ok {
				if _, seen := out[sel]; !seen {
					out[sel] = d
				}
			}
		}
		cerr := sc.Err()
		f.Close()
		if cerr != nil {
			return nil, cerr
		}
	}
	return out, nil
}

// --- dispatch-verb enumeration (cmd/fak/main.go) ---

var reCaseFirstToken = regexp.MustCompile(`^\s*case\s+"([^"]+)"`)

// stripLineComment returns the text before a `//` line comment. Case labels and
// call bodies in the dispatch switch carry no `//` inside string literals, so a
// plain cut at the first `//` is safe here and neutralizes brace-bearing comments.
func stripLineComment(text string) string {
	if i := strings.Index(text, "//"); i >= 0 {
		return text[:i]
	}
	return text
}

// dispatchVerbs returns the canonical verb tokens from cmd/fak/main.go's dispatch
// switches. The original router used one `switch os.Args[1]`; the split router uses
// several `dispatch*Verb` functions with `switch name`. Both shapes are scanned so
// splitting the god-file cannot silently zero this lint. A missing/unreadable main.go
// yields nil (an installed binary outside a repo).
func dispatchVerbs(mainPath string) ([]string, error) {
	f, err := os.Open(mainPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var tokens []string
	seen := map[string]bool{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	inSwitch := false
	depth := 0 // brace depth relative to the switch body
	for sc.Scan() {
		text := sc.Text()
		if !inSwitch {
			// Match code only: comments in this file describe both router shapes.
			code := strings.TrimSpace(stripLineComment(text))
			if strings.HasPrefix(code, "switch os.Args[1]") || strings.HasPrefix(code, "switch name") {
				inSwitch = true
				depth = strings.Count(code, "{") - strings.Count(code, "}")
			}
			continue
		}
		// depth==1 is the switch body itself; case labels live at that level.
		if depth == 1 {
			if m := reCaseFirstToken.FindStringSubmatch(text); m != nil {
				if tok := m[1]; !seen[tok] {
					seen[tok] = true
					tokens = append(tokens, tok)
				}
			}
		}
		// Count braces on the code half only — a `{`/`}` inside a trailing line
		// comment must not drift the depth (the switch body carries such comments).
		code := stripLineComment(text)
		depth += strings.Count(code, "{") - strings.Count(code, "}")
		if depth <= 0 {
			inSwitch = false // keep scanning: the split router has several switches
		}
	}
	return tokens, sc.Err()
}

// --- context classification ---

// contextSegments are the hyphen-delimited name segments that mark a verb as
// context/cache/session-touching. Matching on whole segments (not raw substrings)
// keeps it precise: "relay" matches `relay` but not `chatrelay`; "memory" matches
// `memory-read` but not `memgate`; and `windowgate` is not swept in.
var contextSegments = map[string]bool{
	"ctx": true, "context": true, "contextq": true, "compact": true,
	"session": true, "sessions": true, "headroom": true, "recall": true,
	"rehydrate": true, "vcache": true, "cache": true, "cachevalue": true,
	"relay": true, "resume": true, "dream": true, "memory": true,
	"kv": true, "kvbm": true,
}

// isContextVerbName is true when any hyphen-delimited segment of the verb name is
// a context segment.
func isContextVerbName(name string) bool {
	for _, seg := range strings.Split(strings.ToLower(name), "-") {
		if contextSegments[seg] {
			return true
		}
	}
	return false
}

// --- skill scanning (.claude/skills) ---

var (
	reSkillMgmtVerb = regexp.MustCompile(`(?i)\b(compact|prune|trim|shrink|truncat|hygiene|manage|overlay|clear|/compact|/clear)\b`)
	reSkillCtxNoun  = regexp.MustCompile(`(?i)\b(context|memory|window|memory\.md|auto-memory|token budget)\b`)
	reSkillNameTok  = regexp.MustCompile(`(?i)(memory|context|compact|ctx)`)
)

// scanSkills returns the context-management skills under dir, each marked declared
// if its SKILL.md carries a complete `<!-- fak:ctxplan skill=<name> … -->` directive.
func scanSkills(dir string) ([]Surface, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Surface
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		rel := ".claude/skills/" + e.Name() + "/SKILL.md"
		name, desc, decl, nameLine, err := readSkill(filepath.Join(dir, e.Name(), "SKILL.md"))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if name == "" {
			name = e.Name()
		}
		if !isContextSkill(name, desc) {
			continue
		}
		s := Surface{Kind: KindSkill, Name: name, File: rel, Line: nameLine}
		if decl.complete() {
			s.Declared = true
			s.Line = decl.line
			s.Enters, s.Pages, s.Warms = decl.enters, decl.pages, decl.warms
		}
		out = append(out, s)
	}
	return out, nil
}

// isContextSkill is true when a skill's reason for existing is context/memory
// management: its name carries a context/memory token AND its description pairs a
// management verb with a context noun (the same isolation ctxknobs uses).
func isContextSkill(name, desc string) bool {
	return reSkillNameTok.MatchString(name) &&
		reSkillMgmtVerb.MatchString(desc) &&
		reSkillCtxNoun.MatchString(desc)
}

// readSkill extracts name + description from the leading --- frontmatter (plus the
// 1-based line of name: for provenance) and the first complete fak:ctxplan HTML
// directive anywhere in the file.
func readSkill(path string) (name, desc string, decl declaration, nameLine int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", declaration{}, 0, err
	}
	defer f.Close()

	rel := path
	if i := strings.Index(filepath.ToSlash(path), ".claude/"); i >= 0 {
		rel = filepath.ToSlash(path)[i:]
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	inFrontmatter := false
	for sc.Scan() {
		line++
		text := sc.Text()
		trimmed := strings.TrimSpace(text)
		if line == 1 && trimmed == "---" {
			inFrontmatter = true
			continue
		}
		if inFrontmatter && trimmed == "---" {
			inFrontmatter = false
		} else if inFrontmatter {
			if v, ok := frontmatterValue(text, "name"); ok && name == "" {
				name, nameLine = v, line
			}
			if v, ok := frontmatterValue(text, "description"); ok && desc == "" {
				desc = v
			}
		}
		if decl.line == 0 && reCtxplanSkill.MatchString(text) {
			if _, d, ok := parseDirective(text, rel, line); ok {
				decl = d
			}
		}
	}
	if nameLine == 0 {
		nameLine = 1
	}
	return name, desc, decl, nameLine, sc.Err()
}

// frontmatterValue parses a top-level `key: value` line, stripping optional quotes.
func frontmatterValue(text, key string) (string, bool) {
	if len(text) > 0 && (text[0] == ' ' || text[0] == '\t') {
		return "", false
	}
	prefix := key + ":"
	if !strings.HasPrefix(text, prefix) {
		return "", false
	}
	v := strings.TrimSpace(text[len(prefix):])
	v = strings.TrimSuffix(strings.TrimPrefix(v, `"`), `"`)
	v = strings.TrimSuffix(strings.TrimPrefix(v, `'`), `'`)
	return v, true
}
