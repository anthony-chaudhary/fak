// Package godsplitplan is the boundary + hazard planner for a behavior-preserving
// Go split — the Go port of the retired tools/godsplit_plan.py (fak pythongate).
//
// The /modularize skill retires the code-quality scorecard's `architecture` debt
// (god-files > 1500 lines, god-functions > 200 lines) by MOVING top-level
// declarations into concern-scoped files in the SAME package — a semantic no-op in
// Go EXCEPT for the hazards this planner surfaces.
//
// The error-prone part of a clean split is cutting at the right line: a top-level
// decl's doc comment sits ABOVE its func/type keyword, so a naive cut at the keyword
// orphans the comment. Plan computes, for every top-level declaration, the
// DOC-COMMENT-AWARE block boundaries (the exact block_start..block_end extract range)
// and flags the four things that make code motion NOT a no-op:
//
//   - per-file build tags (//go:build) — moving a decl changes which file carries them;
//   - func init() — init order is filename-ALPHABETICAL across a package, so moving an
//     init() between files can silently reorder initialization;
//   - aliased imports (x "path") — goimports -w re-derives plain imports after a move
//     but does NOT re-infer a local alias, so an alias must be re-added by hand;
//   - multi-line backtick raw strings — this is a LINE parser, not a Go tokenizer;
//     embedded column-0 func/type text inside a raw string is correctly SKIPPED, but
//     the raw_strings count warns you to eyeball the plan for such a file.
//
// It is line-based and best-effort: decl detection ignores raw-string interiors and
// the package/import header, and func length uses brace-depth over
// string/comment-stripped code. The whole fold is pure (Plan drives off an in-memory
// string), so it unit-tests without a repo. Read-only — it never edits, moves, or
// commits; it only plans.
package godsplitplan

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// FileHardMax: a Go file longer than this is a god-file (scorecard architecture debt).
// FuncHardMax: a function/method longer than this is a god-function.
const (
	FileHardMax = 1500
	FuncHardMax = 200
)

var (
	declRE         = regexp.MustCompile(`^(func|type|var|const)\b`)
	packageRE      = regexp.MustCompile(`^package\s+(\w+)`)
	methodRE       = regexp.MustCompile(`^func\s+\([^)]*\)\s+(\w+)`)
	funcRE         = regexp.MustCompile(`^func\s+(\w+)`)
	typeVarConstRE = regexp.MustCompile(`^(?:type|var|const)\s+(\w+)`)
	buildTagRE     = regexp.MustCompile(`^//\s*(?:go:build|\+build)\b`)
	initRE         = regexp.MustCompile(`^func\s+init\s*\(`)
	aliasRE        = regexp.MustCompile(`^([A-Za-z_]\w*)\s+"([^"]+)"`)
)

// Decl is one top-level declaration with its doc-comment-aware extract range. The
// func-only fields (FuncEnd/FuncLines/GodFunction) are pointers so they marshal to
// JSON only for func/method decls — matching the Python payload, which omits them
// entirely for type/var/const while still emitting god_function:false for a func.
type Decl struct {
	Line        int    `json:"line"`
	Kind        string `json:"kind"`
	Name        string `json:"name"`
	BlockStart  int    `json:"block_start"`
	FuncEnd     *int   `json:"func_end,omitempty"`
	FuncLines   *int   `json:"func_lines,omitempty"`
	GodFunction *bool  `json:"god_function,omitempty"`
	BlockEnd    int    `json:"block_end"`
}

// Hazards are the four signals that make code motion NOT a no-op.
type Hazards struct {
	BuildTags     []string `json:"build_tags"`
	InitFuncs     int      `json:"init_funcs"`
	ImportAliases []string `json:"import_aliases"`
	RawStrings    int      `json:"raw_strings"`
}

// Plan is the full split plan for one Go file.
type Plan struct {
	Package   string  `json:"package"`
	LineCount int     `json:"line_count"`
	GodFile   bool    `json:"god_file"`
	Hazards   Hazards `json:"hazards"`
	Decls     []Decl  `json:"decls"`
}

// DeclName returns (kind, name) for a top-level declaration line. A method
// `func (r R) Name(` reports kind "method" and name "Name"; a grouped `var (` /
// `const (` block reports name "(group)".
func DeclName(line string) (string, string) {
	if strings.HasPrefix(line, "func ") {
		if m := methodRE.FindStringSubmatch(line); m != nil {
			return "method", m[1]
		}
		if m := funcRE.FindStringSubmatch(line); m != nil {
			return "func", m[1]
		}
		return "func", "?"
	}
	if m := typeVarConstRE.FindStringSubmatch(line); m != nil {
		return firstToken(line), m[1]
	}
	return firstToken(line), "(group)"
}

// firstToken is Python's line.split(None, 1)[0] — the first whitespace-delimited
// token, or "?" for a blank line.
func firstToken(line string) string {
	f := strings.Fields(line)
	if len(f) == 0 {
		return "?"
	}
	return f[0]
}

// scanLine returns (code, inRawAfter) where code is line with line-comments and the
// contents of "…" / '…' / backtick raw strings removed (braces preserved only when
// they are real code). inRaw carries a multi-line backtick string across lines.
// Best-effort: /* … */ block comments are not handled (rare inside a decl). Iterates
// runes so a multi-byte escape can never desync the index (Python code-point parity).
func scanLine(line string, inRaw bool) (string, bool) {
	rs := []rune(line)
	n := len(rs)
	var out []rune
	i := 0
	for i < n {
		c := rs[i]
		if inRaw {
			if c == '`' {
				inRaw = false
			}
			i++
			continue
		}
		if c == '`' {
			inRaw = true
			i++
			continue
		}
		if c == '/' && i+1 < n && rs[i+1] == '/' {
			break // line comment: rest is not code
		}
		if c == '"' || c == '\'' {
			quote := c
			i++
			for i < n {
				if rs[i] == '\\' {
					i += 2
					continue
				}
				if rs[i] == quote {
					i++
					break
				}
				i++
			}
			continue
		}
		out = append(out, c)
		i++
	}
	return string(out), inRaw
}

// headerEnd returns the 1-based line below which top-level decls legitimately begin —
// the max of the package line, the closing `)` of an import ( … ) block, and any
// single-line import "…". Clamps a decl's block_start so an extract can never swallow
// the package clause (which would duplicate it in the new file → build break).
func headerEnd(lines []string) int {
	pkg, impEnd := 0, 0
	inImport := false
	for i, line := range lines {
		s := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(s, "package "):
			pkg = i + 1
		case s == "import (":
			inImport = true
		case inImport && s == ")":
			impEnd = i + 1
			inImport = false
		case !inImport && strings.HasPrefix(s, "import ") && strings.Contains(s, `"`):
			if i+1 > impEnd {
				impEnd = i + 1
			}
		}
	}
	if pkg > impEnd {
		return pkg
	}
	return impEnd
}

// docStart returns the 1-based line where declIdx's block begins, INCLUDING the
// leading doc comment — the line after the last REAL (not raw-string-interior) blank
// line above the decl, or line 1 if none. declIdx is 0-based into lines.
func docStart(lines []string, declIdx int, inRawBefore []bool) int {
	lastBlank := -1
	for i := 0; i < declIdx; i++ {
		if strings.TrimSpace(lines[i]) == "" && !inRawBefore[i] {
			lastBlank = i
		}
	}
	return lastBlank + 2
}

// funcSpan returns the 1-based END line of the func/method starting at declIdx
// (0-based), by brace-depth over the string/comment-stripped code: the function ends
// when depth returns to 0 after the first `{`. Robust to one-liners, indented
// composite literals, and braces inside strings (which were stripped).
func funcSpan(stripped []string, declIdx int) int {
	depth := 0
	opened := false
	for j := declIdx; j < len(stripped); j++ {
		for _, ch := range stripped[j] {
			if ch == '{' {
				depth++
				opened = true
			} else if ch == '}' {
				depth--
			}
		}
		if opened && depth <= 0 {
			return j + 1
		}
	}
	return len(stripped)
}

// ImportAliases returns the aliased imports (`alias "path"` where alias is a real
// local name, not `_`/`.` and not just the path's basename) — the ones goimports will
// NOT re-create after a move.
func ImportAliases(lines []string) []string {
	aliases := []string{}
	inBlock := false
	for _, line := range lines {
		s := strings.TrimSpace(line)
		if s == "import (" {
			inBlock = true
			continue
		}
		if inBlock && s == ")" {
			inBlock = false
			continue
		}
		target := s
		if !inBlock {
			if strings.HasPrefix(s, "import ") && strings.Contains(s, `"`) {
				target = strings.TrimSpace(s[len("import "):])
			} else {
				continue
			}
		}
		m := aliasRE.FindStringSubmatch(target)
		if m == nil {
			continue
		}
		alias, path := m[1], m[2]
		if alias == "_" || alias == "." {
			continue
		}
		base := path
		if idx := strings.LastIndex(path, "/"); idx >= 0 {
			base = path[idx+1:]
		}
		if alias != base {
			aliases = append(aliases, fmt.Sprintf(`%s "%s"`, alias, path))
		}
	}
	return aliases
}

// Compute folds Go source text into the split plan — package, hazards, and the decl
// list with doc-comment-aware block ranges + size flags. Decl/build-tag/init
// detection ignores raw-string interiors; block_start is clamped below the header. No
// file I/O, so a test drives it with a string.
func Compute(text string) Plan {
	lines := strings.Split(text, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1] // a trailing newline yields one phantom empty element
	}
	n := len(lines)

	// Pass 1: strip strings/comments per line, tracking multi-line raw strings.
	stripped := make([]string, 0, n)
	inRawBefore := make([]bool, 0, n)
	rawStrings := 0
	inRaw := false
	for _, line := range lines {
		inRawBefore = append(inRawBefore, inRaw)
		code, after := scanLine(line, inRaw)
		if !inRaw && after { // a multi-line raw string just opened
			rawStrings++
		}
		stripped = append(stripped, code)
		inRaw = after
	}

	headerEndVal := headerEnd(lines)

	pkg := ""
	buildTags := []string{}
	initFuncs := 0
	decls := []Decl{}

	// Pass 2: detect decls/build-tags/init only on REAL code lines (not raw interiors).
	for i, line := range lines {
		if inRawBefore[i] {
			continue
		}
		if pkg == "" {
			if m := packageRE.FindStringSubmatch(line); m != nil {
				pkg = m[1]
			}
		}
		if buildTagRE.MatchString(line) {
			buildTags = append(buildTags, strings.TrimSpace(line))
		}
		if initRE.MatchString(line) {
			initFuncs++
		}
		if declRE.MatchString(line) {
			kind, name := DeclName(line)
			start := docStart(lines, i, inRawBefore)
			if headerEndVal+1 > start {
				start = headerEndVal + 1
			}
			d := Decl{Line: i + 1, Kind: kind, Name: name, BlockStart: start}
			if kind == "func" || kind == "method" {
				end := funcSpan(stripped, i)
				fl := end - (i + 1) + 1
				god := fl > FuncHardMax
				d.FuncEnd = &end
				d.FuncLines = &fl
				d.GodFunction = &god
			}
			decls = append(decls, d)
		}
	}

	for idx := range decls {
		if idx+1 < len(decls) {
			decls[idx].BlockEnd = decls[idx+1].BlockStart - 1
		} else {
			decls[idx].BlockEnd = n
		}
	}

	return Plan{
		Package:   pkg,
		LineCount: n,
		GodFile:   n > FileHardMax,
		Hazards: Hazards{
			BuildTags:     buildTags,
			InitFuncs:     initFuncs,
			ImportAliases: ImportAliases(lines),
			RawStrings:    rawStrings,
		},
		Decls: decls,
	}
}

// Render is the human table (the non-JSON output). It mirrors the Python layout; the
// machine contract is the JSON, so exact byte-parity of this table is not required.
func Render(path string, p Plan) string {
	var out []string
	flag := ""
	if p.GodFile {
		flag = "  GOD-FILE (>1500)"
	}
	out = append(out, fmt.Sprintf("%s: package %s  (%d lines)%s", path, p.Package, p.LineCount, flag))
	hz := p.Hazards
	out = append(out, "hazards (these make code motion NOT a no-op):")
	out = append(out, "  build_tags     : "+orNone(hz.BuildTags))
	initNote := ""
	if hz.InitFuncs > 0 {
		initNote = " (init order is filename-alpha — moving one between files can reorder)"
	}
	out = append(out, fmt.Sprintf("  init_funcs     : %d%s", hz.InitFuncs, initNote))
	aliasNote := ""
	if len(hz.ImportAliases) > 0 {
		aliasNote = " (goimports will NOT re-add these after a move — copy them by hand)"
	}
	out = append(out, "  import_aliases : "+orNone(hz.ImportAliases)+aliasNote)
	rawNote := ""
	if hz.RawStrings > 0 {
		rawNote = " (line parser may misread embedded code — eyeball the ranges)"
	}
	out = append(out, fmt.Sprintf("  raw_strings    : %d%s", hz.RawStrings, rawNote))
	out = append(out, "")
	out = append(out, "decls (block_start..block_end is the doc-comment-aware sed extract range):")
	for _, d := range p.Decls {
		span := ""
		if d.Kind == "func" || d.Kind == "method" {
			mark := ""
			if d.GodFunction != nil && *d.GodFunction {
				mark = "  <<< GOD-FUNCTION (>200)"
			}
			fl := 0
			if d.FuncLines != nil {
				fl = *d.FuncLines
			}
			span = fmt.Sprintf("  [%dL]%s", fl, mark)
		}
		warn := ""
		if d.BlockEnd < d.BlockStart {
			warn = "  !! INVERTED RANGE — do not trust this plan"
		}
		out = append(out, fmt.Sprintf("  %5d..%-5d %6s %s%s%s", d.BlockStart, d.BlockEnd, d.Kind, d.Name, span, warn))
	}
	return strings.Join(out, "\n")
}

func orNone(xs []string) string {
	if len(xs) == 0 {
		return "none"
	}
	return strings.Join(xs, ", ")
}

// Run is the CLI entry point: `godsplit-plan <file.go> [--json]`. Exit 0 on success,
// 1 on a usage error or an unreadable file — mirroring the Python main.
func Run(stdout, stderr io.Writer, argv []string) int {
	asJSON := false
	file := ""
	for _, a := range argv {
		switch {
		case a == "--json":
			asJSON = true
		case strings.HasPrefix(a, "-"):
			// ignore unknown flags (lenient vs argparse)
		default:
			if file == "" {
				file = a
			}
		}
	}
	if file == "" {
		fmt.Fprintln(stderr, "godsplit-plan: usage: godsplit-plan <file.go> [--json]")
		return 1
	}
	data, err := os.ReadFile(file)
	if err != nil {
		fmt.Fprintf(stderr, "godsplit-plan: %v\n", err)
		return 1
	}
	p := Compute(string(data))
	if asJSON {
		b, _ := json.MarshalIndent(p, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintln(stdout, Render(file, p))
	}
	return 0
}
