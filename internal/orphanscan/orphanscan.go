// Package orphanscan is a small, syntactic detector for the "built but never wired
// up" smell: an unexported top-level function that is defined but referenced nowhere
// in its own package. That is the shape a piece of work takes when it was authored and
// then dropped on the floor — a panel builder that was never added to the registry
// slice, a handler written but never routed, a helper a refactor orphaned when it
// deleted the only call site. The code compiles (Go does not flag an unused func the
// way it flags an unused import or local), the build stays green, and the work is
// silently inert. This package makes that gap answerable as a named finding, in the
// same spirit as internal/devindex's freshness drift and internal/boundarylint's
// boundary tells: a green build is a self-report, and an orphaned function is one place
// a green build hides work that isn't actually connected to anything.
//
// PRECISION OVER RECALL. The scan is deliberately conservative — it would rather miss a
// real orphan than cry wolf on live code — because a noisy detector is one nobody runs.
// It only ever flags an UNEXPORTED, top-level, receiver-less func (an exported symbol
// may be part of the package's API and used from anywhere; a method may satisfy an
// interface and be dispatched dynamically). A candidate is reported only when its name
// appears NOWHERE else in the package's syntax — across every .go file including tests,
// so a test-only helper is never flagged. References are counted syntactically by
// identifier name, which means a candidate whose name collides with a field, a method,
// a local variable, or a recursive self-call anywhere in the package is treated as used
// (a false negative we accept to keep false positives at zero). Special forms that are
// "referenced" invisibly are excluded up front: main, init, the blank func _, anything
// carrying a //go:linkname or //export directive, anything in a generated file, and
// anything a maintainer marks with the //orphanscan:keep escape hatch.
//
// It is a pure, tier-1 syntactic pass: go/parser per file, no type information, no
// build, no network. It parses each file independently, so a package that does not
// fully compile (a common state in a shared tree with in-flight siblings) is still
// scanned — a file that fails to parse is skipped and recorded, never fatal.
//
// # Known limits
//
// These are the boundaries of the syntactic pass, each one FIXTURED in
// precision_test.go (TestPrecisionRecallClasses) rather than merely asserted here —
// #3169. An accepted FALSE POSITIVE flags something that is in fact reachable; an
// accepted FALSE NEGATIVE stays quiet about something that is in fact dead. Every entry
// leans the same way on purpose: a miss costs one un-found orphan, a cry-wolf costs the
// detector its readers.
//
//   - ACCEPTED FALSE POSITIVE — string-keyed / reflection dispatch. A func whose name
//     occurs only inside a string literal (a generated dispatch table, a name-keyed
//     registry resolved by reflection) has no identifier for the pass to count, so it is
//     reported. The remedy is the //orphanscan:keep hatch on the func, which is visible
//     at the definition; inferring wiring from string literals would cost real positives.
//
//   - ACCEPTED FALSE NEGATIVE — build-tag interactions. go/parser never evaluates
//     //go:build, so a definition and its only use under DISJOINT constraints still pool
//     into one reference set and the func is not flagged, even though it is dead on
//     every individual platform. The same mechanism means a helper used only from a
//     //go:build ignore standalone tool in the package dir counts as wired. Evaluating
//     constraints would need a per-platform scan and could only ADD false positives.
//
//   - NOT A LIMIT (pinned so it stays that way) — free funcs satisfying an interface. A
//     receiver-less func adapted through a func-typed adapter (the http.HandlerFunc
//     shape) or assigned to a func field is referenced BY IDENTIFIER at the conversion
//     or assignment, so the pass sees it. The worry only applies across a package
//     boundary, and an unexported name cannot cross one.
//
//   - NOT A LIMIT (pinned so it stays that way) — references from generated code. A
//     generated file is skipped as a source of CANDIDATES but its identifiers are
//     counted as REFERENCES first, so same-package generated wiring is visible. The scan
//     is package-local and cannot see another package's generated code, but only an
//     EXPORTED name is reachable from there and exported names are never candidates —
//     so there is no cross-package reference left to miss.
//
// Whole-tree coverage is a separate question from these limits; see
// docs/notes/ORPHAN-FUNC-SCAN-2026-07-07.md.
package orphanscan

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// Orphan is one unexported top-level function that nothing in its package references.
type Orphan struct {
	Name string // the function's identifier, e.g. "guardAblationPanel"
	File string // slash-separated path as passed to ScanDir, e.g. "cmd/fak/info_panels.go"
	Line int    // 1-based line of the func keyword
}

// String renders the orphan as "file:line: func name is defined but never referenced
// in its package (ORPHAN_FUNC)" — the closed-vocabulary shape a gate or scorecard can
// grep, matching the internal/boundarylint finding grammar.
func (o Orphan) String() string {
	return fmt.Sprintf("%s:%d: func %s is defined but never referenced in its package (ORPHAN_FUNC)", o.File, o.Line, o.Name)
}

// Report is a scan of one package: what was flagged, and what the scan could not read.
// Unparsed is the honest caveat a GATE needs and a human report does not. References
// are counted per file, so a file that will not parse contributes none — and if that
// file held a func's only call site, the func looks unreferenced when it is not. On a
// shared tree with in-flight siblings that is not hypothetical, so a gate asserting on
// this package must be able to see that the input was incomplete and decline to judge
// rather than red the trunk on a mid-save neighbour.
type Report struct {
	Orphans  []Orphan // the findings, sorted by name then file
	Unparsed []string // files skipped because they could not be read or parsed (relPrefix applied)
}

// ScanDir returns just the findings for dir. It is the original, narrow seam kept for
// callers that only report to a human (internal/antipattern folds it into ORPHAN_FUNC
// debt); anything gating on the result wants ScanDirReport, which also says whether the
// package was fully readable.
func ScanDir(dir, relPrefix string) ([]Orphan, error) {
	rep, err := ScanDirReport(dir, relPrefix)
	return rep.Orphans, err
}

// ScanDirReport parses every .go file directly under dir (non-recursive — a Go package is
// a single directory) and returns the unexported, receiver-less top-level functions that
// no file in the package references, sorted by name, plus the files it could not parse.
// relPrefix is prepended to each reported path so a caller can report repo-relative paths
// (pass "" for bare basenames). A dir with no readable Go files yields no findings and no
// error; a single file that fails to parse is skipped and RECORDED, not fatal —
// orphanscan is a syntactic best-effort pass, not a compiler.
func ScanDirReport(dir, relPrefix string) (Report, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return Report{}, err
	}
	fset := token.NewFileSet()

	// candidate: an unexported top-level func we might report, keyed by name.
	type candidate struct {
		file string
		line int
	}
	candidates := map[string]candidate{}
	// refs counts identifier occurrences by name across the whole package, INCLUDING
	// each candidate's own declaration ident (subtracted below). A name with a count
	// above its self-declaration is referenced and therefore not an orphan.
	refs := map[string]int{}
	var unparsed []string

	// relOf renders one entry the way findings report it, so a skipped file and a
	// flagged func name the same path shape.
	relOf := func(name string) string {
		if relPrefix == "" {
			return name
		}
		return relPrefix + "/" + name
	}

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		src, readErr := os.ReadFile(path)
		if readErr != nil {
			// An unreadable file contributes no references, so it is recorded for the
			// same reason an unparseable one is: it may have held the only call site.
			unparsed = append(unparsed, relOf(e.Name()))
			continue
		}
		f, parseErr := parser.ParseFile(fset, path, src, parser.ParseComments)
		if parseErr != nil || f == nil {
			unparsed = append(unparsed, relOf(e.Name())) // skipped, recorded, not fatal
			continue
		}
		generated := isGenerated(f)

		// Every identifier anywhere in the file is a potential reference. Counting the
		// whole file (declarations included) is intentional: the candidate loop below
		// subtracts each candidate's single self-declaration, so only genuine uses
		// remain.
		ast.Inspect(f, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				refs[id.Name]++
			}
			return true
		})

		if generated {
			continue // a generated file may DEFINE a func but never orphan-owns it
		}
		relFile := relOf(e.Name())
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil { // only top-level, receiver-less funcs
				continue
			}
			name := fn.Name.Name
			if !isUnexported(name) || name == "main" || name == "init" || name == "_" {
				continue
			}
			if hasKeepDirective(fn) || hasLinkDirective(f, name) {
				continue
			}
			// Last definition wins if two files somehow share a name (they cannot in a
			// valid package, but the scan tolerates an in-flight duplicate).
			candidates[name] = candidate{file: relFile, line: fset.Position(fn.Pos()).Line}
		}
	}

	var out []Orphan
	for name, c := range candidates {
		// refs[name] includes the one self-declaration ident. Anything above that is a
		// real reference somewhere in the package.
		if refs[name] <= 1 {
			out = append(out, Orphan{Name: name, File: c.file, Line: c.line})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].File < out[j].File
	})
	sort.Strings(unparsed)
	return Report{Orphans: out, Unparsed: unparsed}, nil
}

// isUnexported reports whether name begins with a lowercase letter (a Go unexported
// identifier). A name starting with '_' or a digit is not a normal func name and is
// treated as not-unexported so it is skipped.
func isUnexported(name string) bool {
	if name == "" {
		return false
	}
	r := []rune(name)[0]
	return unicode.IsLower(r)
}

// hasKeepDirective reports whether the func carries the //orphanscan:keep escape hatch
// in its doc comment — the visible, greppable way a maintainer says "this func is wired
// in a way the syntactic scan cannot see" (reflection, a go:generate table, an external
// linkname), matching the //boundarylint:ignore convention.
func hasKeepDirective(fn *ast.FuncDecl) bool {
	if fn.Doc == nil {
		return false
	}
	for _, c := range fn.Doc.List {
		if strings.Contains(c.Text, "orphanscan:keep") {
			return true
		}
	}
	return false
}

// hasLinkDirective reports whether any comment in the file wires name via a
// //go:linkname or //export directive — both make a func reachable from outside the
// package with no in-package reference, so a matching candidate must not be flagged.
func hasLinkDirective(f *ast.File, name string) bool {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			t := c.Text
			if (strings.Contains(t, "go:linkname") || strings.Contains(t, "//export")) && containsWord(t, name) {
				return true
			}
		}
	}
	return false
}

// containsWord reports whether name appears in text delimited by non-identifier
// characters, so "//export Foo" matches name "Foo" but "Foobar" does not.
func containsWord(text, name string) bool {
	idx := 0
	for {
		i := strings.Index(text[idx:], name)
		if i < 0 {
			return false
		}
		i += idx
		before := i == 0 || !isIdentRune(rune(text[i-1]))
		afterPos := i + len(name)
		after := afterPos >= len(text) || !isIdentRune(rune(text[afterPos]))
		if before && after {
			return true
		}
		idx = i + len(name)
	}
}

func isIdentRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

// isGenerated reports whether f carries the standard `// Code generated ... DO NOT
// EDIT.` marker on a line by itself before the package clause, per the go generate
// convention. A generated file's funcs are owned by their generator, not by this gate.
func isGenerated(f *ast.File) bool {
	for _, cg := range f.Comments {
		if cg.Pos() >= f.Package {
			break // comments are position-ordered; past the package clause, stop
		}
		for _, c := range cg.List {
			line := strings.TrimSpace(strings.TrimPrefix(c.Text, "//"))
			if strings.HasPrefix(line, "Code generated ") && strings.HasSuffix(line, " DO NOT EDIT.") {
				return true
			}
		}
	}
	return false
}
