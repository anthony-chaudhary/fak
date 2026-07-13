// Package unwiredscore is the UNWIRED-CODE scorecard -- the recurring detector for the
// failure class the operator named "code complete but not wired into the default path".
//
// A feature is "code complete" long before it delivers value: the Go package compiles,
// carries its own tests, reads clean -- and then nothing ever imports it. It ships in no
// binary, runs on no default path, is exercised by no other package. The 10x-100x of work
// that turns code-complete into useful (wire it to a verb, dogfood it, benchmark it, put it
// on a cadence) silently never happens, and nobody notices because a dead package breaks no
// build. This card makes "a code-complete package wired to nothing" a tracked, unbounded
// unwired_debt integer, and its dispatch half (dispatch.go) fans out one deduped GitHub issue
// per orphan so the backlog is discovered instead of forgotten.
//
// Like internal/defaultvaluescore and internal/propagationscore it is a TREE-READING scorecard
// (no data dir): the import graph of the real Go source IS the data, parsed with go/parser, so
// the score cannot be gamed by editing a JSON file -- only by wiring the package into a default
// path, or retiring it. It sits in the same niche as internal/architest (which pins the layered
// DAG and registration completeness) but catches the invariant architest does NOT: a package
// that declares real API and passes every layering rule, yet is imported by nothing at all.
//
// THE SIGNAL. A candidate is any internal/<pkg> directory (at any depth) that has a non-test
// .go file declaring at least one top-level func/type/const/var -- real, callable API, not a
// doc-only package (an architest-style test harness whose only non-test file is doc.go declares
// nothing and is correctly skipped). A candidate is WIRED iff its import path appears in an
// import statement of ANY .go file anywhere in the module (cmd/internal/pkg/experiments), test
// files included -- a package reached only by an external _test.go is still exercised. A
// candidate that no file imports is UNWIRED: code complete, providing zero durable value.
//
// Imports are read build-tag-blind (every file is parsed regardless of GOOS/GOARCH), so a
// package pulled onto a path only under some build tag is counted WIRED -- the safe direction:
// this card never falsely calls a package dead, it can only miss a dead-subtree (a package
// imported solely by another dead package), which the next run catches once the parent is
// retired. Files under a testdata/ segment are ignored, matching the go toolchain.
package unwiredscore

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id any consumer keys on.
const Schema = "fak-unwired-scorecard/1"

// DebtKey is the headline integer the control-pane folds (corpus.unwired_debt).
const DebtKey = "unwired_debt"

// ModulePrefix is the import-path prefix for every in-module package.
const ModulePrefix = "github.com/anthony-chaudhary/fak/"

// scanRoots are the module subtrees walked for both candidate packages and import edges.
// tools/ is Python (no Go), .dos/.git are not code -- neither is walked.
var scanRoots = []string{"cmd", "internal", "pkg", "experiments"}

// AllowUnwired is the allow-list of code-complete packages that legitimately have no importer,
// each paired with the documented reason -- the same review chokepoint architest's regOffList
// and defaultvaluescore's offWithReason apply. An unwired package NOT on this list is honest
// debt. Keyed by the repo-relative package dir (e.g. "internal/foo"). Seeded deliberately small:
// the point of this card is to SURFACE orphans as tracked issues, not to suppress them, so a
// package earns an entry only when "imported by nothing" is a genuine, reviewed design choice
// (a public test/fixture harness whose only consumers are other packages' _test.go files).
var AllowUnwired = map[string]string{
	// internal/agenttest is the public agent-workflow TEST harness (#238, D-008): deterministic
	// fixtures + a tool-call assertion library meant to be imported by CONSUMERS' _test.go, so a
	// window with no importer is adoption-pending, not dead. Retire this entry (and file the
	// wiring issue) if it stays unadopted past a release.
	"internal/agenttest": "public agent-workflow test/assertion harness meant for external consumers' _test.go; no prod importer is by design",
}

// Pkg is one code-complete internal package with its wiring facts. Dir is repo-relative with
// forward slashes; ImportPath is the full module path a wiring import would name.
type Pkg struct {
	Dir         string `json:"dir"`
	ImportPath  string `json:"import_path"`
	HasTest     bool   `json:"has_test"`
	SourceLines int    `json:"source_lines"`
	Wired       bool   `json:"wired"`
	AllowReason string `json:"allow_reason,omitempty"`
}

// Unwired reports whether the package is code-complete-but-wired-to-nothing AND not allow-listed
// -- i.e. it counts as debt.
func (p Pkg) Unwired() bool { return !p.Wired && p.AllowReason == "" }

// Scan walks root and returns every code-complete internal package with its wiring status,
// sorted worst-first (biggest stranded investment first: most source lines, then tested-but-
// stranded, then name). It is pure over the filesystem at root (no clock, no network, no exec).
func Scan(root string) []Pkg {
	imported := map[string]bool{}     // import paths referenced by SOME .go file in the module
	hasDecl := map[string]bool{}      // internal dir -> a non-test file declares real API
	hasTest := map[string]bool{}      // internal dir -> a _test.go lives here
	srcLines := map[string]int{}      // internal dir -> non-test source line count
	dirsSeen := map[string]struct{}{} // every internal dir with a non-test .go (candidate universe)

	for _, sub := range scanRoots {
		base := filepath.Join(root, sub)
		_ = filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil // unreadable dir/file: skip, never abort the scan
			}
			if d.IsDir() {
				if isTestdataSegment(d.Name()) {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(d.Name(), ".go") {
				return nil
			}
			rel := relSlash(root, path)
			if hasTestdataSegment(rel) {
				return nil
			}
			src, rerr := os.ReadFile(path)
			if rerr != nil {
				return nil
			}
			// Collect the module-internal import edges of EVERY .go file (test files included:
			// an external _test.go importer still exercises the package).
			collectImports(imported, src)

			dir := relSlash(root, filepath.Dir(path))
			isTest := strings.HasSuffix(d.Name(), "_test.go")
			if !strings.HasPrefix(dir, "internal/") && dir != "internal" {
				return nil // only internal/ dirs are candidates; cmd/pkg/experiments count only as importers
			}
			if isTest {
				hasTest[dir] = true
				return nil
			}
			dirsSeen[dir] = struct{}{}
			srcLines[dir] += countLines(src)
			if fileDeclaresAPI(path, src) {
				hasDecl[dir] = true
			}
			return nil
		})
	}

	var out []Pkg
	for dir := range dirsSeen {
		if !hasDecl[dir] {
			continue // doc-only / declaration-free package (an architest-style test harness): skip
		}
		imp := ModulePrefix + dir
		out = append(out, Pkg{
			Dir:         dir,
			ImportPath:  imp,
			HasTest:     hasTest[dir],
			SourceLines: srcLines[dir],
			Wired:       imported[imp],
			AllowReason: AllowUnwired[dir],
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		// Worst-first: the biggest stranded investment (most source lines) is the loudest debt.
		if a.Unwired() != b.Unwired() {
			return a.Unwired() // unwired sorts before wired
		}
		if a.SourceLines != b.SourceLines {
			return a.SourceLines > b.SourceLines
		}
		if a.HasTest != b.HasTest {
			return a.HasTest // tested-but-stranded outranks untested
		}
		return a.Dir < b.Dir
	})
	return out
}

// kpiNoUnwiredPackages (HARD): every code-complete internal package is imported by something.
// Each unwired package is one defect; score is the wired fraction. This is the whole card --
// one axis, an unbounded debt count of orphaned packages.
func kpiNoUnwiredPackages(pkgs []Pkg) (scorecard.KPI, []Pkg) {
	total := len(pkgs)
	var unwired []Pkg
	var defects []string
	for _, p := range pkgs {
		if !p.Unwired() {
			continue
		}
		unwired = append(unwired, p)
		tested := "no"
		if p.HasTest {
			tested = "yes"
		}
		defects = append(defects, fmt.Sprintf(
			"%s: code-complete (%d source line(s), tests: %s) but its import path is referenced by no .go file in the module -- not wired into any default path (UNWIRED); wire it to a verb/default path or retire it",
			p.Dir, p.SourceLines, tested))
	}
	score := 100.0
	if total > 0 {
		score = 100.0 * float64(total-len(unwired)) / float64(total)
	}
	return scorecard.KPI{
		Key: "no_unwired_packages", Group: "wiring",
		Score:   score,
		Detail:  fmt.Sprintf("%d/%d code-complete internal packages are wired into a default path", total-len(unwired), total),
		Defects: defects,
	}, unwired
}

// Build reads the tree, runs the wiring KPI, and folds it into the control-pane payload via the
// shared kernel. root is the repo root.
func Build(root string) scorecard.Payload {
	pkgs := Scan(root)
	kpi, unwired := kpiNoUnwiredPackages(pkgs)

	allow := 0
	tested := 0
	for _, p := range pkgs {
		if p.AllowReason != "" {
			allow++
		}
	}
	for _, p := range unwired {
		if p.HasTest {
			tested++
		}
	}

	finding := "every code-complete internal package is wired into a default path"
	next := "hold -- re-run after a new package lands; a regression means a code-complete package shipped wired to nothing"
	if len(unwired) > 0 {
		finding = fmt.Sprintf("%s: code complete but wired into no default path", scorecard.CountNoun(len(unwired), "orphaned package"))
		next = "run `fak unwired-debt-dispatch` to fan out one tracked issue per orphan, then wire-or-retire worst-first (biggest stranded investment first)"
	}

	p := scorecard.Fold(Schema, []scorecard.KPI{kpi}, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"candidates":     len(pkgs),
			"unwired":        len(unwired),
			"unwired_tested": tested,
			"allowlisted":    allow,
		},
	})
	p.Workspace = root
	return p
}

// --- pure helpers ---------------------------------------------------------------------------

// fileDeclaresAPI reports whether a Go source file declares at least one top-level
// (non-import) func/type/const/var -- i.e. real callable API rather than a doc-only file. The
// import block is parsed but a package whose only decls are imports (or none, a bare doc.go)
// declares nothing. Parse failures are treated as "no decl" (never crash the scan).
func fileDeclaresAPI(path string, src []byte) bool {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
	if err != nil || f == nil {
		return false
	}
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			return true // any func/method is real API
		case *ast.GenDecl:
			if d.Tok != token.IMPORT {
				return true // a top-level type/const/var (not an import) is real API
			}
		}
	}
	return false
}

// collectImports parses only the import block of a file and records every module-internal
// import path into imported.
func collectImports(imported map[string]bool, src []byte) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ImportsOnly)
	if err != nil || f == nil {
		return
	}
	for _, spec := range f.Imports {
		path := strings.Trim(spec.Path.Value, `"`)
		if strings.HasPrefix(path, ModulePrefix) {
			imported[path] = true
		}
	}
}

func countLines(src []byte) int {
	n := 0
	for _, b := range src {
		if b == '\n' {
			n++
		}
	}
	return n + 1
}

func relSlash(root, path string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		rel = path
	}
	return filepath.ToSlash(rel)
}

func isTestdataSegment(name string) bool { return name == "testdata" }

func hasTestdataSegment(relPath string) bool {
	for _, seg := range strings.Split(relPath, "/") {
		if seg == "testdata" {
			return true
		}
	}
	return false
}
