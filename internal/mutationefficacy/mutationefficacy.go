// Package mutationefficacy is a bounded, SOFT mutation-testing probe for the qa-process
// scorecard (#3845): it asks the one question coverage and assertion-strength cannot --
// would the suite actually FAIL if the code were wrong? It applies a tiny set of standard
// operator mutants (flip a comparator, off-by-one a bound, swap +/-) to an allow-list of
// packages, runs each package's tests against the mutated source, and counts SURVIVORS:
// mutants the suite did not catch.
//
// Three properties keep it honest and cheap:
//
//   - SOFT. A survivor is an advisory nudge on the KPI's Soft list, NEVER a HARD defect that
//     gates the card. Mutation is expensive and noisy, so it must not be able to red a gate
//     (the same anti-gaming rule pkg/scorecard states for every Soft signal).
//   - BOUNDED. The probe runs only over a caller-supplied allow-list (never the whole tree)
//     and stops at a per-package mutant cap, so it never dominates a run.
//   - RESTORE-ALWAYS. A mutant is written to disk, tested, and the original bytes are ALWAYS
//     restored -- via defer, so even a panicking test runner cannot leave mutated source on
//     disk. At most one file differs from HEAD at any instant, and only for one test run.
//
// A mutant that fails to COMPILE or fails the tests is "killed" (caught); only a mutant that
// both compiles and leaves every test passing is a survivor. That makes the survivor count
// conservative -- a crude mutant that breaks the build is never miscounted as an efficacy gap.
//
// The pure halves (GenerateMutants, Fold) hold no I/O and are fixture-testable directly; the
// impure runner (ProbePackage + GoTestRunner) is the only part that touches disk or shells to
// the toolchain, and it takes an injected TestRunner so the CLI and tests share one seam.
package mutationefficacy

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// KPIKey is both the KPI Key and the qa-process dogfood finding name (#3845). The qa-process
// card folds this KPI in under `fak score qa-process` when an allow-list is supplied.
const KPIKey = "mutation_efficacy"

// opSwap is the closed mutant menu: each entry rewrites one binary operator to a standard
// single-token mutation of it. The three families the issue names all live here --
// comparator flip (== <-> !=), off-by-one a bound (strictness flip < <-> <=, > <-> >=), and
// arithmetic swap (+ <-> -). Every mutant changes exactly one operator, so it is a minimal,
// localized change of behavior.
var opSwap = map[token.Token]token.Token{
	token.LSS: token.LEQ, // <  -> <=   off-by-one a bound
	token.LEQ: token.LSS, // <= -> <
	token.GTR: token.GEQ, // >  -> >=
	token.GEQ: token.GTR, // >= -> >
	token.EQL: token.NEQ, // == -> !=   flip a comparator
	token.NEQ: token.EQL, // != -> ==
	token.ADD: token.SUB, // +  -> -    arithmetic swap
	token.SUB: token.ADD, // -  -> +
}

// Mutant is one applied source mutation: the file it touches, the 1-based line of the swapped
// operator, a human label ("< -> <="), and the FULL rewritten file source with exactly that one
// operator swapped. The rewrite is a byte splice at the operator's offset, so every other byte
// of the file -- comments, build tags, formatting -- is preserved verbatim.
type Mutant struct {
	File    string
	Line    int
	Op      string
	Mutated string
}

// GenerateMutants parses one Go source file and returns up to cap operator-swap mutants, one
// per mutable binary operator in source order (cap <= 0 means "no cap"). filename is used only
// for positions and the returned File field. It is pure: no I/O, no toolchain. A parse error
// yields no mutants (a file the probe cannot parse is simply not mutated), never a panic.
func GenerateMutants(filename, src string, cap int) []Mutant {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, src, parser.SkipObjectResolution)
	if err != nil {
		return nil
	}
	var mutants []Mutant
	ast.Inspect(file, func(n ast.Node) bool {
		if cap > 0 && len(mutants) >= cap {
			return false
		}
		be, ok := n.(*ast.BinaryExpr)
		if !ok {
			return true
		}
		to, mutable := opSwap[be.Op]
		if !mutable {
			return true
		}
		pos := fset.Position(be.OpPos)
		off := pos.Offset
		orig := be.Op.String()
		// Defensive: only splice when the byte at the operator's offset really is the operator
		// text. A mismatch (a position the printer and the source disagree on) is skipped, never
		// mis-spliced into a corrupt file.
		if off < 0 || off+len(orig) > len(src) || src[off:off+len(orig)] != orig {
			return true
		}
		mutants = append(mutants, Mutant{
			File:    filename,
			Line:    pos.Line,
			Op:      orig + " -> " + to.String(),
			Mutated: src[:off] + to.String() + src[off+len(orig):],
		})
		return true
	})
	return mutants
}

// PackageResult is the folded outcome of probing one package: how many mutants were applied,
// how many survived (the suite stayed green), a bounded human-readable survivor list, and a
// non-empty Err if the package could not be probed at all (no source, unreadable dir).
type PackageResult struct {
	Pkg       string
	Applied   int
	Survived  int
	Survivors []string
	Err       string
}

// TestRunner runs the package's tests against whatever source is currently on disk and reports
// whether they PASSED. It is injected so the pure mutate/restore orchestration can be exercised
// without a toolchain, and so the CLI and the end-to-end test share one definition of "killed".
// run(dir) == true means the suite passed -> the mutant was NOT caught -> it survived.
type TestRunner func(pkgDir string) bool

// ProbePackage applies up to capPerPkg operator mutants across the non-test .go files of pkgDir
// (capPerPkg <= 0 means "no cap"), runs the injected TestRunner after each, and counts survivors.
// It NEVER leaves mutated source on disk: each file's original bytes are captured up front and
// restored by applyAndRun's deferred restore, which runs even if the runner panics.
func ProbePackage(pkgDir string, run TestRunner, capPerPkg int) PackageResult {
	res := PackageResult{Pkg: pkgDir}
	goFiles, err := packageGoFiles(pkgDir)
	if err != nil {
		res.Err = err.Error()
		return res
	}
	if len(goFiles) == 0 {
		res.Err = "no non-test .go files to mutate"
		return res
	}
	for _, gf := range goFiles {
		if capPerPkg > 0 && res.Applied >= capPerPkg {
			break
		}
		orig, err := os.ReadFile(gf)
		if err != nil {
			continue // a file we cannot read is not mutated -- skip, never a false survivor
		}
		remaining := 0
		if capPerPkg > 0 {
			remaining = capPerPkg - res.Applied
		}
		for _, m := range GenerateMutants(filepath.Base(gf), string(orig), remaining) {
			if capPerPkg > 0 && res.Applied >= capPerPkg {
				break
			}
			res.Applied++
			if applyAndRun(gf, orig, []byte(m.Mutated), pkgDir, run) {
				res.Survived++
				res.Survivors = append(res.Survivors,
					fmt.Sprintf("%s:%d %s", filepath.Base(gf), m.Line, m.Op))
			}
		}
	}
	return res
}

// applyAndRun writes mutated over path, runs the suite, and ALWAYS restores orig -- even if run
// panics (defer). It returns whether the suite passed (== the mutant survived). A write failure
// or a panic is treated as a KILL, never a false survivor.
func applyAndRun(path string, orig, mutated []byte, pkgDir string, run TestRunner) (survived bool) {
	if err := os.WriteFile(path, mutated, 0o644); err != nil {
		return false
	}
	defer func() {
		_ = os.WriteFile(path, orig, 0o644) // restore ALWAYS
		if r := recover(); r != nil {
			survived = false
		}
	}()
	return run(pkgDir)
}

// packageGoFiles lists the non-test .go files directly in dir, sorted for determinism. It does
// not recurse -- a Go package is exactly one directory.
func packageGoFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
			files = append(files, filepath.Join(dir, name))
		}
	}
	sort.Strings(files)
	return files, nil
}

// Fold turns per-package probe results into the mutation_efficacy KPI. Survivors are SOFT
// (Soft list, never Defects), so the KPI can NEVER add HARD debt or gate the card -- the
// deliberate SOFT contract of #3845. Score is the kill rate (100 * killed / applied); with no
// mutant applied anywhere the probe is INSUFFICIENT and scores a clean 100 rather than a hollow
// 0 (the absence of a survivor is not a survivor -- the same anti-fail-open discipline the
// qa-process epic turns on, #3833).
func Fold(results []PackageResult) scorecard.KPI {
	applied, survived := 0, 0
	var soft []string
	for _, r := range results {
		applied += r.Applied
		survived += r.Survived
		for _, s := range r.Survivors {
			soft = append(soft, fmt.Sprintf("MUTATION_SURVIVOR %s %s: the suite stayed green under a wrong change -- add an assertion that fails on it", r.Pkg, s))
		}
		if r.Err != "" {
			soft = append(soft, fmt.Sprintf("MUTATION_UNPROBED %s: %s", r.Pkg, r.Err))
		}
	}
	killed := applied - survived
	score := 100.0
	detail := "no mutants applied over the allow-list (INSUFFICIENT) -- mutation_efficacy defaults to 100"
	if applied > 0 {
		score = scorecard.Round1(100 * float64(killed) / float64(applied))
		detail = fmt.Sprintf("%d/%d mutants killed; %d survived (SOFT) across %d package(s)",
			killed, applied, survived, len(results))
	}
	return scorecard.KPI{
		Key:    KPIKey,
		Group:  "mutation",
		Score:  score,
		Detail: detail,
		Soft:   soft,
	}
}

// GoTestRunner returns a TestRunner that runs `go test -count=1 .` in the package dir with a
// bounded per-mutant timeout (the time cap), reporting a PASS as survival. Any non-zero exit --
// a compile failure, a test failure, or a timeout -- is a KILL, the conservative default that
// keeps a crude or slow mutant from being miscounted as an efficacy gap. Subprocess output is
// discarded; only the exit status matters.
func GoTestRunner(timeout time.Duration) TestRunner {
	return func(pkgDir string) bool {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		cmd := windowgate.CommandContext(ctx, "go", "test", "-count=1", ".")
		cmd.Dir = pkgDir
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Run() == nil
	}
}
