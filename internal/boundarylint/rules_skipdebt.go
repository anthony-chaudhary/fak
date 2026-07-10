package boundarylint

import (
	"go/ast"
	"go/token"
)

// SkipDebt flags a test that removes itself from the suite with a bare, unconditional
// t.Skip("...") / t.Skipf(...) / t.SkipNow() — a skip NOT guarded by a platform, short-
// mode, or environment condition. A skipped-into-silence test is invisible to a presence
// KPI (the enclosing Test func still exists, so "we have tests" stays green) while the
// body never runs, which is how a suite quietly rots to all-skips. This is the same
// boundary-tell shape as the rest of the family: the test CLAIMS coverage while its
// assertions never execute.
//
// A skip guarded by a documented condition is an HONEST conditional skip — the test
// still runs in the configuration it targets — and is NOT flagged:
//
//	if testing.Short()           { t.Skip("slow; run without -short") } // short-mode guard
//	if runtime.GOOS == "windows" { t.Skip("POSIX-only") }               // platform guard
//	if os.Getenv("CI") == ""     { t.Skip("needs CI credentials") }     // environment guard
//
// A deliberate always-skip (a quarantined flake tracked by an issue) is a recorded
// decision — suppress it in place with //boundarylint:ignore SKIP_DEBT and the issue
// link, so the exception is greppable rather than silent.
//
// SkipDebt is a SOFT signal: unlike the enforced DefaultRules/DefaultTestRules families
// it is not part of any gate, so surfacing skip debt never reds the build. It is scanned
// over the whole test tree (documented skips DO exist, e.g. platform guards) and reported
// as a trend — `fak boundary` lists it separately from the gating tells, and the
// qa-process scorecard folds it as a SOFT KPI with a work-list of every skip site.
type SkipDebt struct{}

// Code returns this rule's stable finding code, "SKIP_DEBT".
func (SkipDebt) Code() string { return "SKIP_DEBT" }

// skipMethods are the testing.T/B methods that remove the running test from the suite.
// Skipf and SkipNow are unambiguously the testing API; Skip is shared with the odd
// iterator/scanner method, but over _test.go the testing receiver dominates and a rare
// false positive is suppressible in place (SkipDebt is SOFT, never a gate).
var skipMethods = map[string]bool{"Skip": true, "Skipf": true, "SkipNow": true}

// skipGuardSelectors are the pkg.selector references that make an enclosing `if` a
// documented skip guard: a platform check, a short-mode check, or an environment gate.
// A skip inside such an `if` is a conditional skip, not unconditional debt.
var skipGuardSelectors = map[string]map[string]bool{
	"testing": {"Short": true},
	"runtime": {"GOOS": true, "GOARCH": true},
	"os":      {"Getenv": true, "LookupEnv": true},
}

// Check reports every skip call in file that is not guarded by a documented platform/
// short/env condition. Guarded skip lines are collected first, then the walk emits a
// finding for each skip call whose line is not guarded.
func (r SkipDebt) Check(fset *token.FileSet, file *ast.File, relPath string) []Finding {
	guarded := guardedSkipLines(fset, file)
	var out []Finding
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || !skipMethods[sel.Sel.Name] {
			return true
		}
		line := fset.Position(call.Pos()).Line
		if guarded[line] {
			return true // documented platform/short/env-guarded skip — an honest conditional, not debt
		}
		out = append(out, Finding{
			Code: r.Code(),
			File: relPath,
			Line: line,
			Detail: "unconditional t." + sel.Sel.Name + "() removes this test from the suite with no platform/short/env guard; " +
				"restore the assertion, guard the skip with a documented condition (testing.Short / runtime.GOOS / os.Getenv), " +
				"or //boundarylint:ignore SKIP_DEBT with a tracking issue",
		})
		return true
	})
	return out
}

// guardedSkipLines returns the set of source lines holding a skip call that sits inside
// an `if` whose condition tests the platform, short mode, or the environment — the
// conditions that make a skip a documented conditional rather than an unconditional
// removal. Keying by line lets Check exempt exactly those skip calls in a single pass.
func guardedSkipLines(fset *token.FileSet, file *ast.File) map[int]bool {
	guarded := map[int]bool{}
	ast.Inspect(file, func(n ast.Node) bool {
		ifs, ok := n.(*ast.IfStmt)
		if !ok || !isSkipGuardCond(ifs.Cond) {
			return true
		}
		markSkipLines(fset, ifs.Body, guarded)
		if ifs.Else != nil {
			markSkipLines(fset, ifs.Else, guarded)
		}
		return true
	})
	return guarded
}

// isSkipGuardCond reports whether cond references any skipGuardSelectors entry — i.e.
// the `if` is a documented platform/short/env guard.
func isSkipGuardCond(cond ast.Expr) bool {
	found := false
	ast.Inspect(cond, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok {
			return true
		}
		if skipGuardSelectors[pkg.Name][sel.Sel.Name] {
			found = true
			return false
		}
		return true
	})
	return found
}

// markSkipLines records the line of every skip call within node into guarded.
func markSkipLines(fset *token.FileSet, node ast.Node, guarded map[int]bool) {
	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && skipMethods[sel.Sel.Name] {
			guarded[fset.Position(call.Pos()).Line] = true
		}
		return true
	})
}
