package dispatchtick

// Test-integrity rung for the witnessed keep-bit (#3364).
//
// CommitWitnessed (witness.go) grades a `dos commit-audit` row into the non-forgeable
// keep-bit by comparing the claim KIND to the diff SHAPE — it never reads an added test
// body. So a commit whose only new test is `func TestX(t *testing.T){ t.Log("ok") }` — a
// test that structurally CANNOT FAIL — still clears the gate and is stamped
// CLAIM_WITNESSED. This file closes that residue: a deny-by-structure rung that reads the
// ADDED test bodies (Go AST) and denies any commit where a new Test function has no
// reachable failure mechanism at all.
//
// Borrowed from the Martin Loop external self-improvement harness (package
// test-integrity.ts), studied locally 2026-07 (@b06882f) — it flags new tests that never
// import a pre-existing module, trivial/tautological assertions, and empty bodies. fak's
// tools/code_slop_scorecard.py kpi_vacuous_tests already covers empty/zero-observation
// bodies but scores `t.Log`/`t.Skip`-only bodies as NON-vacuous (its assertion regex
// counts Log/Skip as observations) and is a repo-wide advisory scorecard, never the
// keep-bit. This rung is the residue: the `t.Log`/`t.Skip`-only shape, folded into the
// witnessed keep-bit as a structured, legible refusal.
//
// Deliberately conservative: a test is flagged ONLY when the walker finds ZERO can-fail
// signals in its body. Any of {t.Error*/t.Fatal*/t.Fail*, require./assert., panic(,
// t.Run(, a call passing the *testing.T through to a helper} clears the test — so table
// tests, testify/helper wrappers, and subtests are never false-flagged. A file the parser
// cannot trust fails OPEN (counts as witnessed): this rung never denies on a parse error.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

// WitnessTestCannotFail is the structured refusal reason (closed vocabulary, normgate-style)
// the test-integrity rung emits when a commit's added test(s) structurally cannot fail — a
// body whose entire set of testing signals is observation-only (t.Log / t.Skip) with no
// reachable failure mechanism. Emitting a named reason keeps the keep-bit flip legible
// rather than a silent downgrade.
const WitnessTestCannotFail = "TEST_CANNOT_FAIL"

// AddedTestFile is one `*_test.go` file whose test bodies a commit ADDED, carried on the
// wire from the `dos commit-audit --json` row's test_files (the path) plus the diff bytes
// (the added source). Path is repo-relative; Content is the Go source the AST reads. Only
// files whose Path ends in `_test.go` are analyzed; anything else is ignored.
type AddedTestFile struct {
	Path    string
	Content string
}

// AddedTestsWitnessed reports whether the added test files clear the test-integrity rung:
// true unless at least one added Test function structurally CANNOT FAIL. It is the
// deny-by-structure companion to CommitWitnessed. Empty input (no added tests) is witnessed
// — this rung only denies a present-but-inert test, it never demands a test exist.
func AddedTestsWitnessed(files []AddedTestFile) bool {
	ok, _, _ := gradeAddedTests(files)
	return ok
}

// AddedTestsRefusal returns the structured refusal for the FIRST added test that
// structurally cannot fail: (reason, path, funcName). All empty when every added test can
// fail (or there are no added tests) — i.e. the rung is clear.
func AddedTestsRefusal(files []AddedTestFile) (reason, path, fn string) {
	ok, p, f := gradeAddedTests(files)
	if ok {
		return "", "", ""
	}
	return WitnessTestCannotFail, p, f
}

// CommitWitnessedWithIntegrity folds the test-integrity rung INTO the non-forgeable
// keep-bit: true only when CommitWitnessed (claim-vs-diff SHAPE) AND the added test files
// clear AddedTestsWitnessed. When it downgrades a commit that WOULD have been witnessed by
// the claim-vs-diff rung alone, it returns the structured refusal reason
// (WitnessTestCannotFail) so the flip is legible; when the claim-vs-diff rung already
// failed, it returns ("", false) with no new reason — that downgrade is not ours to name.
//
// This is the seam #3364 folds into dispatchtick.CommitWitnessed and
// closureaudit.commitIsWitnessed once the added test bodies are carried on the wire. Per
// the issue's advisory-first fence, promote it behind a SHADOW gate first (report the
// reason without flipping the live keep-bit) until the false-positive rate is witnessed on
// real traffic, then flip to deny-by-structure.
func CommitWitnessedWithIntegrity(verdict, witness string, addedTests []AddedTestFile) (ok bool, reason string) {
	if !CommitWitnessed(verdict, witness) {
		return false, ""
	}
	if r, _, _ := AddedTestsRefusal(addedTests); r != "" {
		return false, r
	}
	return true, ""
}

// gradeAddedTests parses each added `*_test.go` file and returns (false, path, func) for
// the first Test function whose body has no reachable failure mechanism. A parse error
// fails open (the file is skipped) so the rung never denies on source it cannot trust.
func gradeAddedTests(files []AddedTestFile) (ok bool, path, fn string) {
	for _, f := range files {
		if !strings.HasSuffix(f.Path, "_test.go") {
			continue
		}
		fset := token.NewFileSet()
		parsed, err := parser.ParseFile(fset, f.Path, f.Content, 0)
		if err != nil {
			continue // fail open: never deny on an untrusted parse
		}
		for _, decl := range parsed.Decls {
			fd, isFunc := decl.(*ast.FuncDecl)
			if !isFunc || fd.Body == nil || !isTestFunc(fd) {
				continue
			}
			if !funcCanFail(fd) {
				return false, f.Path, fd.Name.Name
			}
		}
	}
	return true, "", ""
}

// isTestFunc reports whether fd is a `func TestXxx(t *testing.T)` — a receiverless func
// whose name has the Test prefix and whose single parameter is a `*<pkg>.T`. Benchmarks,
// Fuzz targets, and Examples have different signatures and are not graded here.
func isTestFunc(fd *ast.FuncDecl) bool {
	if fd.Recv != nil || !strings.HasPrefix(fd.Name.Name, "Test") {
		return false
	}
	_, ok := testParamName(fd)
	return ok
}

// testParamName returns the identifier the test binds its *testing.T to (usually "t") and
// whether the single parameter is a pointer to a selector type named T (`*testing.T`). The
// name lets the walker recognize a helper call that passes the T through (`helper(t)`).
func testParamName(fd *ast.FuncDecl) (string, bool) {
	if fd.Type.Params == nil || len(fd.Type.Params.List) != 1 {
		return "", false
	}
	p := fd.Type.Params.List[0]
	star, ok := p.Type.(*ast.StarExpr)
	if !ok {
		return "", false
	}
	sel, ok := star.X.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "T" {
		return "", false
	}
	if len(p.Names) != 1 {
		return "", false
	}
	return p.Names[0].Name, true
}

// failMethods are the *testing.T methods that can make a test FAIL. Log/Logf/Skip/Skipf/
// SkipNow/Helper/Parallel/Cleanup/Setenv/Name/Deadline are observation or util only and are
// deliberately absent — a body built from only those cannot fail.
var failMethods = map[string]bool{
	"Error": true, "Errorf": true,
	"Fatal": true, "Fatalf": true,
	"Fail": true, "FailNow": true,
}

// funcCanFail walks a test body and reports whether any reachable failure mechanism is
// present. It stops at the first can-fail signal (short-circuit) and, being conservative in
// the SAFE direction, returns true on any doubt — a test is denied only when the walk finds
// nothing that could fail it.
func funcCanFail(fd *ast.FuncDecl) bool {
	tname, _ := testParamName(fd)
	canFail := false
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if canFail {
			return false
		}
		if call, ok := n.(*ast.CallExpr); ok && callCanFail(call, tname) {
			canFail = true
			return false
		}
		return true
	})
	return canFail
}

// callCanFail reports whether a single call could fail the test: a t.Error/Fatal/Fail
// method, a t.Run subtest (which may itself fail — clearing the parent avoids false-flagging
// real table tests), a require./assert. testify call, a panic, or any call that passes the
// bound *testing.T through to a helper (which may Fatal on the parent's behalf).
func callCanFail(call *ast.CallExpr, tname string) bool {
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		if failMethods[fun.Sel.Name] || fun.Sel.Name == "Run" {
			return true
		}
		if id, ok := fun.X.(*ast.Ident); ok && (id.Name == "require" || id.Name == "assert") {
			return true
		}
	case *ast.Ident:
		if fun.Name == "panic" {
			return true
		}
	}
	// A call that hands the test's *testing.T to a helper (`helper(t, ...)`, `p.check(t)`)
	// can fail the test through that helper — clear it regardless of the callee shape.
	return tname != "" && callPassesIdent(call, tname)
}

// callPassesIdent reports whether any argument is the bare identifier name (the test's
// *testing.T passed through to a helper). A method call on the T (`t.Name()`) is a call
// node, not the bare ident, so it does not count.
func callPassesIdent(call *ast.CallExpr, name string) bool {
	for _, a := range call.Args {
		if id, ok := a.(*ast.Ident); ok && id.Name == name {
			return true
		}
	}
	return false
}
