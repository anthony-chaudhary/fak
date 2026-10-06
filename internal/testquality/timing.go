package testquality

import (
	"fmt"
	"go/ast"
	"strconv"
)

// timeImportName returns the local name the file binds the "time" package to, or
// "" when the file does not import it (or dot/blank-imports it, which this
// package does not follow). Matching the literal `time.` selector without this
// check would misread a local variable called `time` as the package.
func timeImportName(file *ast.File) string {
	for _, imp := range file.Imports {
		p, err := strconv.Unquote(imp.Path.Value)
		if err != nil || p != "time" {
			continue
		}
		if imp.Name == nil {
			return "time"
		}
		if imp.Name.Name == "." || imp.Name.Name == "_" {
			return ""
		}
		return imp.Name.Name
	}
	return ""
}

// isTimeCall reports whether call is `<timeName>.<fn>(...)` for one of fns.
func isTimeCall(call *ast.CallExpr, timeName string, fns ...string) bool {
	if timeName == "" {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != timeName {
		return false
	}
	for _, f := range fns {
		if sel.Sel.Name == f {
			return true
		}
	}
	return false
}

// sleepSync reports a TestXxx that calls time.Sleep — the test's verdict now
// depends on the scheduler finishing some other work inside a guessed window.
// Under a loaded CI runner or a busy shared dev box the guess is wrong and the
// test flakes; on a quiet box the same sleep is pure wall-clock cost.
//
// One finding per test function, not per call: the defect is "this test
// synchronises on time", and a count per function keeps the baseline row about
// the test rather than about how many sleeps it happens to have. Sleeps inside
// function literals count, because a goroutine or t.Run body sleeping is the
// same hazard. A sleep in a non-Test helper is not followed, and a sleep inside
// a for/range loop (a condition poll) is not reported (silence-leaning).
func sleepSync(file string, fd *ast.FuncDecl, timeName string, line func(ast.Node) int) []Finding {
	// A sleep inside a loop is the poll-until-condition idiom (usually under a
	// deadline), which is the event-driven fix this rule recommends, not the
	// defect. Telling a good poll from a fixed retry count needs data flow, so
	// every in-loop sleep is left alone (silence-leaning).
	inLoop := map[ast.Node]bool{}
	ast.Inspect(fd.Body, func(node ast.Node) bool {
		var body *ast.BlockStmt
		switch l := node.(type) {
		case *ast.ForStmt:
			body = l.Body
		case *ast.RangeStmt:
			body = l.Body
		default:
			return true
		}
		ast.Inspect(body, func(c ast.Node) bool {
			if call, ok := c.(*ast.CallExpr); ok {
				inLoop[call] = true
			}
			return true
		})
		return true
	})
	var first ast.Node
	n := 0
	ast.Inspect(fd.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if ok && !inLoop[call] && isTimeCall(call, timeName, "Sleep") {
			if first == nil {
				first = call
			}
			n++
		}
		return true
	})
	if n == 0 {
		return nil
	}
	return []Finding{{
		Code: CodeSleepSync, File: file, Func: fd.Name.Name, Line: line(first),
		Detail: fmt.Sprintf("%d time.Sleep call(s): the test synchronises on a guessed wall-clock window "+
			"instead of an event (channel, WaitGroup, condition poll with deadline, injected clock), so it "+
			"flakes under load and burns wall time when idle", n),
	}}
}

// wallclockAsserts reports an `if` whose condition reads elapsed wall-clock time
// (time.Since / time.Until, or a variable assigned from one) and whose body fails
// the test. That assertion measures the machine, not the code: a loaded runner
// fails it with the code under test unchanged, which is the "load-sensitive
// budget test red on base too" shape.
//
// Only the direct shape is matched — the elapsed value has to be produced by
// time.Since/time.Until in the same test, either inline in the condition or via
// one named assignment. An elapsed value computed in a helper is invisible.
func wallclockAsserts(file string, fd *ast.FuncDecl, timeName string, vars map[string]bool, line func(ast.Node) int) []Finding {
	if timeName == "" {
		return nil
	}
	isElapsedCall := func(e ast.Expr) bool {
		call, ok := e.(*ast.CallExpr)
		return ok && isTimeCall(call, timeName, "Since", "Until")
	}
	elapsedVars := map[string]bool{}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != len(as.Rhs) {
			return true
		}
		for i, r := range as.Rhs {
			if !isElapsedCall(r) {
				continue
			}
			if id, ok := as.Lhs[i].(*ast.Ident); ok && id.Name != "_" {
				elapsedVars[id.Name] = true
			}
		}
		return true
	})
	readsElapsed := func(cond ast.Expr) bool {
		found := false
		ast.Inspect(cond, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.CallExpr:
				if isElapsedCall(x) {
					found = true
				}
			case *ast.Ident:
				if elapsedVars[x.Name] {
					found = true
				}
			}
			return !found
		})
		return found
	}
	fails := func(body *ast.BlockStmt) bool {
		found := false
		ast.Inspect(body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if m, ok := callOnTestVar(call, vars); ok && failMethods[m] {
					found = true
				}
			}
			return !found
		})
		return found
	}
	var out []Finding
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		is, ok := n.(*ast.IfStmt)
		if !ok || !readsElapsed(is.Cond) || !fails(is.Body) {
			return true
		}
		out = append(out, Finding{
			Code: CodeWallclockAssert, File: file, Func: fd.Name.Name, Line: line(is),
			Detail: "fails the test on measured wall-clock elapsed time: the verdict depends on how " +
				"loaded the machine is, not on the code under test — assert on a count, an injected " +
				"clock, or move the budget to a benchmark",
		})
		return true
	})
	return out
}

// unconditionalSkips reports a t.Skip/Skipf/SkipNow that is a TOP-LEVEL statement
// of the test body: nothing guards it, so the test never runs anywhere and every
// line after it is dead. That is a deleted test that still counts as one.
// A skip inside any if/switch/select/closure is conditional and not reported.
func unconditionalSkips(file string, fd *ast.FuncDecl, vars map[string]bool, line func(ast.Node) int) []Finding {
	for _, st := range fd.Body.List {
		es, ok := st.(*ast.ExprStmt)
		if !ok {
			// An earlier statement that can return (the helper-process shape:
			// "if invoked as the child, do the work and return; else skip") makes
			// a later skip conditional, so stop looking.
			if containsReturn(st) {
				return nil
			}
			continue
		}
		call, ok := es.X.(*ast.CallExpr)
		if !ok {
			continue
		}
		if m, ok := callOnTestVar(call, vars); ok && skipMethods[m] {
			return []Finding{{
				Code: CodeUnconditionalSkip, File: file, Func: fd.Name.Name, Line: line(call),
				Detail: "t." + m + " is an unguarded top-level statement: the test is skipped on every " +
					"machine and every run, so it reports as a test while checking nothing — fix it, " +
					"guard the skip on the real precondition, or delete it",
			}}
		}
	}
	return nil
}

// containsReturn reports whether st contains a return statement outside any
// nested function literal.
func containsReturn(st ast.Stmt) bool {
	found := false
	ast.Inspect(st, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.ReturnStmt:
			found = true
		}
		return !found
	})
	return found
}
