package testquality

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strings"
)

// Analyze reports the candidates in one test file. name is used for Finding.File
// and for the parse error; src is the file's bytes.
//
// A parse error is RETURNED, never swallowed. A file this package cannot read is
// a file it cannot judge, and reporting zero findings for it would be a lie in
// the shape of a pass — the exact failure mode the whole package exists to catch.
func Analyze(name string, src []byte) ([]Finding, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, name, src, parser.SkipObjectResolution)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", name, err)
	}
	var out []Finding
	for _, d := range file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil || fd.Recv != nil || !isTestFunc(fd) {
			continue
		}
		out = append(out, analyzeTestFunc(fset, name, fd)...)
	}
	// A deterministic order is part of the ratchet: NewFindings calls the Nth
	// finding of a key "new" once the count passes the floor, so which of two
	// same-key findings is reported must not depend on the walk order.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Line != out[j].Line {
			return out[i].Line < out[j].Line
		}
		return out[i].Code < out[j].Code
	})
	return out, nil
}

// analyzeTestFunc runs every rule over one test function.
func analyzeTestFunc(fset *token.FileSet, file string, fd *ast.FuncDecl) []Finding {
	line := func(n ast.Node) int { return fset.Position(n.Pos()).Line }
	var out []Finding
	out = append(out, noAssertion(file, fd, line)...)
	out = append(out, selfComparisons(file, fd, line)...)
	out = append(out, uncheckedErrors(fset, file, fd, line)...)
	out = append(out, unreadExpectations(file, fd, line)...)
	return out
}

// harnessEntryPoints are `TestXxx` functions that are not tests at all, so
// "asserts nothing" is their correct shape rather than a defect.
//
// TestHelperProcess is the os/exec stdlib idiom: the test binary re-executes
// itself and this function IS the fake child, so its whole job is to print and
// exit. Exempting these by name is exact; leaving them to the baseline would put
// permanent rows there that a reader would reasonably try to "fix".
var harnessEntryPoints = map[string]bool{"TestMain": true, "TestHelperProcess": true}

// isTestFunc reports whether fd is a `func TestXxx(t *testing.T)` — the only
// shape analysed. Benchmarks, fuzz targets and examples are out of scope: a
// benchmark that asserts nothing is normal, so including them would be pure
// false positives.
func isTestFunc(fd *ast.FuncDecl) bool {
	n := fd.Name.Name
	if !strings.HasPrefix(n, "Test") || harnessEntryPoints[n] {
		return false
	}
	// The character after "Test" must not be lower-case — the same rule `go test`
	// itself applies, so `Testify`-style helpers are not mistaken for tests.
	if len(n) > 4 && n[4] >= 'a' && n[4] <= 'z' {
		return false
	}
	ps := fd.Type.Params
	if ps == nil || len(ps.List) != 1 {
		return false
	}
	return isTestingTPtr(ps.List[0].Type)
}

// isTestingTPtr matches the type `*testing.T`.
func isTestingTPtr(e ast.Expr) bool {
	st, ok := e.(*ast.StarExpr)
	if !ok {
		return false
	}
	sel, ok := st.X.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	return ok && pkg.Name == "testing" && sel.Sel.Name == "T"
}

// failMethods FAIL the test. A test with none of these reachable cannot report a
// defect no matter what the code under test does.
var failMethods = map[string]bool{
	"Error": true, "Errorf": true, "Fatal": true, "Fatalf": true,
	"Fail": true, "FailNow": true,
}

// skipMethods make a test inert BY DECLARATION rather than by accident, which is
// a different thing from asserting nothing.
var skipMethods = map[string]bool{"Skip": true, "Skipf": true, "SkipNow": true}

// testVars collects every identifier bound to a *testing.T inside fd, including
// the shadowing `t` of `t.Run(name, func(t *testing.T){…})`. Missing the shadow
// would make every subtest's assertions invisible and report the whole tree.
func testVars(fd *ast.FuncDecl) map[string]bool {
	vars := map[string]bool{}
	add := func(fl *ast.FieldList) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			if !isTestingTPtr(f.Type) {
				continue
			}
			for _, n := range f.Names {
				if n.Name != "_" {
					vars[n.Name] = true
				}
			}
		}
	}
	add(fd.Type.Params)
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if fl, ok := n.(*ast.FuncLit); ok {
			add(fl.Type.Params)
		}
		return true
	})
	return vars
}

// callOnTestVar returns the method name when call is `t.Something(...)` for some
// *testing.T in vars.
func callOnTestVar(call *ast.CallExpr, vars map[string]bool) (string, bool) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok || !vars[id.Name] {
		return "", false
	}
	return sel.Sel.Name, true
}

// noAssertion reports a test that cannot fail: no reachable t.Error/t.Fatal, no
// t.Skip, and the *testing.T never handed to another function.
//
// Delegation is TRUSTED, not followed — `wantRefusal(t, err, "…")` counts as an
// assertion. That is what keeps the house helper pattern from reading as a
// tree-wide defect, and it is also the rule's hole (a helper that asserts nothing
// launders every caller), named in the package doc.
func noAssertion(file string, fd *ast.FuncDecl, line func(ast.Node) int) []Finding {
	vars := testVars(fd)
	var hasFail, hasSkip, delegates bool
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if m, ok := callOnTestVar(call, vars); ok {
			if failMethods[m] {
				hasFail = true
			}
			if skipMethods[m] {
				hasSkip = true
			}
			return true
		}
		for _, a := range call.Args {
			if id, ok := a.(*ast.Ident); ok && vars[id.Name] {
				delegates = true
			}
		}
		return true
	})
	if hasFail || hasSkip || delegates {
		return nil
	}
	return []Finding{{
		Code: CodeNoAssertion, File: file, Func: fd.Name.Name, Line: line(fd),
		Detail: "no reachable failure call: no t.Error/t.Fatal, no t.Skip, and the *testing.T is " +
			"never handed to a helper — this test executes code and then reports success " +
			"unconditionally, so the code under test can be deleted with the package green",
	}}
}

// equalFuncs are two-argument equality predicates: `f(a, a)` is true by
// construction whatever a is.
var equalFuncs = map[string]bool{
	"DeepEqual": true, "Equal": true, "EqualFold": true, "EqualValues": true,
	"Is": true, "Contains": true, "HasPrefix": true, "HasSuffix": true,
}

// equalMethods are one-argument equality predicates called ON the value: `a.Equal(a)`.
var equalMethods = map[string]bool{"Equal": true, "Is": true, "Contains": true, "Match": true}

// selfComparisons reports an assertion that compares a value to itself.
//
// The operands must be SIDE-EFFECT-FREE for the two spellings to be the same
// value: `next() == next()` renders identically and is not a self-comparison, and
// neither is `<-ch == <-ch`. So only a restricted expression grammar (identifier,
// selector, index, literal, and pure unary/binary/paren over those) is compared,
// and at least one identifier must be involved.
func selfComparisons(file string, fd *ast.FuncDecl, line func(ast.Node) int) []Finding {
	// `x != x` is the NaN idiom. It is the one place a self-comparison is a real
	// check, so a function that talks about NaN stands the != rule down rather than
	// sending someone to "fix" a correct float test.
	nanAware := false
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && strings.Contains(id.Name, "NaN") {
			nanAware = true
		}
		return true
	})

	var out []Finding
	report := func(n ast.Node, detail string) {
		out = append(out, Finding{
			Code: CodeSelfComparison, File: file, Func: fd.Name.Name, Line: line(n), Detail: detail,
		})
	}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.BinaryExpr:
			switch x.Op {
			case token.EQL, token.NEQ, token.LSS, token.GTR, token.LEQ, token.GEQ:
			default:
				return true
			}
			if x.Op == token.NEQ && nanAware {
				return true
			}
			if !sameSideEffectFree(x.X, x.Y) {
				return true
			}
			report(x, fmt.Sprintf("compares %s to itself (%s %s %s): the condition is decided by the "+
				"expression's own shape, not by the code under test, so this assertion cannot fail",
				types.ExprString(x.X), types.ExprString(x.X), x.Op, types.ExprString(x.Y)))
		case *ast.CallExpr:
			sel, ok := x.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if len(x.Args) == 2 && equalFuncs[sel.Sel.Name] && sameSideEffectFree(x.Args[0], x.Args[1]) {
				report(x, fmt.Sprintf("%s is called with the same argument twice (%s): the predicate is "+
					"true by construction, so this assertion cannot fail",
					types.ExprString(x.Fun), types.ExprString(x.Args[0])))
				return true
			}
			if len(x.Args) == 1 && equalMethods[sel.Sel.Name] && sameSideEffectFree(sel.X, x.Args[0]) {
				report(x, fmt.Sprintf("%s compares %s to itself: the predicate is true by construction, "+
					"so this assertion cannot fail", types.ExprString(x.Fun), types.ExprString(sel.X)))
			}
		}
		return true
	})
	return out
}

// sameSideEffectFree reports whether a and b are the same side-effect-free
// expression naming at least one identifier.
func sameSideEffectFree(a, b ast.Expr) bool {
	if !sideEffectFree(a) || !sideEffectFree(b) {
		return false
	}
	if types.ExprString(a) != types.ExprString(b) {
		return false
	}
	return namesAnIdent(a)
}

// sideEffectFree reports whether evaluating e twice is guaranteed to be the same
// as evaluating it once. Deliberately narrow: anything not in this grammar
// (calls, channel receives, function literals, type assertions) is assumed to be
// effectful, because a false positive here condemns correct code.
func sideEffectFree(e ast.Expr) bool {
	switch x := e.(type) {
	case *ast.Ident, *ast.BasicLit:
		return true
	case *ast.ParenExpr:
		return sideEffectFree(x.X)
	case *ast.SelectorExpr:
		return sideEffectFree(x.X)
	case *ast.IndexExpr:
		return sideEffectFree(x.X) && sideEffectFree(x.Index)
	case *ast.StarExpr:
		return sideEffectFree(x.X)
	case *ast.UnaryExpr:
		return x.Op != token.ARROW && sideEffectFree(x.X)
	case *ast.BinaryExpr:
		return sideEffectFree(x.X) && sideEffectFree(x.Y)
	default:
		return false
	}
}

// namesAnIdent reports whether e mentions an identifier, so `0 == 0` in a
// constant-folding fixture is not reported as somebody's broken assertion.
func namesAnIdent(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if _, ok := n.(*ast.Ident); ok {
			found = true
		}
		return !found
	})
	return found
}

// looksLikeError reports whether an identifier name is conventionally an error.
// A name heuristic standing in for type information, and the rule's main recall
// limit (see the package doc). Deliberately narrow: widening it to every
// nil-compared identifier would sweep in slices and maps, a different and mostly
// benign shape, and false positives are what get a checker disabled.
func looksLikeError(name string) bool {
	return name == "e" || strings.Contains(strings.ToLower(name), "err")
}

// uncheckedErrors reports an error value that is captured and then never
// inspected before it is overwritten or the function ends.
//
// The shape that matters is the one the compiler CANNOT catch. `v, err := f()`
// with err never used again is a build failure ("declared and not used"), so it
// never reaches review. `v, err := f(); w, err := g(); if err != nil {…}` builds
// cleanly and silently drops f's error on the floor — the call's whole failure
// mode is untestable, and the test is green either way.
//
// The window for one assignment runs from the END of its own statement to the END
// of the next assignment to the same name. Ending the window at the NEXT
// assignment's end (not its start) means `err = wrap(err)` counts as inspecting
// the earlier value, which is the silence-leaning reading.
func uncheckedErrors(fset *token.FileSet, file string, fd *ast.FuncDecl, line func(ast.Node) int) []Finding {
	type assign struct {
		name  string
		node  ast.Node
		start token.Pos
		end   token.Pos
	}
	var assigns []assign
	lhs := map[*ast.Ident]bool{}

	ast.Inspect(fd.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || (as.Tok != token.DEFINE && as.Tok != token.ASSIGN) {
			return true
		}
		for _, l := range as.Lhs {
			id, ok := l.(*ast.Ident)
			if !ok || id.Name == "_" || !looksLikeError(id.Name) {
				continue
			}
			lhs[id] = true
			assigns = append(assigns, assign{name: id.Name, node: id, start: as.Pos(), end: as.End()})
		}
		return true
	})
	if len(assigns) == 0 {
		return nil
	}

	uses := map[string][]token.Pos{}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok || lhs[id] || !looksLikeError(id.Name) {
			return true
		}
		uses[id.Name] = append(uses[id.Name], id.Pos())
		return true
	})

	sort.SliceStable(assigns, func(i, j int) bool { return assigns[i].start < assigns[j].start })
	byName := map[string][]assign{}
	for _, a := range assigns {
		byName[a.name] = append(byName[a.name], a)
	}

	var out []Finding
	for name, list := range byName {
		for i, a := range list {
			limit := token.Pos(1 << 30) // to the end of the function
			if i+1 < len(list) {
				limit = list[i+1].end
			}
			checked := false
			for _, u := range uses[name] {
				if u >= a.end && u < limit {
					checked = true
					break
				}
			}
			if checked {
				continue
			}
			detail := fmt.Sprintf("%q is assigned here and never inspected before the function ends: "+
				"the error this call can return cannot fail the test", name)
			if i+1 < len(list) {
				detail = fmt.Sprintf("%q is assigned here and never inspected before it is reassigned at "+
					"line %d: the error this call can return cannot fail the test",
					name, fset.Position(list[i+1].start).Line)
			}
			out = append(out, Finding{
				Code: CodeUncheckedErr, File: file, Func: fd.Name.Name, Line: line(a.node), Detail: detail,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Line < out[j].Line })
	return out
}

// expectationPrefixes name a table field that states what the test EXPECTS. A row
// that declares one and never reads it documents an assertion the test does not
// make — the table looks like coverage and is not.
var expectationPrefixes = []string{"want", "expect", "golden"}

// isExpectationField reports whether a struct field name states an expectation.
func isExpectationField(name string) bool {
	l := strings.ToLower(name)
	for _, p := range expectationPrefixes {
		if strings.HasPrefix(l, p) {
			return true
		}
	}
	return false
}

// unreadExpectations reports a table-test row field named like an expectation
// that nothing in the test ever reads.
//
// The rule stands DOWN whenever the row variable is used as a whole value
// anywhere (passed to a helper, re-bound by the `tc := tc` idiom, stored): the
// fields are then read somewhere this package does not follow, and reporting them
// would condemn a correct test.
func unreadExpectations(file string, fd *ast.FuncDecl, line func(ast.Node) int) []Finding {
	var out []Finding
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		rs, ok := n.(*ast.RangeStmt)
		if !ok {
			return true
		}
		row, ok := rs.Value.(*ast.Ident)
		if !ok || row.Name == "_" {
			return true
		}
		st := rowStructType(rs.X, fd)
		if st == nil || st.Fields == nil || len(st.Fields.List) < 2 {
			return true // a one-field row is not a table
		}
		read, bare := rowFieldUse(fd, row)
		if bare {
			return true
		}
		for _, f := range st.Fields.List {
			for _, nm := range f.Names {
				if !isExpectationField(nm.Name) || read[nm.Name] {
					continue
				}
				out = append(out, Finding{
					Code: CodeUnreadExpectation, File: file, Func: fd.Name.Name, Line: line(nm),
					Detail: fmt.Sprintf("table field %q states an expectation that no code in this test "+
						"reads (no %s.%s anywhere): the rows document an assertion the test does not make",
						nm.Name, row.Name, nm.Name),
				})
			}
		}
		return true
	})
	sort.SliceStable(out, func(i, j int) bool { return out[i].Line < out[j].Line })
	return out
}

// rowFieldUse returns the set of row fields selected anywhere in fd, and whether
// the row variable is ever used as a bare value (which delegates every field).
func rowFieldUse(fd *ast.FuncDecl, row *ast.Ident) (read map[string]bool, bare bool) {
	read = map[string]bool{}
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == row.Name {
				read[sel.Sel.Name] = true
				return false // do not descend: the X here is a field read, not a bare use
			}
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == row.Name && id != row {
			bare = true
		}
		return true
	})
	return read, bare
}

// rowStructType resolves the element struct type a range statement iterates:
// the inline `[]struct{…}{…}` literal, a named local slice/map, or a slice of a
// struct type declared in the same function. It returns nil for anything it
// cannot resolve syntactically — an unresolved table is simply not judged.
func rowStructType(x ast.Expr, fd *ast.FuncDecl) *ast.StructType {
	switch e := x.(type) {
	case *ast.CompositeLit:
		return elemStructType(e.Type, fd)
	case *ast.Ident:
		var found *ast.StructType
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			switch s := n.(type) {
			case *ast.AssignStmt:
				for i, l := range s.Lhs {
					id, ok := l.(*ast.Ident)
					if !ok || id.Name != e.Name || i >= len(s.Rhs) {
						continue
					}
					if cl, ok := s.Rhs[i].(*ast.CompositeLit); ok {
						if st := elemStructType(cl.Type, fd); st != nil {
							found = st
						}
					}
				}
			case *ast.ValueSpec:
				for i, nm := range s.Names {
					if nm.Name != e.Name {
						continue
					}
					if s.Type != nil {
						if st := elemStructType(s.Type, fd); st != nil {
							found = st
						}
					}
					if i < len(s.Values) {
						if cl, ok := s.Values[i].(*ast.CompositeLit); ok {
							if st := elemStructType(cl.Type, fd); st != nil {
								found = st
							}
						}
					}
				}
			}
			return found == nil
		})
		return found
	}
	return nil
}

// elemStructType returns the struct type a slice/array/map type ranges over.
func elemStructType(t ast.Expr, fd *ast.FuncDecl) *ast.StructType {
	var elem ast.Expr
	switch e := t.(type) {
	case *ast.ArrayType:
		elem = e.Elt
	case *ast.MapType:
		elem = e.Value
	default:
		return nil
	}
	switch e := elem.(type) {
	case *ast.StructType:
		return e
	case *ast.StarExpr:
		if st, ok := e.X.(*ast.StructType); ok {
			return st
		}
		if id, ok := e.X.(*ast.Ident); ok {
			return localStructType(fd, id.Name)
		}
	case *ast.Ident:
		return localStructType(fd, e.Name)
	}
	return nil
}

// localStructType finds `type name struct{…}` declared inside fd. A type declared
// at package scope is not resolved: this package parses one file at a time and
// must never guess at a declaration it cannot see.
func localStructType(fd *ast.FuncDecl, name string) *ast.StructType {
	var found *ast.StructType
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		ts, ok := n.(*ast.TypeSpec)
		if !ok || ts.Name.Name != name {
			return true
		}
		if st, ok := ts.Type.(*ast.StructType); ok {
			found = st
		}
		return found == nil
	})
	return found
}
