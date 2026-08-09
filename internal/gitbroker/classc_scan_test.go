package gitbroker

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// THE CORRECTNESS LINE OF EPIC #5619, ENFORCED BY THE BUILD (#5623).
//
// The rule this file holds: a decision-bearing (Class C) query — one whose answer
// feeds a commit gate, a mutation, or a refusal — may never be answered from the
// working-tree cache, and may never join another caller's in-flight execution
// either. It is computed fresh, permanently. Everything else in the epic is a
// performance argument; this is the one claim that, if it broke, would let the
// fleet refuse or admit a commit on a picture of a tree that no longer exists.
//
// WHY A SOURCE SCAN AND NOT A BEHAVIOURAL TEST. The failure being guarded against
// is a future edit, not a present bug: a helper added six months from now that
// reaches for the warm entry "just this once", or a decisional guard quietly
// moved below a cache read during a refactor. A behavioural test only catches a
// Class C path someone remembered to write a test for. A scan catches the shape
// itself, including in code that does not exist yet — it fails the build.
//
// WHY IT LIVES HERE AND NOT IN internal/architest. Same teeth, and the constraint
// stays next to the code it constrains, where the next person to edit treeState
// reads it. (architest is also fak's shared witness machinery, which a guarded
// worker may not ship into.)
//
// The scan is deliberately NAME-based — treeCache's accessors are called lookup
// and store precisely so it can find every call site by name, independent of what
// any receiver happens to be called. A name-based scan can be evaded by renaming,
// so the accessor set on *treeCache is pinned too: a new accessor this scan does
// not know about is itself a failure.

const (
	// treeStateFunc is the ONLY function permitted to touch the working-tree
	// cache or the working-tree single-flight group.
	treeStateFunc  = "treeState"
	treeStateRecv  = "Server"
	treeCacheType  = "treeCache"
	treeFlightName = "treeFlight"
)

// treeCacheAccessors is the complete allowed method set on *treeCache. lookup and
// store are the answer path this scan polices by name; held is a counter read for
// Stats and produces no answer. Anything else is a new way to reach the cache
// that the name scan below would not see.
var treeCacheAccessors = map[string]bool{"lookup": true, "store": true, "held": true}

// classCViolations reports every way the source could let a decisional query
// reach a reused answer. An empty result is the invariant holding.
func classCViolations(fset *token.FileSet, files []*ast.File) []string {
	var out []string
	sawGate := false
	for _, f := range files {
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			recv := receiverTypeName(fn)

			if recv == treeCacheType {
				if !treeCacheAccessors[fn.Name.Name] {
					out = append(out, fmt.Sprintf(
						"%s: %s is a new *%s accessor. This scan finds cache reads by NAME (lookup/store), so an accessor it does not know about is a hole in the Class C guarantee: rename it to lookup/store, or teach treeCacheAccessors and the call-site rule about it",
						fset.Position(fn.Pos()), fn.Name.Name, treeCacheType))
				}
				continue
			}

			permitted := fn.Name.Name == treeStateFunc && recv == treeStateRecv
			if !permitted {
				for _, r := range reusedAnswerReaches(fn.Body) {
					out = append(out, fmt.Sprintf(
						"%s: %s reaches %s. Only (*%s).%s may, because it is the only function that has already refused a decisional caller",
						fset.Position(r.pos), funcLabel(recv, fn.Name.Name), r.what, treeStateRecv, treeStateFunc))
				}
				continue
			}
			sawGate = true
			out = append(out, decisionalGateViolations(fset, fn)...)
		}
	}
	if !sawGate {
		out = append(out, fmt.Sprintf(
			"(*%s).%s is not in the package source at all: the decisional gate this package's correctness rests on has been renamed or removed, so this scan is no longer guarding anything",
			treeStateRecv, treeStateFunc))
	}
	return out
}

type reach struct {
	pos  token.Pos
	what string
}

// reusedAnswerReaches finds every call in n that could hand a caller an answer it
// did not compute: a working-tree cache read or write, or a join on the
// working-tree single-flight group.
//
// The Class A object cache and objFlight are deliberately NOT policed here. Both
// are keyed by a full OID, which names immutable content, so reusing one of those
// answers is safe for a decisional caller as well — there is nothing to go stale.
func reusedAnswerReaches(n ast.Node) []reach {
	var out []reach
	ast.Inspect(n, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		switch sel.Sel.Name {
		case "lookup", "store":
			out = append(out, reach{call.Pos(), "the working-tree cache (." + sel.Sel.Name + ")"})
		case "Do":
			if inner, ok := sel.X.(*ast.SelectorExpr); ok && inner.Sel.Name == treeFlightName {
				out = append(out, reach{call.Pos(), "the working-tree single-flight group (" + treeFlightName + ".Do)"})
			}
		}
		return true
	})
	return out
}

// decisionalGateViolations checks the shape of the one permitted function: it
// must REFUSE a decisional caller before it can possibly reuse an answer.
//
// "Before" is checked structurally rather than by ordering call positions,
// because the guard must not merely come first — it must return, so nothing below
// it is reachable for a Class C caller at all.
func decisionalGateViolations(fset *token.FileSet, fn *ast.FuncDecl) []string {
	where := fset.Position(fn.Pos())
	if len(fn.Body.List) == 0 {
		return []string{fmt.Sprintf("%s: (*%s).%s is empty", where, treeStateRecv, treeStateFunc)}
	}
	guard, ok := fn.Body.List[0].(*ast.IfStmt)
	if !ok || !isDecisionalCall(guard.Cond) {
		return []string{fmt.Sprintf(
			"%s: (*%s).%s does not OPEN with an `if <class>.Decisional()` guard. Whatever runs before that guard runs for a decisional caller too",
			where, treeStateRecv, treeStateFunc)}
	}
	var out []string
	if n := len(guard.Body.List); n == 0 {
		out = append(out, fmt.Sprintf("%s: the Decisional() guard body is empty, so a Class C query falls straight through into the cached path", where))
	} else if _, isReturn := guard.Body.List[n-1].(*ast.ReturnStmt); !isReturn {
		out = append(out, fmt.Sprintf(
			"%s: the Decisional() guard does not RETURN, so a Class C query falls through into the cached path below it",
			where))
	}
	for _, r := range reusedAnswerReaches(guard.Body) {
		out = append(out, fmt.Sprintf(
			"%s: the Decisional() branch itself reaches %s — that branch exists precisely to compute a fresh answer",
			fset.Position(r.pos), r.what))
	}
	return out
}

// isDecisionalCall matches `<expr>.Decisional()` — the one predicate that decides
// whether an answer may be reused.
func isDecisionalCall(cond ast.Expr) bool {
	call, ok := cond.(*ast.CallExpr)
	if !ok || len(call.Args) != 0 {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == "Decisional"
}

func receiverTypeName(fn *ast.FuncDecl) string {
	if fn.Recv == nil || len(fn.Recv.List) == 0 {
		return ""
	}
	t := fn.Recv.List[0].Type
	if star, ok := t.(*ast.StarExpr); ok {
		t = star.X
	}
	if id, ok := t.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

func funcLabel(recv, name string) string {
	if recv == "" {
		return name + "()"
	}
	return "(*" + recv + ")." + name
}

// parsePackageSource parses this package's NON-test files. Test files are
// excluded on purpose: a test may legitimately drive the cache directly to prove
// what it does, and it is production code that must not.
func parsePackageSource(t *testing.T) (*token.FileSet, []*ast.File) {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		files = append(files, f)
	}
	if len(files) == 0 {
		t.Fatal("parsed no package source; the scan would pass vacuously")
	}
	return fset, files
}

// TestClassCQueryCanNeverReachTheCache is the build-failing enforcement the
// acceptance gate of #5623 asks for: if a decision-bearing query can reach the
// working-tree cache — or reach an answer someone else is already computing —
// this package does not build green.
func TestClassCQueryCanNeverReachTheCache(t *testing.T) {
	fset, files := parsePackageSource(t)
	if v := classCViolations(fset, files); len(v) > 0 {
		t.Fatalf("a decision-bearing query can reach a reused answer:\n  %s\n\nAnything feeding a commit gate, a mutation, or a refusal is computed fresh, permanently (#5623). If the answer is only being REPORTED, declare it ClassB at the caller instead of widening this path.",
			strings.Join(v, "\n  "))
	}
}

// TestTheClassCScanHasTeeth is the scan's own negative test.
//
// A source scan that never fires is indistinguishable from one that cannot fire,
// and a guard nobody has watched fail is not a guard. So this mutates the REAL
// parsed package — not a synthetic imitation of it — in the two ways the rule can
// actually be broken, and requires the scan to catch both.
func TestTheClassCScanHasTeeth(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(t *testing.T, fn *ast.FuncDecl)
		wantSub string
	}{
		{
			// The refactor that removes the early return, leaving a Class C caller
			// to fall into the keyed path below it.
			name: "decisional guard deleted",
			mutate: func(t *testing.T, fn *ast.FuncDecl) {
				fn.Body.List = fn.Body.List[1:]
			},
			wantSub: "Decisional()",
		},
		{
			// The refactor that moves the cache read somewhere that has not
			// refused a decisional caller first.
			name: "cache read moved out of the guarded function",
			mutate: func(t *testing.T, fn *ast.FuncDecl) {
				fn.Name = ast.NewIdent("treeStateUnguarded")
			},
			wantSub: "the working-tree cache",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fset, files := parsePackageSource(t)
			fn := findTreeState(t, files)
			tc.mutate(t, fn)
			v := classCViolations(fset, files)
			if len(v) == 0 {
				t.Fatalf("the scan reported NO violation after %q. It cannot fail, so its green result proves nothing about Class C", tc.name)
			}
			if !strings.Contains(strings.Join(v, "\n"), tc.wantSub) {
				t.Fatalf("the scan fired on %q but not for the expected reason (want a violation mentioning %q):\n  %s",
					tc.name, tc.wantSub, strings.Join(v, "\n  "))
			}
		})
	}
}

func findTreeState(t *testing.T, files []*ast.File) *ast.FuncDecl {
	t.Helper()
	for _, f := range files {
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if ok && fn.Name.Name == treeStateFunc && receiverTypeName(fn) == treeStateRecv {
				return fn
			}
		}
	}
	t.Fatalf("(*%s).%s not found in the package source", treeStateRecv, treeStateFunc)
	return nil
}
