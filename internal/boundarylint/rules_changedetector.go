package boundarylint

import (
	"go/ast"
	"go/token"
	"regexp"
	"strconv"
	"strings"
)

// ChangeDetectorTest flags test assertions that freeze a current value instead of
// asserting how two pieces of data must relate: a magic enumeration count
// (len(verbs) != 109), a wholly-literal list equality (reflect.DeepEqual against a
// six-element literal), or a pinned version string (version == "v1.42.7"). Such a
// test passes/fails on churn, not on correctness — it goes red when the enumeration
// legitimately grows and stays green when a real relation breaks — so it rots into
// a change detector the suite drags along. This is the same boundary-tell shape as
// the rest of the family: the assertion CLAIMS "this value is correct" while only
// checking "this value is what it was the day the test was written".
//
// The fix is an invariant: relate the count to the thing it must track (every
// dispatch verb has a help entry), relate the list to its source of truth, assert
// the version PARSES and orders after the previous release rather than equalling a
// literal. A deliberate fixed-width check (sha256 hex is 64 bytes) is a real
// invariant — suppress it in place with //boundarylint:ignore CHANGE_DETECTOR_TEST
// and the reason, so the exception is a recorded decision.
type ChangeDetectorTest struct{}

// Code returns this rule's stable finding code, "CHANGE_DETECTOR_TEST".
func (ChangeDetectorTest) Code() string { return "CHANGE_DETECTOR_TEST" }

// magicCountMin is the smallest integer literal a len() comparison is flagged at.
// Small structural counts (a split into 3 parts, a table expecting 2 findings) are
// the dominant honest assertion shape; frozen enumeration counts are larger.
const magicCountMin = 5

// frozenListMin is the smallest wholly-literal composite a deep-equality call is
// flagged at. A two-element literal is a readable expected value; a five-plus
// element all-literal list is usually a frozen snapshot of an enumeration.
const frozenListMin = 5

// versionLitRe matches a semver-shaped string literal ("1.2", "v1.42.7",
// "2.0.0-rc1"). The rule additionally requires the other operand to be
// version-named, so an incidental dotted id compared to a non-version variable is
// not flagged.
var versionLitRe = regexp.MustCompile(`^v?\d+\.\d+(\.\d+)*(-[0-9A-Za-z.\-]+)?(\+[0-9A-Za-z.\-]+)?$`)

// deepEqualCallees maps package ident → callee names whose call is a whole-value
// equality over its arguments. cmp.Diff is included: asserting an empty diff
// against a frozen literal is the same snapshot.
var deepEqualCallees = map[string]map[string]bool{
	"reflect": {"DeepEqual": true},
	"slices":  {"Equal": true},
	"cmp":     {"Equal": true, "Diff": true},
}

func (r ChangeDetectorTest) Check(fset *token.FileSet, file *ast.File, relPath string) []Finding {
	var out []Finding
	flag := func(pos token.Pos, detail string) {
		out = append(out, Finding{
			Code:   r.Code(),
			File:   relPath,
			Line:   fset.Position(pos).Line,
			Detail: detail,
		})
	}
	ast.Inspect(file, func(n ast.Node) bool {
		switch e := n.(type) {
		case *ast.BinaryExpr:
			if e.Op != token.EQL && e.Op != token.NEQ {
				return true
			}
			if n, ok := magicLenComparison(e); ok {
				flag(e.Pos(), "len() frozen to the magic count "+strconv.Itoa(n)+
					"; assert the relation the count stands for (e.g. every X has a Y), not today's total")
			}
			if lit, ok := versionPinComparison(e); ok {
				flag(e.Pos(), "version pinned to the literal "+strconv.Quote(lit)+
					"; assert an invariant (parses, orders after the prior release), not the current value")
			}
		case *ast.CallExpr:
			if n, ok := frozenListEquality(e); ok {
				flag(e.Pos(), "deep equality against a frozen "+strconv.Itoa(n)+
					"-element literal; compare against the enumeration's source of truth, not a snapshot")
			}
		}
		return true
	})
	return out
}

// magicLenComparison reports whether e compares len(x) (either side) against an
// integer literal >= magicCountMin, returning the literal's value.
func magicLenComparison(e *ast.BinaryExpr) (int, bool) {
	for _, pair := range [2][2]ast.Expr{{e.X, e.Y}, {e.Y, e.X}} {
		call, ok := pair[0].(*ast.CallExpr)
		if !ok {
			continue
		}
		fn, ok := call.Fun.(*ast.Ident)
		if !ok || fn.Name != "len" {
			continue
		}
		lit, ok := pair[1].(*ast.BasicLit)
		if !ok || lit.Kind != token.INT {
			continue
		}
		n, err := strconv.ParseInt(strings.ReplaceAll(lit.Value, "_", ""), 0, 64)
		if err != nil || n < magicCountMin {
			continue
		}
		return int(n), true
	}
	return 0, false
}

// versionPinComparison reports whether e compares a version-named expression
// against a semver-shaped string literal, returning the pinned literal.
func versionPinComparison(e *ast.BinaryExpr) (string, bool) {
	for _, pair := range [2][2]ast.Expr{{e.X, e.Y}, {e.Y, e.X}} {
		lit, ok := pair[0].(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			continue
		}
		val, err := strconv.Unquote(lit.Value)
		if err != nil || !versionLitRe.MatchString(val) {
			continue
		}
		if !mentionsVersion(pair[1]) {
			continue
		}
		return val, true
	}
	return "", false
}

// mentionsVersion reports whether any identifier inside expr is version-named.
func mentionsVersion(expr ast.Expr) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && strings.Contains(strings.ToLower(id.Name), "version") {
			found = true
			return false
		}
		return !found
	})
	return found
}

// frozenListEquality reports whether call is a deep-equality callee with a
// wholly-literal composite argument of >= frozenListMin elements, returning the
// element count.
func frozenListEquality(call *ast.CallExpr) (int, bool) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return 0, false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || !deepEqualCallees[pkg.Name][sel.Sel.Name] {
		return 0, false
	}
	for _, arg := range call.Args {
		lit, ok := arg.(*ast.CompositeLit)
		if !ok || len(lit.Elts) < frozenListMin {
			continue
		}
		if allLiteralElements(lit.Elts) {
			return len(lit.Elts), true
		}
	}
	return 0, false
}

// allLiteralElements reports whether every element is a basic literal (allowing a
// leading unary sign and literal-key/literal-value map entries). Any identifier or
// nested composite means the expected value is derived, not frozen — not flagged.
func allLiteralElements(elts []ast.Expr) bool {
	for _, el := range elts {
		if kv, ok := el.(*ast.KeyValueExpr); ok {
			if !isBasicLiteral(kv.Key) || !isBasicLiteral(kv.Value) {
				return false
			}
			continue
		}
		if !isBasicLiteral(el) {
			return false
		}
	}
	return true
}

func isBasicLiteral(e ast.Expr) bool {
	if u, ok := e.(*ast.UnaryExpr); ok && (u.Op == token.SUB || u.Op == token.ADD) {
		e = u.X
	}
	_, ok := e.(*ast.BasicLit)
	return ok
}
