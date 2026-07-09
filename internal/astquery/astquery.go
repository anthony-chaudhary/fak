// Package astquery is a structural (AST-shape) search over Go source with
// metavariables — the "match code by shape, not text" seam (#3438, epic #3434).
// It is the semgrep/comby idea in miniature: a pattern is a fragment of Go with
// $NAME holes, and it matches any subtree of the same shape, binding each hole to
// the concrete node it lands on. A repeated hole must bind consistently, so
// `$X == $X` matches `a == a` but not `a == b` — the property a plain regex or the
// trigram index (#3437) cannot express.
//
// Scope is expression patterns: the pattern parses as a Go expression, and every
// expression subtree of the target file is a candidate. That already covers the
// high-value queries (call shapes, comparisons, selector chains) and keeps the
// matcher a small, total tree-unification over the go/ast node types it knows.
package astquery

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"regexp"
	"strings"
)

// Match is one structural hit: the source position of the matched subtree and the
// metavariable bindings that made it match (rendered back to source text).
type Match struct {
	Pos      token.Position
	Text     string
	Bindings map[string]string
}

// metavar syntax: $ followed by an identifier. $_ is the anonymous wildcard (matches
// any expression, binds nothing). A named hole binds and must stay consistent.
var (
	metavarSyntax  = regexp.MustCompile(`\$([A-Za-z_][A-Za-z0-9_]*)`)
	metavarPlaceRe = regexp.MustCompile(`^__mv_(.+)__$`)
)

// parsePattern rewrites $NAME holes to parseable placeholder identifiers, then
// parses the fragment as a Go expression.
func parsePattern(pattern string) (ast.Expr, error) {
	rewritten := metavarSyntax.ReplaceAllString(pattern, "__mv_${1}__")
	expr, err := parser.ParseExpr(rewritten)
	if err != nil {
		return nil, fmt.Errorf("astquery: bad pattern %q: %w", pattern, err)
	}
	return expr, nil
}

// metavarName reports the hole name for an identifier that stands in for a
// metavariable, and false if the identifier is an ordinary one.
func metavarName(id *ast.Ident) (string, bool) {
	m := metavarPlaceRe.FindStringSubmatch(id.Name)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// Search parses src (a full Go file) and returns every expression subtree matching
// pattern, with bindings. Matches are returned in source order.
func Search(src, pattern string) ([]Match, error) {
	pat, err := parsePattern(pattern)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "src.go", src, 0)
	if err != nil {
		return nil, fmt.Errorf("astquery: bad source: %w", err)
	}

	var out []Match
	ast.Inspect(file, func(n ast.Node) bool {
		expr, ok := n.(ast.Expr)
		if !ok {
			return true
		}
		binds := map[string]string{}
		if matchExpr(pat, expr, binds, fset) {
			out = append(out, Match{
				Pos:      fset.Position(expr.Pos()),
				Text:     nodeText(fset, expr),
				Bindings: binds,
			})
		}
		return true
	})
	return out, nil
}

// matchExpr unifies pattern node pat against target node n, threading binds for
// metavariable consistency. It is a total structural comparison over the node types
// it knows; an unknown or mismatched shape is a non-match, never a panic.
func matchExpr(pat, n ast.Expr, binds map[string]string, fset *token.FileSet) bool {
	// Metavariable holes are the one place pattern and target shapes may differ.
	if id, ok := pat.(*ast.Ident); ok {
		if name, isMeta := metavarName(id); isMeta {
			return bindMeta(name, n, binds, fset)
		}
	}
	if pat == nil || n == nil {
		return pat == nil && n == nil
	}

	switch p := pat.(type) {
	case *ast.Ident:
		q, ok := n.(*ast.Ident)
		return ok && p.Name == q.Name
	case *ast.BasicLit:
		q, ok := n.(*ast.BasicLit)
		return ok && p.Kind == q.Kind && p.Value == q.Value
	case *ast.ParenExpr:
		// Parentheses are not structure; unwrap on both sides.
		return matchExpr(p.X, unparen(n), binds, fset)
	case *ast.BinaryExpr:
		q, ok := n.(*ast.BinaryExpr)
		return ok && p.Op == q.Op &&
			matchExpr(p.X, q.X, binds, fset) &&
			matchExpr(p.Y, q.Y, binds, fset)
	case *ast.UnaryExpr:
		q, ok := n.(*ast.UnaryExpr)
		return ok && p.Op == q.Op && matchExpr(p.X, q.X, binds, fset)
	case *ast.StarExpr:
		q, ok := n.(*ast.StarExpr)
		return ok && matchExpr(p.X, q.X, binds, fset)
	case *ast.SelectorExpr:
		q, ok := n.(*ast.SelectorExpr)
		return ok && p.Sel.Name == q.Sel.Name && matchExpr(p.X, q.X, binds, fset)
	case *ast.IndexExpr:
		q, ok := n.(*ast.IndexExpr)
		return ok && matchExpr(p.X, q.X, binds, fset) && matchExpr(p.Index, q.Index, binds, fset)
	case *ast.CallExpr:
		q, ok := n.(*ast.CallExpr)
		if !ok || len(p.Args) != len(q.Args) {
			return false
		}
		if !matchExpr(p.Fun, q.Fun, binds, fset) {
			return false
		}
		for i := range p.Args {
			if !matchExpr(p.Args[i], q.Args[i], binds, fset) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

// bindMeta binds a metavariable hole. `_` is the anonymous wildcard: it matches any
// expression and binds nothing. A named hole binds to the target's source text on
// first sight and must match it identically on every later occurrence.
func bindMeta(name string, n ast.Expr, binds map[string]string, fset *token.FileSet) bool {
	if name == "_" {
		return true
	}
	text := nodeText(fset, n)
	if prev, seen := binds[name]; seen {
		return prev == text
	}
	binds[name] = text
	return true
}

func unparen(n ast.Expr) ast.Expr {
	for {
		p, ok := n.(*ast.ParenExpr)
		if !ok {
			return n
		}
		n = p.X
	}
}

// nodeText renders a node back to normalized Go source, the canonical form used to
// compare two metavariable occurrences.
func nodeText(fset *token.FileSet, n ast.Node) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, n); err != nil {
		return ""
	}
	return strings.TrimSpace(buf.String())
}
