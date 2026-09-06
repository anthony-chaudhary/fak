// Package astquery is a structural (AST-shape) search over Go source with
// metavariables — the "match code by shape, not text" seam (#3438, epic #3434).
// It is the semgrep/comby idea in miniature: a pattern is a fragment of Go with
// $NAME holes, and it matches any subtree of the same shape, binding each hole to
// the concrete node it lands on. A repeated hole must bind consistently, so
// `$X == $X` matches `a == a` but not `a == b` — the property a plain regex or the
// trigram index (#3437) cannot express.
//
// Scope covers expression patterns (ast.Expr), statement patterns (ast.Stmt), and
// declaration patterns (ast.Decl). If a pattern parses as a statement, statement
// subtrees are matched; if it parses as an expression, expression subtrees are matched.
package astquery

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Match is one structural hit: the source position of the matched subtree and the
// metavariable bindings that made it match (rendered back to source text).
type Match struct {
	Pos      token.Position    `json:"pos"`
	EndPos   token.Position    `json:"end_pos,omitempty"`
	Text     string            `json:"text"`
	Bindings map[string]string `json:"bindings"`
}

// FileMatch represents a single structural match in a file for multi-file search.
type FileMatch struct {
	File     string            `json:"file"`
	Line     int               `json:"line"`
	Text     string            `json:"text"`
	Bindings map[string]string `json:"bindings"`
}

// metavar syntax: $ followed by an identifier. $_ is the anonymous wildcard (matches
// any node, binds nothing). A named hole binds and must stay consistent.
var (
	metavarSyntax  = regexp.MustCompile(`\$([A-Za-z_][A-Za-z0-9_]*)`)
	metavarPlaceRe = regexp.MustCompile(`^__mv_(.+)__$`)
)

// parsePattern rewrites $NAME holes to parseable placeholder identifiers, then
// parses the fragment as a Go expression, statement, or declaration.
func parsePattern(pattern string) (ast.Node, error) {
	trimmed := strings.TrimSpace(pattern)
	rewritten := metavarSyntax.ReplaceAllString(trimmed, "__mv_${1}__")

	// 1. Try parsing as an expression first.
	if expr, err := parser.ParseExpr(rewritten); err == nil {
		return expr, nil
	}

	// 2. Try parsing as top-level declarations.
	fileSrc := "package p\n" + rewritten
	fset := token.NewFileSet()
	if file, err := parser.ParseFile(fset, "pattern.go", fileSrc, 0); err == nil && len(file.Decls) > 0 {
		if len(file.Decls) == 1 {
			return file.Decls[0], nil
		}
		return file, nil
	}

	// 3. Try parsing as statement(s) wrapped in a function body.
	stmtSrc := "package p\nfunc _() {\n" + rewritten + "\n}"
	if file, err := parser.ParseFile(fset, "pattern.go", stmtSrc, 0); err == nil {
		if len(file.Decls) == 1 {
			if fd, ok := file.Decls[0].(*ast.FuncDecl); ok && fd.Body != nil && len(fd.Body.List) > 0 {
				if len(fd.Body.List) == 1 {
					if ds, ok := fd.Body.List[0].(*ast.DeclStmt); ok {
						return ds.Decl, nil
					}
					return fd.Body.List[0], nil
				}
				return fd.Body, nil
			}
		}
	}

	return nil, fmt.Errorf("astquery: bad pattern %q", pattern)
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

// Search parses src (a full Go file) and returns every subtree matching
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

	return searchNode(file, pat, fset), nil
}

// SearchFile parses the Go source file at filePath and returns all structural matches.
func SearchFile(filePath, pattern string) ([]Match, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	pat, err := parsePattern(pattern)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filePath, data, 0)
	if err != nil {
		return nil, fmt.Errorf("astquery: bad source %s: %w", filePath, err)
	}
	return searchNode(file, pat, fset), nil
}

// SearchDir walks dirPath and searches all .go files for matches of pattern.
// If maxMatches > 0, the search stops after finding maxMatches hits.
func SearchDir(dirPath, pattern string, maxMatches int) ([]FileMatch, error) {
	pat, err := parsePattern(pattern)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(dirPath)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		ms, err := SearchFile(dirPath, pattern)
		if err != nil {
			return nil, err
		}
		var fms []FileMatch
		for _, m := range ms {
			fms = append(fms, FileMatch{
				File:     dirPath,
				Line:     m.Pos.Line,
				Text:     m.Text,
				Bindings: m.Bindings,
			})
			if maxMatches > 0 && len(fms) >= maxMatches {
				break
			}
		}
		return fms, nil
	}

	var matches []FileMatch
	err = filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if strings.HasPrefix(name, ".") && name != "." {
				return filepath.SkipDir
			}
			if name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}
		fileMatches := searchNode(file, pat, fset)
		for _, m := range fileMatches {
			displayPath := path
			if rel, relErr := filepath.Rel(dirPath, path); relErr == nil && !strings.HasPrefix(rel, "..") {
				displayPath = filepath.ToSlash(rel)
			}
			matches = append(matches, FileMatch{
				File:     displayPath,
				Line:     m.Pos.Line,
				Text:     m.Text,
				Bindings: m.Bindings,
			})
			if maxMatches > 0 && len(matches) >= maxMatches {
				return fs.SkipAll
			}
		}
		return nil
	})
	if err != nil && err != fs.SkipAll {
		return nil, err
	}
	return matches, nil
}

func searchNode(root ast.Node, pat ast.Node, fset *token.FileSet) []Match {
	var out []Match
	ast.Inspect(root, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		switch p := pat.(type) {
		case ast.Expr:
			if expr, ok := n.(ast.Expr); ok {
				binds := map[string]string{}
				if matchExpr(p, expr, binds, fset) {
					out = append(out, Match{
						Pos:      fset.Position(expr.Pos()),
						EndPos:   fset.Position(expr.End()),
						Text:     nodeText(fset, expr),
						Bindings: binds,
					})
				}
			}
		case ast.Stmt:
			if stmt, ok := n.(ast.Stmt); ok {
				binds := map[string]string{}
				if matchStmt(p, stmt, binds, fset) {
					out = append(out, Match{
						Pos:      fset.Position(stmt.Pos()),
						EndPos:   fset.Position(stmt.End()),
						Text:     nodeText(fset, stmt),
						Bindings: binds,
					})
				}
			}
		case ast.Decl:
			if decl, ok := n.(ast.Decl); ok {
				binds := map[string]string{}
				if matchDecl(p, decl, binds, fset) {
					out = append(out, Match{
						Pos:      fset.Position(decl.Pos()),
						EndPos:   fset.Position(decl.End()),
						Text:     nodeText(fset, decl),
						Bindings: binds,
					})
				}
			}
		}
		return true
	})
	return out
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
		if !ok || !matchExpr(p.X, q.X, binds, fset) {
			return false
		}
		if name, isMeta := metavarName(p.Sel); isMeta {
			return bindMeta(name, q.Sel, binds, fset)
		}
		return p.Sel.Name == q.Sel.Name
	case *ast.IndexExpr:
		q, ok := n.(*ast.IndexExpr)
		return ok && matchExpr(p.X, q.X, binds, fset) && matchExpr(p.Index, q.Index, binds, fset)
	case *ast.IndexListExpr:
		q, ok := n.(*ast.IndexListExpr)
		if !ok || len(p.Indices) != len(q.Indices) {
			return false
		}
		if !matchExpr(p.X, q.X, binds, fset) {
			return false
		}
		for i := range p.Indices {
			if !matchExpr(p.Indices[i], q.Indices[i], binds, fset) {
				return false
			}
		}
		return true
	case *ast.SliceExpr:
		q, ok := n.(*ast.SliceExpr)
		if !ok || p.Slice3 != q.Slice3 {
			return false
		}
		if !matchExpr(p.X, q.X, binds, fset) {
			return false
		}
		if !matchOptExpr(p.Low, q.Low, binds, fset) || !matchOptExpr(p.High, q.High, binds, fset) {
			return false
		}
		if p.Slice3 && !matchOptExpr(p.Max, q.Max, binds, fset) {
			return false
		}
		return true
	case *ast.CompositeLit:
		q, ok := n.(*ast.CompositeLit)
		if !ok || len(p.Elts) != len(q.Elts) {
			return false
		}
		if p.Type != nil {
			if q.Type == nil || !matchExpr(p.Type, q.Type, binds, fset) {
				return false
			}
		} else if q.Type != nil {
			return false
		}
		for i := range p.Elts {
			if !matchExpr(p.Elts[i], q.Elts[i], binds, fset) {
				return false
			}
		}
		return true
	case *ast.KeyValueExpr:
		q, ok := n.(*ast.KeyValueExpr)
		return ok && matchExpr(p.Key, q.Key, binds, fset) && matchExpr(p.Value, q.Value, binds, fset)
	case *ast.TypeAssertExpr:
		q, ok := n.(*ast.TypeAssertExpr)
		if !ok || !matchExpr(p.X, q.X, binds, fset) {
			return false
		}
		return matchOptExpr(p.Type, q.Type, binds, fset)
	case *ast.Ellipsis:
		q, ok := n.(*ast.Ellipsis)
		if !ok {
			return false
		}
		return matchOptExpr(p.Elt, q.Elt, binds, fset)
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
	case *ast.ArrayType:
		q, ok := n.(*ast.ArrayType)
		return ok && matchOptExpr(p.Len, q.Len, binds, fset) && matchExpr(p.Elt, q.Elt, binds, fset)
	case *ast.MapType:
		q, ok := n.(*ast.MapType)
		return ok && matchExpr(p.Key, q.Key, binds, fset) && matchExpr(p.Value, q.Value, binds, fset)
	case *ast.ChanType:
		q, ok := n.(*ast.ChanType)
		return ok && p.Dir == q.Dir && matchExpr(p.Value, q.Value, binds, fset)
	case *ast.StructType:
		q, ok := n.(*ast.StructType)
		return ok && matchFieldList(p.Fields, q.Fields, binds, fset)
	case *ast.InterfaceType:
		q, ok := n.(*ast.InterfaceType)
		return ok && matchFieldList(p.Methods, q.Methods, binds, fset)
	case *ast.FuncType:
		q, ok := n.(*ast.FuncType)
		return ok && matchFuncType(p, q, binds, fset)
	default:
		return false
	}
}

func matchOptExpr(p, q ast.Expr, binds map[string]string, fset *token.FileSet) bool {
	if p == nil || q == nil {
		return p == nil && q == nil
	}
	return matchExpr(p, q, binds, fset)
}

// matchStmt unifies pattern statement pat against target statement n.
func matchStmt(pat, n ast.Stmt, binds map[string]string, fset *token.FileSet) bool {
	if pat == nil || n == nil {
		return pat == nil && n == nil
	}
	// A metavariable in statement position (parsed as an identifier ExprStmt).
	if es, ok := pat.(*ast.ExprStmt); ok {
		if id, ok := es.X.(*ast.Ident); ok {
			if name, isMeta := metavarName(id); isMeta {
				return bindMeta(name, n, binds, fset)
			}
		}
	}

	switch p := pat.(type) {
	case *ast.ReturnStmt:
		q, ok := n.(*ast.ReturnStmt)
		if !ok || len(p.Results) != len(q.Results) {
			return false
		}
		for i := range p.Results {
			if !matchExpr(p.Results[i], q.Results[i], binds, fset) {
				return false
			}
		}
		return true
	case *ast.AssignStmt:
		q, ok := n.(*ast.AssignStmt)
		if !ok || p.Tok != q.Tok || len(p.Lhs) != len(q.Lhs) || len(p.Rhs) != len(q.Rhs) {
			return false
		}
		for i := range p.Lhs {
			if !matchExpr(p.Lhs[i], q.Lhs[i], binds, fset) {
				return false
			}
		}
		for i := range p.Rhs {
			if !matchExpr(p.Rhs[i], q.Rhs[i], binds, fset) {
				return false
			}
		}
		return true
	case *ast.ExprStmt:
		q, ok := n.(*ast.ExprStmt)
		if !ok {
			return false
		}
		return matchExpr(p.X, q.X, binds, fset)
	case *ast.IfStmt:
		q, ok := n.(*ast.IfStmt)
		if !ok {
			return false
		}
		if !matchOptStmt(p.Init, q.Init, binds, fset) {
			return false
		}
		if !matchExpr(p.Cond, q.Cond, binds, fset) {
			return false
		}
		if !matchBlockStmt(p.Body, q.Body, binds, fset) {
			return false
		}
		if !matchOptStmt(p.Else, q.Else, binds, fset) {
			return false
		}
		return true
	case *ast.DeclStmt:
		q, ok := n.(*ast.DeclStmt)
		if !ok {
			return false
		}
		return matchDecl(p.Decl, q.Decl, binds, fset)
	case *ast.BlockStmt:
		q, ok := n.(*ast.BlockStmt)
		if !ok {
			return false
		}
		return matchBlockStmt(p, q, binds, fset)
	case *ast.ForStmt:
		q, ok := n.(*ast.ForStmt)
		if !ok {
			return false
		}
		if !matchOptStmt(p.Init, q.Init, binds, fset) {
			return false
		}
		if !matchOptExpr(p.Cond, q.Cond, binds, fset) {
			return false
		}
		if !matchOptStmt(p.Post, q.Post, binds, fset) {
			return false
		}
		return matchBlockStmt(p.Body, q.Body, binds, fset)
	case *ast.RangeStmt:
		q, ok := n.(*ast.RangeStmt)
		if !ok || p.Tok != q.Tok {
			return false
		}
		if !matchOptExpr(p.Key, q.Key, binds, fset) || !matchOptExpr(p.Value, q.Value, binds, fset) {
			return false
		}
		if !matchExpr(p.X, q.X, binds, fset) {
			return false
		}
		return matchBlockStmt(p.Body, q.Body, binds, fset)
	case *ast.IncDecStmt:
		q, ok := n.(*ast.IncDecStmt)
		return ok && p.Tok == q.Tok && matchExpr(p.X, q.X, binds, fset)
	case *ast.BranchStmt:
		q, ok := n.(*ast.BranchStmt)
		if !ok || p.Tok != q.Tok {
			return false
		}
		if p.Label != nil {
			return q.Label != nil && p.Label.Name == q.Label.Name
		}
		return q.Label == nil
	case *ast.DeferStmt:
		q, ok := n.(*ast.DeferStmt)
		return ok && matchExpr(p.Call, q.Call, binds, fset)
	case *ast.GoStmt:
		q, ok := n.(*ast.GoStmt)
		return ok && matchExpr(p.Call, q.Call, binds, fset)
	case *ast.SendStmt:
		q, ok := n.(*ast.SendStmt)
		return ok && matchExpr(p.Chan, q.Chan, binds, fset) && matchExpr(p.Value, q.Value, binds, fset)
	case *ast.EmptyStmt:
		_, ok := n.(*ast.EmptyStmt)
		return ok
	case *ast.SwitchStmt:
		q, ok := n.(*ast.SwitchStmt)
		if !ok {
			return false
		}
		if !matchOptStmt(p.Init, q.Init, binds, fset) || !matchOptExpr(p.Tag, q.Tag, binds, fset) {
			return false
		}
		return matchBlockStmt(p.Body, q.Body, binds, fset)
	case *ast.TypeSwitchStmt:
		q, ok := n.(*ast.TypeSwitchStmt)
		if !ok {
			return false
		}
		if !matchOptStmt(p.Init, q.Init, binds, fset) || !matchStmt(p.Assign, q.Assign, binds, fset) {
			return false
		}
		return matchBlockStmt(p.Body, q.Body, binds, fset)
	case *ast.CaseClause:
		q, ok := n.(*ast.CaseClause)
		if !ok || len(p.List) != len(q.List) || len(p.Body) != len(q.Body) {
			return false
		}
		for i := range p.List {
			if !matchExpr(p.List[i], q.List[i], binds, fset) {
				return false
			}
		}
		for i := range p.Body {
			if !matchStmt(p.Body[i], q.Body[i], binds, fset) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func matchOptStmt(p, q ast.Stmt, binds map[string]string, fset *token.FileSet) bool {
	if p == nil || q == nil {
		return p == nil && q == nil
	}
	return matchStmt(p, q, binds, fset)
}

func matchBlockStmt(p, q *ast.BlockStmt, binds map[string]string, fset *token.FileSet) bool {
	if p == nil || q == nil {
		return p == nil && q == nil
	}
	if len(p.List) != len(q.List) {
		return false
	}
	for i := range p.List {
		if !matchStmt(p.List[i], q.List[i], binds, fset) {
			return false
		}
	}
	return true
}

func matchDecl(pat, n ast.Decl, binds map[string]string, fset *token.FileSet) bool {
	if pat == nil || n == nil {
		return pat == nil && n == nil
	}
	if pGen, okP := pat.(*ast.GenDecl); okP {
		qGen, okQ := n.(*ast.GenDecl)
		if !okQ || pGen.Tok != qGen.Tok || len(pGen.Specs) != len(qGen.Specs) {
			return false
		}
		for i := range pGen.Specs {
			if !matchSpec(pGen.Specs[i], qGen.Specs[i], binds, fset) {
				return false
			}
		}
		return true
	}
	if pFunc, okP := pat.(*ast.FuncDecl); okP {
		qFunc, okQ := n.(*ast.FuncDecl)
		if !okQ {
			return false
		}
		if !matchFieldList(pFunc.Recv, qFunc.Recv, binds, fset) {
			return false
		}
		if !matchIdent(pFunc.Name, qFunc.Name, binds, fset) {
			return false
		}
		if !matchFuncType(pFunc.Type, qFunc.Type, binds, fset) {
			return false
		}
		if pFunc.Body != nil {
			if qFunc.Body == nil || !matchBlockStmt(pFunc.Body, qFunc.Body, binds, fset) {
				return false
			}
		}
		return true
	}
	return false
}

func matchIdent(p, q *ast.Ident, binds map[string]string, fset *token.FileSet) bool {
	if p == nil || q == nil {
		return p == nil && q == nil
	}
	if name, isMeta := metavarName(p); isMeta {
		return bindMeta(name, q, binds, fset)
	}
	return p.Name == q.Name
}

func matchFuncType(p, q *ast.FuncType, binds map[string]string, fset *token.FileSet) bool {
	if p == nil || q == nil {
		return p == nil && q == nil
	}
	if !matchFieldList(p.Params, q.Params, binds, fset) {
		return false
	}
	return matchFieldList(p.Results, q.Results, binds, fset)
}

func matchFieldList(p, q *ast.FieldList, binds map[string]string, fset *token.FileSet) bool {
	if p == nil || q == nil {
		return (p == nil || len(p.List) == 0) && (q == nil || len(q.List) == 0)
	}
	if len(p.List) != len(q.List) {
		return false
	}
	for i := range p.List {
		pf, qf := p.List[i], q.List[i]
		if len(pf.Names) != len(qf.Names) {
			return false
		}
		for j := range pf.Names {
			if !matchIdent(pf.Names[j], qf.Names[j], binds, fset) {
				return false
			}
		}
		if !matchOptExpr(pf.Type, qf.Type, binds, fset) {
			return false
		}
	}
	return true
}

func matchSpec(pat, n ast.Spec, binds map[string]string, fset *token.FileSet) bool {
	if pat == nil || n == nil {
		return pat == nil && n == nil
	}
	switch p := pat.(type) {
	case *ast.ValueSpec:
		q, ok := n.(*ast.ValueSpec)
		if !ok || len(p.Names) != len(q.Names) || len(p.Values) != len(q.Values) {
			return false
		}
		for i := range p.Names {
			if !matchExpr(p.Names[i], q.Names[i], binds, fset) {
				return false
			}
		}
		if !matchOptExpr(p.Type, q.Type, binds, fset) {
			return false
		}
		for i := range p.Values {
			if !matchExpr(p.Values[i], q.Values[i], binds, fset) {
				return false
			}
		}
		return true
	case *ast.TypeSpec:
		q, ok := n.(*ast.TypeSpec)
		if !ok || !matchExpr(p.Name, q.Name, binds, fset) {
			return false
		}
		return matchExpr(p.Type, q.Type, binds, fset)
	default:
		return false
	}
}

// bindMeta binds a metavariable hole. `_` is the anonymous wildcard: it matches any
// expression/node and binds nothing. A named hole binds to the target's source text on
// first sight and must match it identically on every later occurrence.
func bindMeta(name string, n ast.Node, binds map[string]string, fset *token.FileSet) bool {
	if name == "_" {
		return true
	}
	text := nodeText(fset, n)
	if prev, seen := binds[name]; seen {
		return prev == text
	}
	binds[name] = text
	binds["$"+name] = text
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
