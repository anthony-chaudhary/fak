package maturity

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Anatomy is a static structural readout for one Go package. Counts describe
// syntax, not runtime frequency or independently executable paths.
type Anatomy struct {
	Schema        string          `json:"schema"`
	Package       string          `json:"package"`
	Directory     string          `json:"directory"`
	Shape         AnatomyShape    `json:"shape"`
	Flow          AnatomyFlow     `json:"flow"`
	Outcomes      AnatomyOutcomes `json:"outcomes"`
	Contracts     AnatomyContract `json:"contracts"`
	Documentation AnatomyDocs     `json:"documentation"`
	Position      AnatomyPosition `json:"position"`
	Caveats       []string        `json:"caveats"`
}

type AnatomyShape struct {
	Files      int `json:"files"`
	TestFiles  int `json:"test_files"`
	Functions  int `json:"functions"`
	Statements int `json:"statements"`
}

type AnatomyFlow struct {
	DecisionPoints       int `json:"decision_points"`
	CyclomaticComplexity int `json:"cyclomatic_complexity"`
	MaximumFunction      int `json:"maximum_function_complexity"`
	MaximumNesting       int `json:"maximum_nesting"`
}

type AnatomyOutcomes struct {
	ReturnSites           int `json:"return_sites"`
	ErrorHandlingBranches int `json:"error_handling_branches"`
	ErrorExits            int `json:"error_exits"`
	SuccessExits          int `json:"success_exits"`
	AmbiguousExits        int `json:"ambiguous_exits"`
}

type AnatomyContract struct {
	GuardClauses       int `json:"guard_clauses"`
	Panics             int `json:"panics"`
	AssumptionComments int `json:"assumption_comments"`
	TODOs              int `json:"todos"`
}

type AnatomyDocs struct {
	ExportedSymbols   int  `json:"exported_symbols"`
	DocumentedExports int  `json:"documented_exports"`
	PackageDoc        bool `json:"package_doc"`
}

type AnatomyPosition struct {
	InternalDependencies []string `json:"internal_dependencies"`
	InternalDependents   []string `json:"internal_dependents"`
	CLIReachable         bool     `json:"cli_reachable"`
}

func AnalyzeAnatomy(root, target string) (Anatomy, error) {
	if root == "" {
		root = "."
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return Anatomy{}, err
	}
	dir := target
	if dir == "" {
		dir = "internal/maturity"
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(absRoot, filepath.FromSlash(dir))
	}
	dir, err = filepath.Abs(dir)
	if err != nil {
		return Anatomy{}, err
	}
	rel, err := filepath.Rel(absRoot, dir)
	if err != nil || strings.HasPrefix(rel, "..") {
		return Anatomy{}, fmt.Errorf("target must be inside root")
	}

	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(info os.FileInfo) bool { return strings.HasSuffix(info.Name(), ".go") }, parser.ParseComments)
	if err != nil {
		return Anatomy{}, err
	}
	if len(pkgs) == 0 {
		return Anatomy{}, fmt.Errorf("no Go package in %s", filepath.ToSlash(rel))
	}
	var names []string
	for name := range pkgs {
		if !strings.HasSuffix(name, "_test") {
			names = append(names, name)
		}
	}
	if len(names) == 0 {
		for name := range pkgs {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	pkg := pkgs[names[0]]
	a := Anatomy{Schema: "fak-maturity-anatomy/1", Package: filepath.ToSlash(rel), Directory: filepath.ToSlash(rel), Caveats: []string{
		"Static production-code syntax counts do not measure runtime frequency or independent executable paths.",
		"Success/error exits and assumption comments are conservative lexical classifications; ambiguous exits remain explicit.",
	}}
	deps := map[string]struct{}{}
	for filename, file := range pkg.Files {
		a.Shape.Files++
		if strings.HasSuffix(filename, "_test.go") {
			a.Shape.TestFiles++
			continue
		}
		if file.Doc != nil {
			a.Documentation.PackageDoc = true
		}
		for _, imp := range file.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			if strings.Contains(path, "/internal/") {
				deps[strings.TrimPrefix(path, "github.com/anthony-chaudhary/fak/")] = struct{}{}
			}
		}
		for _, cg := range file.Comments {
			text := strings.ToLower(cg.Text())
			a.Contracts.TODOs += strings.Count(text, "todo")
			for _, word := range []string{"assume", "assumption", "expect", "invariant", "must "} {
				if strings.Contains(text, word) {
					a.Contracts.AssumptionComments++
					break
				}
			}
		}
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				a.Shape.Functions++
				if d.Name.IsExported() {
					a.Documentation.ExportedSymbols++
					if d.Doc != nil {
						a.Documentation.DocumentedExports++
					}
				}
				if d.Body != nil {
					analyzeFunction(&a, d.Body)
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					var ids []*ast.Ident
					switch s := spec.(type) {
					case *ast.TypeSpec:
						ids = []*ast.Ident{s.Name}
					case *ast.ValueSpec:
						ids = s.Names
					}
					for _, id := range ids {
						if id.IsExported() {
							a.Documentation.ExportedSymbols++
							if d.Doc != nil {
								a.Documentation.DocumentedExports++
							}
						}
					}
				}
			}
		}
	}
	for dep := range deps {
		a.Position.InternalDependencies = append(a.Position.InternalDependencies, dep)
	}
	sort.Strings(a.Position.InternalDependencies)
	a.Position.InternalDependents, a.Position.CLIReachable = anatomyPosition(absRoot, a.Package)
	return a, nil
}

func analyzeFunction(a *Anatomy, body *ast.BlockStmt) {
	complexity, nesting := 1, 0
	ast.Inspect(body, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if _, ok := n.(ast.Stmt); ok {
			a.Shape.Statements++
		}
		switch x := n.(type) {
		case *ast.IfStmt:
			a.Flow.DecisionPoints++
			complexity++
			if exprMentionsError(x.Cond) {
				a.Outcomes.ErrorHandlingBranches++
			}
			if blockTerminates(x.Body) {
				a.Contracts.GuardClauses++
			}
		case *ast.ForStmt, *ast.RangeStmt:
			a.Flow.DecisionPoints++
			complexity++
		case *ast.CaseClause:
			if x.List != nil {
				a.Flow.DecisionPoints++
				complexity++
			}
		case *ast.CommClause:
			if x.Comm != nil {
				a.Flow.DecisionPoints++
				complexity++
			}
		case *ast.BinaryExpr:
			if x.Op == token.LAND || x.Op == token.LOR {
				a.Flow.DecisionPoints++
				complexity++
			}
		case *ast.ReturnStmt:
			a.Outcomes.ReturnSites++
			switch classifyReturn(x) {
			case "error":
				a.Outcomes.ErrorExits++
			case "success":
				a.Outcomes.SuccessExits++
			default:
				a.Outcomes.AmbiguousExits++
			}
		case *ast.CallExpr:
			if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "panic" {
				a.Contracts.Panics++
			}
		}
		return true
	})
	var walkBlock func(*ast.BlockStmt, int)
	walkBlock = func(b *ast.BlockStmt, depth int) {
		if depth > nesting {
			nesting = depth
		}
		for _, st := range b.List {
			switch x := st.(type) {
			case *ast.IfStmt:
				walkBlock(x.Body, depth+1)
				if e, ok := x.Else.(*ast.BlockStmt); ok {
					walkBlock(e, depth+1)
				}
			case *ast.ForStmt:
				walkBlock(x.Body, depth+1)
			case *ast.RangeStmt:
				walkBlock(x.Body, depth+1)
			case *ast.SwitchStmt:
				walkBlock(x.Body, depth+1)
			case *ast.TypeSwitchStmt:
				walkBlock(x.Body, depth+1)
			case *ast.SelectStmt:
				walkBlock(x.Body, depth+1)
			}
		}
	}
	walkBlock(body, 0)
	a.Flow.CyclomaticComplexity += complexity
	if complexity > a.Flow.MaximumFunction {
		a.Flow.MaximumFunction = complexity
	}
	if nesting > a.Flow.MaximumNesting {
		a.Flow.MaximumNesting = nesting
	}
}

func exprMentionsError(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && (id.Name == "err" || strings.HasSuffix(strings.ToLower(id.Name), "error")) {
			found = true
		}
		return !found
	})
	return found
}
func blockTerminates(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	switch b.List[len(b.List)-1].(type) {
	case *ast.ReturnStmt, *ast.BranchStmt:
		return true
	}
	return false
}
func classifyReturn(r *ast.ReturnStmt) string {
	if len(r.Results) == 0 {
		return "success"
	}
	errorish, nilish := false, false
	for _, e := range r.Results {
		s := strings.ToLower(exprText(e))
		if s == "nil" {
			nilish = true
		}
		if strings.Contains(s, "err") || strings.Contains(s, "error") {
			errorish = true
		}
	}
	if errorish {
		return "error"
	}
	if nilish {
		return "success"
	}
	return "ambiguous"
}
func exprText(e ast.Expr) string               { var b strings.Builder; _ = formatNode(&b, e); return b.String() }
func formatNode(w io.Writer, n ast.Node) error { return printer.Fprint(w, token.NewFileSet(), n) }

func anatomyPosition(root, target string) ([]string, bool) {
	graph := internalImportGraph(filepath.Join(root, "internal"))
	trim := strings.TrimPrefix(target, "internal/")
	var dependents []string
	for from, tos := range graph {
		if _, ok := tos[trim]; ok {
			dependents = append(dependents, "internal/"+from)
		}
	}
	sort.Strings(dependents)
	reachable := scanReachable(root)
	_, cli := reachable[trim]
	return dependents, cli
}

func RenderAnatomyText(w io.Writer, a Anatomy) {
	fmt.Fprintf(w, "MATURITY ANATOMY  %s\n", a.Package)
	fmt.Fprintf(w, "shape          files=%d test_files=%d functions=%d statements=%d\n", a.Shape.Files, a.Shape.TestFiles, a.Shape.Functions, a.Shape.Statements)
	fmt.Fprintf(w, "flow           decisions=%d cyclomatic=%d max_function=%d max_nesting=%d\n", a.Flow.DecisionPoints, a.Flow.CyclomaticComplexity, a.Flow.MaximumFunction, a.Flow.MaximumNesting)
	fmt.Fprintf(w, "outcomes       returns=%d success=%d error=%d ambiguous=%d error_branches=%d\n", a.Outcomes.ReturnSites, a.Outcomes.SuccessExits, a.Outcomes.ErrorExits, a.Outcomes.AmbiguousExits, a.Outcomes.ErrorHandlingBranches)
	fmt.Fprintf(w, "contracts      guards=%d panics=%d assumption_comments=%d todos=%d\n", a.Contracts.GuardClauses, a.Contracts.Panics, a.Contracts.AssumptionComments, a.Contracts.TODOs)
	fmt.Fprintf(w, "documentation exported=%d documented=%d package_doc=%t\n", a.Documentation.ExportedSymbols, a.Documentation.DocumentedExports, a.Documentation.PackageDoc)
	fmt.Fprintf(w, "position       dependencies=%d dependents=%d cli_reachable=%t\n", len(a.Position.InternalDependencies), len(a.Position.InternalDependents), a.Position.CLIReachable)
	fmt.Fprintln(w, "note           static production-code counts; outcomes and assumptions are conservative lexical classifications")
}

func EncodeAnatomyJSON(w io.Writer, a Anatomy) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(a)
}
