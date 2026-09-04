package harnessresolve_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"
)

func testIsSubstantiveContractComment(cg *ast.CommentGroup) bool {
	if cg == nil {
		return false
	}
	text := strings.TrimSpace(cg.Text())
	if len(text) < 35 {
		return false
	}
	lower := strings.ToLower(text)

	hasContractMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "assumption:") ||
		strings.Contains(lower, "assumptions:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.Contains(lower, "precondition:") ||
		strings.Contains(lower, "postcondition:") ||
		strings.Contains(lower, "guard:")
	if !hasContractMarker {
		return false
	}

	words := strings.Fields(lower)
	if len(words) < 6 {
		return false
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.4 {
		return false
	}
	return true
}

func testIsFormulaicGamingComment(cg *ast.CommentGroup) (isFormulaic bool, isFiller bool) {
	if cg == nil {
		return false, false
	}
	text := strings.TrimSpace(cg.Text())
	lower := strings.ToLower(text)

	hasMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.HasPrefix(lower, "invariant") ||
		strings.HasPrefix(lower, "guard") ||
		strings.HasPrefix(lower, "contract") ||
		strings.HasPrefix(lower, "fail-closed")

	if !hasMarker {
		return false, false
	}

	words := strings.Fields(lower)
	if len(words) <= 3 {
		return true, true
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.25 || keywordCount >= 3 {
		return true, true
	}

	return true, false
}

func testSplitIdentifierWords(name string) map[string]bool {
	set := make(map[string]bool)
	set[strings.ToLower(name)] = true
	var curr strings.Builder
	for i, r := range name {
		if r == '_' || r == '-' {
			if curr.Len() > 0 {
				set[strings.ToLower(curr.String())] = true
				curr.Reset()
			}
			continue
		}
		if unicode.IsUpper(r) && i > 0 && curr.Len() > 0 {
			set[strings.ToLower(curr.String())] = true
			curr.Reset()
		}
		curr.WriteRune(r)
	}
	if curr.Len() > 0 {
		set[strings.ToLower(curr.String())] = true
	}
	return set
}

func testIsTautologicalDoc(name string, text string) bool {
	nameLower := strings.ToLower(name)
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	firstWord := strings.Trim(strings.ToLower(fields[0]), ":,.-()")
	if firstWord != nameLower && !strings.HasPrefix(strings.ToLower(text), nameLower) {
		return false
	}
	remainder := strings.TrimSpace(text[len(firstWord):])
	words := strings.FieldsFunc(remainder, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	fillers := map[string]bool{
		"is": true, "are": true, "does": true, "do": true, "returns": true, "return": true,
		"represents": true, "represent": true, "holds": true, "hold": true, "the": true,
		"a": true, "an": true, "of": true, "for": true, "to": true, "that": true, "which": true,
		"will": true, "can": true, "provides": true, "provide": true, "specifies": true,
		"specify": true, "defines": true, "define": true, "indicates": true, "indicate": true,
		"details": true, "detail": true, "records": true, "record": true, "encapsulates": true,
		"encapsulate": true, "captures": true, "capture": true, "contains": true, "contain": true,
	}

	nameParts := testSplitIdentifierWords(name)
	meaningfulWords := 0
	for _, w := range words {
		wl := strings.ToLower(w)
		if fillers[wl] || nameParts[wl] {
			continue
		}
		meaningfulWords++
	}
	return meaningfulWords < 2
}

func testIsSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil || len(doc.List) == 0 {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if len(text) < 12 {
		return false
	}
	return !testIsTautologicalDoc(name, text)
}

func testIsSubstantiveBenchmark(fn *ast.FuncDecl) bool {
	if fn.Body == nil || len(fn.Body.List) == 0 {
		return false
	}
	hasLoopOrRun := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if hasLoopOrRun {
			return false
		}
		switch stmt := n.(type) {
		case *ast.ForStmt:
			if stmt.Cond != nil && testReferencesName(stmt.Cond, "N") {
				hasLoopOrRun = true
			}
		case *ast.RangeStmt:
			if testReferencesName(stmt.X, "N") {
				hasLoopOrRun = true
			}
		case *ast.CallExpr:
			if sel, ok := stmt.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Run" {
				hasLoopOrRun = true
			}
		}
		return true
	})
	return hasLoopOrRun
}

func testReferencesName(expr ast.Expr, name string) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return true
	})
	return found
}

func TestHarnessResolveMaturityInvariants(t *testing.T) {
	fset := token.NewFileSet()
	nonTestFiles := []string{"doc.go", "harnessresolve.go"}

	totalContractComments := 0
	totalFormulaic := 0
	hasFiller := false
	totalExported := 0
	totalDocumented := 0

	for _, filename := range nonTestFiles {
		path := filepath.Join(".", filename)
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("failed to read %s: %v", path, err)
		}

		node, err := parser.ParseFile(fset, path, content, parser.ParseComments)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", path, err)
		}

		for _, cg := range node.Comments {
			if testIsSubstantiveContractComment(cg) {
				totalContractComments++
			}
			isForm, isFill := testIsFormulaicGamingComment(cg)
			if isForm {
				totalFormulaic++
			}
			if isFill {
				hasFiller = true
			}
		}

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					totalExported++
					if testIsSubstantiveDoc(d.Name.Name, d.Doc) {
						totalDocumented++
					} else {
						t.Errorf("%s: un-documented or non-substantive export func %s", filename, d.Name.Name)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							totalExported++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if testIsSubstantiveDoc(s.Name.Name, doc) {
								totalDocumented++
							} else {
								t.Errorf("%s: un-documented or non-substantive export type %s", filename, s.Name.Name)
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								totalExported++
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if testIsSubstantiveDoc(name.Name, doc) {
									totalDocumented++
								} else {
									t.Errorf("%s: un-documented or non-substantive export const/var %s", filename, name.Name)
								}
							}
						}
					}
				}
			}
		}
	}

	if totalContractComments < 3 {
		t.Errorf("expected at least 3 substantive contract comments in internal/harnessresolve, got %d", totalContractComments)
	}

	if totalExported == 0 {
		t.Fatal("expected exported symbols in internal/harnessresolve, found 0")
	}

	ratio := float64(totalDocumented) / float64(totalExported)
	if ratio < 0.95 {
		t.Errorf("documented exports ratio %.2f < 0.95 (documented %d / exported %d)", ratio, totalDocumented, totalExported)
	}

	// Verify formulaic gaming guard does not flag excess comments
	if totalFormulaic >= 3 || hasFiller {
		t.Errorf("formulaic comments triggered excess comments: count=%d filler=%v (want count < 3, filler=false)", totalFormulaic, hasFiller)
	}

	// Verify benchmark exists and is substantive
	benchPath := filepath.Join(".", "benchmark_test.go")
	benchContent, err := os.ReadFile(benchPath)
	if err != nil {
		t.Fatalf("failed to read benchmark_test.go: %v", err)
	}
	benchNode, err := parser.ParseFile(fset, benchPath, benchContent, 0)
	if err != nil {
		t.Fatalf("failed to parse benchmark_test.go: %v", err)
	}

	substantiveBenchmarks := 0
	hasBenchmarkHarnessResolve := false
	for _, decl := range benchNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && strings.HasPrefix(fn.Name.Name, "Benchmark") {
			if fn.Name.Name == "BenchmarkHarnessResolve" {
				hasBenchmarkHarnessResolve = true
			}
			if testIsSubstantiveBenchmark(fn) {
				substantiveBenchmarks++
			}
		}
	}
	if !hasBenchmarkHarnessResolve {
		t.Errorf("expected BenchmarkHarnessResolve in benchmark_test.go")
	}
	if substantiveBenchmarks < 1 {
		t.Errorf("expected at least 1 substantive benchmark with b.N loop, found %d", substantiveBenchmarks)
	}
}
