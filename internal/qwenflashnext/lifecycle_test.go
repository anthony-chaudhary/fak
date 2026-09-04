package qwenflashnext

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

// Invariant: Qwen chat parsing must extract thinking analysis and final responses separated by canonical stop tokens.
// Guard: ParseResponse extracts reasoning blocks without leaking thought tags into final outputs.

func TestQwenFlashNextLifecycle(t *testing.T) {
	t.Parallel()

	resp := "<think>\nreasoning block\n</think>\n\nFinal answer.<|im_end|>"
	parsed, err := ParseResponse(resp)
	if err != nil {
		t.Fatalf("ParseResponse failed: %v", err)
	}
	if parsed.Analysis != "reasoning block" || parsed.Final != "Final answer." || !parsed.Stopped {
		t.Fatalf("unexpected parsed response: %+v", parsed)
	}
}

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

// TestQwenFlashNextMaturityDocumentationAndContracts verifies that internal/qwenflashnext
// meets debtlane maturity requirements: substantive contract comments, at least 80% exported
// symbol documentation coverage, and verified benchmark definitions.
func TestQwenFlashNextMaturityDocumentationAndContracts(t *testing.T) {
	files := []string{"chat.go"}
	fset := token.NewFileSet()

	for _, filename := range files {
		path := filepath.Join(".", filename)
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("failed to read %s: %v", path, err)
		}

		node, err := parser.ParseFile(fset, path, content, parser.ParseComments)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", path, err)
		}

		// Verify contract comments presence
		contractCommentsCount := 0
		for _, cg := range node.Comments {
			if testIsSubstantiveContractComment(cg) {
				contractCommentsCount++
			}
		}
		if contractCommentsCount == 0 {
			t.Errorf("%s: expected at least one substantive contract comment, got none", filename)
		}

		// Count exported symbols and documentation
		exported := 0
		documented := 0
		var undocumented []string

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					exported++
					if testIsSubstantiveDoc(d.Name.Name, d.Doc) {
						documented++
					} else {
						undocumented = append(undocumented, d.Name.Name)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							exported++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if testIsSubstantiveDoc(s.Name.Name, doc) {
								documented++
							} else {
								undocumented = append(undocumented, s.Name.Name)
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								exported++
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if testIsSubstantiveDoc(name.Name, doc) {
									documented++
								} else {
									undocumented = append(undocumented, name.Name)
								}
							}
						}
					}
				}
			}
		}

		if exported > 0 {
			ratio := float64(documented) / float64(exported)
			if ratio < 0.80 {
				t.Errorf("%s: documented exports ratio %.2f < 0.80 (undocumented: %v)", filename, ratio, undocumented)
			}
		}
	}

	// Verify benchmark_test.go exists and defines BenchmarkEvaluatePrompt
	benchPath := filepath.Join(".", "benchmark_test.go")
	benchContent, err := os.ReadFile(benchPath)
	if err != nil {
		t.Fatalf("failed to read benchmark_test.go: %v", err)
	}
	benchNode, err := parser.ParseFile(fset, benchPath, benchContent, 0)
	if err != nil {
		t.Fatalf("failed to parse benchmark_test.go: %v", err)
	}

	hasEvaluateBenchmark := false
	for _, decl := range benchNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "BenchmarkEvaluatePrompt" {
			hasEvaluateBenchmark = true
		}
	}
	if !hasEvaluateBenchmark {
		t.Errorf("benchmark_test.go must define BenchmarkEvaluatePrompt")
	}
}
