package architest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIssueCandidateConstructorsDeclareProblemFrame prevents new in-memory issue
// producers from bypassing the canonical intake gate. Decode/review boundaries
// that receive a Candidate are not constructors and therefore do not match.
func TestIssueCandidateConstructorsDeclareProblemFrame(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	fset := token.NewFileSet()
	var missing []string
	for _, base := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, base), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			file, parseErr := parser.ParseFile(fset, path, nil, 0)
			if parseErr != nil {
				return parseErr
			}
			ast.Inspect(file, func(node ast.Node) bool {
				literal, ok := node.(*ast.CompositeLit)
				if !ok || !isIssuePolicyCandidate(literal.Type) {
					return true
				}
				for _, element := range literal.Elts {
					field, keyed := element.(*ast.KeyValueExpr)
					ident, named := field.Key.(*ast.Ident)
					if keyed && named && ident.Name == "ProblemFrame" {
						return true
					}
				}
				pos := fset.Position(literal.Pos())
				missing = append(missing, filepath.ToSlash(strings.TrimPrefix(pos.Filename, root+string(filepath.Separator)))+":"+itoa(pos.Line))
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("issuepolicy.Candidate constructors missing explicit ProblemFrame (declare a canonical frame; do not infer downstream): %s", strings.Join(missing, ", "))
	}
}

func isIssuePolicyCandidate(expr ast.Expr) bool {
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok || selector.Sel.Name != "Candidate" {
		return false
	}
	pkg, ok := selector.X.(*ast.Ident)
	return ok && pkg.Name == "issuepolicy"
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	i := len(digits)
	for value > 0 {
		i--
		digits[i] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[i:])
}
