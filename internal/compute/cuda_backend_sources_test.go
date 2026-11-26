package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// parseCUDABackendSources returns the declarations of every file that defines
// cudaBackend, so a source contract finds a method in whichever file holds it.
// The caller's fset must be the one used to render the returned nodes.
func parseCUDABackendSources(t *testing.T, fset *token.FileSet) *ast.File {
	t.Helper()
	merged := &ast.File{}
	for _, name := range []string{"cuda.go", "cuda_residency.go"} {
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		merged.Decls = append(merged.Decls, file.Decls...)
	}
	return merged
}
