package compute

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"testing"
)

// This CPU source/control-flow witness pins the optional interface and the
// read-only native predicate. It does not compile cgo or execute Vulkan.
// fak-test:runtime fast est=1ms lane=default
func TestV41StepBatchRecordingContract(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		file, name, want string
	}{
		{"v41_step_batch.go", "V41StepBatchBackend", `V41StepBatchBackend interface {
	BeginV41StepBatch() bool
	EndV41StepBatch() error
}`},
		{"v41_step_batch.go", "V41StepBatchRecordingBackend", `V41StepBatchRecordingBackend interface {
	V41StepBatchRecording() bool
}`},
		{"vulkan_v41_step_batch.go", "V41StepBatchRecording", `func (v *vulkanBackend) V41StepBatchRecording() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_v41_step_batch_active() != 0 && bool(C.fvk_batch_active()) && C.fvk_submission_status() == 0
}`},
	} {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, tc.file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		var found ast.Node
		matches := 0
		for _, decl := range file.Decls {
			switch decl := decl.(type) {
			case *ast.GenDecl:
				for _, spec := range decl.Specs {
					if typ, ok := spec.(*ast.TypeSpec); ok && typ.Name.Name == tc.name {
						found = typ
						matches++
					}
				}
			case *ast.FuncDecl:
				if decl.Name.Name == tc.name {
					found = decl
					matches++
				}
			}
		}
		if matches != 1 {
			t.Fatalf("%s: got %d declarations of %s, want exactly one", tc.file, matches, tc.name)
		}
		var got bytes.Buffer
		if err := format.Node(&got, fset, found); err != nil {
			t.Fatal(err)
		}
		if got.String() != tc.want {
			t.Fatalf("%s: %s changed its recording contract:\n%s\nwant:\n%s", tc.file, tc.name, &got, tc.want)
		}
	}
}
