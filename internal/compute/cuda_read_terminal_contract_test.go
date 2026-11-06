package compute

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
func TestCUDAReadTerminalAdmissionContract(t *testing.T) {
	source, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if err := cudaReadTerminalPrefix(source); err != nil {
		t.Fatal(err)
	}
	bridge, err := os.ReadFile("cuda_v41_checked_stream.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(bridge, []byte("//go:build cuda && cgo\n")) {
		t.Fatal("shared terminal query must be linked in every CUDA profile")
	}
	if _, err := os.Stat("cuda_v41_checked_stream_off.go"); !os.IsNotExist(err) {
		t.Fatal("a profile-dependent no-op terminal bridge is forbidden")
	}
}

// The direct top-level prefix excludes delayed, conditional and missing guards.
// The host view and existing invalid-buffer error keep their original precedence.
func cudaReadTerminalPrefix(source []byte) error {
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "cuda.go", source, 0)
	if err != nil {
		return err
	}
	var read *ast.FuncDecl
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == "readLocked" && fn.Recv != nil {
			if read != nil {
				return fmt.Errorf("duplicate readLocked")
			}
			read = fn
		}
	}
	want := []string{
		"if hb, ok := t.buf.(*hostBuf); ok {\n\treturn hb.f32\n}",
		"db := t.buf.(*cudaBuf)",
		"if err := db.invalidStateError(\"buffer\"); err != nil {\n\tpanic(err)\n}",
		"cudaV41CheckedSubmit()",
		"cudaV41CompressorSubmitLocked(db)",
		"pending := cudaV41CompressorReadBeginLocked(c)",
	}
	if read == nil || read.Body == nil || len(read.Body.List) < len(want) {
		return fmt.Errorf("readLocked admission prefix missing")
	}
	for i, expected := range want {
		var actual strings.Builder
		if err := format.Node(&actual, token.NewFileSet(), read.Body.List[i]); err != nil {
			return err
		}
		if actual.String() != expected {
			return fmt.Errorf("readLocked statement %d: got %s; want %s", i, actual.String(), expected)
		}
	}
	return nil
}

// fak-test:runtime fast est=2ms lane=default
func TestCUDAReadTerminalContractRejectsBypasses(t *testing.T) {
	const source = `package compute
func (c *cudaBackend) readLocked(t Tensor) []float32 {
 if hb, ok := t.buf.(*hostBuf); ok { return hb.f32 }
 db := t.buf.(*cudaBuf)
 if err := db.invalidStateError("buffer"); err != nil { panic(err) }
 cudaV41CheckedSubmit()
 cudaV41CompressorSubmitLocked(db)
 pending := cudaV41CompressorReadBeginLocked(c)
 return nil
}`
	if err := cudaReadTerminalPrefix([]byte(source)); err != nil {
		t.Fatal(err)
	}
	for name, replacement := range map[string]string{
		"missing": "", "deferred": "defer cudaV41CheckedSubmit()",
		"conditional": "if c.faultLatch != nil { cudaV41CheckedSubmit() }",
		"after-hooks": "cudaV41CompressorSubmitLocked(db); cudaV41CheckedSubmit()",
	} {
		t.Run(name, func(t *testing.T) {
			mutant := strings.Replace(source, "cudaV41CheckedSubmit()", replacement, 1)
			if err := cudaReadTerminalPrefix([]byte(mutant)); err == nil {
				t.Fatal("contract accepted a bypass")
			}
		})
	}
	if err := cudaReadTerminalPrefix([]byte("package compute")); err == nil {
		t.Fatal("contract accepted a missing function")
	}
}
