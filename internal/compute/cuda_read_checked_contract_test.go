package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
// This source contract complements, never replaces, actual tagged Read tests.
func TestCUDAReadCheckedContract(t *testing.T) {
	src, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "cuda.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == "Read" && fn.Recv != nil {
			body = string(src[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
		}
	}
	if body == "" {
		t.Fatal("actual Read declaration missing")
	}
	for _, required := range []string{`AdmitDevice(c, "read-d2h")`, `status := cudaReadNative.copy(db.device,`, `if status != 0`, `c.faultLatch.Observe(DeviceFaultExecution, "read-d2h", status)`, `Op: "Read"`, `Site: "read-d2h"`, `Code: status`, `if db.device == 0`} {
		if !strings.Contains(body, required) {
			t.Errorf("Read missing %q", required)
		}
	}
	if strings.Contains(body, "C.fcuda_d2h") {
		t.Error("Read bypasses checked status seam")
	}
	if strings.Index(body, "if status != 0") > strings.Index(body, "atomic.AddUint64") {
		t.Error("fence promotion precedes checked failure")
	}
	var native string
	for _, d := range f.Decls {
		g, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, s := range g.Specs {
			v, ok := s.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for _, name := range v.Names {
				if name.Name == "cudaReadNative" {
					native = string(src[fs.Position(v.Pos()).Offset:fs.Position(v.End()).Offset])
				}
			}
		}
	}
	if !strings.Contains(native, "C.fcuda_d2h_on_checked") || strings.Contains(native, "C.fcuda_d2h_checked(") {
		t.Error("production Read must use explicit-device checked D2H, including device0")
	}
}
