package compute

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// This checks the real CUDA admission/launch composition, not device execution.
// fak-test:runtime fast est=2ms lane=default
func TestCUDAHeadlessPrefixKeepsLogitsAdmission(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "cuda_qwen35_sequence.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	functions := make(map[string]*ast.FuncDecl)
	for _, declaration := range f.Decls {
		if fn, ok := declaration.(*ast.FuncDecl); ok {
			functions[fn.Name.Name] = fn
		}
	}
	text := func(n ast.Node) string {
		var out bytes.Buffer
		if err := format.Node(&out, fset, n); err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	preflight := functions["validateQwen35SequenceStoreRequestLocked"]
	if preflight == nil || functions["Qwen35SequenceHeadlessPrefixPath"] == nil {
		t.Fatal("missing CUDA headless prefix admission")
	}
	guarded := false
	for _, statement := range preflight.Body.List {
		branch, ok := statement.(*ast.IfStmt)
		if !ok || text(branch.Cond) != "req.Output.Buf() != nil || req.NeedLogits || req.NeedAllLogits" {
			continue
		}
		body := text(branch.Body)
		guarded = strings.Contains(body, "validateQwen35SequenceIQ2(req.Output,") && strings.Contains(body, "validateQwen35SequenceTensor(\"output\", req.Output,")
	}
	if !guarded {
		t.Fatal("missing full output validation for supplied heads or either logits mode")
	}
	entry := functions["Qwen35SequencePrefill"]
	if entry == nil {
		t.Fatal("missing CUDA sequence entry")
	}
	launches := map[string]bool{"output-head": false, "output-all-head": false}
	for _, statement := range entry.Body.List {
		branch, ok := statement.(*ast.IfStmt)
		if !ok {
			continue
		}
		condition, body := text(branch.Cond), text(branch.Body)
		if condition == "req.NeedLogits" && strings.Contains(body, "\"output-head\"") {
			launches["output-head"] = true
		}
		if condition == "req.NeedAllLogits" && strings.Contains(body, "\"output-all-head\"") {
			launches["output-all-head"] = true
		}
	}
	for name, found := range launches {
		if !found {
			t.Fatalf("lost conditional %s launch", name)
		}
	}
}
