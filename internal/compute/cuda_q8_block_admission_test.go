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
// Unmeasured estimate. Executes production admission and checks its upload
// wiring without CUDA, allocation, or subprocess compilation.
func TestCUDAQ8ResidentBlockAlignmentAdmission(t *testing.T) {
	t.Parallel()
	for _, block := range []int{1, 3, 6} {
		var got any
		func() {
			defer func() { got = recover() }()
			requireCUDAQ8BlockAlignment(block)
		}()
		err, ok := got.(*CUDAOpError)
		if !ok || err.Op != "Upload" || err.Site != "upload-q8-resident" || err.Class != MemoryWeights || !strings.Contains(err.Msg, "divisible by 4") {
			t.Fatalf("block %d refusal = %#v, want typed DP4A alignment error", block, got)
		}
	}
	for _, block := range []int{4, 16, 32} {
		requireCUDAQ8BlockAlignment(block)
	}

	raw, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "cuda.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Name.Name == "uploadQ8Resident" && fn.Recv != nil && fn.Body != nil {
			body = string(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
		}
	}
	previous := -1
	for _, step := range []string{
		"if blk <= 0 || in%blk != 0",
		"requireCUDAQ8BlockAlignment(blk)",
		"codes := hb.I8()",
		"if len(codes) != 0 && len(codes) != out*in",
		"if len(scales) != out*nblk",
		"uploadCache[",
		"c.devQ8(",
		"cudaUploadQuantPayload(",
	} {
		at := strings.Index(body, step)
		if at <= previous {
			t.Fatalf("resident Q8 upload lost admission order at %q", step)
		}
		previous = at
	}
}
