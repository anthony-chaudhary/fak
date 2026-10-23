package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"strings"
	"testing"
)

// Metadata-only admission and production-callsite checks; no CUDA or large allocation.
// Unmeasured estimate.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAReadGeometryAdmission(t *testing.T) {
	t.Parallel()
	t.Run("refusals", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			dtype Dtype
			shape []int
			bytes int
		}{
			{"f16-storage-is-not-f32", F16, []int{4}, 8},
			{"quantized-storage-is-not-f32", Q8_0, []int{32}, 32},
			{"short-allocation", F32, []int{4}, 12},
			{"negative-shape", F32, []int{-1}, 4},
			{"negative-shape-after-zero", F32, []int{0, -1}, 0},
			{"negative-allocation", F32, []int{0}, -1},
			{"shape-product-overflow", F32, []int{math.MaxInt, math.MaxInt}, math.MaxInt},
			{"byte-size-overflow", F32, []int{math.MaxInt/4 + 1}, math.MaxInt},
		} {
			t.Run(tc.name, func(t *testing.T) {
				defer func() {
					got := recover()
					err, ok := got.(*CUDAOpError)
					if !ok || err.Op != "Read" || err.Site != "read-geometry" || err.Class != MemoryActivation {
						t.Fatalf("refusal = %#v, want typed Read geometry error", got)
					}
				}()
				requireCUDAReadGeometry(Tensor{Dtype: tc.dtype, Shape: tc.shape}, tc.bytes, MemoryActivation)
			})
		}
	})
	t.Run("valid-extents", func(t *testing.T) {
		for _, tc := range []struct {
			shape    []int
			bytes    int
			elements int
		}{
			{nil, 4, 1},
			{[]int{2, 3}, 24, 6},
			{[]int{2}, 32, 2},   // a view may use less than the backing allocation
			{[]int{0, 3}, 0, 0}, // empty pointerless views retain their row width
		} {
			elements, bytes := requireCUDAReadGeometry(Tensor{Dtype: F32, Shape: tc.shape}, tc.bytes, MemoryActivation)
			if elements != tc.elements || bytes != tc.elements*4 {
				t.Fatalf("shape %v: got %d elements/%d bytes, want %d/%d", tc.shape, elements, bytes, tc.elements, tc.elements*4)
			}
		}
	})
	t.Run("entrypoint-order", func(t *testing.T) {
		raw, err := os.ReadFile("cuda.go")
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, "cuda.go", raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		var body string
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if ok && fn.Name.Name == "Read" && fn.Recv != nil && fn.Body != nil {
				body = string(raw[fs.Position(fn.Body.Pos()).Offset:fs.Position(fn.Body.End()).Offset])
			}
		}
		previous := -1
		for _, step := range []string{
			"if hb, ok := t.buf.(*hostBuf); ok",
			"if !ok || db == nil || t.be != c",
			"db.invalidStateError(\"buffer\")",
			"elements, nbytes := requireCUDAReadGeometry(t, db.n, db.class)",
			"if db.ptr == nil && (nbytes != 0 || db.n != 0)",
			"make([]float32, elements)",
			"C.fcuda_d2h_on(",
			"C.fcuda_d2h(",
			"atomic.AddUint64(&c.fenceGen, 1)",
		} {
			at := strings.Index(body, step)
			if at <= previous {
				t.Fatalf("Read does not validate before allocation/copy/fence at %q", step)
			}
			previous = at
		}
		if strings.Count(body, "C.size_t(nbytes)") != 2 || strings.Contains(body, "t.Numel()") {
			t.Fatal("Read must use only the checked byte extent for both copy routes")
		}
	})
}
