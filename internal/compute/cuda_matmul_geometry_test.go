package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
// Unmeasured estimate. All tensors are metadata-only: no CUDA or large allocation.
func TestCUDAMatMulGeometryAdmission(t *testing.T) {
	t.Parallel()
	t.Run("refusals", func(t *testing.T) {
		cases := []struct {
			name string
			w    Tensor
			x    Tensor
			rows int
		}{
			{"weight-rank", Tensor{Dtype: F16, Shape: []int{8}}, Tensor{Dtype: F32, Shape: []int{8}}, 1},
			{"zero-rows", Tensor{Dtype: F16, Shape: []int{1, 8}}, Tensor{Dtype: F32, Shape: []int{0, 8}}, 0},
			{"negative-width", Tensor{Dtype: F16, Shape: []int{1, -8}}, Tensor{Dtype: F32, Shape: []int{8}}, 1},
			{"input-mismatch", Tensor{Dtype: F16, Shape: []int{1, 8}}, Tensor{Dtype: F32, Shape: []int{2, 7}}, 2},
			{"input-dtype", Tensor{Dtype: F16, Shape: []int{1, 8}}, Tensor{Dtype: F16, Shape: []int{8}}, 1},
			{"conversion-product", Tensor{Dtype: F16, Shape: []int{1, 32768}}, Tensor{Dtype: F32, Shape: []int{65536, 32768}}, 65536},
			{"conversion-roundup", Tensor{Dtype: F16, Shape: []int{1, math.MaxInt32 - 254}}, Tensor{Dtype: F32, Shape: []int{math.MaxInt32 - 254}}, 1},
			{"output-byte-overflow", Tensor{Dtype: F32, Shape: []int{math.MaxInt32, 1}}, Tensor{Dtype: F32, Shape: []int{math.MaxInt32}}, math.MaxInt32},
			{"input-shape-overflow", Tensor{Dtype: F32, Shape: []int{1, 1}}, Tensor{Dtype: F32, Shape: []int{math.MaxInt, math.MaxInt}}, 1},
		}
		if math.MaxInt > math.MaxInt32 {
			wide := int64(math.MaxInt32) + 1
			cases = append(cases, struct {
				name string
				w    Tensor
				x    Tensor
				rows int
			}{"c-int-narrowing", Tensor{Dtype: F32, Shape: []int{int(wide), 1}}, Tensor{Dtype: F32, Shape: []int{1}}, 1})
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				defer func() {
					got := recover()
					err, ok := got.(*CUDAOpError)
					if !ok || err.Op != "BatchedMatMul" || err.Site != "matmul-geometry" || err.Class != MemoryActivation {
						t.Fatalf("refusal = %#v, want typed geometry error", got)
					}
				}()
				requireCUDAMatMulGeometry("BatchedMatMul", tc.w, tc.x, tc.rows)
			})
		}
	})
	t.Run("valid-extents", func(t *testing.T) {
		for _, shape := range [][]int{{257, 192, 1}, {5120, 17408, 32}} {
			w := Tensor{Dtype: F16, Shape: shape[:2]}
			x := Tensor{Dtype: F32, Shape: []int{shape[2], shape[1]}}
			out, in := requireCUDAMatMulGeometry("BatchedMatMul", w, x, shape[2])
			if out != shape[0] || in != shape[1] {
				t.Fatal("valid dimensions changed")
			}
		}
		if math.MaxInt > math.MaxInt32 {
			// The F16 conversion limit is not a blanket limit on cuBLAS F32.
			requireCUDAMatMulGeometry("BatchedMatMul", Tensor{Dtype: F32, Shape: []int{1, 32768}}, Tensor{Dtype: F32, Shape: []int{65536, 32768}}, 65536)
			requireCUDAMatMulGeometry("MatMul", Tensor{Dtype: F16, Shape: []int{1, math.MaxInt32 - 255}}, Tensor{Dtype: F32, Shape: []int{math.MaxInt32 - 255}}, 1)
		}
	})
	t.Run("entrypoint-order", func(t *testing.T) {
		file, err := parser.ParseFile(token.NewFileSet(), "cuda.go", nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		seen := 0
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || (fn.Name.Name != "MatMul" && fn.Name.Name != "BatchedMatMul") {
				continue
			}
			seen++
			first, ok := fn.Body.List[0].(*ast.AssignStmt)
			if !ok || len(first.Rhs) != 1 {
				t.Fatalf("%s does not preflight before tracing/allocation/native calls", fn.Name.Name)
			}
			call, ok := first.Rhs[0].(*ast.CallExpr)
			if !ok {
				t.Fatalf("%s first statement is not a preflight call", fn.Name.Name)
			}
			name, ok := call.Fun.(*ast.Ident)
			if !ok || name.Name != "requireCUDAMatMulGeometry" {
				t.Fatalf("%s first call is not requireCUDAMatMulGeometry", fn.Name.Name)
			}
		}
		if seen != 2 {
			t.Fatalf("found %d matmul entrypoints, want 2", seen)
		}
	})
}
