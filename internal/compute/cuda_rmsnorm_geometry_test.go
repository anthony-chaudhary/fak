package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
// Unmeasured estimate. Metadata-only tensors require no CUDA or large allocation.
func TestCUDARMSNormGeometryAdmission(t *testing.T) {
	t.Parallel()
	t.Run("refusals", func(t *testing.T) {
		type testCase struct {
			name      string
			x, weight Tensor
		}
		cases := []testCase{
			{"incomplete-row", Tensor{Dtype: F32, Shape: []int{35}}, Tensor{Dtype: F32, Shape: []int{16}}},
			{"input-smaller-than-weight", Tensor{Dtype: F32, Shape: []int{10}}, Tensor{Dtype: F32, Shape: []int{16}}},
			{"empty-input", Tensor{Dtype: F32, Shape: []int{0}}, Tensor{Dtype: F32, Shape: []int{16}}},
			{"empty-weight", Tensor{Dtype: F32, Shape: []int{16}}, Tensor{Dtype: F32, Shape: []int{0}}},
			{"negative-input-dimensions", Tensor{Dtype: F32, Shape: []int{-2, -8}}, Tensor{Dtype: F32, Shape: []int{16}}},
			{"negative-weight-dimensions", Tensor{Dtype: F32, Shape: []int{16}}, Tensor{Dtype: F32, Shape: []int{-2, -8}}},
			{"input-product-overflow", Tensor{Dtype: F32, Shape: []int{65536, 65536, 65536, 65536}}, Tensor{Dtype: F32, Shape: []int{1}}},
			{"weight-product-overflow", Tensor{Dtype: F32, Shape: []int{1}}, Tensor{Dtype: F32, Shape: []int{65536, 65536, 65536, 65536}}},
			{"input-byte-overflow", Tensor{Dtype: F32, Shape: []int{math.MaxInt/F32.Bytes() + 1}}, Tensor{Dtype: F32, Shape: []int{1}}},
			{"input-dtype", Tensor{Dtype: F16, Shape: []int{16}}, Tensor{Dtype: F32, Shape: []int{16}}},
			{"weight-dtype", Tensor{Dtype: F32, Shape: []int{16}}, Tensor{Dtype: F16, Shape: []int{16}}},
		}
		if math.MaxInt > math.MaxInt32 {
			wideRows := int64(math.MaxInt32) + 1
			cases = append(cases,
				testCase{"row-narrowing", Tensor{Dtype: F32, Shape: []int{int(wideRows)}}, Tensor{Dtype: F32, Shape: []int{1}}},
				testCase{"weight-width-product", Tensor{Dtype: F32, Shape: []int{65536, 65536}}, Tensor{Dtype: F32, Shape: []int{65536, 65536}}},
				testCase{"width-loop-increment", Tensor{Dtype: F32, Shape: []int{math.MaxInt32 - 254}}, Tensor{Dtype: F32, Shape: []int{math.MaxInt32 - 254}}},
				testCase{"valid-row-width-byte-overflow", Tensor{Dtype: F32, Shape: []int{math.MaxInt32, math.MaxInt32 - 255}}, Tensor{Dtype: F32, Shape: []int{math.MaxInt32 - 255}}},
			)
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				defer func() {
					got := recover()
					err, ok := got.(*CUDAOpError)
					if !ok || err.Op != "RMSNorm" || err.Site != "rmsnorm-geometry" || err.Class != MemoryActivation {
						t.Fatalf("refusal = %#v, want typed RMSNorm geometry error", got)
					}
				}()
				requireCUDARMSNormGeometry(tc.x, tc.weight)
			})
		}
	})
	t.Run("valid-flattened-extents", func(t *testing.T) {
		type testCase struct {
			x, weight   []int
			rows, width int
		}
		cases := []testCase{
			{[]int{32}, []int{16}, 2, 16},
			{[]int{1, 3, 16}, []int{2, 8}, 3, 16},
			{[]int{4, 8}, []int{4, 4}, 2, 16},
			{nil, []int{}, 1, 1},
			{[]int{3}, nil, 3, 1},
		}
		if math.MaxInt > math.MaxInt32 {
			cases = append(cases,
				testCase{[]int{math.MaxInt32}, []int{1}, math.MaxInt32, 1},
				testCase{[]int{math.MaxInt32 - 255}, []int{math.MaxInt32 - 255}, 1, math.MaxInt32 - 255},
				testCase{[]int{65536, 65536}, []int{65536}, 65536, 65536},
			)
		}
		for _, tc := range cases {
			rows, width := requireCUDARMSNormGeometry(Tensor{Dtype: F32, Shape: tc.x}, Tensor{Dtype: F32, Shape: tc.weight})
			if rows != tc.rows || width != tc.width {
				t.Fatalf("x=%v weight=%v: geometry=(%d,%d), want (%d,%d)", tc.x, tc.weight, rows, width, tc.rows, tc.width)
			}
		}
	})
	t.Run("entrypoint-order", func(t *testing.T) {
		file, err := parser.ParseFile(token.NewFileSet(), "cuda.go", nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Name.Name != "RMSNorm" {
				continue
			}
			first, ok := fn.Body.List[0].(*ast.AssignStmt)
			if !ok || len(first.Rhs) != 1 {
				t.Fatal("RMSNorm does not preflight before allocation/native calls")
			}
			call, ok := first.Rhs[0].(*ast.CallExpr)
			if !ok {
				t.Fatal("RMSNorm first statement is not a preflight call")
			}
			name, ok := call.Fun.(*ast.Ident)
			if !ok || name.Name != "requireCUDARMSNormGeometry" {
				t.Fatal("RMSNorm first call is not requireCUDARMSNormGeometry")
			}
			return
		}
		t.Fatal("RMSNorm entrypoint not found")
	})
}
