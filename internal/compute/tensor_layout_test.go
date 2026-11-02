package compute

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"slices"
	"testing"
)

func tensorLayoutFixture(be *cpuBackend, dtype Dtype) (Tensor, Tensor, Tensor) {
	const width = 32
	weights, codes := make([]float32, 2*width), make([]int8, 2*width)
	input, panel := make([]float32, width), make([]float32, 2*width)
	for i := range width {
		weights[i], weights[width+i] = float32(i+1), -float32(i+1)
		codes[i], codes[width+i] = int8(i+1), -int8(i+1)
		input[i], panel[i], panel[width+i] = 127, 127, -127
	}
	w := NewF32(be, []int{2, width}, weights)
	if dtype == Q8_0 {
		w = NewQ8(be, []int{2, width}, codes, []float32{1, 1}, width)
	}
	return w, NewF32(be, []int{width}, input), NewF32(be, []int{2, width}, panel)
}

func tensorLayoutCall(fn func() error) (err error) {
	defer func() {
		if value := recover(); value != nil {
			if failure, ok := value.(error); ok {
				err = failure
			} else {
				err = fmt.Errorf("non-error panic: %v", value)
			}
		}
	}()
	return fn()
}

// fak-test:justify why=regression when=changed:internal/compute/**
// fak-test:runtime fast est=1s
func TestTensorLayoutAdmission(t *testing.T) {
	be := &cpuBackend{}
	accepted, total := 0, 0
	for _, dtype := range []Dtype{F32, Q8_0} {
		for _, layout := range []Layout{ColMajor, Tiled, Layout(255)} {
			w, x, panel := tensorLayoutFixture(be, dtype)
			badW, badX, badPanel := w, x, panel
			badW.Layout, badX.Layout, badPanel.Layout = layout, layout, layout
			cases := []struct {
				name string
				run  func() error
			}{
				{"upload-weight", func() error { be.Upload(badW, dtype); return nil }},
				{"matmul-weight", func() error { be.MatMul(badW, x); return nil }},
				{"matmul-input", func() error { be.MatMul(w, badX); return nil }},
				{"batched-weight", func() error { be.BatchedMatMul(badW, panel, 2); return nil }},
				{"batched-input", func() error { be.BatchedMatMul(w, badPanel, 2); return nil }},
			}
			if dtype == Q8_0 {
				cases = append(cases,
					struct {
						name string
						run  func() error
					}{"prepare-weight", func() error { _, err := be.PrepareMatMul(badW, x); return err }},
					struct {
						name string
						run  func() error
					}{"prepare-input", func() error { _, err := be.PrepareMatMul(w, badX); return err }},
				)
			}
			for _, tc := range cases {
				total++
				t.Run(fmt.Sprintf("%s/layout-%d/%s", dtype, layout, tc.name), func(t *testing.T) {
					err := tensorLayoutCall(tc.run)
					if err == nil {
						accepted++
					}
					if !errors.Is(err, ErrUnsupportedLayout) {
						t.Errorf("expected unsupported-layout identity, got %v", err)
					}
				})
			}
		}
	}
	t.Logf("layout-admission metric: silently_accepted=%d invalid_cases=%d", accepted, total)
	if total != 36 {
		t.Fatalf("layout-admission matrix changed: cases=%d", total)
	}
}

// fak-test:justify why=guard when=changed:internal/compute/** twin=TestTensorLayoutAdmission
// fak-test:runtime fast est=1s
func TestTensorLayoutRowMajorNumerics(t *testing.T) {
	be := &cpuBackend{}
	for _, dtype := range []Dtype{F32, Q8_0} {
		t.Run(dtype.String(), func(t *testing.T) {
			w, x, panel := tensorLayoutFixture(be, dtype)
			uploaded := be.Upload(w, dtype)
			if uploaded.buf != w.buf || uploaded.Layout != RowMajor {
				t.Fatal("row-major upload changed resident storage or layout")
			}
			if got := be.f32(be.MatMul(uploaded, x)); !slices.Equal(got, []float32{67056, -67056}) {
				t.Fatalf("matmul result=%v", got)
			}
			if got := be.f32(be.BatchedMatMul(uploaded, panel, 2)); !slices.Equal(got, []float32{67056, -67056, -67056, 67056}) {
				t.Fatalf("batched result=%v", got)
			}
			if dtype == Q8_0 {
				if _, err := be.PrepareMatMul(w, x); err != nil {
					t.Fatalf("prepare row-major Q8: %v", err)
				}
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/compute/**
// fak-test:runtime fast est=1s
func TestTensorLayoutColumnMajorPhysicalBytes(t *testing.T) {
	be := &cpuBackend{}
	raw := []float32{1, 4, 2, 5, 3, 6}
	x := NewF32(be, []int{3}, []float32{1, 2, 3})
	w := NewF32(be, []int{2, 3}, raw)
	if got := be.f32(be.MatMul(w, x)); !slices.Equal(got, []float32{15, 29}) {
		t.Fatalf("row-major physical-byte control=%v", got)
	}
	w.Layout = ColMajor
	if err := tensorLayoutCall(func() error { be.MatMul(w, x); return nil }); !errors.Is(err, ErrUnsupportedLayout) {
		t.Fatalf("column-major physical storage requires rejection, got %v", err)
	}
	packed := make([]float32, len(raw))
	for row := range 2 {
		for col := range 3 {
			packed[row*3+col] = raw[col*2+row]
		}
	}
	if got := be.f32(be.MatMul(NewF32(be, w.Shape, packed), x)); !slices.Equal(got, []float32{14, 32}) {
		t.Fatalf("explicitly packed column-major oracle=%v", got)
	}
}

// fak-test:justify why=guard when=changed:internal/compute/** twin=TestTensorLayoutAdmission
// fak-test:runtime fast est=1s
func TestTensorLayoutCUDAUploadPackingContract(t *testing.T) {
	be := &cpuBackend{}
	w, _, _ := tensorLayoutFixture(be, F32)
	for _, tc := range []struct {
		name   string
		layout Layout
		dtype  Dtype
		as     Dtype
		class  MemoryClass
		shape  []int
		valid  bool
	}{
		{"row-major", RowMajor, Q8_0, Q8_0, MemoryOffload, []int{2, 32}, true},
		{"existing-load-time-f16-pack", ColMajor, F32, F16, MemoryWeights, []int{2, 32}, true},
		{"col-major-no-pack", ColMajor, F32, F32, MemoryWeights, []int{2, 32}, false},
		{"col-major-quantized", ColMajor, Q8_0, F16, MemoryWeights, []int{2, 32}, false},
		{"col-major-offload", ColMajor, F32, F16, MemoryOffload, []int{2, 32}, false},
		{"col-major-vector", ColMajor, F32, F16, MemoryWeights, []int{32}, false},
		{"tiled", Tiled, F32, F16, MemoryWeights, []int{2, 32}, false},
		{"unknown", Layout(255), F32, F16, MemoryWeights, []int{2, 32}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := w
			source.Layout, source.Dtype, source.Shape = tc.layout, tc.dtype, tc.shape
			err := checkCUDAUploadLayout("test upload", source, tc.as, tc.class)
			if tc.valid {
				if err != nil {
					t.Fatalf("supported upload packing rejected: %v", err)
				}
			} else if !errors.Is(err, ErrUnsupportedLayout) {
				t.Fatalf("unsupported upload packing lost error identity: %v", err)
			}
		})
	}
}

func tensorLayoutMethod(t *testing.T, filename, backend, method string) *ast.FuncDecl {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), filename, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", filename, err)
	}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != method || fn.Recv == nil {
			continue
		}
		pointer, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
		if name, named := tensorLayoutPointerName(pointer); ok && named && name == backend {
			return fn
		}
	}
	t.Fatalf("missing %s.%s in %s", backend, method, filename)
	return nil
}

func tensorLayoutPointerName(pointer *ast.StarExpr) (string, bool) {
	if pointer == nil {
		return "", false
	}
	name, ok := pointer.X.(*ast.Ident)
	if !ok {
		return "", false
	}
	return name.Name, true
}

func tensorLayoutRequireGuard(t *testing.T, statement ast.Stmt, name string) {
	t.Helper()
	first, ok := statement.(*ast.ExprStmt)
	if !ok {
		t.Fatal("upload does not admit layout before its first operation")
	}
	call, ok := first.X.(*ast.CallExpr)
	if !ok {
		t.Fatal("first upload operation is not layout admission")
	}
	guard, ok := call.Fun.(*ast.Ident)
	if !ok || guard.Name != name || len(call.Args) < 2 {
		t.Fatal("upload layout admission is missing before locks, cache or allocation")
	}
	input, ok := call.Args[1].(*ast.Ident)
	if !ok || input.Name != "t" {
		t.Fatal("upload guard does not receive the source tensor")
	}
	if name == "requireCUDAUploadLayout" {
		for i, name := range []string{"as", "class"} {
			if len(call.Args) <= i+2 {
				t.Fatal("CUDA guard lacks upload conversion context")
			}
			argument, ok := call.Args[i+2].(*ast.Ident)
			if !ok || argument.Name != name {
				t.Fatal("CUDA guard does not receive the original upload conversion context")
			}
		}
	}
}

// This guards tagged source wiring; the CPU test above executes the shared
// rejection semantics. It does not attest that a GPU or its packing kernel ran.
// fak-test:justify why=guard when=changed:internal/compute/** twin=TestTensorLayoutAdmission
// fak-test:runtime fast est=1s
func TestTensorLayoutGPUUploadAdmissionBeforeSideEffects(t *testing.T) {
	for _, tc := range []struct{ filename, backend, method, guard string }{
		{"cuda.go", "cudaBackend", "uploadClass", "requireCUDAUploadLayout"},
		{"metal.go", "metalBackend", "uploadClass", "requireRowMajor"},
		{"vulkan.go", "vulkanBackend", "uploadClass", "requireRowMajor"},
	} {
		t.Run(tc.backend+"/"+tc.method, func(t *testing.T) {
			fn := tensorLayoutMethod(t, tc.filename, tc.backend, tc.method)
			if len(fn.Body.List) == 0 {
				t.Fatal("empty upload implementation")
			}
			tensorLayoutRequireGuard(t, fn.Body.List[0], tc.guard)
		})
	}
	t.Run("cudaBackend/uploadAsyncShape", func(t *testing.T) {
		fn := tensorLayoutMethod(t, "cuda_async_upload.go", "cudaBackend", "uploadAsyncShape")
		selection, ok := fn.Body.List[0].(*ast.SwitchStmt)
		if !ok || len(selection.Body.List) != 2 {
			t.Fatal("async dispatcher must preserve its two transfer cases and synchronous fallback")
		}
		for _, statement := range selection.Body.List {
			branch := statement.(*ast.CaseClause)
			if len(branch.Body) < 2 {
				t.Fatal("async transfer case has no admission")
			}
			tensorLayoutRequireGuard(t, branch.Body[0], "requireCUDAUploadLayout")
		}
		last, ok := fn.Body.List[len(fn.Body.List)-1].(*ast.ReturnStmt)
		if !ok || len(last.Results) != 3 {
			t.Fatal("async dispatcher lost unsupported-shape fallback")
		}
		matched, ok := last.Results[2].(*ast.Ident)
		if !ok || matched.Name != "false" {
			t.Fatal("unmatched async shapes cannot reach the guarded synchronous upload")
		}
		for _, route := range []struct{ method, fallback string }{{"UploadAsync", "Upload"}, {"UploadAsyncClass", "UploadClass"}} {
			fn := tensorLayoutMethod(t, "cuda_async_upload.go", "cudaBackend", route.method)
			last, ok := fn.Body.List[len(fn.Body.List)-1].(*ast.ReturnStmt)
			if !ok || len(last.Results) == 0 {
				t.Fatal("async public entrypoint lost synchronous fallback")
			}
			call, ok := last.Results[0].(*ast.CallExpr)
			if !ok {
				t.Fatal("async fallback does not call the synchronous entrypoint")
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || selector.Sel.Name != route.fallback {
				t.Fatal("async fallback bypasses guarded synchronous admission")
			}
		}
	})
	t.Run("cudaBackend/UploadRank", func(t *testing.T) {
		fn := tensorLayoutMethod(t, "cuda_collective.go", "cudaBackend", "UploadRank")
		first, ok := fn.Body.List[0].(*ast.IfStmt)
		if !ok {
			t.Fatal("rank upload does not check layout before other operations")
		}
		assignment, ok := first.Init.(*ast.AssignStmt)
		if !ok || len(assignment.Rhs) != 1 {
			t.Fatal("rank upload lacks error-returning admission")
		}
		call, ok := assignment.Rhs[0].(*ast.CallExpr)
		if !ok {
			t.Fatal("rank upload first operation does not check layout")
		}
		guard, ok := call.Fun.(*ast.Ident)
		if !ok || guard.Name != "checkRowMajor" || len(first.Body.List) != 1 {
			t.Fatal("rank upload lost its layout guard")
		}
		ret, ok := first.Body.List[0].(*ast.ReturnStmt)
		if !ok || len(ret.Results) != 2 {
			t.Fatal("rank upload does not return admission failures")
		}
		failure, ok := ret.Results[1].(*ast.Ident)
		if !ok || failure.Name != "err" {
			t.Fatal("rank upload discards its layout failure")
		}
	})
	for _, tc := range []struct{ filename, backend string }{
		{"cuda.go", "cudaBackend"}, {"metal.go", "metalBackend"}, {"vulkan.go", "vulkanBackend"},
	} {
		for _, method := range []string{"Upload", "UploadClass"} {
			fn := tensorLayoutMethod(t, tc.filename, tc.backend, method)
			returns := 0
			ast.Inspect(fn.Body, func(node ast.Node) bool {
				ret, ok := node.(*ast.ReturnStmt)
				if !ok {
					return true
				}
				returns++
				if len(ret.Results) != 1 {
					t.Fatalf("%s.%s bypasses upload admission", tc.backend, method)
				}
				call, ok := ret.Results[0].(*ast.CallExpr)
				if !ok {
					t.Fatalf("%s.%s returns without upload admission", tc.backend, method)
				}
				selector, ok := call.Fun.(*ast.SelectorExpr)
				if !ok || selector.Sel.Name != "uploadClass" {
					t.Fatalf("%s.%s does not route to guarded uploadClass", tc.backend, method)
				}
				return true
			})
			if returns == 0 {
				t.Fatalf("%s.%s has no upload return", tc.backend, method)
			}
		}
	}
}
