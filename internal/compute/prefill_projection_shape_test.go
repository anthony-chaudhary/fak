package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestPrefillBatchProjectionShape(t *testing.T) {
	for _, name := range []string{"Wq", "Wk", "Wv", "Wo"} {
		for _, shape := range []string{"valid", "transposed", "flat", "extra-axis"} {
			t.Run(name+"/"+shape, func(t *testing.T) {
				x := NewF32(Default(), []int{2, 3}, make([]float32, 6))
				q := NewF32(Default(), []int{4, 3}, make([]float32, 12))
				kv := NewF32(Default(), []int{2, 3}, make([]float32, 6))
				o := NewF32(Default(), []int{3, 4}, make([]float32, 12))
				args := PrefillBatchArgs{X: x, Wq: q, Wk: kv, Wv: kv, Wo: o, NumHeads: 2, NumKVHeads: 1, HeadDim: 2}
				var target *Tensor
				switch name {
				case "Wq":
					target = &args.Wq
				case "Wk":
					target = &args.Wk
				case "Wv":
					target = &args.Wv
				case "Wo":
					target = &args.Wo
				}
				rows, cols := target.Shape[0], target.Shape[1]
				switch shape {
				case "transposed":
					target.Shape = []int{cols, rows}
				case "flat":
					target.Shape = []int{rows * cols}
				case "extra-axis":
					target.Shape = []int{rows, cols, 1}
				}
				// Every malformed shape has the SAME element count as the valid
				// matrix. A numel-only check cannot protect the GEMM axes.
				if target.Numel() != rows*cols {
					t.Fatal("test changed element count")
				}
				p, d, err := validatePrefillBatchArgs(&args)
				if shape == "valid" {
					if err != nil || p != 2 || d != 3 {
						t.Fatalf("valid projection rejected: P=%d D=%d error=%v", p, d, err)
					}
				} else if err == nil {
					t.Fatal("malformed projection admitted before GEMM")
				}
			})
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPrefillBatchProjectionShapeCompatibility(t *testing.T) {
	for _, layout := range []Layout{RowMajor, ColMajor, Tiled} {
		for _, flat := range []bool{false, true} {
			x := NewF32(Default(), []int{2, 3}, make([]float32, 6))
			q := NewF32(Default(), []int{4, 3}, make([]float32, 12))
			kv := NewF32(Default(), []int{2, 3}, make([]float32, 6))
			q.Layout, kv.Layout = layout, layout
			args := PrefillBatchArgs{X: x, Wq: q, Wk: kv, Wv: kv, NumHeads: 2, NumKVHeads: 1, HeadDim: 2}
			if flat {
				args.X.Shape = []int{6}
				args.Tokens = 2
			}
			// Layout is a physical descriptor, not a change to logical axes.
			// This is validator compatibility only, not backend support for Tiled.
			if p, d, err := validatePrefillBatchArgs(&args); err != nil || p != 2 || d != 3 {
				t.Fatalf("layout=%v flat=%v rejected: P=%d D=%d err=%v", layout, flat, p, d, err)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestPrefillBatchProjectionShapePrecedesBackendEffects(t *testing.T) {
	for _, path := range []string{"prefill_batch.go", "cuda.go", "metal.go", "vulkan_ops_vulkan.go"} {
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || fn.Name.Name != "PrefillBatch" {
				continue
			}
			found = true
			if len(fn.Body.List) < 2 {
				t.Fatalf("%s: incomplete PrefillBatch", path)
			}
			assign, ok := fn.Body.List[0].(*ast.AssignStmt)
			if !ok || len(assign.Rhs) != 1 {
				t.Fatalf("%s: validation is not the first operation", path)
			}
			call, ok := assign.Rhs[0].(*ast.CallExpr)
			if !ok {
				t.Fatalf("%s: first operation is not validation call", path)
			}
			name, ok := call.Fun.(*ast.Ident)
			if !ok || name.Name != "validatePrefillBatchArgs" {
				t.Fatalf("%s: projection validation moved after effects", path)
			}
			guard, ok := fn.Body.List[1].(*ast.IfStmt)
			if !ok || len(guard.Body.List) != 1 {
				t.Fatalf("%s: missing immediate validation error return", path)
			}
			condition, ok := guard.Cond.(*ast.BinaryExpr)
			if !ok || condition.Op != token.NEQ {
				t.Fatalf("%s: unexpected validation guard", path)
			}
			left, leftOK := condition.X.(*ast.Ident)
			right, rightOK := condition.Y.(*ast.Ident)
			if !leftOK || !rightOK || left.Name != "err" || right.Name != "nil" {
				t.Fatalf("%s: validation error not checked", path)
			}
			if _, ok := guard.Body.List[0].(*ast.ReturnStmt); !ok {
				t.Fatalf("%s: error guard does not return", path)
			}
		}
		if !found {
			t.Fatalf("%s: backend PrefillBatch missing", path)
		}
	}
}
