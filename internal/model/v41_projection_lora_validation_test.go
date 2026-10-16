package model

import (
	"bytes"
	"errors"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41LoRAValidationBackend struct {
	*v41DenseTestBackend
	invalidRead bool
}

func (b *v41LoRAValidationBackend) Read(x compute.Tensor) []float32 {
	values := b.v41DenseTestBackend.Read(x)
	if b.invalidRead {
		values[0] = float32(math.NaN())
	}
	return values
}

// fak-test:runtime fast est=20ms lane=default
func TestV41ProjectionNoLoRADuplicateValidation(t *testing.T) {
	t.Parallel()
	t.Run("nil_adapter_has_one_output_traversal", func(t *testing.T) {
		// Bind the optimization to the actual production control flow. Runtime
		// values alone cannot distinguish one finite traversal from two identical ones.
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "v41_forward_weights.go", nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		text := func(n ast.Node) string {
			var b bytes.Buffer
			if err := format.Node(&b, fset, n); err != nil {
				t.Fatal(err)
			}
			return b.String()
		}
		found := 0
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Name.Name != "v41BindDenseProjection" {
				continue
			}
			ast.Inspect(fn, func(node ast.Node) bool {
				block, ok := node.(*ast.BlockStmt)
				if !ok {
					return true
				}
				for i, stmt := range block.List {
					guard, ok := stmt.(*ast.IfStmt)
					if !ok || text(guard.Cond) != "s.M.lora != nil" {
						continue
					}
					found++
					if i < 2 || guard.Else != nil || len(guard.Body.List) != 2 {
						t.Fatal("LoRA guard must contain only row application and validation")
					}
					before, ok := block.List[i-2].(*ast.IfStmt)
					if !ok || text(before.Cond) != "head || guarded" || text(block.List[i-1]) != "opStage = \"lora\"" {
						t.Fatal("first finite check or error stage changed")
					}
					after, ok := guard.Body.List[1].(*ast.IfStmt)
					if !ok || text(before) != text(after) {
						t.Fatal("active LoRA lost the complete finite check")
					}
					if len(before.Body.List) != 1 {
						t.Fatal("unexpected first validation body")
					}
					loop, ok := before.Body.List[0].(*ast.RangeStmt)
					if !ok || text(loop.X) != "values" {
						t.Fatal("first validation must traverse output values")
					}
					calls := 0
					ast.Inspect(loop, func(n ast.Node) bool {
						if call, ok := n.(*ast.CallExpr); ok && text(call.Fun) == "finite32" {
							calls++
						}
						return true
					})
					if calls != 1 {
						t.Fatal("output finite validation missing or duplicated")
					}
					if _, ok := guard.Body.List[0].(*ast.ForStmt); !ok {
						t.Fatal("row application order changed")
					}
				}
				return true
			})
		}
		if found != 1 {
			t.Fatalf("got %d no-adapter guards, want one", found)
		}
	})

	for _, tc := range []struct {
		name                           string
		adapter, overflow, invalidRead bool
		stage                          string
	}{
		{name: "nil_adapter"},
		{name: "nil_adapter_invalid_read", invalidRead: true, stage: "readback"},
		{name: "active_adapter", adapter: true},
		{name: "active_adapter_overflow", adapter: true, overflow: true, stage: "lora"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Only a 2x2 head is needed; no full model/session forward or device is used.
			m := &Model{Cfg: Config{HiddenSize: 2, VocabSize: 2}}
			defer func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			}()
			m.manifest, m.raw = synthBuildRaw([]synthTensor{{"lm_head.weight", []int{2, 2}}}, func(string, func() float32) float32 { return .25 })
			if tc.adapter {
				set := NewLoRASet()
				b := []float32{.25, .5}
				if tc.overflow {
					b[0] = math.MaxFloat32
				}
				if err := set.Add(&LoRAAdapter{Adapter: "validation", Target: "lm_head.weight", In: 2, Out: 2, Rank: 1, Alpha: 1, A: []float32{1, 1}, B: b}); err != nil {
					t.Fatal(err)
				}
				set.Activate("validation")
				m.SetLoRA(set)
			}
			b := &v41LoRAValidationBackend{v41DenseTestBackend: newV41DenseTestBackend(), invalidRead: tc.invalidRead}
			s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
			defer s.Close()
			got, outcome, err := s.v41DenseProjectionFunc()(-1, "lm_head.weight", []float32{1, 1, 2, 2}, 2, 2, 2)
			if tc.stage != "" {
				var closed *BackendForwardOperationError
				if outcome != v41ProjectionError || got != nil || !errors.Is(err, errV41ProjectionResult) || !errors.As(err, &closed) || closed.Stage != tc.stage || !s.BackendSessionClosed() {
					t.Fatalf("invalid result changed selected failure: outcome=%v got=%v err=%v", outcome, got, err)
				}
			} else {
				want := []float32{.5, .5, 1, 1}
				if tc.adapter {
					want = []float32{1, 1.5, 2, 3}
				}
				if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) || s.BackendSessionClosed() {
					t.Fatalf("result=%v want=%v outcome=%v err=%v", got, want, outcome, err)
				}
			}
			if len(b.live) != 0 {
				t.Fatalf("projection leaked %d transient tensors", len(b.live))
			}
		})
	}
}
