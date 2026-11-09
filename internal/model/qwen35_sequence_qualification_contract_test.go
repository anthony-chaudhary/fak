package model

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// TestCUDAQwen35SequenceQualificationSource checks the composition of the
// CUDA-only qualification test without requiring a CUDA build. It is not a
// device, numerical-parity, or execution-owner runtime witness.
//
// fak-test:runtime fast est=5ms lane=default
func TestCUDAQwen35SequenceQualificationSource(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "qwen35_sequence_prefill_cuda_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	functions := make(map[string]*ast.FuncDecl)
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			functions[fn.Name.Name] = fn
		}
	}
	text := func(node ast.Node) string {
		var out bytes.Buffer
		if err := format.Node(&out, fset, node); err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	function := func(name string) *ast.FuncDecl {
		fn := functions[name]
		if fn == nil {
			t.Fatalf("missing qualification function %s", name)
		}
		return fn
	}
	positive := function("TestCUDAQwen35SequencePrefillMatchesCPUAndPersistsDecodeState")
	negative := function("TestCUDAQwen35SequenceMalformedRequestFailsBeforeMutation")

	t.Run("admitted_fixture_geometry", func(t *testing.T) {
		cfg := functions["cudaQwen35SequenceParityTestCfg"]
		if cfg == nil {
			t.Fatal("CUDA qualification still uses the inadmissible generic fixture")
		}
		body := text(cfg.Body)
		if !strings.Contains(body, "cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 24, 4, 256") ||
			!strings.Contains(body, "cfg.PartialRotaryFactor = 0.25") {
			t.Fatal("qualification fixture does not match the CUDA prompt-attention geometry")
		}
		for _, fn := range []*ast.FuncDecl{positive, negative} {
			if !strings.Contains(text(fn.Body), "NewSynthetic(cudaQwen35SequenceParityTestCfg())") {
				t.Fatalf("%s does not use the CUDA-admitted fixture", fn.Name.Name)
			}
		}
	})
	t.Run("owned_states_before_dispatch", func(t *testing.T) {
		var callback *ast.FuncLit
		var dispatches []*ast.CallExpr
		ast.Inspect(positive.Body, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch text(call.Fun) {
			case "device.qwen35HAL.mutateSequence":
				if callback != nil || len(call.Args) != 2 {
					t.Fatal("qualification must use one real mutation-owner callback")
				}
				callback, _ = call.Args[1].(*ast.FuncLit)
			case "sequence.Qwen35SequencePrefill":
				dispatches = append(dispatches, call)
			}
			return true
		})
		if callback == nil || len(dispatches) != 1 {
			t.Fatal("qualification bypasses the real mutation-owner callback")
		}
		dispatch := dispatches[0]
		if dispatch.Pos() <= callback.Body.Pos() || dispatch.End() >= callback.Body.End() {
			t.Fatal("sequence dispatch escaped the protected owner callback")
		}
		if len(callback.Body.List) < 3 || text(callback.Body.List[0]) != "req.States = states" {
			t.Fatal("request states must be replaced by owned handles before capture or dispatch")
		}
		body := text(callback.Body)
		for _, required := range []string{
			"state.Conv.Buf() == nil", "!state.Conv.Ready()",
			"state.Recurrent.Buf() == nil", "!state.Recurrent.Ready()",
			"stateHandles[layer] = [2]compute.Buffer{state.Conv.Buf(), state.Recurrent.Buf()}",
			"be.ResetHostXfer()", "be.ResetH2DXfer()", "be.ResetQwen35GDNOperationCount()",
		} {
			if !strings.Contains(body, required) {
				t.Fatalf("protected qualification callback missing %s", required)
			}
		}
		if text(callback.Body.List[len(callback.Body.List)-1]) != "return sequence.Qwen35SequencePrefill(req)" {
			t.Fatal("qualification must dispatch after binding, handle capture and counter reset")
		}
		ownedScope := false
		ast.Inspect(positive.Body, func(node ast.Node) bool {
			fn, ok := node.(*ast.FuncLit)
			if !ok || fn.Pos() >= callback.Pos() || fn.End() <= callback.End() || len(fn.Body.List) < 3 {
				return true
			}
			ownedScope = text(fn.Body.List[0]) == "scope := device.beginBackendExecution()" &&
				text(fn.Body.List[1]) == "defer scope.finish()"
			return true
		})
		if !ownedScope {
			t.Fatal("qualification dispatch lacks its backend execution scope")
		}
	})
	t.Run("specific_negative_control", func(t *testing.T) {
		if !strings.Contains(text(negative.Body), `sequenceErr.Stage != "embedding-gather"`) {
			t.Fatal("malformed-token control can pass on an unrelated preflight failure")
		}
	})
	t.Run("prefill_oracle_survives_step", func(t *testing.T) {
		if !strings.Contains(text(positive.Body), "wantPrefill := append([]float32(nil), cpu.Prefill(prompt)...)") {
			t.Fatal("prefill oracle is not preserved before the next CPU step")
		}
	})
}
