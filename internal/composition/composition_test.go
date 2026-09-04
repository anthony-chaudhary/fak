package composition

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"
)

func qwen(id string) Snapshot {
	return Snapshot{Intent: Intent{WorkID: "turn-1", Quality: "complete", LatencyClass: "interactive", CostClass: "local", PolicyID: "readonly"}, Model: Model{ID: id, Revision: "1", Provenance: "witnessed", Engine: "fak-native", Capabilities: []string{"hybrid_attention", "gdn_state"}}, Execution: Execution{Backend: "metal", Quantization: "q4_k", Phases: []string{"prefill", "decode"}}, Claims: []ResourceClaim{{Kind: "weights", Owner: "session", Lifetime: "model", Locality: "device", Compatibility: "qwen/v1", Bytes: 100}, {Kind: "attention_kv", Owner: "session", Lifetime: "session", Locality: "device", Compatibility: "kv/v1", Bytes: 20}, {Kind: "gdn_state", Owner: "session", Lifetime: "session", Locality: "device", Compatibility: "gdn/v1", Bytes: 4}}, Edges: []Edge{{From: "weights", To: "prefill", Kind: "load", Bytes: 100}, {From: "prefill", To: "decode", Kind: "state_transfer", Bytes: 24}}}
}

func TestQwenAndSyntheticResolveToScrubbedReceipt(t *testing.T) {
	for _, id := range []string{"qwen3.8-hybrid", "synthetic-variation"} {
		h, r, err := Resolve(qwen(id))
		if err != nil {
			t.Fatal(err)
		}
		if h.Snapshot().Digest == "" || r.GraphDigest != h.Snapshot().Digest || r.Engine != "fak-native" || len(r.StateKinds) != 3 {
			t.Fatalf("receipt=%+v", r)
		}
	}
}

func TestInvalidCombinationFailsBeforeAllocation(t *testing.T) {
	s := qwen("bad")
	s.Forbidden = [][]string{{"hybrid_attention", "metal", "q4_k", "gdn_state"}}
	h, _, err := Resolve(s)
	if !IsReason(err, ReasonForbiddenCombination) || h.Snapshot() != nil {
		t.Fatalf("handle=%+v err=%v", h, err)
	}
	s = qwen("bad")
	s.Model.Engine = "llama.cpp"
	if _, _, err = Resolve(s); !IsReason(err, ReasonEngineAmbiguous) {
		t.Fatal(err)
	}
}

func TestBenchmarkMaturity(t *testing.T) {
	benchPath := filepath.Join(".", "benchmark_test.go")
	content, err := os.ReadFile(benchPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", benchPath, err)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, benchPath, content, 0)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", benchPath, err)
	}

	hasBenchmarkComposition := false
	for _, decl := range node.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "BenchmarkComposition" {
			continue
		}
		if fn.Body == nil || len(fn.Body.List) == 0 {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if forStmt, ok := n.(*ast.ForStmt); ok {
				if forStmt.Cond != nil {
					ast.Inspect(forStmt.Cond, func(cn ast.Node) bool {
						if id, ok := cn.(*ast.Ident); ok && id.Name == "N" {
							hasBenchmarkComposition = true
						}
						return true
					})
				}
			}
			return true
		})
	}
	if !hasBenchmarkComposition {
		t.Errorf("benchmark_test.go must define BenchmarkComposition containing a b.N loop")
	}
}
