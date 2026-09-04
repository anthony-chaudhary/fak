package codexsession

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// BenchmarkSessionMarshal measures serialization and deserialization throughput
// for core Codex session protocol messages and approval journal entries.
func BenchmarkSessionMarshal(b *testing.B) {
	entry := ApprovalJournalEntry{
		ApprovalID:         "thread-1:turn-2:item-3:req-4",
		Kind:               "command",
		Status:             "approved",
		Reason:             "human operator approved command execution within workspace bounds",
		Scope:              "/workspace/project",
		ThreadID:           "thread-1",
		TurnID:             "turn-2",
		ItemID:             "item-3",
		RequestID:          "req-4",
		FakCapabilityFloor: "allow",
		CodexSandboxPolicy: "workspace-write",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(entry)
		if err != nil {
			b.Fatalf("marshal failed: %v", err)
		}
		var decoded ApprovalJournalEntry
		if err := json.Unmarshal(data, &decoded); err != nil {
			b.Fatalf("unmarshal failed: %v", err)
		}
		if decoded.ApprovalID != entry.ApprovalID {
			b.Fatalf("round-trip mismatch: got %q, want %q", decoded.ApprovalID, entry.ApprovalID)
		}
	}
}

func TestBenchmarkSessionMarshalSanity(t *testing.T) {
	entry := ApprovalJournalEntry{
		ApprovalID:         "thread-test:turn-1:item-1:req-1",
		Kind:               "patch",
		Status:             "pending",
		Reason:             "testing sanity check",
		Scope:              "/workspace",
		ThreadID:           "thread-test",
		TurnID:             "turn-1",
		ItemID:             "item-1",
		RequestID:          "req-1",
		FakCapabilityFloor: "deny",
		CodexSandboxPolicy: "untrusted",
	}

	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var decoded ApprovalJournalEntry
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	if decoded.ApprovalID != entry.ApprovalID || decoded.Kind != entry.Kind {
		t.Fatalf("unexpected decoded entry: %+v", decoded)
	}

	res := testing.Benchmark(BenchmarkSessionMarshal)
	if res.N <= 0 {
		t.Fatalf("expected benchmark to execute iterations, got %d", res.N)
	}
}

func TestCodexSessionMaturityRequirements(t *testing.T) {
	files := []string{"adapter.go", "approval.go", "compatibility.go"}
	fset := token.NewFileSet()

	totalContractComments := 0
	totalExported := 0
	totalDocumented := 0

	for _, filename := range files {
		content, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("failed to read %s: %v", filename, err)
		}
		node, err := parser.ParseFile(fset, filename, content, parser.ParseComments)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", filename, err)
		}

		fileContracts := 0
		for _, cg := range node.Comments {
			if isTestSubstantiveContractComment(cg) {
				fileContracts++
				totalContractComments++
			}
		}
		if fileContracts == 0 {
			t.Errorf("%s: expected at least one substantive contract comment", filename)
		}

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(d.Name.Name) {
					totalExported++
					if d.Doc != nil && len(strings.TrimSpace(d.Doc.Text())) > 12 {
						totalDocumented++
					} else {
						t.Errorf("%s: undocumented exported func: %s", filename, d.Name.Name)
					}
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if ast.IsExported(s.Name.Name) {
							totalExported++
							doc := s.Doc
							if doc == nil {
								doc = d.Doc
							}
							if doc != nil && len(strings.TrimSpace(doc.Text())) > 12 {
								totalDocumented++
							} else {
								t.Errorf("%s: undocumented exported type: %s", filename, s.Name.Name)
							}
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if ast.IsExported(name.Name) {
								totalExported++
								doc := s.Doc
								if doc == nil {
									doc = d.Doc
								}
								if doc != nil && len(strings.TrimSpace(doc.Text())) > 12 {
									totalDocumented++
								} else {
									t.Errorf("%s: undocumented exported var/const: %s", filename, name.Name)
								}
							}
						}
					}
				}
			}
		}
	}

	if totalContractComments < len(files) {
		t.Errorf("expected at least %d contract comments across files, got %d", len(files), totalContractComments)
	}

	if totalExported == 0 || totalDocumented != totalExported {
		t.Errorf("expected 100%% documented exports: %d/%d documented", totalDocumented, totalExported)
	}

	// Verify BenchmarkSessionMarshal exists in benchmark_test.go and contains a b.N loop
	benchPath := filepath.Join(".", "benchmark_test.go")
	benchContent, err := os.ReadFile(benchPath)
	if err != nil {
		t.Fatalf("failed to read benchmark_test.go: %v", err)
	}
	benchNode, err := parser.ParseFile(fset, benchPath, benchContent, 0)
	if err != nil {
		t.Fatalf("failed to parse benchmark_test.go: %v", err)
	}
	foundBench := false
	for _, decl := range benchNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "BenchmarkSessionMarshal" {
			foundBench = true
			hasNLoop := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if forStmt, ok := n.(*ast.ForStmt); ok {
					if forStmt.Cond != nil {
						ast.Inspect(forStmt.Cond, func(cn ast.Node) bool {
							if id, ok := cn.(*ast.Ident); ok && id.Name == "N" {
								hasNLoop = true
							}
							return true
						})
					}
				}
				return true
			})
			if !hasNLoop {
				t.Error("BenchmarkSessionMarshal does not contain a loop referencing b.N")
			}
		}
	}
	if !foundBench {
		t.Error("BenchmarkSessionMarshal function not found in benchmark_test.go")
	}
}

func isTestSubstantiveContractComment(cg *ast.CommentGroup) bool {
	if cg == nil {
		return false
	}
	text := strings.TrimSpace(cg.Text())
	if len(text) < 35 {
		return false
	}
	lower := strings.ToLower(text)

	hasContractMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "assumption:") ||
		strings.Contains(lower, "assumptions:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.Contains(lower, "precondition:") ||
		strings.Contains(lower, "postcondition:") ||
		strings.Contains(lower, "guard:")
	if !hasContractMarker {
		return false
	}

	words := strings.Fields(lower)
	if len(words) < 6 {
		return false
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.4 {
		return false
	}
	return true
}
