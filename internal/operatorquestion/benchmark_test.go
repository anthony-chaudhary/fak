package operatorquestion

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

var (
	benchQuestionSink OperatorQuestion
	benchSignalSink   choicetriage.Signal
	benchFoundSink    bool
)

func BenchmarkOperatorQuestion(b *testing.B) {
	claudeGate := NativeGate{
		HarnessCommand: "claude",
		Tool:           "AskUserQuestion",
		Payload:        []byte(`{"questions":[{"header":"Isolation","multiSelect":false,"question":"Which isolation should I use?","options":[{"label":"Explicit paths","description":"Commit only owned files"},{"label":"Wait","description":"Wait for peer edits"}]}]}`),
	}
	codexPlanGate := NativeGate{
		HarnessCommand: "codex",
		Tool:           "update_plan",
		Payload:        []byte(`{"explanation":"safe sequence","file_tree":["internal/x/**"],"done_criterion":"dos verify plan phase","plan":[{"step":"inspect","status":"pending","tool":"Read","args":{"path":"x"}},{"step":"edit","status":"pending"}]}`),
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		q1, err := Normalize(claudeGate)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q1
		benchSignalSink = q1.ToSignal()

		q2, err := Normalize(codexPlanGate)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q2
		benchSignalSink = q2.ToSignal()
	}
}

func BenchmarkNormalizeClaudeQuestion(b *testing.B) {
	gate := NativeGate{
		HarnessCommand: "claude",
		Tool:           "AskUserQuestion",
		Payload:        []byte(`{"questions":[{"header":"Approach","question":"Select strategy:","options":[{"label":"Fast","description":"Fast path"},{"label":"Careful","description":"Full check"}]}]}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, err := Normalize(gate)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q
	}
}

func BenchmarkNormalizeClaudePlan(b *testing.B) {
	gate := NativeGate{
		HarnessCommand: "claude-code",
		Tool:           "ExitPlanMode",
		Payload:        []byte(`{"plan":"run tests and commit","file_tree":["internal/operatorquestion/**"],"steps":[{"text":"verify","tool":"bash","args":{"command":"go test ./..."}}],"done_criterion":"all green"}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, err := Normalize(gate)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q
	}
}

func BenchmarkNormalizeCodexQuestion(b *testing.B) {
	gate := NativeGate{
		HarnessCommand: "codex",
		Tool:           "functions.request_user_input",
		Payload:        []byte(`{"questions":[{"id":"q1","header":"Choice","question":"Select option:","options":[{"label":"A","description":"First"},{"label":"B","description":"Second"}]}]}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, err := Normalize(gate)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q
	}
}

func BenchmarkNormalizeCodexPlan(b *testing.B) {
	gate := NativeGate{
		HarnessCommand: "codex",
		Tool:           "update_plan",
		Payload:        []byte(`{"explanation":"two-phase update","file_tree":["internal/operatorquestion/**"],"done_criterion":"witnessed commit","plan":[{"step":"audit","status":"pending","tool":"Read","args":{"path":"doc.go"}},{"step":"implement","status":"pending"}]}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, err := Normalize(gate)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q
	}
}

func BenchmarkToSignal(b *testing.B) {
	q := OperatorQuestion{
		Kind:     ChooseApproach,
		Harness:  "claude",
		Question: "Which isolation should I use?",
		Detail:   "Detailed question context",
		Options: []Option{
			{Label: "Explicit paths", Rationale: "Commit only owned files"},
			{Label: "Wait", Rationale: "Wait for peer edits"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSignalSink = q.ToSignal()
	}
}

func BenchmarkLastFromTranscript(b *testing.B) {
	tmpDir := b.TempDir()
	path := filepath.Join(tmpDir, "bench_transcript.jsonl")
	content := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"path":"doc.go"}}]}}` + "\n" +
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"AskUserQuestion","input":{"questions":[{"question":"Proceed?","options":[{"label":"Yes","description":"Continue"},{"label":"No","description":"Halt"}]}]}}]}}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, found, err := LastFromTranscript(path, "claude")
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q
		benchFoundSink = found
	}
}

func BenchmarkLastFromTranscriptAny(b *testing.B) {
	tmpDir := b.TempDir()
	path := filepath.Join(tmpDir, "bench_mixed_transcript.jsonl")
	content := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"path":"doc.go"}}]}}` + "\n" +
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"functions.request_user_input","input":{"questions":[{"id":"q","header":"H","question":"Proceed?","options":[{"label":"Y","description":"yes"}]}]}}]}}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, found, err := LastFromTranscriptAny(path)
		if err != nil {
			b.Fatal(err)
		}
		benchQuestionSink = q
		benchFoundSink = found
	}
}

func TestBenchmarkOperatorQuestionExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkOperatorQuestion)
	if res.N <= 0 {
		t.Fatalf("expected BenchmarkOperatorQuestion iterations > 0, got %d", res.N)
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

	hasBenchmarkOperatorQuestion := false
	substantiveBenchmarks := 0
	for _, decl := range node.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || len(fn.Name.Name) < 9 || fn.Name.Name[:9] != "Benchmark" {
			continue
		}
		if fn.Body == nil || len(fn.Body.List) == 0 {
			continue
		}
		hasLoop := false
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if forStmt, ok := n.(*ast.ForStmt); ok {
				if forStmt.Cond != nil {
					ast.Inspect(forStmt.Cond, func(cn ast.Node) bool {
						if id, ok := cn.(*ast.Ident); ok && id.Name == "N" {
							hasLoop = true
							if fn.Name.Name == "BenchmarkOperatorQuestion" {
								hasBenchmarkOperatorQuestion = true
							}
						}
						return true
					})
				}
			}
			return true
		})
		if hasLoop {
			substantiveBenchmarks++
		}
	}

	if !hasBenchmarkOperatorQuestion {
		t.Errorf("benchmark_test.go must define BenchmarkOperatorQuestion containing a b.N loop")
	}
	if substantiveBenchmarks < 5 {
		t.Errorf("expected at least 5 substantive benchmarks with b.N loop, found %d", substantiveBenchmarks)
	}
}
