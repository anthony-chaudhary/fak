package trajectory

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func evalCalls(names ...string) []EvalToolCall {
	calls := make([]EvalToolCall, len(names))
	for i, name := range names {
		calls[i] = EvalToolCall{Name: name}
	}
	return calls
}

func evalCall(name string, params map[string]any) EvalToolCall {
	return EvalToolCall{Name: name, InputParameters: params}
}

func assertEvalScore(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("score = %v, want %v", got, want)
	}
}

func TestTrajectoryEvalToolCorrectness(t *testing.T) {
	params := ToolCorrectnessOptions{CompareInputParameters: true}
	ordered := ToolCorrectnessOptions{ConsiderOrdering: true}
	orderedParams := ToolCorrectnessOptions{ConsiderOrdering: true, CompareInputParameters: true}
	exact := ToolCorrectnessOptions{ExactMatch: true}
	exactParams := ToolCorrectnessOptions{ExactMatch: true, CompareInputParameters: true}
	cases := []struct {
		name     string
		expected []EvalToolCall
		called   []EvalToolCall
		opts     ToolCorrectnessOptions
		want     float64
	}{
		{"unordered all present", evalCalls("A", "B"), evalCalls("B", "A"), ToolCorrectnessOptions{}, 1},
		{"unordered missing tools", evalCalls("A", "B", "C"), evalCalls("A", "D"), ToolCorrectnessOptions{}, 1.0 / 3},
		{"unordered extra tools not penalized", evalCalls("A"), evalCalls("A", "B", "C"), ToolCorrectnessOptions{}, 1},
		{"unordered repeated expectation consumes per call", evalCalls("A", "A"), evalCalls("A"), ToolCorrectnessOptions{}, 0.5},
		{"unordered repeated calls satisfy repeated expectation", evalCalls("A", "A"), evalCalls("A", "A"), ToolCorrectnessOptions{}, 1},
		{"unordered type mismatch", []EvalToolCall{{Name: "A", Type: "mcp"}}, []EvalToolCall{{Name: "A", Type: "function"}}, ToolCorrectnessOptions{}, 0},
		{"unordered args off ignores params",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "y": 2.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "y": 3.0})}, ToolCorrectnessOptions{}, 1},
		{"unordered args on partial credit",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "y": 2.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "y": 3.0})}, params, 0.5},
		{"unordered args on extra key widens union",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "z": 9.0})}, params, 0.5},
		{"unordered args on nested partial credit",
			[]EvalToolCall{evalCall("A", map[string]any{"q": map[string]any{"a": 1.0, "b": 2.0}, "z": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"q": map[string]any{"a": 1.0, "b": 3.0}, "z": 1.0})}, params, 0.75},
		{"unordered args on picks best call",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 2.0}), evalCall("A", map[string]any{"x": 1.0})}, params, 1},
		{"unordered nil and empty params are equal",
			[]EvalToolCall{evalCall("A", nil)}, []EvalToolCall{evalCall("A", map[string]any{})}, params, 1},
		{"unordered output mismatch zeroes the pair",
			[]EvalToolCall{{Name: "A", Output: "ok"}}, []EvalToolCall{{Name: "A", Output: "bad"}},
			ToolCorrectnessOptions{CompareOutput: true}, 0},
		{"unordered empty both", nil, nil, ToolCorrectnessOptions{}, 1},
		{"unordered empty expected with calls", nil, evalCalls("A"), ToolCorrectnessOptions{}, 0},

		{"ordered in order", evalCalls("A", "B", "C"), evalCalls("A", "B", "C"), ordered, 1},
		{"ordered one swap keeps lcs of two", evalCalls("A", "B", "C"), evalCalls("A", "C", "B"), ordered, 2.0 / 3},
		{"ordered reversed keeps lcs of one", evalCalls("A", "B", "C"), evalCalls("C", "B", "A"), ordered, 1.0 / 3},
		{"ordered extra calls interleaved", evalCalls("A", "B"), evalCalls("X", "A", "Y", "B"), ordered, 1},
		{"ordered missing tool", evalCalls("A", "B", "C"), evalCalls("A", "C"), ordered, 2.0 / 3},
		{"ordered args mismatch drops the step",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0}), evalCall("B", map[string]any{"y": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0}), evalCall("B", map[string]any{"y": 2.0})}, orderedParams, 0.5},
		{"ordered args partial weight",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "y": 1.0}), evalCall("B", nil)},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0, "y": 2.0}), evalCall("B", nil)}, orderedParams, 0.75},
		{"ordered args off ignores params",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 2.0})}, ordered, 1},
		{"ordered empty both", nil, nil, ordered, 1},
		{"ordered empty expected with calls", nil, evalCalls("A"), ordered, 0},

		{"exact identical", evalCalls("A", "B"), evalCalls("A", "B"), exact, 1},
		{"exact reordered", evalCalls("A", "B"), evalCalls("B", "A"), exact, 0},
		{"exact extra call", evalCalls("A"), evalCalls("A", "B"), exact, 0},
		{"exact args on mismatch",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 2.0})}, exactParams, 0},
		{"exact args off ignores params",
			[]EvalToolCall{evalCall("A", map[string]any{"x": 1.0})},
			[]EvalToolCall{evalCall("A", map[string]any{"x": 2.0})}, exact, 1},
		{"exact empty both", nil, nil, exact, 1},
		{"exact empty expected with calls", nil, evalCalls("A"), exact, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ScoreToolCorrectness(tc.expected, tc.called, tc.opts)
			assertEvalScore(t, got.Score, tc.want)
			if got.Metric != "tool_correctness" || got.Threshold != DefaultEvalThreshold {
				t.Fatalf("metric/threshold = %q/%v", got.Metric, got.Threshold)
			}
			if got.Success != (tc.want >= DefaultEvalThreshold) {
				t.Fatalf("success = %v for score %v", got.Success, got.Score)
			}
		})
	}
}

func TestTrajectoryEvalStrictModeCollapses(t *testing.T) {
	expected, called := evalCalls("A", "B"), evalCalls("A")

	lenient := ScoreToolCorrectness(expected, called, ToolCorrectnessOptions{})
	assertEvalScore(t, lenient.Score, 0.5)
	if !lenient.Success {
		t.Fatal("0.5 should pass the default 0.5 threshold")
	}

	raised := ScoreToolCorrectness(expected, called, ToolCorrectnessOptions{EvalGate: EvalGate{Threshold: 0.6}})
	assertEvalScore(t, raised.Score, 0.5)
	if raised.Success {
		t.Fatal("0.5 should fail a 0.6 threshold without collapsing the score")
	}

	strict := ScoreToolCorrectness(expected, called, ToolCorrectnessOptions{EvalGate: EvalGate{Threshold: 0.2, StrictMode: true}})
	if strict.Score != 0 || strict.Threshold != 1 || strict.Success {
		t.Fatalf("strict partial = %+v, want score 0 threshold 1 fail", strict)
	}

	strictFull := ScoreToolCorrectness(expected, evalCalls("A", "B"), ToolCorrectnessOptions{EvalGate: EvalGate{StrictMode: true}})
	if strictFull.Score != 1 || !strictFull.Success {
		t.Fatalf("strict full = %+v, want score 1 pass", strictFull)
	}
}

func TestTrajectoryEvalStepEfficiency(t *testing.T) {
	cases := []struct {
		name              string
		reference, actual int
		gate              EvalGate
		want              float64
		success           bool
	}{
		{"double the reference", 3, 6, EvalGate{}, 0.5, true},
		{"fewer than reference clamps to one", 4, 2, EvalGate{}, 1, true},
		{"exactly reference", 5, 5, EvalGate{}, 1, true},
		{"zero reference and zero actual", 0, 0, EvalGate{}, 1, true},
		{"zero actual with work required", 3, 0, EvalGate{}, 0, false},
		{"zero reference with steps taken", 0, 5, EvalGate{}, 0, false},
		{"strict collapses partial", 3, 6, EvalGate{StrictMode: true}, 0, false},
		{"threshold above score", 3, 4, EvalGate{Threshold: 0.8}, 0.75, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScoreStepEfficiency(tc.reference, tc.actual, tc.gate)
			if err != nil {
				t.Fatal(err)
			}
			assertEvalScore(t, got.Score, tc.want)
			if got.Success != tc.success {
				t.Fatalf("success = %v, want %v", got.Success, tc.success)
			}
		})
	}
	if _, err := ScoreStepEfficiency(-1, 3, EvalGate{}); err == nil {
		t.Fatal("negative reference should be refused")
	}
}

func TestTrajectoryEvalTaskCompletionGate(t *testing.T) {
	clean := ScoreTaskCompletion(AuditTranscriptRow{}, EvalGate{StrictMode: true})
	if clean.Score != 1 || !clean.Success {
		t.Fatalf("clean = %+v, want pass", clean)
	}
	failed := ScoreTaskCompletion(AuditTranscriptRow{
		TerminalFailures:       2,
		TerminalFailureClasses: map[string]int{"http_502_upstream": 1, "http_429_rate_limit": 1},
	}, EvalGate{})
	if failed.Score != 0 || failed.Success {
		t.Fatalf("failed = %+v, want fail", failed)
	}
	if !strings.Contains(failed.Reason, "[http_429_rate_limit http_502_upstream]") {
		t.Fatalf("reason %q should list sorted failure classes", failed.Reason)
	}
}

const trajectoryEvalAuditJSONL = `{"schema":"fak-trajectory-audit/1","kind":"summary","sources":1,"sessions":2}
{"schema":"fak-trajectory-audit/1","kind":"source_denominator","source":"codex","root":"codex/sessions","root_present":true,"files_discovered":2,"files_scanned":2,"records":40,"record_types":{},"token_semantics":"disjoint"}
{"schema":"fak-trajectory-audit/1","kind":"session","source":"codex","session_id":"sess-ok","source_path":"sessions/a.jsonl","models":["gpt-5"],"tokens":{"input_tokens":100,"output_tokens":20,"cache_create_tokens":0,"cache_read_tokens":0},"tool_calls":3,"tool_errors":0,"tool_distribution":[{"name":"shell","bytes":300,"share":0.75,"calls":2},{"name":"apply_patch","bytes":100,"share":0.25,"calls":1}],"unknown_exemplars":{},"repeated_failures":0,"expected_wait_timeouts":0,"mutation_churn":0,"hook_p95_ms":null,"usage_records":2}
{"schema":"fak-trajectory-audit/1","kind":"session","source":"codex","session_id":"sess-fail","source_path":"sessions/b.jsonl","models":["gpt-5"],"tokens":{"input_tokens":50,"output_tokens":5,"cache_create_tokens":0,"cache_read_tokens":0},"tool_calls":1,"tool_errors":1,"tool_distribution":[{"name":"shell","bytes":10,"share":1,"calls":1}],"unknown_exemplars":{},"repeated_failures":0,"expected_wait_timeouts":0,"mutation_churn":0,"hook_p95_ms":null,"usage_records":1,"terminal_failures":1,"terminal_failure_classes":{"http_502_upstream":1}}
`

func TestTrajectoryEvalAuditJSONLEndToEnd(t *testing.T) {
	spec := TrajectoryEvalSpec{
		SessionID:      "sess-ok",
		ExpectedTools:  evalCalls("shell", "read_file", "apply_patch"),
		ReferenceSteps: 2,
	}
	report, err := EvaluateAuditTrajectory(strings.NewReader(trajectoryEvalAuditJSONL), spec)
	if err != nil {
		t.Fatal(err)
	}
	if report.Source != "codex" || report.SessionID != "sess-ok" {
		t.Fatalf("identity = %s/%s", report.Source, report.SessionID)
	}
	assertEvalScore(t, report.ToolCorrectness.Score, 2.0/3)
	assertEvalScore(t, report.StepEfficiency.Score, 2.0/3)
	if report.TaskCompletion.Score != 1 || !report.TaskCompletion.Success {
		t.Fatalf("task completion = %+v", report.TaskCompletion)
	}

	spec.SessionID = "sess-fail"
	spec.ExpectedTools = evalCalls("shell")
	report, err = EvaluateAuditTrajectory(strings.NewReader(trajectoryEvalAuditJSONL), spec)
	if err != nil {
		t.Fatal(err)
	}
	assertEvalScore(t, report.ToolCorrectness.Score, 1)
	if report.TaskCompletion.Success {
		t.Fatalf("terminal failure should fail the completion gate: %+v", report.TaskCompletion)
	}
}

func TestTrajectoryEvalAuditJSONLRefusals(t *testing.T) {
	if _, err := EvaluateAuditTrajectory(strings.NewReader(trajectoryEvalAuditJSONL), TrajectoryEvalSpec{}); err == nil {
		t.Fatal("two session rows without a session id should be refused")
	}
	if _, err := EvaluateAuditTrajectory(strings.NewReader(trajectoryEvalAuditJSONL), TrajectoryEvalSpec{SessionID: "absent"}); err == nil {
		t.Fatal("missing session should be refused")
	}
	foreign := `{"schema":"other/1","kind":"session","session_id":"x"}`
	if _, err := EvaluateAuditTrajectory(strings.NewReader(foreign), TrajectoryEvalSpec{}); err == nil {
		t.Fatal("foreign schema should be refused")
	}
	for _, opts := range []ToolCorrectnessOptions{{ExactMatch: true}, {ConsiderOrdering: true}, {CompareInputParameters: true}, {CompareOutput: true}} {
		_, err := EvaluateAuditTrajectory(strings.NewReader(trajectoryEvalAuditJSONL), TrajectoryEvalSpec{SessionID: "sess-ok", ToolCorrectness: opts})
		if !errors.Is(err, ErrEvalNeedsOrderedCalls) {
			t.Fatalf("opts %+v: err = %v, want ErrEvalNeedsOrderedCalls", opts, err)
		}
	}
}
