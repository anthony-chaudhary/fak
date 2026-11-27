package agentbench

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

func floatPtr(v float64) *float64 { return &v }

func makeTasksSummary(model, suite string, trials, accepted, loops, cyclicRepeats int, validity float64, rows []TaskRepRow) TasksSummary {
	return TasksSummary{
		Schema:         "fak.agentbench.tasks-summary.v1",
		Model:          model,
		Suite:          suite,
		IncludeHeldout: suite == "extended",
		Reps:           1,
		CompletedReps:  1,
		Tasks:          rows,
		Aggregate: taskrun.TrialSummary{
			Trials:           trials,
			Accepted:         accepted,
			AcceptRate:       floatPtr(float64(accepted) / float64(trials)),
			MeasuredTrials:   trials,
			ToolCallsTotal:   trials * 5,
			ToolCallsInvalid: int(math.Round(float64(trials*5) * (1 - validity))),
			ToolCallValidity: floatPtr(validity),
			Loops:            loops,
			StuckRate:        floatPtr(float64(loops) / float64(trials)),
			CyclicRepeats:    cyclicRepeats,
			CyclicRepeatRate: floatPtr(float64(cyclicRepeats) / float64(trials)),
		},
	}
}

func makeTaskRows(n, accepted int) []TaskRepRow {
	rows := make([]TaskRepRow, 0, n)
	for i := 0; i < n; i++ {
		acc := i < accepted
		cnt := 0
		if acc {
			cnt = 1
		}
		rows = append(rows, TaskRepRow{
			ID:            "task-" + string(rune('a'+i%26)) + string(rune('0'+i/26)),
			Family:        "demo",
			Accepted:      []bool{acc},
			AcceptedCount: cnt,
			Loops:         []bool{false},
		})
	}
	return rows
}

// fak-test:runtime fast est=10ms lane=default
func TestEvaluateTasksGatePass(t *testing.T) {
	rows := makeTaskRows(8, 7)
	cand := makeTasksSummary("qwen3.8-27b", "default", 8, 7, 1, 1, 1.0, rows)
	got := EvaluateTasksGate(cand, nil, DefaultGatePolicy())
	if got.Verdict != GatePass || len(got.Reasons) != 0 {
		t.Fatalf("expected PASS with no reasons, got %s %v", got.Verdict, got.Reasons)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEvaluateTasksGateAbsoluteFailures(t *testing.T) {
	rows := makeTaskRows(10, 5)
	cand := makeTasksSummary("bad-quant", "default", 10, 5, 3, 4, 0.90, rows)
	got := EvaluateTasksGate(cand, nil, DefaultGatePolicy())
	if got.Verdict != GateFail {
		t.Fatalf("expected FAIL, got %s", got.Verdict)
	}
	for _, want := range []string{
		ReasonAcceptRateBelowFloor,
		ReasonToolValidityBelowFloor,
		ReasonLoopRateAboveCeiling,
		ReasonCyclicRepeatAboveCeiling,
	} {
		if !slices.Contains(got.Reasons, want) {
			t.Fatalf("expected reason %s in %v", want, got.Reasons)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEvaluateTasksGateInconclusivePreconditions(t *testing.T) {
	rows := makeTaskRows(8, 7)

	// Run error
	errSummary := makeTasksSummary("m", "default", 8, 7, 0, 0, 1.0, rows)
	errSummary.Error = "endpoint died"
	if got := EvaluateTasksGate(errSummary, nil, DefaultGatePolicy()); got.Verdict != GateInconclusive || !slices.Contains(got.Reasons, ReasonRunError) {
		t.Fatalf("expected INCONCLUSIVE RUN_ERROR, got %s %v", got.Verdict, got.Reasons)
	}

	// Insufficient trials
	small := makeTasksSummary("m", "default", 4, 4, 0, 0, 1.0, makeTaskRows(4, 4))
	if got := EvaluateTasksGate(small, nil, DefaultGatePolicy()); got.Verdict != GateInconclusive || !slices.Contains(got.Reasons, ReasonInsufficientTrials) {
		t.Fatalf("expected INCONCLUSIVE INSUFFICIENT_TRIALS, got %s %v", got.Verdict, got.Reasons)
	}

	// Unmeasured tool calls
	unmeasured := makeTasksSummary("m", "default", 8, 7, 0, 0, 1.0, rows)
	unmeasured.Aggregate.MeasuredTrials = 0
	unmeasured.Aggregate.ToolCallValidity = nil
	if got := EvaluateTasksGate(unmeasured, nil, DefaultGatePolicy()); got.Verdict != GateInconclusive || !slices.Contains(got.Reasons, ReasonUnmeasuredToolCalls) {
		t.Fatalf("expected INCONCLUSIVE UNMEASURED_TOOL_CALLS, got %s %v", got.Verdict, got.Reasons)
	}

	// Unpinned sampling when required
	pol := DefaultGatePolicy()
	pol.RequirePinnedSampling = true
	cand := makeTasksSummary("m", "default", 8, 7, 0, 0, 1.0, rows)
	if got := EvaluateTasksGate(cand, nil, pol); got.Verdict != GateInconclusive || !slices.Contains(got.Reasons, ReasonSamplingUnpinned) {
		t.Fatalf("expected INCONCLUSIVE SAMPLING_UNPINNED, got %s %v", got.Verdict, got.Reasons)
	}

	// Suite mismatch
	base := makeTasksSummary("base", "extended", 8, 7, 0, 0, 1.0, rows)
	if got := EvaluateTasksGate(cand, &base, DefaultGatePolicy()); got.Verdict != GateInconclusive || !slices.Contains(got.Reasons, ReasonSuiteMismatch) {
		t.Fatalf("expected INCONCLUSIVE SUITE_MISMATCH, got %s %v", got.Verdict, got.Reasons)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEvaluateTasksGateBaselineRegression(t *testing.T) {
	// Planted regression modeled on the 2026-11-22 35-task empirical delta:
	// baseline passes 6 tasks that candidate fails, 0 vice-versa (p = 2 * 2^-6 = 0.03125 <= 0.05).
	baseRows := make([]TaskRepRow, 35)
	candRows := make([]TaskRepRow, 35)
	for i := 0; i < 35; i++ {
		id := "task-" + string(rune('a'+i%26)) + string(rune('0'+i/26))
		basePass := i < 31
		candPass := i < 25
		baseCnt, candCnt := 0, 0
		if basePass {
			baseCnt = 1
		}
		if candPass {
			candCnt = 1
		}
		baseRows[i] = TaskRepRow{ID: id, Accepted: []bool{basePass}, AcceptedCount: baseCnt}
		candRows[i] = TaskRepRow{ID: id, Accepted: []bool{candPass}, AcceptedCount: candCnt}
	}
	base := makeTasksSummary("q4", "extended", 105, 92, 7, 8, 1.0, baseRows)
	cand := makeTasksSummary("q2live", "extended", 105, 80, 13, 14, 1.0, candRows)

	got := EvaluateTasksGate(cand, &base, DefaultGatePolicy())
	if got.Verdict != GateFail {
		t.Fatalf("expected FAIL on planted regression, got %s %v", got.Verdict, got.Reasons)
	}
	if !slices.Contains(got.Reasons, ReasonAcceptRateRegression) {
		t.Fatalf("expected %s in %v", ReasonAcceptRateRegression, got.Reasons)
	}
	if !slices.Contains(got.Reasons, ReasonPairedTaskRegression) {
		t.Fatalf("expected %s in %v", ReasonPairedTaskRegression, got.Reasons)
	}
	if got.Paired == nil || got.Paired.CandidateWorse != 6 || got.Paired.CandidateBetter != 0 || math.Abs(got.Paired.TwoSidedSignP-0.03125) > 1e-6 {
		t.Fatalf("unexpected paired stats: %+v", got.Paired)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRunGateCLI(t *testing.T) {
	dir := t.TempDir()
	candDir := filepath.Join(dir, "cand")
	if err := os.MkdirAll(candDir, 0700); err != nil {
		t.Fatal(err)
	}
	passSummary := makeTasksSummary("qwen3.8", "default", 8, 7, 1, 1, 1.0, makeTaskRows(8, 7))
	if err := writeSummaryJSON(filepath.Join(candDir, "summary.json"), passSummary); err != nil {
		t.Fatal(err)
	}

	var out, errBuf bytes.Buffer
	code := RunCLI(t.Context(), &out, &errBuf, []string{"gate", "--candidate", candDir})
	if code != 0 {
		t.Fatalf("expected exit 0 for directory summary.json, got %d stderr=%s", code, errBuf.String())
	}
	var dec GateDecision
	if err := json.Unmarshal(out.Bytes(), &dec); err != nil {
		t.Fatalf("decode gate json: %v", err)
	}
	if dec.Verdict != GatePass {
		t.Fatalf("expected PASS, got %s", dec.Verdict)
	}

	// Failing summary -> exit 1
	failPath := filepath.Join(dir, "fail.json")
	failSummary := makeTasksSummary("bad", "default", 8, 2, 4, 4, 0.80, makeTaskRows(8, 2))
	if err := writeSummaryJSON(failPath, failSummary); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errBuf.Reset()
	if code := RunCLI(t.Context(), &out, &errBuf, []string{"gate", "--candidate", failPath}); code != 1 {
		t.Fatalf("expected exit 1 for FAIL, got %d", code)
	}

	// Inconclusive summary -> exit 3
	incPath := filepath.Join(dir, "inc.json")
	incSummary := makeTasksSummary("small", "default", 2, 2, 0, 0, 1.0, makeTaskRows(2, 2))
	if err := writeSummaryJSON(incPath, incSummary); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errBuf.Reset()
	if code := RunCLI(t.Context(), &out, &errBuf, []string{"gate", "--candidate", incPath}); code != 3 {
		t.Fatalf("expected exit 3 for INCONCLUSIVE, got %d", code)
	}

	// Missing candidate -> exit 2
	out.Reset()
	errBuf.Reset()
	if code := RunCLI(t.Context(), &out, &errBuf, []string{"gate"}); code != 2 {
		t.Fatalf("expected exit 2 for missing --candidate, got %d", code)
	}
}
