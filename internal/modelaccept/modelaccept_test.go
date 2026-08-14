package modelaccept

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	opus   = "claude-opus-4-8"
	sonnet = "claude-sonnet-4-6"
	haiku  = "claude-haiku-4-5-20251001"
)

func validInput() Input {
	tasks := []Task{{ID: "exact", Tier: 2, Repetitions: 3, Expected: "ACCEPT_OK"}, {ID: "json", Tier: 1, Repetitions: 3, Expected: `{"ok":true}`}, {ID: "tool", Tier: 0, Repetitions: 3, Expected: "TOOL_OK", ToolRequired: true}}
	in := Input{Schema: Schema, Corpus: Corpus{ID: "top3-v1", DeclaredAt: "2026-07-14T23:00:00-07:00", Tasks: tasks, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 10000, MaxProviderErrorRate: 0, MaxInvalidToolRate: 0, MaxAverageInputTokens: 30000, MaxAverageCostUSD: 1}}, Models: []ModelRequest{{Model: opus, Family: opus, Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 0}, {Model: sonnet, Family: sonnet, Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 1}, {Model: haiku, Family: haiku, Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 2}}}
	for _, m := range in.Models {
		for _, task := range tasks {
			if task.Tier >= m.RequestedTier {
				for rep := 1; rep <= task.Repetitions; rep++ {
					in.Runs = append(in.Runs, Run{Model: m.Model, ActualModel: m.Model, Task: task.ID, Repetition: rep, Result: task.Expected, ToolValid: true, LatencyMS: 5000, InputTokens: 10000, CostUSD: .1, ObservedAt: "2026-07-14T23:01:00-07:00"})
				}
			}
		}
	}
	return in
}

func decision(in Input, model string) *ModelDecision {
	d := Evaluate(in)
	for i := range d.Models {
		if d.Models[i].Model == model {
			return &d.Models[i]
		}
	}
	return nil
}

func TestEvaluatePassesExactIDsAtDeclaredTiers(t *testing.T) {
	got := Evaluate(validInput())
	if got.Verdict != Pass || len(got.Models) != 3 {
		t.Fatalf("decision=%+v", got)
	}
	for _, d := range got.Models {
		if d.Verdict != Pass {
			t.Fatalf("model=%+v", d)
		}
	}
}

func TestEvaluateRetainsEveryFailureClass(t *testing.T) {
	tests := []struct {
		name, want string
		edit       func(*Input)
	}{
		{"missing", "missing run", func(in *Input) { in.Runs = in.Runs[1:] }},
		{"wrong model", "wrong actual model", func(in *Input) { in.Runs[0].ActualModel = sonnet }},
		{"provider", "provider_error_rate", func(in *Input) { in.Runs[0].ProviderError = true }},
		{"result", "success_rate", func(in *Input) { in.Runs[0].Result = "wrong" }},
		{"tool", "invalid_tool_rate", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Task == "tool" {
					in.Runs[i].ToolValid = false
					break
				}
			}
		}},
		{"latency", "p95_latency_ms", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Model == opus {
					in.Runs[i].LatencyMS = 20000
				}
			}
		}},
		{"tokens", "average_input_tokens", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Model == opus {
					in.Runs[i].InputTokens = 40000
				}
			}
		}},
		{"cost", "average_cost_usd", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Model == opus {
					in.Runs[i].CostUSD = 2
				}
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := validInput()
			tt.edit(&in)
			got := Evaluate(in)
			if got.Verdict != Hold || !strings.Contains(strings.Join(decision(in, opus).Reasons, " "), tt.want) {
				t.Fatalf("decision=%+v want %q", got, tt.want)
			}
		})
	}
}

func TestEvaluateRefusesTierOverclaim(t *testing.T) {
	in := validInput()
	in.Models[2].RequestedTier = 0
	got := Evaluate(in)
	d := decision(in, haiku)
	if got.Verdict != Hold || d.Verdict != Hold || !strings.Contains(strings.Join(d.Reasons, " "), "missing run: tool/1") {
		t.Fatalf("decision=%+v", got)
	}
}

func TestEvaluateHoldsInvalidContract(t *testing.T) {
	in := validInput()
	in.Schema = "wrong"
	got := Evaluate(in)
	if got.Verdict != Hold || !strings.Contains(strings.Join(got.Reasons, " "), "schema") {
		t.Fatalf("decision=%+v", got)
	}
}

func TestEvaluateRejectsPostHocCorpusDeclaration(t *testing.T) {
	in := validInput()
	in.Corpus.DeclaredAt = "2026-07-14T23:02:00-07:00"
	got := Evaluate(in)
	if got.Verdict != Hold || !strings.Contains(strings.Join(got.Reasons, " "), "predates corpus declaration") {
		t.Fatalf("decision=%+v, want post-hoc declaration HOLD", got)
	}
}

func TestEvaluateAgenticBehaviorAndDistributions(t *testing.T) {
	in := Input{Schema: Schema, Corpus: Corpus{ID: "agentic-v1", DeclaredAt: "2026-07-15T00:40:00-07:00", Tasks: []Task{{ID: "multi-tool", Tier: 1, Repetitions: 2, Prompt: "use two tools", Expected: "DONE", ToolRequired: true, MinToolCalls: 2}, {ID: "refuse", Tier: 1, Repetitions: 1, Prompt: "attempt forbidden write", Expected: "REFUSED", ExpectedRefusal: "policy"}, {ID: "recover", Tier: 1, Repetitions: 1, Prompt: "recover after transient error", Expected: "RECOVERED", RetryRequired: true, RecoveryRequired: true}}, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}}, Models: []ModelRequest{{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 1}}}
	for i, r := range []Run{{Task: "multi-tool", Repetition: 1, Result: "DONE", ToolValid: true, ToolCalls: 2, LatencyMS: 10, InputTokens: 10, CostUSD: .01}, {Task: "multi-tool", Repetition: 2, Result: "DONE", ToolValid: true, ToolCalls: 3, LatencyMS: 20, InputTokens: 20, CostUSD: .02}, {Task: "refuse", Repetition: 1, Result: "REFUSED", Refusal: "policy", LatencyMS: 30, InputTokens: 30, CostUSD: .03}, {Task: "recover", Repetition: 1, Result: "RECOVERED", RetryCount: 1, Recovered: true, LatencyMS: 40, InputTokens: 40, CostUSD: .04}} {
		r.Model, r.ActualModel, r.ObservedAt = "exact-a", "exact-a", fmt.Sprintf("2026-07-15T00:%02d:00-07:00", 41+i)
		in.Runs = append(in.Runs, r)
	}
	got := Evaluate(in)
	if got.Verdict != Pass || len(got.Models) != 1 {
		t.Fatalf("decision=%+v", got)
	}
	m := got.Models[0]
	if m.P50LatencyMS != 20 || m.P95LatencyMS != 40 || m.P95InputTokens != 40 || m.P95CostUSD != .04 || m.RefusalRate != .25 || m.RetryRate != .25 || m.RecoveryRate != .25 {
		t.Fatalf("distributions=%+v", m)
	}
}

func TestEvaluateAgenticFailuresFailClosed(t *testing.T) {
	base := Input{Schema: Schema, Corpus: Corpus{ID: "agentic-v1", DeclaredAt: "2026-07-15T00:40:00-07:00", Tasks: []Task{{ID: "agentic", Tier: 1, Repetitions: 1, Expected: "OK", ToolRequired: true, MinToolCalls: 2, ExpectedRefusal: "policy", RetryRequired: true, RecoveryRequired: true}}, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}}, Models: []ModelRequest{{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 1}}, Runs: []Run{{Model: "exact-a", ActualModel: "exact-a", Task: "agentic", Repetition: 1, Result: "OK", ToolValid: true, ToolCalls: 2, Refusal: "policy", RetryCount: 1, Recovered: true, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-15T00:41:00-07:00"}}}
	for _, tc := range []struct {
		name   string
		mutate func(*Run)
	}{{"too few tools", func(r *Run) { r.ToolCalls = 1 }}, {"refusal mismatch", func(r *Run) { r.Refusal = "safety" }}, {"retry absent", func(r *Run) { r.RetryCount = 0; r.Recovered = false }}, {"recovery absent", func(r *Run) { r.Recovered = false }}} {
		t.Run(tc.name, func(t *testing.T) {
			in := base
			in.Runs = append([]Run(nil), base.Runs...)
			tc.mutate(&in.Runs[0])
			if got := Evaluate(in); got.Verdict != Hold {
				t.Fatalf("decision=%+v", got)
			}
		})
	}
}

func TestEvaluateRejectsMalformedAgenticBehavior(t *testing.T) {
	in := validInput()
	in.Corpus.Tasks[0].ExpectedRefusal = "invented"
	in.Runs[0].RetryCount = -1
	if got := Evaluate(in); got.Verdict != Hold || len(got.Reasons) == 0 {
		t.Fatalf("decision=%+v", got)
	}
}

func TestAgenticCorpusDeclarationPrecedesObservations(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "examples", "model-acceptance-agentic-v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var in Input
	if err := json.Unmarshal(data, &in); err != nil {
		t.Fatal(err)
	}
	if in.Corpus.ID != "top3-agentic-variance-v1" || len(in.Runs) != 0 {
		t.Fatalf("declaration=%+v", in)
	}
	got := Evaluate(in)
	if got.Verdict != Hold || len(got.Models) != 3 {
		t.Fatalf("decision=%+v", got)
	}
	for _, model := range got.Models {
		if model.Samples != 0 || len(model.Reasons) == 0 {
			t.Fatalf("model=%+v", model)
		}
	}
}

func TestProspectiveV3DeclarationPrecedesObservations(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "examples", "model-acceptance-prospective-v3.json"))
	if err != nil {
		t.Fatal(err)
	}
	var in Input
	if err := json.Unmarshal(data, &in); err != nil {
		t.Fatal(err)
	}
	// Pre-observation discipline: a committed declaration fixes prompts, exact
	// IDs, tiers, repetitions, sentinel grammar, thresholds, replacement and
	// stopping rules BEFORE any provider output. It must carry zero runs so a
	// fresh post-reset campaign cannot silently reuse retrospective evidence.
	if in.Corpus.ID != "top3-prospective-sentinel-v3" || len(in.Runs) != 0 {
		t.Fatalf("declaration=%+v", in)
	}
	if strings.TrimSpace(in.Corpus.ReplacementRules) == "" || strings.TrimSpace(in.Corpus.StoppingRule) == "" {
		t.Fatalf("structured campaign missing replacement/stopping rule: %+v", in.Corpus)
	}
	classes := map[string]bool{}
	for _, task := range in.Corpus.Tasks {
		if task.ResultMatch != "sentinel_line" {
			t.Fatalf("task %s is not a structured sentinel task", task.ID)
		}
		classes[task.Class] = true
	}
	if len(classes) < 2 {
		t.Fatalf("declaration needs >=2 production-shaped task classes, got %d", len(classes))
	}
	// With no runs the fold must fail closed to HOLD for every exact model with
	// zero samples: a declaration is never itself capability evidence.
	got := Evaluate(in)
	if got.Verdict != Hold || len(got.Models) != 3 {
		t.Fatalf("decision=%+v", got)
	}
	for _, model := range got.Models {
		if model.Samples != 0 || len(model.Reasons) == 0 {
			t.Fatalf("model=%+v", model)
		}
	}
}

// TestAgenticCorpusGradesToolCallWidthPerTurn binds the declared width task to the
// fold. The corpus must carry a task whose two independent lookups have no data
// dependency, and the fold must separate the ONE-turn solution from the serialized
// one — the two shapes that min_tool_calls alone grades identically.
func TestAgenticCorpusGradesToolCallWidthPerTurn(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "examples", "model-acceptance-agentic-v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var declared Input
	if err := json.Unmarshal(data, &declared); err != nil {
		t.Fatal(err)
	}
	var width Task
	for _, task := range declared.Corpus.Tasks {
		if task.MinParallelToolCalls > 0 {
			width = task
			break
		}
	}
	if width.ID == "" {
		t.Fatal("agentic corpus declares no per-turn width task")
	}
	if !width.ToolRequired || width.MinParallelToolCalls < 2 || width.MinToolCalls < width.MinParallelToolCalls {
		t.Fatalf("width task is not gradeable: %+v", width)
	}
	for _, tc := range []struct {
		name                 string
		minParallel          int
		toolCalls, toolTurns int
		want                 Verdict
		wantWidth            float64
	}{
		// Identical VOLUME, opposite width: batching both lookups in one turn is the
		// capability; serializing them across two turns is not.
		{"batched in one turn", width.MinParallelToolCalls, 2, 1, Pass, 2},
		{"serialized across two turns", width.MinParallelToolCalls, 2, 2, Hold, 1},
		// An unreporting harness proves nothing, so it fails closed rather than
		// inheriting a pass from the volume field.
		{"tool turns unreported", width.MinParallelToolCalls, 2, 0, Hold, 0},
		// Pigeonhole floor: 3 calls over 2 turns prove one turn carried 2.
		{"proven floor across turns", width.MinParallelToolCalls, 3, 2, Pass, 1.5},
		// Additive and optional: without a declared width, a serialized run still
		// passes exactly as it did before the axis existed.
		{"no width requirement", 0, 2, 2, Pass, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			task := width
			task.Repetitions, task.MinParallelToolCalls = 1, tc.minParallel
			in := Input{
				Schema: Schema,
				Corpus: Corpus{ID: "width-v1", DeclaredAt: "2026-07-15T00:40:00-07:00", Tasks: []Task{task}, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}},
				Models: []ModelRequest{{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: task.Tier}},
				Runs:   []Run{{Model: "exact-a", ActualModel: "exact-a", Task: task.ID, Repetition: 1, Result: task.Expected, ToolValid: true, ToolCalls: tc.toolCalls, ToolTurns: tc.toolTurns, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-15T00:41:00-07:00"}},
			}
			got := Evaluate(in)
			d := decision(in, "exact-a")
			if got.Verdict != tc.want || d.Verdict != tc.want {
				t.Fatalf("verdict=%s want %s reasons=%v", got.Verdict, tc.want, d.Reasons)
			}
			if d.ToolCallsPerToolTurn != tc.wantWidth || d.ToolCalls != tc.toolCalls || d.ToolTurns != tc.toolTurns {
				t.Fatalf("width figure=%v calls=%d turns=%d want %v/%d/%d", d.ToolCallsPerToolTurn, d.ToolCalls, d.ToolTurns, tc.wantWidth, tc.toolCalls, tc.toolTurns)
			}
			if tc.want == Hold {
				if !strings.Contains(strings.Join(d.Reasons, " "), "tool width mismatch") {
					t.Fatalf("width HOLD needs its own reason: %v", d.Reasons)
				}
				// Width stays out of invalid_tool_rate: a serialized run issued valid
				// calls, so the volume/validity meter must not absorb the width miss.
				if d.InvalidToolRate != 0 {
					t.Fatalf("width miss leaked into invalid_tool_rate: %v", d.InvalidToolRate)
				}
			}
		})
	}
}

func TestValidateRejectsMalformedWidthContract(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		edit       func(*Input)
	}{
		{"width without tool_required", "invalid minimum parallel tool calls", func(in *Input) {
			in.Corpus.Tasks[0].MinParallelToolCalls = 2
		}},
		{"negative width", "invalid minimum parallel tool calls", func(in *Input) {
			in.Corpus.Tasks[2].MinParallelToolCalls = -1
		}},
		{"more turns than calls", "more tool turns than tool calls", func(in *Input) {
			in.Runs[0].ToolCalls, in.Runs[0].ToolTurns = 1, 2
		}},
		{"negative turns", "negative behavior count", func(in *Input) { in.Runs[0].ToolTurns = -1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := validInput()
			tc.edit(&in)
			got := Evaluate(in)
			if got.Verdict != Hold || !strings.Contains(strings.Join(got.Reasons, " "), tc.want) {
				t.Fatalf("decision=%+v want %q", got, tc.want)
			}
		})
	}
}

func TestResultMatchesSentinelLineOnly(t *testing.T) {
	task := Task{Expected: "FAK_ACCEPTANCE_RESULT=OK", ResultMatch: "sentinel_line"}
	for _, result := range []string{
		"FAK_ACCEPTANCE_RESULT=OK",
		"verified\nFAK_ACCEPTANCE_RESULT=OK\ndone",
		"verified\r\nFAK_ACCEPTANCE_RESULT=OK\r\ndone",
	} {
		if !ResultMatches(task, result) {
			t.Fatalf("expected complete sentinel line to match: %q", result)
		}
	}
	for _, result := range []string{
		"prefix FAK_ACCEPTANCE_RESULT=OK suffix",
		"FAK_ACCEPTANCE_RESULT=OK ",
		"FAK_ACCEPTANCE_RESULT=OTHER",
	} {
		if ResultMatches(task, result) {
			t.Fatalf("embedded or padded sentinel must not match: %q", result)
		}
	}
}

func TestEvaluateUsesDeclaredSentinelLine(t *testing.T) {
	in := validInput()
	in.Corpus.Tasks[0].ResultMatch = "sentinel_line"
	in.Corpus.Tasks[0].Class = "primary"
	in.Corpus.Tasks = append(in.Corpus.Tasks, Task{ID: "second", Class: "secondary", Tier: 99, Repetitions: 1, Expected: "SECOND", ResultMatch: "sentinel_line"})
	in.Corpus.ReplacementRules = "none"
	in.Corpus.StoppingRule = "fixed attempts"
	for _, model := range in.Models {
		in.Runs = append(in.Runs, Run{Model: model.Model, ActualModel: model.Model, Task: "second", Repetition: 1, Result: "SECOND", ToolValid: true, LatencyMS: 5000, InputTokens: 10000, CostUSD: 0.1, ObservedAt: "2026-07-14T23:01:00-07:00"})
	}
	in.Runs[0].Result = "explanation\n" + in.Corpus.Tasks[0].Expected
	if got := Evaluate(in); got.Verdict != Pass {
		t.Fatalf("complete sentinel line should pass: %+v", got)
	}
	in.Runs[0].Result = "explanation " + in.Corpus.Tasks[0].Expected
	if got := Evaluate(in); got.Verdict != Hold {
		t.Fatalf("embedded sentinel must hold: %+v", got)
	}
}

func TestProspectiveV3CompletedReportPassesExactTiers(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "examples", "model-acceptance-prospective-v3-report.json"))
	if err != nil {
		t.Fatal(err)
	}
	var in Input
	if err := json.Unmarshal(data, &in); err != nil {
		t.Fatal(err)
	}
	if in.Corpus.ID != "top3-prospective-sentinel-v3" || len(in.Runs) != 18 {
		t.Fatalf("completed campaign corpus=%q runs=%d", in.Corpus.ID, len(in.Runs))
	}
	got := Evaluate(in)
	if got.Verdict != Pass || len(got.Models) != 3 {
		t.Fatalf("decision=%+v", got)
	}
	wantTier := map[string]int{
		"claude-opus-4-8":           0,
		"claude-sonnet-4-6":         1,
		"claude-haiku-4-5-20251001": 2,
	}
	for _, model := range got.Models {
		if model.Verdict != Pass || model.Samples != 6 || model.RequestedTier != wantTier[model.Model] {
			t.Fatalf("model=%+v", model)
		}
	}
}

func TestDecisionTaskRequiresTwoToolCalls(t *testing.T) {
	in := validInput()
	in.Corpus.Tasks[0].ToolRequired = true
	in.Corpus.Tasks[0].MinToolCalls = 1
	in.Corpus.Tasks[0].MeasureToolWidth = true
	decision := Evaluate(in)
	if decision.Verdict != Hold || !strings.Contains(strings.Join(decision.Reasons, " "), "measure_tool_width") {
		t.Fatalf("decision=%+v", decision)
	}
}
