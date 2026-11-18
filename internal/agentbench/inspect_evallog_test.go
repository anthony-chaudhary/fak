package agentbench

import (
	"encoding/json"
	"strings"
	"testing"
)

// testRunResult builds a valid RunResult for the happy-path projection.
func testRunResult() RunResult {
	return RunResult{
		Arm:   "candidate",
		Model: "fixture-model",
		Suite: "default",
		Tasks: []taskrunTaskReceipt{
			{
				ID:        "task-a",
				Family:    "c2",
				Passed:    true,
				Accepted:  true,
				Model:     taskrunModelObservation{Requests: 4, UpstreamRequests: 4, SuccessfulRequests: 4, ModelIdentityStatus: "matched"},
				ToolCalls: taskrunToolCallMetrics{Measured: true, Total: 3, Invalid: 0, Validity: ptrFloat(1.0)},
			},
			{
				ID:        "task-b",
				Family:    "c2",
				Passed:    false,
				Accepted:  true,
				Model:     taskrunModelObservation{Requests: 2, UpstreamRequests: 2, SuccessfulRequests: 2, ModelIdentityStatus: "matched"},
				ToolCalls: taskrunToolCallMetrics{Measured: true, Total: 1, Invalid: 1, Validity: ptrFloat(0.0)},
			},
		},
	}
}

func ptrFloat(f float64) *float64 { return &f }

func TestInspectEvalLog(t *testing.T) {
	t.Run("schema shape", func(t *testing.T) {
		log, err := ExportInspectEvalLog(testRunResult())
		if err != nil {
			t.Fatalf("ExportInspectEvalLog = %v", err)
		}
		if log.Version != 2 {
			t.Fatalf("version = %d, want 2", log.Version)
		}
		if log.Status != "success" {
			t.Fatalf("status = %q, want success", log.Status)
		}
		if log.Eval.Task != inspectTaskName {
			t.Fatalf("eval.task = %q, want %q", log.Eval.Task, inspectTaskName)
		}
		if log.Eval.Model != "fixture-model" {
			t.Fatalf("eval.model = %q, want fixture-model", log.Eval.Model)
		}
		if log.Eval.Dataset.Samples != 2 {
			t.Fatalf("eval.dataset.samples = %d, want 2", log.Eval.Dataset.Samples)
		}
		if log.Plan.Name != "plan" || len(log.Plan.Steps) != 1 || log.Plan.Steps[0].Solver != "agentbench" {
			t.Fatalf("plan = %+v", log.Plan)
		}
		if log.Results == nil {
			t.Fatalf("results is nil")
		}
		if log.Results.TotalSamples != 2 || log.Results.CompletedSamples != 2 {
			t.Fatalf("results totals = %+v", log.Results)
		}
		if log.Metadata["projection"] != inspectEvalLogSchema {
			t.Fatalf("metadata.projection = %v, want %q", log.Metadata["projection"], inspectEvalLogSchema)
		}
		if log.Metadata["arm"] != "candidate" {
			t.Fatalf("metadata.arm = %v, want candidate", log.Metadata["arm"])
		}
		if len(log.Samples) != 2 {
			t.Fatalf("samples = %d, want 2", len(log.Samples))
		}
	})

	t.Run("samples", func(t *testing.T) {
		log, err := ExportInspectEvalLog(testRunResult())
		if err != nil {
			t.Fatalf("ExportInspectEvalLog = %v", err)
		}
		for i, s := range log.Samples {
			if s.Epoch != 1 {
				t.Fatalf("sample %d epoch = %d, want 1", i, s.Epoch)
			}
			if s.ID == "" || s.Input == "" || s.Target != "pass" {
				t.Fatalf("sample %d identity = %+v", i, s)
			}
			if len(s.Messages) != 1 || s.Messages[0].Role != "assistant" {
				t.Fatalf("sample %d messages = %+v", i, s.Messages)
			}
			if s.Output.Model != "fixture-model" || len(s.Output.Choices) != 1 {
				t.Fatalf("sample %d output = %+v", i, s.Output)
			}
			if s.Output.Choices[0].StopReason != "stop" {
				t.Fatalf("sample %d stop_reason = %q", i, s.Output.Choices[0].StopReason)
			}
			if s.Output.Usage == nil {
				t.Fatalf("sample %d usage is nil", i)
			}
			if len(s.Scores) != 1 {
				t.Fatalf("sample %d scores = %d, want 1", i, len(s.Scores))
			}
			if len(s.Events) != 2 {
				t.Fatalf("sample %d events = %d, want 2", i, len(s.Events))
			}
		}
		// task-a passed, task-b failed.
		if log.Samples[0].Scores["task_passed"].Answer != "passed" {
			t.Fatalf("task-a answer = %q, want passed", log.Samples[0].Scores["task_passed"].Answer)
		}
		if log.Samples[1].Scores["task_passed"].Answer != "failed" {
			t.Fatalf("task-b answer = %q, want failed", log.Samples[1].Scores["task_passed"].Answer)
		}
	})

	t.Run("model and score events", func(t *testing.T) {
		log, err := ExportInspectEvalLog(testRunResult())
		if err != nil {
			t.Fatalf("ExportInspectEvalLog = %v", err)
		}
		for i, s := range log.Samples {
			modelEvent, ok := s.Events[0].(InspectModelEvent)
			if !ok {
				t.Fatalf("sample %d event 0 is %T, want InspectModelEvent", i, s.Events[0])
			}
			if modelEvent.Event != "model" || modelEvent.Model != "fixture-model" {
				t.Fatalf("sample %d model event = %+v", i, modelEvent)
			}
			if len(modelEvent.Input) != 1 || modelEvent.Input[0].Role != "user" {
				t.Fatalf("sample %d model event input = %+v", i, modelEvent.Input)
			}
			if modelEvent.ToolChoice != "auto" {
				t.Fatalf("sample %d tool_choice = %q", i, modelEvent.ToolChoice)
			}
			scoreEvent, ok := s.Events[1].(InspectScoreEvent)
			if !ok {
				t.Fatalf("sample %d event 1 is %T, want InspectScoreEvent", i, s.Events[1])
			}
			if scoreEvent.Event != "score" || scoreEvent.Scorer != "task_passed" {
				t.Fatalf("sample %d score event = %+v", i, scoreEvent)
			}
			if scoreEvent.Target != "pass" {
				t.Fatalf("sample %d score target = %q", i, scoreEvent.Target)
			}
		}
	})

	t.Run("scores", func(t *testing.T) {
		log, err := ExportInspectEvalLog(testRunResult())
		if err != nil {
			t.Fatalf("ExportInspectEvalLog = %v", err)
		}
		if len(log.Results.Scores) != 2 {
			t.Fatalf("results.scores = %d, want 2", len(log.Results.Scores))
		}
		passRate, ok := log.Results.Scores[0].Metrics["mean"]
		if !ok || log.Results.Scores[0].Name != "pass_rate" {
			t.Fatalf("pass_rate score = %+v", log.Results.Scores[0])
		}
		if passRate.Value != 0.5 {
			t.Fatalf("pass_rate mean = %v, want 0.5", passRate.Value)
		}
		toolValidity, ok := log.Results.Scores[1].Metrics["mean"]
		if !ok || log.Results.Scores[1].Name != "tool_call_validity" {
			t.Fatalf("tool_call_validity score = %+v", log.Results.Scores[1])
		}
		if toolValidity.Value != 0.5 {
			t.Fatalf("tool_call_validity mean = %v, want 0.5", toolValidity.Value)
		}
	})

	t.Run("json round trip", func(t *testing.T) {
		log, err := ExportInspectEvalLog(testRunResult())
		if err != nil {
			t.Fatalf("ExportInspectEvalLog = %v", err)
		}
		body, err := json.Marshal(log)
		if err != nil {
			t.Fatalf("marshal = %v", err)
		}
		var decoded map[string]any
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("unmarshal = %v", err)
		}
		if decoded["version"].(float64) != 2 {
			t.Fatalf("json version = %v, want 2", decoded["version"])
		}
		samples, ok := decoded["samples"].([]any)
		if !ok || len(samples) != 2 {
			t.Fatalf("json samples = %v", decoded["samples"])
		}
		first := samples[0].(map[string]any)
		events := first["events"].([]any)
		if events[0].(map[string]any)["event"] != "model" {
			t.Fatalf("json first event = %v", events[0])
		}
		if events[1].(map[string]any)["event"] != "score" {
			t.Fatalf("json second event = %v", events[1])
		}
	})

	t.Run("fail closed on unknown shape", func(t *testing.T) {
		cases := map[string]func(RunResult) RunResult{
			"empty arm":         func(r RunResult) RunResult { r.Arm = ""; return r },
			"empty model":       func(r RunResult) RunResult { r.Model = ""; return r },
			"no tasks":          func(r RunResult) RunResult { r.Tasks = nil; return r },
			"empty task id":     func(r RunResult) RunResult { r.Tasks[0].ID = ""; return r },
			"no observation":    func(r RunResult) RunResult { r.Tasks[0].Model.Requests = 0; return r },
			"inconsistent reqs": func(r RunResult) RunResult { r.Tasks[0].Model.UpstreamRequests = 99; return r },
			"unknown identity":  func(r RunResult) RunResult { r.Tasks[0].Model.ModelIdentityStatus = "weird"; return r },
		}
		for name, mutate := range cases {
			t.Run(name, func(t *testing.T) {
				_, err := ExportInspectEvalLog(mutate(testRunResult()))
				if err == nil {
					t.Fatalf("expected refusal for %s", name)
				}
				if !inspectEvalLogIsRefused(err) {
					t.Fatalf("refusal = %v, want wrapped refusal", err)
				}
				if !strings.Contains(err.Error(), "refused") {
					t.Fatalf("refusal text = %q", err.Error())
				}
			})
		}
	})

	t.Run("comparison projection", func(t *testing.T) {
		receipt := compareDirectoryReceipt{
			Schema:    "fak.agentbench.comparison.v1",
			Status:    "COMPARABLE",
			Candidate: comparisonArmArtifact{ArmID: "candidate", ModelID: "fixture-model"},
			Comparison: ComparisonResult{
				Metrics: map[string]PairedMetric{
					"accepted_task_ms": {Pairs: 5, MedianDeltaMillis: 10, BootstrapLowMillis: -5, BootstrapHighMillis: 25, Confidence: .95},
				},
			},
		}
		log, err := InspectEvalLogFromComparison(receipt)
		if err != nil {
			t.Fatalf("InspectEvalLogFromComparison = %v", err)
		}
		if log.Eval.Model != "fixture-model" || log.Eval.Dataset.Name != "comparison" {
			t.Fatalf("comparison eval = %+v", log.Eval)
		}
		if len(log.Samples) != 1 || log.Samples[0].ID != "metric-1-accepted_task_ms" {
			t.Fatalf("comparison samples = %+v", log.Samples)
		}
		// The interval straddles zero, so the metric is "passed".
		if log.Samples[0].Scores["task_passed"].Answer != "passed" {
			t.Fatalf("comparison answer = %q, want passed", log.Samples[0].Scores["task_passed"].Answer)
		}
	})

	t.Run("comparison fail closed", func(t *testing.T) {
		cases := map[string]compareDirectoryReceipt{
			"unknown schema": {Schema: "other", Status: "COMPARABLE", Candidate: comparisonArmArtifact{ArmID: "c", ModelID: "m"}},
			"unknown status": {Schema: "fak.agentbench.comparison.v1", Status: "INCONCLUSIVE", Candidate: comparisonArmArtifact{ArmID: "c", ModelID: "m"}},
			"no model":       {Schema: "fak.agentbench.comparison.v1", Status: "COMPARABLE", Candidate: comparisonArmArtifact{ArmID: "c"}},
			"no metrics":     {Schema: "fak.agentbench.comparison.v1", Status: "COMPARABLE", Candidate: comparisonArmArtifact{ArmID: "c", ModelID: "m"}},
		}
		for name, receipt := range cases {
			t.Run(name, func(t *testing.T) {
				_, err := InspectEvalLogFromComparison(receipt)
				if err == nil || !inspectEvalLogIsRefused(err) {
					t.Fatalf("expected refusal for %s, got %v", name, err)
				}
			})
		}
	})
}
