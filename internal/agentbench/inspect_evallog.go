package agentbench

// Inspect AI eval-log export projection.
//
// Schema adoption (MIT) from UKGovernmentBEIS/inspect_ai, pinned at
// src/inspect_ai/log/_log.py:1;src/inspect_ai/log/_transcript.py:1 @
// b9768e290af000efae66bcc272d9c8af6a710334. The JSON field names and
// nesting below mirror the pinned Inspect EvalLog (version 2) shape:
// the top-level log carries version/status/eval/plan/results/stats/
// metadata/samples; each sample carries id/epoch/input/target/
// messages/output/scores/events; each event is discriminated by its
// "event" field ("model" for a ModelEvent, "score" for a ScoreEvent);
// and each score carries value/answer/explanation. No Python harness is
// ported; this is a host-side projection from agentbench run results
// only. No benchmark claims.

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

// inspectEvalLogSchema is the projection receipt schema.
const inspectEvalLogSchema = "fak.agentbench.inspect-eval-log.v1"

// inspectEvalLogVersion is the pinned Inspect eval-log file format
// version (EvalLog.version, default 2).
const inspectEvalLogVersion = 2

// inspectTaskName is the task name carried on projected logs.
const inspectTaskName = "fak-agentbench"

// inspectEvalLogTimestamp is the deterministic UTC timestamp carried on
// projected logs. agentbench receipts carry per-event times, but the
// projection is a pure function of the result shape; a fixed timestamp
// keeps the output byte-stable for a given input.
const inspectEvalLogTimestamp = "2026-01-01T00:00:00.000000+00:00"

// RunResult is the minimal agentbench run-result shape the Inspect
// eval-log projection consumes. It is a projection input, not a new
// receipt: callers fill it from a taskrun.Receipt (or a comparison
// receipt's candidate arm) and hand it to ExportInspectEvalLog.
//
// The shape is deliberately narrow: arm identity, model, suite, and the
// per-task receipts. Endpoint, filesystem paths, and prompt content are
// out of scope for the projection (privacy policy governs content
// export; the eval log carries task identity and model-observation
// counts only).
type RunResult struct {
	// Arm is the arm identity the run belongs to (e.g. "candidate").
	Arm string
	// Model is the resolved model ID the run served.
	Model string
	// Suite is the task suite name (e.g. "default").
	Suite string
	// Tasks are the per-task receipts, in run order.
	Tasks []taskrunTaskReceipt
}

// taskrunTaskReceipt is the projection's view of a taskrun.TaskReceipt.
// It mirrors the fields the Inspect eval log carries: task identity,
// pass/accept verdicts, and the model-observation counts.
type taskrunTaskReceipt struct {
	ID             string
	Family         string
	Passed         bool
	Accepted       bool
	ExternalPassed bool
	// Model is the task's model observation.
	Model taskrunModelObservation
	// ToolCalls is the task's tool-call metrics.
	ToolCalls taskrunToolCallMetrics
}

// taskrunModelObservation mirrors taskrun.ModelObservation's projection
// fields: request counts and identity status.
type taskrunModelObservation struct {
	Requests            int
	UpstreamRequests    int
	SuccessfulRequests  int
	ModelIdentityStatus string
}

// taskrunToolCallMetrics mirrors taskrun.ToolCallMetrics' projection
// fields: measured flag, total/invalid counts, and validity.
type taskrunToolCallMetrics struct {
	Measured bool
	Total    int
	Invalid  int
	Validity *float64
}

// InspectEvalLog is the projected Inspect AI eval log. Field names and
// nesting mirror the pinned Inspect EvalLog (version 2) shape.
type InspectEvalLog struct {
	Version  int                 `json:"version"`
	Status   string              `json:"status"`
	Eval     InspectEvalSpec     `json:"eval"`
	Plan     InspectEvalPlan     `json:"plan"`
	Results  *InspectEvalResults `json:"results"`
	Stats    InspectEvalStats    `json:"stats"`
	Metadata map[string]any      `json:"metadata"`
	Samples  []InspectEvalSample `json:"samples"`
}

// InspectEvalSpec mirrors the pinned Inspect EvalSpec shape (the subset
// the projection fills).
type InspectEvalSpec struct {
	EvalID      string             `json:"eval_id,omitempty"`
	RunID       string             `json:"run_id,omitempty"`
	Created     string             `json:"created"`
	Task        string             `json:"task"`
	TaskID      string             `json:"task_id,omitempty"`
	TaskVersion int                `json:"task_version,omitempty"`
	Dataset     InspectEvalDataset `json:"dataset"`
	Model       string             `json:"model"`
	Config      InspectEvalConfig  `json:"config"`
	ModelArgs   map[string]any     `json:"model_args,omitempty"`
}

// InspectEvalDataset mirrors the pinned Inspect EvalDataset shape.
type InspectEvalDataset struct {
	Name    string `json:"name,omitempty"`
	Samples int    `json:"samples,omitempty"`
}

// InspectEvalConfig mirrors the pinned Inspect EvalConfig shape (the
// subset the projection fills).
type InspectEvalConfig struct {
	LogSamples bool `json:"log_samples"`
}

// InspectEvalPlan mirrors the pinned Inspect EvalPlan shape.
type InspectEvalPlan struct {
	Name  string                `json:"name"`
	Steps []InspectEvalPlanStep `json:"steps"`
}

// InspectEvalPlanStep mirrors the pinned Inspect EvalPlanStep shape.
type InspectEvalPlanStep struct {
	Solver string `json:"solver"`
}

// InspectEvalResults mirrors the pinned Inspect EvalResults shape.
type InspectEvalResults struct {
	TotalSamples     int                `json:"total_samples"`
	CompletedSamples int                `json:"completed_samples"`
	Scores           []InspectEvalScore `json:"scores"`
}

// InspectEvalScore mirrors the pinned Inspect EvalScore shape.
type InspectEvalScore struct {
	Name          string                       `json:"name"`
	Scorer        string                       `json:"scorer"`
	ScoredSamples int                          `json:"scored_samples"`
	Metrics       map[string]InspectEvalMetric `json:"metrics"`
}

// InspectEvalMetric mirrors the pinned Inspect EvalMetric shape.
type InspectEvalMetric struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

// InspectEvalStats mirrors the pinned Inspect EvalStats shape.
type InspectEvalStats struct {
	StartedAt   string                       `json:"started_at"`
	CompletedAt string                       `json:"completed_at"`
	ModelUsage  map[string]InspectModelUsage `json:"model_usage"`
}

// InspectEvalSample mirrors the pinned Inspect EvalSample shape (the
// subset the projection fills).
type InspectEvalSample struct {
	ID          string                  `json:"id"`
	Epoch       int                     `json:"epoch"`
	Input       string                  `json:"input"`
	Target      string                  `json:"target"`
	Messages    []InspectChatMessage    `json:"messages"`
	Output      InspectModelOutput      `json:"output"`
	Scores      map[string]InspectScore `json:"scores"`
	Events      []any                   `json:"events"`
	StartedAt   string                  `json:"started_at"`
	CompletedAt string                  `json:"completed_at"`
}

// InspectChatMessage mirrors the pinned Inspect ChatMessageAssistant
// shape (the subset the projection fills).
type InspectChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Model   string `json:"model,omitempty"`
}

// InspectModelOutput mirrors the pinned Inspect ModelOutput shape (the
// subset the projection fills).
type InspectModelOutput struct {
	Model   string             `json:"model"`
	Choices []InspectChoice    `json:"choices"`
	Usage   *InspectModelUsage `json:"usage,omitempty"`
}

// InspectChoice mirrors the pinned Inspect ChatCompletionChoice shape.
type InspectChoice struct {
	Message    InspectChatMessage `json:"message"`
	StopReason string             `json:"stop_reason"`
}

// InspectModelUsage mirrors the pinned Inspect ModelUsage shape.
type InspectModelUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// InspectScore mirrors the pinned Inspect Score shape.
type InspectScore struct {
	Value       any    `json:"value"`
	Answer      string `json:"answer"`
	Explanation string `json:"explanation,omitempty"`
}

// InspectModelEvent mirrors the pinned Inspect ModelEvent shape (the
// subset the projection fills). The "event" discriminator is "model".
type InspectModelEvent struct {
	Event      string               `json:"event"`
	UUID       string               `json:"uuid"`
	Timestamp  string               `json:"timestamp"`
	Model      string               `json:"model"`
	Input      []InspectChatMessage `json:"input"`
	Tools      []any                `json:"tools"`
	ToolChoice string               `json:"tool_choice"`
	Output     InspectModelOutput   `json:"output"`
}

// InspectScoreEvent mirrors the pinned Inspect ScoreEvent shape (the
// subset the projection fills). The "event" discriminator is "score".
type InspectScoreEvent struct {
	Event     string       `json:"event"`
	UUID      string       `json:"uuid"`
	Timestamp string       `json:"timestamp"`
	Score     InspectScore `json:"score"`
	Target    string       `json:"target"`
	Scorer    string       `json:"scorer"`
}

// errInspectEvalLogRefused is the fail-closed refusal for unknown
// result shapes.
var errInspectEvalLogRefused = errors.New("inspect eval-log projection refused")

// validateRunResult fail-closed checks the RunResult shape. An unknown
// shape (empty arm/model, empty task ID/family, mismatched request
// counts, or a task with no model observation) is refused, never
// fabricated into a log.
func validateRunResult(r RunResult) error {
	if r.Arm == "" {
		return fmt.Errorf("%w: arm identity is empty", errInspectEvalLogRefused)
	}
	if r.Model == "" {
		return fmt.Errorf("%w: model identity is empty", errInspectEvalLogRefused)
	}
	if len(r.Tasks) == 0 {
		return fmt.Errorf("%w: run result carries no tasks", errInspectEvalLogRefused)
	}
	for i, task := range r.Tasks {
		if task.ID == "" || task.Family == "" {
			return fmt.Errorf("%w: task %d has empty identity", errInspectEvalLogRefused, i)
		}
		if task.Model.Requests == 0 {
			return fmt.Errorf("%w: task %s has no model observation", errInspectEvalLogRefused, task.ID)
		}
		if task.Model.Requests != task.Model.UpstreamRequests || task.Model.UpstreamRequests != task.Model.SuccessfulRequests {
			return fmt.Errorf("%w: task %s model observation request counts are inconsistent", errInspectEvalLogRefused, task.ID)
		}
		if task.Model.ModelIdentityStatus != "matched" && task.Model.ModelIdentityStatus != "observed" {
			return fmt.Errorf("%w: task %s model identity status %q is unknown", errInspectEvalLogRefused, task.ID, task.Model.ModelIdentityStatus)
		}
	}
	return nil
}

// ExportInspectEvalLog projects a RunResult into an Inspect AI eval
// log. It fail-closed refuses unknown result shapes (see
// validateRunResult). The projection is deterministic: a given
// RunResult yields a byte-stable InspectEvalLog.
func ExportInspectEvalLog(r RunResult) (InspectEvalLog, error) {
	if err := validateRunResult(r); err != nil {
		return InspectEvalLog{}, err
	}

	// Per-task samples, events, and scores.
	passed := 0
	toolCallMeasured := 0
	toolCallValid := 0
	totalInput := 0
	totalOutput := 0
	samples := make([]InspectEvalSample, 0, len(r.Tasks))
	for i, task := range r.Tasks {
		sampleID := task.ID
		verdict := "failed"
		if task.Passed && task.Accepted {
			verdict = "passed"
			passed++
		}
		if task.ToolCalls.Measured {
			toolCallMeasured++
			if task.ToolCalls.Invalid == 0 {
				toolCallValid++
			}
		}
		totalInput += task.Model.Requests
		totalOutput += task.Model.SuccessfulRequests

		// The sample's model output: a single assistant choice
		// carrying the task verdict, with the model-observation
		// counts as usage.
		usage := InspectModelUsage{
			InputTokens:  task.Model.Requests,
			OutputTokens: task.Model.SuccessfulRequests,
			TotalTokens:  task.Model.Requests + task.Model.SuccessfulRequests,
		}
		assistant := InspectChatMessage{
			Role:    "assistant",
			Content: "task " + task.ID + " " + verdict,
			Model:   r.Model,
		}
		output := InspectModelOutput{
			Model:   r.Model,
			Choices: []InspectChoice{{Message: assistant, StopReason: "stop"}},
			Usage:   &usage,
		}

		// Per-sample events: a model event carrying the
		// model-observation counts, then a score event carrying the
		// pass verdict.
		modelEvent := InspectModelEvent{
			Event:      "model",
			UUID:       fmt.Sprintf("e%d-1", i+1),
			Timestamp:  inspectEvalLogTimestamp,
			Model:      r.Model,
			Input:      []InspectChatMessage{{Role: "user", Content: "task " + task.ID + " " + task.Family}},
			Tools:      []any{},
			ToolChoice: "auto",
			Output:     output,
		}
		scoreValue := "I"
		if verdict == "passed" {
			scoreValue = "C"
		}
		score := InspectScore{
			Value:       scoreValue,
			Answer:      verdict,
			Explanation: "task " + task.ID + " " + task.Family,
		}
		scoreEvent := InspectScoreEvent{
			Event:     "score",
			UUID:      fmt.Sprintf("e%d-2", i+1),
			Timestamp: inspectEvalLogTimestamp,
			Score:     score,
			Target:    "pass",
			Scorer:    "task_passed",
		}

		samples = append(samples, InspectEvalSample{
			ID:          sampleID,
			Epoch:       1,
			Input:       "task " + task.ID + " " + task.Family,
			Target:      "pass",
			Messages:    []InspectChatMessage{assistant},
			Output:      output,
			Scores:      map[string]InspectScore{"task_passed": score},
			Events:      []any{modelEvent, scoreEvent},
			StartedAt:   inspectEvalLogTimestamp,
			CompletedAt: inspectEvalLogTimestamp,
		})
	}

	// Aggregate scores: pass rate and tool-call validity.
	passRate := float64(passed) / float64(len(r.Tasks))
	results := InspectEvalResults{
		TotalSamples:     len(r.Tasks),
		CompletedSamples: len(r.Tasks),
		Scores: []InspectEvalScore{
			{
				Name:          "pass_rate",
				Scorer:        "task_passed",
				ScoredSamples: len(r.Tasks),
				Metrics: map[string]InspectEvalMetric{
					"mean": {Name: "mean", Value: passRate},
				},
			},
		},
	}
	if toolCallMeasured > 0 {
		toolCallValidity := float64(toolCallValid) / float64(toolCallMeasured)
		results.Scores = append(results.Scores, InspectEvalScore{
			Name:          "tool_call_validity",
			Scorer:        "tool_call_validity",
			ScoredSamples: toolCallMeasured,
			Metrics: map[string]InspectEvalMetric{
				"mean": {Name: "mean", Value: toolCallValidity},
			},
		})
	}

	log := InspectEvalLog{
		Version: inspectEvalLogVersion,
		Status:  "success",
		Eval: InspectEvalSpec{
			Created: inspectEvalLogTimestamp,
			Task:    inspectTaskName,
			Dataset: InspectEvalDataset{Name: r.Suite, Samples: len(r.Tasks)},
			Model:   r.Model,
			Config:  InspectEvalConfig{LogSamples: true},
		},
		Plan: InspectEvalPlan{
			Name:  "plan",
			Steps: []InspectEvalPlanStep{{Solver: "agentbench"}},
		},
		Results: &results,
		Stats: InspectEvalStats{
			StartedAt:   inspectEvalLogTimestamp,
			CompletedAt: inspectEvalLogTimestamp,
			ModelUsage: map[string]InspectModelUsage{
				r.Model: {
					InputTokens:  totalInput,
					OutputTokens: totalOutput,
					TotalTokens:  totalInput + totalOutput,
				},
			},
		},
		Metadata: map[string]any{
			"projection": inspectEvalLogSchema,
			"arm":        r.Arm,
		},
		Samples: samples,
	}

	return log, nil
}

// InspectEvalLogFromComparison projects a comparison receipt's
// candidate arm into an Inspect AI eval log. It fail-closed refuses a
// receipt whose schema or status is unknown, or whose candidate arm
// carries no model or no task receipts.
func InspectEvalLogFromComparison(receipt compareDirectoryReceipt) (InspectEvalLog, error) {
	if receipt.Schema != "fak.agentbench.comparison.v1" {
		return InspectEvalLog{}, fmt.Errorf("%w: comparison receipt schema %q is unknown", errInspectEvalLogRefused, receipt.Schema)
	}
	if receipt.Status != "COMPARABLE" && receipt.Status != "NOISY_INCONCLUSIVE" {
		return InspectEvalLog{}, fmt.Errorf("%w: comparison status %q is not a comparable result", errInspectEvalLogRefused, receipt.Status)
	}
	candidate := receipt.Candidate
	if candidate.ModelID == "" {
		return InspectEvalLog{}, fmt.Errorf("%w: candidate arm carries no model identity", errInspectEvalLogRefused)
	}
	// The candidate arm's task receipts are not carried on the
	// comparison receipt directly; the projection uses the candidate
	// arm's model and the comparison's paired metric keys as the task
	// identity. A comparison with no matched metrics has no task shape
	// to project.
	metricKeys := make([]string, 0, len(receipt.Comparison.Metrics))
	for key := range receipt.Comparison.Metrics {
		metricKeys = append(metricKeys, key)
	}
	if len(metricKeys) == 0 {
		return InspectEvalLog{}, fmt.Errorf("%w: comparison carries no matched metrics to project", errInspectEvalLogRefused)
	}
	sort.Strings(metricKeys)
	tasks := make([]taskrunTaskReceipt, 0, len(metricKeys))
	for i, key := range metricKeys {
		metric := receipt.Comparison.Metrics[key]
		tasks = append(tasks, taskrunTaskReceipt{
			ID:       fmt.Sprintf("metric-%d-%s", i+1, key),
			Family:   "paired-metric",
			Passed:   metric.BootstrapHighMillis >= 0 && metric.BootstrapLowMillis <= 0,
			Accepted: true,
			Model: taskrunModelObservation{
				Requests:            metric.Pairs,
				UpstreamRequests:    metric.Pairs,
				SuccessfulRequests:  metric.Pairs,
				ModelIdentityStatus: "matched",
			},
		})
	}
	return ExportInspectEvalLog(RunResult{
		Arm:   candidate.ArmID,
		Model: candidate.ModelID,
		Suite: "comparison",
		Tasks: tasks,
	})
}

// WriteInspectEvalLog writes a projected Inspect eval log to path as
// pretty-printed JSON.
func WriteInspectEvalLog(path string, log InspectEvalLog) error {
	body, err := json.MarshalIndent(log, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	if err := os.WriteFile(path, body, 0600); err != nil {
		return err
	}
	return nil
}

// inspectEvalLogSchemaName returns the projection receipt schema.
func inspectEvalLogSchemaName() string { return inspectEvalLogSchema }

// inspectEvalLogTaskName returns the task name carried on projected
// logs.
func inspectEvalLogTaskName() string { return inspectTaskName }

// inspectEvalLogVersionNumber returns the pinned Inspect eval-log file
// format version.
func inspectEvalLogVersionNumber() int { return inspectEvalLogVersion }

// inspectEvalLogTimestampValue returns the deterministic UTC timestamp
// carried on projected logs.
func inspectEvalLogTimestampValue() string { return inspectEvalLogTimestamp }

// inspectEvalLogRefusedError returns the fail-closed refusal error for
// unknown result shapes.
func inspectEvalLogRefusedError() error { return errInspectEvalLogRefused }

// inspectEvalLogIsRefused reports whether err is the fail-closed
// refusal for an unknown result shape.
func inspectEvalLogIsRefused(err error) bool { return errors.Is(err, errInspectEvalLogRefused) }

// inspectEvalLogStringify renders a projected log's sample IDs in
// order, for operator-facing summaries.
func inspectEvalLogStringify(log InspectEvalLog) string {
	ids := make([]string, 0, len(log.Samples))
	for _, s := range log.Samples {
		ids = append(ids, s.ID)
	}
	return strings.Join(ids, ",")
}
