// Adapted from confident-ai/deepeval (Apache-2.0) at commit
// deepeval/metrics/step_efficiency/step_efficiency.py and
// deepeval/metrics/task_completion/task_completion.py. This is a Go
// reimplementation of the deterministic parts only; LLM-judged scoring stays
// external.

package trajectory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
)

// DefaultEvalThreshold is the pass threshold used when a gate leaves Threshold unset.
const DefaultEvalThreshold = 0.5

// ErrEvalNeedsOrderedCalls refuses tool-correctness modes that need ordered,
// argument-bearing calls, which content-free audit rollups do not carry.
var ErrEvalNeedsOrderedCalls = errors.New("trajectory eval: audit rows carry unordered tool names only")

// EvalToolCall is one expected or observed tool invocation.
type EvalToolCall struct {
	Name            string         `json:"name"`
	Type            string         `json:"type,omitempty"`
	InputParameters map[string]any `json:"input_parameters,omitempty"`
	Output          any            `json:"output,omitempty"`
}

// EvalGate is the shared threshold and strict-mode pair; StrictMode raises the threshold to 1
// and a Threshold of 0 or less means DefaultEvalThreshold.
type EvalGate struct {
	Threshold  float64 `json:"threshold,omitempty"`
	StrictMode bool    `json:"strict_mode,omitempty"`
}

func (g EvalGate) threshold() float64 {
	if g.StrictMode {
		return 1
	}
	if g.Threshold <= 0 {
		return DefaultEvalThreshold
	}
	return g.Threshold
}

func (g EvalGate) result(metric string, score float64, reason string) EvalMetricResult {
	threshold := g.threshold()
	if g.StrictMode && score < threshold {
		score = 0
	}
	return EvalMetricResult{Metric: metric, Score: score, Threshold: threshold, Success: score >= threshold, Reason: reason}
}

// ToolCorrectnessOptions selects the matching mode; ExactMatch wins over ConsiderOrdering,
// and nil and empty input parameters compare equal.
type ToolCorrectnessOptions struct {
	EvalGate
	ExactMatch             bool `json:"exact_match,omitempty"`
	ConsiderOrdering       bool `json:"consider_ordering,omitempty"`
	CompareInputParameters bool `json:"compare_input_parameters,omitempty"`
	CompareOutput          bool `json:"compare_output,omitempty"`
}

// EvalMetricResult is one scored metric.
type EvalMetricResult struct {
	Metric    string  `json:"metric"`
	Score     float64 `json:"score"`
	Threshold float64 `json:"threshold"`
	Success   bool    `json:"success"`
	Reason    string  `json:"reason"`
}

// ScoreToolCorrectness compares observed calls with expected calls.
func ScoreToolCorrectness(expected, called []EvalToolCall, opts ToolCorrectnessOptions) EvalMetricResult {
	var score float64
	var mode string
	switch {
	case opts.ExactMatch:
		mode, score = "exact", exactToolMatchScore(expected, called, opts)
	case opts.ConsiderOrdering:
		mode = "ordered"
		switch {
		case len(expected) == 0 && len(called) == 0:
			score = 1
		case len(expected) == 0:
			score = 0
		default:
			score = weightedToolLCS(expected, called, opts) / float64(len(expected))
		}
	default:
		mode, score = "unordered", unorderedToolMatchScore(expected, called, opts)
	}
	reason := fmt.Sprintf("%s match: expected %v, called %v", mode, evalToolNames(expected), evalToolNames(called))
	return opts.EvalGate.result("tool_correctness", score, reason)
}

func exactToolMatchScore(expected, called []EvalToolCall, opts ToolCorrectnessOptions) float64 {
	if len(expected) != len(called) {
		return 0
	}
	for i := range expected {
		if !sameTool(expected[i], called[i]) {
			return 0
		}
		if opts.CompareInputParameters && !reflect.DeepEqual(nonNilParams(expected[i].InputParameters), nonNilParams(called[i].InputParameters)) {
			return 0
		}
		if opts.CompareOutput && !reflect.DeepEqual(expected[i].Output, called[i].Output) {
			return 0
		}
	}
	return 1
}

// Each observed call is consumed at most once, by occurrence rather than by value,
// so repeated identical calls can each satisfy a repeated expectation. Upstream
// de-duplicates called tools by value and scores such repeats lower; this differs on purpose.
func unorderedToolMatchScore(expected, called []EvalToolCall, opts ToolCorrectnessOptions) float64 {
	if len(expected) == 0 {
		if len(called) == 0 {
			return 1
		}
		return 0
	}
	used := make([]bool, len(called))
	total := 0.0
	for _, want := range expected {
		best, bestIndex := 0.0, -1
		for j, got := range called {
			if used[j] || !sameTool(want, got) {
				continue
			}
			if s := pairScore(want, got, opts); s > best {
				best, bestIndex = s, j
			}
		}
		if bestIndex >= 0 {
			total += best
			used[bestIndex] = true
		}
	}
	return total / float64(len(expected))
}

func weightedToolLCS(expected, called []EvalToolCall, opts ToolCorrectnessOptions) float64 {
	m, n := len(expected), len(called)
	dp := make([][]float64, m+1)
	for i := range dp {
		dp[i] = make([]float64, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			want, got := expected[i-1], called[j-1]
			best := max(dp[i-1][j], dp[i][j-1])
			if sameTool(want, got) {
				if s := pairScore(want, got, opts); s > 0 {
					best = max(best, dp[i-1][j-1]+s)
				}
			}
			dp[i][j] = best
		}
	}
	total := 0.0
	for i, j := m, n; i > 0 && j > 0; {
		switch {
		case dp[i][j] == dp[i-1][j]:
			i--
		case dp[i][j] == dp[i][j-1]:
			j--
		default:
			total += dp[i][j] - dp[i-1][j-1]
			i, j = i-1, j-1
		}
	}
	return total
}

func sameTool(a, b EvalToolCall) bool { return a.Name == b.Name && a.Type == b.Type }

func pairScore(want, got EvalToolCall, opts ToolCorrectnessOptions) float64 {
	score := 1.0
	if opts.CompareInputParameters {
		score *= paramOverlap(nonNilParams(want.InputParameters), nonNilParams(got.InputParameters))
	}
	if opts.CompareOutput && !reflect.DeepEqual(want.Output, got.Output) {
		score = 0
	}
	return score
}

// paramOverlap credits each shared key with an equal value 1/|union|, recursing
// into nested objects for partial credit.
func paramOverlap(a, b map[string]any) float64 {
	if reflect.DeepEqual(a, b) {
		return 1
	}
	union := len(a)
	for key := range b {
		if _, ok := a[key]; !ok {
			union++
		}
	}
	score := 0.0
	for key, av := range a {
		bv, ok := b[key]
		if !ok {
			continue
		}
		if reflect.DeepEqual(av, bv) {
			score += 1 / float64(union)
			continue
		}
		am, aIsMap := av.(map[string]any)
		bm, bIsMap := bv.(map[string]any)
		if aIsMap && bIsMap {
			score += paramOverlap(am, bm) / float64(union)
		}
	}
	return score
}

func nonNilParams(p map[string]any) map[string]any {
	if p == nil {
		return map[string]any{}
	}
	return p
}

func evalToolNames(calls []EvalToolCall) []string {
	names := make([]string, len(calls))
	for i, call := range calls {
		names[i] = call.Name
	}
	return names
}

// ScoreStepEfficiency is a deterministic stand-in for the upstream judged metric:
// min(1, referenceSteps/actualSteps), with zero actual steps scoring 1 only when the reference is also zero.
func ScoreStepEfficiency(referenceSteps, actualSteps int, gate EvalGate) (EvalMetricResult, error) {
	if referenceSteps < 0 || actualSteps < 0 {
		return EvalMetricResult{}, fmt.Errorf("trajectory eval: step counts must be non-negative (reference %d, actual %d)", referenceSteps, actualSteps)
	}
	var score float64
	switch {
	case actualSteps == 0 && referenceSteps == 0:
		score = 1
	case actualSteps == 0:
		score = 0
	default:
		score = min(1, float64(referenceSteps)/float64(actualSteps))
	}
	reason := fmt.Sprintf("reference %d steps, actual %d steps", referenceSteps, actualSteps)
	return gate.result("step_efficiency", score, reason), nil
}

// ScoreTaskCompletion is a deterministic stand-in for the upstream judged metric:
// it passes when no terminal failure is recorded; this is a gate on recorded outcome, not proof the task was achieved.
func ScoreTaskCompletion(row AuditTranscriptRow, gate EvalGate) EvalMetricResult {
	if row.TerminalFailures > 0 {
		return gate.result("task_completion", 0, fmt.Sprintf("%d terminal failures recorded %v", row.TerminalFailures, sortedFailureClasses(row.TerminalFailureClasses)))
	}
	return gate.result("task_completion", 1, "no terminal failure recorded")
}

func sortedFailureClasses(classes map[string]int) []string {
	names := make([]string, 0, len(classes))
	for name := range classes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// TrajectoryEvalSpec is the expected side of a trajectory evaluation.
type TrajectoryEvalSpec struct {
	SessionID       string                 `json:"session_id,omitempty"`
	ExpectedTools   []EvalToolCall         `json:"expected_tools"`
	ToolCorrectness ToolCorrectnessOptions `json:"tool_correctness"`
	ReferenceSteps  int                    `json:"reference_steps"`
	StepEfficiency  EvalGate               `json:"step_efficiency"`
	TaskCompletion  EvalGate               `json:"task_completion"`
}

// TrajectoryEvalReport holds the three deterministic metrics for one session row.
type TrajectoryEvalReport struct {
	Source          string           `json:"source"`
	SessionID       string           `json:"session_id"`
	ToolCorrectness EvalMetricResult `json:"tool_correctness"`
	StepEfficiency  EvalMetricResult `json:"step_efficiency"`
	TaskCompletion  EvalMetricResult `json:"task_completion"`
}

// EvaluateAuditTrajectory scores one session row of fak-trajectory-audit/1 JSONL
// against spec; an empty SessionID requires exactly one session row.
func EvaluateAuditTrajectory(r io.Reader, spec TrajectoryEvalSpec) (TrajectoryEvalReport, error) {
	opts := spec.ToolCorrectness
	if opts.ExactMatch || opts.ConsiderOrdering || opts.CompareInputParameters || opts.CompareOutput {
		return TrajectoryEvalReport{}, ErrEvalNeedsOrderedCalls
	}
	row, err := selectAuditSessionRow(r, spec.SessionID)
	if err != nil {
		return TrajectoryEvalReport{}, err
	}
	steps, err := ScoreStepEfficiency(spec.ReferenceSteps, row.ToolCalls, spec.StepEfficiency)
	if err != nil {
		return TrajectoryEvalReport{}, err
	}
	return TrajectoryEvalReport{
		Source:          row.Source,
		SessionID:       row.TranscriptID,
		ToolCorrectness: ScoreToolCorrectness(spec.ExpectedTools, auditRowToolCalls(row), opts),
		StepEfficiency:  steps,
		TaskCompletion:  ScoreTaskCompletion(row, spec.TaskCompletion),
	}, nil
}

func selectAuditSessionRow(r io.Reader, sessionID string) (AuditTranscriptRow, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 32*1024*1024)
	var matches []AuditTranscriptRow
	for line := 1; scanner.Scan(); line++ {
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 {
			continue
		}
		var envelope struct {
			Schema    string `json:"schema"`
			Kind      string `json:"kind"`
			SessionID string `json:"session_id"`
		}
		if err := json.Unmarshal(raw, &envelope); err != nil {
			return AuditTranscriptRow{}, fmt.Errorf("trajectory eval: line %d: decode envelope: %w", line, err)
		}
		if envelope.Schema != AuditSchema {
			return AuditTranscriptRow{}, fmt.Errorf("trajectory eval: line %d: schema %q is not %s", line, envelope.Schema, AuditSchema)
		}
		if envelope.Kind != "session" || (sessionID != "" && envelope.SessionID != sessionID) {
			continue
		}
		var row AuditTranscriptRow
		if err := json.Unmarshal(raw, &row); err != nil {
			return AuditTranscriptRow{}, fmt.Errorf("trajectory eval: line %d: decode session row: %w", line, err)
		}
		matches = append(matches, row)
	}
	if err := scanner.Err(); err != nil {
		return AuditTranscriptRow{}, fmt.Errorf("trajectory eval: scan input: %w", err)
	}
	if len(matches) != 1 {
		return AuditTranscriptRow{}, fmt.Errorf("trajectory eval: want exactly one session row (session_id %q), found %d", sessionID, len(matches))
	}
	return matches[0], nil
}

// auditRowToolCalls expands the per-tool call counts into name-only calls in name order.
func auditRowToolCalls(row AuditTranscriptRow) []EvalToolCall {
	dist := append([]AuditDistributionRow(nil), row.ToolDistribution...)
	sort.Slice(dist, func(i, j int) bool { return dist[i].Name < dist[j].Name })
	var calls []EvalToolCall
	for _, tool := range dist {
		for range tool.Calls {
			calls = append(calls, EvalToolCall{Name: tool.Name})
		}
	}
	return calls
}
