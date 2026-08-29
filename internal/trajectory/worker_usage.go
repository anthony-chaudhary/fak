package trajectory

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

const QwenEmptyUsageAssessmentSchema = "fak-qwen-empty-usage-assessment/1"

const (
	QwenUsageStateExcluded     = "excluded"
	QwenUsageStatePending      = "pending"
	QwenUsageStateHealthy      = "healthy"
	QwenUsageStateEmpty        = "empty"
	QwenUsageStateUnobservable = "unobservable"

	QwenUsageReasonNotApplicable             = "not_applicable"
	QwenUsageReasonUsageNotExpected          = "usage_not_expected"
	QwenUsageReasonLaunchNotStarted          = "launch_not_started"
	QwenUsageReasonWindowOpen                = "observation_window_open"
	QwenUsageReasonEvidenceUnobservable      = "evidence_unobservable"
	QwenUsageReasonProviderUsageObserved     = "provider_usage_observed"
	QwenUsageReasonTurnCompletedWithoutUsage = "turn_completed_without_usage"
	QwenUsageReasonProcessExitedWithoutUsage = "process_exited_without_usage"
	QwenUsageReasonWindowElapsedWithoutUsage = "window_elapsed_without_usage"

	QwenWorkloadKindModelPerformance = "model_performance"
	QwenTargetModelFamily            = "qwen"
	QwenWorkerKindExecution          = "execution"
	QwenUsageExpectationProvider     = "provider"
	QwenUsageExpectationNone         = "none"
)

// CodexExecUsage is the content-free usage projection of one `codex exec --json`
// stream. Only typed event names and provider usage counters are inspected.
type CodexExecUsage struct {
	LogReadable       bool   `json:"log_readable"`
	ParseErrors       int    `json:"parse_errors"`
	TurnsStarted      int    `json:"turns_started"`
	TurnsCompleted    int    `json:"turns_completed"`
	LastEvent         string `json:"last_event,omitempty"`
	InputTokens       int64  `json:"input_tokens"`
	CachedInputTokens int64  `json:"cached_input_tokens"`
	OutputTokens      int64  `json:"output_tokens"`
	ProviderTokens    int64  `json:"provider_tokens"`
	UsageCovered      bool   `json:"usage_covered"`
}

// InspectCodexExecUsage reads the structural Codex event stream used by
// orchestration workers. Missing files and incomplete trailing JSON are normal
// while a worker is starting, so both yield the observation accumulated so far.
func InspectCodexExecUsage(path string) (CodexExecUsage, error) {
	var out CodexExecUsage
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return out, fmt.Errorf("inspect Codex worker usage: %w", err)
	}
	defer file.Close()
	out.LogReadable = true

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "{") {
			continue
		}
		var event struct {
			Type  string `json:"type"`
			Usage *struct {
				InputTokens       int64 `json:"input_tokens"`
				CachedInputTokens int64 `json:"cached_input_tokens"`
				OutputTokens      int64 `json:"output_tokens"`
			} `json:"usage,omitempty"`
		}
		if json.Unmarshal([]byte(line), &event) != nil {
			out.ParseErrors++
			continue
		}
		if event.Type == "" {
			continue
		}
		out.LastEvent = event.Type
		switch event.Type {
		case "turn.started":
			out.TurnsStarted++
		case "turn.completed":
			out.TurnsCompleted++
			if event.Usage != nil && (event.Usage.InputTokens != 0 || event.Usage.CachedInputTokens != 0 || event.Usage.OutputTokens != 0) {
				out.InputTokens += event.Usage.InputTokens
				out.CachedInputTokens += event.Usage.CachedInputTokens
				out.OutputTokens += event.Usage.OutputTokens
				out.ProviderTokens += event.Usage.InputTokens + event.Usage.OutputTokens
				out.UsageCovered = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return out, fmt.Errorf("inspect Codex worker usage: %w", err)
	}
	return out, nil
}

type QwenEmptyUsageInput struct {
	WorkloadKind      string
	TargetModelFamily string
	WorkerKind        string
	UsageExpectation  string
	WorkerModel       string
	LaunchStatus      string
	PID               int
	StartedAt         time.Time
	ObservedAt        time.Time
	Window            time.Duration
	ProcessAlive      bool
	Usage             CodexExecUsage
}

// QwenEmptyUsageAssessment is the typed classification persisted by the
// orchestration launch receipt and rendered by orchestration status.
type QwenEmptyUsageAssessment struct {
	Schema            string         `json:"schema"`
	State             string         `json:"state"`
	Reason            string         `json:"reason"`
	WorkloadKind      string         `json:"workload_kind"`
	TargetModelFamily string         `json:"target_model_family"`
	WorkerKind        string         `json:"worker_kind"`
	UsageExpectation  string         `json:"usage_expectation"`
	WorkerModel       string         `json:"worker_model"`
	StartedAt         time.Time      `json:"started_at,omitempty"`
	ObservedAt        time.Time      `json:"observed_at"`
	WindowEndsAt      time.Time      `json:"window_ends_at,omitempty"`
	ProcessAlive      bool           `json:"process_alive"`
	Usage             CodexExecUsage `json:"usage"`
}

// AssessQwenEmptyUsage defines the exact launch window as the half-open
// interval [StartedAt, StartedAt+Window). Provider usage makes the launch
// healthy immediately. A structurally completed turn is also terminal: when
// its readable, parseable event stream contains no provider counters it settles
// as explicitly empty without manufacturing usage. Other zero-usage observations
// remain pending until the configured end because process state may still be
// followed by buffered usage evidence. At the end, an exited process or still-live
// launch without usage is empty. Structurally non-applicable launches,
// explicitly usage-not-expected workers (such as preflight-only workers), and
// launches that never started are the closed valid exclusions. Missing or
// malformed evidence is unobservable, never silently reclassified as empty.
func AssessQwenEmptyUsage(in QwenEmptyUsageInput) QwenEmptyUsageAssessment {
	observedAt := in.ObservedAt.UTC()
	out := QwenEmptyUsageAssessment{
		Schema:       QwenEmptyUsageAssessmentSchema,
		WorkloadKind: in.WorkloadKind, TargetModelFamily: in.TargetModelFamily,
		WorkerKind: in.WorkerKind, UsageExpectation: in.UsageExpectation, WorkerModel: in.WorkerModel,
		StartedAt: in.StartedAt.UTC(), ObservedAt: observedAt,
		ProcessAlive: in.ProcessAlive, Usage: in.Usage,
	}
	if in.UsageExpectation == QwenUsageExpectationNone {
		out.State = QwenUsageStateExcluded
		out.Reason = QwenUsageReasonUsageNotExpected
		return out
	}
	if in.WorkloadKind != QwenWorkloadKindModelPerformance ||
		in.TargetModelFamily != QwenTargetModelFamily ||
		in.WorkerKind != QwenWorkerKindExecution ||
		in.UsageExpectation != QwenUsageExpectationProvider {
		out.State = QwenUsageStateExcluded
		out.Reason = QwenUsageReasonNotApplicable
		return out
	}
	if in.PID <= 0 || in.StartedAt.IsZero() {
		out.State = QwenUsageStateExcluded
		out.Reason = QwenUsageReasonLaunchNotStarted
		return out
	}
	out.WindowEndsAt = in.StartedAt.UTC().Add(in.Window)
	windowOpen := in.Window > 0 && observedAt.Before(out.WindowEndsAt)
	if in.Usage.UsageCovered {
		out.State = QwenUsageStateHealthy
		out.Reason = QwenUsageReasonProviderUsageObserved
		return out
	}
	if !in.Usage.LogReadable || in.Usage.ParseErrors > 0 {
		if windowOpen {
			out.State = QwenUsageStatePending
			out.Reason = QwenUsageReasonWindowOpen
			return out
		}
		out.State = QwenUsageStateUnobservable
		out.Reason = QwenUsageReasonEvidenceUnobservable
		return out
	}
	if in.Usage.TurnsStarted > 0 && in.Usage.TurnsCompleted >= in.Usage.TurnsStarted {
		out.State = QwenUsageStateEmpty
		out.Reason = QwenUsageReasonTurnCompletedWithoutUsage
		return out
	}
	if windowOpen {
		out.State = QwenUsageStatePending
		out.Reason = QwenUsageReasonWindowOpen
		return out
	}
	if !in.ProcessAlive {
		out.State = QwenUsageStateEmpty
		out.Reason = QwenUsageReasonProcessExitedWithoutUsage
		return out
	}
	out.State = QwenUsageStateEmpty
	out.Reason = QwenUsageReasonWindowElapsedWithoutUsage
	return out
}
