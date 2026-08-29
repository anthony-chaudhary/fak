package trajectory

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestInspectCodexExecUsageReadsTypedEvents(t *testing.T) {
	path := filepath.Join(t.TempDir(), "worker.jsonl")
	log := "{\"type\":\"thread.started\"}\n" +
		"{\"type\":\"turn.started\"}\n" +
		"guard diagnostic prose is ignored\n" +
		"{not-json\n" +
		"{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":12,\"cached_input_tokens\":7,\"output_tokens\":3}}\n"
	if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := InspectCodexExecUsage(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.TurnsStarted != 1 || got.TurnsCompleted != 1 || got.LastEvent != "turn.completed" ||
		got.InputTokens != 12 || got.CachedInputTokens != 7 || got.OutputTokens != 3 ||
		got.ProviderTokens != 15 || !got.UsageCovered || !got.LogReadable || got.ParseErrors != 1 {
		t.Fatalf("usage = %+v", got)
	}
}

func TestAssessQwenEmptyUsageWindowAndExclusions(t *testing.T) {
	started := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	base := QwenEmptyUsageInput{
		WorkloadKind: QwenWorkloadKindModelPerformance, TargetModelFamily: QwenTargetModelFamily,
		WorkerKind: QwenWorkerKindExecution, UsageExpectation: QwenUsageExpectationProvider,
		WorkerModel: "gpt-5.6-sol", LaunchStatus: "started", PID: 42,
		StartedAt: started, ObservedAt: started.Add(30 * time.Second),
		Window: time.Minute, ProcessAlive: true, Usage: CodexExecUsage{LogReadable: true},
	}
	tests := []struct {
		name   string
		mutate func(*QwenEmptyUsageInput)
		state  string
		reason string
	}{
		{
			name: "healthy usage",
			mutate: func(in *QwenEmptyUsageInput) {
				in.Usage = CodexExecUsage{LogReadable: true, TurnsStarted: 1, TurnsCompleted: 1, UsageCovered: true, ProviderTokens: 9}
			},
			state: QwenUsageStateHealthy, reason: QwenUsageReasonProviderUsageObserved,
		},
		{
			name: "window remains open",
			mutate: func(in *QwenEmptyUsageInput) {
				in.Usage = CodexExecUsage{LogReadable: true, TurnsStarted: 1}
			},
			state: QwenUsageStatePending, reason: QwenUsageReasonWindowOpen,
		},
		{
			name: "completed turn is empty",
			mutate: func(in *QwenEmptyUsageInput) {
				in.Usage = CodexExecUsage{LogReadable: true, TurnsStarted: 1, TurnsCompleted: 1}
			},
			state: QwenUsageStateEmpty, reason: QwenUsageReasonTurnCompletedWithoutUsage,
		},
		{
			name: "elapsed active launch is empty",
			mutate: func(in *QwenEmptyUsageInput) {
				in.ObservedAt = started.Add(time.Minute)
				in.Usage = CodexExecUsage{LogReadable: true, TurnsStarted: 1}
			},
			state: QwenUsageStateEmpty, reason: QwenUsageReasonWindowElapsedWithoutUsage,
		},
		{
			name: "explicit preflight is excluded",
			mutate: func(in *QwenEmptyUsageInput) {
				in.UsageExpectation = QwenUsageExpectationNone
			},
			state: QwenUsageStateExcluded, reason: QwenUsageReasonUsageNotExpected,
		},
		{
			name: "unmarked Qwen prose is not applicable",
			mutate: func(in *QwenEmptyUsageInput) {
				in.TargetModelFamily = ""
				in.WorkerModel = "Qwen appears only in prose-like metadata"
			},
			state: QwenUsageStateExcluded, reason: QwenUsageReasonNotApplicable,
		},
		{
			name: "malformed evidence is unobservable at deadline",
			mutate: func(in *QwenEmptyUsageInput) {
				in.ObservedAt = started.Add(time.Minute)
				in.Usage = CodexExecUsage{LogReadable: true, ParseErrors: 1}
			},
			state: QwenUsageStateUnobservable, reason: QwenUsageReasonEvidenceUnobservable,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			in := base
			test.mutate(&in)
			got := AssessQwenEmptyUsage(in)
			if got.Schema != QwenEmptyUsageAssessmentSchema || got.State != test.state || got.Reason != test.reason {
				t.Fatalf("assessment = %+v, want %s/%s", got, test.state, test.reason)
			}
			if test.name == "completed turn is empty" {
				if got.Usage != in.Usage {
					t.Fatalf("completed-turn usage changed: got %+v, want %+v", got.Usage, in.Usage)
				}
				if got.Usage.UsageCovered || got.Usage.InputTokens != 0 || got.Usage.CachedInputTokens != 0 || got.Usage.OutputTokens != 0 || got.Usage.ProviderTokens != 0 {
					t.Fatalf("completed empty turn invented provider usage: %+v", got.Usage)
				}
			}
			if got.WindowEndsAt != started.Add(time.Minute) &&
				got.Reason != QwenUsageReasonNotApplicable &&
				got.Reason != QwenUsageReasonUsageNotExpected {
				t.Fatalf("window end = %s", got.WindowEndsAt)
			}
		})
	}
}

func TestAssessQwenEmptyUsageDeadlineBoundary(t *testing.T) {
	started := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	input := QwenEmptyUsageInput{
		WorkloadKind: QwenWorkloadKindModelPerformance, TargetModelFamily: QwenTargetModelFamily,
		WorkerKind: QwenWorkerKindExecution, UsageExpectation: QwenUsageExpectationProvider,
		WorkerModel: "gpt-5.6-sol", LaunchStatus: "started", PID: 42,
		StartedAt: started, Window: time.Minute, ProcessAlive: true,
		Usage: CodexExecUsage{LogReadable: true, TurnsStarted: 1},
	}
	input.ObservedAt = started.Add(time.Minute - time.Nanosecond)
	if got := AssessQwenEmptyUsage(input); got.State != QwenUsageStatePending {
		t.Fatalf("deadline-1ns = %+v, want pending", got)
	}
	input.ObservedAt = started.Add(time.Minute)
	if got := AssessQwenEmptyUsage(input); got.State != QwenUsageStateEmpty || got.Reason != QwenUsageReasonWindowElapsedWithoutUsage {
		t.Fatalf("deadline = %+v, want empty", got)
	}
}
