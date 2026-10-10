package agentbench

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

// fak-test:runtime fast est=20ms lane=default
func TestRunTaskRepsFoldsPerTaskAcceptance(t *testing.T) {
	temp := 0.6
	opts := tasksOptions{endpoint: "http://h/v1", model: "m", out: t.TempDir(), suite: "extended", reps: 3, concurrency: 1, sampling: taskrun.Sampling{Temperature: &temp}}
	run := func(rep int, out string) (taskrun.Receipt, error) {
		if rep == 3 {
			return taskrun.Receipt{}, errors.New("sandbox unavailable")
		}
		tasks := []taskrun.TaskReceipt{
			{ID: "clamp-above-range", Family: "clamp", Accepted: rep == 1, ToolCalls: taskrun.ToolCallMetrics{Measured: true, Total: 4, Invalid: 1, Loop: rep == 2}, Model: taskrun.ModelObservation{SamplingSent: []taskrun.Sampling{{Temperature: &temp}}}},
			{ID: "wrap-is", Family: "error-wrap", Accepted: true, ToolCalls: taskrun.ToolCallMetrics{Measured: true, Total: 4}},
		}
		return taskrun.Receipt{Tasks: tasks, Aggregate: taskrun.SummarizeTrials(tasks)}, nil
	}
	s := runTaskReps(context.Background(), opts, run)
	if s.CompletedReps != 2 || len(s.Receipts) != 2 || len(s.PerRep) != 2 || s.Error == "" {
		t.Fatalf("reps completed=%d receipts=%d per_rep=%d error=%q", s.CompletedReps, len(s.Receipts), len(s.PerRep), s.Error)
	}
	if len(s.Tasks) != 2 || s.Tasks[0].ID != "clamp-above-range" || s.Tasks[0].AcceptedCount != 1 || len(s.Tasks[0].Accepted) != 2 || !s.Tasks[0].Accepted[0] || s.Tasks[0].Accepted[1] || !s.Tasks[0].Loops[1] {
		t.Fatalf("task rows = %+v", s.Tasks)
	}
	a := s.Aggregate
	if a.Trials != 4 || a.Accepted != 3 || a.ToolCallsTotal != 16 || a.ToolCallsInvalid != 2 || a.Loops != 1 || a.StuckRate == nil || *a.StuckRate != 0.25 {
		t.Fatalf("aggregate = %+v", a)
	}
	if s.SamplingRequested == nil || len(s.SamplingObserved) != 1 || *s.SamplingObserved[0].Temperature != temp {
		t.Fatalf("sampling requested=%v observed=%v", s.SamplingRequested, s.SamplingObserved)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestParseTasksOptionsRefusesIncompleteRuns(t *testing.T) {
	base := []string{"--endpoint", "http://h/v1", "--model", "m", "--out", t.TempDir()}
	for name, args := range map[string][]string{
		"missing endpoint": {"--model", "m", "--out", "o"},
		"unknown suite":    append(append([]string(nil), base...), "--suite", "nope"),
		"zero reps":        append(append([]string(nil), base...), "--reps", "0"),
		"bad top-p":        append(append([]string(nil), base...), "--top-p", "1.5"),
		"bad seed":         append(append([]string(nil), base...), "--seed", "x"),
	} {
		if _, err := parseTasksOptions(io.Discard, args); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	opts, err := parseTasksOptions(io.Discard, append(append([]string(nil), base...), "--suite", "extended", "--reps", "3", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--max-tokens", "2048", "--seed", "42"))
	if err != nil {
		t.Fatal(err)
	}
	s := opts.sampling
	if opts.suite != "extended" || opts.reps != 3 || *s.Temperature != 0.6 || *s.TopP != 0.95 || *s.TopK != 20 || *s.MaxTokens != 2048 || *s.Seed != 42 {
		t.Fatalf("parsed options = %+v", opts)
	}
}
