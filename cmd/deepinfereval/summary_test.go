package main

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestDeepInferSummaryCoverageAndRaggedPassAtK(t *testing.T) {
	t.Parallel()
	row := func(task string, sample int, text, finish string) summaryRollout {
		return summaryRollout{TaskID: task, SampleIndex: &sample, RawText: text, FinishReason: finish}
	}
	input := func() summaryInput {
		return summaryInput{
			Schema: summaryInputSchema, NSample: 2,
			Tasks: []summaryTask{{ID: "a", Answer: "A"}, {ID: "b", Answer: "C"}},
			Rollouts: []summaryRollout{
				row("a", 0, "ANSWER: A", "length"),
				row("a", 1, "no answer", "error"),
				row("b", 0, "ANSWER: B", "stop"),
			},
		}
	}
	runInput := func(t *testing.T, benchmark string, value summaryInput) (int, string, string) {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "rollouts.json")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return runCommand(t, "-benchmark", benchmark, "-summary", path, "-json")
	}
	decode := func(t *testing.T, output string) summaryReceipt {
		t.Helper()
		var result summaryReceipt
		if err := json.Unmarshal([]byte(output), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	assertRefusal := func(t *testing.T, output, code string) {
		t.Helper()
		var refusal refusalReceipt
		if err := json.Unmarshal([]byte(output), &refusal); err != nil {
			t.Fatal(err)
		}
		if refusal.Schema != summarySchema || refusal.Code != code {
			t.Fatalf("refusal schema=%q code=%q", refusal.Schema, refusal.Code)
		}
	}

	t.Run("ragged counts retain short tasks and fail complete-result status", func(t *testing.T) {
		status, stdout, stderr := runInput(t, "gpqa", input())
		if status != 3 {
			t.Fatalf("status=%d, want 3", status)
		}
		assertRefusal(t, stderr, "INCOMPLETE_COVERAGE")
		got := decode(t, stdout)
		if got.Schema != summarySchema || got.Benchmark != "gpqa" || got.Judge != "rule" || got.ReferenceComparable || got.CoverageComplete || !got.RaggedSamples {
			t.Fatal("summary mislabeled rule-only ragged coverage")
		}
		if got.NumTasks != 2 || got.NumRollouts != 3 || got.ExpectedTasks != 2 || got.ExpectedRollouts != 4 || got.MissingTasks != 0 {
			t.Fatal("summary lost expected or observed coverage")
		}
		if math.Abs(got.Score-1.0/3) > 1e-12 || got.PassAtK["Pass@1"] != .25 || got.PassAtK["Pass@2"] != .5 {
			t.Fatalf("score=%g pass@1=%g pass@2=%g", got.Score, got.PassAtK["Pass@1"], got.PassAtK["Pass@2"])
		}
		if got.TaskAccuracy["a"] != .5 || got.TaskAccuracy["b"] != 0 || got.TruncatedRollouts != 1 || got.InvalidRollouts != 1 || got.ExtractSources["none"] != 1 || math.Abs(got.ExtractFailed-1.0/3) > 1e-12 {
			t.Fatal("summary lost per-task or failure diagnostics")
		}
	})
	t.Run("complete coverage is not reference acceptance", func(t *testing.T) {
		value := input()
		value.Rollouts = append(value.Rollouts, row("b", 1, "ANSWER: C", "stop"))
		status, stdout, stderr := runInput(t, "gpqa", value)
		if status != 0 || stderr != "" {
			t.Fatalf("status=%d stderr=%q", status, stderr)
		}
		got := decode(t, stdout)
		if !got.CoverageComplete || got.RaggedSamples || got.ReferenceComparable || got.Score != .5 || got.PassAtK["Pass@2"] != 1 {
			t.Fatal("complete summary contract differs")
		}
	})
	t.Run("invalid finish remains diagnostic rather than rewriting correctness", func(t *testing.T) {
		value := input()
		value.Rollouts[1].RawText = "ANSWER: A"
		got, err := summarizeRollouts("gpqa", value)
		if err != nil {
			t.Fatal(err)
		}
		if got.InvalidRollouts != 1 || math.Abs(got.Score-2.0/3) > 1e-12 || got.TaskAccuracy["a"] != 1 {
			t.Fatal("invalid finish silently rewrote upstream rule correctness")
		}
	})
	t.Run("missing task remains explicit", func(t *testing.T) {
		value := input()
		value.Tasks = append(value.Tasks, summaryTask{ID: "c", Answer: "D"})
		status, stdout, stderr := runInput(t, "gpqa", value)
		if status != 3 {
			t.Fatalf("status=%d", status)
		}
		assertRefusal(t, stderr, "INCOMPLETE_COVERAGE")
		got := decode(t, stdout)
		if got.MissingTasks != 1 || got.ExpectedRollouts != 6 || got.CoverageComplete {
			t.Fatal("missing task concealed by observed-task denominator")
		}
	})
	for name, alter := range map[string]func(*summaryInput){
		"duplicate sample":     func(value *summaryInput) { value.Rollouts = append(value.Rollouts, value.Rollouts[0]) },
		"missing sample index": func(value *summaryInput) { value.Rollouts[0].SampleIndex = nil },
		"out of range sample":  func(value *summaryInput) { value.Rollouts[0] = row("a", 2, "A", "stop") },
		"unknown task":         func(value *summaryInput) { value.Rollouts[0].TaskID = "unknown" },
		"duplicate task":       func(value *summaryInput) { value.Tasks = append(value.Tasks, value.Tasks[0]) },
		"empty rollouts":       func(value *summaryInput) { value.Rollouts = nil },
	} {
		t.Run(name, func(t *testing.T) {
			value := input()
			alter(&value)
			status, stdout, stderr := runInput(t, "gpqa", value)
			if status != 2 || stdout != "" {
				t.Fatalf("status=%d stdout=%q", status, stdout)
			}
			assertRefusal(t, stderr, "SUMMARY_INVALID")
		})
	}
	t.Run("HLE still requires a judge", func(t *testing.T) {
		status, stdout, stderr := runInput(t, "hle", input())
		if status != 3 || stdout != "" {
			t.Fatalf("status=%d stdout=%q", status, stdout)
		}
		assertRefusal(t, stderr, "JUDGE_REQUIRED")
	})
	t.Run("default k ladder and strict manifest parser", func(t *testing.T) {
		for n, want := range map[int][]int{1: {1}, 5: {1, 2, 4, 5}, 32: {1, 2, 4, 8, 16, 32}} {
			if got := summaryKList(n); !reflect.DeepEqual(got, want) {
				t.Fatalf("k list(%d)=%v, want %v", n, got, want)
			}
		}
		for _, text := range []string{`{} {}`, `{"unexpected":true}`} {
			if _, err := readSummary(strings.NewReader(text)); !errors.Is(err, errSummaryInput) {
				t.Fatalf("error=%v, want invalid-manifest sentinel", err)
			}
		}
	})
}
