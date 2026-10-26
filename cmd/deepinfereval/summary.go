package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
	"github.com/anthony-chaudhary/fak/internal/reasoneval"
)

const (
	summaryInputSchema = "fak.deepinfer-rollouts.v1"
	summarySchema      = "fak.deepinfer-summary.v1"
	maxSummaryBytes    = 16 << 20
)

var errSummaryInput = errors.New("invalid rollout manifest")

type summaryTask struct {
	ID     string `json:"task_id"`
	Answer string `json:"answer"`
}

type summaryRollout struct {
	TaskID       string `json:"task_id"`
	SampleIndex  *int   `json:"sample_idx"`
	RawText      string `json:"raw_text"`
	FinishReason string `json:"finish_reason"`
}

type summaryInput struct {
	Schema   string           `json:"schema"`
	NSample  int              `json:"n_sample"`
	Tasks    []summaryTask    `json:"tasks"`
	Rollouts []summaryRollout `json:"rollouts"`
}

type summaryReceipt struct {
	Schema              string             `json:"schema"`
	Benchmark           string             `json:"benchmark"`
	Judge               string             `json:"judge"`
	ReferenceComparable bool               `json:"reference_comparable"`
	NSample             int                `json:"n_sample"`
	NumTasks            int                `json:"num_tasks"`
	NumRollouts         int                `json:"num_rollouts"`
	ExpectedTasks       int                `json:"expected_tasks"`
	ExpectedRollouts    int                `json:"expected_rollouts"`
	CoverageComplete    bool               `json:"coverage_complete"`
	MissingTasks        int                `json:"missing_tasks"`
	RaggedSamples       bool               `json:"ragged_samples"`
	Score               float64            `json:"score"`
	TaskAccuracy        map[string]float64 `json:"task_accuracy"`
	PassAtK             map[string]float64 `json:"pass_at_k"`
	FinishReasons       map[string]int     `json:"finish_reasons"`
	ExtractSources      map[string]int     `json:"extract_sources"`
	ExtractFailed       float64            `json:"extract_failed"`
	TruncatedRollouts   int                `json:"truncated_rollouts"`
	InvalidRollouts     int                `json:"invalid_rollouts"`
}

func runSummary(benchmark, path string, stdout, stderr io.Writer) int {
	f, err := os.Open(path)
	if err != nil {
		return summaryRefusal(benchmark, err, stderr)
	}
	defer f.Close()
	input, err := readSummary(f)
	if err != nil {
		return summaryRefusal(benchmark, err, stderr)
	}
	receipt, err := summarizeRollouts(benchmark, input)
	if err != nil {
		return summaryRefusal(benchmark, err, stderr)
	}
	if err := json.NewEncoder(stdout).Encode(receipt); err != nil {
		return summaryRefusal(benchmark, err, stderr)
	}
	if !receipt.CoverageComplete {
		_ = json.NewEncoder(stderr).Encode(refusalReceipt{
			Schema: summarySchema, Benchmark: receipt.Benchmark,
			Code: "INCOMPLETE_COVERAGE", Detail: "summary is diagnostic, not a complete result",
		})
		return 3
	}
	return 0
}

func summaryRefusal(benchmark string, err error, stderr io.Writer) int {
	code, status := "SUMMARY_INVALID", 2
	if errors.Is(err, reasoneval.ErrJudgeRequired) {
		code, status = "JUDGE_REQUIRED", 3
	}
	_ = json.NewEncoder(stderr).Encode(refusalReceipt{
		Schema: summarySchema, Benchmark: benchmark, Code: code, Detail: err.Error(),
	})
	return status
}

func readSummary(r io.Reader) (summaryInput, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxSummaryBytes+1))
	if err != nil || len(data) > maxSummaryBytes {
		return summaryInput{}, fmt.Errorf("%w: unreadable or exceeds byte limit", errSummaryInput)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var input summaryInput
	if err := decoder.Decode(&input); err != nil {
		return summaryInput{}, fmt.Errorf("%w: %v", errSummaryInput, err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return summaryInput{}, fmt.Errorf("%w: expected one JSON object", errSummaryInput)
	}
	return input, nil
}

// Aggregation adapts DeepInfer's eval/common/passk.py and eval/score.py at
// 553741557a57df44a46fb3ab72b6aafeaf945337 (Apache-2.0). The existing Fak
// estimator is reused; repeated or out-of-range sample identities are refused
// rather than hidden by the upstream last-write-wins merge.
func summarizeRollouts(benchmark string, input summaryInput) (summaryReceipt, error) {
	benchmark = strings.ToLower(strings.TrimSpace(benchmark))
	if _, err := reasoneval.ScoreRule(benchmark, "", ""); err != nil {
		return summaryReceipt{}, err
	}
	if input.Schema != summaryInputSchema || input.NSample < 1 || len(input.Tasks) == 0 || len(input.Rollouts) == 0 || len(input.Tasks) > int(^uint(0)>>1)/input.NSample {
		return summaryReceipt{}, fmt.Errorf("%w: schema, tasks, samples, or rollout count", errSummaryInput)
	}
	answers := make(map[string]string, len(input.Tasks))
	for _, task := range input.Tasks {
		if strings.TrimSpace(task.ID) == "" || strings.TrimSpace(task.Answer) == "" {
			return summaryReceipt{}, fmt.Errorf("%w: task identity and answer required", errSummaryInput)
		}
		if _, exists := answers[task.ID]; exists {
			return summaryReceipt{}, fmt.Errorf("%w: duplicate task", errSummaryInput)
		}
		answers[task.ID] = task.Answer
	}
	result := summaryReceipt{
		Schema: summarySchema, Benchmark: benchmark, Judge: "rule", NSample: input.NSample,
		ExpectedTasks: len(input.Tasks), ExpectedRollouts: len(input.Tasks) * input.NSample,
		NumRollouts: len(input.Rollouts), TaskAccuracy: map[string]float64{},
		PassAtK: map[string]float64{}, FinishReasons: map[string]int{}, ExtractSources: map[string]int{},
	}
	tallies := make(map[string]livecodebench.SampleTally, len(input.Tasks))
	type sampleKey struct {
		task  string
		index int
	}
	seen := make(map[sampleKey]struct{}, len(input.Rollouts))
	correct, failed := 0, 0
	for _, row := range input.Rollouts {
		answer, known := answers[row.TaskID]
		if !known || row.SampleIndex == nil || *row.SampleIndex < 0 || *row.SampleIndex >= input.NSample || strings.TrimSpace(row.FinishReason) == "" {
			return summaryReceipt{}, fmt.Errorf("%w: unknown task, invalid sample, or missing finish reason", errSummaryInput)
		}
		key := sampleKey{row.TaskID, *row.SampleIndex}
		if _, exists := seen[key]; exists {
			return summaryReceipt{}, fmt.Errorf("%w: duplicate sample", errSummaryInput)
		}
		seen[key] = struct{}{}
		score, err := reasoneval.ScoreRule(benchmark, row.RawText, answer)
		if err != nil {
			return summaryReceipt{}, err
		}
		tally := tallies[row.TaskID]
		tally.Samples++
		if score.Correct {
			tally.Correct++
			correct++
		}
		tallies[row.TaskID] = tally
		if score.Source == "none" {
			failed++
		}
		result.ExtractSources[score.Source]++
		result.FinishReasons[row.FinishReason]++
		switch row.FinishReason {
		case "length", "prompt_length_exceeded":
			result.TruncatedRollouts++
		case "ignored", "aborted", "error", "preempted":
			result.InvalidRollouts++
		}
	}
	result.NumTasks = len(tallies)
	result.MissingTasks = len(input.Tasks) - len(tallies)
	result.CoverageComplete = result.NumRollouts == result.ExpectedRollouts
	result.Score = float64(correct) / float64(result.NumRollouts)
	result.ExtractFailed = float64(failed) / float64(result.NumRollouts)
	firstCount := 0
	for _, task := range input.Tasks {
		tally, present := tallies[task.ID]
		if !present {
			continue
		}
		if firstCount != 0 && tally.Samples != firstCount {
			result.RaggedSamples = true
		}
		firstCount = tally.Samples
		result.TaskAccuracy[task.ID] = float64(tally.Correct) / float64(tally.Samples)
		for _, k := range summaryKList(input.NSample) {
			value, err := livecodebench.PassAtK(tally.Samples, tally.Correct, min(k, tally.Samples))
			if err != nil {
				return summaryReceipt{}, err
			}
			result.PassAtK["Pass@"+strconv.Itoa(k)] += value / float64(result.NumTasks)
		}
	}
	return result, nil
}

func summaryKList(n int) []int {
	var values []int
	for k := 1; k < n; k *= 2 {
		values = append(values, k)
		if k > n/2 {
			break
		}
	}
	return append(values, n)
}
