package tb4bench

import (
	"errors"
	"math"
)

// BenchmarkTierMetrics contains the official, authoritative solve and duration numbers.
// This struct is strictly separated from internal kernel telemetry.
type BenchmarkTierMetrics struct {
	TotalTasks              int             `json:"total_tasks"`
	SolvedTasks             int             `json:"solved_tasks"`
	SolveRate               float64         `json:"solve_rate"` // 0.0 to 1.0
	TaskPassMap             map[string]bool `json:"task_pass_map"`
	MeanTaskDurationSeconds float64         `json:"mean_task_duration_seconds"`
}

// TelemetryTierMetrics captures internal engine, harness, and system performance telemetry.
// These metrics are operational/informational and do not alter the official solve rate.
type TelemetryTierMetrics struct {
	TotalPromptTokens     int64   `json:"total_prompt_tokens"`
	TotalCompletionTokens int64   `json:"total_completion_tokens"`
	TokenEfficiency       float64 `json:"token_efficiency"` // tokens per solved task
	VDSOHits              int64   `json:"vdso_hits,omitempty"`
	CompactedTokens       int64   `json:"compacted_tokens,omitempty"`
	PolicyBlocks          int64   `json:"policy_blocks,omitempty"`
	PeakMemoryMB          float64 `json:"peak_memory_mb,omitempty"`
}

// ArmMetrics holds the two distinct tiers of metrics for a single benchmark arm.
type ArmMetrics struct {
	ArmID     string               `json:"arm_id"`
	Official  BenchmarkTierMetrics `json:"official"`
	Telemetry TelemetryTierMetrics `json:"telemetry"`
}

// ComputeArmMetrics aggregates task oracle results and telemetry records into ArmMetrics.
func ComputeArmMetrics(armID string, results []OracleResult, telemetry TelemetryTierMetrics) (ArmMetrics, error) {
	if len(results) == 0 {
		return ArmMetrics{}, errors.New("cannot compute metrics from empty results")
	}

	passMap := make(map[string]bool, len(results))
	solved := 0
	var totalDurationMs int64

	for _, res := range results {
		passMap[res.TaskID] = res.Passed
		if res.Passed {
			solved++
		}
		totalDurationMs += res.DurationMs
	}

	solveRate := float64(solved) / float64(len(results))
	meanDurationSec := (float64(totalDurationMs) / 1000.0) / float64(len(results))

	totalTokens := telemetry.TotalPromptTokens + telemetry.TotalCompletionTokens
	var tokenEfficiency float64
	if solved > 0 {
		tokenEfficiency = float64(totalTokens) / float64(solved)
	} else {
		tokenEfficiency = math.NaN()
	}
	telemetry.TokenEfficiency = tokenEfficiency

	return ArmMetrics{
		ArmID: armID,
		Official: BenchmarkTierMetrics{
			TotalTasks:              len(results),
			SolvedTasks:             solved,
			SolveRate:               solveRate,
			TaskPassMap:             passMap,
			MeanTaskDurationSeconds: meanDurationSec,
		},
		Telemetry: telemetry,
	}, nil
}
