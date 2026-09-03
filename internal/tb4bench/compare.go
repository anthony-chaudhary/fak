package tb4bench

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// TaskComparison compares outcomes and costs for a single task across both arms.
type TaskComparison struct {
	TaskID            string        `json:"task_id"`
	Category          string        `json:"category"`
	ArmAVerdict       string        `json:"arm_a_verdict"`
	ArmAFailureReason FailureReason `json:"arm_a_failure_reason,omitempty"`
	ArmADurationMs    int64         `json:"arm_a_duration_ms"`
	ArmATokens        int64         `json:"arm_a_tokens"`
	ArmBVerdict       string        `json:"arm_b_verdict"`
	ArmBFailureReason FailureReason `json:"arm_b_failure_reason,omitempty"`
	ArmBDurationMs    int64         `json:"arm_b_duration_ms"`
	ArmBTokens        int64         `json:"arm_b_tokens"`
	Winner            string        `json:"winner"` // ARM_A, ARM_B, TIE_SOLVED, TIE_FAILED
}

// CategoryComparison aggregates solve rates per task category.
type CategoryComparison struct {
	Category       string  `json:"category"`
	TotalTasks     int     `json:"total_tasks"`
	ArmASolved     int     `json:"arm_a_solved"`
	ArmBSolved     int     `json:"arm_b_solved"`
	ArmASolveRate  float64 `json:"arm_a_solve_rate"`
	ArmBSolveRate  float64 `json:"arm_b_solve_rate"`
	SolveRateDelta float64 `json:"solve_rate_delta"`
}

// WinTieLossMatrix counts win/tie/loss outcomes between Arm A and Arm B.
type WinTieLossMatrix struct {
	ArmAWins   int `json:"arm_a_wins"`
	BothSolved int `json:"both_solved"`
	BothFailed int `json:"both_failed"`
	ArmBWins   int `json:"arm_b_wins"`
}

// CompareReport is the authoritative comparative analysis between Arm A and Arm B.
type CompareReport struct {
	Benchmark            string               `json:"benchmark"`
	GeneratedAt          string               `json:"generated_at"`
	ContractDigest       string               `json:"contract_digest"`
	ModelCheckpoint      string               `json:"model_checkpoint"`
	ModelSha256          string               `json:"model_sha256"`
	ArmAMetrics          ArmMetrics           `json:"arm_a_metrics"`
	ArmBMetrics          ArmMetrics           `json:"arm_b_metrics"`
	SolveRateDelta       float64              `json:"solve_rate_delta"`
	TokenEfficiencyDelta float64              `json:"token_efficiency_delta"`
	WinTieLoss           WinTieLossMatrix     `json:"win_tie_loss"`
	CategoryBreakdown    []CategoryComparison `json:"category_breakdown"`
	Tasks                []TaskComparison     `json:"tasks"`
}

// BuildCompareReport constructs a CompareReport from contracts, receipts, and telemetry.
func BuildCompareReport(
	contract *OfficialRunContract,
	armAReceipts map[string]*GradingReceipt,
	armBReceipts map[string]*GradingReceipt,
	armATelemetry, armBTelemetry TelemetryTierMetrics,
	tasks []TaskManifest,
) (*CompareReport, error) {
	if contract == nil {
		return nil, errors.New("contract cannot be nil")
	}
	if len(tasks) == 0 {
		return nil, errors.New("tasks list cannot be empty")
	}

	contractDigest, _ := contract.Digest()

	var armAOracleResults []OracleResult
	var armBOracleResults []OracleResult
	var comparisons []TaskComparison

	catMap := make(map[string]*CategoryComparison)
	var matrix WinTieLossMatrix

	for _, task := range tasks {
		rA := armAReceipts[task.TaskID]
		rB := armBReceipts[task.TaskID]

		passedA := rA != nil && rA.Verdict == "SOLVED"
		passedB := rB != nil && rB.Verdict == "SOLVED"

		durA := int64(0)
		durB := int64(0)
		if rA != nil {
			durA = rA.DurationMs
			armAOracleResults = append(armAOracleResults, OracleResult{
				TaskID:        task.TaskID,
				ExitCode:      rA.ExitCode,
				DurationMs:    rA.DurationMs,
				Passed:        passedA,
				FailureReason: rA.FailureReason,
			})
		}
		if rB != nil {
			durB = rB.DurationMs
			armBOracleResults = append(armBOracleResults, OracleResult{
				TaskID:        task.TaskID,
				ExitCode:      rB.ExitCode,
				DurationMs:    rB.DurationMs,
				Passed:        passedB,
				FailureReason: rB.FailureReason,
			})
		}

		winner := "TIE_FAILED"
		if passedA && passedB {
			winner = "TIE_SOLVED"
			matrix.BothSolved++
		} else if passedA && !passedB {
			winner = "ARM_A"
			matrix.ArmAWins++
		} else if !passedA && passedB {
			winner = "ARM_B"
			matrix.ArmBWins++
		} else {
			matrix.BothFailed++
		}

		tc := TaskComparison{
			TaskID:         task.TaskID,
			Category:       task.Category,
			ArmAVerdict:    "FAILED",
			ArmADurationMs: durA,
			ArmBVerdict:    "FAILED",
			ArmBDurationMs: durB,
			Winner:         winner,
		}
		if rA != nil {
			tc.ArmAVerdict = rA.Verdict
			tc.ArmAFailureReason = rA.FailureReason
		}
		if rB != nil {
			tc.ArmBVerdict = rB.Verdict
			tc.ArmBFailureReason = rB.FailureReason
		}

		comparisons = append(comparisons, tc)

		// Category tally
		cat, exists := catMap[task.Category]
		if !exists {
			cat = &CategoryComparison{Category: task.Category}
			catMap[task.Category] = cat
		}
		cat.TotalTasks++
		if passedA {
			cat.ArmASolved++
		}
		if passedB {
			cat.ArmBSolved++
		}
	}

	// Compute category solve rates
	var categories []CategoryComparison
	for _, cat := range catMap {
		if cat.TotalTasks > 0 {
			cat.ArmASolveRate = float64(cat.ArmASolved) / float64(cat.TotalTasks)
			cat.ArmBSolveRate = float64(cat.ArmBSolved) / float64(cat.TotalTasks)
			cat.SolveRateDelta = cat.ArmASolveRate - cat.ArmBSolveRate
		}
		categories = append(categories, *cat)
	}
	sort.Slice(categories, func(i, j int) bool {
		return categories[i].Category < categories[j].Category
	})

	metricsA, err := ComputeArmMetrics("fak_inkernel", armAOracleResults, armATelemetry)
	if err != nil {
		return nil, fmt.Errorf("failed to compute arm A metrics: %w", err)
	}
	metricsB, err := ComputeArmMetrics("opencode_llamacpp", armBOracleResults, armBTelemetry)
	if err != nil {
		return nil, fmt.Errorf("failed to compute arm B metrics: %w", err)
	}

	solveRateDelta := metricsA.Official.SolveRate - metricsB.Official.SolveRate
	tokenDelta := metricsA.Telemetry.TokenEfficiency - metricsB.Telemetry.TokenEfficiency

	return &CompareReport{
		Benchmark:            BenchmarkName,
		GeneratedAt:          time.Now().UTC().Format(time.RFC3339),
		ContractDigest:       contractDigest,
		ModelCheckpoint:      contract.Model.Checkpoint,
		ModelSha256:          contract.Model.Sha256,
		ArmAMetrics:          metricsA,
		ArmBMetrics:          metricsB,
		SolveRateDelta:       solveRateDelta,
		TokenEfficiencyDelta: tokenDelta,
		WinTieLoss:           matrix,
		CategoryBreakdown:    categories,
		Tasks:                comparisons,
	}, nil
}

// GenerateMarkdown formats the comparative report into GitHub-flavored Markdown.
func (r *CompareReport) GenerateMarkdown() string {
	var sb strings.Builder

	sb.WriteString("# Terminal-Bench 4 Comparative Analysis\n\n")
	sb.WriteString(fmt.Sprintf("**Benchmark:** %s | **Generated:** %s\n", r.Benchmark, r.GeneratedAt))
	sb.WriteString(fmt.Sprintf("**Model:** `%s` (`%s`)\n", r.ModelCheckpoint, r.ModelSha256))
	sb.WriteString(fmt.Sprintf("**Contract Digest:** `%s`\n\n", r.ContractDigest))

	// Executive Summary Table
	sb.WriteString("## 1. Executive Summary & Authoritative Solve Rates\n\n")
	sb.WriteString("| Metric | Arm A: fak (In-Kernel) | Arm B: OpenCode (llama.cpp) | Delta (A - B) |\n")
	sb.WriteString("|---|---|---|---|\n")
	sb.WriteString(fmt.Sprintf("| **Solve Rate** | **%.1f%%** (%d/%d) | **%.1f%%** (%d/%d) | **%+.1f%%** |\n",
		r.ArmAMetrics.Official.SolveRate*100, r.ArmAMetrics.Official.SolvedTasks, r.ArmAMetrics.Official.TotalTasks,
		r.ArmBMetrics.Official.SolveRate*100, r.ArmBMetrics.Official.SolvedTasks, r.ArmBMetrics.Official.TotalTasks,
		r.SolveRateDelta*100))
	sb.WriteString(fmt.Sprintf("| Mean Duration | %.1fs | %.1fs | %+.1fs |\n",
		r.ArmAMetrics.Official.MeanTaskDurationSeconds, r.ArmBMetrics.Official.MeanTaskDurationSeconds,
		r.ArmAMetrics.Official.MeanTaskDurationSeconds-r.ArmBMetrics.Official.MeanTaskDurationSeconds))
	sb.WriteString(fmt.Sprintf("| Total Prompt Tokens | %d | %d | %+d |\n",
		r.ArmAMetrics.Telemetry.TotalPromptTokens, r.ArmBMetrics.Telemetry.TotalPromptTokens,
		r.ArmAMetrics.Telemetry.TotalPromptTokens-r.ArmBMetrics.Telemetry.TotalPromptTokens))
	sb.WriteString(fmt.Sprintf("| Total Completion Tokens | %d | %d | %+d |\n",
		r.ArmAMetrics.Telemetry.TotalCompletionTokens, r.ArmBMetrics.Telemetry.TotalCompletionTokens,
		r.ArmAMetrics.Telemetry.TotalCompletionTokens-r.ArmBMetrics.Telemetry.TotalCompletionTokens))
	sb.WriteString(fmt.Sprintf("| vDSO Hits (fak) | %d | N/A | +%d |\n",
		r.ArmAMetrics.Telemetry.VDSOHits, r.ArmAMetrics.Telemetry.VDSOHits))
	sb.WriteString(fmt.Sprintf("| Policy Blocks (fak) | %d | N/A | +%d |\n\n",
		r.ArmAMetrics.Telemetry.PolicyBlocks, r.ArmAMetrics.Telemetry.PolicyBlocks))

	// Win/Tie/Loss Matrix
	sb.WriteString("## 2. Head-to-Head Win/Tie/Loss Matrix\n\n")
	sb.WriteString(fmt.Sprintf("- **Arm A Wins (fak only):** %d\n", r.WinTieLoss.ArmAWins))
	sb.WriteString(fmt.Sprintf("- **Both Solved (Tie):** %d\n", r.WinTieLoss.BothSolved))
	sb.WriteString(fmt.Sprintf("- **Both Failed (Tie):** %d\n", r.WinTieLoss.BothFailed))
	sb.WriteString(fmt.Sprintf("- **Arm B Wins (OpenCode only):** %d\n\n", r.WinTieLoss.ArmBWins))

	// Category Breakdown Table
	sb.WriteString("## 3. Category Breakdown\n\n")
	sb.WriteString("| Category | Tasks | Arm A Solve Rate | Arm B Solve Rate | Delta |\n")
	sb.WriteString("|---|---|---|---|---|\n")
	for _, cat := range r.CategoryBreakdown {
		sb.WriteString(fmt.Sprintf("| `%s` | %d | %.1f%% (%d) | %.1f%% (%d) | %+.1f%% |\n",
			cat.Category, cat.TotalTasks,
			cat.ArmASolveRate*100, cat.ArmASolved,
			cat.ArmBSolveRate*100, cat.ArmBSolved,
			cat.SolveRateDelta*100))
	}
	sb.WriteString("\n")

	// Task-by-Task Details
	sb.WriteString("## 4. Task-by-Task Adjudication Roster\n\n")
	sb.WriteString("| Task ID | Category | Arm A Verdict | Arm B Verdict | Winner |\n")
	sb.WriteString("|---|---|---|---|---|\n")
	for _, t := range r.Tasks {
		sb.WriteString(fmt.Sprintf("| `%s` | `%s` | %s | %s | **%s** |\n",
			t.TaskID, t.Category, t.ArmAVerdict, t.ArmBVerdict, t.Winner))
	}

	return sb.String()
}

// Save outputs both machine-readable JSON and human-readable Markdown reports.
func (r *CompareReport) Save(jsonPath, mdPath string) error {
	if jsonPath != "" {
		data, err := json.MarshalIndent(r, "", "  ")
		if err != nil {
			return err
		}
		_ = os.MkdirAll(filepath.Dir(jsonPath), 0755)
		if err := os.WriteFile(jsonPath, data, 0644); err != nil {
			return err
		}
	}

	if mdPath != "" {
		mdContent := r.GenerateMarkdown()
		_ = os.MkdirAll(filepath.Dir(mdPath), 0755)
		if err := os.WriteFile(mdPath, []byte(mdContent), 0644); err != nil {
			return err
		}
	}

	return nil
}
