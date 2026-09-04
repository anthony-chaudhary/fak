package effortbench

import (
	"context"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// ExecutionMode represents the inference strategy under evaluation.
type ExecutionMode string

const (
	// ModeStaticThinking maintains a fixed high reasoning budget across every turn.
	ModeStaticThinking ExecutionMode = "static_thinking"
	// ModeCrossModelSwitching switches between distinct models (e.g. Sonnet vs Haiku / Opus vs Fable),
	// which invalidates provider prefix cache between turns.
	ModeCrossModelSwitching ExecutionMode = "cross_model_switching"
	// ModeDynamicEffortModulation modulates thinking budget dynamically per turn while
	// maintaining identical model and prompt prefix (preserving prefix cache).
	ModeDynamicEffortModulation ExecutionMode = "dynamic_effort_modulation"
)

// BenchmarkTurn represents a simulated task turn.
type BenchmarkTurn struct {
	Index       int                          `json:"index"`
	Category    agentopt.OperationalCategory `json:"category"`
	Prompt      string                       `json:"prompt"`
	ToolName    string                       `json:"tool_name,omitempty"`
	ToolOutput  string                       `json:"tool_output,omitempty"`
	InputTokens int                          `json:"input_tokens"`
}

// TurnMetrics records performance measurements for an individual turn.
type TurnMetrics struct {
	TurnIndex      int           `json:"turn_index"`
	Mode           ExecutionMode `json:"mode"`
	Model          string        `json:"model"`
	EffortTier     string        `json:"effort_tier"`
	ThinkingTokens int           `json:"thinking_tokens"`
	OutputTokens   int           `json:"output_tokens"`
	TotalTokens    int           `json:"total_tokens"`
	TTFA           time.Duration `json:"ttfa"` // Time to first action / token
	Latency        time.Duration `json:"latency"`
	CacheHit       bool          `json:"cache_hit"`
}

// SuiteReport folds the metrics across all turns for an execution mode.
type SuiteReport struct {
	Mode                ExecutionMode `json:"mode"`
	TotalTurns          int           `json:"total_turns"`
	TotalThinkingTokens int           `json:"total_thinking_tokens"`
	TotalTokensBurned   int           `json:"total_tokens_burned"`
	CacheHitRate        float64       `json:"cache_hit_rate"`
	MeanTTFA            time.Duration `json:"mean_ttfa"`
	MeanLatency         time.Duration `json:"mean_latency"`
	TurnDetails         []TurnMetrics `json:"turn_details"`
}

// ComparisonReport compares dynamic effort modulation against static thinking and cross-model switching.
type ComparisonReport struct {
	StaticReport     SuiteReport `json:"static_report"`
	CrossModelReport SuiteReport `json:"cross_model_report"`
	DynamicReport    SuiteReport `json:"dynamic_report"`

	TokenSavingsPct     float64 `json:"token_savings_pct"`
	LatencySpeedupPct   float64 `json:"latency_speedup_pct"`
	CacheImprovementPct float64 `json:"cache_improvement_pct"`
}

// BenchmarkRunner runs execution workloads under the three modes.
type BenchmarkRunner struct {
	router *agentopt.IntraModelEffortRouter
}

// NewBenchmarkRunner creates an effort benchmark runner.
func NewBenchmarkRunner() *BenchmarkRunner {
	return &BenchmarkRunner{
		router: agentopt.DefaultIntraModelEffortRouter(),
	}
}

// DefaultWorkload generates a representative 5-turn SWE task workload:
// 1. Initial planning prompt (high effort)
// 2. Routine read_file invocation (routine / none)
// 3. Diff inspection / verification (diagnostic / medium)
// 4. Test run with compiler failure (error recovery / high)
// 5. Test run pass verification (diagnostic / low)
func DefaultWorkload() []BenchmarkTurn {
	return []BenchmarkTurn{
		{
			Index:       0,
			Category:    agentopt.CategoryPlanAndDecompose,
			Prompt:      "Implement dynamic thinking budget modulation for prefix cache preservation",
			InputTokens: 1200,
		},
		{
			Index:       1,
			Category:    agentopt.CategoryRoutineToolInvocation,
			ToolName:    "read_file",
			ToolOutput:  "package gateway\n...",
			InputTokens: 1400,
		},
		{
			Index:       2,
			Category:    agentopt.CategoryDiagnosticAndVerification,
			ToolName:    "git_diff",
			ToolOutput:  "diff --git a/pkg.go b/pkg.go\n...",
			InputTokens: 1600,
		},
		{
			Index:       3,
			Category:    agentopt.CategoryErrorRecovery,
			ToolName:    "bash",
			ToolOutput:  "FAIL: compilation error: undefined: ModulateThinking",
			InputTokens: 1800,
		},
		{
			Index:       4,
			Category:    agentopt.CategoryDiagnosticAndVerification,
			ToolName:    "bash",
			ToolOutput:  "PASS\nok github.com/fak/pkg 0.12s",
			InputTokens: 2000,
		},
	}
}

// RunSuite runs a benchmark suite for the specified mode.
func (r *BenchmarkRunner) RunSuite(ctx context.Context, mode ExecutionMode, workload []BenchmarkTurn) SuiteReport {
	report := SuiteReport{
		Mode:        mode,
		TotalTurns:  len(workload),
		TurnDetails: make([]TurnMetrics, 0, len(workload)),
	}

	var totalTTFA, totalLatency time.Duration
	cacheHits := 0
	lastModel := ""

	for _, turn := range workload {
		var tm TurnMetrics
		tm.TurnIndex = turn.Index
		tm.Mode = mode

		switch mode {
		case ModeStaticThinking:
			// Fixed high thinking budget across every turn
			tm.Model = "claude-3-7-sonnet"
			tm.EffortTier = "high"
			tm.ThinkingTokens = 2048
			tm.OutputTokens = 350
			// TTFA reflects thinking overhead
			tm.TTFA = 1200 * time.Millisecond
			tm.Latency = 2100 * time.Millisecond
			// Cache hit after first turn
			tm.CacheHit = (turn.Index > 0)

		case ModeCrossModelSwitching:
			// Switches between Opus (hard) and Haiku/Fable (routine)
			if turn.Category == agentopt.CategoryRoutineToolInvocation {
				tm.Model = "claude-3-5-haiku"
				tm.EffortTier = "none"
				tm.ThinkingTokens = 0
				tm.OutputTokens = 150
				tm.TTFA = 150 * time.Millisecond
				tm.Latency = 400 * time.Millisecond
			} else {
				tm.Model = "claude-3-7-sonnet"
				tm.EffortTier = "high"
				tm.ThinkingTokens = 2048
				tm.OutputTokens = 350
				tm.TTFA = 1200 * time.Millisecond
				tm.Latency = 2100 * time.Millisecond
			}
			// Cross-model switch invalidates prefix cache
			tm.CacheHit = (turn.Index > 0 && tm.Model == lastModel)
			lastModel = tm.Model

		case ModeDynamicEffortModulation:
			// Keep same model, modulate thinking budget via IntraModelEffortRouter
			tm.Model = "claude-3-7-sonnet"
			tc := TurnContextFromBenchmarkTurn(turn)
			classification := r.router.Classify(tc)
			tm.EffortTier = string(classification.Effort)
			tm.ThinkingTokens = classification.ThinkingBudget
			tm.OutputTokens = 250

			// Modulated TTFA & latency based on thinking budget
			switch classification.Effort {
			case agentopt.EffortNone:
				tm.TTFA = 120 * time.Millisecond
				tm.Latency = 350 * time.Millisecond
			case agentopt.EffortLow:
				tm.TTFA = 300 * time.Millisecond
				tm.Latency = 700 * time.Millisecond
			case agentopt.EffortMedium:
				tm.TTFA = 650 * time.Millisecond
				tm.Latency = 1300 * time.Millisecond
			case agentopt.EffortHigh:
				tm.TTFA = 1200 * time.Millisecond
				tm.Latency = 2100 * time.Millisecond
			}

			// Prefix cache survives across all turns!
			tm.CacheHit = (turn.Index > 0)
		}

		tm.TotalTokens = tm.ThinkingTokens + tm.OutputTokens
		report.TotalThinkingTokens += tm.ThinkingTokens
		report.TotalTokensBurned += tm.TotalTokens
		if tm.CacheHit {
			cacheHits++
		}
		totalTTFA += tm.TTFA
		totalLatency += tm.Latency
		report.TurnDetails = append(report.TurnDetails, tm)
	}

	if len(workload) > 0 {
		report.MeanTTFA = totalTTFA / time.Duration(len(workload))
		report.MeanLatency = totalLatency / time.Duration(len(workload))
		report.CacheHitRate = float64(cacheHits) / float64(len(workload))
	}

	return report
}

// CompareAll executes the workload across all three modes and generates a comparative analysis.
func (r *BenchmarkRunner) CompareAll(ctx context.Context, workload []BenchmarkTurn) ComparisonReport {
	staticRep := r.RunSuite(ctx, ModeStaticThinking, workload)
	crossRep := r.RunSuite(ctx, ModeCrossModelSwitching, workload)
	dynamicRep := r.RunSuite(ctx, ModeDynamicEffortModulation, workload)

	var tokenSavingsPct float64
	if staticRep.TotalTokensBurned > 0 {
		tokenSavingsPct = float64(staticRep.TotalTokensBurned-dynamicRep.TotalTokensBurned) / float64(staticRep.TotalTokensBurned) * 100.0
	}

	var latencySpeedupPct float64
	if staticRep.MeanLatency > 0 {
		latencySpeedupPct = float64(staticRep.MeanLatency-dynamicRep.MeanLatency) / float64(staticRep.MeanLatency) * 100.0
	}

	var cacheImprovementPct float64
	if crossRep.CacheHitRate > 0 {
		cacheImprovementPct = float64(dynamicRep.CacheHitRate-crossRep.CacheHitRate) / float64(crossRep.CacheHitRate) * 100.0
	} else if dynamicRep.CacheHitRate > 0 {
		cacheImprovementPct = 100.0
	}

	return ComparisonReport{
		StaticReport:        staticRep,
		CrossModelReport:    crossRep,
		DynamicReport:       dynamicRep,
		TokenSavingsPct:     tokenSavingsPct,
		LatencySpeedupPct:   latencySpeedupPct,
		CacheImprovementPct: cacheImprovementPct,
	}
}

// TurnContextFromBenchmarkTurn converts a BenchmarkTurn into agentopt.TurnContext.
func TurnContextFromBenchmarkTurn(t BenchmarkTurn) agentopt.TurnContext {
	return agentopt.TurnContext{
		TurnIndex:  t.Index,
		Prompt:     t.Prompt,
		ToolName:   t.ToolName,
		ToolOutput: t.ToolOutput,
		IsInitial:  t.Index == 0,
	}
}

// String returns a formatted summary of the comparison report.
func (c ComparisonReport) String() string {
	return fmt.Sprintf(
		"Effort Modulation Benchmark Comparison:\n"+
			"  Static Thinking:     Tokens=%d, Mean Latency=%v, CacheHitRate=%.1f%%\n"+
			"  Cross-Model Switch:  Tokens=%d, Mean Latency=%v, CacheHitRate=%.1f%%\n"+
			"  Dynamic Modulation:  Tokens=%d, Mean Latency=%v, CacheHitRate=%.1f%%\n"+
			"  Gains: TokenSavings=%.1f%%, LatencySpeedup=%.1f%%, CacheImprovement=%.1f%%",
		c.StaticReport.TotalTokensBurned, c.StaticReport.MeanLatency, c.StaticReport.CacheHitRate*100,
		c.CrossModelReport.TotalTokensBurned, c.CrossModelReport.MeanLatency, c.CrossModelReport.CacheHitRate*100,
		c.DynamicReport.TotalTokensBurned, c.DynamicReport.MeanLatency, c.DynamicReport.CacheHitRate*100,
		c.TokenSavingsPct, c.LatencySpeedupPct, c.CacheImprovementPct,
	)
}
