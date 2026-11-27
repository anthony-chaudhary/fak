package agentbench

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

const (
	GateSchema = "fak.agentbench.quality-gate.v1"

	GatePass         = "PASS"
	GateFail         = "FAIL"
	GateInconclusive = "INCONCLUSIVE"
)

// Closed reason tokens for multi-turn agentic quality gating.
const (
	ReasonRunError                 = "RUN_ERROR"
	ReasonInsufficientTrials       = "INSUFFICIENT_TRIALS"
	ReasonInsufficientReps         = "INSUFFICIENT_REPS"
	ReasonUnmeasuredToolCalls      = "UNMEASURED_TOOL_CALLS"
	ReasonSamplingUnpinned         = "SAMPLING_UNPINNED"
	ReasonSuiteMismatch            = "SUITE_MISMATCH"
	ReasonNoPairedTasks            = "NO_PAIRED_TASKS"
	ReasonAcceptRateBelowFloor     = "ACCEPT_RATE_BELOW_FLOOR"
	ReasonToolValidityBelowFloor   = "TOOL_VALIDITY_BELOW_FLOOR"
	ReasonLoopRateAboveCeiling     = "LOOP_RATE_ABOVE_CEILING"
	ReasonCyclicRepeatAboveCeiling = "CYCLIC_REPEAT_RATE_ABOVE_CEILING"
	ReasonAcceptRateRegression     = "ACCEPT_RATE_REGRESSION"
	ReasonToolValidityRegression   = "TOOL_VALIDITY_REGRESSION"
	ReasonLoopRateRegression       = "LOOP_RATE_REGRESSION"
	ReasonPairedTaskRegression     = "PAIRED_TASK_REGRESSION"
)

const gateUsage = "usage: fak bench agent gate --candidate PATH [--baseline PATH] [--min-accept-rate F] [--min-tool-validity F] [--max-loop-rate F] [--max-cyclic-repeat-rate F] [--min-trials N] [--min-reps N] [--max-accept-drop F] [--max-tool-validity-drop F] [--max-loop-increase F] [--require-pinned-sampling] [--json]"

// GatePolicy defines the thresholds for multi-turn agentic quality gating.
type GatePolicy struct {
	MinAcceptRate         float64 `json:"min_accept_rate"`
	MinToolValidity       float64 `json:"min_tool_validity"`
	MaxLoopRate           float64 `json:"max_loop_rate"`
	MaxCyclicRepeatRate   float64 `json:"max_cyclic_repeat_rate"`
	MinTrials             int     `json:"min_trials"`
	MinCompletedReps      int     `json:"min_completed_reps"`
	MaxAcceptDrop         float64 `json:"max_accept_drop"`
	MaxToolValidityDrop   float64 `json:"max_tool_validity_drop"`
	MaxLoopRateIncrease   float64 `json:"max_loop_rate_increase"`
	PairedSignAlpha       float64 `json:"paired_sign_alpha"`
	RequirePinnedSampling bool    `json:"require_pinned_sampling"`
}

// DefaultGatePolicy returns conservative default floors for inference-core gating.
func DefaultGatePolicy() GatePolicy {
	return GatePolicy{
		MinAcceptRate:       0.75,
		MinToolValidity:     0.98,
		MaxLoopRate:         0.20,
		MaxCyclicRepeatRate: 0.25,
		MinTrials:           8,
		MinCompletedReps:    1,
		MaxAcceptDrop:       0.05,
		MaxToolValidityDrop: 0.02,
		MaxLoopRateIncrease: 0.05,
		PairedSignAlpha:     0.05,
	}
}

// PairedTaskComparison summarizes task-level paired differences between candidate and baseline.
type PairedTaskComparison struct {
	MatchedTasks      int     `json:"matched_tasks"`
	CandidateBetter   int     `json:"candidate_better"`
	CandidateWorse    int     `json:"candidate_worse"`
	Tied              int     `json:"tied"`
	TwoSidedSignP     float64 `json:"two_sided_sign_p"`
	AcceptRateDelta   float64 `json:"accept_rate_delta"`
	ToolValidityDelta float64 `json:"tool_validity_delta"`
	LoopRateDelta     float64 `json:"loop_rate_delta"`
}

// GateDecision is the fail-closed output of EvaluateTasksGate.
type GateDecision struct {
	Schema         string                `json:"schema"`
	Verdict        string                `json:"verdict"`
	Reasons        []string              `json:"reasons"`
	Policy         GatePolicy            `json:"policy"`
	CandidateModel string                `json:"candidate_model,omitempty"`
	CandidateSuite string                `json:"candidate_suite,omitempty"`
	BaselineModel  string                `json:"baseline_model,omitempty"`
	BaselineSuite  string                `json:"baseline_suite,omitempty"`
	Candidate      taskrun.TrialSummary  `json:"candidate"`
	Baseline       *taskrun.TrialSummary `json:"baseline,omitempty"`
	Paired         *PairedTaskComparison `json:"paired,omitempty"`
}

// LoadTasksSummary loads a TasksSummary from a summary.json file, a directory
// containing summary.json, or a single taskrun.Receipt JSON file.
func LoadTasksSummary(path string) (TasksSummary, error) {
	if path == "" {
		return TasksSummary{}, errors.New("empty path")
	}
	info, err := os.Stat(path)
	if err != nil {
		return TasksSummary{}, err
	}
	if info.IsDir() {
		path = filepath.Join(path, "summary.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return TasksSummary{}, err
	}
	var summary TasksSummary
	if err := json.Unmarshal(raw, &summary); err != nil {
		return TasksSummary{}, fmt.Errorf("parse %s: %w", path, err)
	}
	if summary.Schema == "fak.agentbench.tasks-summary.v1" {
		return summary, nil
	}
	var receipt taskrun.Receipt
	if err := json.Unmarshal(raw, &receipt); err == nil && len(receipt.Tasks) > 0 {
		return wrapTaskrunReceipt(receipt, path), nil
	}
	return TasksSummary{}, fmt.Errorf("unsupported summary schema %q in %s", summary.Schema, path)
}

func wrapTaskrunReceipt(r taskrun.Receipt, path string) TasksSummary {
	rows := make([]TaskRepRow, 0, len(r.Tasks))
	for _, t := range r.Tasks {
		accCount := 0
		if t.Accepted {
			accCount = 1
		}
		var errs []string
		if t.Error != "" {
			errs = []string{t.Error}
		}
		rows = append(rows, TaskRepRow{
			ID:            t.ID,
			Family:        t.Family,
			Accepted:      []bool{t.Accepted},
			AcceptedCount: accCount,
			Loops:         []bool{t.ToolCalls.Loop},
			Errors:        errs,
		})
	}
	return TasksSummary{
		Schema:            "fak.agentbench.tasks-summary.v1",
		Endpoint:          r.Endpoint,
		Model:             r.Model,
		Suite:             r.Suite,
		IncludeHeldout:    r.IncludeHeldout,
		Reps:              1,
		CompletedReps:     1,
		Concurrency:       r.ConfiguredConcurrency,
		SamplingRequested: r.SamplingRequested,
		Receipts:          []string{path},
		Tasks:             rows,
		PerRep:            []taskrun.TrialSummary{r.Aggregate},
		Aggregate:         r.Aggregate,
		StartedAt:         r.StartedAt,
		Duration:          r.Duration,
	}
}

// EvaluateTasksGate evaluates a candidate TasksSummary (and optional baseline) against policy.
func EvaluateTasksGate(candidate TasksSummary, baseline *TasksSummary, policy GatePolicy) GateDecision {
	decision := GateDecision{
		Schema:         GateSchema,
		Policy:         policy,
		CandidateModel: candidate.Model,
		CandidateSuite: candidate.Suite,
		Candidate:      candidate.Aggregate,
	}
	var preconditionReasons []string
	var failureReasons []string

	if candidate.Error != "" {
		preconditionReasons = append(preconditionReasons, ReasonRunError)
	}
	if candidate.CompletedReps < policy.MinCompletedReps {
		preconditionReasons = append(preconditionReasons, ReasonInsufficientReps)
	}
	if candidate.Aggregate.Trials < policy.MinTrials || candidate.Aggregate.AcceptRate == nil {
		preconditionReasons = append(preconditionReasons, ReasonInsufficientTrials)
	}
	if candidate.Aggregate.MeasuredTrials == 0 || candidate.Aggregate.ToolCallValidity == nil {
		preconditionReasons = append(preconditionReasons, ReasonUnmeasuredToolCalls)
	}
	if policy.RequirePinnedSampling && (candidate.SamplingRequested == nil || candidate.SamplingRequested.Temperature == nil) {
		preconditionReasons = append(preconditionReasons, ReasonSamplingUnpinned)
	}

	if baseline != nil {
		decision.BaselineModel = baseline.Model
		decision.BaselineSuite = baseline.Suite
		bAgg := baseline.Aggregate
		decision.Baseline = &bAgg
		if baseline.Error != "" {
			preconditionReasons = append(preconditionReasons, ReasonRunError)
		}
		if baseline.Aggregate.Trials <= 0 || baseline.Aggregate.AcceptRate == nil {
			preconditionReasons = append(preconditionReasons, ReasonInsufficientTrials)
		}
		if baseline.Aggregate.MeasuredTrials == 0 || baseline.Aggregate.ToolCallValidity == nil {
			preconditionReasons = append(preconditionReasons, ReasonUnmeasuredToolCalls)
		}
		if candidate.Suite != baseline.Suite || candidate.IncludeHeldout != baseline.IncludeHeldout {
			preconditionReasons = append(preconditionReasons, ReasonSuiteMismatch)
		}
	}

	if len(preconditionReasons) > 0 {
		decision.Verdict = GateInconclusive
		decision.Reasons = preconditionReasons
		return decision
	}

	// Absolute quality floors.
	if *candidate.Aggregate.AcceptRate < policy.MinAcceptRate {
		failureReasons = append(failureReasons, ReasonAcceptRateBelowFloor)
	}
	if *candidate.Aggregate.ToolCallValidity < policy.MinToolValidity {
		failureReasons = append(failureReasons, ReasonToolValidityBelowFloor)
	}
	if candidate.Aggregate.StuckRate != nil && *candidate.Aggregate.StuckRate > policy.MaxLoopRate {
		failureReasons = append(failureReasons, ReasonLoopRateAboveCeiling)
	}
	if policy.MaxCyclicRepeatRate > 0 && candidate.Aggregate.CyclicRepeatRate != nil && *candidate.Aggregate.CyclicRepeatRate > policy.MaxCyclicRepeatRate {
		failureReasons = append(failureReasons, ReasonCyclicRepeatAboveCeiling)
	}

	// Optional paired baseline regression checks.
	if baseline != nil {
		paired := comparePairedTasks(candidate, *baseline)
		decision.Paired = &paired
		if paired.MatchedTasks == 0 {
			decision.Verdict = GateInconclusive
			decision.Reasons = []string{ReasonNoPairedTasks}
			return decision
		}
		if paired.AcceptRateDelta < -policy.MaxAcceptDrop {
			failureReasons = append(failureReasons, ReasonAcceptRateRegression)
		}
		if paired.ToolValidityDelta < -policy.MaxToolValidityDrop {
			failureReasons = append(failureReasons, ReasonToolValidityRegression)
		}
		if paired.LoopRateDelta > policy.MaxLoopRateIncrease {
			failureReasons = append(failureReasons, ReasonLoopRateRegression)
		}
		if paired.CandidateWorse > paired.CandidateBetter && paired.TwoSidedSignP <= policy.PairedSignAlpha {
			failureReasons = append(failureReasons, ReasonPairedTaskRegression)
		}
	}

	if len(failureReasons) > 0 {
		decision.Verdict = GateFail
		decision.Reasons = failureReasons
		return decision
	}
	decision.Verdict = GatePass
	decision.Reasons = []string{}
	return decision
}

func comparePairedTasks(candidate, baseline TasksSummary) PairedTaskComparison {
	baseTasks := make(map[string]TaskRepRow, len(baseline.Tasks))
	for _, t := range baseline.Tasks {
		baseTasks[t.ID] = t
	}
	comp := PairedTaskComparison{
		AcceptRateDelta:   *candidate.Aggregate.AcceptRate - *baseline.Aggregate.AcceptRate,
		ToolValidityDelta: *candidate.Aggregate.ToolCallValidity - *baseline.Aggregate.ToolCallValidity,
	}
	if candidate.Aggregate.StuckRate != nil && baseline.Aggregate.StuckRate != nil {
		comp.LoopRateDelta = *candidate.Aggregate.StuckRate - *baseline.Aggregate.StuckRate
	}
	for _, ct := range candidate.Tasks {
		bt, ok := baseTasks[ct.ID]
		if !ok || len(ct.Accepted) == 0 || len(bt.Accepted) == 0 {
			continue
		}
		comp.MatchedTasks++
		cRate := float64(ct.AcceptedCount) / float64(len(ct.Accepted))
		bRate := float64(bt.AcceptedCount) / float64(len(bt.Accepted))
		switch {
		case cRate > bRate+1e-9:
			comp.CandidateBetter++
		case bRate > cRate+1e-9:
			comp.CandidateWorse++
		default:
			comp.Tied++
		}
	}
	comp.TwoSidedSignP = exactTwoSidedSignTest(comp.CandidateBetter, comp.CandidateWorse)
	return comp
}

// exactTwoSidedSignTest computes the exact two-sided binomial p-value for discordant pairs.
func exactTwoSidedSignTest(wins, losses int) float64 {
	n := wins + losses
	if n == 0 {
		return 1.0
	}
	k := wins
	if losses < k {
		k = losses
	}
	// Sum binomial probabilities P(X <= k) for X ~ Binomial(n, 0.5).
	tail := 0.0
	choose := 1.0
	scale := math.Exp2(-float64(n))
	for i := 0; i <= k; i++ {
		tail += choose * scale
		choose *= float64(n-i) / float64(i+1)
	}
	return math.Min(1.0, 2.0*tail)
}

func runGateCLI(stdout, stderr io.Writer, args []string) int {
	policy := DefaultGatePolicy()
	var candidatePath, baselinePath string
	var asJSON bool

	flags := flag.NewFlagSet("bench agent gate", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&candidatePath, "candidate", "", "path to candidate summary.json, output directory, or taskrun receipt.json")
	flags.StringVar(&baselinePath, "baseline", "", "optional path to baseline summary.json, output directory, or taskrun receipt.json")
	flags.Float64Var(&policy.MinAcceptRate, "min-accept-rate", policy.MinAcceptRate, "minimum aggregate task acceptance rate [0..1]")
	flags.Float64Var(&policy.MinToolValidity, "min-tool-validity", policy.MinToolValidity, "minimum tool-call schema/JSON validity rate [0..1]")
	flags.Float64Var(&policy.MaxLoopRate, "max-loop-rate", policy.MaxLoopRate, "maximum consecutive-loop / turn-cap rate [0..1]")
	flags.Float64Var(&policy.MaxCyclicRepeatRate, "max-cyclic-repeat-rate", policy.MaxCyclicRepeatRate, "maximum non-consecutive cyclic repeat rate [0..1] (<=0 disables)")
	flags.IntVar(&policy.MinTrials, "min-trials", policy.MinTrials, "minimum number of evaluated trials")
	flags.IntVar(&policy.MinCompletedReps, "min-reps", policy.MinCompletedReps, "minimum number of completed reps")
	flags.Float64Var(&policy.MaxAcceptDrop, "max-accept-drop", policy.MaxAcceptDrop, "maximum allowed acceptance-rate drop vs baseline")
	flags.Float64Var(&policy.MaxToolValidityDrop, "max-tool-validity-drop", policy.MaxToolValidityDrop, "maximum allowed tool-validity drop vs baseline")
	flags.Float64Var(&policy.MaxLoopRateIncrease, "max-loop-increase", policy.MaxLoopRateIncrease, "maximum allowed loop-rate increase vs baseline")
	flags.BoolVar(&policy.RequirePinnedSampling, "require-pinned-sampling", policy.RequirePinnedSampling, "require explicit pinned sampling configuration")
	flags.BoolVar(&asJSON, "json", true, "emit machine-readable JSON decision")

	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if flags.NArg() != 0 || candidatePath == "" {
		fmt.Fprintln(stderr, gateUsage)
		return 2
	}

	cand, err := LoadTasksSummary(candidatePath)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: gate: candidate: %v\n", err)
		return 2
	}
	var basePtr *TasksSummary
	if baselinePath != "" {
		base, err := LoadTasksSummary(baselinePath)
		if err != nil {
			fmt.Fprintf(stderr, "agentbench: gate: baseline: %v\n", err)
			return 2
		}
		basePtr = &base
	}

	decision := EvaluateTasksGate(cand, basePtr, policy)
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(decision); err != nil {
			fmt.Fprintf(stderr, "agentbench: gate: encode: %v\n", err)
			return 2
		}
	} else {
		fmt.Fprintf(stdout, "%s reasons=%v\n", decision.Verdict, decision.Reasons)
	}

	switch decision.Verdict {
	case GatePass:
		return 0
	case GateFail:
		return 1
	default:
		return 3
	}
}
