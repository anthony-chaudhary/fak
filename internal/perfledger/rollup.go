// Adapted from ai-dynamo/aiperf
// src/aiperf/sweep_controller/main.py:119@ab610add4e8876fca223197596c728e17365f381
// (Apache-2.0, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES;
// modified from the original): four-way terminal-phase rollup
// (Succeeded/Failed/PartiallyFailed/Cancelled) honoring a maxFailures budget
// and abort policy while isolating cancelled children from genuine failures.
package perfledger

import (
	"fmt"
)

// TerminalPhase is the 4-way terminal classification for a sweep run.
type TerminalPhase string

const (
	PhaseSucceeded       TerminalPhase = "Succeeded"
	PhaseFailed          TerminalPhase = "Failed"
	PhasePartiallyFailed TerminalPhase = "PartiallyFailed"
	PhaseCancelled       TerminalPhase = "Cancelled"
)

// ChildOutcome is the terminal (or pending) status of one sweep child.
type ChildOutcome string

const (
	ChildSucceeded ChildOutcome = "Succeeded"
	ChildFailed    ChildOutcome = "Failed"
	ChildCancelled ChildOutcome = "Cancelled"
	ChildPending   ChildOutcome = "Pending"
)

// FailurePolicy controls how child failures exceeding MaxFailures are handled.
type FailurePolicy string

const (
	FailurePolicyContinue FailurePolicy = "continue"
	FailurePolicyAbort    FailurePolicy = "abort"
)

// RollupConfig configures terminal-phase resolution.
type RollupConfig struct {
	MaxFailures    int           `json:"max_failures"`
	Policy         FailurePolicy `json:"policy"`
	SweepCancelled bool          `json:"sweep_cancelled,omitempty"`
}

// RollupCounts summarizes child outcomes with Cancelled isolated from Failed.
type RollupCounts struct {
	Succeeded int `json:"succeeded"`
	Failed    int `json:"failed"`
	Cancelled int `json:"cancelled"`
	Pending   int `json:"pending"`
}

// RollupResult captures the resolved terminal phase and failure-budget metrics.
type RollupResult struct {
	Phase          TerminalPhase `json:"phase"`
	Succeeded      int           `json:"succeeded"`
	Failed         int           `json:"failed"`
	Cancelled      int           `json:"cancelled"`
	Pending        int           `json:"pending"`
	Total          int           `json:"total"`
	MaxFailures    int           `json:"max_failures"`
	BudgetExceeded bool          `json:"budget_exceeded"`
	Aborted        bool          `json:"aborted"`
	Reason         string        `json:"reason"`
}

// ResolveTerminalPhase resolves the 4-way terminal phase from explicit child
// counts. Cancelled children are tracked separately from Failed so out-of-band
// cancellations cannot consume the MaxFailures budget.
func ResolveTerminalPhase(cfg RollupConfig, counts RollupCounts) (RollupResult, error) {
	if cfg.MaxFailures < 0 {
		return RollupResult{}, fmt.Errorf("perfledger: RollupConfig.MaxFailures must be non-negative, got %d", cfg.MaxFailures)
	}
	if cfg.Policy == "" {
		cfg.Policy = FailurePolicyContinue
	}
	if cfg.Policy != FailurePolicyContinue && cfg.Policy != FailurePolicyAbort {
		return RollupResult{}, fmt.Errorf("perfledger: unknown FailurePolicy %q", cfg.Policy)
	}
	if counts.Succeeded < 0 || counts.Failed < 0 || counts.Cancelled < 0 || counts.Pending < 0 {
		return RollupResult{}, fmt.Errorf("perfledger: negative child counts in rollup: %+v", counts)
	}

	total := counts.Succeeded + counts.Failed + counts.Cancelled + counts.Pending
	budgetExceeded := counts.Failed > cfg.MaxFailures
	aborted := cfg.Policy == FailurePolicyAbort && budgetExceeded

	if counts.Pending > 0 && !aborted && !cfg.SweepCancelled {
		return RollupResult{}, fmt.Errorf("perfledger: cannot resolve terminal phase with %d pending children", counts.Pending)
	}

	res := RollupResult{
		Succeeded:      counts.Succeeded,
		Failed:         counts.Failed,
		Cancelled:      counts.Cancelled,
		Pending:        counts.Pending,
		Total:          total,
		MaxFailures:    cfg.MaxFailures,
		BudgetExceeded: budgetExceeded,
		Aborted:        aborted,
	}

	switch {
	case aborted:
		res.Phase = PhaseFailed
		res.Reason = "failure_budget_aborted"
	case cfg.SweepCancelled:
		res.Phase = PhaseCancelled
		res.Reason = "sweep_cancelled"
	case budgetExceeded:
		res.Phase = PhaseFailed
		res.Reason = "failure_budget_exceeded"
	case counts.Failed > 0 && counts.Succeeded == 0:
		res.Phase = PhaseFailed
		res.Reason = "all_attempted_failed"
	case counts.Failed > 0:
		res.Phase = PhasePartiallyFailed
		res.Reason = "partial_failures_within_budget"
	case counts.Cancelled > 0:
		res.Phase = PhaseCancelled
		res.Reason = "children_cancelled_without_failures"
	default:
		res.Phase = PhaseSucceeded
		res.Reason = "all_succeeded"
	}

	return res, nil
}

// ResolveTerminalPhaseFromOutcomes tallies child outcomes into isolated
// buckets and resolves the 4-way terminal phase.
func ResolveTerminalPhaseFromOutcomes(cfg RollupConfig, outcomes []ChildOutcome) (RollupResult, error) {
	var counts RollupCounts
	for i, o := range outcomes {
		switch o {
		case ChildSucceeded:
			counts.Succeeded++
		case ChildFailed:
			counts.Failed++
		case ChildCancelled:
			counts.Cancelled++
		case ChildPending:
			counts.Pending++
		default:
			return RollupResult{}, fmt.Errorf("perfledger: unknown ChildOutcome at index %d: %q", i, o)
		}
	}
	return ResolveTerminalPhase(cfg, counts)
}
