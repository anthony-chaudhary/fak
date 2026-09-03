package gateway

// batchbudget.go — composable admission budgets for native batch construction.
//
// The shape is adapted from Modular's TokenBudgetCollection
// (max/python/max/serve/scheduler/batch_constructor/token_budget.py
// @1c9fd2e03331f77d3a1034127cb3700b7fa43c02), but the fold deliberately
// evaluates every independent budget and makes exhaustion dominate. That keeps
// admission fail-closed and independent of the order in which budgets are
// installed: a "reached" result from one axis can never hide an "exhausted"
// result from another.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/session"
)

type batchBudgetStatus uint8

const (
	// batchBudgetAvailable means the request fits and this budget retains room.
	batchBudgetAvailable batchBudgetStatus = iota
	// batchBudgetReached means the request fits, but no later request should be
	// added to this batch under this budget.
	batchBudgetReached
	// batchBudgetExhausted means the request does not fit and must not be admitted.
	batchBudgetExhausted
)

func (s batchBudgetStatus) valid() bool {
	return s >= batchBudgetAvailable && s <= batchBudgetExhausted
}

// batchBudgetSnapshot is the immutable admission state presented to every
// independent budget in one fold. Budget checks must not mutate controller state.
type batchBudgetSnapshot struct {
	running int
	tokens  int
}

// batchBudgetCheck is one budget's typed result. budget is a stable identity used
// to make equal-status refusals deterministic; reason is surfaced for impossible
// requests at the live admission boundary.
type batchBudgetCheck struct {
	status batchBudgetStatus
	budget string
	reason string
}

// batchBudget evaluates one independent capacity axis against the same immutable
// snapshot and request. State is charged only after the composed fold admits.
type batchBudget func(batchBudgetSnapshot, SeqRequest) batchBudgetCheck

// foldBatchBudgets composes all independent budgets into one status. Exhausted
// dominates reached, which dominates available. Equal non-available statuses use
// the lexicographically smallest (budget, reason) pair, so both the admission
// decision and its typed refusal are stable across installation order.
func foldBatchBudgets(state batchBudgetSnapshot, req SeqRequest, budgets []batchBudget) batchBudgetCheck {
	folded := batchBudgetCheck{status: batchBudgetAvailable}
	for _, budget := range budgets {
		check := evaluateBatchBudget(budget, state, req)
		if strongerBatchBudgetCheck(check, folded) {
			folded = check
		}
	}
	return folded
}

func evaluateBatchBudget(budget batchBudget, state batchBudgetSnapshot, req SeqRequest) batchBudgetCheck {
	if budget == nil {
		return batchBudgetCheck{
			status: batchBudgetExhausted,
			budget: "invalid_batch_budget",
			reason: "nil batch budget",
		}
	}
	check := budget(state, req)
	if check.budget == "" {
		return batchBudgetCheck{
			status: batchBudgetExhausted,
			budget: "invalid_batch_budget",
			reason: "batch budget returned an empty identity",
		}
	}
	if !check.status.valid() {
		return batchBudgetCheck{
			status: batchBudgetExhausted,
			budget: check.budget,
			reason: fmt.Sprintf("batch budget %q returned invalid status %d", check.budget, check.status),
		}
	}
	if check.status == batchBudgetExhausted && check.reason == "" {
		check.reason = fmt.Sprintf("batch budget %q exhausted", check.budget)
	}
	return check
}

func strongerBatchBudgetCheck(candidate, current batchBudgetCheck) bool {
	if candidate.status != current.status {
		return candidate.status > current.status
	}
	if candidate.status == batchBudgetAvailable {
		return false
	}
	if candidate.budget != current.budget {
		return candidate.budget < current.budget
	}
	return candidate.reason < current.reason
}

func admissionBatchBudgets(policy AdmissionPolicy) []batchBudget {
	budgets := make([]batchBudget, 0, 2)
	if policy.MaxNumSeqs > 0 {
		budgets = append(budgets, maxNumSeqsBatchBudget(policy.MaxNumSeqs))
	}
	if policy.TokenBudget > 0 {
		budgets = append(budgets, tokenBatchBudget(policy.TokenBudget))
	}
	return budgets
}

func maxNumSeqsBatchBudget(capacity int) batchBudget {
	return func(state batchBudgetSnapshot, _ SeqRequest) batchBudgetCheck {
		next := state.running + 1
		switch {
		case next > capacity:
			return batchBudgetCheck{
				status: batchBudgetExhausted,
				budget: "max_num_seqs",
				reason: fmt.Sprintf("scheduler max-num-seqs budget exhausted (%d running, capacity %d)", state.running, capacity),
			}
		case next == capacity:
			return batchBudgetCheck{status: batchBudgetReached, budget: "max_num_seqs"}
		default:
			return batchBudgetCheck{status: batchBudgetAvailable, budget: "max_num_seqs"}
		}
	}
}

func tokenBatchBudget(capacity int) batchBudget {
	return func(state batchBudgetSnapshot, req SeqRequest) batchBudgetCheck {
		next := state.tokens + req.Tokens
		switch {
		case next > capacity:
			reason := fmt.Sprintf("scheduler token budget exhausted (%d in use + %d requested > %d)", state.tokens, req.Tokens, capacity)
			if state.tokens == 0 && req.Tokens > capacity {
				// Preserve the established live-boundary refusal text for an
				// envelope that can never fit, even on an idle controller.
				reason = fmt.Sprintf("request tokens %d exceed scheduler token budget %d", req.Tokens, capacity)
			}
			return batchBudgetCheck{
				status: batchBudgetExhausted,
				budget: "tokens",
				reason: reason,
			}
		case next == capacity:
			return batchBudgetCheck{status: batchBudgetReached, budget: "tokens"}
		default:
			return batchBudgetCheck{status: batchBudgetAvailable, budget: "tokens"}
		}
	}
}

func fleetBatchBudget(pool *session.Pool) batchBudget {
	return func(_ batchBudgetSnapshot, req SeqRequest) batchBudgetCheck {
		if pool == nil {
			return batchBudgetCheck{status: batchBudgetAvailable, budget: "fleet_tokens"}
		}
		rem := pool.Remaining()
		if rem < 0 {
			return batchBudgetCheck{status: batchBudgetAvailable, budget: "fleet_tokens"}
		}
		switch {
		case rem < req.Tokens:
			return batchBudgetCheck{
				status: batchBudgetExhausted,
				budget: "fleet_tokens",
				reason: fmt.Sprintf("session pool budget exhausted (%d remaining < %d requested)", rem, req.Tokens),
			}
		case rem == req.Tokens:
			return batchBudgetCheck{status: batchBudgetReached, budget: "fleet_tokens"}
		default:
			return batchBudgetCheck{status: batchBudgetAvailable, budget: "fleet_tokens"}
		}
	}
}
