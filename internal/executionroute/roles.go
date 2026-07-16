package executionroute

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ---------------------------------------------------------------------------
// SUB-MODEL DELEGATION ROLES — who does what within one execution.
// ---------------------------------------------------------------------------

// Role is the delegation duty a sub-model plays within a single execution. Roles
// are a CLOSED, additive set: each names a DISTINCT job (probe / do / adjudicate
// / drive) so a plan can bind a role to its own model constraints and budget
// WITHOUT overloading one model id to mean several things. Adding a role is a new
// constant plus a knownRole arm, never a manifest free-text field.
type Role string

const (
	// RoleScout is the cheap classify-first probe: a small/fast model labels the
	// subject before the expensive roles commit (the scout-then-route pattern).
	RoleScout Role = "scout"
	// RoleWorker is a bounded doer: it runs the actual task under a tight budget
	// and may fan out into several attempts, each capped.
	RoleWorker Role = "worker"
	// RoleJudge is the stronger adjudicator: a larger model that scores or folds
	// the workers' outputs, and the usual escalation target when a worker's budget
	// is spent.
	RoleJudge Role = "judge"
	// RolePrimary is the driver that owns the turn and delegates to the others.
	RolePrimary Role = "primary"
)

// knownRole reports whether r is one of the closed Role set.
func knownRole(r Role) bool {
	switch r {
	case RoleScout, RoleWorker, RoleJudge, RolePrimary:
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// BUDGETS — independent axes that bound a role's consumption.
// ---------------------------------------------------------------------------

// Budget bounds one role's consumption across independent axes. A zero field is
// "unbounded on that axis". The axes are orthogonal: ANY bounded axis reaching
// its limit exhausts the whole budget (a fail-closed OR across axes), so a role
// that is cheap on tokens but chatty on attempts is still stopped.
type Budget struct {
	MaxTokens    int     `json:"max_tokens,omitempty"`     // total prompt+completion tokens
	MaxLatencyMS int     `json:"max_latency_ms,omitempty"` // wall-clock budget in milliseconds
	MaxAttempts  int     `json:"max_attempts,omitempty"`   // calls / retries allowed
	MaxCost      float64 `json:"max_cost,omitempty"`       // spend ceiling in currency units
}

// Spend is the consumed-so-far tally for a role, in the same axes as Budget. The
// wiring layer fills it from live meters; RouteRoles only reads it.
type Spend struct {
	Tokens    int     `json:"tokens,omitempty"`
	LatencyMS int     `json:"latency_ms,omitempty"`
	Attempts  int     `json:"attempts,omitempty"`
	Cost      float64 `json:"cost,omitempty"`
}

// Exhausted reports whether spend has met or exceeded any bounded axis of the
// budget, and names the first axis (in a stable order: tokens, latency, attempts,
// cost) that did. A zero-limit axis is unbounded and never exhausts, so an
// all-zero Budget is never exhausted.
func (b Budget) Exhausted(s Spend) (axis string, exhausted bool) {
	if b.MaxTokens > 0 && s.Tokens >= b.MaxTokens {
		return "tokens", true
	}
	if b.MaxLatencyMS > 0 && s.LatencyMS >= b.MaxLatencyMS {
		return "latency", true
	}
	if b.MaxAttempts > 0 && s.Attempts >= b.MaxAttempts {
		return "attempts", true
	}
	if b.MaxCost > 0 && s.Cost >= b.MaxCost {
		return "cost", true
	}
	return "", false
}

// ---------------------------------------------------------------------------
// THE ROLE PLAN — bind a role to model constraints, a budget, escalation, fold.
// ---------------------------------------------------------------------------

// RolePlan binds one Role to its OWN model-routing subject (its independent model
// constraints), an independent Budget, an escalation target, and a result-fold.
// Keeping the Subject per-role is what stops a single model id from being
// overloaded across duties: the scout, worker, and judge each route through the
// shared modelroute oracle on their own terms.
type RolePlan struct {
	Role Role `json:"role"`
	// Subject is routed through the shared modelroute Manifest to pick this role's
	// model(s); the role and the model policy stay separate.
	Subject modelroute.Subject `json:"subject"`
	// Budget bounds this role independently of the others.
	Budget Budget `json:"budget"`
	// EscalateTo is the role that takes over when this role's budget is exhausted
	// ("" == no escalation; the role simply stops). It MUST name a role also
	// declared in the same RolePlanSet.
	EscalateTo Role `json:"escalate_to,omitempty"`
	// Fold is how THIS role's own (possibly ensemble) outputs reduce into one.
	Fold modelroute.Reduction `json:"fold,omitempty"`
}

// RolePlanSet is the declared multi-role plan: an ordered list of RolePlan values plus
// the top-level Fold that composes the roles' results into the final answer.
// Order is operator policy and is preserved into the envelope, so the trace is
// stable and diffable.
type RolePlanSet struct {
	Plans []RolePlan `json:"plans"`
	// Fold composes the roles' results into one answer when no escalation changes
	// the adjudication (see RoleEnvelope.EffectiveFold).
	Fold modelroute.Reduction `json:"fold,omitempty"`
}

// ---------------------------------------------------------------------------
// THE RESOLVED ENVELOPE — inspectable role decisions for one execution.
// ---------------------------------------------------------------------------

// RoleStatus is the resolved lifecycle state of a role after budgeting.
type RoleStatus string

const (
	// RoleActive means the role is within budget and runs its own plan.
	RoleActive RoleStatus = "active"
	// RoleExhausted means a bounded axis is spent; the declared escalation (if any)
	// has fired.
	RoleExhausted RoleStatus = "exhausted"
)

// RoleDecision is one role's resolved routing: the model Plan chosen by the shared
// modelroute oracle, the budget it runs under, the spend against it, whether that
// budget is exhausted (and on which axis), and — when exhausted — the role it
// escalated to. Every field is inspectable so the delegation is auditable rather
// than collapsed into a single model id.
type RoleDecision struct {
	Role          Role                 `json:"role"`
	Plan          modelroute.Plan      `json:"plan"`
	Budget        Budget               `json:"budget"`
	Spend         Spend                `json:"spend"`
	Status        RoleStatus           `json:"status"`
	ExhaustedAxis string               `json:"exhausted_axis,omitempty"`
	EscalatedTo   Role                 `json:"escalated_to,omitempty"`
	Fold          modelroute.Reduction `json:"fold,omitempty"`
	Reason        string               `json:"reason"`
}

// RoleEnvelope is the parent, inspectable record of every role decision for one
// execution. It preserves declared role order and keeps each role's budget,
// escalation, and fold visible — the delegation analogue of the Decision envelope
// that keeps harness/model/session distinct instead of flattening them.
type RoleEnvelope struct {
	Roles []RoleDecision `json:"roles"`
	// Fold is the set's top-level reduction, echoed for inspection.
	Fold modelroute.Reduction `json:"fold,omitempty"`
}

// Role returns the resolved decision for role r, if present.
func (e RoleEnvelope) Role(r Role) (RoleDecision, bool) {
	for _, d := range e.Roles {
		if d.Role == r {
			return d, true
		}
	}
	return RoleDecision{}, false
}

// Escalated reports whether any role exhausted its budget and escalated.
func (e RoleEnvelope) Escalated() bool {
	for _, d := range e.Roles {
		if d.Status == RoleExhausted && d.EscalatedTo != "" {
			return true
		}
	}
	return false
}

// EffectiveFold is the reduction that folds the execution's results into one
// answer. It encodes the declared escalation/fold policy: when a role has
// exhausted its budget and escalated, the escalation TARGET's fold governs the
// final adjudication (e.g. the judge's best_of decides among the workers'
// partials); absent any escalation, the set's top-level Fold applies. The first
// escalation in declared order wins, so the result is deterministic.
func (e RoleEnvelope) EffectiveFold() modelroute.Reduction {
	for _, d := range e.Roles {
		if d.Status == RoleExhausted && d.EscalatedTo != "" {
			if target, ok := e.Role(d.EscalatedTo); ok && target.Fold != "" {
				return target.Fold
			}
		}
	}
	return e.Fold
}

// ---------------------------------------------------------------------------
// ROUTING — resolve a declared plan into an inspectable envelope.
// ---------------------------------------------------------------------------

// RouteRoles resolves each declared RolePlan into an inspectable RoleDecision: it
// routes the role's Subject through the shared modelroute oracle (so role policy
// and model policy stay separate, and no model id is overloaded across duties),
// tallies the role's spend against its independent Budget, and — when a bounded
// axis is exhausted — fires the declared escalation to a stronger role. The
// returned RoleEnvelope is the parent record; declared role order is preserved.
// spends is the consumed-so-far tally per role (a nil / missing entry is a fresh,
// zero spend). It is pure and deterministic: no I/O, no goroutines.
func RouteRoles(set RolePlanSet, manifest modelroute.Manifest, spends map[Role]Spend) (RoleEnvelope, error) {
	if len(set.Plans) == 0 {
		return RoleEnvelope{}, fmt.Errorf("execution route: no role plans are declared")
	}
	declared := make(map[Role]bool, len(set.Plans))
	for _, p := range set.Plans {
		if !knownRole(p.Role) {
			return RoleEnvelope{}, fmt.Errorf("execution route: unknown role %q", p.Role)
		}
		if declared[p.Role] {
			return RoleEnvelope{}, fmt.Errorf("execution route: role %q is declared more than once", p.Role)
		}
		declared[p.Role] = true
	}
	env := RoleEnvelope{Fold: set.Fold, Roles: make([]RoleDecision, 0, len(set.Plans))}
	for _, p := range set.Plans {
		if p.EscalateTo != "" && !declared[p.EscalateTo] {
			return RoleEnvelope{}, fmt.Errorf("execution route: role %q escalates to undeclared role %q", p.Role, p.EscalateTo)
		}
		plan := manifest.Route(p.Subject).Plan
		spend := spends[p.Role]
		axis, spent := p.Budget.Exhausted(spend)
		dec := RoleDecision{
			Role:   p.Role,
			Plan:   plan,
			Budget: p.Budget,
			Spend:  spend,
			Status: RoleActive,
			Fold:   p.Fold,
			Reason: fmt.Sprintf("role %s within budget", p.Role),
		}
		if spent {
			dec.Status = RoleExhausted
			dec.ExhaustedAxis = axis
			if p.EscalateTo != "" {
				dec.EscalatedTo = p.EscalateTo
				dec.Reason = fmt.Sprintf("role %s exhausted %s budget; escalated to %s", p.Role, axis, p.EscalateTo)
			} else {
				dec.Reason = fmt.Sprintf("role %s exhausted %s budget; no escalation declared", p.Role, axis)
			}
		}
		env.Roles = append(env.Roles, dec)
	}
	return env, nil
}

// RouteWithRoles composes the harness/model/session envelope with the sub-model
// role plan, keeping the role decisions inspectable inside the parent Decision.
// It reuses the same modelroute Manifest for both the top-level model route and
// the per-role routing, so one policy governs the whole execution.
func RouteWithRoles(req Request, set RolePlanSet, profiles []harnessprofile.HarnessProfile, manifest modelroute.Manifest, spends map[Role]Spend) (Decision, error) {
	dec, err := Route(req, profiles, manifest)
	if err != nil {
		return Decision{}, err
	}
	env, err := RouteRoles(set, manifest, spends)
	if err != nil {
		return Decision{}, err
	}
	dec.Roles = &env
	return dec, nil
}
