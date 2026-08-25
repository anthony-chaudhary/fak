package supervisionpolicy

import (
	"math"
	"time"
)

// Role is a closed member role. Unknown values fail closed.
type Role uint8

const (
	RoleAuthority Role = iota + 1
	RoleAdapter
	RoleProjection
	RoleHelper
)

// Strategy is a closed supervision strategy. Unknown values fail closed.
type Strategy uint8

const (
	StrategyOneForOne Strategy = iota + 1
	StrategyHoldAuthority
	StrategyEscalateDomain
)

// Action is the only effect a caller may enact from a Decision.
type Action uint8

const (
	ActionRestart Action = iota + 1
	ActionReattach
	ActionHold
	ActionEscalate
)

// Outcome is a closed, payload-free explanation of a decision.
type Outcome uint8

const (
	OutcomeRecover Outcome = iota + 1
	OutcomeUnknownRole
	OutcomeUnknownStrategy
	OutcomeStaleGeneration
	OutcomeUncertainEffect
	OutcomeCheckpointRequired
	OutcomeBackoff
	OutcomeBudgetExhausted
	OutcomeStrategyHold
	OutcomeStrategyEscalation
	OutcomeInvalidIdentity
	OutcomeInvalidBudget
	OutcomeWriterEpochExhausted
)

// LogicalSessionID and MemberID are stable identities across physical recovery.
type LogicalSessionID string
type MemberID string
type DomainID string
type CheckpointRef string
type EvidenceRef string

// Budget bounds recovery intensity. Backoff is deterministic: the delay after
// the nth failure in a window is BaseBackoff*n, capped at MaxBackoff when set.
type Budget struct {
	MaxRestarts uint32
	Window      time.Duration
	BaseBackoff time.Duration
	MaxBackoff  time.Duration
}

// Request is an observed failure submitted to the pure policy. Failures are
// prior restart attempt times; callers supply Now so evaluation is deterministic.
type Request struct {
	Role               Role
	Strategy           Strategy
	Domain             DomainID
	Session            LogicalSessionID
	Member             MemberID
	Generation         uint64
	ObservedGeneration uint64
	WriterEpoch        uint64
	EffectCertain      bool
	Checkpoint         CheckpointRef
	Evidence           []EvidenceRef
	Now                time.Time
	Failures           []time.Time
	Budget             Budget
}

// BudgetState is the receipt's complete, bounded intensity accounting.
type BudgetState struct {
	Used       uint32
	Remaining  uint32
	WindowFrom time.Time
	RetryAt    time.Time
}

// Decision is a closed receipt. It deliberately has no message, transcript, or
// arbitrary payload field. NextWriterEpoch is fenced only for authority restart.
type Decision struct {
	Role            Role
	Domain          DomainID
	Session         LogicalSessionID
	Member          MemberID
	Generation      uint64
	WriterEpoch     uint64
	NextWriterEpoch uint64
	Action          Action
	Outcome         Outcome
	Budget          BudgetState
	Evidence        []EvidenceRef
}

// Decide returns a deterministic, fail-closed recovery receipt.
func Decide(r Request) Decision {
	d := Decision{
		Role:        r.Role,
		Domain:      r.Domain,
		Session:     r.Session,
		Member:      r.Member,
		Generation:  r.Generation,
		WriterEpoch: r.WriterEpoch,
		Action:      ActionHold,
		Evidence:    append([]EvidenceRef(nil), r.Evidence...),
	}

	if !validBudget(r.Budget) {
		d.Outcome = OutcomeInvalidBudget
		return d
	}
	d.Budget = budgetState(r)

	if r.Domain == "" || r.Session == "" || r.Member == "" {
		d.Outcome = OutcomeInvalidIdentity
		return d
	}
	if !validRole(r.Role) {
		d.Outcome = OutcomeUnknownRole
		return d
	}
	if !validStrategy(r.Strategy) {
		d.Outcome = OutcomeUnknownStrategy
		return d
	}
	if r.Generation != r.ObservedGeneration {
		d.Outcome = OutcomeStaleGeneration
		return d
	}
	if !r.EffectCertain {
		d.Outcome = OutcomeUncertainEffect
		return d
	}
	if r.Strategy == StrategyEscalateDomain {
		d.Action = ActionEscalate
		d.Outcome = OutcomeStrategyEscalation
		return d
	}
	if r.Role == RoleAuthority && r.Strategy == StrategyHoldAuthority {
		d.Outcome = OutcomeStrategyHold
		return d
	}
	if r.Role == RoleAuthority && r.Checkpoint == "" {
		d.Outcome = OutcomeCheckpointRequired
		return d
	}
	if r.Role == RoleAuthority && r.WriterEpoch == math.MaxUint64 {
		d.Outcome = OutcomeWriterEpochExhausted
		return d
	}
	if d.Budget.Used >= r.Budget.MaxRestarts {
		d.Action = ActionEscalate
		d.Outcome = OutcomeBudgetExhausted
		return d
	}
	if !d.Budget.RetryAt.IsZero() && r.Now.Before(d.Budget.RetryAt) {
		d.Outcome = OutcomeBackoff
		return d
	}

	d.Outcome = OutcomeRecover
	if r.Role == RoleProjection {
		d.Action = ActionReattach
		return d
	}
	d.Action = ActionRestart
	if r.Role == RoleAuthority {
		d.NextWriterEpoch = r.WriterEpoch + 1
	} else {
		d.NextWriterEpoch = r.WriterEpoch
	}
	return d
}

func validRole(role Role) bool {
	return role >= RoleAuthority && role <= RoleHelper
}

func validStrategy(strategy Strategy) bool {
	return strategy >= StrategyOneForOne && strategy <= StrategyEscalateDomain
}

func validBudget(b Budget) bool {
	return b.MaxRestarts > 0 && b.Window > 0 && b.BaseBackoff >= 0 && b.MaxBackoff >= 0 &&
		(b.MaxBackoff == 0 || b.BaseBackoff <= b.MaxBackoff)
}

func budgetState(r Request) BudgetState {
	from := r.Now.Add(-r.Budget.Window)
	var used uint32
	var latest time.Time
	for _, failure := range r.Failures {
		if failure.After(r.Now) || failure.Before(from) {
			continue
		}
		used++
		if latest.IsZero() || failure.After(latest) {
			latest = failure
		}
	}
	remaining := uint32(0)
	if used < r.Budget.MaxRestarts {
		remaining = r.Budget.MaxRestarts - used
	}
	state := BudgetState{Used: used, Remaining: remaining, WindowFrom: from}
	if used > 0 && r.Budget.BaseBackoff > 0 {
		state.RetryAt = latest.Add(scaledBackoff(r.Budget, used))
	}
	return state
}

func scaledBackoff(b Budget, used uint32) time.Duration {
	limit := time.Duration(math.MaxInt64)
	if b.MaxBackoff > 0 {
		limit = b.MaxBackoff
	}
	if time.Duration(used) > limit/b.BaseBackoff {
		return limit
	}
	return min(time.Duration(used)*b.BaseBackoff, limit)
}
