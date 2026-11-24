package routing

// DefaultMaxAttempts is the total upstream attempts one client request may
// spend when the routing config declares no max_attempts. It is finite and
// positive, and large enough that every declared chain the router ships (a
// primary plus a few fallbacks, each with a short same-backend key rollover)
// walks exactly as it did before the budget existed.
const DefaultMaxAttempts = 16

// AttemptBudget is the ONE bound on the upstream attempts a single client
// request may spend across every layer that re-executes it: the chain walk,
// the same-backend key retry (still capped per class by Failure.RetryBudget),
// the durable zero-output requeue, lease-expiry reclaim, and a client's
// POST .../retry (RA-01 leaf 6). The unit is one upstream dial.
//
// The fabric owns it next to Failure.RetryBudget (docs/ROUTER-FAILOVER-AUTHORITY.md,
// "Budget" plane): Failure.RetryBudget answers "how many more tries at THIS
// target", AttemptBudget answers "how many dials in total for this request".
// A caller records what it spent; it never invents a ceiling of its own.
type AttemptBudget struct {
	// Max is the total upstream attempts; <= 0 means DefaultMaxAttempts.
	Max int `json:"max"`
}

// NewAttemptBudget returns a budget of max attempts, or DefaultMaxAttempts when
// max <= 0 (an unset config key).
func NewAttemptBudget(max int) AttemptBudget {
	if max <= 0 {
		max = DefaultMaxAttempts
	}
	return AttemptBudget{Max: max}
}

// Remaining is how many further upstream attempts the budget authorizes after
// spent have been made: the limit minus spent, floored at 0. The limit is Max,
// or DefaultMaxAttempts when Max <= 0, so the zero value (a caller that never
// resolved a decision) is still bounded: never unbounded, never zero.
func (b AttemptBudget) Remaining(spent int) int {
	limit := b.Max
	if limit <= 0 {
		limit = DefaultMaxAttempts
	}
	return max(limit-max(spent, 0), 0)
}
