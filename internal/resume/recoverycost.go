package resume

// recoverycost.go — the OBSERVED recovery-cost governor (#4146): a pure fold that sums the
// provider tokens/cost a session has spent on POST-resume turns, plus the declared cap whose
// breach flips a launch to a REVERSIBLE hold carrying RESUME_COST_EXCEEDED.
//
// # A SEPARATE governor, never mixed with the witnessed budget
//
// Automatic resume is rationed today only by attempt COUNT (EarnedResumeBudget, #3124) — a
// witnessed quantity. This governor adds a second, ORTHOGONAL rein: how much OBSERVED
// provider spend the recovery has already cost. The two never mix. An OBSERVED token/cost sum
// is never added into EarnedResumeBudget or any witnessed number (the doctrine's
// WITNESSED-is-never-summed-with-OBSERVED rule); the attempt cap stays exactly as it was, and
// this fold lives in its own field on NextInput (RecoveryCostBreached). Keeping them apart is
// the whole point: a cheap thrash and an expensive single recovery are different failures.
//
// # Reversible-first
//
// A breach never tears anything down. It tops the launch ladder out at ActHoldAdmission — a
// deferred, operator-forceable HOLD (next.go) — exactly the reversible posture the resume
// package takes everywhere. Below the cap, launch normally.
//
// # Pure fold; attribution mirrors NewTurnsAfter
//
// Like NewTurnsAfter (the witnessed progress count), the cost sum attributes a turn to the
// recovery iff its timestamp is strictly after the launch boundary. Same boundary rule, a
// different quantity (OBSERVED cost instead of a count). No clock, no I/O — the shell reads
// the per-turn provider numbers and hands them in.

// ResumeCostExceeded is the closed refusal reason a recovery-cost breach emits (#4146). It is
// emittable (this governor stamps it), verifiable (FoldRecoveryCost vs RecoveryCostCap re-
// derives it from OBSERVED numbers), and refusable (FoldNextAction routes it to a reversible
// hold). The token is byte-identical to the dos.toml [reasons] entry the orchestrator adds.
const ResumeCostExceeded = "RESUME_COST_EXCEEDED"

// RecoveryTurnCost is one OBSERVED post-resume turn's provider cost — the meter this governor
// sums. Tokens and CostMicroUSD are provider-reported OBSERVED numbers (never a witnessed or
// earned quantity); UnixSeconds attributes the turn to a launch boundary the same way
// NewTurnsAfter attributes a progress turn.
type RecoveryTurnCost struct {
	// UnixSeconds is the turn's wall-clock timestamp in epoch-seconds (the same unit
	// LastLaunchUnix and NewTurnsAfter use). A turn counts toward the recovery cost iff this is
	// strictly greater than the launch boundary.
	UnixSeconds int64 `json:"unix_seconds"`
	// Tokens is the OBSERVED provider token count for this turn (prompt+completion, as the
	// shell chooses to attribute). Negative values are ignored (a bad reading never lowers the
	// sum below a real one).
	Tokens int `json:"tokens"`
	// CostMicroUSD is the OBSERVED provider cost for this turn in micro-USD (1e-6 USD), an
	// integer to keep the sum exact. Negative values are ignored.
	CostMicroUSD int64 `json:"cost_micro_usd"`
}

// RecoveryCost is the folded OBSERVED recovery spend attributed to post-resume turns for one
// session (across relaunches when the shell keys the boundary at the FIRST launch). Every
// field is OBSERVED — this struct is never summed into a witnessed budget.
type RecoveryCost struct {
	Turns        int   `json:"turns"`
	Tokens       int   `json:"tokens"`
	CostMicroUSD int64 `json:"cost_micro_usd"`
}

// FoldRecoveryCost sums the OBSERVED tokens/cost of every turn strictly after sinceUnix — the
// recovery spend attributed to the resume. The boundary rule mirrors NewTurnsAfter exactly:
// sinceUnix <= 0 (no launch on record) attributes nothing, and only turns with UnixSeconds >
// sinceUnix count, so a turn at or before the launch is never charged to the recovery. To
// meter a session's CUMULATIVE recovery cost across relaunches, the shell passes the FIRST
// launch's UnixSeconds; to meter only the latest launch, it passes the last. Total over any
// input; a nil slice folds to the zero cost.
func FoldRecoveryCost(turns []RecoveryTurnCost, sinceUnix int64) RecoveryCost {
	var rc RecoveryCost
	if sinceUnix <= 0 {
		return rc
	}
	for _, t := range turns {
		if t.UnixSeconds <= sinceUnix {
			continue
		}
		rc.Turns++
		if t.Tokens > 0 {
			rc.Tokens += t.Tokens
		}
		if t.CostMicroUSD > 0 {
			rc.CostMicroUSD += t.CostMicroUSD
		}
	}
	return rc
}

// RecoveryCostCap is the declared per-session recovery-cost envelope: the OBSERVED post-resume
// spend ceiling above which a NEW automatic resume flips to a reversible hold. It is a
// declared policy input, NOT a measurement. A zero field means that dimension is UNBOUNDED (no
// cap on it) — an all-zero cap gates nothing, the fail-open default so an un-configured
// deployment behaves exactly as it did before this governor existed.
type RecoveryCostCap struct {
	// MaxTokens caps cumulative OBSERVED recovery tokens; 0 = unbounded.
	MaxTokens int `json:"max_tokens,omitempty"`
	// MaxCostMicroUSD caps cumulative OBSERVED recovery cost in micro-USD; 0 = unbounded.
	MaxCostMicroUSD int64 `json:"max_cost_micro_usd,omitempty"`
}

// Exceeded reports whether the folded OBSERVED cost has crossed any bounded dimension of the
// cap, and the closed reason to emit when it has. A dimension with a zero cap never gates. The
// compare is strictly greater: spending exactly up to the declared ceiling is still allowed;
// crossing it is the breach. Total over any input.
func (c RecoveryCostCap) Exceeded(rc RecoveryCost) (bool, string) {
	if c.MaxTokens > 0 && rc.Tokens > c.MaxTokens {
		return true, ResumeCostExceeded
	}
	if c.MaxCostMicroUSD > 0 && rc.CostMicroUSD > c.MaxCostMicroUSD {
		return true, ResumeCostExceeded
	}
	return false, ""
}
