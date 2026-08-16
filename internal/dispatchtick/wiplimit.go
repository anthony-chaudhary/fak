package dispatchtick

import "fmt"

// wiplimit.go -- the WIP cap term: a flow limit on STARTED-AND-UNFINISHED units,
// folded into dispatch admission so a new unit is not started onto a fleet that
// already owns more unfinished work than it can finish.
//
// Every other admission term in this package is a RESOURCE limit. ConfiguredCap is
// an operator's number, LeaseCap the kernel's session target, HostCap the box's
// cores/RAM/threads, SeatCap the account-seat inventory, WorkerFloor a forecast
// pre-warm, ContractionCap a pending drain, WorktreeCap a proven isolation ceiling;
// the post-folds add gate latency, rate-limit bursts, host churn, and fresh-seat
// slots. Each answers "can the BOX or the ACCOUNT POOL carry another worker?".
//
// None of them answers "does the fleet already OWN too much unfinished work?", and
// the two questions come apart because the quantity admission measures -- res.Live,
// which is max(kernel alive, OS worker procs) -- counts live worker PROCESSES. A
// worker process is not a unit of work. When a session ends, crashes, or is reaped,
// the process disappears from res.Live but the UNIT it started does not disappear:
// it keeps its branch, its commits, and its unclosed issue. So res.Live systematically
// UNDER-counts owned work by exactly the number of started-then-abandoned units, and
// headroom = Cap - Live reads as free capacity on a fleet whose real in-hand inventory
// is already large. Admission then starts another unit on top of it, which is the
// mechanism by which started-and-unfinished work accumulates without bound: nothing
// in the loop ever charges a start against work already in hand.
//
// This term charges it. It is a pure LOWERING fold in the established shape of
// ApplyGateBackpressure / ApplyRateLimitBackpressure / ApplyChurnBackpressure: it can
// only tighten the cap, never raise it, and its zero value abstains, so a caller that
// supplies no census gets a byte-identical preflight.
//
// INVENTORY IS NOT WIP. The census keeps the two quantities in separate fields and
// charges only one of them. Started (admitted AND begun, not yet ended) is WIP and
// charges the limit. Inventory (filed or admitted but never begun) is NOT charged at
// any weight. This is load-bearing rather than cosmetic: this repo carries a backlog
// three orders of magnitude larger than its real concurrency, so a fold that charged
// inventory would refuse every spawn forever and the gate would be turned off within
// a day -- which is the actual failure mode of most WIP limits. internal/flowmetrics
// draws the same line for the same reason ("An unstarted issue is backlog, not WIP:
// conflating the two is what makes a 1300-issue backlog look like 1300 units of work
// in progress when the real concurrent count is far smaller"). Inventory is carried
// through only so the refusal reason can SAY it was not charged.

const (
	// PreflightRefuseWIPLimit is the closed refusal token emitted when the flow limit
	// is the sole binding term: the box and the account pool would both carry another
	// worker, and admission refuses anyway because the fleet already owns its limit of
	// started-and-unfinished units. It is deliberately distinct from PreflightRefuseAtCap
	// so an operator reading a tick can tell "out of machine" from "out of flow", which
	// have opposite remedies -- the first wants more capacity, the second wants the
	// in-hand units FINISHED, and answering the second with the first is what grows WIP.
	PreflightRefuseWIPLimit = "REFUSE_WIP_LIMIT"

	// WIPLimiting is the cap_terms.limiting value recorded when this term binds.
	WIPLimiting = "wip_limit"

	// WIPLimitEnv is the operator knob that declares the flow limit. It is opt-in and
	// unset by default: a limit is a policy choice about how much unfinished work this
	// fleet will carry, and guessing one for an operator would either be so loose it
	// never binds or so tight it refuses a working fleet on its first tick. Unset means
	// the census is not binding and admission is byte-identical to before this term.
	WIPLimitEnv = "FAK_WIP_LIMIT"
)

// WIPCensus is the measured flow-limit input: how much started-and-unfinished work
// the fleet owns right now, and how much it is allowed to own.
//
// The zero value is the ABSTAIN value (Measured false, Limit 0) and makes ApplyWIPLimit
// an identity function, so every existing caller is byte-identical to before this term
// existed. Measured is explicit rather than inferred from Started > 0 because a census
// that legitimately reads zero WIP ("nothing is in flight") and a census that could not
// be taken ("the ledger was unreadable") are different facts, and only the first should
// license a spawn on the strength of the WIP term.
type WIPCensus struct {
	// Measured is true only when the producer actually read the WIP surface. False means
	// no signal -- the fold abstains rather than admitting or refusing on absent data.
	Measured bool `json:"measured"`

	// Started is the count of units that were BEGUN and have not FINISHED. This is the
	// WIP quantity, and the only one charged against Limit.
	Started int `json:"started"`

	// Inventory is the count of units that are filed or admitted but NEVER BEGUN. It is
	// carried for the refusal reason and is never charged -- see the file comment.
	Inventory int `json:"inventory"`

	// Limit is the flow limit: the most started-and-unfinished units the fleet may own.
	// Zero or negative means "no flow limit declared" and the fold abstains.
	Limit int `json:"limit"`
}

// Binding reports whether the census carries a usable flow limit.
func (c WIPCensus) Binding() bool { return c.Measured && c.Limit > 0 }

// Allowance is how many NEW units the flow limit still permits, floored at zero.
// A fleet already at or over its limit has an allowance of 0, not a negative one:
// the term refuses further starts, it never demands that workers be killed.
func (c WIPCensus) Allowance() int {
	if n := c.Limit - c.Started; n > 0 {
		return n
	}
	return 0
}

// capFor converts the allowance into a cap comparable with the other terms. Every
// term in this package caps CONCURRENT WORKERS, so the flow limit has to be expressed
// in the same units before it can join the min()-fold: holding at live + allowance
// leaves exactly Allowance() spawns of headroom above whatever is already running.
func (c WIPCensus) capFor(live int) int {
	hold := live + c.Allowance()
	if hold < 0 {
		return 0
	}
	return hold
}

// ApplyWIPLimit folds the measured WIP census into a preflight verdict as a lowering
// cap term, so a start is charged against work the fleet already owns.
//
// It abstains -- returning res untouched -- in four cases:
//
//   - res is already a refusal. A higher-precedence term (host, seat, account, at-cap)
//     is already stopping growth, so the flow limit is not the SOLE binding term and
//     the existing verdict, which names a more actionable cause, stands. This mirrors
//     ApplyGateBackpressure's !res.OK guard.
//   - the census is unmeasured. No signal, no opinion.
//   - no flow limit is declared (Limit <= 0). The term is opt-in.
//   - the hold meets or exceeds the existing cap. The term only ever LOWERS; it cannot
//     manufacture capacity that the box and the seat pool have not already granted.
//
// When it does bind, it lowers the cap to live + allowance. If that still leaves
// headroom the verdict stays SPAWN_OK with a smaller wave -- the fleet may start
// fewer units, which is the point of a flow limit. Only when the allowance is
// exhausted does it refuse with PreflightRefuseWIPLimit.
//
// Note the asymmetry with ApplyGateBackpressure's cold-start floor: that fold keeps a
// pressured-but-cold fleet SPAWN_OK because a slow kernel needs its first worker to
// clear the very backlog that gates it, so a zero cap would deadlock. There is no such
// deadlock here and so no floor: WIP is only ever reduced by FINISHING units, which is
// done by workers that are already running, so refusing new starts cannot wedge the
// fold shut. A fleet at its flow limit recovers by closing what it holds.
func ApplyWIPLimit(res PreflightResult, c WIPCensus) PreflightResult {
	if !res.OK || !c.Binding() {
		return res
	}
	hold := c.capFor(res.Live)
	if hold >= res.Cap {
		return res
	}
	res.Cap = hold
	res.Headroom = hold - res.Live
	res.CapTerms.EffectiveCap = hold
	res.CapTerms.Limiting = WIPLimiting
	res.CapTerms.WIPStarted = c.Started
	res.CapTerms.WIPLimit = c.Limit
	if res.Headroom > 0 {
		// The limit binds but is not exhausted: admit a SMALLER wave.
		return res
	}
	res.OK = false
	res.Verdict = PreflightRefuseWIPLimit
	res.Reason = wipLimitReason(c)
	return res
}

// wipLimitReason names the refusal and, critically, states that inventory was NOT
// charged. An operator who reads "at the flow limit" on a repo with a four-digit
// backlog will assume the backlog caused it and disable the gate; saying the quiet
// part out loud -- that N unstarted units were excluded by construction -- is what
// keeps the inventory-vs-WIP distinction legible at the moment it matters.
func wipLimitReason(c WIPCensus) string {
	return fmt.Sprintf("%s: the fleet already owns %d started-and-unfinished unit(s) against a flow limit of %d - finish work in hand before starting more (%d unstarted unit(s) are inventory, not WIP, and are not charged)",
		PreflightRefuseWIPLimit, c.Started, c.Limit, c.Inventory)
}
