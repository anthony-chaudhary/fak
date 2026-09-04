// Package dispatchaging is the deterministic anti-starvation term the fak issue-dispatch
// order is missing: given a set of READY (already dispatchable) work units, it decides which
// one a worker should pick FIRST when raw priority alone would let a low-priority unit wait
// forever. It is the computable answer to the operator question "issue #42 is only a P2, and a
// steady drip of P0/P1 work keeps outranking it — will it EVER run?".
//
// # The gap it closes
//
// The fleet's two dispatch-order leaves rank by base priority ABSOLUTELY:
//
//   - internal/dispatchtick.OrderLaneCandidates sorts by priority weight (P0=1000, P1=400,
//     P2=150, default=60) descending, with only a by-number recency TIEBREAK.
//   - internal/dispatchorder.Plan sorts DispKeep units by declared Priority descending first,
//     then a recency/PreferOldest/ID tiebreak.
//
// In both, priority LEADS: a lower-weight unit never overtakes a higher-weight one, only ties
// break by age. So a ready unit that is perpetually out-weighted by fresher higher-priority
// arrivals is never picked — classic priority-scheduling STARVATION. internal/dispatchconservation
// can already SEE the symptom (its re-storm "churn" count flags issues that burned units while
// others got none), but nothing feeds "how long has this unit been waiting" back into the order.
// This package is that missing feedback: an EFFECTIVE weight = base priority + a bounded aging
// boost that grows with wait time, plus a hard starvation deadline that force-promotes any ready
// unit that has waited past a floor. The invariant it buys: no ready unit waits forever.
//
// # Two independent guarantees
//
//  1. Soft aging (monotonic): effective weight rises by BoostPerInterval for every full
//     IntervalSeconds a unit has waited. Because it is unbounded by default and monotonic, a
//     long-waiting light unit eventually out-weighs any FIXED heavier tier — so permanent
//     starvation is impossible even with the hard deadline disabled. A caller can cap the soft
//     boost with MaxBoostPoints when it wants priority inversion bounded.
//  2. Hard starvation deadline: a unit that has waited >= StarvationSeconds is StandingStarved
//     and is admitted ahead of every non-starved unit THIS tick, whatever its base weight. This
//     bounds the worst-case wait to a fixed number independent of the boost arithmetic — the
//     fail-closed rung an operator can reason about directly.
//
// # Both lenses
//
// Fairness control AND throughput optimization are the same mechanism here: draining a unit that
// would otherwise rot prevents the later re-storm churn dispatchconservation charges as wasted
// capacity, and removes the operator toil of hand-bumping a starved issue's priority label.
//
// # Evidence, not claims; data, not code; additive, no regression
//
// The standing is a pure function of the wait CLOCK (Now, supplied as data — the leaf never reads
// one) against each unit's ReadySince stamp, counting ELIGIBLE time only: a declared cooldown
// window pauses the clock, so time a unit could not be picked never accrues starvation pressure
// (see Candidate.CoolingSince). It trusts no worker's self-report. The knobs live in
// a small declared Params struct with documented defaults (candidates for dos.toml later). And the
// term is purely additive: the zero-value Params disables aging entirely, so Fold then orders by
// base weight (then wait, then ID) — byte-identical to the pre-aging order (see the golden test).
//
// # Pure and total
//
// Fold takes a clock reading as data, imports nothing internal, does no I/O — same input, same
// Result. The impure half (gather the ready candidates from the live backlog, then act on the
// order) belongs in the cmd/fak shell, exactly the leaf/shell split dispatchorder (the decision)
// and its wire use. Total over any input; an empty candidate set yields an empty, defined Result.
package dispatchaging

import "sort"

// Schema tags the machine-readable Result, mirroring fleet-dispatch-conservation/1.
const Schema = "fleet-dispatch-aging/1"

// Documented defaults (see DefaultParams). They are tuned against the priority-weight taxonomy
// dispatchtick declares (P0=1000, P1=400, P2=150, default=60): one default-tier's worth of weight
// (60) accrues every ten minutes, so an unlabeled unit climbs past a P2 in ~20 min, past a P1 in
// ~1 h, and past a P0 in ~2.6 h of waiting — and the hard deadline force-serves it by 6 h no
// matter how heavy the competition.
const (
	// DefaultIntervalSeconds is the aging quantum: every full interval waited adds BoostPerInterval.
	DefaultIntervalSeconds = 600 // 10 minutes
	// DefaultBoostPerInterval is the priority points a unit earns per full interval waited. 60
	// equals dispatchtick.PriorityWeightDefault — one unlabeled tier per quantum.
	DefaultBoostPerInterval = 60
	// DefaultMaxBoostPoints caps the soft aging boost. 0 == uncapped: soft aging alone can, given
	// enough time, out-weigh any fixed tier (the no-permanent-starvation property). Set positive to
	// bound how far soft aging may invert priority and lean on the hard deadline for big leaps.
	DefaultMaxBoostPoints = 0
	// DefaultStarvationSeconds is the hard deadline: a ready unit waiting this long is force-served.
	// It matches dispatchconservation's default reporting window so a starving unit surfaces within
	// one conservation window. 0 == disabled (soft aging is then the only anti-starvation term).
	DefaultStarvationSeconds = 6 * 3600 // 6 hours

	// boostClamp bounds the computed soft boost so an absurd (or corrupt) wait can never overflow
	// the effective weight; it is far above any real priority tier, so it never perturbs ordering.
	boostClamp = 1 << 30
)

// Standing is the closed anti-starvation verdict for one ready unit — the whole vocabulary.
type Standing string

const (
	// StandingFresh: waited less than one aging interval; ranked purely on base weight (aging is a
	// no-op for it). The majority case, and the one that preserves the pre-aging order exactly.
	StandingFresh Standing = "fresh"
	// StandingAging: has accrued a positive aging boost, so its effective weight exceeds its base
	// weight — it is climbing the order but has not hit the hard deadline.
	StandingAging Standing = "aging"
	// StandingStarved: waited past StarvationSeconds; force-promoted ahead of every non-starved unit
	// this tick regardless of base weight. The hard, fail-closed anti-starvation guarantee.
	StandingStarved Standing = "starved"
)

// Params tunes the fold; the ZERO value disables aging entirely (the additive no-regression
// baseline). Use DefaultParams for the tuned, anti-starvation-on configuration.
type Params struct {
	// NowUnix is the current time as data (the leaf never reads a clock). Waits are Now - ReadySince.
	NowUnix int64 `json:"now_unix"`
	// IntervalSeconds is the aging quantum. <= 0 disables SOFT aging (no boost is ever added).
	IntervalSeconds int64 `json:"interval_seconds"`
	// BoostPerInterval is the weight added per full interval waited. <= 0 disables soft aging.
	BoostPerInterval int `json:"boost_per_interval"`
	// MaxBoostPoints caps the soft boost. <= 0 means uncapped (see DefaultMaxBoostPoints).
	MaxBoostPoints int `json:"max_boost_points"`
	// StarvationSeconds is the hard deadline. <= 0 disables the hard force-serve rung.
	StarvationSeconds int64 `json:"starvation_seconds"`
}

// DefaultParams returns the tuned, anti-starvation-on configuration for the given clock. Callers
// override individual fields (e.g. from flags or dos.toml) after taking this baseline.
func DefaultParams(nowUnix int64) Params {
	return Params{
		NowUnix:           nowUnix,
		IntervalSeconds:   DefaultIntervalSeconds,
		BoostPerInterval:  DefaultBoostPerInterval,
		MaxBoostPoints:    DefaultMaxBoostPoints,
		StarvationSeconds: DefaultStarvationSeconds,
	}
}

func (p Params) softAgingOn() bool    { return p.IntervalSeconds > 0 && p.BoostPerInterval > 0 }
func (p Params) hardDeadlineOn() bool { return p.StarvationSeconds > 0 }

// Candidate is one READY unit of dispatchable work — the facts the aging order needs, none of the
// payload. It carries only its identity, its base priority weight, when it became ready, and its
// latest cooldown window; the caller has already filtered to units that are actually dispatchable
// (not live, not superseded — those dispositions are dispatchorder's job, upstream of this term).
// Cooling is the one disposition that must ALSO be reported here: a cooling unit is ineligible, so
// its wait clock pauses over the declared window instead of accruing phantom starvation pressure.
type Candidate struct {
	// ID is the unit's identity (an issue number as a string, a task id). Echoed in the result.
	ID string `json:"id"`
	// BaseWeight is the priority weight from the label taxonomy (dispatchtick.PriorityWeight:
	// P0=1000, P1=400, P2=150, unlabeled=60). It is the weight aging BOOSTS, never replaces.
	BaseWeight int `json:"base_weight"`
	// ReadySince is the unix time (seconds) the unit first became dispatchable — the wait clock.
	// 0 == unknown, which conservatively waits ZERO seconds: an unknown ready time never invents a
	// boost and never trips the starvation deadline (the fail-closed direction).
	ReadySince int64 `json:"ready_since"`
	// CoolingSince / CoolingUntil describe the unit's current or most recent cooldown window (unix
	// seconds) — the span it was INELIGIBLE to dispatch after a failed attempt (dispatchorder's
	// DispCooling disposition, dispatch_tick's recently-attempted screen). The wait clock PAUSES
	// over this span — its overlap with the waited span is subtracted from the wait — so ineligible
	// time never accrues starvation pressure and a cooled unit never queue-jumps into exactly the
	// re-storm churn dispatchconservation charges as waste. The semantics are pause, NOT reset:
	// wait accrued before the cooldown is preserved, and accrual resumes from that paused value the
	// moment the window ends.
	//
	// Zero values are the legacy no-cooling input and change nothing. A window needs a declared
	// end: CoolingUntil == 0 means no window (CoolingSince alone is ignored). CoolingSince == 0
	// with a declared end conservatively treats the whole wait up to CoolingUntil as cooled — an
	// unknown start may suppress pressure, never invent it (the fail-closed direction). A caller
	// tracking several completed cooldowns folds prior spans by advancing ReadySince by their sum;
	// these fields carry only the latest window.
	CoolingSince int64 `json:"cooling_since,omitempty"`
	CoolingUntil int64 `json:"cooling_until,omitempty"`
}

// waitSeconds is the unit's ELIGIBLE wait against the clock: Now - ReadySince minus the slice of
// that span covered by the declared cooldown window, clamped to >= 0, and 0 when ReadySince is
// unknown. Clock skew (a ReadySince after Now) waits zero, never negative.
func (c Candidate) waitSeconds(now int64) int64 {
	if c.ReadySince <= 0 {
		return 0
	}
	w := now - c.ReadySince
	if w <= 0 {
		return 0
	}
	if w -= c.cooledOverlap(now); w <= 0 {
		return 0
	}
	return w
}

// cooledOverlap is how many of the unit's waited seconds fall inside its declared cooldown window —
// the ineligible time the wait clock skips. The window is clipped to the waited span [ReadySince,
// now]; a window with no declared end (CoolingUntil <= 0) does not exist. While the unit is still
// cooling (now < CoolingUntil) the overlap grows exactly as fast as the raw wait, so the eligible
// wait — and with it the boost and the standing — holds constant: the pause.
func (c Candidate) cooledOverlap(now int64) int64 {
	if c.CoolingUntil <= 0 {
		return 0
	}
	start := c.CoolingSince
	if start < c.ReadySince {
		start = c.ReadySince
	}
	end := c.CoolingUntil
	if end > now {
		end = now
	}
	if end <= start {
		return 0
	}
	return end - start
}

// Ranked is one candidate with the aging verdict attached.
type Ranked struct {
	Candidate
	// WaitSeconds is how long the unit has been ready AND eligible — cooldown windows pause the
	// clock (the evidence the standing rests on).
	WaitSeconds int64 `json:"wait_seconds"`
	// AgingBoost is the soft boost added to the base weight (>= 0, capped at MaxBoostPoints).
	AgingBoost int `json:"aging_boost"`
	// EffectiveWeight is BaseWeight + AgingBoost — the weight the non-starved order sorts on.
	EffectiveWeight int `json:"effective_weight"`
	// Standing is the closed anti-starvation verdict (fresh | aging | starved).
	Standing Standing `json:"standing"`
	// Rank is the 0-based dispatch position over ALL candidates (starved first, then by effective
	// weight). Rank 0 is the unit a worker should take this tick.
	Rank int `json:"rank"`
}

// Result is the full deterministic verdict: every ready candidate in dispatch order, each with its
// aging verdict, plus the pick list and a standing census for a one-line operator summary.
type Result struct {
	Schema string `json:"schema"`
	// Order is every candidate in dispatch order (Rank 0 first): starved units ahead of the rest,
	// then by descending effective weight.
	Order []Ranked `json:"order"`
	// PickOrder is Order's IDs, in dispatch order — the sequence a drain should consume.
	PickOrder []string `json:"pick_order"`
	// Standing census, so a summary needs no fold.
	StarvedCount int `json:"starved_count"`
	AgingCount   int `json:"aging_count"`
	FreshCount   int `json:"fresh_count"`
	// OldestWaitSeconds is the longest any ready unit has waited (the worst starvation pressure).
	OldestWaitSeconds int64 `json:"oldest_wait_seconds"`
}

// Pick is the single unit a worker should take this tick — PickOrder[0], or "" when there is
// nothing ready.
func (r Result) Pick() string {
	if len(r.PickOrder) == 0 {
		return ""
	}
	return r.PickOrder[0]
}

// boostFor computes the soft aging boost for a wait, honoring the interval/per-interval/cap knobs.
// Disabled soft aging yields 0. The result is clamped so a corrupt or absurd wait cannot overflow.
func (p Params) boostFor(wait int64) int {
	if !p.softAgingOn() || wait < p.IntervalSeconds {
		return 0
	}
	intervals := wait / p.IntervalSeconds
	boost := intervals * int64(p.BoostPerInterval)
	if boost > boostClamp {
		boost = boostClamp
	}
	b := int(boost)
	if p.MaxBoostPoints > 0 && b > p.MaxBoostPoints {
		b = p.MaxBoostPoints
	}
	return b
}

// Fold is THE deterministic anti-starvation decision: same candidates + same clock in, same Result
// out — no clock read, no I/O. Total over any input.
//
// Invariant: dispatch aging calculations are fail-closed and monotonic: unknown ready timestamps never accrue boost.
// Guard: cooldown intervals pause aging accrual without resetting previously elapsed wait duration.
//
// The policy, per candidate:
//  1. wait = Now - ReadySince, minus the slice of that span inside the declared cooldown window
//     (clamped >= 0; 0 when ReadySince is unknown). A cooling unit's wait is PAUSED: it cannot
//     grow — and cannot starve — on ineligible time alone.
//  2. boost = soft aging boost for the wait (0 when soft aging is disabled or wait < one interval).
//  3. effective = BaseWeight + boost.
//  4. standing = Starved if the hard deadline is on and wait >= StarvationSeconds; else Aging when
//     boost > 0; else Fresh.
//
// The order is: all Starved units first (worst-starved — longest wait — first), then the rest by
// descending effective weight. Ties, in both bands, break by longer wait, then higher base weight,
// then ID ascending — a total, deterministic order. With aging disabled (zero-value Params) no unit
// is boosted or starved, so the order is base-weight-desc then wait then ID: the pre-aging order.
func Fold(cands []Candidate, p Params) Result {
	ranked := make([]Ranked, 0, len(cands))
	out := Result{Schema: Schema}
	for _, c := range cands {
		wait := c.waitSeconds(p.NowUnix)
		boost := p.boostFor(wait)
		r := Ranked{
			Candidate:       c,
			WaitSeconds:     wait,
			AgingBoost:      boost,
			EffectiveWeight: c.BaseWeight + boost,
			Rank:            -1,
		}
		switch {
		case p.hardDeadlineOn() && wait >= p.StarvationSeconds:
			r.Standing = StandingStarved
		case boost > 0:
			r.Standing = StandingAging
		default:
			r.Standing = StandingFresh
		}
		ranked = append(ranked, r)
		if wait > out.OldestWaitSeconds {
			out.OldestWaitSeconds = wait
		}
	}

	sort.SliceStable(ranked, func(i, j int) bool { return dispatchBefore(ranked[i], ranked[j]) })

	out.Order = ranked
	out.PickOrder = make([]string, 0, len(ranked))
	for i := range out.Order {
		out.Order[i].Rank = i
		out.PickOrder = append(out.PickOrder, out.Order[i].ID)
		switch out.Order[i].Standing {
		case StandingStarved:
			out.StarvedCount++
		case StandingAging:
			out.AgingCount++
		default:
			out.FreshCount++
		}
	}
	return out
}

// dispatchBefore is the total dispatch order over two ranked units: starved ahead of non-starved;
// within each band, higher effective weight first, then longer wait, then higher base weight, then
// smaller ID. Starved units compare on wait first (serve the worst-starved first) — their effective
// weight is reported for transparency but never gates their promotion.
func dispatchBefore(a, b Ranked) bool {
	sa, sb := a.Standing == StandingStarved, b.Standing == StandingStarved
	if sa != sb {
		return sa // starved units sort ahead of everything else
	}
	if !sa { // non-starved band: effective weight leads
		if a.EffectiveWeight != b.EffectiveWeight {
			return a.EffectiveWeight > b.EffectiveWeight
		}
	}
	if a.WaitSeconds != b.WaitSeconds {
		return a.WaitSeconds > b.WaitSeconds // longer wait first (oldest-first)
	}
	if a.BaseWeight != b.BaseWeight {
		return a.BaseWeight > b.BaseWeight
	}
	return a.ID < b.ID
}
