package dispatchtick

// admission_edf.go -- earliest-deadline-first (EDF) dispatch ordering plus
// predicted-miss load-shedding (#5292, parent epic #5289, composes with the
// disaggregation admission-kernel epic #3259).
//
// WHY. The package already knows two ways to pick and to refuse work, and
// neither reasons about deadlines:
//
//   - priority.go orders candidates by PRIORITY WEIGHT (heaviest label first,
//     optional aging) -- a static rank, blind to when an item must finish.
//   - footprint_ceiling.go sheds via a MEASURED resource ceiling
//     (RefuseHostFootprint) -- reactive backpressure that fires only once the
//     host is already at its limit.
//
// What is missing is the predict-then-shed axis: order pending work by the
// deadline it must meet, simulate the queue against a service-time estimate,
// and shed -- up front, with a typed reason -- the items that are ALREADY
// doomed to miss, so the survivors keep their deadline instead of everyone
// missing together. Clean-room borrow of the policy shape of Mooncake's tent
// admission queue (Apache-2.0; EDF pick order + predicted-miss drop of
// degradation-eligible requests). Policy only: plain ints, injected clock and
// estimator, no I/O, no internal imports -- deterministic and CPU-witnessable.
//
// SEMANTICS. Time is an abstract monotonic int64 tick (the caller picks the
// unit; millis in the shell, small ints in tests). An item must FINISH by its
// deadline tick (finish <= deadline meets it; finish > deadline misses). The
// planner walks the EDF order simulating one serial service lane: each
// dispatched item occupies the lane for its estimated cost. A predicted miss
// is handled by the item's Degradable bit:
//
//   - Degradable: SHED with the typed ShedPredictedDeadlineMiss reason and the
//     predicted start/finish evidence. The lane clock does NOT advance -- the
//     freed time is what rescues the items behind it.
//   - Not degradable: NEVER silently dropped. It stays in the dispatch order,
//     flagged PredictedLate, and honestly occupies the lane (which may doom a
//     degradable item behind it -- that cascade is the point, not a bug).
//
// Fail-closed direction: a shed is always explicit (typed reason + numbers a
// reader can re-derive), never a silent disappearance; an unpriceable cost
// clamps to zero so an estimator bug can only FAIL TO SHED (keeping today's
// admit-everything behavior), never invent a miss and drop real work.
//
// NOT_YET: this file is the pure policy plus its closure tests. Wiring it into
// the live tick loop as an opt-in ordering/shedding mode (the way aging_order.go
// opts into OrderLaneCandidates) is follow-on work on #5292.

import (
	"fmt"
	"sort"
)

// ShedPredictedDeadlineMiss is the typed reason a shed decision carries when the
// planner predicts the item cannot finish by its deadline given the queue ahead
// of it. The string is the WIRE token; renaming the Go symbol must never change
// the wire string (pinned by test), matching the RefuseHostFootprint convention.
const ShedPredictedDeadlineMiss = "SHED_PREDICTED_DEADLINE_MISS"

// DeadlineItem is one pending unit of work the EDF planner orders: an opaque ID,
// the absolute tick it must FINISH by, the caller's service-cost hint in the same
// tick unit, and whether the item may be degraded (shed) when it is predicted to
// miss. Plain values only -- the planner never looks anywhere else.
type DeadlineItem struct {
	ID string
	// DeadlineTick is the absolute tick the item must finish by (inclusive).
	DeadlineTick int64
	// CostHintTicks is the caller's predicted service time. An injected
	// estimator (EDFPlanner.EstimateCostTicks) overrides it; negative values
	// clamp to zero (unknown cost must never invent a predicted miss).
	CostHintTicks int64
	// Degradable marks the item as eligible for predicted-miss shedding. A
	// non-degradable item is never dropped by this planner -- only flagged.
	Degradable bool
}

// DispatchDecision is one admitted slot in the EDF order, with the simulated
// evidence behind it: when the item is predicted to start and finish on the
// serial lane, and whether it is predicted to finish late (possible only for
// non-degradable items, which are retained rather than shed).
type DispatchDecision struct {
	Item                DeadlineItem
	PredictedStartTick  int64
	PredictedFinishTick int64
	// PredictedLate reports a non-degradable item predicted to miss its
	// deadline: retained (never silently dropped), but flagged so the caller
	// can surface the debt instead of discovering it at the deadline.
	PredictedLate bool
}

// ShedDecision is one item shed up front because the planner predicts it will
// miss its deadline and it is degradable. Reason is always a typed wire token
// (ShedPredictedDeadlineMiss today); the predicted ticks are the evidence a
// reader needs to re-derive the verdict instead of trusting it.
type ShedDecision struct {
	Item                DeadlineItem
	Reason              string
	PredictedStartTick  int64
	PredictedFinishTick int64
}

// EDFPlan is the planner's full, deterministic answer: the dispatch order
// (earliest deadline first) and the set shed up front, each with its predicted
// timeline. Every input item appears in exactly one of the two lists.
type EDFPlan struct {
	Order []DispatchDecision
	Shed  []ShedDecision
}

// ShedIDs materializes just the shed item IDs, in shed order, for terse logging
// and assertions.
func (p EDFPlan) ShedIDs() []string {
	ids := make([]string, len(p.Shed))
	for i, s := range p.Shed {
		ids[i] = s.Item.ID
	}
	return ids
}

// OrderIDs materializes just the admitted item IDs, in dispatch order.
func (p EDFPlan) OrderIDs() []string {
	ids := make([]string, len(p.Order))
	for i, d := range p.Order {
		ids[i] = d.Item.ID
	}
	return ids
}

// EDFPlanner holds the two injected dependencies that keep the plan
// deterministic and testable: the clock and the service-time estimator. The
// zero value is usable -- tick 0 and cost hints -- and deliberately never falls
// back to a wall clock (that would smuggle nondeterminism into a pure fold).
type EDFPlanner struct {
	// NowTick returns the current tick. Nil means tick 0, which keeps a
	// relative-deadline caller and every test deterministic by construction.
	NowTick func() int64
	// EstimateCostTicks predicts one item's service time in ticks. Nil falls
	// back to the item's CostHintTicks. Negative results clamp to zero: an
	// unpriceable item can only fail to shed, never be invented into a miss.
	EstimateCostTicks func(DeadlineItem) int64
}

// now resolves the injected clock, defaulting to tick 0 (deterministic; never
// the wall clock).
func (p EDFPlanner) now() int64 {
	if p.NowTick == nil {
		return 0
	}
	return p.NowTick()
}

// cost resolves one item's estimated service ticks: the injected estimator when
// wired, else the item's own hint, clamped at zero so unknown cost fails open
// on shedding (admit) rather than dropping work on a guess.
func (p EDFPlanner) cost(item DeadlineItem) int64 {
	c := item.CostHintTicks
	if p.EstimateCostTicks != nil {
		c = p.EstimateCostTicks(item)
	}
	if c < 0 {
		return 0
	}
	return c
}

// Plan orders the pending items earliest-deadline-first and sheds, up front,
// every degradable item predicted to miss its deadline given the queue ahead of
// it. The input slice is never mutated. Determinism: the EDF sort is stable
// with an explicit (DeadlineTick, ID, input-order) tiebreak, the clock and
// estimator are injected, and the serial-lane simulation is a pure fold -- the
// same inputs always produce byte-for-byte the same plan.
func (p EDFPlanner) Plan(items []DeadlineItem) EDFPlan {
	ordered := append([]DeadlineItem(nil), items...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].DeadlineTick != ordered[j].DeadlineTick {
			return ordered[i].DeadlineTick < ordered[j].DeadlineTick
		}
		return ordered[i].ID < ordered[j].ID
	})

	plan := EDFPlan{
		Order: make([]DispatchDecision, 0, len(ordered)),
		Shed:  make([]ShedDecision, 0),
	}
	lane := p.now() // when the serial service lane next frees up
	for _, item := range ordered {
		start := lane
		finish := start + p.cost(item)
		miss := finish > item.DeadlineTick
		if miss && item.Degradable {
			// Predicted miss on a degradable item: shed it now, with the
			// evidence. The lane does not advance -- the freed time is what
			// keeps the items behind it meetable.
			plan.Shed = append(plan.Shed, ShedDecision{
				Item:                item,
				Reason:              ShedPredictedDeadlineMiss,
				PredictedStartTick:  start,
				PredictedFinishTick: finish,
			})
			continue
		}
		plan.Order = append(plan.Order, DispatchDecision{
			Item:                item,
			PredictedStartTick:  start,
			PredictedFinishTick: finish,
			PredictedLate:       miss,
		})
		lane = finish
	}
	return plan
}

// String renders one shed decision as a single evidence-bearing line: the wire
// reason plus the numbers (predicted finish vs deadline) a reader needs to
// re-derive the verdict.
func (s ShedDecision) String() string {
	return fmt.Sprintf("%s id=%s predicted_finish=%d deadline=%d predicted_start=%d",
		s.Reason, s.Item.ID, s.PredictedFinishTick, s.Item.DeadlineTick, s.PredictedStartTick)
}
