package model

import (
	"errors"
	"fmt"
	"math"
)

// placement_delta.go — typed PlacementDelta + closed-loop rebalance cadence for hot
// expert-placement change (CONTINUAL-ENGINE design §5, phase 3; ticket
// continual-engine-placement-delta-cadence-20261104).
//
// WHY THIS EXISTS. expert_balance.go computes a load-balanced expert→rank banding ONCE
// from a load snapshot and "never REBALANCES" (expert_balance.go:17): the placement is
// stale until an incident. The CONTINUAL-ENGINE design (§5) turns that one-shot into a
// closed loop — measured load → cost-model recompute → PlacementDelta (proposed) →
// cadence gate → apply via ActiveRankMask-style membership → receipt — where the unit
// of change is a typed delta applied against a live placement, never a restart, and
// the engine never churns placement every step: the first rebalance fires at ~1/4 of
// the horizon, then on a steady cadence, and never while a rebalance is in flight.
//
// WHAT LANDS HERE (the pure, CPU-witnessable core of that loop):
//
//   - PlacementDelta: the typed placement change, closed kind set, generation-stamped
//     at apply time;
//   - RebalanceCadence: the fire-gate policy (first-fire fraction, steady interval,
//     in-flight guard), driven by explicit logical tick steps — no wall clock;
//   - PlacementStore: the live placement state (mask + plan + placement generation)
//     that deltas apply against, whole-old/whole-new per apply, CAS-guarded on the
//     delta's From;
//   - PlacementDeltaSource: the dry-run-capable applier that turns a
//     LoadBalancedExpertBands recommendation into a gated, applied, receipted delta.
//
// BUILT ON, NOT REBUILT. LoadBalancedExpertBands / RankLoads / Balancedness
// (expert_balance.go) stay PURE policy functions — this file CALLS them and adds no
// step/decision machinery to that file. ActiveRankMask (active_rank_membership.go) is
// the From/To membership snapshot type; applying a delta constructs the NEXT mask via
// NewActiveRankMask and bumps the placement generation exactly once. TPPlan
// (tensor_parallel.go) is the plan type a re-rank delta carries.
//
// PHASE-0/1 EXTENSION POINT. ForwardSlot and WeightDelta land in sibling phases on
// other branches; they are ABSENT from this base. Per the EngineDelta grammar (design
// §2 invariant 3) each change class owns an INDEPENDENT generation counter — this file
// owns placement-gen only and coins no forward-gen or weight-epoch field, so those
// classes plug in as siblings without touching this type surface.
//
// NO WALL CLOCK. Every decision here is a function of explicit logical tick steps
// (step, horizon) supplied by the caller — deterministic, testable without sleeps.

// ErrPlacementDeltaKind is the closed-set sentinel: a PlacementDelta whose Kind is not
// one of PlacementReRank / PlacementPromote / PlacementDemote is refused, never
// coerced.
var ErrPlacementDeltaKind = errors.New("model: unsupported placement-delta kind")

// ErrPlacementDelta is the general refusal sentinel for a placement delta or its
// application: malformed payload, a membership transition inconsistent with the kind,
// a CAS base mismatch (the delta's From is not the live mask), or an apply/abort with
// nothing pending. A delta refused here did not happen — the live placement is
// untouched.
var ErrPlacementDelta = errors.New("model: placement delta refused")

// ErrRebalanceDenied is returned when the cadence gate refuses a rebalance at a given
// tick; the closed-vocabulary reason is extracted with DenialReason.
var ErrRebalanceDenied = errors.New("model: rebalance denied by cadence")

// ErrRebalanceState is returned on an illegal cadence/store state transition: Begin
// while already in flight, Complete/Abort with nothing in flight, or Propose while a
// delta is still pending.
var ErrRebalanceState = errors.New("model: rebalance state transition refused")

// PlacementDeltaKind is the CLOSED set of placement change classes. Adding a kind is a
// deliberate API change: every kind names what the delta moves and how Validate checks
// it, and an unknown kind fails closed with ErrPlacementDeltaKind.
type PlacementDeltaKind int

const (
	// PlacementReRank re-partitions the expert→rank bands under UNCHANGED membership:
	// which rank OWNS which experts changes (a new TPPlan); who is alive does not.
	PlacementReRank PlacementDeltaKind = iota
	// PlacementPromote moves ranks from dead to alive (To ⊋ From): recovered ranks
	// rejoin the EP group.
	PlacementPromote
	// PlacementDemote moves ranks from alive to dead (To ⊂ From): failed ranks leave
	// the EP group; their traffic routes around per the mask contract.
	PlacementDemote
)

// String renders the kind for receipts and logs.
func (k PlacementDeltaKind) String() string {
	switch k {
	case PlacementReRank:
		return "re-rank"
	case PlacementPromote:
		return "promote"
	case PlacementDemote:
		return "demote"
	default:
		return fmt.Sprintf("unknown(%d)", int(k))
	}
}

// PlacementDelta is a typed placement change applied against a live placement — never
// raw struct surgery on the engine. It names the membership before (From) and after
// (To) as ActiveRankMask snapshots, the expert→rank banding it installs (Plan, for
// re-ranks), and why (Reason). Generation and AppliedAt are STAMPED by the store at
// apply time (a proposed delta carries zero values); a delta missing both stamps has
// not been applied.
type PlacementDelta struct {
	Kind       PlacementDeltaKind
	From       ActiveRankMask
	To         ActiveRankMask
	Plan       TPPlan // re-rank payload: the new expert→rank banding; zero for membership moves
	Reason     string
	AppliedAt  int    // logical step at which the store applied it; stamped at apply
	Generation uint64 // placement-generation stamp assigned by the store; stamped at apply
}

// Ranks derives the affected ranks from the delta itself (never stored, so it cannot
// disagree with From/To/Plan): promote lists the ranks that come alive, demote lists
// the ranks that die, and re-rank lists every rank whose banding the new partition
// re-declares (all of them — the partition is global). Ascending, deduped.
func (d PlacementDelta) Ranks() []int {
	switch d.Kind {
	case PlacementPromote:
		out := make([]int, 0, d.To.ActiveCount()-d.From.ActiveCount())
		for r := 0; r < d.To.WorldSize(); r++ {
			if !d.From.Active(r) && d.To.Active(r) {
				out = append(out, r)
			}
		}
		return out
	case PlacementDemote:
		out := make([]int, 0, d.From.ActiveCount()-d.To.ActiveCount())
		for r := 0; r < d.From.WorldSize(); r++ {
			if d.From.Active(r) && !d.To.Active(r) {
				out = append(out, r)
			}
		}
		return out
	case PlacementReRank:
		out := make([]int, len(d.Plan.Shards))
		for i, s := range d.Plan.Shards {
			out[i] = s.Rank
		}
		return out
	default:
		return nil
	}
}

// Validate fails closed on any delta that could not be a real placement change: an
// unknown kind (ErrPlacementDeltaKind), unusable or world-mismatched From/To, a
// membership transition inconsistent with the kind, or a re-rank carrying a malformed
// plan or one whose shard count disagrees with the world size.
func (d PlacementDelta) Validate() error {
	if d.From.WorldSize() <= 0 || d.From.WorldSize() != d.To.WorldSize() {
		return fmt.Errorf("%w: From world %d, To world %d (want equal, > 0)", ErrPlacementDelta, d.From.WorldSize(), d.To.WorldSize())
	}
	if d.From.ActiveCount() == 0 || d.To.ActiveCount() == 0 {
		return fmt.Errorf("%w: empty membership snapshot", ErrPlacementDelta)
	}
	switch d.Kind {
	case PlacementReRank:
		if err := d.Plan.Validate(); err != nil {
			return fmt.Errorf("%w: re-rank plan: %v", ErrPlacementDelta, err)
		}
		if len(d.Plan.Shards) != d.From.WorldSize() {
			return fmt.Errorf("%w: re-rank plan has %d shards, want world size %d", ErrPlacementDelta, len(d.Plan.Shards), d.From.WorldSize())
		}
		if !sameMembership(d.From, d.To) {
			return fmt.Errorf("%w: re-rank must leave membership unchanged (live=%v target=%v)", ErrPlacementDelta, d.From.ActiveRanks(), d.To.ActiveRanks())
		}
	case PlacementPromote:
		if !isStrictSuperset(d.From, d.To) {
			return fmt.Errorf("%w: promote requires To a strict superset of From (live=%v target=%v)", ErrPlacementDelta, d.From.ActiveRanks(), d.To.ActiveRanks())
		}
	case PlacementDemote:
		if !isStrictSubset(d.From, d.To) {
			return fmt.Errorf("%w: demote requires To a strict subset of From (live=%v target=%v)", ErrPlacementDelta, d.From.ActiveRanks(), d.To.ActiveRanks())
		}
	default:
		return fmt.Errorf("%w: %v", ErrPlacementDeltaKind, d.Kind)
	}
	return nil
}

// NewPlacementDelta validates and returns a delta ready to propose. It is the
// sanctioned constructor: hand-built literals skip the closed-kind check and are
// refused again by Validate (which Propose runs).
func NewPlacementDelta(kind PlacementDeltaKind, from, to ActiveRankMask, plan TPPlan, reason string) (PlacementDelta, error) {
	d := PlacementDelta{Kind: kind, From: from, To: to, Plan: plan, Reason: reason}
	if err := d.Validate(); err != nil {
		return PlacementDelta{}, err
	}
	return d, nil
}

// sameMembership reports whether two masks span the same world with identical active
// sets.
func sameMembership(a, b ActiveRankMask) bool {
	if a.WorldSize() != b.WorldSize() {
		return false
	}
	for r := 0; r < a.WorldSize(); r++ {
		if a.Active(r) != b.Active(r) {
			return false
		}
	}
	return true
}

// isStrictSuperset reports sub ⊊ super: every live rank of sub is live in super, and
// super has at least one more.
func isStrictSuperset(sub, super ActiveRankMask) bool {
	if sub.WorldSize() != super.WorldSize() || sub.ActiveCount() >= super.ActiveCount() {
		return false
	}
	for r := 0; r < sub.WorldSize(); r++ {
		if sub.Active(r) && !super.Active(r) {
			return false
		}
	}
	return true
}

// isStrictSubset reports sub ⊊ super.
func isStrictSubset(sub, super ActiveRankMask) bool {
	return isStrictSuperset(super, sub)
}

// plansEqual reports whether two plans tile the same dimension with identical shards.
func plansEqual(a, b TPPlan) bool {
	if a.Dim != b.Dim || len(a.Shards) != len(b.Shards) {
		return false
	}
	for i := range a.Shards {
		if a.Shards[i] != b.Shards[i] {
			return false
		}
	}
	return true
}

// balancedOrNil scores a plan's balancedness against a load vector, returning nil when
// the metric is undefined (no routed traffic, or no load supplied) — never a
// fabricated 1.0.
func balancedOrNil(load []int, p TPPlan) *float64 {
	b, err := Balancedness(load, p)
	if err != nil {
		return nil
	}
	return &b
}

// CadenceDenyReason is the CLOSED set of reasons the rebalance gate can deny a fire;
// CadenceOK means allowed.
type CadenceDenyReason int

const (
	CadenceOK       CadenceDenyReason = iota // allow
	CadenceTooEarly                          // before the first-fire threshold (~horizon/4)
	CadenceTooSoon                           // inside the steady interval of the last fire
	CadenceInFlight                          // a rebalance is proposed but unapplied
)

// String renders the reason for receipts and logs.
func (r CadenceDenyReason) String() string {
	switch r {
	case CadenceOK:
		return "ok"
	case CadenceTooEarly:
		return "too-early"
	case CadenceTooSoon:
		return "too-soon"
	case CadenceInFlight:
		return "in-flight"
	default:
		return fmt.Sprintf("unknown(%d)", int(r))
	}
}

// RebalanceCadence is the fire-gate policy for placement churn: the FIRST rebalance of
// a run may not fire before FirstFireFraction of the horizon (default 1/4), subsequent
// fires are spaced by MinIntervalSteps (0 = the first-fire threshold, i.e. a steady
// quarter-period cadence), and no fire is ever allowed while a rebalance is in flight.
// All decisions are pure functions of the caller-supplied logical tick (step, horizon)
// — there is no wall clock anywhere in this path, so tests drive ticks explicitly.
//
// State transitions are paired: Begin marks in flight (a delta was proposed),
// Complete(step) records the fire and clears in flight (it was applied), Abort clears
// in flight WITHOUT recording a fire (it was discarded). An unpaired transition fails
// closed with ErrRebalanceState. Do not copy a cadence after use.
type RebalanceCadence struct {
	FirstFireFraction float64
	MinIntervalSteps  int

	inFlight     bool
	lastFireStep int // -1 until the first fire
}

// DefaultRebalanceCadence is the shipped default: first fire at a quarter of the
// horizon, steady cadence at the same quarter period.
func DefaultRebalanceCadence() *RebalanceCadence {
	return &RebalanceCadence{FirstFireFraction: 0.25, MinIntervalSteps: 0, lastFireStep: -1}
}

// NewRebalanceCadence validates the knobs: fraction in (0,1], interval >= 0.
func NewRebalanceCadence(firstFireFraction float64, minIntervalSteps int) (*RebalanceCadence, error) {
	if !(firstFireFraction > 0 && firstFireFraction <= 1) {
		return nil, fmt.Errorf("model: RebalanceCadence FirstFireFraction = %g, want (0,1]", firstFireFraction)
	}
	if minIntervalSteps < 0 {
		return nil, fmt.Errorf("model: RebalanceCadence MinIntervalSteps = %d, want >= 0", minIntervalSteps)
	}
	return &RebalanceCadence{FirstFireFraction: firstFireFraction, MinIntervalSteps: minIntervalSteps, lastFireStep: -1}, nil
}

// FirstFireThreshold is the step at which the first fire becomes legal: ceil(horizon ×
// fraction), snapping to the exact integer when the product is integral within float
// noise (so 0.25 × 100 is exactly 25, never 26). A malformed horizon yields 0, which
// Allow treats as "not yet".
func (c *RebalanceCadence) FirstFireThreshold(horizon int) int {
	if horizon <= 0 {
		return 0
	}
	x := float64(horizon) * c.FirstFireFraction
	r := math.Round(x)
	if math.Abs(x-r) < 1e-9 {
		return int(r)
	}
	return int(math.Ceil(x))
}

// SteadyInterval is the minimum spacing between consecutive fires: MinIntervalSteps if
// set, else the first-fire threshold (the quarter-period default).
func (c *RebalanceCadence) SteadyInterval(horizon int) int {
	if c.MinIntervalSteps > 0 {
		return c.MinIntervalSteps
	}
	return c.FirstFireThreshold(horizon)
}

// Allow reports whether a rebalance may fire at logical step `step` of a `horizon`-step
// window. It is a read-only query: it neither begins nor completes anything. Malformed
// ticks (negative step, non-positive horizon) fail closed as TooEarly — never allow on
// garbage.
func (c *RebalanceCadence) Allow(step, horizon int) (bool, CadenceDenyReason) {
	if c.inFlight {
		return false, CadenceInFlight
	}
	if step < 0 || horizon <= 0 {
		return false, CadenceTooEarly
	}
	if c.lastFireStep < 0 {
		if step < c.FirstFireThreshold(horizon) {
			return false, CadenceTooEarly
		}
		return true, CadenceOK
	}
	if step-c.lastFireStep < c.SteadyInterval(horizon) {
		return false, CadenceTooSoon
	}
	return true, CadenceOK
}

// Begin marks a rebalance in flight. Refused while one already is (ErrRebalanceState).
func (c *RebalanceCadence) Begin() error {
	if c.inFlight {
		return fmt.Errorf("%w: rebalance already in flight", ErrRebalanceState)
	}
	c.inFlight = true
	return nil
}

// Complete records that the in-flight rebalance fired at `step` and clears in flight.
// Refused when nothing is in flight or on a negative step.
func (c *RebalanceCadence) Complete(step int) error {
	if !c.inFlight {
		return fmt.Errorf("%w: no rebalance in flight to complete", ErrRebalanceState)
	}
	if step < 0 {
		return fmt.Errorf("%w: fire step %d, want >= 0", ErrRebalanceState, step)
	}
	c.inFlight = false
	c.lastFireStep = step
	return nil
}

// Abort clears in flight WITHOUT recording a fire: a discarded proposal consumes no
// cadence. Refused when nothing is in flight.
func (c *RebalanceCadence) Abort() error {
	if !c.inFlight {
		return fmt.Errorf("%w: no rebalance in flight to abort", ErrRebalanceState)
	}
	c.inFlight = false
	return nil
}

// cadenceDenial wraps ErrRebalanceDenied with the closed-vocabulary reason.
type cadenceDenial struct{ reason CadenceDenyReason }

func (e cadenceDenial) Error() string { return "model: rebalance denied: " + e.reason.String() }
func (e cadenceDenial) Unwrap() error { return ErrRebalanceDenied }

// DenialReason extracts the closed-vocabulary reason from a gate denial.
func DenialReason(err error) (CadenceDenyReason, bool) {
	var d cadenceDenial
	if errors.As(err, &d) {
		return d.reason, true
	}
	return CadenceOK, false
}

// PlacementStore is the live placement state a PlacementDelta applies against: the
// current membership mask, the current expert→rank banding, and the placement
// generation counter (EngineDelta grammar §2 invariant 3 — independent of forward-gen
// and weight-epoch). Applies are whole-old/whole-new: one atomic state swap per delta,
// CAS-guarded on the delta's From, and the generation bumps EXACTLY ONCE per applied
// delta. Single-owner: the rebalance loop drives it; it holds no lock because it takes
// no wall clock and does no I/O.
type PlacementStore struct {
	mask       ActiveRankMask
	plan       TPPlan
	generation uint64
	pending    *PlacementDelta
	last       *PlacementReceipt
}

// NewPlacementStore pins the initial placement: a usable mask (non-empty world with at
// least one live rank) and a valid plan.
func NewPlacementStore(mask ActiveRankMask, plan TPPlan) (*PlacementStore, error) {
	if mask.WorldSize() <= 0 || mask.ActiveCount() == 0 {
		return nil, fmt.Errorf("%w: initial mask unusable (world %d, active %d)", ErrPlacementDelta, mask.WorldSize(), mask.ActiveCount())
	}
	if err := plan.Validate(); err != nil {
		return nil, fmt.Errorf("%w: initial plan: %v", ErrPlacementDelta, err)
	}
	return &PlacementStore{mask: mask, plan: plan}, nil
}

// Mask returns the live membership snapshot. The mask is immutable by convention (no
// mutators), so the shared backing array is safe to read.
func (s *PlacementStore) Mask() ActiveRankMask { return s.mask }

// Plan returns the live expert→rank banding.
func (s *PlacementStore) Plan() TPPlan { return s.plan }

// Generation is the placement-generation stamp of the live placement.
func (s *PlacementStore) Generation() uint64 { return s.generation }

// Pending returns the in-flight (proposed, unapplied) delta, or nil.
func (s *PlacementStore) Pending() *PlacementDelta { return s.pending }

// LastReceipt returns the receipt of the most recently applied delta, or nil.
func (s *PlacementStore) LastReceipt() *PlacementReceipt { return s.last }

// Propose records d as the in-flight delta. Refused while another delta is pending,
// when d fails Validate, or when d.From is not the live mask (the CAS base check at
// propose time — a late catch of a delta built against a stale snapshot).
func (s *PlacementStore) Propose(d PlacementDelta) error {
	if s.pending != nil {
		return fmt.Errorf("%w: delta already in flight", ErrRebalanceState)
	}
	if err := d.Validate(); err != nil {
		return err
	}
	if !sameMembership(d.From, s.mask) {
		return fmt.Errorf("%w: delta From is not the live mask (base drifted since the delta was built)", ErrPlacementDelta)
	}
	cp := d
	s.pending = &cp
	return nil
}

// Apply applies the pending delta at logical step `step`, swapping the mask (and the
// plan, for re-ranks) whole-old/whole-new, stamping the delta's Generation and
// AppliedAt, and returning the receipt. `load` is the measured load window the receipt
// scores balancedness against; pass nil to leave the metric undefined (membership
// moves are not scored on expert load). Refused when nothing is pending or the delta's
// From no longer matches the live mask (fail-closed CAS: a delta whose base is not the
// live snapshot is refused, not merged).
func (s *PlacementStore) Apply(step int, load []int) (PlacementReceipt, error) {
	if s.pending == nil {
		return PlacementReceipt{}, fmt.Errorf("%w: nothing pending to apply", ErrPlacementDelta)
	}
	d := *s.pending
	if !sameMembership(d.From, s.mask) {
		return PlacementReceipt{}, fmt.Errorf("%w: CAS mismatch — delta From is not the live mask", ErrPlacementDelta)
	}
	nextPlan := s.plan
	if d.Kind == PlacementReRank {
		nextPlan = d.Plan
	}
	before := balancedOrNil(load, s.plan)
	after := balancedOrNil(load, nextPlan)
	nextMask, err := NewActiveRankMask(s.mask.WorldSize(), d.To.ActiveRanks())
	if err != nil {
		return PlacementReceipt{}, fmt.Errorf("%w: reconstruct To mask: %v", ErrPlacementDelta, err)
	}
	s.generation++
	d.AppliedAt = step
	d.Generation = s.generation
	receipt := PlacementReceipt{Delta: d, Step: step, BalancedBefore: before, BalancedAfter: after}
	s.mask = nextMask
	s.plan = nextPlan
	s.last = &receipt
	s.pending = nil
	return receipt, nil
}

// Abort discards the pending delta without applying it. Refused when nothing is pending.
func (s *PlacementStore) Abort() error {
	if s.pending == nil {
		return fmt.Errorf("%w: nothing pending to abort", ErrRebalanceState)
	}
	s.pending = nil
	return nil
}

// PlacementReceipt is the witness of an applied PlacementDelta: the stamped delta, the
// step it landed at, and the balancedness (mean/max per-rank load) before and after,
// scored against the measured load window. A nil Balanced* means the metric was
// undefined on that side (no routed traffic, or no load supplied) — never a fabricated
// 1.0.
type PlacementReceipt struct {
	Delta          PlacementDelta
	Step           int
	BalancedBefore *float64
	BalancedAfter  *float64
}

// PlacementDeltaSource is the closed-loop driver: it gates proposals on the
// RebalanceCadence, turns a LoadBalancedExpertBands recommendation into a
// PlacementDelta, and applies/aborts it through the PlacementStore. Plan is dry-run:
// it proposes (marks in flight) but changes nothing else — no generation bump, no mask
// or plan swap — until Apply commits.
type PlacementDeltaSource struct {
	store   *PlacementStore
	cadence *RebalanceCadence
}

// NewPlacementDeltaSource wires a store and its cadence gate into one driver.
func NewPlacementDeltaSource(store *PlacementStore, cadence *RebalanceCadence) *PlacementDeltaSource {
	return &PlacementDeltaSource{store: store, cadence: cadence}
}

// Plan proposes a load-aware re-rank at logical step `step` of a `horizon`-step
// window: gate on the cadence, recompute the load-balanced banding over the store's
// world, and — only if it differs from the live plan — propose a PlacementReRank delta
// and mark the cadence in flight. It returns changed=false with the zero delta when
// the live plan is already load-optimal: a no-op recommendation proposes nothing,
// marks nothing, and consumes no cadence.
func (s *PlacementDeltaSource) Plan(step, horizon int, load []int) (PlacementDelta, bool, error) {
	if ok, reason := s.cadence.Allow(step, horizon); !ok {
		return PlacementDelta{}, false, cadenceDenial{reason: reason}
	}
	world := s.store.Mask().WorldSize()
	p, err := LoadBalancedExpertBands(load, world)
	if err != nil {
		return PlacementDelta{}, false, err
	}
	if plansEqual(p, s.store.Plan()) {
		return PlacementDelta{}, false, nil
	}
	m := s.store.Mask()
	d, err := NewPlacementDelta(PlacementReRank, m, m, p, fmt.Sprintf("load-aware re-rank at step %d of %d", step, horizon))
	if err != nil {
		return PlacementDelta{}, false, err
	}
	if err := s.store.Propose(d); err != nil {
		return PlacementDelta{}, false, err
	}
	if err := s.cadence.Begin(); err != nil {
		s.store.Abort()
		return PlacementDelta{}, false, err
	}
	return d, true, nil
}

// PlanMembership proposes a promote/demote at the gated tick: the target membership
// must be a strict superset (promote) or strict subset (demote) of the live mask. A
// mixed change (some ranks in AND some out) is refused — one delta is one typed
// change, and the caller sequences two across cadence windows. A target equal to the
// live mask is a no-op (changed=false, nothing proposed).
func (s *PlacementDeltaSource) PlanMembership(step, horizon int, alive []int) (PlacementDelta, bool, error) {
	if ok, reason := s.cadence.Allow(step, horizon); !ok {
		return PlacementDelta{}, false, cadenceDenial{reason: reason}
	}
	cur := s.store.Mask()
	target, err := NewActiveRankMask(cur.WorldSize(), alive)
	if err != nil {
		return PlacementDelta{}, false, err
	}
	if sameMembership(cur, target) {
		return PlacementDelta{}, false, nil
	}
	var kind PlacementDeltaKind
	switch {
	case isStrictSuperset(cur, target):
		kind = PlacementPromote
	case isStrictSubset(cur, target):
		kind = PlacementDemote
	default:
		return PlacementDelta{}, false, fmt.Errorf("%w: membership change mixes promotions and demotions (live=%v target=%v); sequence two deltas", ErrPlacementDelta, cur.ActiveRanks(), target.ActiveRanks())
	}
	d, err := NewPlacementDelta(kind, cur, target, TPPlan{}, fmt.Sprintf("%s at step %d of %d", kind, step, horizon))
	if err != nil {
		return PlacementDelta{}, false, err
	}
	if err := s.store.Propose(d); err != nil {
		return PlacementDelta{}, false, err
	}
	if err := s.cadence.Begin(); err != nil {
		s.store.Abort()
		return PlacementDelta{}, false, err
	}
	return d, true, nil
}

// Apply commits the pending delta at logical step `step`: the store swaps the placement
// whole-old/whole-new and stamps the generation exactly once; the cadence then records
// the fire. `load` is the measured window the receipt scores (nil leaves the metric
// undefined).
func (s *PlacementDeltaSource) Apply(step int, load []int) (PlacementReceipt, error) {
	r, err := s.store.Apply(step, load)
	if err != nil {
		return r, err
	}
	if err := s.cadence.Complete(step); err != nil {
		return r, err
	}
	return r, nil
}

// Abort discards the pending delta and releases the in-flight mark without consuming
// cadence.
func (s *PlacementDeltaSource) Abort() error {
	if err := s.store.Abort(); err != nil {
		return err
	}
	return s.cadence.Abort()
}
