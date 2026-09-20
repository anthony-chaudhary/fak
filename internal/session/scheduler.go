package session

// scheduler.go — the first CONSUMER of Table.Snapshot (#627). The table HOLDS the
// drive state and exposes Snapshot already sorted into scheduler-consumption order
// (Priority ascending, ties by Rev descending, then TraceID); it deliberately never
// PICKS a winner. This file is the separate type that does: a Scheduler READS the
// snapshot each boundary and names the one live session that should run next, and
// CONSUMES budget-exhaustion / pause / stop as "a slot just freed" scheduling events
// (delivered through the table's existing WatchBudget / WatchTransitions observer
// seams) instead of re-deriving them from a process scan.
//
// POSTURE. The Scheduler is policy; the Table is a value. The scheduler only READS
// Snapshot and SUBSCRIBES to events — it adds nothing to the table and holds no lock
// the table knows about, so the table stays a passive, policy-free data structure.
// Because Pick reads the LIVE snapshot every call, an operator who cuts one session's
// budget and raises its Priority (SetBudget + SetPriority) flips which session Pick
// returns on the very next boundary — the priority-queue "let an urgent one pass"
// move falls straight out of reading fresh state, no scheduler bookkeeping required.
//
// Foundation-only: stdlib (sync/time), off the request path, deterministic (no hidden
// clock, no randomness) so every policy is unit-testable to an exact pick sequence.

import (
	"math"
	"sort"
	"sync"
	"time"
)

// DefaultReservationGrace is the reclaim window after a known-coming turn's predicted
// arrival. The reservation is advisory and lower-class than real work: if the matching
// request does not promote before this grace closes, the slot is reclaimed.
const DefaultReservationGrace = time.Second

// Policy selects how Pick breaks contention among the live, eligible sessions that
// share one gateway. The zero value is StrictPriority — the safe, trivially-correct
// default that simply honors the snapshot's existing sort.
type Policy uint8

const (
	// StrictPriority returns the FIRST eligible session in Snapshot order. Snapshot is
	// already sorted (Priority ascending, then Rev descending, then TraceID), so this
	// is deterministic and correct by construction: the lowest Priority value that is
	// eligible wins, and a budget cut / priority raise re-sorts the snapshot the next
	// time Pick reads it.
	StrictPriority Policy = iota
	// WeightedFair returns a deterministic weighted round-robin winner, giving a lower
	// Priority value a proportionally larger share of the picks while still letting
	// every eligible session make progress. See pickWeightedFairLocked for the exact
	// algorithm (smooth weighted round-robin — no clock, no randomness).
	WeightedFair
)

// String renders a Policy as its lowercase token; an out-of-range value renders
// "unknown" rather than panicking, matching the rest of the package's enums.
func (p Policy) String() string {
	switch p {
	case StrictPriority:
		return "strict-priority"
	case WeightedFair:
		return "weighted-fair"
	}
	return "unknown"
}

// SlotCause is the closed reason a scheduling slot freed — the "why" on a SlotEvent.
// It is a small total enum so a host reacts on a checkable token, never free text.
type SlotCause uint8

const (
	// CauseBudgetExhausted: a session drained a configured budget axis (observed via the
	// table's BudgetExhausted event on the WatchBudget seam).
	CauseBudgetExhausted SlotCause = iota
	// CausePaused: an operator paused the session (a hold, not an end) — its slot is
	// free while it waits.
	CausePaused
	// CauseDraining: a stop was requested; the session takes it at the next boundary.
	CauseDraining
	// CauseStopped: the session reached its terminal state.
	CauseStopped
)

// String renders a SlotCause as its lowercase token; an out-of-range value renders
// "unknown" rather than panicking.
func (c SlotCause) String() string {
	switch c {
	case CauseBudgetExhausted:
		return "budget-exhausted"
	case CausePaused:
		return "paused"
	case CauseDraining:
		return "draining"
	case CauseStopped:
		return "stopped"
	}
	return "unknown"
}

// SlotEvent is the immutable "a slot freed" signal the Scheduler emits to its host
// when a session leaves the eligible set (budget exhaustion, pause, drain, or stop).
// It is the scheduling-EVENT framing the package design names: the supervisor learns
// a slot opened the instant it happens, rather than re-deriving liveness from a
// process scan. Rev is the table revision at the freeing write, so a host can order
// or de-duplicate events against a /v1/fak/changes cursor.
type SlotEvent struct {
	TraceID string    `json:"trace_id"`
	Cause   SlotCause `json:"cause"`
	Rev     uint64    `json:"rev"`
}

// SlotReservation is the scheduler's advisory hold for a known-coming turn (#811).
// It carries the exact prefix identity a host should keep resident and the expiry
// boundary after which the hold must be reclaimed. It is deliberately not a table
// write: reservations are scheduler-local policy, lower-class than real requests.
type SlotReservation struct {
	TraceID            string `json:"trace_id"`
	Prefix             string `json:"prefix"`
	SourceRev          uint64 `json:"source_rev,omitempty"`
	ReservedAtUnixNano int64  `json:"reserved_at_unix_nano"`
	ArrivesAtUnixNano  int64  `json:"arrives_at_unix_nano"`
	ExpiresAtUnixNano  int64  `json:"expires_at_unix_nano"`
}

// SlotPromotion is returned when a real request matches an active reservation. Promoted
// false is never returned; the bool return on PromoteReservation carries that bit. The
// struct exists so a host can record "warm prefix adopted" with the reservation facts.
type SlotPromotion struct {
	SlotReservation
	PromotedAtUnixNano int64 `json:"promoted_at_unix_nano"`
}

// AttachOptions carries the optional pass-through observers and warn fraction Attach
// installs alongside the scheduler's own handlers. WatchBudget / WatchTransitions each
// hold exactly ONE observer, so the scheduler composes: it installs a fan-out handler
// that first calls the host's pass-through (if any) and then interprets the event for
// its own slot-freed accounting. A zero AttachOptions means the scheduler takes sole
// ownership of both seams (no pass-through, warning disabled).
type AttachOptions struct {
	// WarnFraction is forwarded verbatim to Table.WatchBudget, so a host that wants the
	// #743 pre-exhaustion warning still receives it through its pass-through observer.
	// The scheduler itself ignores BudgetWarn — a warning does not free a slot; only
	// BudgetExhausted does. A value outside (0,1) disables the warning (the table's
	// documented behavior), leaving only the exhaustion event firing.
	WarnFraction float64
	// Budget, if non-nil, is the host's pass-through budget observer (e.g. an operator
	// webhook). It is invoked for EVERY BudgetEvent before the scheduler interprets the
	// event. nil means the scheduler owns the budget seam alone.
	Budget BudgetObserver
	// Transitions, if non-nil, is the host's pass-through transition observer, invoked
	// for every TransitionEvent before the scheduler maps it to a slot-freed cause.
	Transitions TransitionObserver
}

// Scheduler reads a *Table via Snapshot and picks the live session that should run
// next under a chosen Policy, and consumes budget-exhaustion / pause / stop as
// slot-freed scheduling events. It is the policy layer that keeps the table policy-
// free. The zero value is not usable — construct with NewScheduler. A nil *Scheduler
// is a sane no-op (Pick returns no winner; OnSlotFreed / Attach do nothing), so a
// host with no scheduler wired behaves like the pre-scheduler path.
type Scheduler struct {
	mu     sync.Mutex
	policy Policy
	table  *Table
	// credits is the WeightedFair virtual-time state: a per-TraceID running credit,
	// keyed by trace, rebuilt each Pick to hold only the currently-eligible set (so it
	// is bounded and a session that leaves contention forgets its stale credit).
	credits      map[string]int64
	reservations map[reservationKey]SlotReservation
	// intents is the reservation LEDGER (#13384): the first-observed deadline for every
	// (trace, prefix) intent, retained for the LIFETIME OF THE INTENT GENERATION rather
	// than the lifetime of the reservation. A reservation is pruned the moment its grace
	// closes; this ledger is what remembers the deadline AFTER the reservation is gone,
	// so a scan at now >= the original expiry can tell "this intent already expired"
	// apart from "this intent is new" and decline instead of re-basing a fresh hold.
	// Keyed exactly like a reservation, it is released whenever the trace leaves the
	// snapshot or its hint turns terminal, so it tracks live membership (R-5).
	intents map[reservationKey]reservationIntent
	// epochs remembers, per trace, the last intent identity the scheduler observed.
	// It is what distinguishes a fresh publication (a NEW identity -> a new intent
	// generation, so a fresh deadline is minted) from a repeated scan of the SAME hint
	// (the ledger tombstone governs). A terminal hint overwrites the epoch with the
	// terminal identity, so a clear-then-republish is a witnessed change. Keyed by
	// trace, one row per live trace, released when the trace leaves the snapshot (R-5).
	epochs map[string]reservationIntentIdentity
	onSlot func(SlotEvent)
}

// reservationIntent is one entry of the first-deadline ledger: the identity of the
// intent generation that minted a reservation and its FIRST-observed arrival/expiry
// stamps, which are never re-based. A later scan whose intent identity matches treats
// the ledger as a tombstone: before expiresAtUnixNano it returns the SAME stamps, and
// at/after it declines rather than minting a fresh deadline (R-1 / R-2).
type reservationIntent struct {
	identity           reservationIntentIdentity
	reservedAtUnixNano int64
	arrivesAtUnixNano  int64
	expiresAtUnixNano  int64
}

// reservationIntentIdentity is the scheduler-VISIBLE identity of a TurnIntent for
// reservation purposes. State.Rev is deliberately NOT part of it: Rev is a record
// version bumped by any write (SetPriority, SetPace, a byte-identical SetTurnIntent),
// so keying on it would renew the hold on an unrelated write (SPEC R-3). The identity
// is exactly the fields that change what reservation a hint asks for — the prefix and
// the forward arrival delay — so a prefix change is a NEW generation while a repeated
// scan of the same hint is the SAME one.
type reservationIntentIdentity struct {
	prefix           string
	arrivingInMillis int64
}

func intentIdentityOf(st State) reservationIntentIdentity {
	return reservationIntentIdentity{prefix: st.Intent.Prefix, arrivingInMillis: st.Intent.ArrivingInMillis}
}

// NewScheduler builds an unattached scheduler under the given Policy. Bind it to a
// table with Attach before calling Pick; an unattached scheduler's Pick returns no
// winner.
func NewScheduler(policy Policy) *Scheduler {
	return &Scheduler{policy: policy, credits: map[string]int64{}, reservations: map[reservationKey]SlotReservation{}, intents: map[reservationKey]reservationIntent{}, epochs: map[string]reservationIntentIdentity{}}
}

// Attach binds the scheduler to a table and installs the internal slot-freed handlers
// on the table's existing WatchBudget / WatchTransitions seams, composing with any
// pass-through observers in opts. After Attach, Pick reads this table's live Snapshot
// and slot-freed events flow to the OnSlotFreed callback. Calling Attach again re-binds
// (the last Attach wins — it overwrites the table's single observer slots, by design).
// A nil receiver or nil table is a no-op.
//
// The composed handlers are safe against deadlock: the table fires observers AFTER it
// releases its own lock, and the scheduler's handler only takes the scheduler's lock
// (a distinct lock the table never holds) before invoking the host callback.
func (s *Scheduler) Attach(t *Table, opts AttachOptions) {
	if s == nil || t == nil {
		return
	}
	s.mu.Lock()
	s.table = t
	s.mu.Unlock()

	passBudget := opts.Budget
	t.WatchBudget(opts.WarnFraction, func(ev BudgetEvent) {
		if passBudget != nil {
			passBudget(ev)
		}
		// A warning is early progress, not a freed slot; only exhaustion frees one.
		if ev.Kind == BudgetExhausted {
			s.emitSlot(SlotEvent{TraceID: ev.TraceID, Cause: CauseBudgetExhausted, Rev: ev.Rev})
		}
	})

	passTrans := opts.Transitions
	t.WatchTransitions(func(ev TransitionEvent) {
		if passTrans != nil {
			passTrans(ev)
		}
		// WatchTransitions only fires on notable moves (into Paused/Draining/Stopped), so
		// Running/Throttled never reach here; slotCauseFor guards the mapping defensively.
		if cause, ok := slotCauseFor(ev.To); ok {
			s.emitSlot(SlotEvent{TraceID: ev.TraceID, Cause: cause, Rev: ev.Rev})
		}
	})
}

// OnSlotFreed registers the host callback invoked once per slot-freed event. Passing
// nil clears it. Safe to call any time, including before Attach; a nil receiver is a
// no-op. The callback runs on the table's observer goroutine (after the table lock is
// released), so it may block on slow work without stalling the table — the host owns
// fan-out and failure policy, exactly like the table's own observer seams.
func (s *Scheduler) OnSlotFreed(fn func(SlotEvent)) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.onSlot = fn
	s.mu.Unlock()
}

// emitSlot delivers one SlotEvent to the host callback (if any). It copies the callback
// reference under the lock and invokes it AFTER releasing the lock, so a slow host
// callback never holds the scheduler's mutex (and a re-entrant Pick from the callback
// cannot self-deadlock).
func (s *Scheduler) emitSlot(ev SlotEvent) {
	s.mu.Lock()
	cb := s.onSlot
	s.mu.Unlock()
	if cb != nil {
		cb(ev)
	}
}

// Pick names the live session that should run next, reading the attached table's LIVE
// Snapshot so an operator's budget/priority change is reflected on the very next call.
// ok is false when there is no eligible session — an empty table, an all-ineligible
// table (everyone paused/draining/stopped or budget-exhausted), or an unattached / nil
// scheduler. On ok==false the returned State is the zero value; a caller checks ok and
// idles the gateway rather than running a phantom session.
func (s *Scheduler) Pick() (State, bool) {
	if s == nil {
		return State{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.table == nil {
		return State{}, false
	}
	// Snapshot takes the table's own RLock (a distinct lock); it never calls back into
	// the scheduler, so holding s.mu across it cannot deadlock.
	snap := s.table.Snapshot()
	switch s.policy {
	case WeightedFair:
		return s.pickWeightedFairLocked(snap)
	default:
		return pickStrictPriority(snap)
	}
}

// PickFrom selects the next session to run from an explicit candidate snapshot under
// the scheduler's configured Policy.
func (s *Scheduler) PickFrom(snap []State) (State, bool) {
	if s == nil {
		return State{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.policy == WeightedFair {
		return s.pickWeightedFairLocked(snap)
	}
	return pickStrictPriority(snap)
}

// PickWeightedFair selects the next session to run from an explicit candidate snapshot
// using the weighted fair-share algorithm, regardless of the scheduler's configured policy.
func (s *Scheduler) PickWeightedFair(snap []State) (State, bool) {
	if s == nil {
		return State{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pickWeightedFairLocked(snap)
}

// Policy returns the scheduler's configured policy.
func (s *Scheduler) Policy() Policy {
	if s == nil {
		return StrictPriority
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.policy
}

// lockAndPruneReservations acquires s.mu and reclaims stale advisory reservations at
// now, returning the dropped reservations plus an unlock func the caller must defer —
// e.g. `defer s.lockAndPruneReservations(now)()`, whose immediate call locks+prunes and
// whose deferred result unlocks. Shared by ReserveKnownComing, Reservations, and
// ExpireReservations so the lock-then-prune boundary can't drift between the three.
func (s *Scheduler) lockAndPruneReservations(now time.Time) (dropped []SlotReservation, unlock func()) {
	s.mu.Lock()
	return s.pruneReservationsLocked(now), s.mu.Unlock
}

// ReserveKnownComing scans the live snapshot for #811 forward-looking TurnIntent
// hints and records advisory reservations for those known-coming turns. The returned
// reservations are the holds a host may use to pin matching KV residency and keep a
// best-effort slot warm. They are strictly lower class than real requests: Pick ignores
// them, and ExpireReservations/PromoteReservation reclaim them without touching table
// state. now is supplied by the caller so the policy remains deterministic in tests.
//
// FIRST-DEADLINE WINS (#13384). The first time a (trace, prefix) intent is observed
// establishes its arrival and expiry from THAT call's now, and every later scan of the
// same intent identity returns those SAME stamps. Reservations are pruned the moment
// their grace closes, so the stamps are remembered in the intents ledger, not in the
// reservation map: this is what lets a scan at now >= the original expiry recognize the
// hint as ALREADY EXPIRED and decline, instead of re-deriving a fresh unexpired hold
// from the unchanged hint and resurrecting it forever. A NEW generation is minted only
// on a witnessed identity change (prefix moved, or the hint was cleared and then
// re-published); an identical hint that was never observed as cleared is conservatively
// declined (R-4).
//
// BOOKKEEPING follows live membership, not the reservation lifetime: a trace that
// leaves the snapshot, or whose hint turns terminal (zero/negative arrival, empty
// prefix, WillDiscard), releases both its reservation and its ledger row, so a churning
// table cannot grow the scheduler without bound. A reservation whose grace has closed
// keeps its ledger row as the expiry tombstone; the row is released when the intent
// generation changes or the trace itself leaves.
func (s *Scheduler) ReserveKnownComing(now time.Time) []SlotReservation {
	if s == nil {
		return nil
	}
	_, unlock := s.lockAndPruneReservations(now)
	defer unlock()
	if s.table == nil {
		return nil
	}
	snap := s.table.Snapshot()
	// One row of deferred state: a ledger row and/or a minted reservation. Writing
	// decisions are staged and applied after the scan so a mint this scan observes is
	// always paired with its ledger row, and a superseded generation's delete lands
	// after (never before) the row it replaces.
	type intentUpdate struct {
		key     reservationKey
		ledger  reservationIntent
		res     SlotReservation
		hasRes  bool
		hasLedg bool
	}
	pending := make([]intentUpdate, 0, len(snap))
	// live keys the ledger rows this snapshot owns; present keys the traces in it, so
	// the release pass can drop every row belonging to a trace that has left the table.
	live := make(map[reservationKey]struct{}, len(snap))
	present := make(map[string]struct{}, len(snap))
	out := make([]SlotReservation, 0)
	for _, st := range snap {
		present[st.TraceID] = struct{}{}
		ident := intentIdentityOf(st)
		prevIdent, hadEpoch := s.epochs[st.TraceID]
		r, ok := reservationFromState(st, now)
		if !ok {
			// Terminal hint (zero/negative/unrepresentable arrival, empty prefix,
			// WillDiscard): release the trace's advisory state and mint nothing. The
			// epoch records the terminal identity so a later clear-then-republish IS a
			// witnessed generation change, while a re-scan of the same terminal hint
			// stays a no-op.
			s.dropReservationsForTraceLocked(st.TraceID, "")
			pending = append(pending, intentUpdate{
				key:     reservationKey{trace: st.TraceID, prefix: prevIdent.prefix},
				hasLedg: true, // clearing this trace's ledger rows (empty identity below)
			})
			s.epochs[st.TraceID] = ident
			continue
		}
		key := reservationKey{trace: r.TraceID, prefix: r.Prefix}
		live[key] = struct{}{}
		// sameGeneration is true only when the trace's intent identity is unchanged AND
		// the scheduler has actually observed this trace before. A first-ever sighting
		// (or a prefix/arrival change) is a NEW generation and gets a fresh deadline.
		sameGeneration := hadEpoch && prevIdent == ident
		if !sameGeneration {
			// A new generation supersedes this trace's ENTIRE older advisory state: drop
			// every reservation and ledger row for the trace so a stale prefix can never
			// be promoted and a stale deadline (even at the SAME prefix but a changed
			// arrival delay) can never shadow the new generation (R-4a).
			s.dropReservationsForTraceLocked(st.TraceID, "")
			pending = append(pending, intentUpdate{
				key:     reservationKey{trace: st.TraceID, prefix: prevIdent.prefix},
				hasLedg: true,
			})
			s.epochs[st.TraceID] = ident
		}
		if prevRes, hasRes := s.reservations[key]; hasRes {
			// A live reservation is this generation's first-observed deadline; a repeat
			// scan returns it verbatim and never re-bases the stamps (R-1).
			out = append(out, prevRes)
			continue
		}
		if led, hasLedger := s.intents[key]; sameGeneration && hasLedger {
			// The reservation is gone but the generation is unchanged: its grace either
			// closed (decline, R-2) or it was promoted (re-mint the SAME stamps).
			if now.UTC().UnixNano() >= led.expiresAtUnixNano {
				continue
			}
			r.ReservedAtUnixNano = led.reservedAtUnixNano
			r.ArrivesAtUnixNano = led.arrivesAtUnixNano
			r.ExpiresAtUnixNano = led.expiresAtUnixNano
			pending = append(pending, intentUpdate{key: key, res: r, hasRes: true, ledger: led, hasLedg: true})
			out = append(out, r)
			continue
		}
		// First observation of this generation's deadline: the established stamps come
		// from THIS call's now, and the ledger remembers them past the reservation (R-1).
		led := reservationIntent{
			identity:           ident,
			reservedAtUnixNano: r.ReservedAtUnixNano,
			arrivesAtUnixNano:  r.ArrivesAtUnixNano,
			expiresAtUnixNano:  r.ExpiresAtUnixNano,
		}
		pending = append(pending, intentUpdate{key: key, res: r, hasRes: true, ledger: led, hasLedg: true})
		out = append(out, r)
	}
	for _, u := range pending {
		if u.hasRes {
			s.reservations[u.key] = u.res
		}
		if u.hasLedg {
			if u.ledger.identity == (reservationIntentIdentity{}) {
				delete(s.intents, u.key)
			} else {
				s.intents[u.key] = u.ledger
			}
		}
	}
	// Release every ledger row the current snapshot does not own: the trace left the
	// table (LRU eviction / Reset), its prefix moved, or its hint turned terminal. This
	// is the R-5 bound — live membership is the only state that persists.
	for key := range s.intents {
		if _, ok := live[key]; !ok {
			delete(s.intents, key)
		}
	}
	// A reservation whose trace left the table is released immediately rather than left
	// to age out of its grace: the hold exists to pin KV for a LIVE session, so a Reset /
	// LRU eviction (or a terminal hint, already dropped above) must not keep scheduler
	// state alive (R-5). This also makes a churn loop return to an empty reservation map.
	for key := range s.reservations {
		if _, ok := present[key.trace]; !ok {
			delete(s.reservations, key)
		}
	}
	for trace := range s.epochs {
		if _, ok := present[trace]; !ok {
			delete(s.epochs, trace)
		}
	}
	sortReservations(out)
	return out
}

// PromoteReservation adopts an active reservation into the real request slot when the
// arriving request proves it is the same session and prefix. A prefix mismatch or an
// expired/missing reservation returns ok=false and leaves any non-matching reservation
// intact, so the caller falls back to the normal cold path rather than fabricating reuse.
func (s *Scheduler) PromoteReservation(trace, prefix string, now time.Time) (SlotPromotion, bool) {
	if s == nil || trace == "" || prefix == "" {
		return SlotPromotion{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneReservationsLocked(now)
	key := reservationKey{trace: trace, prefix: prefix}
	r, ok := s.reservations[key]
	if !ok {
		return SlotPromotion{}, false
	}
	delete(s.reservations, key)
	return SlotPromotion{SlotReservation: r, PromotedAtUnixNano: now.UTC().UnixNano()}, true
}

// ExpireReservations reclaims stale advisory reservations and returns the slots it
// dropped. It never mutates session state and never reports a live request as refused;
// expiry only means the warm hold is gone and a later request must take the normal path.
func (s *Scheduler) ExpireReservations(now time.Time) []SlotReservation {
	if s == nil {
		return nil
	}
	dropped, unlock := s.lockAndPruneReservations(now)
	defer unlock()
	return dropped
}

// Reservations returns the currently-active advisory reservations after first applying
// expiry at now. The returned slice is a copy sorted in deterministic map iteration
// replacement order by the table-free key fields.
func (s *Scheduler) Reservations(now time.Time) []SlotReservation {
	if s == nil {
		return nil
	}
	_, unlock := s.lockAndPruneReservations(now)
	defer unlock()
	out := make([]SlotReservation, 0, len(s.reservations))
	for _, r := range s.reservations {
		out = append(out, r)
	}
	sortReservations(out)
	return out
}

// pickStrictPriority returns the first eligible session in snapshot order. Snapshot is
// already sorted into consumption order, so "first eligible" IS the lowest-Priority
// eligible session, broken by Rev (most recently changed) then TraceID — deterministic
// and correct without any per-scheduler state.
func pickStrictPriority(snap []State) (State, bool) {
	for _, st := range snap {
		if Eligible(st) {
			return st, true
		}
	}
	return State{}, false
}

// pickWeightedFairLocked implements a deterministic SMOOTH weighted round-robin (the
// nginx algorithm) over the currently-eligible sessions. Caller holds s.mu.
//
// ALGORITHM. Each eligible session i is assigned an integer weight derived from its
// Priority relative to the current contention set: weight_i = (maxPriority - Priority_i)
// + 1, where maxPriority is the largest Priority value among the eligible sessions. So
// the highest Priority value (lowest share) gets weight 1, every step lower adds 1, and
// every weight is >= 1 — a lower Priority value yields a proportionally larger share.
// Let W = sum of the weights. On each Pick:
//
//  1. add every session's weight to its running credit c_i;
//  2. the winner is the session with the greatest credit (ties broken toward the
//     earlier-in-snapshot, i.e. higher-share, session — strict-greater keeps the first);
//  3. the winner repays the round by subtracting W from its credit.
//
// Over any window of W consecutive Picks (with a stable eligible set) each session i is
// chosen EXACTLY weight_i times, smoothly interleaved, and the credits return to their
// starting point — so the distribution is exact and the sequence is fully periodic,
// deterministic, and unit-testable to an exact pick order. No clock, no randomness.
//
// The credit map is rebuilt each call to hold only the currently-eligible traces, so it
// stays bounded and a session that leaves contention (paused, exhausted, evicted) drops
// its stale credit and rejoins fair; an empty contention set clears the state entirely.
func (s *Scheduler) pickWeightedFairLocked(snap []State) (State, bool) {
	// Eligible sessions in snapshot order — already Priority-ascending, so a larger-share
	// (lower Priority value) session sorts earlier and therefore wins credit ties.
	elig := make([]State, 0, len(snap))
	for _, st := range snap {
		if Eligible(st) {
			elig = append(elig, st)
		}
	}
	if len(elig) == 0 {
		s.credits = map[string]int64{} // nothing in contention: forget stale virtual time
		return State{}, false
	}

	maxPriority := elig[0].Priority
	for _, st := range elig {
		if st.Priority > maxPriority {
			maxPriority = st.Priority
		}
	}

	next := make(map[string]int64, len(elig))
	var total int64
	bestIdx := -1
	var bestCredit int64
	for i, st := range elig {
		w := int64(maxPriority-st.Priority) + 1
		total += w
		c := s.credits[st.TraceID] + w // carry prior credit (0 if newly eligible), then add weight
		next[st.TraceID] = c
		if bestIdx == -1 || c > bestCredit {
			bestIdx, bestCredit = i, c
		}
	}
	winner := elig[bestIdx]
	next[winner.TraceID] -= total
	s.credits = next // prune to the current contention set
	return winner, true
}

// Eligible reports whether a session may be PICKED to run next: its run-state advances
// (Running or Throttled — Paused/Draining/Stopped are held or ending, never eligible)
// AND it has not exhausted a configured budget axis. It is the helper both policies
// share, exported so a host can apply the same eligibility test (e.g. to render which
// sessions are in contention) without re-deriving the rule.
func Eligible(st State) bool {
	switch st.Run {
	case Running, Throttled:
		// advancing — fall through to the budget check
	default:
		return false
	}
	return !budgetExhausted(st.Budget)
}

// budgetExhausted reports whether any CONFIGURED budget axis has hit zero. An Unbounded
// (negative) turns/tokens axis never exhausts; a context axis is "configured" only when
// its cap is set (ContextTokensCap > 0 — the table stamps the cap from the remaining at
// set-time, and context 0-with-no-cap means "not configured", never exhausted). A
// configured axis at or below zero is exhausted.
func budgetExhausted(b Budget) bool {
	if !b.turnsUnbounded() && b.TurnsLeft <= 0 {
		return true
	}
	if !b.tokensUnbounded() && b.TokensLeft <= 0 {
		return true
	}
	if b.ContextTokensCap > 0 && b.ContextTokensLeft <= 0 {
		return true
	}
	return false
}

// slotCauseFor maps a notable run-state landing to its slot-freed cause. The bool is
// false for any non-notable state (Running/Throttled), so the transition handler never
// emits a slot event for a session that is merely advancing.
func slotCauseFor(to RunState) (SlotCause, bool) {
	switch to {
	case Paused:
		return CausePaused, true
	case Draining:
		return CauseDraining, true
	case Stopped:
		return CauseStopped, true
	}
	return 0, false
}

type reservationKey struct {
	trace  string
	prefix string
}

// MaxArrivingMillis is the STATIC upper bound on an ArrivingInMillis hint: the
// integer-division ceiling that keeps time.Duration(ms)*time.Millisecond from wrapping
// int64. The multiply overflows for MaxArrivingMillis+1 (MaxInt64 ns wraps to a NEGATIVE
// duration, so an oversized hint would otherwise look like an arrival in the past).
//
// It guards the MULTIPLY only, and is deliberately retained as the static half of the
// R-6 gate (SPEC §5): a delay anywhere in [1, MaxArrivingMillis] multiplies cleanly. The
// ADD is a separate hazard and is bounded separately by arrivingMillisOK, which subtracts
// the caller's nowNs (and the grace tail) from the ceiling. Do NOT use this constant alone
// as an admission test — it is a multiply bound, not an arrival bound.
const MaxArrivingMillis = math.MaxInt64 / int64(time.Millisecond)

// arrivingMillisOK is the R-6 gate: a hint must be a positive, representable delay whose
// arrival AND expiry from now both stay inside the int64 nanosecond timeline. Two bounds
// are required, because the two arithmetic steps overflow independently:
//
//   - the MULTIPLY bound: ms <= MaxArrivingMillis keeps ms*time.Millisecond <= MaxInt64.
//     Static arithmetic alone is insufficient: it says nothing about the SUM.
//   - the ADD bound: the reservation computes ReservedAtUnixNano = nowNs and
//     ArrivesAtUnixNano = nowNs + ms*1e6, ExpiresAtUnixNano = that + grace. So the ceiling
//     must leave room for nowNs AND the grace tail:
//     ms <= (MaxInt64 - nowNs - graceNs) / 1e6.
//     Without the nowNs term a near-max hint wraps the SUM negative and mints an inverted
//     row (ExpiresAt < ArrivesAt < ReservedAt) — exactly what R-6 forbids. Without the
//     grace term an arrival landing at MaxInt64 overflows one step later.
//
// Checked BEFORE any multiply or Add, so a pathological hint can never overflow into a
// bogus (negative or past) arrival; rejection mints nothing and never panics.
func arrivingMillisOK(ms int64, now time.Time) bool {
	if ms <= 0 || ms > MaxArrivingMillis {
		return false
	}
	nowNs := now.UTC().UnixNano()
	// Leave room for the constant grace tail as well as the delay itself. graceNs is a
	// small positive constant (DefaultReservationGrace), so the subtraction is exact and
	// cannot itself underflow for any representable now.
	ceilingNs := int64(math.MaxInt64) - nowNs - int64(DefaultReservationGrace)
	if ceilingNs < 0 {
		// now is already within one grace of the representable timeline end: no positive
		// delay can be added without wrapping, so decline.
		return false
	}
	return ms <= ceilingNs/int64(time.Millisecond)
}

func reservationFromState(st State, now time.Time) (SlotReservation, bool) {
	if st.Intent.Prefix == "" {
		return SlotReservation{}, false
	}
	if st.Intent.WillDiscard {
		return SlotReservation{}, false
	}
	if !arrivingMillisOK(st.Intent.ArrivingInMillis, now) {
		return SlotReservation{}, false
	}
	nowT := now.UTC()
	nowNs := nowT.UnixNano()
	arrives := nowT.Add(time.Duration(st.Intent.ArrivingInMillis) * time.Millisecond)
	expires := arrives.Add(DefaultReservationGrace)
	res := SlotReservation{
		TraceID:            st.TraceID,
		Prefix:             st.Intent.Prefix,
		SourceRev:          st.Rev,
		ReservedAtUnixNano: nowNs,
		ArrivesAtUnixNano:  arrives.UnixNano(),
		ExpiresAtUnixNano:  expires.UnixNano(),
	}
	// Belt-and-suspenders: reject rather than mint any row that violates the R-6 ordering
	// invariant ReservedAtNano <= ArrivesAtNano <= ExpiresAtNano. arrivingMillisOK should
	// make this unreachable for a live hint, but the invariant is cheap to assert and is the
	// property callers actually rely on, so it is enforced where the row is built.
	if res.ArrivesAtUnixNano < res.ReservedAtUnixNano || res.ExpiresAtUnixNano < res.ArrivesAtUnixNano {
		return SlotReservation{}, false
	}
	return res, true
}

func (s *Scheduler) pruneReservationsLocked(now time.Time) []SlotReservation {
	if s.reservations == nil {
		s.reservations = map[reservationKey]SlotReservation{}
		return nil
	}
	nowNs := now.UTC().UnixNano()
	var expired []SlotReservation
	for key, r := range s.reservations {
		if r.ExpiresAtUnixNano > 0 && nowNs >= r.ExpiresAtUnixNano {
			expired = append(expired, r)
			delete(s.reservations, key)
		}
	}
	sortReservations(expired)
	return expired
}

func (s *Scheduler) dropReservationsForTraceLocked(trace, keepPrefix string) {
	for key := range s.reservations {
		if key.trace == trace && (keepPrefix == "" || key.prefix != keepPrefix) {
			delete(s.reservations, key)
		}
	}
}

func sortReservations(rs []SlotReservation) {
	sort.Slice(rs, func(i, j int) bool {
		if rs[i].TraceID != rs[j].TraceID {
			return rs[i].TraceID < rs[j].TraceID
		}
		if rs[i].Prefix != rs[j].Prefix {
			return rs[i].Prefix < rs[j].Prefix
		}
		return rs[i].ReservedAtUnixNano < rs[j].ReservedAtUnixNano
	})
}
