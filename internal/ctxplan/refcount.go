package ctxplan

// REFCOUNT WITNESS OVER THE OUTCOME (#846; the reachability-layer epic #844).
//
// Compaction is garbage collection: a resident span survives because something LIVE still
// refers to it. The refcount of a resident span is its count of LIVE REFERENTS — an open
// goal/pin that needs it, a later turn that referenced it (a Hits entry), or a pending
// consumer. ctxplan already witnesses the raw feedback (Outcome{Hits,Faults,Wasted} and
// signal = Hits ∪ pins); this file reads that ONE loop through a refcount lens, naming the
// two failures a refcount makes visible. It adds NO new measurement loop and frees NOTHING.
//
//	false-retain — a span KEPT after nothing references it (silent budget rot). Concretely:
//	               a PINNED span (its only referent is the pin) that lands in Outcome.Wasted
//	               for K consecutive turns. The pin holds refcount >= 1, but no turn-level
//	               referent has touched it in K turns — the pin is idle. Advisory down-weight
//	               only; the pin is NEVER auto-freed (collection of a root is the explicit
//	               goal-discharge child's job, #847).
//
//	false-free   — a span FREED (elided) while a live goal still needs it: an Outcome.Faults
//	               id that is in the active goal's reachable set. This is the bug the whole
//	               frame exists to kill. Once the goal-as-pin-root child (#845) pins the
//	               goal's reachable set, false-free should be STRUCTURALLY impossible — so a
//	               nonzero false-free count is a regression sentinel proving that child works.
//
// The two are kept as DISTINCT classes (not one "compaction miss"): false-retain is
// pinned-but-wasted-for-K (over-resident); false-free is faulted-while-goal-live (under-
// resident). They live on opposite axes, exactly like NoiseTokens vs FaultTokens on
// SignalNoise — collapsing them would hide which way the window is wrong.
//
// Everything here is MEASUREMENT plus an ADVISORY down-weight. Absent or empty, the
// heuristic selection decides unchanged: a refcount signal that gated correctness would be
// a bug, not a feature.

// LiveGoal is the caller-supplied notion of "a goal is live and these spans are its
// reachable set" — the minimal input the false-free arm needs WITHOUT importing
// internal/agent (where the goal pin lives, #845). The caller (the session planner) knows
// the active goal's reachable span ids and passes them in; ctxplan stays a pure metric over
// ids it is handed. An empty/nil LiveGoal (Active false or no Reachable ids) means "no live
// goal" — the false-free arm then reports nothing, so an offline / goal-less session is a
// clean no-op on this axis.
type LiveGoal struct {
	// Active is whether a goal is currently live. When false the false-free arm is inert.
	Active bool
	// Reachable is the set of span ids the live goal needs resident — its reachable set.
	// A fault on any of these is a false-free (the loop paged out a span the goal demands).
	Reachable []string
}

// reachableSet folds LiveGoal.Reachable into a lookup, or nil when no goal is live.
func (g LiveGoal) reachableSet() map[string]bool {
	if !g.Active || len(g.Reachable) == 0 {
		return nil
	}
	m := make(map[string]bool, len(g.Reachable))
	for _, id := range g.Reachable {
		m[id] = true
	}
	return m
}

// Refcount is the two-class refcount verdict for one turn, read off the witnessed Outcome
// through the reachability lens. FalseRetain and FalseFree are SEPARATE classes by
// construction (different fields, different axes) so an operator can tell over-resident rot
// from under-resident loss at a glance.
type Refcount struct {
	// FalseRetain names the pinned spans that have been resident-but-untouched for K
	// consecutive turns — the advisory down-weight candidates (over-resident rot). Their pin
	// is NOT freed; this is a signal, not a collector. Empty when nothing has rotted K turns.
	FalseRetain []string `json:"false_retain,omitempty"`
	// FalseFree names the spans that faulted (were paged out) while the live goal still
	// needed them — the regression sentinel (under-resident loss). Empty when no goal is
	// live, or when the goal's reachable set stayed resident (the target: structurally none).
	FalseFree []string `json:"false_free,omitempty"`
}

// Any reports whether either class flagged anything this turn — a cheap "is the window
// healthy on the refcount axes?" check for an operator surface.
func (r Refcount) Any() bool { return len(r.FalseRetain) > 0 || len(r.FalseFree) > 0 }

// falseRetainK is the default number of CONSECUTIVE turns a pinned span must be Wasted
// before it is flagged false-retain. One idle turn is normal (a pin earns its keep across
// turns, not every turn); K turns of idleness is rot worth an advisory down-weight. Coarse
// on purpose — this is a gauge, not a controller.
const falseRetainK = 3

// RefcountWitness carries the CROSS-TURN state the false-retain arm needs: a per-span
// counter of how many consecutive turns a pinned span has been Wasted. It is the only
// stateful piece (false-free is per-turn and stateless). Keyed by span id (Cell.ID, the
// same key everywhere else). Construct with NewRefcountWitness; the zero value is also a
// usable empty witness (K defaults to falseRetainK).
type RefcountWitness struct {
	// k is the consecutive-wasted threshold; 0 means use falseRetainK.
	k int
	// wastedStreak[id] is how many consecutive turns the pinned span id has been Wasted. An
	// entry is reset (deleted) the moment the span re-enters signal (Hits) or stops being a
	// pinned-and-wasted span — so a span that idles, gets used, then idles again starts over,
	// matching "consecutive".
	wastedStreak map[string]int
}

// NewRefcountWitness builds a witness with an explicit K (consecutive-wasted threshold for
// false-retain). A k <= 0 falls back to the falseRetainK default.
func NewRefcountWitness(k int) *RefcountWitness {
	if k <= 0 {
		k = falseRetainK
	}
	return &RefcountWitness{k: k, wastedStreak: map[string]int{}}
}

// threshold returns the effective K (handles the zero-value witness).
func (rw *RefcountWitness) threshold() int {
	if rw.k <= 0 {
		return falseRetainK
	}
	return rw.k
}

// Observe folds ONE turn's Plan + witnessed Outcome (and the optional LiveGoal) into the
// two-class Refcount verdict, advancing the cross-turn false-retain streaks. It reads only
// the loop ctxplan already witnesses:
//
//   - FALSE-RETAIN: for each PINNED resident span, if it is in Outcome.Wasted (resident,
//     untouched) AND not in Hits, bump its consecutive-wasted streak; the moment the streak
//     reaches K it is flagged false-retain (advisory down-weight). Any pinned span that is
//     Hit this turn (or is no longer pinned/wasted) has its streak RESET to zero — the
//     streak counts CONSECUTIVE idle turns, so a single use clears the rot. The pin itself
//     is never touched: Observe returns a signal, it does not free.
//
//   - FALSE-FREE: for each id in Outcome.Faults that is in the live goal's reachable set,
//     flag it false-free (the loop paged out a span the live goal still needs). Stateless
//     and per-turn. No live goal => no false-free.
//
// Pure given its state: the same (plan, outcome, goal) advances the streaks deterministically
// (ids are flagged in plan-resident / Faults order, no map iteration in the output). Calling
// Observe with no pins and no live goal is a clean no-op that flags nothing.
func (rw *RefcountWitness) Observe(p Plan, o Outcome, goal LiveGoal) Refcount {
	if rw.wastedStreak == nil {
		rw.wastedStreak = map[string]int{}
	}
	hit := make(map[string]bool, len(o.Hits))
	for _, id := range o.Hits {
		hit[id] = true
	}
	wasted := make(map[string]bool, len(o.Wasted))
	for _, id := range o.Wasted {
		wasted[id] = true
	}

	var rc Refcount
	k := rw.threshold()

	// FALSE-RETAIN: walk the resident pins in plan order (deterministic output) and advance
	// each pinned span's consecutive-wasted streak.
	seenPinned := make(map[string]bool, len(p.Selected))
	for _, sel := range p.Selected {
		if !sel.Pinned {
			continue
		}
		seenPinned[sel.ID] = true
		// A pinned span that the turn referenced (Hit) is doing its job — reset the streak.
		// A pinned span resident-but-untouched (Wasted, not Hit) is idle — extend the streak.
		// A pinned span neither hit nor wasted this turn (unaccounted) is NOT proof of rot;
		// treat it as "no evidence of idleness" and reset, so only WITNESSED idleness counts.
		if wasted[sel.ID] && !hit[sel.ID] {
			rw.wastedStreak[sel.ID]++
			if rw.wastedStreak[sel.ID] >= k {
				rc.FalseRetain = append(rc.FalseRetain, sel.ID)
			}
		} else {
			delete(rw.wastedStreak, sel.ID)
		}
	}
	// A span that dropped out of the resident pin set entirely (no longer pinned) is no longer
	// a false-retain candidate — clear any stale streak so it cannot resurface later.
	for id := range rw.wastedStreak {
		if !seenPinned[id] {
			delete(rw.wastedStreak, id)
		}
	}

	// FALSE-FREE: a fault on a span the live goal still needs. Stateless, per-turn.
	if reach := goal.reachableSet(); reach != nil {
		for _, id := range o.Faults {
			if reach[id] {
				rc.FalseFree = append(rc.FalseFree, id)
			}
		}
	}

	return rc
}

// Streak returns the current consecutive-wasted streak for a pinned span id (0 if none) —
// for tests and an EXPLAIN surface that wants to show "rotting for N of K turns" before the
// flag fires. Read-only; it does not advance state.
func (rw *RefcountWitness) Streak(id string) int {
	if rw.wastedStreak == nil {
		return 0
	}
	return rw.wastedStreak[id]
}
