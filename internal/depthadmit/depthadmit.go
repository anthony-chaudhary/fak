package depthadmit

// depthadmit.go — the pure depth fold.
//
// THE GAP THIS FILLS. The trajectory-control substrate already answers two
// questions about a live objective, and neither of them is depth:
//
//   - internal/trajctl curve.go folds the witnessed progress curve into MOTION —
//     HEALTHY / STALL / DRIFT / DETOUR_OVERRUN. "Is this objective moving?"
//   - internal/focusscore folds the objective tree into BREADTH — how many
//     objectives are active at once against a pinned WIP cap, which the dispatch
//     tick spends as FOCUS_WIP_SATURATED. "Is the fleet fanning out?"
//
// Both are brakes on breadth. Neither can see how FAR down one line of work the
// fleet actually got, because both read the score curve and the objective COUNT,
// never the declared plan. So an objective whose Plan declares six phases can be
// closed `met` with one phase witnessed and score a perfect focus/curve read: one
// active objective, a rising curve, closed on time. The substrate has no way to
// say "that line stopped short" — a shallow close is indistinguishable from a
// deep one, and what is not measured is not driven.
//
// This fold reads the ONE piece of evidence both of the others skip: the declared
// plan (trajctl.Objective.Plan) against the phases a commit actually witnessed
// (the `Trajctl-Phase:` trailer bindings, trajctl phasecommits.go). From those two
// it derives the depth FRONTIER — the first declared phase with no witness — and
// that single value does all the work here:
//
//	DRIVE depth   Admit refuses a `met` closure whose declared plan is not carried,
//	              so "done" costs a witnessed phase per declared phase instead of a
//	              self-report. Declaring the plan becomes the price of claiming done.
//	ALLOW depth   Persist compares two attempts' frontiers. Deep work on a hard
//	              problem looks exactly like thrash to a counter — both are N
//	              attempts on one issue — and the breadth machinery
//	              (internal/attemptbudget) can only count. Frontier MOVEMENT is the
//	              signal that separates them: an attempt that carried a phase the
//	              last one did not is depth and has earned its next attempt; a
//	              frontier that has not moved is thrash.
//	HAND OFF      HandoffLine renders the frontier as the one line a successor
//	              session needs to resume mid-line instead of re-planning from the
//	              top — the concrete next phase, not a restatement of the goal.
//
// PURITY. Same Input in, same Report out: no I/O, no clock reads, no git. The
// impure half — reading the ledger, resolving whether a phase's commit SHA still
// exists — belongs to the caller, exactly the seam trajctl's own
// GitEvidenceResolver and PhaseCommitsFromTrailers already use. Like
// internal/stepbaton this package takes plain SCALARS (phase ids and titles),
// never a trajctl type, so it stays stdlib-only, tier-1, and usable for any
// declared-and-witnessed plan rather than only the trajctl ledger's.
//
// SCOPE — this is the KERNEL plus its one shipped consumer, the `fak trajctl
// close` gate. It deliberately does NOT fold itself into internal/attemptbudget's
// dispatchability decision (Persist is the value that fold would read, but the
// wiring is a separate change with its own golden), and it does not inject the
// handoff line at SessionStart (that is the stepbaton consumer seam).

import (
	"fmt"
	"strings"
)

// Schema is the pinned schema id for a depth report. Downstream consumers pin to
// this string; bump it (never mutate a shipped field's meaning) if the shape
// changes, so a stale reader recognizes a foreign schema instead of misreading it.
const Schema = "fak-depthadmit-report/1"

// RefusalReason is the closed refusal token Admit spends when a closure claims a
// depth the witnessed plan does not support. Registered in dos.toml so
// `dos_check_reason` validates it and an operator sees a depth hold distinctly
// from a breadth hold (FOCUS_WIP_SATURATED) or an attempt hold.
const RefusalReason = "DEPTH_NOT_CARRIED"

// Verdict is the closed depth vocabulary. A consumer that sees any other string
// is reading a foreign or newer schema.
type Verdict string

const (
	// VerdictUndeclared: the objective declared no plan, so its depth is not
	// merely shallow but UNKNOWABLE. Distinct from Shallow on purpose — nothing
	// can be carried against a line that was never drawn, and the honest read is
	// "no claim is checkable here", not "zero progress".
	VerdictUndeclared Verdict = "UNDECLARED"
	// VerdictShallow: a plan is declared and NOT ONE of its phases is witnessed.
	// The line was drawn and never walked.
	VerdictShallow Verdict = "SHALLOW"
	// VerdictAdvancing: at least one declared phase is witnessed and a frontier
	// remains. This is the healthy mid-line state — depth in progress, not a
	// defect.
	VerdictAdvancing Verdict = "ADVANCING"
	// VerdictCarried: every declared phase has a witness. The line was carried to
	// its declared end, which is the ONLY state that admits a `met` closure.
	VerdictCarried Verdict = "CARRIED"
)

// validVerdicts is the membership set every Verdict must belong to — used by
// tests and any deserializing caller to fail closed on a corrupt value.
var validVerdicts = map[Verdict]bool{
	VerdictUndeclared: true,
	VerdictShallow:    true,
	VerdictAdvancing:  true,
	VerdictCarried:    true,
}

// ValidVerdict reports whether v is a member of the closed vocabulary.
func ValidVerdict(v Verdict) bool { return validVerdicts[v] }

// Closure is the closed terminal-transition vocabulary Admit adjudicates. It
// mirrors trajctl's two terminal statuses verbatim, held as plain strings so this
// package keeps no trajctl dependency.
type Closure string

const (
	// ClosureMet claims the objective's goal was reached — the claim that costs a
	// carried plan.
	ClosureMet Closure = "met"
	// ClosureAbandoned drops the objective without claiming its goal. Never
	// refused: abandoning is a legitimate operator decision, and refusing it would
	// only trap dead objectives open. The Report still records the depth reached,
	// so the ledger shows what was left on the table.
	ClosureAbandoned Closure = "abandoned"
)

// ValidClosure reports whether c is a member of the closed vocabulary.
func ValidClosure(c Closure) bool { return c == ClosureMet || c == ClosureAbandoned }

// Phase is one declared unit of the line of work, in declared order. It is the
// scalar projection of trajctl.PlanPhase — id plus optional human title, no
// trajctl import.
type Phase struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`
}

// Input is the witnessed depth evidence for exactly ONE objective.
type Input struct {
	// Plan is the declared phases in declared order. Order is load-bearing: the
	// frontier is the FIRST unwitnessed phase, which is what makes it a next step
	// rather than a set of leftovers.
	Plan []Phase `json:"plan,omitempty"`
	// Witnessed is the set of phase ids for which the caller RESOLVED a witness —
	// a commit that still exists, not a commit that was merely claimed. Resolving
	// is the caller's impure job; this fold trusts the list it is handed and
	// nothing else. Ids naming no declared phase are reported as Foreign, never
	// silently credited.
	Witnessed []string `json:"witnessed,omitempty"`
}

// Frontier is the first declared phase with no witness: the concrete next step of
// this line of work. It is nil exactly when the plan is fully carried (or when
// there is no well-formed plan at all).
type Frontier struct {
	// Index is the 0-based position of the frontier phase among the well-formed
	// declared phases.
	Index   int    `json:"index"`
	PhaseID string `json:"phase_id"`
	Title   string `json:"title,omitempty"`
	// Remaining counts the unwitnessed declared phases, this one included, so a
	// consumer can render "4 remaining" without re-walking the plan.
	Remaining int `json:"remaining"`
}

// Coverage is the raw tally the Verdict is derived from. Every field is
// re-derivable from Input alone; nothing here is self-reported.
type Coverage struct {
	// Declared counts the well-formed, distinct declared phases.
	Declared int `json:"declared"`
	// Carried counts the declared phases with a resolved witness.
	Carried int `json:"carried"`
	// Unwitnessed lists the declared phase ids with no witness, in declared order.
	Unwitnessed []string `json:"unwitnessed,omitempty"`
	// Foreign lists witnessed ids that name NO declared phase, sorted by first
	// appearance. A non-empty Foreign means the work drifted off the declared plan
	// — the commits landed somewhere the plan does not describe. It is surfaced,
	// never credited, and never blocks a closure on its own: the plan being an
	// incomplete description of the work is a planning signal, not a depth defect.
	Foreign []string `json:"foreign,omitempty"`
	// Malformed counts declared phases whose id is blank after trimming. Such a
	// phase can never be witnessed (a `Trajctl-Phase:` trailer cannot name it), so
	// it is excluded from Declared and instead makes the plan uncheckable — Admit
	// refuses a `met` closure while any exist rather than quietly scoring around
	// them.
	Malformed int `json:"malformed,omitempty"`
}

// Report is the folded depth read for one objective.
type Report struct {
	Schema   string    `json:"schema"`
	Verdict  Verdict   `json:"verdict"`
	Coverage Coverage  `json:"coverage"`
	Frontier *Frontier `json:"frontier,omitempty"`
}

// Fold derives the depth report from witnessed evidence. It is total: every Input,
// including a zero one, yields exactly one member of the closed Verdict vocabulary.
//
// Normalization, all fail-closed:
//   - phase ids and witnessed ids are trimmed of surrounding space;
//   - a declared phase whose id is blank is Malformed, not Declared;
//   - a repeated declared id counts once (first occurrence wins, keeping its
//     title), so a duplicated phase cannot inflate the denominator;
//   - a witnessed id matching no declared phase is Foreign, never Carried.
func Fold(in Input) Report {
	r := Report{Schema: Schema}

	witnessed := make(map[string]bool, len(in.Witnessed))
	for _, w := range in.Witnessed {
		if w = strings.TrimSpace(w); w != "" {
			witnessed[w] = true
		}
	}

	// Walk the plan in declared order so the frontier is the FIRST gap.
	seen := make(map[string]bool, len(in.Plan))
	matched := make(map[string]bool, len(witnessed))
	for _, p := range in.Plan {
		id := strings.TrimSpace(p.ID)
		if id == "" {
			r.Coverage.Malformed++
			continue
		}
		if seen[id] {
			continue
		}
		seen[id] = true
		idx := r.Coverage.Declared
		r.Coverage.Declared++
		if witnessed[id] {
			r.Coverage.Carried++
			matched[id] = true
			continue
		}
		r.Coverage.Unwitnessed = append(r.Coverage.Unwitnessed, id)
		if r.Frontier == nil {
			r.Frontier = &Frontier{Index: idx, PhaseID: id, Title: strings.TrimSpace(p.Title)}
		}
	}
	if r.Frontier != nil {
		r.Frontier.Remaining = len(r.Coverage.Unwitnessed)
	}

	// Foreign witnesses, in first-appearance order — deterministic without a sort.
	emitted := make(map[string]bool, len(witnessed))
	for _, w := range in.Witnessed {
		w = strings.TrimSpace(w)
		if w == "" || matched[w] || emitted[w] {
			continue
		}
		emitted[w] = true
		r.Coverage.Foreign = append(r.Coverage.Foreign, w)
	}

	switch {
	case r.Coverage.Declared == 0:
		r.Verdict = VerdictUndeclared
	case r.Coverage.Carried == 0:
		r.Verdict = VerdictShallow
	case r.Coverage.Carried < r.Coverage.Declared:
		r.Verdict = VerdictAdvancing
	default:
		r.Verdict = VerdictCarried
	}
	return r
}

// Decision is the closure adjudication: may this objective take the requested
// terminal transition, given the depth its plan actually witnessed?
type Decision struct {
	Admitted bool    `json:"admitted"`
	Closure  Closure `json:"closure"`
	// Reason is the closed refusal token (RefusalReason) when Admitted is false,
	// and empty when admitted. It is never free text.
	Reason string `json:"reason,omitempty"`
	// Detail is the human sentence explaining THIS refusal or admission — which
	// phase is the frontier, how much of the plan is carried. Advisory prose; the
	// machine-readable facts are Reason, Report, and Frontier.
	Detail string `json:"detail"`
	Report Report `json:"report"`
}

// Admit adjudicates a terminal transition against witnessed depth.
//
// Invariant: depth admission decisions are fail-closed and monotonic.
//
// The rule, in one line: `met` costs a carried plan; `abandoned` costs nothing but
// still records what was left.
//
//   - ClosureMet is admitted ONLY on VerdictCarried with no malformed phases.
//     SHALLOW and ADVANCING are refused because the declared line is not walked;
//     UNDECLARED is refused because a `met` with no plan claims a depth no one can
//     check — which is the shallow-close hole this fold exists to close, in its
//     purest form. A malformed plan is refused for the same reason: a phase that
//     cannot be named cannot be witnessed.
//   - ClosureAbandoned is ALWAYS admitted. Refusing it would trap dead objectives
//     open, and abandoning claims nothing that depth could contradict. The Report
//     rides along so the ledger records the depth reached at the moment of the
//     drop.
//
// An unrecognized Closure is refused: fail closed on a foreign vocabulary rather
// than defaulting to the permissive arm.
func Admit(in Input, c Closure) Decision {
	rep := Fold(in)
	d := Decision{Closure: c, Report: rep}

	switch {
	case c == ClosureAbandoned:
		d.Admitted = true
		d.Detail = fmt.Sprintf("abandoned: depth %s at %d/%d declared phases carried — recorded, not refused",
			rep.Verdict, rep.Coverage.Carried, rep.Coverage.Declared)
	case c != ClosureMet:
		d.Reason = RefusalReason
		d.Detail = fmt.Sprintf("unrecognized closure %q: not a member of the closed vocabulary (%q, %q), so the transition fails closed",
			string(c), string(ClosureMet), string(ClosureAbandoned))
	case rep.Coverage.Malformed > 0:
		d.Reason = RefusalReason
		d.Detail = fmt.Sprintf("met refused: %d declared phase(s) have a blank id and can never be witnessed, so this plan's depth is uncheckable — give every phase an id a %s trailer can name",
			rep.Coverage.Malformed, "Trajctl-Phase:")
	case rep.Verdict == VerdictUndeclared:
		d.Reason = RefusalReason
		d.Detail = "met refused: no plan is declared, so `met` claims a depth nothing can check — declare the phases this line of work has to carry, then close it once they are witnessed"
	case rep.Verdict == VerdictCarried:
		d.Admitted = true
		d.Detail = fmt.Sprintf("met admitted: all %d declared phases are witnessed", rep.Coverage.Declared)
	default:
		d.Reason = RefusalReason
		d.Detail = fmt.Sprintf("met refused: %d of %d declared phases are witnessed; the line stops at %s",
			rep.Coverage.Carried, rep.Coverage.Declared, frontierPhrase(rep.Frontier))
	}
	return d
}

// frontierPhrase renders a frontier for a refusal sentence. A nil frontier cannot
// reach the refusal arms (a plan with no frontier is CARRIED), but rendering it
// defensively keeps the message total rather than panicking on a future caller.
func frontierPhrase(f *Frontier) string {
	if f == nil {
		return "an unresolved phase"
	}
	if f.Title == "" {
		return fmt.Sprintf("phase %q (%d remaining)", f.PhaseID, f.Remaining)
	}
	return fmt.Sprintf("phase %q — %s (%d remaining)", f.PhaseID, f.Title, f.Remaining)
}

// Persistence is the closed vocabulary for what changed between two attempts at
// the SAME line of work. It exists because depth and thrash are the same shape to
// a counter: both are N attempts on one issue, and internal/attemptbudget — which
// holds an issue once its attempt budget is crossed — can only count. Frontier
// movement is what tells them apart.
type Persistence string

const (
	// PersistenceAdvanced: the later attempt carried declared phases the earlier
	// one had not. This is depth, not repetition — the attempt bought witnessed
	// ground, and a budget that holds it is punishing the deep work it wanted.
	PersistenceAdvanced Persistence = "ADVANCED"
	// PersistenceStuck: the same declared phases are carried. Nothing was bought;
	// this is the shape a budget SHOULD hold on.
	PersistenceStuck Persistence = "STUCK"
	// PersistenceRegressed: fewer declared phases are carried than before — a
	// witness was LOST (a commit went dangling, a phase was re-opened). Strictly
	// worse than stuck, and never grounds for extending a budget.
	PersistenceRegressed Persistence = "REGRESSED"
	// PersistenceUnknown: at least one side declared no plan, so there is no
	// comparable frontier. Abstains rather than guessing — an undeclared plan is
	// not evidence of either depth or thrash.
	PersistenceUnknown Persistence = "UNKNOWN"
)

// ValidPersistence reports whether p is a member of the closed vocabulary.
func ValidPersistence(p Persistence) bool {
	switch p {
	case PersistenceAdvanced, PersistenceStuck, PersistenceRegressed, PersistenceUnknown:
		return true
	default:
		return false
	}
}

// PersistenceReport is the compared depth of two attempts at one line of work.
type PersistenceReport struct {
	Persistence Persistence `json:"persistence"`
	// CarriedDelta is later.Carried - earlier.Carried: the witnessed ground the
	// later attempt bought (negative when a witness was lost). Zero when Unknown.
	CarriedDelta int `json:"carried_delta"`
	// From and To are the two frontiers, so a consumer can render "was at p2, now
	// at p4" without re-folding either side.
	From *Frontier `json:"from,omitempty"`
	To   *Frontier `json:"to,omitempty"`
	// Detail is the human sentence. Advisory prose.
	Detail string `json:"detail"`
}

// Persist compares an earlier and a later depth report for the SAME line of work
// and names what the later attempt bought.
//
// The measure is CARRIED COUNT, not frontier index, and that choice is
// load-bearing: a phase witnessed out of declared order (a later phase landed
// first) advances the count while leaving the frontier parked, and that is still
// real ground bought. Counting is the forgiving-but-honest read; frontier index
// alone would score genuine progress as thrash.
//
// A plan that GREW between attempts (the line was re-planned deeper) does not by
// itself count as advancing — carrying more phases does, declaring more does not.
// That is deliberate: re-planning is not progress, and a fold that rewarded it
// would make "add a phase" the cheapest way to buy another attempt.
func Persist(earlier, later Report) PersistenceReport {
	pr := PersistenceReport{From: earlier.Frontier, To: later.Frontier}
	if earlier.Verdict == VerdictUndeclared || later.Verdict == VerdictUndeclared {
		pr.Persistence = PersistenceUnknown
		pr.Detail = "persistence unknown: at least one attempt declared no plan, so there is no comparable frontier — this is an abstention, not a pass"
		return pr
	}
	pr.CarriedDelta = later.Coverage.Carried - earlier.Coverage.Carried
	switch {
	case pr.CarriedDelta > 0:
		pr.Persistence = PersistenceAdvanced
		pr.Detail = fmt.Sprintf("advanced: %d more declared phase(s) witnessed (%d/%d -> %d/%d) — this attempt bought ground",
			pr.CarriedDelta, earlier.Coverage.Carried, earlier.Coverage.Declared, later.Coverage.Carried, later.Coverage.Declared)
	case pr.CarriedDelta < 0:
		pr.Persistence = PersistenceRegressed
		pr.Detail = fmt.Sprintf("regressed: %d fewer declared phase(s) witnessed (%d/%d -> %d/%d) — a witness was lost",
			-pr.CarriedDelta, earlier.Coverage.Carried, earlier.Coverage.Declared, later.Coverage.Carried, later.Coverage.Declared)
	default:
		pr.Persistence = PersistenceStuck
		pr.Detail = fmt.Sprintf("stuck: the same %d/%d declared phase(s) are witnessed — this attempt bought no ground",
			later.Coverage.Carried, later.Coverage.Declared)
	}
	return pr
}

// HandoffLine renders the one line a successor session needs to resume this line
// of work mid-stream instead of re-planning from the top. It names the concrete
// next phase, never restates the goal — the goal is already in the objective, and
// a successor that re-reads the goal re-plans, while a successor handed the
// frontier continues.
func HandoffLine(objectiveID string, r Report) string {
	id := strings.TrimSpace(objectiveID)
	if id == "" {
		id = "(unnamed objective)"
	}
	switch r.Verdict {
	case VerdictUndeclared:
		return fmt.Sprintf("depth: %s declares no plan — depth is unknowable; declare the phases before claiming this line is done", id)
	case VerdictCarried:
		return fmt.Sprintf("depth: %s carried %d/%d phases — the declared line is complete", id, r.Coverage.Carried, r.Coverage.Declared)
	default:
		next := ""
		if r.Frontier != nil {
			next = fmt.Sprintf(" next phase %q", r.Frontier.PhaseID)
			if r.Frontier.Title != "" {
				next += fmt.Sprintf(" (%s)", r.Frontier.Title)
			}
			next += fmt.Sprintf(", %d remaining", r.Frontier.Remaining)
		}
		return fmt.Sprintf("depth: %s carried %d/%d phases —%s", id, r.Coverage.Carried, r.Coverage.Declared, next)
	}
}
