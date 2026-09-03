// Package superloop is the operator-intent META-LOOP: a SUPER LOOP walks a curated
// set of member loops/gardens/scorecards, reads their status FIRST, and selects
// worst-first which member to enter — the layer that sits ABOVE a normal loop.
//
// # The differentiation (super loop vs normal loop) — the conceptual crux
//
// The fleet already runs many NORMAL loops: the dispatch loop resolves one issue,
// the garden tick reaps one class of stale work, a scorecard run reports one debt,
// `fak loop drive` settles one GOAL.md witness. Each is keyed on a TASK and a
// cadence; each tick DOES one unit of concrete work; each is a LEAF in the work
// graph — it acts on the codebase/world directly, and its health is "did it tick
// recently + keep-rate". A normal loop has no members, no read-first phase, and no
// selection step: it just runs its body.
//
// A SUPER LOOP is keyed on an operator INTENT ("improve quality"), not a task. Its
// tick is a TRAVERSAL over OTHER loops, in four moves a normal loop never makes:
//
//  1. WALK    — read each member's STATUS before doing anything (orient-over-loops).
//  2. SELECT  — worst-first pick the member most in debt / dark / stale.
//  3. DESCEND — enter that member's loop (which may itself be a super loop: recursion).
//  4. FOLD    — exit on the AGGREGATE clearing (folded debt <= floor), not on any
//     single task's witness.
//
// So a super loop is an INTERIOR node: it mutates nothing at its own altitude — its
// only effect is reading members and driving them; the MEMBERS mutate. That is the
// load-bearing line. Five properties separate the two, and [Classify] checks all
// five against a [LoopFacts] descriptor so "this is a super loop" is a witnessed
// verdict, not a label. A super loop generalizes the garden bundle
// (internal/gardenbundle): the garden is a FIXED bundle folded into one OK/RED gate;
// a super loop is an intent-named, worst-first-selecting, recursively-nestable bundle
// whose members are themselves loops, and whose output is a WORKLIST (what to enter
// next), not just a pass/fail.
//
// The package is PURE: the registry is data, [Classify] and [Walk] are deterministic
// folds over inputs the impure shell (cmd/fak/superloop.go) supplies. It reads no
// files and no clock; the shell collects member status (scorecard baseline debt,
// loopfleet loop health) and hands it in — the same shell/core split loopindex and
// loopfleet use.
package superloop

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// WalkSchema is the versioned payload tag the `--json` walk emits.
const WalkSchema = "fak.superloop-walk.v1"

// MemberKind tags which existing surface a member references, so the shell knows how
// to read its status and a reader knows what altitude the member sits at.
type MemberKind string

const (
	// KindScorecard is a control-pane scorecard key (debt-bearing); status is read
	// from the pinned scorecard baseline / a fresh run.
	KindScorecard MemberKind = "scorecard"
	// KindLoop is a ledgered loop id; status is read from the cross-ledger loop-health
	// fold (internal/loopfleet).
	KindLoop MemberKind = "loop"
	// KindGarden is the garden bundle (itself a fold-over-folds); a member to descend.
	KindGarden MemberKind = "garden"
	// KindSuperloop is another super loop — the recursion case: a super loop whose
	// member is a super loop walks it as a sub-traversal.
	KindSuperloop MemberKind = "superloop"
	// KindSurface is a named command/control surface whose own status fold is outside
	// this generic registry. It is surfaced as a descend pointer, not weighed here.
	KindSurface MemberKind = "surface"
	// KindUtilization is a live CAPACITY-utilization signal: a resource pool whose
	// UNUSED headroom is the debt. Unlike a scorecard (a committed baseline value) or a
	// loop (a ledger liveness fold), its status is read LIVE by the shell at walk time
	// — an account pool's offerable-but-unused seats, or a lab node pool's up-but-idle
	// boxes. The debt is "capacity available but not being used": a resource sitting
	// warm while work backs up. Ref names which pool ("account-limits", "node-resources").
	KindUtilization MemberKind = "utilization"
	// KindTrajectory is a trajectory-control OBJECTIVE (internal/trajctl): a steered
	// goal whose worst-first status is its curve SIGNAL severity
	// (HEALTHY < STALL < DRIFT < DETOUR_OVERRUN). Like KindUtilization its status is
	// read LIVE by the shell — here from the trajctl ledger's folded curve, not a
	// committed baseline — and a SINGLE registry member ENUMERATES into one status per
	// OPEN objective, so "improve trajectory health" walks its objectives worst-first
	// through the same fold. Ref names the objective set on the registry member ("open"
	// = every active/paused objective; any other value selects one objective by id) and
	// the concrete objective id on each enumerated status. This joins the existing
	// controller-of-controllers walk rather than growing a rival walker (issue #2563).
	KindTrajectory MemberKind = "trajectory"
	// KindLoopFleet is the WHOLE-FLEET member kind (issue #4955): the same
	// fleet-enumerating precedent KindTrajectory set, one level up. A SINGLE registry
	// member with Ref="all" ENUMERATES into one status per ledgered loop on the
	// canonical roster (the deduped union [BuildRoster] folds: loopfleet's folded
	// loops + the loopmgr job registry + the registered super loops), so no
	// ledgered-and-folding loop can be invisible to a walk just because no intent
	// hand-named it. Like KindTrajectory the statuses are read LIVE by the shell —
	// from the cross-ledger loop-health fold — via [LoopFleetStatuses]: each
	// enumerated status carries the concrete loop identity as its Ref, an
	// absent/unreadable ledger surfaces as an UNMEASURED known gap (never dropped,
	// never a healthy zero), and any other Ref selects one loop by identity. The
	// worst-first walk over this roster is the meta-walker follow-on (#4958).
	KindLoopFleet MemberKind = "loopfleet"
)

// WorkClass is the gardening-vs-throughput bucket a member's work falls into — the
// axis a night has to keep in balance (#3126). A fleet that spends every tick tending
// quality while open issues pile up is as out of balance as one that drains issues
// while the gardens rot; the walk classifies each member so the mix is visible and a
// soft tie-break can nudge a lopsided run back toward its declared target.
type WorkClass string

const (
	// WorkGardening is tend-the-house work: scorecards, the garden bundle, and named
	// control surfaces. It retires quality/hygiene debt rather than moving the backlog.
	WorkGardening WorkClass = "gardening"
	// WorkThroughput is issue-drain work: the dispatch loops and drain-* intents that
	// move open issues toward done. It is the backlog-clearing output of a run.
	WorkThroughput WorkClass = "throughput"
	// WorkNeutral is everything the gardening/throughput axis does not name — capacity
	// signals (utilization) and non-drain report loops (cadence, dojo, dogfood). It is
	// surfaced for visibility but never weighed by the mix tie-break.
	WorkNeutral WorkClass = "neutral"
)

// DefaultThroughputTargetPct is the soft mix target a Super uses when it declares none:
// a balanced 50% throughput share. It only ever breaks a tie between two equally-urgent
// members of opposite class, so the exact number is a gentle lean, not a quota.
const DefaultThroughputTargetPct = 50

// classifyWork buckets a member into the gardening/throughput/neutral axis. It reads
// only the member's declared kind and ref — a pure classification, no status needed:
//
//   - scorecard / garden / surface → gardening (tend quality & hygiene);
//   - a loop or sub-super-loop whose ref names issue-drain work → throughput;
//   - every other loop (report/calibration cadences) and utilization → neutral.
//
// The throughput test is ref-based rather than kind-based because KindLoop and
// KindSuperloop carry BOTH backlog-drain members (dispatch, drain-*) and non-drain ones
// (cadence, dojo, the dogfood probe): only the issue-drain refs are throughput.
// KindLoopFleet rides the same ref test (#4955): each ENUMERATED fleet loop carries its
// concrete loop identity as Ref, so every roster loop lands on the gardening/throughput
// axis the mix fold (and #4956's gate) reads, exactly as if an intent had hand-named it.
func classifyWork(m Member) WorkClass {
	switch m.Kind {
	case KindScorecard, KindGarden, KindSurface:
		return WorkGardening
	case KindLoop, KindSuperloop, KindLoopFleet:
		if isIssueDrain(m.Ref) {
			return WorkThroughput
		}
		return WorkNeutral
	default:
		return WorkNeutral
	}
}

// isIssueDrain reports whether a loop/super-loop ref names backlog-draining work. The
// three tokens cover every issue-drain surface in the registry — the legacy aggregate
// "dispatch" ledger, the named "issue-resolve" dispatch goals, and the "drain-*"
// intents — while leaving report/calibration loops (cadence, dojo, dogfood) neutral.
func isIssueDrain(ref string) bool {
	r := strings.ToLower(ref)
	return strings.Contains(r, "dispatch") ||
		strings.Contains(r, "drain") ||
		strings.Contains(r, "issue-resolve")
}

// --- the per-member PROGRESS verdict: SPINNING vs advancing (issue #4956) ----
//
// The liveness verdict (loopmgr.DeriveState, folded by the shell into Dark/Debt)
// answers "is it ticking?"; it can never answer "is it PRODUCING?". A member that
// fires on cadence while its ledger-verified progress advances by nothing used to
// read clean — the walk saw DARK (not ticking) but never SPINNING (ticking, zero
// verified progress). The progress verdict sits ALONGSIDE the liveness verdict —
// it deliberately does not overload Dark/Debt/Measured — and reuses the relay
// G-track machinery wholesale rather than forking it: "advanced" is the EXACT
// [relay.NoProgressEscape.Advances] rule (the verified step count rose past the
// high-water mark), the progress itself arrives as a [relay.VerifiedProgress]
// (read by the shell through relay.ReadVerifiedProgress — the intent ledger's own
// rows, NEVER a self-reported keep), the benign park is the exact
// [relay.IdleAwareEscape] rule, and the two closed reason tokens are relay's own
// (RELAY_NO_PROGRESS / RELAY_IDLE_PARKED) so a supervisor can dos_check_reason
// the verdict instead of parsing free text.

// MemberProgress is the closed per-member progress verdict. The zero value ""
// means the progress axis was not read at all (non-loop members, dark members):
// surface-only, never weighed.
type MemberProgress string

const (
	// ProgressAdvancing: the ledger-verified step count rose past the window's
	// high-water mark — real, re-verifiable forward movement. Healthy.
	ProgressAdvancing MemberProgress = "advancing"
	// ProgressSpinning: the member is TICKING (live/stale) but its verified
	// progress did not advance, and its work class is throughput (issue-drain) —
	// live-but-producing-nothing, the state this verdict exists to surface. It
	// carries relay.ReasonNoProgress (RELAY_NO_PROGRESS): an OPERATOR_GATE —
	// escalate for a revive/redirect, never auto-replan.
	ProgressSpinning MemberProgress = "spinning"
	// ProgressIdleParked: a non-throughput (report/calibration/watch) member with
	// no advance whose idleness is POSITIVELY proven — the watched invariant HOLDS
	// against a durable witness AND the ledger confirms zero admitted pending work
	// (the relay idle rule). Benign; carries relay.ReasonIdleParked
	// (RELAY_IDLE_PARKED). Never an alarm, never debt.
	ProgressIdleParked MemberProgress = "idle-parked"
	// ProgressIdleUnproven: a non-throughput member with no advance that could NOT
	// prove it is idle (unknown invariant, unread pending count, or pending work).
	// Fail closed: it never reads as a benign park — but it is not slandered as
	// SPINNING either, because a report/calibration cadence is legitimately idle;
	// only a throughput member trips SPINNING (the classifyWork mix axis decides).
	ProgressIdleUnproven MemberProgress = "idle-unproven"
	// ProgressUnmeasured: the verified-progress read itself was unverifiable — no
	// intent-ledger anchor, or an unreachable ledger. Treated as NO progress
	// (never as clean/advancing) but SURFACE-ONLY: it adds no debt and never
	// fabricates a measured zero, the same posture nightIssueProgress keeps when
	// no ledger exists (the shell's night issue-progress read in cmd/fak).
	ProgressUnmeasured MemberProgress = "unmeasured"
)

// ProgressRead is the evidence the shell hands [ClassifyProgress] for one TICKING
// member — all of it read from durable state through the relay seams, none of it
// narrated by the member itself.
type ProgressRead struct {
	// Baseline is the verified-progress read pinning the high-water mark the
	// window starts from. The zero value (no verified read) fails closed to
	// high-water 0 — exactly the relay.NoProgressEscape zero-value start, where a
	// first verified step counts as progress.
	Baseline relay.VerifiedProgress `json:"baseline"`
	// Now is the verified-progress read at walk time; "advanced" is judged Now
	// against the Baseline high-water with the exact Advances rule.
	Now relay.VerifiedProgress `json:"now"`
	// Idle is the durable idle evidence (invariant witness verdict + admitted
	// pending count) consulted ONLY for a non-throughput member with no advance —
	// the relay watch-goal idle rule. The zero value proves nothing, so it never
	// parks a member benign.
	Idle relay.IdleObservation `json:"idle"`
}

// ClassifyProgress folds one member's ledger-verified progress read into the
// closed progress verdict plus its closed reason token ("" when no closed reason
// binds). ticking is the LIVENESS input: SPINNING is specifically
// ticking-with-no-verified-advance, so a non-ticking (dark) member returns the
// zero verdict — its urgency is already carried by the liveness verdict, never
// double-counted here. Decision order, each step reusing the relay rule it names:
//
//  1. advanced?     relay.NoProgressEscape.Advances against the Baseline
//     high-water — ProgressAdvancing, healthy.
//  2. unverifiable? a Now read that is not ProgressVerified is treated as NO
//     progress but stays surface-only — ProgressUnmeasured, never a fabricated
//     zero, never clean.
//  3. throughput?   classifyWork decides who may trip SPINNING: only an
//     issue-drain member earns RELAY_NO_PROGRESS on zero advance.
//  4. idle?         everything else takes the relay.IdleAwareEscape park rule —
//     RELAY_IDLE_PARKED only when the invariant HOLDS with zero admitted pending
//     work; an unproven idle stays unproven (fail closed, no benign park).
func ClassifyProgress(m Member, ticking bool, read ProgressRead) (MemberProgress, string) {
	if !ticking {
		return "", ""
	}
	esc := relay.IdleAwareEscape{}
	esc.Escape.ObserveLeg(read.Baseline) // pin the window's verified high-water mark
	if esc.Escape.Advances(read.Now) {
		return ProgressAdvancing, ""
	}
	if read.Now.Verdict != relay.ProgressVerified {
		return ProgressUnmeasured, ""
	}
	if classifyWork(m) == WorkThroughput {
		return ProgressSpinning, relay.ReasonNoProgress
	}
	if out := esc.ObserveLeg(read.Now, read.Idle); out.Parked {
		return ProgressIdleParked, out.Reason
	}
	return ProgressIdleUnproven, ""
}

// --- the per-member ORPHANED-FOLLOWON verdict (issue #4957) -----------------
//
// The progress verdict (#4956) answers "is this loop producing at its OWN grain?";
// it can never answer "is anyone advancing what it produced?". A loop tick that
// emits a downstream issue — a durable relay.ArtifactIssue baton pointer ("#1234"),
// or the issue an a2achan.WorkerStatus names — reads as advancing even while that
// emitted work sits untouched: progress at the loop grain, zero progress at the
// fleet grain. The follow-on verdict sits ALONGSIDE the liveness and progress
// verdicts (it deliberately does not overload Dark/Debt/Progress): the shell joins
// the member's emitted refs against LIVE issue state — durable, re-verifiable
// ground truth, never a self-narrated field — and this package folds the pure read
// into a closed verdict, fail-closed exactly like ClassifyProgress: an unreadable
// emission is NEVER fabricated into an orphan (the missing-ledger asymmetry
// relay.ReadVerifiedProgress keeps). This leaf only WITNESSES orphaned emissions;
// the chase/close action points at the member's own front door — it never re-files
// or re-dispatches the emitted work (that is #4958's live-binding job).

// MemberFollowon is the closed per-member follow-on verdict. The zero value ""
// means the follow-on axis was not read at all (no emissions read — non-loop
// members, loops that emitted nothing): surface-only, never weighed.
type MemberFollowon string

const (
	// FollowonAdvancing: every resolved emission advanced or closed within the
	// member's cadence window — the emitted work is being carried. Clean.
	FollowonAdvancing MemberFollowon = "advancing"
	// FollowonOrphaned: at least one emitted issue is OPEN with no advance within
	// the cadence window — the member produced work nobody picks up. DEBT: it
	// carries relay.ReasonOrphanedFollowon (RELAY_ORPHANED_FOLLOWON) and ranks in
	// the debt band so the operator chases/closes the orphan.
	FollowonOrphaned MemberFollowon = "orphaned"
	// FollowonUnknown: an emission was unreadable/unresolvable (gh outage, a ref
	// this host cannot join). FAIL CLOSED, surface-only: never orphaned — an
	// orphan is never fabricated from an absence (the ProgressUnmeasured
	// asymmetry) — but never clean either.
	FollowonUnknown MemberFollowon = "unknown"
)

// FollowonEmission is one emitted downstream ref plus the LIVE issue state the
// shell resolved it against — durable open/advance ground truth, never the
// member's own narration.
type FollowonEmission struct {
	// Ref is the emitted downstream ref in relay artifact form ("#1234" — a
	// relay.ArtifactIssue pointer, or the issue an a2achan.WorkerStatus names).
	Ref string `json:"ref"`
	// Resolved is true when the live issue state behind Ref was actually read.
	// False fails the whole member's verdict closed to FollowonUnknown.
	Resolved bool `json:"resolved"`
	// Open is the live issue state (true = still open). Meaningful only when
	// Resolved.
	Open bool `json:"open,omitempty"`
	// Advanced is true when the issue advanced (was updated or closed) within
	// the member's cadence window, judged by the shell against the durable
	// updated/closed timestamps. Meaningful only when Resolved.
	Advanced bool `json:"advanced,omitempty"`
}

// FollowonRead is the evidence the shell hands [ClassifyFollowon] for one member —
// the emitted refs joined against live issue state, all of it read from durable
// state (baton artifacts + issue state), none of it narrated by the member itself.
type FollowonRead struct {
	// Emissions are the member's emitted follow-on refs with their resolved live
	// state. Empty means the axis was not read at all (surface-only).
	Emissions []FollowonEmission `json:"emissions,omitempty"`
}

// ClassifyFollowon folds one member's emitted-work read into the closed follow-on
// verdict plus its closed reason token ("" when no closed reason binds). Decision
// order, fail-closed like ClassifyProgress:
//
//  1. nothing read?  no emissions — the zero verdict: the axis was not read,
//     surface-only, never weighed (a loop that emitted nothing owes nothing HERE).
//  2. unresolvable?  ANY emission whose live state could not be read fails the
//     whole verdict closed to FollowonUnknown — surface-only, NEVER orphaned: an
//     orphan is never fabricated from an absence, the same asymmetry
//     relay.ReadVerifiedProgress keeps for a missing ledger.
//  3. all carried?   every resolved emission advanced or closed within the
//     cadence window — FollowonAdvancing, clean.
//  4. orphaned.      at least one emission is OPEN with no advance within the
//     window — FollowonOrphaned + relay.ReasonOrphanedFollowon.
func ClassifyFollowon(m Member, read FollowonRead) (MemberFollowon, string) {
	if len(read.Emissions) == 0 {
		return "", ""
	}
	orphaned := false
	for _, e := range read.Emissions {
		if !e.Resolved {
			return FollowonUnknown, ""
		}
		if e.Open && !e.Advanced {
			orphaned = true
		}
	}
	if orphaned {
		return FollowonOrphaned, relay.ReasonOrphanedFollowon
	}
	return FollowonAdvancing, ""
}

// Member is one constituent a super loop walks. Ref names the surface (scorecard
// key / loop id / garden name / super-loop name); Why is the one-line reason it
// belongs under this intent. Enter, when set, is the CONCRETE command or skill an
// operator runs to retire this member's debt (e.g. "/slop-score",
// "python tools/learning_scorecard.py --json") — it makes the worklist action
// directly enterable instead of the generic "its skill" pointer.
type Member struct {
	Kind  MemberKind `json:"kind"`
	Ref   string     `json:"ref"`
	Why   string     `json:"why"`
	Enter string     `json:"enter,omitempty"`
}

// Super is a named operator-intent super loop: an intent bound to an ordered member
// set plus the aggregate Floor below which the intent reads as satisfied.
type Super struct {
	// Name is the operator intent token, e.g. "improve-quality".
	Name string `json:"name"`
	// Title is the human one-liner shown in `list`.
	Title string `json:"title"`
	// About explains, in one sentence, what walking this intent does.
	About string `json:"about"`
	// Members are walked in order; SELECT reorders them worst-first.
	Members []Member `json:"members"`
	// Floor is the aggregate-debt threshold at or below which the intent is satisfied
	// (0 = the intent wants every member clear).
	Floor int `json:"floor"`
	// Budget is the DECLARED generation-budget envelope this intent reserves across
	// the four dimensions the budget contract names (Time/Tokens/Workers/Review;
	// docs/generation-super-loop-budgets.md). It is the TOP of the cascade — a
	// planned reservation, not a measured consumption — that [Walk] divides down into
	// a per-member allocation. A zero cap in any dimension is UNBUDGETED for that
	// dimension: the contract's "no row = hold for later-horizon work" case, surfaced
	// as a HOLD, never as an implicit unlimited grant. Priority stays debt-ordered; a
	// budget only reserves attention (the contract's Core Rule).
	Budget GenerationBudget `json:"budget"`
	// IssueTarget is a DECLARED operator headline number — the count of issues the
	// intent wants a run to progress (e.g. run-the-night's ~200-issue overnight
	// target). Like Floor and Budget it is a policy the operator states, NOT a
	// measured consumption: the walk always SURFACES it so the number is an explicit,
	// testable part of the intent rather than buried prose. When the impure shell hands
	// [Walk] a LIVE progress count (via [WithIssueProgress]) the walk also BINDS it: a
	// shortfall (progressed < target) is folded as a gate that keeps the intent
	// unsatisfied, so the headline stops being decorative and becomes a witnessed
	// number. With no progress handed in it stays surface-only (declared, never gating)
	// — the same declared-vs-measured posture the budget rows keep. 0 = no headline
	// issue target for this intent.
	IssueTarget int `json:"issue_target,omitempty"`
	// ThroughputTargetPct is the DECLARED soft mix target: the percentage of this
	// intent's attention the operator wants spent on throughput (issue-drain) work
	// versus gardening (quality/hygiene) work. Like Floor and IssueTarget it is a stated
	// policy, but a far GENTLER one — it never overrides worst-first urgency or debt
	// order; it only breaks a tie between two equally-urgent members of opposite class,
	// nudging a lopsided run back toward the target (#3126). 0 = unset, which takes
	// DefaultThroughputTargetPct (a balanced 50%). Values are clamped to [0,100].
	ThroughputTargetPct int `json:"throughput_target_pct,omitempty"`
}

// The four budget-dimension tokens the generation-budget contract names
// (docs/generation-super-loop-budgets.md §Budget Dimensions). They are the stable
// keys a walk's budget rows and a member's held-dimension list use.
const (
	BudgetTime    = "time"    // wall-clock window spent walking/descending (max_minutes)
	BudgetTokens  = "tokens"  // model/context spend for planning, workers, reviews (token_ceiling)
	BudgetWorkers = "workers" // concurrent agent/worker seats admitted (max_workers)
	BudgetReview  = "review"  // stronger-rung review slots consumed before promotion (review_slots)
)

// GenerationBudget is a super loop's DECLARED reservation across the four budget
// dimensions. Each cap is a PLANNED operator policy (like [Super].Floor), not a
// measured number: it says how much recurring capacity this intent reserves, so
// current work is not diluted and future-facing streams are not silently starved.
// A zero cap means UNBUDGETED for that dimension — [Walk] renders it as a HOLD row
// and lists it under each member's held dimensions.
type GenerationBudget struct {
	// Stream is the generation horizon this reservation serves (e.g. "gen/now",
	// "gen/next"), reported so a walk can say which streams received capacity and
	// which were held. Empty means the envelope is horizon-agnostic.
	Stream string `json:"stream,omitempty"`
	// MaxMinutes caps the wall-clock window (Time dimension); 0 = unbudgeted.
	MaxMinutes int `json:"max_minutes,omitempty"`
	// TokenCeiling caps model/context spend (Tokens dimension); 0 = unbudgeted.
	TokenCeiling int `json:"token_ceiling,omitempty"`
	// MaxWorkers caps concurrent worker seats (Workers dimension); 0 = unbudgeted.
	MaxWorkers int `json:"max_workers,omitempty"`
	// ReviewSlots caps stronger-rung review slots (Review dimension); 0 = unbudgeted.
	ReviewSlots int `json:"review_slots,omitempty"`
	// Expiry is the recheck date a later-horizon reservation must carry: the contract
	// requires an expiry on gen/second-next and gen/future shares so a research/design
	// budget cannot become permanent. Empty is valid only for now/near-term streams.
	Expiry string `json:"expiry,omitempty"`
}

// cap returns the declared cap for a dimension token, or 0 (unbudgeted) for an
// unknown token — the single place the four fields map to their contract keys.
func (b GenerationBudget) cap(dim string) int {
	switch dim {
	case BudgetTime:
		return b.MaxMinutes
	case BudgetTokens:
		return b.TokenCeiling
	case BudgetWorkers:
		return b.MaxWorkers
	case BudgetReview:
		return b.ReviewSlots
	default:
		return 0
	}
}

// budgetDims is the ordered dimension table the divide-down folds over: each entry
// pairs a contract dimension token with its reporting unit. Ordered Time, Tokens,
// Workers, Review to match the contract's table.
var budgetDims = []struct {
	Name string
	Unit string
}{
	{BudgetTime, "minutes"},
	{BudgetTokens, "tokens"},
	{BudgetWorkers, "workers"},
	{BudgetReview, "slots"},
}

// registry is the curated set of named super loops. It is deliberately small and
// data-only: each entry binds an operator intent to REAL existing surfaces (every
// scorecard Ref is a control-pane card key; the no-drift test enforces it), so a
// super loop can never point an operator at a member that does not exist.
//
// Each entry also declares a generation Budget: a PLANNED reservation (not a
// measured consumption) across the four contract dimensions. The numbers are
// conservative operator policy — the top of the cascade [Walk] divides down into
// per-member shares. A dimension left at zero (e.g. Review on the gen/next intents)
// is deliberately UNBUDGETED, so the walk surfaces the contract's HOLD state instead
// of pretending a cap exists.
var registry = []Super{
	{
		Name:   "improve-quality",
		Title:  "improve code & content quality",
		About:  "descend the seven-surface sweep, walk the remaining quality scorecards + the gardening loop, then enter the worst-first member to retire its debt",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 30, TokenCeiling: 200000, MaxWorkers: 2, ReviewSlots: 1},
		Members: []Member{
			// The seven quality SURFACES live in their own nested intent (below) so an
			// operator can sweep exactly them; improve-quality DESCENDS it, which keeps
			// each surface's debt counted exactly once at the tend root.
			{Kind: KindSuperloop, Ref: "sweep-surfaces", Why: "the seven quality surfaces (code, doc-appeal, agent-readiness, code-slop, concept-disambiguation, learning, tooling-quality), swept worst-first"},
			{Kind: KindScorecard, Ref: "conflation", Why: "conflation debt: metrics that report an upstream value without an OBSERVED qualifier", Enter: "/conflation-score"},
			{Kind: KindScorecard, Ref: "intent_literal", Why: "intent-literal debt: code that drifts from the operator's literal intent"},
			{Kind: KindScorecard, Ref: "ui_quality", Why: "UI/UX-quality debt over the terminal render surface"},
			{Kind: KindScorecard, Ref: "claim_repro", Why: "claim-repro debt: claims a reader cannot reproduce", Enter: "/claim-repro-score"},
			{Kind: KindGarden, Ref: "garden", Why: "the gardening bundle tends the rest (scorecard ratchet, fresh-status, stale work)"},
		},
	},
	{
		// sweep-surfaces is the seven-surface sweep, one intent that walks exactly the
		// seven quality SURFACES worst-first. Every member carries an Enter hint (the
		// owning skill, or the scorecard script where no skill exists yet), so the
		// worklist's action column is directly runnable. It is NESTED under
		// improve-quality (which descends it), so the root tend fold counts each
		// surface's debt exactly once — the once-only test pins that.
		Name:   "sweep-surfaces",
		Title:  "sweep the seven quality surfaces worst-first",
		About:  "walk the seven surface scorecards (code, doc-appeal, agent-readiness, code-slop, concept-disambiguation, learning, tooling-quality), then enter the worst-first surface's reduce loop",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 25, TokenCeiling: 180000, MaxWorkers: 2, ReviewSlots: 1},
		Members: []Member{
			{Kind: KindScorecard, Ref: "code", Why: "code-quality debt: the core correctness/clarity signal", Enter: "/quality-score"},
			{Kind: KindScorecard, Ref: "appeal", Why: "doc-appeal debt: docs that read machine-written repel human readers", Enter: "/appeal-score"},
			{Kind: KindScorecard, Ref: "agent", Why: "agent-readiness friction debt: how hard an agent must fight the repo to get work done", Enter: "/agent-readiness"},
			{Kind: KindScorecard, Ref: "slop", Why: "code-slop debt is the heaviest quality drag; retire it worst-first", Enter: "/slop-score"},
			{Kind: KindScorecard, Ref: "disambiguation", Why: "concept-disambiguation debt: ambiguous concepts confuse agents and readers", Enter: "/disambiguation-score"},
			{Kind: KindScorecard, Ref: "learning", Why: "learning debt: lessons the fleet keeps re-paying instead of encoding", Enter: "python tools/learning_scorecard.py --json"},
			{Kind: KindScorecard, Ref: "tooling_quality", Why: "tooling-quality (py) debt: Python tooling not yet ported to the Go-first hygiene path", Enter: "python tools/tooling_quality_scorecard.py --json"},
		},
	},
	{
		Name:   "improve-loops",
		Title:  "improve the agentic + background loops",
		About:  "walk the loop-index scorecard + goal-scoped issue dispatch + the live loop ledgers, then enter the worst-first loop that is in debt or has gone dark",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 20, TokenCeiling: 150000, MaxWorkers: 2},
		Members: []Member{
			{Kind: KindScorecard, Ref: "loopindex", Why: "the agentic-coding loop-index: orient->plan->act->verify->ship->learn stages not yet witnessed at floor"},
			{Kind: KindScorecard, Ref: "dogfood", Why: "dogfood-loop debt: are we running our own loops, and does packet friction reach the tracker before an outsider does?", Enter: "go run ./cmd/fak dogfood-score"},
			{Kind: KindLoop, Ref: "loopmgr:recent-feature-dogfood", Why: "the recent-feature packet loop that probes fak like an outsider — dark means friction is found by outsiders, not by the loop", Enter: "make dogfood-recent && go run ./cmd/fak dogfood-issues --live"},
			{Kind: KindSuperloop, Ref: "drain-issues", Why: "issue dispatch has multiple operator goals now; descend to see throughput and high-priority loops separately"},
			{Kind: KindLoop, Ref: "cadence", Why: "the regular-cadence report loop — dark means the pacing pulse stopped"},
			{Kind: KindLoop, Ref: "dojo", Why: "the dojo gym loop — dark means calibration stopped"},
			{Kind: KindGarden, Ref: "garden", Why: "the gardening bundle surfaces orphaned/unwitnessed runs across loops"},
		},
	},
	{
		Name:   "drain-issues",
		Title:  "drain the issue backlog by operator goal",
		About:  "walk aggregate dispatch progress plus the throughput and high-priority dispatch intents, then revive the goal whose loop is darkest",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 30, TokenCeiling: 200000, MaxWorkers: 3, ReviewSlots: 1},
		Members: []Member{
			{Kind: KindLoop, Ref: "dispatch", Why: "legacy aggregate issue-resolve progress ledger; keeps the historical backlog-drain signal visible"},
			{Kind: KindSuperloop, Ref: "drain-throughput", Why: "the throughput issue-drain intent, with its own ledger and enter command"},
			{Kind: KindSuperloop, Ref: "drain-high-priority", Why: "the high-priority issue-drain intent, with its own ledger and enter command"},
		},
	},
	{
		Name:   "drain-throughput",
		Title:  "drain open issues for throughput",
		About:  "walk the named throughput dispatch loop ledger and revive the step-budget lane-pressure picker when it goes dark or stale",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 15, TokenCeiling: 100000, MaxWorkers: 2},
		Members: []Member{
			{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/throughput", Why: "the named throughput dispatch goal: move open tasks by lane pressure while preserving its own ledger identity", Enter: "go run ./cmd/fak dispatch auto --goal throughput"},
		},
	},
	{
		Name:   "drain-high-priority",
		Title:  "drain high-priority issues",
		About:  "walk the named high-priority dispatch loop ledger and revive the priority-label picker when it goes dark or stale",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 15, TokenCeiling: 100000, MaxWorkers: 2, ReviewSlots: 1},
		Members: []Member{
			{Kind: KindLoop, Ref: "loopmgr:issue-resolve-dispatch/claude/high-priority", Why: "the named high-priority dispatch goal: favor the strongest priority label while sharing the same lane/tree lease fabric", Enter: "go run ./cmd/fak dispatch auto --goal high-priority"},
		},
	},
	{
		Name:   "manage-benchmarks",
		Title:  "manage benchmark collection and publishing",
		About:  "walk benchmark DX, the nightrun collection loop, and the benchmark control surface, then enter the worst-first benchmark action",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 20, TokenCeiling: 120000, MaxWorkers: 1},
		Members: []Member{
			{Kind: KindScorecard, Ref: "bench_dx", Why: "benchmark-DX debt means the benchmark surfaces are confusing or incomplete"},
			{Kind: KindLoop, Ref: "nightrun", Why: "the local benchmark data-collection loop; dark/stale means collection throughput stalled"},
			{Kind: KindSurface, Ref: "fak bench-loop status", Why: "the benchmark domain super-loop folds registry, catalog, local next selection, request/rollup surfaces, and the authority gap"},
		},
	},
	{
		// run-the-night is the OVERNIGHT meta-loop: the intent that keeps a whole night
		// of fleet work productive by walking the three dimensions that must move
		// TOGETHER or the night wastes itself — issues drained, account limits used, and
		// lab/node silicon used. It is deliberately NOT quality/loop hygiene (that is
		// improve-quality/improve-loops): it is the "is the night actually producing?"
		// fold. Two of its members are live CAPACITY signals (KindUtilization) whose
		// debt is UNUSED headroom — an offerable-but-idle account seat or an up-but-idle
		// box is capacity the night is paying for and not spending. The third descends
		// the issue-drain intent (shared with improve-loops; a super loop may be reached
		// by two parents — each parent folds its own descended status). Worst-first, the
		// walk enters whichever dimension is most underused: if boxes sit idle it enters
		// node-resources, if seats sit idle it enters account-limits, else it drives the
		// issue backlog — the loop that carries a night toward the ~200-issue target.
		Name:        "run-the-night",
		Title:       "run a productive night: drain issues, use every account limit and every node",
		About:       "walk the three night-productivity dimensions live — open-issue drain, account-limit utilization, and lab/node resource utilization — and enter the worst-first (most underused) dimension",
		Floor:       0,
		IssueTarget: 200, // the operator's declared overnight headline: progress ~200 issues.
		Budget:      GenerationBudget{Stream: "gen/now", MaxMinutes: 60, TokenCeiling: 400000, MaxWorkers: 4, ReviewSlots: 1},
		Members: []Member{
			{Kind: KindSuperloop, Ref: "drain-issues", Why: "the night's headline output: open issues progressed toward the ~200-issue target, by throughput and high-priority goals"},
			{Kind: KindUtilization, Ref: "account-limits", Why: "account-limit utilization: offerable seats sitting idle are limits the night is paying for and not spending", Enter: "go run ./cmd/fak accounts next && go run ./cmd/fak dispatch auto --goal throughput"},
			{Kind: KindUtilization, Ref: "node-resources", Why: "lab/node resource utilization: boxes (Mac, A100s, dgx) up but idle are silicon the night is wasting", Enter: "go run ./cmd/fak lab status --all"},
		},
	},
	{
		// improve-trajectory joins the existing controller-of-controllers walk with a
		// TRAJECTORY intent (issue #2563, epic #2533): its one KindTrajectory member
		// enumerates every OPEN trajectory-control objective into a worst-first status
		// by curve signal (DETOUR_OVERRUN/DRIFT/STALL ahead of HEALTHY), so "improve
		// trajectory health" is an intent walked worst-first without growing a rival
		// walker. It holds no scorecard, so it double-counts nothing at the root fold;
		// the aggregate clears (Satisfied) exactly when every open objective reads
		// HEALTHY — an on-course fleet.
		Name:   "improve-trajectory",
		Title:  "improve trajectory health — steer open objectives worst-first",
		About:  "walk every OPEN trajectory-control objective by its curve signal (DETOUR_OVERRUN/DRIFT/STALL worst-first) and enter the objective whose trajectory is most off-course",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 20, TokenCeiling: 150000, MaxWorkers: 1},
		Members: []Member{
			{Kind: KindTrajectory, Ref: "open", Why: "every open (active|paused) trajectory objective, folded by curve signal — a DRIFT/STALL/DETOUR_OVERRUN objective is an agent going off its intended curve"},
		},
	},
	{
		// tend-scoreboards is the REPORTING-family intent: the scoreboard/feed surfaces
		// fak publishes to Slack (internal/scoreboard names the family — scoreboard,
		// blockers, bench, cachevalue, capacity, node-usage, backlog, dojo, product,
		// releases, steering — all folded onto one CI/CD report channel). None of the
		// quality/loop/night intents walk them, so "is every scoreboard number healthy?"
		// had no worst-first walk. This intent gives it one, honestly: the members that
		// carry a MEASURABLE debt are the outward-facing scorecards whose numbers get
		// posted (product, release-readiness, steerability, milestone, osp-residual) — each a real
		// control-pane card key NOT walked by another intent, so the once-only fold still
		// counts each scorecard exactly once. The feeds that have no scorecard (blockers,
		// cachevalue, capacity, node-usage, backlog) are a DELIVERY-liveness question — are
		// the channels actually receiving posts? — which the Slack-beat surface folds, so
		// it rides here as a KindSurface descend pointer (surfaced for entry, never weighed
		// or counted). The intent is SATISFIED when every posted scorecard is at floor; the
		// descend pointer keeps the feed-liveness check in view without letting an unread
		// surface red a clean walk.
		//
		// The operator-steerability overlay (#5039) joins as a KindLoop member rather than a
		// sixth intent: its number IS outward-facing steering, so tend-scoreboards is its
		// home, and a new intent walking the same debt would fragment the fold. What is
		// registered here is the overlay's LIVENESS — the maintenance loop (#5023) whose tick
		// re-folds the pending dev->release delta and appends docs/nightrun/steerpr-overlay.jsonl,
		// read through the same cross-ledger loop-health fold every other KindLoop member uses.
		// It carries NO scorecard ref, so the once-only invariant is untouched (this addition
		// double-counts nothing at the root). Its Enter is the
		// concrete verb that retires the debt the loop measures — `fak steer prs`, a read-only
		// fold that exits 0 — so the worklist's action column is runnable exactly as printed.
		// A host with no overlay ledger, or a ledger loopfleet cannot fold, reads UNMEASURED
		// (the shell's KindLoop miss path), never a clean zero: an overlay that stopped
		// ticking for a week must surface, which is the whole reason to register it.
		Name:   "tend-scoreboards",
		Title:  "tend the scoreboard/reporting surfaces",
		About:  "walk the outward-facing report scorecards fak posts to Slack (product, release, steerability, milestone, osp-residual) plus the steerability-overlay maintenance loop's liveness, surface the feed-delivery liveness, then enter the worst-first report in debt",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 15, TokenCeiling: 100000, MaxWorkers: 1},
		Members: []Member{
			{Kind: KindScorecard, Ref: "product", Why: "product-scorecard debt: the #product feed publishes a stale or incomplete product number", Enter: "go run ./cmd/fak product-scorecard --json"},
			{Kind: KindScorecard, Ref: "release", Why: "release-readiness debt: the release-cadence feed reports an unready release", Enter: "python tools/release_readiness_scorecard.py --json"},
			{Kind: KindScorecard, Ref: "steer", Why: "steerability debt: the steering-guard feed reports the fleet is hard to steer", Enter: "/steerability-score"},
			{Kind: KindScorecard, Ref: "milestone", Why: "milestone debt: the milestone feed reports the roadmap climb has stalled", Enter: "/milestone-score"},
			{Kind: KindScorecard, Ref: "osp_residual", Why: "operator-steerability residual debt (#5022): forming units in the pending dev->release delta the kernel could not witness — the pile that owes an operator a look, which without a posted number only exists in an ad-hoc command run", Enter: "go run ./cmd/fak steer prs --json"},
			{Kind: KindLoop, Ref: "steerpr-overlay", Why: "operator-steerability overlay liveness (#5039): the maintenance loop (#5023) that re-folds the pending dev->release delta as commits land — dark or unread means the residual pile an operator steers by stopped being recomputed, and an unwitnessed claim can sit unseen for a week", Enter: "fak steer prs"},
			{Kind: KindSurface, Ref: "fak slack beat", Why: "reporting-feed delivery liveness: are the scoreboard channels actually receiving the posts the feeds send (blockers, cachevalue, capacity, node-usage, backlog have no scorecard, only a delivery pulse)?"},
		},
	},
	{
		// tend-reporting is the FEEDER-LIVENESS intent (issue #4863), the counterpart of
		// tend-scoreboards. The two split the reporting family along the only line that
		// matters operationally: tend-scoreboards asks "is the posted NUMBER healthy?"
		// (scorecard debt); tend-reporting asks "is the FEEDER that posts it still alive?"
		// (loop liveness). Since #4958 it is parented under tend-fleet — a loop FAMILY is
		// exactly what the generic fleet meta-walker supervises — while tend-scoreboards
		// stays a direct root member (numbers are not loops). A feed can go dark for a week with every scorecard at floor, so the
		// number-walk cannot answer the liveness question — before this intent, nothing
		// entered the worst-first feeder to bring it back up; `fak slack beat` (the surface
		// pointer tend-scoreboards carries) only pulses DELIVERY, and is never weighed.
		//
		// The members are the feeders that fold onto the ONE CI/CD report channel
		// (scoreboard.CICDReportChannel) — the family the `fak slack check`/`fak slack walk`
		// registry names. They are KindLoop (a feeder IS a ledgered post cadence), so each
		// carries liveness debt and NO scorecard: the once-only fold is untouched (this
		// intent double-counts nothing at the root). Each ref is `report:<surface>` — the
		// FEED, deliberately distinct from any same-named underlying loop ("dojo" the gym
		// loop under improve-loops is not "report:dojo" the rollup feed), so two intents can
		// never fold one ref's debt twice. Every ref is WorkNeutral by classifyWork (a report
		// cadence is neither gardening nor backlog-drain), which is the correct benign class
		// for an idle report feed.
		//
		// Each Enter is the ONE uniformly-runnable revival verb — `fak slack refresh
		// --surface <name>` (cmd/fak/slack_refresh.go) — rather than each feeder's raw post
		// command: the raw commands are not uniformly runnable as printed (blockers/backlog
		// need a GitHub payload, scoreboard needs a KPI), and a FrontRunnable Enter promises
		// a headless drive can execute it. The refresh verb keeps that promise for every row.
		//
		// TODO(#4865): swap this inline roster for the canonical roster source once it lands;
		// until then this list is the hand-kept mirror of the CICDReportChannel family, and
		// TestTendReportingMirrorsTheReportChannelFamily pins it against drift.
		Name:   "tend-reporting",
		Title:  "tend the reporting-family feeders (are they still posting?)",
		About:  "walk the Slack feeders that fold onto the one CI/CD report channel by LIVENESS — darkest/most-overdue first — and enter the worst feeder to revive it; folds only when every feeder is confirmed up",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 15, TokenCeiling: 80000, MaxWorkers: 1},
		Members: []Member{
			{Kind: KindLoop, Ref: "report:product", Why: "the product feed: dark means product direction / persona findings stopped reaching the report channel", Enter: "fak slack refresh --surface product"},
			{Kind: KindLoop, Ref: "report:blockers", Why: "the blockers feed: dark means a fleet blocker can stand unpaged — the feed an operator trusts to interrupt them", Enter: "fak slack refresh --surface blockers"},
			{Kind: KindLoop, Ref: "report:cachevalue", Why: "the cache-value P&L feed: dark means the witnessed kernel-reuse trend stops being published", Enter: "fak slack refresh --surface cachevalue"},
			{Kind: KindLoop, Ref: "report:bench", Why: "the bench feed: dark means benchmark rollups / run-requests stop reaching the bench nodes", Enter: "fak slack refresh --surface bench"},
			{Kind: KindLoop, Ref: "report:dojo", Why: "the dojo rollup feed: dark means calibration trends stop being posted (distinct from the dojo gym loop itself)", Enter: "fak slack refresh --surface dojo"},
			{Kind: KindLoop, Ref: "report:backlog", Why: "the backlog feed: dark means the issue-triage + bottleneck digest stops surfacing", Enter: "fak slack refresh --surface backlog --backlog-channel " + scoreboard.CICDReportChannel},
			{Kind: KindLoop, Ref: "report:node-usage", Why: "the node-usage feed: dark means compute-node usage snapshots stop, and idle silicon goes unnoticed", Enter: "fak slack refresh --surface node-usage"},
			{Kind: KindLoop, Ref: "report:steering", Why: "the steering feed: dark means the steering-guard surface stops reporting how steerable the fleet is", Enter: "fak slack refresh --surface steering"},
		},
	},
	{
		// tend-fleet is the GENERIC operator meta-walker (issue #4958): ONE intent that
		// walks the whole supervised loop fleet worst-first on the PRODUCT of the three
		// per-loop dimensions — liveness (dark/stale), verified progress (SPINNING,
		// #4956), and follow-on (ORPHANED, #4957) — so a loop that is up-but-not-working
		// or emitting work nobody advances can no longer hide behind a live cadence.
		//
		// Its first member is the KindLoopFleet enumeration over the canonical roster
		// (#4955): every ledgered loop becomes one status ranked by fleetwalk.go's
		// product debt, and every unfoldable ledger surfaces as an UNMEASURED known gap
		// that blocks Satisfied. Its second member is the reporting family (#4862)
		// re-parented here as ONE KindSuperloop child — the feeder loops are exactly a
		// loop family the meta-walker supervises, read through the identical
		// SubwalkStatus fold every other descended intent gets (no special claim path,
		// no new field). Driving the worst member goes through `fak superloop drive
		// tend-fleet` / `fak superloop fleet run` — the SAME region-admission gate any
		// spawn passes; the meta-walker holds no private spawn path.
		Name:   "tend-fleet",
		Title:  "meta-walk the loop fleet — worst-first on liveness × progress × follow-on",
		About:  "enumerate every ledgered loop from the canonical roster and rank worst-first on the product of the three loop dimensions (dark/stale × spinning × orphaned), then enter the worst member through its own front door",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/next", MaxMinutes: 20, TokenCeiling: 150000, MaxWorkers: 2},
		Members: []Member{
			{Kind: KindLoopFleet, Ref: "all", Why: "every ledgered loop the operator supervises (#4955), each counted once — a spinning or orphaned loop out-ranks a clean live leaf by the product, and a skipped ledger surfaces as a known gap"},
			{Kind: KindSuperloop, Ref: "tend-reporting", Why: "the reporting-family feeders (#4862) as ONE child of the fleet meta-walk: a loop family supervised through the identical descend fold, so a dark feeder surfaces exactly like any other fleet loop"},
		},
	},
	{
		// tend is the ROOT intent — the recursion case made real: a super loop whose
		// every member is itself a super loop. Walking it descends each registered
		// intent (the shell walks sub-super-loops inline and folds each sub-report via
		// [SubwalkStatus]), so one command answers "across every operator intent, what
		// is worst right now?". The no-drift test pins that every other registered
		// intent is REACHABLE from here over KindSuperloop edges (directly a member,
		// or nested like sweep-surfaces under improve-quality), so a new intent cannot
		// silently escape the root — while a nested intent is still counted only once.
		Name:   "tend",
		Title:  "tend every operator intent — the super loop of super loops",
		About:  "walk every registered super loop (directly or by descent), and enter the worst-first intent",
		Floor:  0,
		Budget: GenerationBudget{Stream: "gen/now", MaxMinutes: 60, TokenCeiling: 400000, MaxWorkers: 4, ReviewSlots: 1},
		Members: []Member{
			{Kind: KindSuperloop, Ref: "improve-quality", Why: "quality debt is the broadest drag on everything the fleet ships"},
			{Kind: KindSuperloop, Ref: "improve-loops", Why: "the loops keep everything else tended; a dark loop below starves the rest"},
			{Kind: KindSuperloop, Ref: "improve-trajectory", Why: "trajectory health: are the steered objectives on-course, or drifting/stalling off their intended curve?"},
			{Kind: KindSuperloop, Ref: "manage-benchmarks", Why: "benchmark collection feeds the outward-facing numbers"},
			{Kind: KindSuperloop, Ref: "tend-scoreboards", Why: "the reporting family: are the scoreboard numbers fak posts to Slack (product, release, steerability, milestone, osp-residual) healthy, and are the feeds delivering?"},
			{Kind: KindSuperloop, Ref: "tend-fleet", Why: "the generic fleet meta-walker (#4958): every ledgered loop worst-first on liveness × progress × follow-on, with the reporting-family feeders (#4862) re-parented beneath it as one child"},
			{Kind: KindSuperloop, Ref: "run-the-night", Why: "the overnight productivity meta-loop: are issues draining and is every account limit + node actually being used?"},
		},
	},
}

// Registry returns a copy of the named super loops in declaration order.
func Registry() []Super {
	out := make([]Super, len(registry))
	copy(out, registry)
	return out
}

// Lookup returns the named super loop and true, or a zero Super and false.
func Lookup(name string) (Super, bool) {
	name = strings.TrimSpace(name)
	for _, s := range registry {
		if s.Name == name {
			return s, true
		}
	}
	return Super{}, false
}

// Names returns the registered intent names in declaration order.
func Names() []string {
	out := make([]string, 0, len(registry))
	for _, s := range registry {
		out = append(out, s.Name)
	}
	return out
}

// ScorecardRefs returns the distinct scorecard keys referenced across the whole
// registry — the set the no-drift witness checks against the control-pane cards.
func ScorecardRefs() []string {
	seen := map[string]struct{}{}
	for _, s := range registry {
		for _, m := range s.Members {
			if m.Kind == KindScorecard {
				seen[m.Ref] = struct{}{}
			}
		}
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// --- the differentiation: super loop vs normal loop -------------------------

// LoopFacts is the spawn-free descriptor [Classify] judges. The shell fills it: for
// a registered super loop via [FactsFor]; for a normal ledgered loop via [LeafFacts].
// Keeping it a plain struct keeps Classify pure and the differentiation testable.
type LoopFacts struct {
	Name string `json:"name"`
	// MemberCount is how many member loops/gardens/scorecards it walks (0 = a leaf).
	MemberCount int `json:"member_count"`
	// WalksFirst: does its tick READ member status before acting?
	WalksFirst bool `json:"walks_first"`
	// SelectsWorstFirst: does it pick which member to enter by worst-first?
	SelectsWorstFirst bool `json:"selects_worst_first"`
	// ExitsOnAggregate: does it exit on folded debt<=floor (vs a single task witness)?
	ExitsOnAggregate bool `json:"exits_on_aggregate"`
	// ActsAtOwnAltitude: does it write a domain artifact itself (a leaf), rather than
	// only driving its members (an interior node)? A super loop does NOT.
	ActsAtOwnAltitude bool `json:"acts_at_own_altitude"`
}

// Property is one differentiation rung: the super-loop-satisfying value (Want), what
// this loop has (Got), and a one-line account.
type Property struct {
	Name   string `json:"name"`
	Want   bool   `json:"want"`
	Got    bool   `json:"got"`
	Holds  bool   `json:"holds"`
	Detail string `json:"detail"`
}

// Verdict is the classification: whether the loop is a super loop, and the five
// properties that decided it. A super loop satisfies ALL five; a normal loop fails
// at least one, and Reason names the first failing rung.
type Verdict struct {
	Name       string     `json:"name"`
	IsSuper    bool       `json:"is_super"`
	Properties []Property `json:"properties"`
	Reason     string     `json:"reason"`
}

// Classify judges a LoopFacts against the five differentiating properties. It is the
// executable form of "what makes a super loop a super loop":
//
//	has_members         — it walks >=1 member loop (a leaf has none)
//	walks_first         — its tick READS member status before acting
//	selects_worst_first — it picks which member to enter, worst-first
//	exits_on_aggregate  — it stops when the FOLD clears, not a single witness
//	interior_node       — it mutates nothing at its own altitude (members mutate)
//
// All five must hold for IsSuper. The check is monotone in the obvious direction: a
// loop that does any of these is "more super" than one that does none.
func Classify(f LoopFacts) Verdict {
	props := []Property{
		{
			Name: "has_members", Want: true, Got: f.MemberCount > 0,
			Detail: fmt.Sprintf("walks %d member loop(s); a normal loop has none", f.MemberCount),
		},
		{
			Name: "walks_first", Want: true, Got: f.WalksFirst,
			Detail: "its tick READS each member's status before acting (orient-over-loops)",
		},
		{
			Name: "selects_worst_first", Want: true, Got: f.SelectsWorstFirst,
			Detail: "it selects which member to enter, worst-first (a normal loop just runs its body)",
		},
		{
			Name: "exits_on_aggregate", Want: true, Got: f.ExitsOnAggregate,
			Detail: "it exits on the folded aggregate clearing, not on a single task's witness",
		},
		{
			Name: "interior_node", Want: true, Got: !f.ActsAtOwnAltitude,
			Detail: "it mutates nothing at its own altitude — only its members mutate the world",
		},
	}
	is := true
	reason := "all five super-loop properties hold: it walks members, reads them first, selects worst-first, exits on the aggregate, and acts only through its members"
	for i := range props {
		props[i].Holds = props[i].Got == props[i].Want
		if !props[i].Holds && is {
			is = false
			reason = fmt.Sprintf("not a super loop: %s does not hold — %s", props[i].Name, props[i].Detail)
		}
	}
	return Verdict{Name: f.Name, IsSuper: is, Properties: props, Reason: reason}
}

// FactsFor returns the LoopFacts of a registered super loop. By construction a
// registered Super satisfies all five properties (the registry only holds intents
// the walk treats as super loops): it has members, the walk reads them first, the
// fold selects worst-first, the Floor is an aggregate exit, and it drives members
// rather than acting itself.
func FactsFor(s Super) LoopFacts {
	return LoopFacts{
		Name:              s.Name,
		MemberCount:       len(s.Members),
		WalksFirst:        true,
		SelectsWorstFirst: true,
		ExitsOnAggregate:  true,
		ActsAtOwnAltitude: false,
	}
}

// LeafFacts returns the LoopFacts of a NORMAL leaf loop (a task loop like the
// dispatch tick or a scorecard run): it has no members, no read-first phase, no
// selection, exits on its own witness, and acts directly. Used by `explain` to show
// the contrast Classify draws.
func LeafFacts(name string) LoopFacts {
	return LoopFacts{
		Name:              name,
		MemberCount:       0,
		WalksFirst:        false,
		SelectsWorstFirst: false,
		ExitsOnAggregate:  false,
		ActsAtOwnAltitude: true,
	}
}

// --- the walk: read member status, fold worst-first -------------------------

// MemberStatus is the status the shell read for one member. Debt is the member's
// debt in its own units (scorecard debt, or a dark/stale penalty for a loop). Dark
// marks a member loop gone quiet or a member that errored. Measured is false when
// the member's status could not be read — surfaced, never silently treated as clean.
//
// Container marks a member whose status this walk did NOT read — a descend pointer,
// not a leaf to weigh. A container is always surfaced in the worklist as a "descend"
// item, but it is NOT counted toward aggregate debt or the measured/unmeasured tally
// and never blocks Satisfied: this walk did not claim to have read it, so it neither
// inflates the debt nor is slandered as an unreadable failure. In practice only a
// garden or an external surface stays a container: a registered sub-super-loop is
// DESCENDED inline by the shell (recursion) and arrives here as a measured leaf via
// [SubwalkStatus].
type MemberStatus struct {
	Member    Member `json:"member"`
	Debt      int    `json:"debt"`
	Dark      bool   `json:"dark"`
	Measured  bool   `json:"measured"`
	Container bool   `json:"container"`
	Detail    string `json:"detail"`
	// Progress is the per-member PROGRESS verdict (issue #4956) sitting ALONGSIDE
	// the liveness verdict: spinning / advancing / idle-parked / idle-unproven /
	// unmeasured, from [ClassifyProgress]. "" means the progress axis was not read
	// (non-loop members, dark members) — surface-only, never weighed.
	Progress MemberProgress `json:"progress,omitempty"`
	// ProgressReason is the closed reason token bound to Progress —
	// relay.ReasonNoProgress on spinning, relay.ReasonIdleParked on a proven
	// benign park, "" otherwise — so the verdict is checkable against
	// dos_check_reason, never free text.
	ProgressReason string `json:"progress_reason,omitempty"`
	// FollowOn is the per-member FOLLOW-ON verdict (issue #4957) sitting alongside
	// liveness and progress: orphaned / advancing / unknown, from [ClassifyFollowon]
	// over the shell's emitted-refs-vs-live-issue-state read. "" means the
	// follow-on axis was not read — surface-only, never weighed (see fleetwalk.go
	// for how the meta-walk folds it into the worst-first product, #4958).
	FollowOn MemberFollowon `json:"follow_on,omitempty"`
	// FollowOnReason is the closed reason token bound to FollowOn —
	// relay.ReasonOrphanedFollowon on orphaned, "" otherwise — so the verdict is
	// checkable against dos_check_reason, never free text (the ProgressReason
	// discipline, #4957).
	FollowOnReason string `json:"follow_on_reason,omitempty"`
	// Subwalk carries the full folded report summary of a descended sub-super-loop
	// (KindSuperloop), preserving its roll-up metrics, leaf counts, and denominator
	// so the parent walk can compute a true aggregate roll-up. Nil for non-superloop members.
	Subwalk *SubwalkSummary `json:"subwalk,omitempty"`
}

// SubwalkSummary is the summary of a descended sub-super-loop walk preserved by
// [SubwalkStatus] on a [MemberStatus]. It records the sub-walk's verdict, debt,
// and both its direct and recursive leaf denominator metrics so the parent walk can
// fold a true aggregate roll-up across the entire tree.
type SubwalkSummary struct {
	Intent           string         `json:"intent"`
	Title            string         `json:"title,omitempty"`
	Verdict          string         `json:"verdict"`
	Finding          string         `json:"finding"`
	Satisfied        bool           `json:"satisfied"`
	TotalDebt        int            `json:"total_debt"`
	Floor            int            `json:"floor"`
	Members          int            `json:"members"`
	Walked           int            `json:"walked"`
	Unmeasured       int            `json:"unmeasured"`
	Dark             int            `json:"dark"`
	Spinning         int            `json:"spinning,omitempty"`
	Orphaned         int            `json:"orphaned,omitempty"`
	IssueTarget      int            `json:"issue_target,omitempty"`
	IssueProgressed  int            `json:"issue_progressed,omitempty"`
	IssueShortfall   int            `json:"issue_shortfall,omitempty"`
	Rollup           RollupSummary  `json:"rollup"`
	LeafStatuses     []MemberStatus `json:"leaf_statuses,omitempty"`
	DescendedIntents []string       `json:"descended_intents,omitempty"`
}

// RollupSummary is the hierarchical aggregate roll-up across this intent and all
// descended sub-super-loops. It provides the true leaf-level denominator and
// health roll-up, deduplicating leaves across shared sub-super-loops so each
// distinct surface in the fleet is counted exactly once.
type RollupSummary struct {
	// Intents is the count of super-loop intents in this roll-up hierarchy
	// (1 for a leaf intent with no sub-super-loops; >1 when descending).
	Intents int `json:"intents"`

	// DescendedIntents lists the names of every super loop intent included in this
	// roll-up (sorted, deduplicated).
	DescendedIntents []string `json:"descended_intents,omitempty"`

	// LeafMembers is the true leaf denominator: total distinct leaf surfaces/members
	// evaluated across the whole hierarchy. By conservation:
	// LeafMembers == Walked + Unmeasured.
	LeafMembers int `json:"leaf_members"`
	Walked      int `json:"walked"`
	Unmeasured  int `json:"unmeasured"`
	Dark        int `json:"dark"`
	Spinning    int `json:"spinning,omitempty"`
	Orphaned    int `json:"orphaned,omitempty"`
	Containers  int `json:"containers,omitempty"`

	// Debt & Target
	TotalDebt       int `json:"total_debt"`
	Floor           int `json:"floor"`
	IssueTarget     int `json:"issue_target,omitempty"`
	IssueProgressed int `json:"issue_progressed,omitempty"`
	IssueShortfall  int `json:"issue_shortfall,omitempty"`

	// Satisfied is true iff the entire rolled-up tree is satisfied:
	// no unmeasured leaves, no dark loops, no spinning, no orphaned,
	// debt <= floor, and no issue shortfall.
	Satisfied bool `json:"satisfied"`
}

// WorkItem is one worst-first entry in the walk's plan: enter this member next, and
// why. Rank is 1-based in worst-first order. Container carries the status's descend-
// pointer bit through to the renderer, so an unread container shows "→" while a
// descended sub-super-loop shows its real folded debt.
type WorkItem struct {
	Rank      int    `json:"rank"`
	Member    Member `json:"member"`
	Debt      int    `json:"debt"`
	Dark      bool   `json:"dark"`
	Container bool   `json:"container"`
	// Progress / ProgressReason carry the member's progress verdict (#4956)
	// through to the rendered worklist, so a SPINNING entry names its closed
	// reason (RELAY_NO_PROGRESS) machine-readably alongside the revive action.
	Progress       MemberProgress `json:"progress,omitempty"`
	ProgressReason string         `json:"progress_reason,omitempty"`
	// FollowOn / FollowOnReason carry the member's follow-on verdict (#4957) the
	// same way, so an ORPHANED-FOLLOWON entry names its closed reason
	// (RELAY_ORPHANED_FOLLOWON) machine-readably alongside its chase/redirect
	// action.
	FollowOn       MemberFollowon `json:"follow_on,omitempty"`
	FollowOnReason string         `json:"follow_on_reason,omitempty"`
	Action         string         `json:"action"`
	Detail         string         `json:"detail"`
	// Allocation is this member's divided share of the intent's declared budget —
	// the top-of-cascade input the drive rung binds to the member's budget.* / cap
	// env when it enters. It is a reservation, not an enforcement: no cap is applied
	// here (see [Allocation]).
	Allocation Allocation `json:"allocation"`
}

// Allocation is the budget share one worklist member receives from the divide-down.
// Under even division each worklist member gets the same share, so a per-dimension
// cap is the intent's declared cap divided (floor) by the number of worklist
// members. Held names the dimensions the intent left UNBUDGETED (zero cap) — the
// contract's "no row = hold for later-horizon work" list, carried explicitly so a
// reader (and the future bind) can tell "0 share of a budgeted dimension" (timeshare
// a scarce cap) apart from "this dimension is not budgeted at all" (hold).
type Allocation struct {
	MaxMinutes   int      `json:"max_minutes,omitempty"`
	TokenCeiling int      `json:"token_ceiling,omitempty"`
	MaxWorkers   int      `json:"max_workers,omitempty"`
	ReviewSlots  int      `json:"review_slots,omitempty"`
	Held         []string `json:"held,omitempty"`
}

// BudgetRow is one budget dimension folded for the whole walk: the intent's declared
// cap (Total), whether the dimension is budgeted at all, and the PerMember share the
// divide-down hands each worklist member. An unbudgeted dimension (Budgeted false)
// is a HOLD — zero total, zero share, and a Hold reason — surfaced for the operator
// rather than silently treated as an unlimited grant.
type BudgetRow struct {
	Dimension string `json:"dimension"`
	Unit      string `json:"unit"`
	Stream    string `json:"stream,omitempty"`
	Budgeted  bool   `json:"budgeted"`
	Total     int    `json:"total"`
	Members   int    `json:"members"`
	PerMember int    `json:"per_member"`
	Hold      string `json:"hold,omitempty"`
}

// WalkReport is the folded intent-level verdict + the worst-first worklist: the
// answer to "I asked to <intent> — what is the status of everything under it, and
// what should I enter first?"
type WalkReport struct {
	Schema    string `json:"schema"`
	Name      string `json:"name"`
	Title     string `json:"title"`
	TotalDebt int    `json:"total_debt"`
	Floor     int    `json:"floor"`
	// IssueTarget echoes the intent's DECLARED headline issue count (0 = none). It is
	// always surfaced for the operator; when a live progress count is handed in (see
	// IssueProgressed) it is also the target the shortfall gate measures against.
	IssueTarget int `json:"issue_target,omitempty"`
	// IssueProgressed is the LIVE count of issues the intent has progressed this run, as
	// measured by the impure shell and handed to Walk via WithIssueProgress. It is only
	// meaningful when IssueProgressMeasured is true; the pure package reads no ledger, so
	// an unmeasured walk leaves this 0 and never fabricates progress it did not witness.
	IssueProgressed int `json:"issue_progressed,omitempty"`
	// IssueProgressMeasured records whether a live progress count was handed in at all.
	// It disambiguates a real measured zero (nothing progressed yet — a shortfall) from
	// "not measured" (surface-only, never gating), so an unread issue layer cannot make
	// the night falsely read as having hit its headline.
	IssueProgressMeasured bool `json:"issue_progress_measured,omitempty"`
	// IssueShortfall is max(0, IssueTarget - IssueProgressed) when the intent declares a
	// target AND progress was measured: the issues still owed against the headline. A
	// positive shortfall keeps Satisfied false — the declared target is a gate, not a
	// decoration. 0 when there is no target, no measurement, or the target is met.
	IssueShortfall int  `json:"issue_shortfall,omitempty"`
	Satisfied      bool `json:"satisfied"`
	// Members is the evaluated direct member count (the direct denominator).
	// When template members expand (KindLoopFleet, KindTrajectory), it reflects
	// the expanded candidate count so Walked + Unmeasured + Containers == Members.
	Members int `json:"members"`
	// DeclaredMembers is the count of declared member templates on the intent (len(s.Members)).
	DeclaredMembers int `json:"declared_members,omitempty"`
	Walked          int `json:"walked"`
	Unmeasured      int `json:"unmeasured"`
	Dark            int `json:"dark"`
	// Spinning counts measured members whose progress verdict is SPINNING —
	// ticking on cadence with zero advanced verified progress (#4956). Like Dark
	// it blocks Satisfied: a fleet that fires without producing is not tended.
	Spinning int `json:"spinning,omitempty"`
	// Orphaned counts measured members whose follow-on verdict is ORPHANED —
	// emitting work nobody advances (#4957). Like Dark and Spinning it blocks
	// Satisfied: a loop whose output piles up unowned is not tended.
	Orphaned int `json:"orphaned,omitempty"`
	// Containers is the count of unread container / descend-pointer members at this direct level.
	Containers int `json:"containers,omitempty"`

	// Rollup is the hierarchical aggregate roll-up across this intent and all descended
	// sub-super-loops. It provides the true leaf-level denominator and health roll-up.
	Rollup RollupSummary `json:"rollup"`

	// LeafStatuses contains the distinct evaluated leaf member statuses across the
	// roll-up, deduplicated by member key.
	LeafStatuses []MemberStatus `json:"leaf_statuses,omitempty"`

	Worklist []WorkItem     `json:"worklist"`
	Statuses []MemberStatus `json:"statuses"`
	// Budget is the intent's declared generation-budget envelope folded into one row
	// per contract dimension (Time/Tokens/Workers/Review), each carrying the declared
	// cap and the per-worklist-member share. Always four rows: an unbudgeted dimension
	// renders as a HOLD row, which is itself the contract's operator warning.
	Budget []BudgetRow `json:"budget"`
	// Mix is the gardening-vs-throughput split of the worklist — how much of what this
	// walk says to enter is quality-tending versus backlog-draining, and which way the
	// soft target-ratio tie-break leaned (#3126). Always present; a walk with no work
	// (an empty worklist) reports all-zero counts and no favor.
	Mix        WorkMix `json:"mix"`
	Verdict    string  `json:"verdict"`
	Finding    string  `json:"finding"`
	Reason     string  `json:"reason"`
	NextAction string  `json:"next_action"`
}

// WorkMix is the folded gardening-vs-throughput split of a walk's worklist. The three
// counts tally the worklist members by [WorkClass]; TargetThroughputPct echoes the
// resolved soft target (declared or default); Favor names the class the tie-break
// nudged toward this walk, or "" when the mix is already on target or one side is
// absent (a tie-break needs BOTH classes present to have anything to trade).
type WorkMix struct {
	Gardening           int       `json:"gardening"`
	Throughput          int       `json:"throughput"`
	Neutral             int       `json:"neutral"`
	TargetThroughputPct int       `json:"target_throughput_pct"`
	Favor               WorkClass `json:"favor,omitempty"`
}

// resolveThroughputTarget clamps a Super's declared soft mix target into [0,100],
// substituting the balanced default when the intent declares none (0). Pure and total.
func resolveThroughputTarget(pct int) int {
	if pct <= 0 {
		return DefaultThroughputTargetPct
	}
	if pct > 100 {
		return 100
	}
	return pct
}

// favoredClass decides which class a soft tie-break should lean toward, given the
// worklist's current gardening/throughput counts and the resolved target throughput
// share. It returns "" (no lean) unless BOTH classes are present — with only one class
// on the worklist there is no tie to trade — and unless the measured share actually
// differs from the target. When throughput is under its target share the walk leans
// throughput; when it is over, it leans gardening. Neutral members never tip it.
func favoredClass(gardening, throughput, targetPct int) WorkClass {
	if gardening == 0 || throughput == 0 {
		return ""
	}
	sharePct := throughput * 100 / (gardening + throughput)
	switch {
	case sharePct < targetPct:
		return WorkThroughput
	case sharePct > targetPct:
		return WorkGardening
	default:
		return ""
	}
}

// WalkOpt configures a Walk beyond the declared registry data — the seam the impure
// shell uses to fold LIVE measurements (which the pure package cannot read: it reads no
// ledger, disk, or clock) into the verdict. Options apply in order; a nil option is
// ignored. It is the same shell-measures / core-folds split the member statuses use.
type WalkOpt func(*walkConfig)

// walkConfig accumulates the live measurements the WalkOpts hand in.
type walkConfig struct {
	issueProgressMeasured bool
	issueProgressed       int
}

// WithIssueProgress hands Walk the LIVE count of issues the intent has progressed this
// run, as measured by the shell. When the intent declares an IssueTarget (> 0) the walk
// folds this against it: a shortfall (progressed < target) becomes a gate that keeps the
// intent unsatisfied, turning the declared headline into a witnessed number instead of
// decorative prose. A negative count is clamped to 0. Handing progress in for an intent
// with no target is harmless (surfaced, never gating). Passing it at all marks the walk
// as having MEASURED progress, so a real zero reads as a shortfall, not as "unmeasured".
func WithIssueProgress(progressed int) WalkOpt {
	return func(c *walkConfig) {
		if progressed < 0 {
			progressed = 0
		}
		c.issueProgressMeasured = true
		c.issueProgressed = progressed
	}
}

// Walk folds the member statuses the shell read into the intent-level verdict and a
// worst-first worklist. Ordering (the SELECT step): dark/unmeasured members first
// (a gone-dark loop or an unreadable member is the most urgent thing to enter), then
// by debt descending, ties broken by the member's declared order (stable). The
// intent is SATISFIED only when total debt is at-or-below Floor AND every member was
// measured AND none is dark AND none is SPINNING (#4956: ticking with zero advanced
// verified progress) AND any declared issue-target with measured progress is met — an
// unread, dark, or spinning member, or an unmet headline, can never read as clean. Live
// measurements the pure package cannot read (e.g. issue progress) arrive via WalkOpts.
func Walk(s Super, statuses []MemberStatus, opts ...WalkOpt) WalkReport {
	var cfg walkConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	rep := WalkReport{
		Schema:                WalkSchema,
		Name:                  s.Name,
		Title:                 s.Title,
		Floor:                 s.Floor,
		IssueTarget:           s.IssueTarget,
		IssueProgressMeasured: cfg.issueProgressMeasured,
		IssueProgressed:       cfg.issueProgressed,
		DeclaredMembers:       len(s.Members),
		Members:               len(s.Members),
		Statuses:              statuses,
	}

	// Preserve declared order as the stable tiebreaker.
	order := map[string]int{}
	for i, m := range s.Members {
		order[memberKey(m)] = i
	}

	ranked := append([]MemberStatus(nil), statuses...)
	for _, st := range statuses {
		if st.Container {
			rep.Containers++
			continue // a descend-pointer: not weighed, not counted (see MemberStatus).
		}
		if st.Measured {
			rep.Walked++
			rep.TotalDebt += st.Debt
		} else {
			rep.Unmeasured++
		}
		if st.Dark {
			rep.Dark++
		}
		if st.Progress == ProgressSpinning {
			rep.Spinning++
		}
		if st.FollowOn == FollowonOrphaned {
			rep.Orphaned++
		}
	}

	// When template members expand (e.g. KindLoopFleet:all or KindTrajectory:open),
	// the evaluated direct candidate denominator is Walked + Unmeasured + Containers.
	if rep.Walked+rep.Unmeasured > rep.Members {
		rep.Members = rep.Walked + rep.Unmeasured + rep.Containers
	}

	descendedIntents := map[string]bool{s.Name: true}
	leafMap := make(map[string]MemberStatus)
	var rollupContainers int
	var subwalkShortfall int
	var subwalkTarget int
	var subwalkProgressed int

	for _, st := range statuses {
		if st.Container {
			rollupContainers++
			continue
		}
		if st.Subwalk != nil {
			descendedIntents[st.Subwalk.Intent] = true
			for _, in := range st.Subwalk.DescendedIntents {
				descendedIntents[in] = true
			}
			subwalkShortfall += st.Subwalk.IssueShortfall
			subwalkTarget += st.Subwalk.IssueTarget
			subwalkProgressed += st.Subwalk.IssueProgressed
			rollupContainers += st.Subwalk.Rollup.Containers

			if len(st.Subwalk.LeafStatuses) > 0 {
				for _, ls := range st.Subwalk.LeafStatuses {
					k := memberKey(ls.Member)
					if _, exists := leafMap[k]; !exists {
						leafMap[k] = ls
					}
				}
			} else {
				k := memberKey(st.Member)
				if _, exists := leafMap[k]; !exists {
					synth := st
					if st.Subwalk.Unmeasured > 0 {
						synth.Measured = false
					}
					if st.Subwalk.Dark > 0 {
						synth.Dark = true
					}
					leafMap[k] = synth
				}
			}
		} else {
			k := memberKey(st.Member)
			if _, exists := leafMap[k]; !exists {
				leafMap[k] = st
			}
		}
	}

	rep.Rollup.Intents = len(descendedIntents)
	rep.Rollup.Floor = s.Floor
	rep.Rollup.Containers = rollupContainers

	descendedList := make([]string, 0, len(descendedIntents))
	for in := range descendedIntents {
		descendedList = append(descendedList, in)
	}
	sort.Strings(descendedList)
	rep.Rollup.DescendedIntents = descendedList

	keys := make([]string, 0, len(leafMap))
	for k := range leafMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		rep.LeafStatuses = append(rep.LeafStatuses, leafMap[k])
	}

	if rep.Rollup.Intents == 1 {
		rep.Rollup.LeafMembers = rep.Walked + rep.Unmeasured
		rep.Rollup.Walked = rep.Walked
		rep.Rollup.Unmeasured = rep.Unmeasured
		rep.Rollup.Dark = rep.Dark
		rep.Rollup.Spinning = rep.Spinning
		rep.Rollup.Orphaned = rep.Orphaned
		rep.Rollup.TotalDebt = rep.TotalDebt
		rep.Rollup.IssueTarget = rep.IssueTarget
		rep.Rollup.IssueProgressed = rep.IssueProgressed
		rep.Rollup.IssueShortfall = rep.IssueShortfall
		rep.Rollup.Satisfied = rep.Unmeasured == 0 && rep.Dark == 0 && rep.Spinning == 0 && rep.Orphaned == 0 && rep.TotalDebt <= s.Floor && rep.IssueShortfall == 0
	} else {
		for _, ls := range rep.LeafStatuses {
			if ls.Measured {
				rep.Rollup.Walked++
				rep.Rollup.TotalDebt += ls.Debt
			} else {
				rep.Rollup.Unmeasured++
			}
			if ls.Dark {
				rep.Rollup.Dark++
			}
			if ls.Progress == ProgressSpinning {
				rep.Rollup.Spinning++
			}
			if ls.FollowOn == FollowonOrphaned {
				rep.Rollup.Orphaned++
			}
		}
		rep.Rollup.LeafMembers = rep.Rollup.Walked + rep.Rollup.Unmeasured
		rep.Rollup.IssueTarget = rep.IssueTarget + subwalkTarget
		rep.Rollup.IssueProgressed = rep.IssueProgressed + subwalkProgressed
		rep.Rollup.IssueShortfall = rep.IssueShortfall + subwalkShortfall
		rep.Rollup.TotalDebt += rep.Rollup.IssueShortfall
		rep.Rollup.Satisfied = rep.Rollup.Unmeasured == 0 && rep.Rollup.Dark == 0 && rep.Rollup.Spinning == 0 && rep.Rollup.Orphaned == 0 && rep.Rollup.TotalDebt <= s.Floor && rep.Rollup.IssueShortfall == 0
	}

	// Classify the worklist-bound members into the gardening/throughput/neutral mix and
	// resolve the soft target. favor is computed ONCE over the whole candidate set, so
	// the tie-break below is a stable global lean toward rebalancing the mix rather than
	// a per-pair ratchet that could depend on comparison order.
	targetPct := resolveThroughputTarget(s.ThroughputTargetPct)
	var gardening, throughput, neutral int
	for _, st := range statuses {
		if !workEligible(st) {
			continue
		}
		switch classifyWork(st.Member) {
		case WorkGardening:
			gardening++
		case WorkThroughput:
			throughput++
		default:
			neutral++
		}
	}
	favor := favoredClass(gardening, throughput, targetPct)
	rep.Mix = WorkMix{
		Gardening:           gardening,
		Throughput:          throughput,
		Neutral:             neutral,
		TargetThroughputPct: targetPct,
		Favor:               favor,
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		// urgency tier (low number = earlier): a dark/unmeasured leaf is most urgent,
		// then debt-bearing leaves, then descend-pointers, then clean.
		ti, tj := tier(ranked[i]), tier(ranked[j])
		if ti != tj {
			return ti < tj
		}
		if ranked[i].Debt != ranked[j].Debt {
			return ranked[i].Debt > ranked[j].Debt
		}
		// Soft mix tie-break (#3126): only reached when two members share urgency AND
		// debt — a genuine tie the code above leaves to declared order. When the walk is
		// leaning toward a class to rebalance the mix, a member of the favored class
		// sorts ahead of one that is not; otherwise declared order still decides. This
		// can never move a member past one that is more urgent or carries more debt, so
		// it is a nudge, not an override.
		if favor != "" {
			fi := classifyWork(ranked[i].Member) == favor
			fj := classifyWork(ranked[j].Member) == favor
			if fi != fj {
				return fi
			}
		}
		return order[memberKey(ranked[i].Member)] < order[memberKey(ranked[j].Member)]
	})

	for _, st := range ranked {
		// A clean, measured leaf is nothing to enter. A container is ALWAYS surfaced
		// (its status is only knowable by descending). Everything with work stays.
		if !workEligible(st) {
			continue
		}
		rep.Worklist = append(rep.Worklist, WorkItem{
			Member:         st.Member,
			Debt:           st.Debt,
			Dark:           st.Dark,
			Container:      st.Container,
			Progress:       st.Progress,
			ProgressReason: st.ProgressReason,
			FollowOn:       st.FollowOn,
			FollowOnReason: st.FollowOnReason,
			Action:         actionFor(st),
			Detail:         workDetail(st),
		})
	}
	// Re-rank the filtered worklist 1..N so the printed ranks are contiguous.
	for i := range rep.Worklist {
		rep.Worklist[i].Rank = i + 1
	}

	// Divide the declared budget down across the worklist members: each budgeted
	// dimension's cap splits evenly (floored) among the members with work, and every
	// worklist member is annotated with its share; an unbudgeted dimension is held
	// (see divideBudget). No cap is enforced HERE — this is the reservation the drive
	// rung binds when it enters (#2224). The bind has teeth for the Time dimension: the
	// `--execute` exec rung turns [Allocation.MaxMinutes] into the front-door run's real
	// deadline (superloopEffectiveTimeout, cmd/fak). The other dimensions
	// (Tokens/Workers/Review) have no headless enforcement point yet and stay reservations
	// the operator and the member's own child machinery observe.
	rows, alloc := divideBudget(s.Budget, len(rep.Worklist))
	rep.Budget = rows
	for i := range rep.Worklist {
		rep.Worklist[i].Allocation = alloc
	}

	// Fold the declared issue-target headline against the measured live progress: an
	// unmet target with progress in hand is a shortfall that gates satisfaction (the
	// number is a promise, not a decoration). Only bites when the intent DECLARES a
	// target and the shell actually MEASURED progress — a surface-only walk never gates.
	if s.IssueTarget > 0 && cfg.issueProgressMeasured && cfg.issueProgressed < s.IssueTarget {
		rep.IssueShortfall = s.IssueTarget - cfg.issueProgressed
	}

	directSatisfied := rep.Unmeasured == 0 && rep.Dark == 0 && rep.Spinning == 0 && rep.Orphaned == 0 && rep.TotalDebt <= s.Floor && rep.IssueShortfall == 0
	rep.Satisfied = directSatisfied && rep.Rollup.Satisfied
	rep.Verdict, rep.Finding, rep.Reason, rep.NextAction = walkVerdict(s, rep)
	return rep
}

// divideBudget folds a declared budget into the walk's per-dimension rows and the
// per-member allocation. n is the number of worklist members the reservation is
// divided across. Division is even and FLOORED — the sum of member shares never
// exceeds a dimension's cap — so a small integer cap spread over many members can
// floor to a zero per-member share (a budgeted-but-scarce dimension the members must
// timeshare), which is deliberately distinct from an UNBUDGETED dimension (zero cap),
// reported as a hold. When n is 0 (a satisfied walk with an empty worklist) every
// share is 0: there is nothing to enter, so nothing to reserve.
func divideBudget(b GenerationBudget, n int) ([]BudgetRow, Allocation) {
	rows := make([]BudgetRow, 0, len(budgetDims))
	var alloc Allocation
	for _, d := range budgetDims {
		total := b.cap(d.Name)
		row := BudgetRow{
			Dimension: d.Name,
			Unit:      d.Unit,
			Stream:    b.Stream,
			Budgeted:  total > 0,
			Total:     total,
			Members:   n,
		}
		if total <= 0 {
			row.Hold = "unbudgeted — held for later-horizon work; declare a " + d.Name + " cap to reserve capacity"
			alloc.Held = append(alloc.Held, d.Name)
			rows = append(rows, row)
			continue
		}
		if n > 0 {
			row.PerMember = total / n
		}
		switch d.Name {
		case BudgetTime:
			alloc.MaxMinutes = row.PerMember
		case BudgetTokens:
			alloc.TokenCeiling = row.PerMember
		case BudgetWorkers:
			alloc.MaxWorkers = row.PerMember
		case BudgetReview:
			alloc.ReviewSlots = row.PerMember
		}
		rows = append(rows, row)
	}
	return rows, alloc
}

// SubwalkStatus folds a completed sub-walk into the member status the parent walk
// weighs — the DESCEND move made real for a sub-super-loop member. The mapping is
// conservative-honest: the member is Measured (the walk actually read it, so it is
// no longer a container), it carries the sub-walk's aggregate debt PLUS any unmet
// headline shortfall, and an UNSATISFIED sub-intent can never read as clean — when its
// own measured debt is zero (unmeasured or dark members inside), it still carries one
// unit of debt at the parent's altitude. Dark propagates when any member loop below has
// gone dark, so the parent's SELECT ranks a dark subtree with leaf-dark urgency.
//
// Folding IssueShortfall into the descended debt (#3151) is what lets a large headline
// miss out-rank trivial member debt at the root: a run-the-night that missed its
// ~200-issue headline with zero member debt contributes debt 200 (not a single floored
// unit), so the root worst-first ranks the night's biggest gap ahead of a sibling
// carrying debt 2. The shortfall is 0 whenever the sub-walk declares no target or met
// it, so this is a no-op for shortfall-free subs and the 1-unit floor still guards the
// zero-shortfall unsatisfied case (dark/unmeasured members with no headline).
//
// Subwalk preserves the sub-walk's roll-up summary, leaf counts, and denominator so
// the parent walk can compute a true hierarchical aggregate roll-up without losing
// sight of the underlying population.
func SubwalkStatus(m Member, rep WalkReport) MemberStatus {
	debt := rep.TotalDebt + rep.IssueShortfall
	if !rep.Satisfied && debt <= 0 {
		debt = 1
	}
	dark := rep.Dark > 0 || rep.Rollup.Dark > 0
	var prog MemberProgress
	var progReason string
	if rep.Spinning > 0 || rep.Rollup.Spinning > 0 {
		prog = ProgressSpinning
		progReason = relay.ReasonNoProgress
	}
	var followOn MemberFollowon
	var followOnReason string
	if rep.Orphaned > 0 || rep.Rollup.Orphaned > 0 {
		followOn = FollowonOrphaned
		followOnReason = relay.ReasonOrphanedFollowon
	}

	leaves := rep.Rollup.LeafMembers
	if leaves == 0 {
		leaves = rep.Members
	}
	unm := rep.Unmeasured
	if rep.Rollup.Unmeasured > unm {
		unm = rep.Rollup.Unmeasured
	}
	darkCount := rep.Dark
	if rep.Rollup.Dark > darkCount {
		darkCount = rep.Rollup.Dark
	}

	detail := fmt.Sprintf("descended: %s (%s) — debt %d, shortfall %d, unmeasured %d, dark %d across %d member(s)",
		rep.Verdict, rep.Finding, rep.TotalDebt, rep.IssueShortfall, unm, darkCount, leaves)

	descended := make([]string, 0, len(rep.Rollup.DescendedIntents)+1)
	descended = append(descended, rep.Name)
	for _, in := range rep.Rollup.DescendedIntents {
		if in != rep.Name {
			descended = append(descended, in)
		}
	}
	sort.Strings(descended)

	sub := &SubwalkSummary{
		Intent:           rep.Name,
		Title:            rep.Title,
		Verdict:          rep.Verdict,
		Finding:          rep.Finding,
		Satisfied:        rep.Satisfied,
		TotalDebt:        rep.TotalDebt,
		Floor:            rep.Floor,
		Members:          rep.Members,
		Walked:           rep.Walked,
		Unmeasured:       rep.Unmeasured,
		Dark:             rep.Dark,
		Spinning:         rep.Spinning,
		Orphaned:         rep.Orphaned,
		IssueTarget:      rep.IssueTarget,
		IssueProgressed:  rep.IssueProgressed,
		IssueShortfall:   rep.IssueShortfall,
		Rollup:           rep.Rollup,
		LeafStatuses:     rep.LeafStatuses,
		DescendedIntents: descended,
	}

	return MemberStatus{
		Member:         m,
		Measured:       true,
		Debt:           debt,
		Dark:           dark,
		Progress:       prog,
		ProgressReason: progReason,
		FollowOn:       followOn,
		FollowOnReason: followOnReason,
		Detail:         detail,
		Subwalk:        sub,
	}
}
