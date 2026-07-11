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
func classifyWork(m Member) WorkClass {
	switch m.Kind {
	case KindScorecard, KindGarden, KindSurface:
		return WorkGardening
	case KindLoop, KindSuperloop:
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
	Action    string `json:"action"`
	Detail    string `json:"detail"`
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
	IssueShortfall int            `json:"issue_shortfall,omitempty"`
	Satisfied      bool           `json:"satisfied"`
	Members        int            `json:"members"`
	Walked         int            `json:"walked"`
	Unmeasured     int            `json:"unmeasured"`
	Dark           int            `json:"dark"`
	Worklist       []WorkItem     `json:"worklist"`
	Statuses       []MemberStatus `json:"statuses"`
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
// measured AND none is dark AND any declared issue-target with measured progress is met
// — an unread or dark member, or an unmet headline, can never read as clean. Live
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
			Member:    st.Member,
			Debt:      st.Debt,
			Dark:      st.Dark,
			Container: st.Container,
			Action:    actionFor(st),
			Detail:    workDetail(st),
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

	rep.Satisfied = rep.Unmeasured == 0 && rep.Dark == 0 && rep.TotalDebt <= s.Floor && rep.IssueShortfall == 0
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
func SubwalkStatus(m Member, rep WalkReport) MemberStatus {
	debt := rep.TotalDebt + rep.IssueShortfall
	if !rep.Satisfied && debt <= 0 {
		debt = 1
	}
	return MemberStatus{
		Member:   m,
		Measured: true,
		Debt:     debt,
		Dark:     rep.Dark > 0,
		Detail: fmt.Sprintf("descended: %s (%s) — debt %d, shortfall %d, unmeasured %d, dark %d across %d member(s)",
			rep.Verdict, rep.Finding, rep.TotalDebt, rep.IssueShortfall, rep.Unmeasured, rep.Dark, rep.Members),
	}
}

// tier ranks a member status into a worst-first band (lower = enter sooner):
//
//	0  a dark / unmeasured LEAF — its status is bad or unknown; most urgent
//	1  a measured leaf carrying debt
//	2  a container (garden / super loop) — descend to learn its status
//	3  a measured, clean, live leaf — nothing to do
//
// workEligible reports whether a status belongs on the worklist — the exact inverse of
// the clean-and-measured drop condition. A container (always surfaced for descent), an
// unmeasured or dark member, or any debt-bearing leaf is work to enter; a measured,
// clean, live leaf is not. Shared by the worklist filter and the mix pre-count so the
// two can never disagree on what counts as "work to enter".
func workEligible(st MemberStatus) bool {
	return st.Container || !st.Measured || st.Dark || st.Debt > 0
}

func tier(st MemberStatus) int {
	if st.Container {
		return 2
	}
	if st.Dark || !st.Measured {
		return 0
	}
	if st.Debt > 0 {
		return 1
	}
	return 3
}

func actionFor(st MemberStatus) string {
	switch st.Member.Kind {
	case KindScorecard:
		if !st.Measured {
			return fmt.Sprintf("run `fak scorecard` / the %s scorecard to measure it", st.Member.Ref)
		}
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			return fmt.Sprintf("enter `%s` to retire %s debt", e, st.Member.Ref)
		}
		return fmt.Sprintf("enter the %s scorecard's reduce loop (its skill) to retire debt", st.Member.Ref)
	case KindLoop:
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			if st.Dark {
				return fmt.Sprintf("revive via `%s` — %s has gone dark", e, st.Member.Ref)
			}
			return fmt.Sprintf("drive via `%s`", e)
		}
		if st.Dark {
			return fmt.Sprintf("revive the %s loop — it has gone dark", st.Member.Ref)
		}
		return fmt.Sprintf("drive the %s loop", st.Member.Ref)
	case KindGarden:
		return "run `fak garden` then `fak garden tick` to tend the bundle"
	case KindSuperloop:
		return fmt.Sprintf("descend: `fak superloop walk %s`", st.Member.Ref)
	case KindSurface:
		return fmt.Sprintf("enter `%s`", st.Member.Ref)
	case KindUtilization:
		if !st.Measured {
			return fmt.Sprintf("read the %s pool's live utilization to measure its unused headroom", st.Member.Ref)
		}
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			return fmt.Sprintf("enter `%s` to spend the idle %s capacity", e, st.Member.Ref)
		}
		return fmt.Sprintf("put the idle %s capacity to work", st.Member.Ref)
	case KindTrajectory:
		if !st.Measured {
			return fmt.Sprintf("read the trajctl ledger to fold objective %q's curve", st.Member.Ref)
		}
		if e := strings.TrimSpace(st.Member.Enter); e != "" {
			return fmt.Sprintf("enter `%s` to steer objective %s back on-course", e, st.Member.Ref)
		}
		return fmt.Sprintf("steer trajectory objective %q worst-first (`fak trajctl curve --objective %s`)", st.Member.Ref, st.Member.Ref)
	default:
		return "enter the member's loop"
	}
}

func workDetail(st MemberStatus) string {
	if st.Container {
		return "DESCEND — " + firstNonEmpty(st.Detail, st.Member.Why)
	}
	if !st.Measured {
		if strings.TrimSpace(st.Detail) != "" {
			return "UNMEASURED — " + st.Detail
		}
		return "UNMEASURED — status could not be read"
	}
	if st.Dark {
		return "DARK — " + firstNonEmpty(st.Detail, "loop has gone quiet past its cadence")
	}
	return firstNonEmpty(st.Detail, st.Member.Why)
}

func walkVerdict(s Super, rep WalkReport) (verdict, finding, reason, next string) {
	if rep.Unmeasured > 0 {
		return "ACTION", "superloop_unmeasured",
			fmt.Sprintf("walking %q: %d/%d member(s) could not be read, so the intent is not proven tended (debt %d across %d measured)",
				s.Name, rep.Unmeasured, rep.Members, rep.TotalDebt, rep.Walked),
			"repair/read the unmeasured member(s) first: " + worklistHead(rep)
	}
	if rep.Dark > 0 {
		return "ACTION", "superloop_dark",
			fmt.Sprintf("walking %q: %d member loop(s) have gone DARK; revive them before chasing debt (debt %d)", s.Name, rep.Dark, rep.TotalDebt),
			"worst-first: " + worklistHead(rep)
	}
	if rep.TotalDebt > s.Floor {
		return "ACTION", "superloop_debt",
			fmt.Sprintf("walking %q: aggregate debt %d > floor %d across %d member(s); enter the worst first", s.Name, rep.TotalDebt, s.Floor, rep.Members),
			"worst-first: " + worklistHead(rep)
	}
	if rep.IssueShortfall > 0 {
		return "ACTION", "superloop_issue_shortfall",
			fmt.Sprintf("walking %q: debt at-or-below floor, but %d/%d headline issue(s) still owed (progressed %d) — the target is a gate, not a decoration",
				s.Name, rep.IssueShortfall, rep.IssueTarget, rep.IssueProgressed),
			"progress the remaining issues: " + worklistHead(rep)
	}
	target := ""
	if rep.IssueProgressMeasured && rep.IssueTarget > 0 {
		target = fmt.Sprintf("; headline %d/%d issue(s) progressed", rep.IssueProgressed, rep.IssueTarget)
	}
	return "OK", "superloop_satisfied",
		fmt.Sprintf("walking %q: aggregate debt %d at-or-below floor %d; every member measured and live%s", s.Name, rep.TotalDebt, s.Floor, target),
		"hold the line; the member loops keep it tended"
}

func worklistHead(rep WalkReport) string {
	if len(rep.Worklist) == 0 {
		return "(nothing to enter)"
	}
	w := rep.Worklist[0]
	return fmt.Sprintf("%s %q — %s", w.Member.Kind, w.Member.Ref, w.Action)
}

func memberKey(m Member) string { return string(m.Kind) + ":" + m.Ref }

func firstNonEmpty(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}
