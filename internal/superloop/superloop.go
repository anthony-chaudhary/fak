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
)

// Member is one constituent a super loop walks. Ref names the surface (scorecard
// key / loop id / garden name / super-loop name); Why is the one-line reason it
// belongs under this intent.
type Member struct {
	Kind MemberKind `json:"kind"`
	Ref  string     `json:"ref"`
	Why  string     `json:"why"`
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
}

// registry is the curated set of named super loops. It is deliberately small and
// data-only: each entry binds an operator intent to REAL existing surfaces (every
// scorecard Ref is a control-pane card key; the no-drift test enforces it), so a
// super loop can never point an operator at a member that does not exist.
var registry = []Super{
	{
		Name:  "improve-quality",
		Title: "improve code & content quality",
		About: "walk the quality-bearing scorecards + the gardening loop, then enter the worst-first member to retire its debt",
		Floor: 0,
		Members: []Member{
			{Kind: KindScorecard, Ref: "slop", Why: "code-slop debt is the heaviest quality drag; retire it worst-first"},
			{Kind: KindScorecard, Ref: "code", Why: "code-quality debt: the core correctness/clarity signal"},
			{Kind: KindScorecard, Ref: "disambiguation", Why: "concept-disambiguation debt: ambiguous concepts confuse agents and readers"},
			{Kind: KindScorecard, Ref: "conflation", Why: "conflation debt: metrics that report an upstream value without an OBSERVED qualifier"},
			{Kind: KindScorecard, Ref: "intent_literal", Why: "intent-literal debt: code that drifts from the operator's literal intent"},
			{Kind: KindScorecard, Ref: "ui_quality", Why: "UI/UX-quality debt over the terminal render surface"},
			{Kind: KindScorecard, Ref: "claim_repro", Why: "claim-repro debt: claims a reader cannot reproduce"},
			{Kind: KindGarden, Ref: "garden", Why: "the gardening bundle tends the rest (scorecard ratchet, fresh-status, stale work)"},
		},
	},
	{
		Name:  "improve-loops",
		Title: "improve the agentic + background loops",
		About: "walk the loop-index scorecard + the live loop ledgers, then enter the worst-first loop that is in debt or has gone dark",
		Floor: 0,
		Members: []Member{
			{Kind: KindScorecard, Ref: "loopindex", Why: "the agentic-coding loop-index: orient->plan->act->verify->ship->learn stages not yet witnessed at floor"},
			{Kind: KindScorecard, Ref: "dogfood", Why: "dogfood-loop debt: are we running our own loops?"},
			{Kind: KindLoop, Ref: "dispatch", Why: "the issue-resolve dispatch loop — dark means throughput stalled"},
			{Kind: KindLoop, Ref: "cadence", Why: "the regular-cadence report loop — dark means the pacing pulse stopped"},
			{Kind: KindLoop, Ref: "dojo", Why: "the dojo gym loop — dark means calibration stopped"},
			{Kind: KindGarden, Ref: "garden", Why: "the gardening bundle surfaces orphaned/unwitnessed runs across loops"},
		},
	},
	{
		Name:  "manage-benchmarks",
		Title: "manage benchmark collection and publishing",
		About: "walk benchmark DX, the nightrun collection loop, and the benchmark control surface, then enter the worst-first benchmark action",
		Floor: 0,
		Members: []Member{
			{Kind: KindScorecard, Ref: "bench_dx", Why: "benchmark-DX debt means the benchmark surfaces are confusing or incomplete"},
			{Kind: KindLoop, Ref: "nightrun", Why: "the local benchmark data-collection loop; dark/stale means collection throughput stalled"},
			{Kind: KindSurface, Ref: "fak bench-loop status", Why: "the benchmark domain super-loop folds registry, catalog, local next selection, request/rollup surfaces, and the authority gap"},
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
// Container marks a member (a garden or a super loop) whose status is only knowable
// by DESCENDING into its own walk — it is a recursion pointer, not a leaf to weigh.
// A container is always surfaced in the worklist as a "descend" item, but it is NOT
// counted toward aggregate debt or the measured/unmeasured tally and never blocks
// Satisfied: this walk did not claim to have read it, so it neither inflates the
// debt nor is slandered as an unreadable failure.
type MemberStatus struct {
	Member    Member `json:"member"`
	Debt      int    `json:"debt"`
	Dark      bool   `json:"dark"`
	Measured  bool   `json:"measured"`
	Container bool   `json:"container"`
	Detail    string `json:"detail"`
}

// WorkItem is one worst-first entry in the walk's plan: enter this member next, and
// why. Rank is 1-based in worst-first order.
type WorkItem struct {
	Rank   int    `json:"rank"`
	Member Member `json:"member"`
	Debt   int    `json:"debt"`
	Dark   bool   `json:"dark"`
	Action string `json:"action"`
	Detail string `json:"detail"`
}

// WalkReport is the folded intent-level verdict + the worst-first worklist: the
// answer to "I asked to <intent> — what is the status of everything under it, and
// what should I enter first?"
type WalkReport struct {
	Schema     string         `json:"schema"`
	Name       string         `json:"name"`
	Title      string         `json:"title"`
	TotalDebt  int            `json:"total_debt"`
	Floor      int            `json:"floor"`
	Satisfied  bool           `json:"satisfied"`
	Members    int            `json:"members"`
	Walked     int            `json:"walked"`
	Unmeasured int            `json:"unmeasured"`
	Dark       int            `json:"dark"`
	Worklist   []WorkItem     `json:"worklist"`
	Statuses   []MemberStatus `json:"statuses"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
}

// Walk folds the member statuses the shell read into the intent-level verdict and a
// worst-first worklist. Ordering (the SELECT step): dark/unmeasured members first
// (a gone-dark loop or an unreadable member is the most urgent thing to enter), then
// by debt descending, ties broken by the member's declared order (stable). The
// intent is SATISFIED only when total debt is at-or-below Floor AND every member was
// measured AND none is dark — an unread or dark member can never read as clean.
func Walk(s Super, statuses []MemberStatus) WalkReport {
	rep := WalkReport{
		Schema:   WalkSchema,
		Name:     s.Name,
		Title:    s.Title,
		Floor:    s.Floor,
		Members:  len(s.Members),
		Statuses: statuses,
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
		return order[memberKey(ranked[i].Member)] < order[memberKey(ranked[j].Member)]
	})

	for _, st := range ranked {
		// A clean, measured leaf is nothing to enter. A container is ALWAYS surfaced
		// (its status is only knowable by descending). Everything with work stays.
		if !st.Container && st.Measured && !st.Dark && st.Debt <= 0 {
			continue
		}
		rep.Worklist = append(rep.Worklist, WorkItem{
			Member: st.Member,
			Debt:   st.Debt,
			Dark:   st.Dark,
			Action: actionFor(st),
			Detail: workDetail(st),
		})
	}
	// Re-rank the filtered worklist 1..N so the printed ranks are contiguous.
	for i := range rep.Worklist {
		rep.Worklist[i].Rank = i + 1
	}

	rep.Satisfied = rep.Unmeasured == 0 && rep.Dark == 0 && rep.TotalDebt <= s.Floor
	rep.Verdict, rep.Finding, rep.Reason, rep.NextAction = walkVerdict(s, rep)
	return rep
}

// tier ranks a member status into a worst-first band (lower = enter sooner):
//
//	0  a dark / unmeasured LEAF — its status is bad or unknown; most urgent
//	1  a measured leaf carrying debt
//	2  a container (garden / super loop) — descend to learn its status
//	3  a measured, clean, live leaf — nothing to do
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
		return fmt.Sprintf("enter the %s scorecard's reduce loop (its skill) to retire debt", st.Member.Ref)
	case KindLoop:
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
	return "OK", "superloop_satisfied",
		fmt.Sprintf("walking %q: aggregate debt %d at-or-below floor %d; every member measured and live", s.Name, rep.TotalDebt, s.Floor),
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
