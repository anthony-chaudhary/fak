package gardenbundle

// walk.go — the ITEM side of the garden. The bundle fold (gardenbundle.go) and the
// act-tick (tick.go) operate over the ~8 ORCHESTRATOR MEMBERS. This file adds the
// walk over the HUNDREDS of INDIVIDUAL garden items a member surfaces (today:
// issues; the seam takes any scored item set), so "tend the garden" can mean one
// resource-aware pass over a 300-item backlog instead of 300 separate reports.
//
// The decision of WHAT to do per item — or whether to SKIP it to save the budget —
// is PURE and TESTABLE here (PlanWalk). The cmd verb supplies the classified item
// set (e.g. `gh issue list` run through the issue gardener) and renders/ledgers.
//
// RESOURCE-AWARENESS — the load-bearing properties the goal names ("aware of what
// to do or not to save resources/time"), all enforced by PlanWalk and all tested:
//
//   - skip-fresh: an item touched within SkipFreshDays is skipped WITHOUT being put
//     on the worklist, whatever its tags — a freshly-active item is being worked, so
//     spending attention on it now is waste. The cheap pre-filter that drops the
//     bulk of a live backlog before any follow-up cost is paid.
//   - budget: at most Budget items earn a disposition that needs follow-up, picked
//     WORST-FIRST by the gardener's score; the remainder are Deferred to the next
//     pass. So the walk's output — and the operator/agent work it implies — is
//     BOUNDED no matter how large the set, and a recurring walk drains the backlog
//     worst-first over passes instead of dumping it all at once.
//   - propose, don't execute: a decision carries the exact command but Perform is
//     false under DryRun — the same propose-don't-mutate discipline as the garden
//     tick and trajectory-garden. Auto-apply is a later, witness-gated rung.

import (
	"fmt"
	"sort"
	"strings"
)

// Schema identifiers for the walk envelope.
const WalkSchema = "fak.garden-walk.v1"

// WalkDisposition is the closed set of per-item outcomes the walk assigns.
type WalkDisposition string

const (
	// DispAct: a concrete, low-judgment gardening action with a ready command
	// (close a dormant question, mark a stale issue). Eligible to Perform.
	DispAct WalkDisposition = "act"
	// DispReview: the item surfaces a condition that needs human judgment
	// (needs-area / needs-kind / likely-dup). Surfaced, never auto-performed.
	DispReview WalkDisposition = "review"
	// DispSkip: nothing to do — the item is healthy (no condition) or fresh
	// (touched within the freshness window). Counted, never emitted in detail.
	DispSkip WalkDisposition = "skip"
)

// WalkItem is one classified garden item handed to the pure planner. The caller
// (the cmd verb) does the source-specific classification (the issue gardener's
// tags/score/action) and fills these fields; PlanWalk decides per-item handling
// and applies the resource policy. Source-agnostic on purpose: an issue, a
// trajectory turn, or an orphaned run all reduce to (id, score, idle, disposition).
type WalkItem struct {
	ID       int
	Title    string
	Score    int
	IdleDays int
	// InProgress marks an item already being actively worked (the source's own
	// "someone is on it" signal, e.g. the in-progress label). The cheapest, most
	// reliable resource pre-filter: re-surfacing an item under active work is waste,
	// and unlike IdleDays it is immune to bot churn bumping the timestamp.
	InProgress bool
	// Disposition is the source's classification BEFORE the resource policy: act,
	// review, or skip (healthy). PlanWalk may downgrade act/review to skip when the
	// item is fresh or in-progress, but never upgrades a skip.
	Disposition WalkDisposition
	// Action is the action kind (e.g. "mark-stale"); Command is the ready-to-run
	// shell command for an act (empty for review/skip). Reason explains the tags.
	Action  string
	Command string
	Reason  string
}

// WalkPolicy is the resource-awareness knob set. Zero values mean: no freshness
// skip, unbounded budget, act for real (not dry-run) — but the cmd defaults are
// resource-safe (a freshness window, a finite budget, dry-run on).
type WalkPolicy struct {
	// Budget caps how many attention items (act + review) earn a follow-up
	// decision, worst-first. <= 0 means unbounded.
	Budget int
	// SkipFreshDays skips any item idle FEWER than this many days, regardless of its
	// tags (a recently-touched item is left alone). <= 0 disables the freshness skip.
	// NOTE: keyed on the item's update timestamp, so it is only meaningful where that
	// timestamp tracks real work — on a bot-churned tracker it bumps constantly and
	// the filter over-skips, which is why the cmd default is off.
	SkipFreshDays int
	// SkipInProgress skips any item the source marks actively-in-progress — the cheap
	// pre-filter that genuinely fires even when timestamps are unreliable.
	SkipInProgress bool
	// DryRun forces every decision's Perform=false: the plan proposes, never acts.
	DryRun bool
}

// WalkDecision is one budgeted item's outcome: which handling applies, whether the
// walk would Perform it (act + not dry-run), and the ready command.
type WalkDecision struct {
	ID          int             `json:"id"`
	Title       string          `json:"title"`
	Score       int             `json:"score"`
	IdleDays    int             `json:"idle_days"`
	Disposition WalkDisposition `json:"disposition"`
	Action      string          `json:"action,omitempty"`
	Command     string          `json:"cmd,omitempty"`
	Perform     bool            `json:"perform"`
	Reason      string          `json:"reason,omitempty"`
}

// WalkPlan is the folded walk: the resource counts plus the bounded, worst-first
// worklist. It carries the control-pane envelope fields so it speaks the same
// schema/ok/verdict/reason/next_action language as the rest of the garden and can
// join the bundle as a member later.
type WalkPlan struct {
	Schema     string         `json:"schema"`
	Source     string         `json:"source"`
	DryRun     bool           `json:"dry_run"`
	Budget     int            `json:"budget"`
	Total      int            `json:"total"`     // items walked
	Active     int            `json:"active"`    // cheap-skipped: actively in-progress
	Fresh      int            `json:"fresh"`     // cheap-skipped: idle < SkipFreshDays
	Healthy    int            `json:"healthy"`   // skipped: no condition
	Attention  int            `json:"attention"` // act+review needing follow-up (pre-budget)
	Acted      int            `json:"acted"`     // act decisions emitted (Perform under !dry-run)
	Review     int            `json:"review"`    // review decisions emitted
	Deferred   int            `json:"deferred"`  // attention items beyond budget -> next pass
	Decisions  []WalkDecision `json:"decisions"` // budgeted, worst-first
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
}

// PlanWalk folds a classified item set into the resource-aware walk plan. It is
// pure: same items + policy in, same plan out, no I/O. The order of operations is
// the resource policy:
//
//  1. skip-fresh — drop items idle fewer than SkipFreshDays (cheap pre-filter),
//     counting them Fresh, before anything else looks at them.
//  2. partition the rest into attention (act/review) vs healthy (skip).
//  3. sort attention worst-first by score (tie: lower id first, stable).
//  4. take the first Budget as the emitted worklist; the rest are Deferred.
func PlanWalk(source string, items []WalkItem, policy WalkPolicy) WalkPlan {
	plan := WalkPlan{
		Schema: WalkSchema,
		Source: source,
		DryRun: policy.DryRun,
		Budget: policy.Budget,
		Total:  len(items),
	}

	attention := make([]WalkItem, 0, len(items))
	for _, it := range items {
		// 1a. Cheapest pre-filter: an actively-in-progress item is being handled — leave
		// it alone whatever its tags (immune to timestamp churn, unlike skip-fresh).
		if policy.SkipInProgress && it.InProgress {
			plan.Active++
			continue
		}
		// 1b. Freshness pre-filter: a recently-touched item is left alone whatever its tags.
		if policy.SkipFreshDays > 0 && it.IdleDays < policy.SkipFreshDays {
			plan.Fresh++
			continue
		}
		// 2. Partition. A skip (healthy) item needs no follow-up.
		if it.Disposition == DispSkip || it.Disposition == "" {
			plan.Healthy++
			continue
		}
		attention = append(attention, it)
	}
	plan.Attention = len(attention)

	// 3. Worst-first by score; stable, deterministic tie-break on id.
	sort.SliceStable(attention, func(i, j int) bool {
		if attention[i].Score != attention[j].Score {
			return attention[i].Score > attention[j].Score
		}
		return attention[i].ID < attention[j].ID
	})

	// 4. Budget: emit the worst Budget, defer the rest to the next pass.
	emit := attention
	if policy.Budget > 0 && len(attention) > policy.Budget {
		emit = attention[:policy.Budget]
		plan.Deferred = len(attention) - policy.Budget
	}

	plan.Decisions = make([]WalkDecision, 0, len(emit))
	for _, it := range emit {
		d := WalkDecision{
			ID:          it.ID,
			Title:       it.Title,
			Score:       it.Score,
			IdleDays:    it.IdleDays,
			Disposition: it.Disposition,
			Action:      it.Action,
			Command:     it.Command,
			Reason:      it.Reason,
		}
		switch it.Disposition {
		case DispAct:
			plan.Acted++
			d.Perform = !policy.DryRun
		case DispReview:
			plan.Review++
			d.Perform = false
		}
		plan.Decisions = append(plan.Decisions, d)
	}

	foldWalkVerdict(&plan)
	return plan
}

// foldWalkVerdict sets the control-pane envelope fields from the counts. A walk
// that surfaces a worklist is the pass WORKING (ok=true, verdict ACTION) — only an
// empty walk is "clear". OK is never false here: a backlog is not a broken garden.
func foldWalkVerdict(p *WalkPlan) {
	p.OK = true
	skipped := p.Active + p.Fresh + p.Healthy
	if p.Attention == 0 {
		p.Verdict = "OK"
		p.Finding = "garden_walk_clear"
		p.Reason = fmt.Sprintf("walked %d %s; none need attention (%d in-progress, %d fresh, %d healthy)",
			p.Total, plural(p.Source, p.Total), p.Active, p.Fresh, p.Healthy)
		p.NextAction = "hold the line; the scheduled walk re-checks the set next pass"
		return
	}
	p.Verdict = "ACTION"
	p.Finding = "garden_walk_worklist"
	emitted := p.Acted + p.Review
	var parts []string
	parts = append(parts, fmt.Sprintf("%d need attention", p.Attention))
	parts = append(parts, fmt.Sprintf("%d act / %d review emitted", p.Acted, p.Review))
	if p.Deferred > 0 {
		parts = append(parts, fmt.Sprintf("%d deferred to next pass (budget %d)", p.Deferred, p.Budget))
	}
	parts = append(parts, fmt.Sprintf("%d skipped (%d in-progress, %d fresh, %d healthy)", skipped, p.Active, p.Fresh, p.Healthy))
	p.Reason = fmt.Sprintf("walked %d %s: %s", p.Total, plural(p.Source, p.Total), strings.Join(parts, ", "))
	if p.DryRun {
		p.NextAction = fmt.Sprintf("propose-only (dry-run): run the %d emitted command(s)/review(s) worst-first", emitted)
	} else {
		p.NextAction = fmt.Sprintf("run the %d act command(s); review the rest", p.Acted)
	}
	if p.Deferred > 0 {
		p.NextAction += "; widen --budget or let the next pass take the deferred items"
	}
}

// plural renders a source noun with a rough plural for the reason line.
func plural(source string, n int) string {
	if source == "" {
		source = "item"
	}
	if n == 1 {
		return strings.TrimSuffix(source, "s")
	}
	if strings.HasSuffix(source, "s") {
		return source
	}
	return source + "s"
}
