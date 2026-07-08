package metrics

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

// Generation-aware learning-agenda selection (issue #1675, gen/future).
//
// An operator brief has a bounded attention budget, and the learning items competing
// for it do not share a horizon: some are urgent knowledge (a gen/now regression the
// operator must understand today), some are future optionality (a gen/future standard
// or benchmark whose value shows up only if it is not crowded out). A naive selector
// ranks every item by urgency and greedily fills the budget. That selector has a bug
// that looks like a policy: urgent items always win, so optionality is never learned,
// and the horizon label silently becomes a priority label.
//
// This file is the SHAPE of the selection rule that fixes that, expressed as a pure,
// stdlib-only model (no clock, no disk, no default exposure) in the same idiom as
// generation_roadmap.go. The rule has two parts:
//
//  1. AN OPTIONALITY RESERVE. A fraction of the attention budget is reserved for
//     horizons PAST "now". Only optionality items may draw from it, so a flood of
//     high-urgency gen/now items cannot starve every gen/future item. The reserve is a
//     FLOOR, not a cap: whatever the reserve does not spend flows back into the open
//     pool, so protecting optionality never wastes attention.
//
//  2. HORIZON-BLIND RANKING WITHIN A POOL. Inside either pool, items are ranked by
//     Urgency (a decay/expiry property carried by the ITEM), never by horizon rank.
//     A gen/future item with higher urgency outranks a gen/now item with lower urgency.
//     This is the mechanism that keeps the horizon label from becoming a value judgment.
//
// Generation stays orthogonal to three things, and the selector is built to keep them
// separate (OrthogonalityNote, shared with the roadmap dashboard, is rendered in the
// header):
//
//   - PRIORITY — horizon never enters the ranking comparator. Urgency is a per-item
//     field, independent of Horizon; the reserve changes WHICH POOL an item competes
//     in, never its rank inside that pool.
//   - SHARED TRUNK — an agenda is a view over items that all live on main. No horizon
//     implies a branch or a worktree; Horizon is a label filter, nothing more.
//   - RUNTIME FEATURE GATES — an item's horizon says WHEN its subject is expected to
//     mature, not WHETHER anything is exposed at runtime. Nothing here reads or sets a
//     feature gate; default exposure remains an explicit, independent decision.
//
// Promotion / demotion / invalidating-assumption for this artifact itself:
//   - Promotion evidence: internal/operatorbrief's learningAgendaFor stops selecting a
//     single Focus by Pace alone and instead folds real candidate items through
//     SelectLearningAgenda against the brief's existing Attention.BudgetMinutes, so a
//     rendered brief demonstrably carries at least one non-"now" item under load.
//   - Demotion/retirement evidence: retire this rule if measurement shows the reserve is
//     never binding (optionality items always fit anyway, so the reserve is ceremony),
//     or if operators observably ignore reserved items — a reserved-but-unread item is
//     worse than an honest deferral, and the reserve should then drop to zero.
//   - Invalidating assumption: that attention is a SINGLE fungible budget measured in
//     minutes, and that a learning item's cost is separable from the others'. If
//     learning is actually context-switch-dominated (three 5-minute items cost far more
//     than one 15-minute item), then minutes are the wrong unit, the greedy fill is the
//     wrong algorithm, and this model must be rebuilt on a switch-cost metric before it
//     is promoted. A second, weaker assumption: that Urgency is knowable per item at
//     selection time rather than only in hindsight.

// DefaultOptionalityReserve is the starting fraction of an attention budget held for
// horizons past "now". It is a policy default, not a discovered constant: the honest
// witness for tuning it is the demotion evidence named above (is the reserve binding,
// and are reserved items actually read?).
const DefaultOptionalityReserve = 0.25

// AgendaPool names which half of the budget paid for a selected item. It is a closed
// vocabulary: an item is either bought with attention reserved for optionality, or with
// attention every horizon competes for on equal terms.
type AgendaPool string

const (
	// PoolReserve means the item was bought with the optionality reserve — attention
	// that only a horizon past "now" may draw from.
	PoolReserve AgendaPool = "optionality-reserve"
	// PoolOpen means the item won attention in the open pool, ranked purely by
	// urgency against every other horizon.
	PoolOpen AgendaPool = "open"
)

// AgendaDeferral is the closed vocabulary of reasons an item did not reach the brief.
// A deferral is always explicit: an item is never silently dropped, because "the brief
// did not mention it" and "the brief could not afford it" are different facts.
type AgendaDeferral string

const (
	// DeferBudgetExhausted means the item was valid but did not fit the budget.
	DeferBudgetExhausted AgendaDeferral = "budget-exhausted"
	// DeferUnknownHorizon means the item named a horizon outside RoadmapGenerations.
	// Selection fails closed on it rather than guessing a horizon.
	DeferUnknownHorizon AgendaDeferral = "unknown-horizon"
	// DeferInvalidCost means the item declared a non-positive attention cost, which
	// would let it ride any budget for free.
	DeferInvalidCost AgendaDeferral = "invalid-cost"
)

// AgendaDeferrals is the ordered, closed deferral vocabulary, for a test or a renderer
// that must enumerate every reason an item can miss the brief.
var AgendaDeferrals = []AgendaDeferral{
	DeferBudgetExhausted,
	DeferUnknownHorizon,
	DeferInvalidCost,
}

// LearningItem is one candidate for the operator's learning agenda.
//
// Horizon and Urgency are deliberately independent fields. Horizon says WHEN the
// subject is expected to mature; Urgency says how fast the value of learning it decays.
// A gen/future item can be urgent (a standards comment window closes Friday) and a
// gen/now item can be unurgent (a shipped subsystem that is not going anywhere).
// Collapsing the two is the exact error this model exists to prevent.
type LearningItem struct {
	// Topic is the item's stable identity, and the deterministic tie-break key.
	Topic string `json:"topic"`
	// Horizon is the generation stream, one of RoadmapGenerations.
	Horizon string `json:"horizon"`
	// Minutes is the attention cost of learning the item. Must be > 0.
	Minutes int `json:"minutes"`
	// Urgency is how fast the value of this item decays, in [0,1]. It is a property
	// of the ITEM, never derived from Horizon.
	Urgency float64 `json:"urgency"`
	// Why is an optional one-line rationale carried into the rendered brief.
	Why string `json:"why,omitempty"`
}

// IsOptionality reports whether the item sits past the "now" horizon — the class the
// reserve exists to protect.
func (it LearningItem) IsOptionality() bool {
	return it.Horizon != "now"
}

// AgendaBudget is the operator's bounded attention for one brief.
type AgendaBudget struct {
	// TotalMinutes is the whole attention budget for the agenda.
	TotalMinutes int `json:"total_minutes"`
	// OptionalityReserve is the fraction of TotalMinutes reserved for horizons past
	// "now", clamped to [0,1]. Zero disables the reserve (and reproduces the naive
	// urgency-greedy selector, which is useful as a control in a measurement).
	OptionalityReserve float64 `json:"optionality_reserve"`
}

// SelectedItem is a LearningItem that reached the brief, plus which pool paid for it.
type SelectedItem struct {
	LearningItem
	Pool AgendaPool `json:"pool"`
}

// DeferredItem is a LearningItem that did not reach the brief, plus the closed-vocabulary
// reason. Every input item appears in exactly one of Selected or Deferred.
type DeferredItem struct {
	LearningItem
	Reason AgendaDeferral `json:"reason"`
}

// LearningAgendaPlan is the full selection result: what the brief should recommend, what
// it consciously deferred, and how the budget was spent across the two pools.
type LearningAgendaPlan struct {
	Budget         AgendaBudget   `json:"budget"`
	Selected       []SelectedItem `json:"selected"`
	Deferred       []DeferredItem `json:"deferred,omitempty"`
	MinutesUsed    int            `json:"minutes_used"`
	ReserveMinutes int            `json:"reserve_minutes"`
	ReserveUsed    int            `json:"reserve_used"`
}

// reserveMinutes computes the optionality floor for a budget, clamped so a nonsense
// fraction can neither exceed the budget nor go negative.
func (b AgendaBudget) reserveMinutes() int {
	if b.TotalMinutes <= 0 {
		return 0
	}
	f := b.OptionalityReserve
	if f <= 0 {
		return 0
	}
	if f > 1 {
		f = 1
	}
	r := int(math.Round(float64(b.TotalMinutes) * f))
	if r > b.TotalMinutes {
		r = b.TotalMinutes
	}
	return r
}

// knownHorizon reports whether h is in the closed horizon vocabulary. Selection fails
// closed on anything else rather than inventing a horizon for it.
func knownHorizon(h string) bool {
	for _, s := range RoadmapGenerations {
		if s == h {
			return true
		}
	}
	return false
}

// rankAgendaItems orders candidates by urgency descending, tie-broken by Topic ascending.
// Horizon is deliberately absent from the comparator — that absence IS the orthogonality
// guarantee, and a test binds it.
func rankAgendaItems(items []LearningItem) []LearningItem {
	out := make([]LearningItem, len(items))
	copy(out, items)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Urgency != out[j].Urgency {
			return out[i].Urgency > out[j].Urgency
		}
		return out[i].Topic < out[j].Topic
	})
	return out
}

// SelectLearningAgenda applies the two-part rule to a candidate set and a budget:
// fill the optionality reserve from horizons past "now" (ranked by urgency), then fill
// the remaining budget from everything still unselected (ranked by urgency, horizon-blind).
//
// It is pure and deterministic: the same candidates and budget always produce the same
// plan, so a test can assert it exactly and a brief can render it.
func SelectLearningAgenda(items []LearningItem, b AgendaBudget) LearningAgendaPlan {
	plan := LearningAgendaPlan{Budget: b, ReserveMinutes: b.reserveMinutes()}

	// Validate first, so an invalid item is reported as invalid rather than as
	// "budget-exhausted" — the two send an operator to different fixes.
	var valid []LearningItem
	var invalid []DeferredItem
	for _, it := range items {
		switch {
		case !knownHorizon(it.Horizon):
			invalid = append(invalid, DeferredItem{LearningItem: it, Reason: DeferUnknownHorizon})
		case it.Minutes <= 0:
			invalid = append(invalid, DeferredItem{LearningItem: it, Reason: DeferInvalidCost})
		default:
			valid = append(valid, it)
		}
	}
	sort.SliceStable(invalid, func(i, j int) bool { return invalid[i].Topic < invalid[j].Topic })

	ranked := rankAgendaItems(valid)
	taken := make(map[int]AgendaPool, len(ranked))

	// Phase 1 — the optionality reserve. Only horizons past "now" may draw from it.
	// A too-large item is skipped, not fatal: a smaller optionality item behind it can
	// still claim the reserve, which is what makes the floor actually bind.
	reserveLeft := plan.ReserveMinutes
	for i, it := range ranked {
		if !it.IsOptionality() {
			continue
		}
		if it.Minutes <= reserveLeft {
			taken[i] = PoolReserve
			reserveLeft -= it.Minutes
		}
	}
	plan.ReserveUsed = plan.ReserveMinutes - reserveLeft

	// Phase 2 — the open pool: every remaining item, every horizon, ranked by urgency
	// alone. Unspent reserve flows back here, so the floor never wastes attention.
	openLeft := b.TotalMinutes - plan.ReserveUsed
	for i, it := range ranked {
		if _, ok := taken[i]; ok {
			continue
		}
		if it.Minutes <= openLeft {
			taken[i] = PoolOpen
			openLeft -= it.Minutes
		}
	}

	for i, it := range ranked {
		if pool, ok := taken[i]; ok {
			plan.Selected = append(plan.Selected, SelectedItem{LearningItem: it, Pool: pool})
			plan.MinutesUsed += it.Minutes
			continue
		}
		plan.Deferred = append(plan.Deferred, DeferredItem{LearningItem: it, Reason: DeferBudgetExhausted})
	}
	plan.Deferred = append(plan.Deferred, invalid...)
	return plan
}

// MinutesByHorizon rolls the selected items up per horizon, so a brief can show the
// urgent/optionality split it actually bought. Every horizon in RoadmapGenerations is
// present, including the zero ones — a starved horizon must be visible, not absent.
func (p LearningAgendaPlan) MinutesByHorizon() map[string]int {
	out := make(map[string]int, len(RoadmapGenerations))
	for _, s := range RoadmapGenerations {
		out[s] = 0
	}
	for _, s := range p.Selected {
		out[s.Horizon] += s.Minutes
	}
	return out
}

// OptionalityMinutes is the attention the plan spent on horizons past "now", whichever
// pool paid for it. This is the number that answers the issue's question directly: did
// the brief separate urgent knowledge from future optionality, or only claim to?
func (p LearningAgendaPlan) OptionalityMinutes() int {
	var n int
	for _, s := range p.Selected {
		if s.IsOptionality() {
			n += s.Minutes
		}
	}
	return n
}

// Render produces a deterministic text agenda: the orthogonality header, the selected
// items with the pool that paid for each, the per-horizon mix, then the explicit
// deferrals. Pure (no clock, no disk), so a test can assert its bytes.
func (p LearningAgendaPlan) Render() string {
	var b strings.Builder
	b.WriteString("Learning agenda (budget: " + strconv.Itoa(p.Budget.TotalMinutes) +
		"m, optionality reserve: " + strconv.Itoa(p.ReserveMinutes) + "m)\n")
	b.WriteString(OrthogonalityNote)
	b.WriteString("\n\n")

	b.WriteString("  selected (" + strconv.Itoa(p.MinutesUsed) + "m used, " +
		strconv.Itoa(p.OptionalityMinutes()) + "m on optionality):\n")
	if len(p.Selected) == 0 {
		b.WriteString("    - (none)\n")
	}
	for _, s := range p.Selected {
		b.WriteString("    - " + pad("gen/"+s.Horizon, agendaHorizonWidth) + " " +
			pad(s.Topic, agendaTopicWidth) + " " +
			strconv.Itoa(s.Minutes) + "m urgency=" +
			strconv.FormatFloat(s.Urgency, 'f', -1, 64) + " [" + string(s.Pool) + "]")
		if s.Why != "" {
			b.WriteString(" | " + s.Why)
		}
		b.WriteString("\n")
	}

	b.WriteString("\n  horizon mix:")
	mix := p.MinutesByHorizon()
	for _, s := range RoadmapGenerations {
		b.WriteString(" " + s + "=" + strconv.Itoa(mix[s]) + "m")
	}
	b.WriteString("\n")

	if len(p.Deferred) > 0 {
		b.WriteString("\n  deferred:\n")
		for _, d := range p.Deferred {
			b.WriteString("    - " + pad("gen/"+d.Horizon, agendaHorizonWidth) + " " +
				pad(d.Topic, agendaTopicWidth) + " " +
				strconv.Itoa(d.Minutes) + "m -> " + string(d.Reason) + "\n")
		}
	}
	return b.String()
}

const (
	agendaHorizonWidth = 16
	agendaTopicWidth   = 28
)
