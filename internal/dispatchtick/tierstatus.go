package dispatchtick

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ---------------------------------------------------------------------------
// C8 TIER-DECISION OBSERVABILITY (#3045) — turn the tier account chooser's
// decisions into an operator-readable tier_decision ledger row + status report.
// ---------------------------------------------------------------------------
//
// This is the OBSERVABILITY node of the model-tier working path:
//
//	tier route (C5)  ->  THIS (tier_decision readout)  ->  operator status
//
// C5 (RouteAccountForTier) decides WHICH account serves an issue and already
// carries the routing verdict (over-tier, under-tier-refused, reason). C8 does
// NOT re-decide anything: it JOINS that pure verdict to the surrounding context
// an operator needs — issue number, lane, chosen model, required/optimal/chosen
// tier, escalation, witnessed outcome, and a MODELED cost delta — and folds a set
// of decisions into a status report that shows over-tier waste and under-tier
// refusals side by side. Without this, a tiering system silently wastes frontier
// seats or silently degrades work quality (the issue's "why this is next").
//
// PURITY & SURFACE. BuildTierStatusReport is a pure fold over the passed rows: no
// I/O, no wall clock, inputs untouched, so a hand-verified fixture stays valid.
// It imports only modelroute (the WorkTier vocabulary) + stdlib, matching this
// leaf's documented architest surface — in particular it does NOT import
// issuecost. The TierOutcome vocabulary below MIRRORS issuecost.Bucket (the C9
// calibration fold, #3046) by name so the two readouts agree, but C8 only
// RENDERS an outcome supplied by a ledger/fixture; it never re-derives a witness.
//
// COST IS MODELED, NOT NETTED. CostDelta is a dimensionless MODELED cost-point
// delta versus the optimal tier, a stand-in until provider bills land (the
// issue's assumption). Per the issue's confusion risks the report keeps gross
// modeled cost SEPARATE from the witnessed outcome column and never presents a
// modeled saving as a net win without quality parity — see the report Note.

// TierOutcome is the witnessed terminal state of the work a tier decision drove.
// Closed vocabulary mirroring issuecost.Bucket (C9, #3046) WITHOUT importing that
// leaf, so dispatchtick keeps its modelroute+stdlib surface. The readout only
// renders an outcome a ledger/fixture supplied — it does not witness anything.
type TierOutcome string

const (
	// TierOutcomePending: a decision was made but no outcome is witnessed yet.
	TierOutcomePending TierOutcome = "pending"
	// TierOutcomeShipped: diff-witnessed commit + green tests + close (issuecost success).
	TierOutcomeShipped TierOutcome = "shipped"
	// TierOutcomeStall: no witnessed ship and no revert.
	TierOutcomeStall TierOutcome = "stall"
	// TierOutcomeEscalated: the chosen tier could not finish; work bumped up a tier.
	TierOutcomeEscalated TierOutcome = "escalated"
	// TierOutcomeReverted: shipped then reverted/reopened — rework.
	TierOutcomeReverted TierOutcome = "reverted"
	// TierOutcomeRefused: a correct under-tier refusal (never a demerit).
	TierOutcomeRefused TierOutcome = "refused"
)

// Valid reports whether o is one of the defined outcomes.
func (o TierOutcome) Valid() bool {
	switch o {
	case TierOutcomePending, TierOutcomeShipped, TierOutcomeStall,
		TierOutcomeEscalated, TierOutcomeReverted, TierOutcomeRefused:
		return true
	default:
		return false
	}
}

// TierStatusSchema versions the machine-readable readout so a status consumer can
// pin the shape.
const TierStatusSchema = "fak-tier-decision-status/1"

// CostProvenanceModeled labels every cost delta as MODELED (not provider-observed)
// per the net-true-value standard — a status consumer must not read it as billed.
const CostProvenanceModeled = "modeled"

// modelTierUnitCost is a RELATIVE, MODELED cost weight per account ModelTier: a
// frontier seat (tier 1) is the most expensive, a routine seat (tier 3) the
// cheapest. Units are dimensionless "cost points", NOT dollars — a stand-in until
// provider bills are wired (the issue's assumption). The spread is roughly
// geometric so over-tiering a routine unit onto a frontier seat reads as a large
// modeled waste, matching the asymmetry C3/C5 encode.
func modelTierUnitCost(modelTier int) int {
	switch modelTier {
	case 1:
		return 9
	case 2:
		return 3
	default: // tier 3 and any out-of-range value are the routine floor
		return 1
	}
}

// TierDecisionInput is one fixture/ledger row the readout folds: the issue and
// lane the decision was made for, an optional product filter + account pool, the
// issue's tier metadata, and an optional witnessed outcome (default pending). An
// explicit Escalated witness marks a decision the chosen tier could not carry.
type TierDecisionInput struct {
	Issue   int       `json:"issue"`
	Lane    string    `json:"lane"`
	Product string    `json:"product,omitempty"`
	Tier    IssueTier `json:"tier"`
	// Labels, when present, is the raw per-issue GitHub label set — the REAL tier
	// source. BuildTierDecisionRow derives Tier and the tag flags from it via
	// IssueTierFromLabels, so a live readout is fed the same tier signal the
	// dispatcher parses. When empty, Tier is used as given (the fixture path).
	Labels    []string     `json:"labels,omitempty"`
	Rows      []AccountRow `json:"-"`
	Outcome   TierOutcome  `json:"outcome,omitempty"`
	Escalated bool         `json:"escalated,omitempty"`
}

// TierDecisionRow is one observable tier_decision: the routing verdict joined to
// the operator context (issue, lane, model) plus escalation, witnessed outcome,
// and a MODELED cost delta versus the optimal tier. It is the row the issue's
// working spine names ("tier_decision rows joined to issue number, lane, model,
// required_tier, optimal_tier, chosen_tier, reason, escalation, outcome, and
// estimated cost delta").
type TierDecisionRow struct {
	Issue            int                 `json:"issue"`
	Lane             string              `json:"lane"`
	Account          string              `json:"account,omitempty"`
	Model            string              `json:"model,omitempty"`
	RequiredTier     modelroute.WorkTier `json:"required_tier"`
	OptimalTier      modelroute.WorkTier `json:"optimal_tier"`
	ChosenTier       modelroute.WorkTier `json:"chosen_tier"`
	ChosenModelTier  int                 `json:"chosen_model_tier"`
	Reason           string              `json:"reason"`
	OverTier         bool                `json:"over_tier"`
	UnderTierRefused bool                `json:"under_tier_refused"`
	Escalation       bool                `json:"escalation"`
	Outcome          TierOutcome         `json:"outcome"`
	CostDeltaModeled int                 `json:"cost_delta_modeled"`
	CostProvenance   string              `json:"cost_provenance"`
	Blocked          []string            `json:"blocked_accounts,omitempty"`
	// TagFlags names why an issue's tier tags were not cleanly usable
	// (model_tier_*_missing|invalid|conflict, model_tier_contradiction), empty when
	// the labels parsed to a clean tier. Surfaced so the readout shows tagging
	// health, not just the routing verdict — a conservative frontier route driven
	// by a missing/contradictory tag reads differently from a genuine T0 issue.
	TagFlags []string `json:"tag_flags,omitempty"`
}

// BuildTierDecisionRow routes one input through the C5 chooser and folds the
// verdict + context into an observable row. CostDeltaModeled is chosen_cost -
// optimal_cost in modeled cost points: POSITIVE is over-tier waste (spent above
// the optimal tier), NEGATIVE is a saving (below optimal but still at/above the
// required floor). A refusal spends nothing, so its delta is 0 and it carries the
// too-weak seats it turned away.
func BuildTierDecisionRow(in TierDecisionInput) TierDecisionRow {
	// Labels are the REAL tier source when present: derive the typed tier (and the
	// closed-vocab tag flags) exactly as the dispatcher would. An issue whose labels
	// are missing/invalid/conflicting/contradictory yields HasTier=false, so
	// IssueTier.resolve applies the conservative frontier floor — the safe default,
	// and the tag flags say why it stayed conservative.
	tier := in.Tier
	var tagFlags []string
	if len(in.Labels) > 0 {
		tier, tagFlags = IssueTierFromLabels(in.Labels)
	}
	res := RouteAccountForTier(in.Rows, in.Product, tier)

	outcome := in.Outcome
	if outcome == "" {
		outcome = TierOutcomePending
	}
	escalation := in.Escalated || outcome == TierOutcomeEscalated

	row := TierDecisionRow{
		Issue:            in.Issue,
		Lane:             strings.TrimSpace(in.Lane),
		RequiredTier:     res.RequiredTier,
		OptimalTier:      res.OptimalTier,
		Reason:           res.FallbackReason,
		OverTier:         res.OverTier,
		UnderTierRefused: res.UnderTierRefused,
		Escalation:       escalation,
		Outcome:          outcome,
		CostProvenance:   CostProvenanceModeled,
		TagFlags:         tagFlags,
	}

	if res.UnderTierRefused {
		// A refusal launched nothing: no chosen tier, no spend. Surface it as the
		// refused outcome unless a ledger already labeled it something stronger.
		if outcome == TierOutcomePending {
			row.Outcome = TierOutcomeRefused
		}
		row.ChosenTier = modelroute.WorkTier(-1) // renders as T? — nothing was chosen
		for _, b := range res.Blocked {
			row.Blocked = append(row.Blocked, b.Account)
		}
		return row
	}

	if !res.OK {
		// No routable accounts at all — distinct from a floor refusal.
		row.ChosenTier = modelroute.WorkTier(-1)
		return row
	}

	row.Account = res.Account.Account
	row.Model = res.Account.Model
	row.ChosenTier = res.ChosenTier
	row.ChosenModelTier = res.ChosenModelTier

	optimalModelTier := modelTierFloorForWork(res.OptimalTier)
	row.CostDeltaModeled = modelTierUnitCost(res.ChosenModelTier) - modelTierUnitCost(optimalModelTier)
	return row
}

// TierStatusReport is the operator-facing fold: every decision row plus the
// counters the issue's working spine names — how many decisions routed, how many
// were over-tier or under-tier refused, how many escalated, the modeled cost
// waste (over-tier) and modeled saving (below-optimal) kept SEPARATE, and the
// witnessed-outcome tally. Note carries the quality-parity caveat so a modeled
// saving is never read as a net win.
type TierStatusReport struct {
	Schema              string              `json:"schema"`
	Rows                []TierDecisionRow   `json:"rows"`
	Decisions           int                 `json:"decisions"`
	Routed              int                 `json:"routed"`
	OverTier            int                 `json:"over_tier"`
	UnderTierRefused    int                 `json:"under_tier_refused"`
	Escalations         int                 `json:"escalations"`
	OverTierCostModeled int                 `json:"over_tier_cost_modeled"`
	SavingsCostModeled  int                 `json:"savings_cost_modeled"`
	CostProvenance      string              `json:"cost_provenance"`
	OutcomeTally        map[TierOutcome]int `json:"outcome_tally"`
	Note                string              `json:"note"`
}

// tierStatusNote is the standing caveat the issue's confusion risks demand: the
// cost columns are MODELED and are NOT netted against quality, so a gross modeled
// saving must not be presented as a net win without witnessed quality parity (the
// outcome column). It also warns against summing these with provider-observed
// cache savings.
const tierStatusNote = "cost_delta is MODELED cost points, not billed dollars, and is NOT netted against quality; do not read a modeled saving as a net win without witnessed outcome parity, and do not sum it with provider-observed cache savings"

// BuildTierStatusReport folds a set of tier decisions into the operator readout.
// It is pure: no I/O, no wall clock, inputs untouched, OutcomeTally always
// non-nil. Rows preserve input order so a captured fixture is deterministic.
func BuildTierStatusReport(inputs []TierDecisionInput) TierStatusReport {
	rep := TierStatusReport{
		Schema:         TierStatusSchema,
		Decisions:      len(inputs),
		CostProvenance: CostProvenanceModeled,
		OutcomeTally:   map[TierOutcome]int{},
		Note:           tierStatusNote,
	}
	for _, in := range inputs {
		row := BuildTierDecisionRow(in)
		rep.Rows = append(rep.Rows, row)
		rep.OutcomeTally[row.Outcome]++
		if row.Escalation {
			rep.Escalations++
		}
		switch {
		case row.UnderTierRefused:
			rep.UnderTierRefused++
		default:
			rep.Routed++
			if row.OverTier {
				rep.OverTier++
			}
			switch {
			case row.CostDeltaModeled > 0:
				rep.OverTierCostModeled += row.CostDeltaModeled
			case row.CostDeltaModeled < 0:
				rep.SavingsCostModeled += -row.CostDeltaModeled
			}
		}
	}
	return rep
}

// Render formats a TierStatusReport as a compact, deterministic operator readout:
// a header line of counters, one line per decision row, the outcome tally, and
// the modeled-cost caveat. It is the human mirror of the JSON surface.
func (r TierStatusReport) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "tier decisions: %d (routed=%d over_tier=%d under_tier_refused=%d escalations=%d)\n",
		r.Decisions, r.Routed, r.OverTier, r.UnderTierRefused, r.Escalations)
	fmt.Fprintf(&b, "  modeled cost points [%s]: over_tier_waste=%d savings=%d\n",
		r.CostProvenance, r.OverTierCostModeled, r.SavingsCostModeled)
	for _, row := range r.Rows {
		chosen := row.ChosenTier.String()
		seat := row.Account
		if seat == "" {
			seat = "-"
		}
		flags := tierRowFlags(row)
		fmt.Fprintf(&b, "  #%-5d %-10s req=%s opt=%s chosen=%s seat=%-10s cost=%+d outcome=%-9s %s [%s]\n",
			row.Issue, truncateLane(row.Lane), row.RequiredTier, row.OptimalTier, chosen,
			seat, row.CostDeltaModeled, row.Outcome, flags, row.Reason)
	}
	b.WriteString("  outcomes:")
	for _, o := range tierOutcomeOrder {
		if n, ok := r.OutcomeTally[o]; ok && n > 0 {
			fmt.Fprintf(&b, " %s=%d", o, n)
		}
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "  note: %s\n", r.Note)
	return b.String()
}

// tierOutcomeOrder fixes the render iteration order over the outcome tally.
var tierOutcomeOrder = []TierOutcome{
	TierOutcomeShipped, TierOutcomeStall, TierOutcomeEscalated,
	TierOutcomeReverted, TierOutcomeRefused, TierOutcomePending,
}

// tierRowFlags renders the load-bearing verdict flags for a row so the operator
// sees over-tier waste and under-tier refusals at a glance.
func tierRowFlags(row TierDecisionRow) string {
	var f []string
	if row.OverTier {
		f = append(f, "OVER-TIER")
	}
	if row.UnderTierRefused {
		f = append(f, "REFUSED")
	}
	if row.Escalation {
		f = append(f, "ESCALATED")
	}
	if len(f) == 0 {
		return "ok"
	}
	sort.Strings(f)
	return strings.Join(f, "+")
}

// truncateLane keeps the render column aligned for a long lane label.
func truncateLane(lane string) string {
	if lane == "" {
		return "-"
	}
	if len(lane) <= 10 {
		return lane
	}
	return lane[:9] + "…"
}
