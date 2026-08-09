package quality

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// This file is the MULTIPLICITY controller (#4568, under epic #4509): the layer
// that turns a GRID of quality comparisons into one trustworthy verdict. A suite
// never runs a single test — it runs |models| x |slices| x |metrics| of them, and
// every cell is its own hypothesis test with its own p-value. Judged at a nominal
// per-comparison alpha the grid manufactures alarms: over m = 20 comparisons of a
// perfectly clean engine, P(at least one "regression") = 1 - (1 - 0.05)^20 ~= 0.64.
// A gate that cries wolf on two of every three clean runs is one its operators
// learn to ignore, and an ignored gate is worse than no gate — it launders the
// real regression, when it finally lands, as "probably another false alarm".
//
// The controller therefore makes every family DECLARE two things and then holds
// it to them:
//
//   - a CORRECTION with a documented bound. Holm-Bonferroni gives STRONG control
//     of the family-wise error rate — P(any false rejection) <= alpha under every
//     configuration of true nulls. Benjamini-Hochberg controls the false discovery
//     rate — E[false rejections / rejections] <= alpha for independent or PRDS
//     p-values — which keeps far more power on a wide grid. Both are exact step
//     procedures over the sorted p-values, so a decision is a pure function of the
//     family and replays identically.
//   - a set of PRIMARY metrics, everything else secondary and adjudicated only
//     behind a serial gate: the secondary family is tested AT ALL only when the
//     primary family produced a rejection. Under the global null that gate opens
//     with probability <= alpha, so the whole grid inherits the primary family's
//     bound instead of spending the error budget twice.
//
// The layer is additive in the same sense as release_gate.go and suite_split.go:
// it registers no oracle and edits no core, consuming only the Evidence /
// Divergence / FailureBundle the spine already emits. It is fail-closed wherever
// the epic requires it — a comparison whose provenance is incomplete, whose tier
// is undeclared, whose runtime cost is undocumented, whose p-value is
// uninterpretable, or which declares a discovery it cannot hand over a scrubbed
// replay artifact for, BLOCKS. Missing and inconclusive evidence are never a pass.
//
// The assumption worth stating, because it is the one that can be wrong: the
// serial gate buys the family-wise bound with secondary POWER. A genuine
// secondary-metric regression on a run whose primary metrics are all clean is
// reported as GATED, not as a discovery — "not tested", which is honest, but is
// not the same claim as "not there". A suite that cares about a metric enough to
// block on it must declare that metric primary.

// MultiplicityDecisionSchema is the versioned tag on a multiplicity decision.
// Consumers pin the major so a schema bump is a conscious migration (the #4519
// house rule), not a silent field drift.
const MultiplicityDecisionSchema = "fak-quality-multiplicity/1"

// MultiplicityCorrection is the closed set of correction policies a family may
// declare. Each names the error rate it bounds; CorrectionNone names the absence
// of one and is admitted only so the cost of skipping correction can be MEASURED
// (see the null simulation in the test file) — a family that declares it is
// refused a pass.
type MultiplicityCorrection string

const (
	// CorrectionHolm is Holm-Bonferroni step-down: strong family-wise control,
	// uniformly more powerful than plain Bonferroni, no independence assumption.
	CorrectionHolm MultiplicityCorrection = "holm"
	// CorrectionBenjaminiHochberg is BH step-up: false-discovery-rate control for
	// independent or PRDS p-values, the right trade on a wide model x slice grid
	// where insisting on zero false alarms costs every real detection.
	CorrectionBenjaminiHochberg MultiplicityCorrection = "benjamini-hochberg"
	// CorrectionNone applies no correction. It bounds nothing.
	CorrectionNone MultiplicityCorrection = "none"
)

// MultiplicityPolicy is the declaration a family is held to: the error budget,
// the correction that spends it, and which metrics are PRIMARY. Declaring the
// primary set is mandatory — the hierarchy is what the secondary gate opens
// behind, and a family with no primary has nothing to gate on.
type MultiplicityPolicy struct {
	Alpha      float64                `json:"alpha"`
	Correction MultiplicityCorrection `json:"correction"`
	Primary    []string               `json:"primary_metrics"`
}

// Validate refuses a policy that cannot bound anything: an alpha outside (0, 1),
// an unknown correction, or an undeclared primary metric set.
func (p MultiplicityPolicy) Validate() error {
	if !(p.Alpha > 0 && p.Alpha < 1) {
		return fmt.Errorf("alpha %v must be in the open interval (0, 1)", p.Alpha)
	}
	switch p.Correction {
	case CorrectionHolm, CorrectionBenjaminiHochberg, CorrectionNone:
	default:
		return fmt.Errorf("correction %q is not one of %q, %q, %q",
			p.Correction, CorrectionHolm, CorrectionBenjaminiHochberg, CorrectionNone)
	}
	if len(p.Primary) == 0 {
		return fmt.Errorf("no primary metric declared: the secondary family has nothing to gate behind")
	}
	for _, m := range p.Primary {
		if strings.TrimSpace(m) == "" {
			return fmt.Errorf("primary metric names must be non-empty")
		}
	}
	return nil
}

func (p MultiplicityPolicy) isPrimary(metric string) bool {
	for _, m := range p.Primary {
		if m == metric {
			return true
		}
	}
	return false
}

// MetricComparison is one cell of the grid: the p-value one (model, slice,
// metric) comparison produced, the tier and runtime cost of producing it, and the
// spine Evidence it came from. Evidence carries the provenance the epic requires
// (model, tokenizer, engine/backend, seed or deterministic oracle, code revision,
// tolerance/baseline) plus — on a failing run — the localized first divergence and
// the scrubbed replay bundle, so a discovery arrives already actionable.
// EvidenceFromResult (release_gate.go) is the seam that builds it from a RunCase
// Result.
type MetricComparison struct {
	Model       string   `json:"model"`
	Slice       string   `json:"slice"`
	Metric      string   `json:"metric"`
	P           float64  `json:"p_value"`
	Tier        Tier     `json:"tier"`
	CostSeconds float64  `json:"cost_seconds"`
	Evidence    Evidence `json:"evidence"`
}

func (c MetricComparison) cell() string { return c.Model + "/" + c.Slice + "/" + c.Metric }

// ComparisonDecision is one comparison's adjudicated outcome: its adjusted
// p-value, whether the correction rejected it, whether the hierarchical gate left
// it untested, and — when it blocks — the reason plus the divergence and replay
// artifact an operator acts on. Rejected is the PURE statistical outcome and is
// never overwritten by an administrative refusal, so a null simulation can read
// the procedure's own error rate off it.
type ComparisonDecision struct {
	Index           int            `json:"index"`
	Cell            string         `json:"cell"`
	Model           string         `json:"model"`
	Slice           string         `json:"slice"`
	Metric          string         `json:"metric"`
	CaseID          string         `json:"case_id"`
	Primary         bool           `json:"primary"`
	Tier            Tier           `json:"tier"`
	CostSeconds     float64        `json:"cost_seconds"`
	P               float64        `json:"p_value"`
	Adjusted        float64        `json:"adjusted_p"`
	Rejected        bool           `json:"rejected"`
	Gated           bool           `json:"gated"`
	State           EvidenceState  `json:"state"`
	Reason          string         `json:"reason"`
	FirstDivergence *Divergence    `json:"first_divergence,omitempty"`
	Replay          *FailureBundle `json:"replay,omitempty"`
}

// TierCost is one tier's share of the family's documented evidence cost, so an
// operator can read what the grid costs per cadence rather than as one number
// (#4568 acceptance: assign a tier and document runtime/resource cost).
type TierCost struct {
	Tier        Tier    `json:"tier"`
	Comparisons int     `json:"comparisons"`
	CostSeconds float64 `json:"cost_seconds"`
}

// MultiplicityDecision is the machine-readable output of the controller: the
// declared policy, the bound that policy actually buys, every comparison's
// decision, the blocking subset ordered strongest-evidence-first, any family-level
// refusal, and the per-tier cost. Pass is true iff nothing blocked and nothing was
// refused.
type MultiplicityDecision struct {
	Schema      string               `json:"schema"`
	Policy      MultiplicityPolicy   `json:"policy"`
	Bound       string               `json:"bound"`
	Comparisons int                  `json:"comparisons"`
	Tested      int                  `json:"tested"`
	Gated       int                  `json:"gated"`
	Decisions   []ComparisonDecision `json:"decisions"`
	Blocks      []ComparisonDecision `json:"blocks,omitempty"`
	Refusals    []string             `json:"refusals,omitempty"`
	Cost        []TierCost           `json:"cost"`
	Pass        bool                 `json:"pass"`
}

// ControlMultiplicity adjudicates one family of comparisons under one policy. It
// admits each comparison (provenance, tier, cost, interpretable p-value), applies
// the declared correction to the PRIMARY sub-family, opens the secondary gate only
// if the primary family rejected, and folds the result into a decision whose
// blocking entries lead with the strongest evidence. It is a pure function of
// (policy, family) — same inputs, same decision — so a multiplicity verdict
// replays.
func ControlMultiplicity(policy MultiplicityPolicy, family []MetricComparison) (MultiplicityDecision, error) {
	if err := policy.Validate(); err != nil {
		return MultiplicityDecision{}, fmt.Errorf("multiplicity policy: %w", err)
	}
	if len(family) == 0 {
		return MultiplicityDecision{}, fmt.Errorf("multiplicity family is empty: a family of no comparisons proves nothing")
	}

	d := MultiplicityDecision{
		Schema:      MultiplicityDecisionSchema,
		Policy:      policy,
		Comparisons: len(family),
		Decisions:   make([]ComparisonDecision, len(family)),
		Cost:        rollupTierCost(family),
	}
	var primary, secondary []int
	for i, c := range family {
		dec := ComparisonDecision{
			Index: i, Cell: c.cell(), Model: c.Model, Slice: c.Slice, Metric: c.Metric,
			CaseID: c.Evidence.CaseID, Primary: policy.isPrimary(c.Metric), Tier: c.Tier,
			CostSeconds: c.CostSeconds, P: c.P, Adjusted: 1, State: StatePass,
			FirstDivergence: c.Evidence.FirstDivergence, Replay: c.Evidence.Replay,
		}
		if why, ok := admitComparison(c); !ok {
			dec.State, dec.Reason = StateInconclusive, why
			d.Decisions[i] = dec
			continue
		}
		d.Decisions[i] = dec
		if dec.Primary {
			primary = append(primary, i)
		} else {
			secondary = append(secondary, i)
		}
	}

	rejectedPrimary := 0
	if len(primary) > 0 {
		rejectedPrimary = applyMultiplicityCorrection(policy, d.Decisions, primary)
		d.Tested = len(primary)
	} else {
		d.Refusals = append(d.Refusals, fmt.Sprintf(
			"no admissible comparison carries a declared primary metric (%s): the primary family is what the secondary gate opens behind, so the grid cannot be adjudicated",
			strings.Join(policy.Primary, ", ")))
	}
	if len(secondary) > 0 {
		if rejectedPrimary > 0 {
			applyMultiplicityCorrection(policy, d.Decisions, secondary)
			d.Tested += len(secondary)
		} else {
			for _, i := range secondary {
				d.Decisions[i].Gated = true
				d.Decisions[i].Reason = "secondary metric not tested: the primary family produced no rejection, so the hierarchical gate stayed closed"
			}
			d.Gated = len(secondary)
		}
	}
	d.Bound = boundStatement(policy, len(primary), len(secondary), rejectedPrimary > 0)

	if policy.Correction == CorrectionNone && d.Tested > 0 {
		d.Refusals = append(d.Refusals, fmt.Sprintf(
			"policy declares correction %q: an uncorrected family of %d comparison(s) honors no documented family-wise or FDR bound — at per-comparison alpha %.4f the global null alone yields P(>=1 false discovery) = %.4f — so the family's verdict is inconclusive, never a pass",
			CorrectionNone, d.Tested, policy.Alpha, 1-math.Pow(1-policy.Alpha, float64(d.Tested))))
	}

	for i := range d.Decisions {
		explainComparison(&d.Decisions[i], policy)
		if d.Decisions[i].State != StatePass {
			d.Blocks = append(d.Blocks, d.Decisions[i])
		}
	}
	// Strongest evidence first: a rejection an operator can act on outranks a
	// comparison that could not be interpreted, and among rejections the smallest
	// adjusted p-value is the least likely to be noise. Index breaks every tie, so
	// the order is total and the plan replays.
	sort.SliceStable(d.Blocks, func(a, b int) bool {
		x, y := d.Blocks[a], d.Blocks[b]
		if x.Rejected != y.Rejected {
			return x.Rejected
		}
		if x.Rejected && x.Adjusted != y.Adjusted {
			return x.Adjusted < y.Adjusted
		}
		return x.Index < y.Index
	})
	d.Pass = len(d.Blocks) == 0 && len(d.Refusals) == 0
	return d, nil
}

// admitComparison is the fail-closed admission gate for one cell. It returns the
// FIRST unmet requirement so a refusal is actionable rather than a bare
// "inconclusive". Everything it checks is an #4568 acceptance criterion: an
// explicit tier and documented cost, the full execution/baseline provenance, an
// interpretable p-value, and evidence that was actually produced.
func admitComparison(c MetricComparison) (string, bool) {
	switch {
	case strings.TrimSpace(c.Model) == "":
		return "comparison declares no model", false
	case strings.TrimSpace(c.Slice) == "":
		return "comparison declares no slice", false
	case strings.TrimSpace(c.Metric) == "":
		return "comparison declares no metric", false
	}
	switch c.Tier {
	case TierPR, TierNightly, TierRelease:
	default:
		return fmt.Sprintf("tier %q is not one of pr, nightly, release: every comparison must be assigned to an explicit tier", c.Tier), false
	}
	if !(c.CostSeconds > 0) || math.IsInf(c.CostSeconds, 0) {
		return fmt.Sprintf("cost_seconds %v: the runtime cost of the evidence must be documented as a positive finite number", c.CostSeconds), false
	}
	if ok, why := c.Evidence.Provenance.complete(); !ok {
		return "incomplete provenance: " + why, false
	}
	switch c.Evidence.State {
	case StatePass, StateFail:
	default:
		return fmt.Sprintf("underlying evidence state %q: only produced pass/fail evidence carries an interpretable p-value", c.Evidence.State), false
	}
	if math.IsNaN(c.P) || c.P < 0 || c.P > 1 {
		return fmt.Sprintf("p-value %v is not in [0, 1]: an uninterpretable p-value cannot be corrected, so the comparison is inconclusive", c.P), false
	}
	return "", true
}

// applyMultiplicityCorrection writes the adjusted p-value and rejection flag for
// one sub-family (the indices idx into decs) and returns how many it rejected.
//
// Both procedures are expressed as ADJUSTED p-values compared to alpha, which is
// exactly equivalent to their step formulations and makes every cell's own margin
// readable:
//
//	Holm (step-down):  adj_(k) = max_{j<=k} min(1, (m - j + 1) * p_(j))
//	BH   (step-up):    adj_(k) = min_{j>=k} min(1, (m / j) * p_(j))
//
// The running max / running min are what enforce monotonicity, and without them
// the comparison to alpha would NOT reproduce the step procedures: Holm must stop
// at the first non-rejection, and BH must reject everything below its largest
// passing rank.
func applyMultiplicityCorrection(policy MultiplicityPolicy, decs []ComparisonDecision, idx []int) int {
	m := len(idx)
	order := append([]int(nil), idx...)
	sort.SliceStable(order, func(a, b int) bool {
		if decs[order[a]].P != decs[order[b]].P {
			return decs[order[a]].P < decs[order[b]].P
		}
		return order[a] < order[b]
	})

	adj := make([]float64, m)
	switch policy.Correction {
	case CorrectionHolm:
		running := 0.0
		for k := 0; k < m; k++ {
			if v := math.Min(1, float64(m-k)*decs[order[k]].P); v > running {
				running = v
			}
			adj[k] = running
		}
	case CorrectionBenjaminiHochberg:
		running := 1.0
		for k := m - 1; k >= 0; k-- {
			if v := math.Min(1, float64(m)/float64(k+1)*decs[order[k]].P); v < running {
				running = v
			}
			adj[k] = running
		}
	default: // CorrectionNone: the raw per-comparison p-value, bounding nothing.
		for k := 0; k < m; k++ {
			adj[k] = decs[order[k]].P
		}
	}

	rejected := 0
	for k, i := range order {
		decs[i].Adjusted = adj[k]
		decs[i].Rejected = adj[k] <= policy.Alpha
		if decs[i].Rejected {
			rejected++
		}
	}
	return rejected
}

// explainComparison stamps the operator-facing reason on one adjudicated cell and
// promotes a rejection to a blocking state. A rejection is a DECLARED REGRESSION,
// so it must arrive replayable: a discovery whose evidence carries no scrubbed
// replay artifact is downgraded to inconclusive — it still blocks, but it is
// labelled as evidence nobody can independently reproduce rather than as an
// actionable finding.
func explainComparison(d *ComparisonDecision, policy MultiplicityPolicy) {
	switch {
	case d.State == StateInconclusive: // admission already recorded the reason
		return
	case d.Gated:
		return // the gate reason is already stamped
	case !d.Rejected:
		d.Reason = fmt.Sprintf("no discovery: %s-adjusted p = %.4g > alpha %.4f (raw p = %.4g)",
			policy.Correction, d.Adjusted, policy.Alpha, d.P)
		return
	}
	d.State = StateFail
	d.Reason = fmt.Sprintf("regression declared: %s-adjusted p = %.4g <= alpha %.4f (raw p = %.4g)",
		policy.Correction, d.Adjusted, policy.Alpha, d.P)
	if d.FirstDivergence != nil {
		d.Reason += fmt.Sprintf("; first divergence at token %d (reference %q, engine %q)",
			d.FirstDivergence.Index, d.FirstDivergence.Reference, d.FirstDivergence.Engine)
	}
	if d.Replay == nil || !d.Replay.Scrubbed {
		d.State = StateInconclusive
		d.Reason += "; but no scrubbed replay artifact travels with it, so the discovery cannot be independently reproduced and is not actionable evidence"
	}
}

// boundStatement writes down, in words, the bound the declared policy actually
// buys for THIS family shape. It is emitted on the decision so a consumer reads
// the guarantee off the artifact instead of assuming one.
func boundStatement(policy MultiplicityPolicy, primary, secondary int, primaryRejected bool) string {
	var head string
	switch policy.Correction {
	case CorrectionHolm:
		head = fmt.Sprintf("family-wise error rate <= %.4f: holm-bonferroni step-down over %d primary comparison(s), strong control under any configuration of true nulls",
			policy.Alpha, primary)
	case CorrectionBenjaminiHochberg:
		head = fmt.Sprintf("false discovery rate <= %.4f: benjamini-hochberg step-up over %d primary comparison(s), for independent or PRDS p-values",
			policy.Alpha, primary)
	default:
		head = fmt.Sprintf("NO documented bound: %d primary comparison(s) judged at the raw per-comparison alpha %.4f",
			primary, policy.Alpha)
	}
	if secondary == 0 {
		return head
	}
	if primaryRejected {
		return head + fmt.Sprintf("; the primary family rejected, so the hierarchical gate opened and the same correction was applied to the %d secondary comparison(s) as a separate family",
			secondary)
	}
	return head + fmt.Sprintf("; %d secondary comparison(s) were gated untested — under the global null the gate opens with probability <= alpha, so the grid inherits the primary bound",
		secondary)
}

// rollupTierCost sums the family's documented evidence cost per tier, in the fixed
// pr / nightly / release order with any undeclared tier last, so the readout is
// deterministic. Every comparison counts, including the inconclusive ones: their
// cost was paid whether or not their evidence could be used.
func rollupTierCost(family []MetricComparison) []TierCost {
	counts := map[Tier]*TierCost{}
	var seen []Tier
	for _, c := range family {
		tc, ok := counts[c.Tier]
		if !ok {
			tc = &TierCost{Tier: c.Tier}
			counts[c.Tier] = tc
			seen = append(seen, c.Tier)
		}
		tc.Comparisons++
		if c.CostSeconds > 0 && !math.IsInf(c.CostSeconds, 0) {
			tc.CostSeconds += c.CostSeconds
		}
	}
	rank := map[Tier]int{TierPR: 0, TierNightly: 1, TierRelease: 2}
	sort.SliceStable(seen, func(a, b int) bool {
		ra, oka := rank[seen[a]]
		rb, okb := rank[seen[b]]
		if oka != okb {
			return oka
		}
		if oka && ra != rb {
			return ra < rb
		}
		return seen[a] < seen[b]
	})
	out := make([]TierCost, 0, len(seen))
	for _, t := range seen {
		out = append(out, *counts[t])
	}
	return out
}

// ExplainMultiplicity renders a decision as an operator readout: the bound the
// family bought, every family-level refusal, each blocking cell strongest-first
// with its divergence and replay artifact, and the per-tier cost. It mirrors
// ExplainRelease (release_gate.go) and ExplainPlan (suite_split.go) — the bridge
// from a machine decision to "here is which cell to open first, and what the grid
// cost to learn it".
func ExplainMultiplicity(d MultiplicityDecision) string {
	var b strings.Builder
	verdict := "CONTROLLED"
	if !d.Pass {
		verdict = "BLOCKED"
	}
	fmt.Fprintf(&b, "%s  %d comparison(s): %d tested, %d gated\n", verdict, d.Comparisons, d.Tested, d.Gated)
	fmt.Fprintf(&b, "  bound: %s\n", d.Bound)
	for _, r := range d.Refusals {
		fmt.Fprintf(&b, "  refused: %s\n", r)
	}
	for i, blk := range d.Blocks {
		marker := "  "
		if i == 0 {
			marker = "->" // the first actionable divergence
		}
		fmt.Fprintf(&b, "%s %-8s %-13s %s\n", marker, blk.Tier, blk.State, blk.Cell)
		fmt.Fprintf(&b, "     reason: %s\n", blk.Reason)
		if r := blk.Replay; r != nil {
			fmt.Fprintf(&b, "     replay: case %s, failing oracle %s (%s), scrubbed=%t\n",
				r.CaseID, r.FailingOracle, r.FailingKind, r.Scrubbed)
		}
	}
	for _, tc := range d.Cost {
		fmt.Fprintf(&b, "  cost %-8s %d comparison(s), %.1fs\n", tc.Tier, tc.Comparisons, tc.CostSeconds)
	}
	return b.String()
}
