package quality

import (
	"fmt"
	"math"
	"strings"
)

// This file PRICES POST-SELECTION SEARCH (#5666, under epic #1279): the layer that
// refuses to read confidence measured at the winning point of a searched sweep as
// if it had been measured at a point declared before looking.
//
// The RSI loop proposes, witnesses, and keeps. Proposing means SEARCHING — over
// thresholds, over variants, over which comparison to run — and keeping means
// retaining the most favorable arm the search found. The p-value that arm reports
// is conditioned on having been selected as the winner, and that conditioning is
// not free: over m independent null arms judged at alpha, P(min p <= alpha) =
// 1 - (1 - alpha)^m, so a sweep of 20 clean thresholds at alpha = 0.05 hands back a
// "significant improvement" about 64% of the time. Search breadth alone
// manufactures the passing claim. The winner's curse is the same effect on the
// effect size: the arm that won won partly because its noise was favorable.
//
// The controller therefore makes a sweep DECLARE its complete search family and
// then prices the winner against it, exposing BOTH numbers side by side:
//
//   - the UNADJUSTED p at the retained arm — what a naive read reports, and what
//     an operator will otherwise quote;
//   - the SELECTION-ADJUSTED p over the whole family — Sidak (1 - (1 - p)^m, exact
//     for independent arms) or Bonferroni (min(1, m*p), valid under any dependence).
//
// The gap between them is the PRICE of the search, and the case where the first
// certifies and the second does not is flagged in its own field: that is exactly
// the claim breadth manufactured, and naming it is the point of the layer.
//
// It is additive in the same sense as multiple_comparisons.go and release_gate.go:
// it registers no oracle, edits no core, and consumes only the Evidence the spine
// already emits. It is fail-closed everywhere the family cannot be established —
// an undeclared family, a declared count that disagrees with the arms handed over,
// a winner that is not in the family or is not the arm the declared rule would have
// picked, an arm whose evidence cannot be interpreted, or an adjustment that prices
// nothing all REFUSE. A refusal is never a pass, and unlike the sibling controllers
// every one of them rides on the result as a typed, machine-readable refusal rather
// than as a Go error, because the receipt has to record the refusal too.
//
// The assumption worth stating, because it is the one that can be wrong: Sidak is
// EXACT only when the arms are independent. A sweep over a monotone threshold grid
// evaluated on one shared sample has strongly positively dependent arms, and Sidak
// is anti-conservative there — it prices the search too cheaply. Bonferroni holds
// under arbitrary dependence and is the honest default for a threshold sweep; the
// declaration is recorded on the decision so a reader can check which was claimed.

// PostSelectionSchema is the versioned tag on a post-selection decision. Consumers
// pin the major so a schema bump is a conscious migration (the #4519 house rule),
// not a silent field drift.
const PostSelectionSchema = "fak-quality-postselection/1"

// SelectionRule is the closed set of rules by which a sweep may pick the arm it
// retains. The rule is what makes the price computable: pricing the minimum of m
// p-values is only correct if the minimum is what was actually kept.
type SelectionRule string

const (
	// SelectMinP retains the most significant arm. This is a search, and it is
	// priced over the whole family.
	SelectMinP SelectionRule = "min-p"
	// SelectPredeclared names the absence of a search: exactly one point, fixed
	// before any of the evidence was seen. It is priced at m = 1 — that is, not at
	// all — which is precisely the claim a sweep is not entitled to make.
	SelectPredeclared SelectionRule = "predeclared"
)

// SelectionAdjustment is the closed set of policies that may price the search.
// AdjustNone is admitted only so the cost of skipping the price can be MEASURED
// (see the null simulation in the test file) — a sweep that declares it is refused.
type SelectionAdjustment string

const (
	// AdjustSidak is the Sidak correction on the minimum p-value, 1 - (1 - p)^m.
	// Exact for INDEPENDENT arms; anti-conservative under positive dependence.
	AdjustSidak SelectionAdjustment = "sidak"
	// AdjustBonferroni is min(1, m*p): valid under arbitrary dependence between
	// arms, and therefore the honest default for a sweep over one shared sample.
	AdjustBonferroni SelectionAdjustment = "bonferroni"
	// AdjustNone prices nothing. It bounds nothing.
	AdjustNone SelectionAdjustment = "none"
)

// PostSelectionRefusalCode is the closed vocabulary of typed refusals the contract
// uses. A structured code is what lets a consumer branch on WHY a certification was
// withheld instead of pattern-matching prose.
type PostSelectionRefusalCode string

const (
	// RefuseCertificationInvalid: the declaration itself cannot bound anything —
	// an alpha outside (0, 1), an unknown rule or adjustment, or no retained arm
	// named.
	RefuseCertificationInvalid PostSelectionRefusalCode = "certification_invalid"
	// RefuseFamilyUndeclared: no search family was handed over, or its size was
	// never declared. The winner cannot be priced against a family that does not
	// exist on the record.
	RefuseFamilyUndeclared PostSelectionRefusalCode = "search_family_undeclared"
	// RefuseFamilyIncomplete: the declared family size disagrees with the arms
	// handed over. Fewer arms than declared means evidence was evaluated and not
	// reported (the file drawer) and the selection cannot be audited; more arms
	// than declared means the declaration understates the breadth actually searched
	// and would underprice it.
	RefuseFamilyIncomplete PostSelectionRefusalCode = "search_family_incomplete"
	// RefuseWinnerNotInFamily: the retained arm is not one of the arms handed over,
	// so the family the price is computed over is not the family it was drawn from.
	RefuseWinnerNotInFamily PostSelectionRefusalCode = "winner_not_in_family"
	// RefuseWinnerNotExtremal: the retained arm is not the arm the declared rule
	// would have picked. The selection rule on the record is then not the rule that
	// was used, and the adjustment derived from it does not apply.
	RefuseWinnerNotExtremal PostSelectionRefusalCode = "winner_not_extremal"
	// RefuseArmInadmissible: an arm's evidence cannot be interpreted (missing
	// provenance, unproduced evidence, a p-value outside [0, 1]). An uninterpretable
	// arm leaves the true breadth of the search unknown, so the family is refused
	// rather than silently shrunk to the arms that happened to parse.
	RefuseArmInadmissible PostSelectionRefusalCode = "arm_inadmissible"
	// RefuseAdjustmentUndeclared: the sweep searched more than one point and
	// declared no adjustment, so the reported confidence honors no bound at all.
	RefuseAdjustmentUndeclared PostSelectionRefusalCode = "selection_adjustment_undeclared"
)

// PostSelectionRefusal is one typed reason the contract withheld a conclusion.
// Detail localizes it to the specific arm or field so the refusal is actionable.
type PostSelectionRefusal struct {
	Code   PostSelectionRefusalCode `json:"code"`
	Detail string                   `json:"detail"`
}

// SearchArm is one candidate point the sweep evaluated: the threshold, variant, or
// comparison it names, the p-value that point produced, and the spine Evidence it
// came from. Every arm the search touched belongs here, including the ones that
// lost — the losers are the family, and the family is what the winner is priced
// against.
type SearchArm struct {
	ID       string   `json:"id"`
	Point    string   `json:"point"`
	P        float64  `json:"p_value"`
	Evidence Evidence `json:"evidence"`
}

// SweepCertification is the declaration a sweep is held to: the error budget, the
// rule that picked the winner, the adjustment that prices the search, the size of
// the COMPLETE family that was searched, and which arm was retained. DeclaredArms
// is stated separately from the arms handed over on purpose — it is the operator's
// assertion about how wide the search was, and cross-checking it against the arms
// on the record is what catches an understated family.
type SweepCertification struct {
	Alpha        float64             `json:"alpha"`
	Rule         SelectionRule       `json:"rule"`
	Adjustment   SelectionAdjustment `json:"adjustment"`
	DeclaredArms int                 `json:"declared_arms"`
	Winner       string              `json:"winner"`
}

// ArmDecision is one arm's adjudicated record: what it was, what it reported,
// whether it was the retained arm, and — when it could not be interpreted — why.
type ArmDecision struct {
	Index    int     `json:"index"`
	ID       string  `json:"id"`
	Point    string  `json:"point"`
	P        float64 `json:"p_value"`
	Winner   bool    `json:"winner"`
	Admitted bool    `json:"admitted"`
	Reason   string  `json:"reason,omitempty"`
}

// PostSelectionDecision is the machine-readable output of the controller: the
// declaration, the family it was checked against, every arm, and — the point of
// the layer — the unadjusted and selection-adjusted results side by side with the
// price of the search between them.
//
// Certified is true only when nothing was refused AND the SELECTION-ADJUSTED
// p-value clears alpha. UnadjustedCertifies records what the naive read would have
// concluded, so the two are never collapsed into one number.
type PostSelectionDecision struct {
	Schema        string             `json:"schema"`
	Certification SweepCertification `json:"certification"`
	// FamilySize is the number of arms actually handed over, which is what the
	// adjustment is computed over once the declaration has been reconciled with it.
	FamilySize int          `json:"family_size"`
	Winner     *ArmDecision `json:"winner,omitempty"`
	// Unadjusted is the retained arm's raw p-value: confidence at the winning point,
	// ignoring that the point was chosen by looking.
	Unadjusted float64 `json:"unadjusted_p"`
	// Adjusted is that p-value priced over the complete search family. On a refusal
	// it is pinned at 1 — no family was established, so no bound was bought.
	Adjusted float64 `json:"selection_adjusted_p"`
	// SelectionPrice is Adjusted - Unadjusted: what the search breadth cost, in
	// p-value, stated as its own number so it can be tracked over time.
	SelectionPrice      float64 `json:"selection_price"`
	UnadjustedCertifies bool    `json:"unadjusted_certifies"`
	Certified           bool    `json:"certified"`
	// ManufacturedByBreadth is the headline finding: the sweep's unadjusted result
	// clears alpha and its priced result does not, so the passing claim was produced
	// by how widely it searched rather than by the improvement it found. It is only
	// ever set on a decision that was priced — a refusal is a different failure and
	// is not laundered into this one.
	ManufacturedByBreadth bool                   `json:"manufactured_by_breadth"`
	Bound                 string                 `json:"bound"`
	Arms                  []ArmDecision          `json:"arms"`
	Refusals              []PostSelectionRefusal `json:"refusals,omitempty"`
}

// PriceSearch adjudicates one searched sweep against its declaration. It admits
// every arm, reconciles the declared family size with the arms on the record,
// checks that the retained arm is the one the declared rule would have picked, and
// prices the winner's p-value over the complete family.
//
// It returns no error: every way the contract can fail is a typed refusal ON the
// decision, because the receipt has to record refusals as faithfully as it records
// passes. A caller that ignores Refusals entirely and reads only Certified still
// fails closed — Certified is false whenever anything was refused.
//
// It is a pure function of (certification, family): same inputs, same decision, so
// a certification replays.
func PriceSearch(cert SweepCertification, family []SearchArm) PostSelectionDecision {
	d := PostSelectionDecision{
		Schema:        PostSelectionSchema,
		Certification: cert,
		FamilySize:    len(family),
		Adjusted:      1,
		Arms:          make([]ArmDecision, len(family)),
	}

	// 1. The declaration itself. A sweep that cannot state a budget, a rule, an
	// adjustment, and which arm it kept has declared nothing to hold it to.
	if !(cert.Alpha > 0 && cert.Alpha < 1) {
		d.refuse(RefuseCertificationInvalid, fmt.Sprintf("alpha %v must be in the open interval (0, 1)", cert.Alpha))
	}
	switch cert.Rule {
	case SelectMinP, SelectPredeclared:
	default:
		d.refuse(RefuseCertificationInvalid, fmt.Sprintf("selection rule %q is not one of %q, %q", cert.Rule, SelectMinP, SelectPredeclared))
	}
	switch cert.Adjustment {
	case AdjustSidak, AdjustBonferroni, AdjustNone:
	default:
		d.refuse(RefuseCertificationInvalid, fmt.Sprintf("selection adjustment %q is not one of %q, %q, %q",
			cert.Adjustment, AdjustSidak, AdjustBonferroni, AdjustNone))
	}
	if strings.TrimSpace(cert.Winner) == "" {
		d.refuse(RefuseCertificationInvalid, "no retained arm named: a certification must say which point of the sweep it kept")
	}

	// 2. The family. Both directions of a size mismatch are refused: fewer arms on
	// the record than declared means the selection cannot be audited, and more means
	// the declaration would underprice the breadth actually searched.
	switch {
	case len(family) == 0:
		d.refuse(RefuseFamilyUndeclared, "no search arms handed over: the winner cannot be priced against a family that is not on the record")
	case cert.DeclaredArms <= 0:
		d.refuse(RefuseFamilyUndeclared, fmt.Sprintf(
			"declared_arms %d: the size of the complete search family must be declared as a positive count", cert.DeclaredArms))
	case cert.DeclaredArms != len(family):
		d.refuse(RefuseFamilyIncomplete, fmt.Sprintf(
			"declared_arms %d but %d arm(s) handed over: %s", cert.DeclaredArms, len(family),
			familyMismatchDetail(cert.DeclaredArms, len(family))))
	case cert.Rule == SelectPredeclared && len(family) != 1:
		d.refuse(RefuseFamilyIncomplete, fmt.Sprintf(
			"rule %q asserts a single point fixed before the evidence was seen, but %d arm(s) were searched: a sweep is not a predeclaration",
			SelectPredeclared, len(family)))
	}

	// 3. The arms. An arm nobody can interpret leaves the true breadth unknown.
	minP, haveMin := math.Inf(1), false
	var winner *ArmDecision
	for i, a := range family {
		dec := ArmDecision{Index: i, ID: a.ID, Point: a.Point, P: a.P, Winner: a.ID == cert.Winner, Admitted: true}
		if why, ok := admitArm(a); !ok {
			dec.Admitted, dec.Reason = false, why
			d.refuse(RefuseArmInadmissible, fmt.Sprintf("arm %d (%q): %s", i, a.ID, why))
		} else {
			if a.P < minP {
				minP, haveMin = a.P, true
			}
		}
		d.Arms[i] = dec
		if dec.Winner && winner == nil {
			winner = &d.Arms[i]
		}
	}

	// 4. The winner. It must be in the family, and it must be the arm the declared
	// rule would have picked — otherwise the rule on the record is not the rule used.
	if winner == nil {
		if strings.TrimSpace(cert.Winner) != "" {
			d.refuse(RefuseWinnerNotInFamily, fmt.Sprintf(
				"retained arm %q is not among the %d arm(s) handed over: the price would be computed over a family the winner was not drawn from",
				cert.Winner, len(family)))
		}
	} else {
		d.Winner = winner
		d.Unadjusted = winner.P
		d.UnadjustedCertifies = winner.Admitted && winner.P <= cert.Alpha
		if winner.Admitted && haveMin && cert.Rule == SelectMinP && winner.P > minP {
			d.refuse(RefuseWinnerNotExtremal, fmt.Sprintf(
				"rule %q but retained arm %q reports p = %.4g while the family's minimum is %.4g: the declared rule is not the one that picked the winner, so its adjustment does not apply",
				SelectMinP, winner.ID, winner.P, minP))
		}
	}

	// 5. The price. An unpriced multi-arm sweep honors no bound at all; say how much
	// that is worth under the global null rather than leaving it as an assertion.
	if cert.Adjustment == AdjustNone && len(family) > 1 {
		d.refuse(RefuseAdjustmentUndeclared, fmt.Sprintf(
			"adjustment %q over %d searched arm(s): the retained p-value honors no bound — under the global null the minimum of %d independent arms clears alpha %.4f with probability %.4f, so a passing claim here is not evidence of an improvement",
			AdjustNone, len(family), len(family), cert.Alpha, 1-math.Pow(1-cert.Alpha, float64(len(family)))))
	}

	if len(d.Refusals) > 0 {
		d.Bound = "no bound established: " + string(d.Refusals[0].Code) + " — the search family could not be reconciled with the certification, so the retained arm was not priced"
		return d
	}

	d.Adjusted = adjustForSelection(cert.Adjustment, d.Unadjusted, len(family))
	d.SelectionPrice = d.Adjusted - d.Unadjusted
	d.Certified = d.Adjusted <= cert.Alpha
	d.ManufacturedByBreadth = d.UnadjustedCertifies && !d.Certified
	d.Bound = selectionBoundStatement(cert, len(family))
	return d
}

func (d *PostSelectionDecision) refuse(code PostSelectionRefusalCode, detail string) {
	d.Refusals = append(d.Refusals, PostSelectionRefusal{Code: code, Detail: detail})
}

// familyMismatchDetail names which way the ledger and the declaration disagree,
// because the two directions are different failures with different fixes.
func familyMismatchDetail(declared, handed int) string {
	if handed < declared {
		return "arms were evaluated but not reported, so the selection cannot be audited from the record"
	}
	return "the declaration understates the breadth actually searched, which would underprice the selection"
}

// admitArm is the fail-closed admission gate for one candidate point. It returns
// the FIRST unmet requirement so a refusal localizes rather than reading as a bare
// "inadmissible". It mirrors admitComparison (multiple_comparisons.go): the same
// provenance, produced-evidence, and interpretable-p-value bar, since an arm of a
// sweep is a comparison that happens to have lost.
func admitArm(a SearchArm) (string, bool) {
	if strings.TrimSpace(a.ID) == "" {
		return "arm declares no id", false
	}
	if strings.TrimSpace(a.Point) == "" {
		return "arm declares no searched point: the threshold, variant, or comparison it names is what makes the family auditable", false
	}
	if ok, why := a.Evidence.Provenance.complete(); !ok {
		return "incomplete provenance: " + why, false
	}
	switch a.Evidence.State {
	case StatePass, StateFail:
	default:
		return fmt.Sprintf("underlying evidence state %q: only produced pass/fail evidence carries an interpretable p-value", a.Evidence.State), false
	}
	if math.IsNaN(a.P) || a.P < 0 || a.P > 1 {
		return fmt.Sprintf("p-value %v is not in [0, 1]: an uninterpretable p-value cannot be priced, so the family's breadth is unknown", a.P), false
	}
	return "", true
}

// adjustForSelection prices a retained p-value over a family of m searched arms.
// Both procedures reduce to p exactly at m = 1, which is what makes a genuine
// predeclaration cost nothing.
func adjustForSelection(adj SelectionAdjustment, p float64, m int) float64 {
	switch adj {
	case AdjustSidak:
		return math.Min(1, 1-math.Pow(1-p, float64(m)))
	case AdjustBonferroni:
		return math.Min(1, float64(m)*p)
	default:
		return p
	}
}

// ExplainPostSelection renders a PostSelectionDecision as an operator readout. It
// mirrors ExplainRelease (release_gate.go): the bridge from a machine verdict to
// "here is exactly what the search cost you". BOTH readings are always printed, on
// adjacent lines — collapsing them to one number is the confusion the layer exists
// to prevent, so the renderer is not allowed to drop either.
func ExplainPostSelection(d PostSelectionDecision) string {
	var b strings.Builder
	retained := strings.TrimSpace(d.Certification.Winner)
	if retained == "" {
		retained = "(no arm named)"
	}
	if d.Winner != nil {
		retained = fmt.Sprintf("%s [%s]", d.Winner.ID, d.Winner.Point)
	}
	switch {
	case len(d.Refusals) > 0:
		fmt.Fprintf(&b, "REFUSED  retained %s — %d typed refusal(s); the search was never priced\n", retained, len(d.Refusals))
	case d.Certified:
		fmt.Fprintf(&b, "CERTIFIED  retained %s — survived being priced over %d searched arm(s)\n", retained, d.FamilySize)
	default:
		fmt.Fprintf(&b, "NOT CERTIFIED  retained %s — priced over %d searched arm(s)\n", retained, d.FamilySize)
	}
	fmt.Fprintf(&b, "  breadth: %d arm(s) declared, %d on the record; rule %s\n",
		d.Certification.DeclaredArms, d.FamilySize, d.Certification.Rule)
	fmt.Fprintf(&b, "  unadjusted          p = %-10.4g %s\n",
		d.Unadjusted, selectionVerdictWord(d.UnadjustedCertifies, d.Certification.Alpha))
	fmt.Fprintf(&b, "  selection-adjusted  p = %-10.4g %s [%s]\n",
		d.Adjusted, selectionVerdictWord(d.Certified, d.Certification.Alpha), d.Certification.Adjustment)
	fmt.Fprintf(&b, "  price of the search: %+.4g\n", d.SelectionPrice)
	if d.ManufacturedByBreadth {
		b.WriteString("-> MANUFACTURED BY BREADTH: the unadjusted read certifies and the priced read does not, so the passing claim came from how widely the sweep searched rather than from the improvement it found\n")
	}
	fmt.Fprintf(&b, "  bound: %s\n", d.Bound)
	for i, r := range d.Refusals {
		marker := "  "
		if i == 0 {
			marker = "->" // the first thing to fix
		}
		fmt.Fprintf(&b, "%s %s: %s\n", marker, r.Code, r.Detail)
	}
	return b.String()
}

// selectionVerdictWord states one reading against the budget in words, so the two
// p-value lines are readable without the operator re-doing the comparison.
func selectionVerdictWord(certifies bool, alpha float64) string {
	if certifies {
		return fmt.Sprintf("<= alpha %.4g: certifies", alpha)
	}
	return fmt.Sprintf("> alpha %.4g: does not certify", alpha)
}

// selectionBoundStatement writes down, in words, the bound the declaration
// actually buys for THIS sweep shape — including the dependence assumption Sidak
// rests on, which is the one a threshold sweep most often violates. It is emitted
// on the decision so a consumer reads the guarantee off the artifact instead of
// assuming one.
func selectionBoundStatement(cert SweepCertification, m int) string {
	if m == 1 && cert.Rule == SelectPredeclared {
		return fmt.Sprintf("no selection to price: a single point predeclared before the evidence was seen, judged at alpha %.4f", cert.Alpha)
	}
	switch cert.Adjustment {
	case AdjustSidak:
		return fmt.Sprintf("selection-adjusted at alpha %.4f: sidak 1-(1-p)^%d over the complete search family, EXACT for independent arms and anti-conservative under positive dependence (a sweep sharing one sample violates this — prefer bonferroni there)", cert.Alpha, m)
	case AdjustBonferroni:
		return fmt.Sprintf("selection-adjusted at alpha %.4f: bonferroni min(1, %d*p) over the complete search family, valid under arbitrary dependence between arms", cert.Alpha, m)
	default:
		return fmt.Sprintf("NO documented bound: the winner of a %d-arm search judged at the raw alpha %.4f", m, cert.Alpha)
	}
}
