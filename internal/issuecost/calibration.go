// This file is the C9 calibration fold (#3046): join per-issue TIER DECISIONS to
// WITNESSED OUTCOMES and propose — never auto-apply — threshold changes to the
// routing policy. It is the missing "tier decision -> witnessed outcome ->
// calibration report" join the model-tier chain needs so a static score blend
// does not go stale.
//
// # Why a witnessed join, not worker self-report
//
// The whole point is to learn from what THIS repo actually did, without trusting
// a worker's word for it. So an outcome only counts as SUCCESS when it carries a
// non-forgeable witness: a diff-witnessed commit-audit AND green lane tests AND a
// real close. A merely-closed issue with no commit/test witness is NOT success —
// it folds to Stall (the issue's confusion risk: "a closed issue is not enough if
// the commit/test witness is missing").
//
// # The asymmetry this preserves
//
// A cheaper tier that REFUSES high-risk work it may not take is behaving
// correctly; punishing it would teach the policy to over-tier. So a witnessed
// refusal lands in its own CorrectRefuse bucket and never counts as a demerit
// against the tier (the issue's second confusion risk).
//
// # Advisory / shadow only
//
// Calibrate proposes; it never mutates a threshold. Every Recommendation carries
// AutoApply=false. Automatic policy retuning is explicitly out of scope and needs
// a separate keep-bit and a human-visible diff (the issue's stated assumption).
//
// # Purity
//
// Calibrate is a pure fold over the passed decision + outcome rows: no wall clock,
// no I/O, no mutation of the inputs. The same fixtures produce the same report on
// every run, so a hand-verified fixture stays valid. This file adds only stdlib
// imports, keeping the leaf's stdlib+fleetmetrics-only surface.
package issuecost

import (
	"fmt"
	"sort"
)

// Tier is the work/decision tier label in a calibration row: T0 (the MOST
// demanding — ultra-hard / high-risk), T1 (normal implementation), T2 (routine /
// bounded). It mirrors modelroute.WorkTier's labels but stays a plain string so a
// fixture row is human-readable and this leaf keeps its narrow import surface.
type Tier string

const (
	TierT0 Tier = "T0"
	TierT1 Tier = "T1"
	TierT2 Tier = "T2"
)

// tierOrder is the fixed, most-demanding-first order used for deterministic
// rendering and recommendation output.
var tierOrder = []Tier{TierT0, TierT1, TierT2}

// demand returns a tier's demand rank: 0 for T0 (most demanding) up to 2 for T2.
// A LOWER rank is MORE demanding — the same numbering trap modelroute warns about,
// confined here to this one helper so no caller does a raw `<` on the label.
func (t Tier) demand() int {
	switch t {
	case TierT0:
		return 0
	case TierT1:
		return 1
	case TierT2:
		return 2
	default:
		return 3 // unknown tiers sort last and never read as "cheapest routine"
	}
}

// Valid reports whether t is one of the three defined tiers.
func (t Tier) Valid() bool { return t == TierT0 || t == TierT1 || t == TierT2 }

// moreDemandingThan reports whether t is strictly more demanding than u — i.e.
// running t on work whose optimal tier is u is over-tier waste.
func (t Tier) moreDemandingThan(u Tier) bool { return t.demand() < u.demand() }

// TierDecision is one routing decision the dispatcher made for an issue: the tier
// it actually ran (Chosen), the risk floor it had to meet (Required), and the
// ideal fit (Optimal). Chosen more demanding than Optimal is the over-tier case.
type TierDecision struct {
	Issue    int  `json:"issue"`
	Chosen   Tier `json:"chosen"`
	Required Tier `json:"required"`
	Optimal  Tier `json:"optimal"`
}

// WitnessedOutcome is what THIS repo witnessed for an issue after the decision ran.
// Every bool names a witness source (see WitnessSource): CommitWitnessed is a
// diff-witnessed commit-audit; TestsGreen is a green lane/CI gate; Closed is the
// close-gate; Reverted is a git revert or issue reopen; Escalated is a dispatch-
// ledger bump to a more capable tier; Refused is a witnessed under-tier refusal.
// Turns is elapsed dispatch turns (carried for reporting, not yet a bucket key).
type WitnessedOutcome struct {
	Issue           int  `json:"issue"`
	CommitWitnessed bool `json:"commit_witnessed"`
	TestsGreen      bool `json:"tests_green"`
	Closed          bool `json:"closed"`
	Reverted        bool `json:"reverted"`
	Escalated       bool `json:"escalated"`
	Refused         bool `json:"refused"`
	Turns           int  `json:"turns"`
}

// Bucket is the terminal calibration classification of one joined
// (decision, outcome) pair. It is a closed vocabulary so a report is explainable
// without free text.
type Bucket string

const (
	// BucketSuccess: shipped GREEN and closed with a non-forgeable witness.
	BucketSuccess Bucket = "success"
	// BucketStall: no witnessed ship and no revert — stuck, or closed without a
	// commit/test witness (which does not earn success).
	BucketStall Bucket = "stall"
	// BucketEscalation: the chosen tier could not finish and the work was bumped
	// to a more capable tier.
	BucketEscalation Bucket = "escalation"
	// BucketReverted: the work shipped then had to be reverted or the issue was
	// reopened — rework.
	BucketReverted Bucket = "revert-reopen"
	// BucketCorrectRefuse: the chosen (usually cheaper) tier correctly REFUSED
	// work above its floor. This is right behavior, never a demerit.
	BucketCorrectRefuse Bucket = "correct-refuse"
)

// bucketOrder fixes the render/JSON iteration order over buckets.
var bucketOrder = []Bucket{BucketSuccess, BucketStall, BucketEscalation, BucketReverted, BucketCorrectRefuse}

// witnessSource names, per outcome bucket, the non-forgeable evidence that puts a
// pair in that bucket. The report echoes this so an operator sees WHICH witness
// backs each metric — the issue's done-condition requirement.
var witnessSource = map[Bucket]string{
	BucketSuccess:       "diff-witnessed commit-audit + green lane tests + close-gate close",
	BucketStall:         "no witnessed close/ship and no revert in the window",
	BucketEscalation:    "dispatch-ledger escalation to a more capable tier",
	BucketReverted:      "git revert or issue-reopen witness",
	BucketCorrectRefuse: "witnessed under-tier refusal (correct — not a demerit)",
}

// WitnessSource returns the witness-source description for a bucket (empty for an
// unknown bucket). Exported so a CLI or status surface can render the same names.
func WitnessSource(b Bucket) string { return witnessSource[b] }

// classify maps one joined (decision, outcome) pair to exactly one bucket. The
// order is load-bearing:
//
//  1. a witnessed REFUSAL is correct behavior first — never re-read as a stall;
//  2. a REVERT/reopen is rework even if it also closed;
//  3. an ESCALATION means the chosen tier did not carry it;
//  4. SUCCESS requires the full witness (commit + tests + close) — a bare close
//     is not enough;
//  5. everything else is a STALL.
func classify(o WitnessedOutcome) Bucket {
	switch {
	case o.Refused:
		return BucketCorrectRefuse
	case o.Reverted:
		return BucketReverted
	case o.Escalated:
		return BucketEscalation
	case o.Closed && o.CommitWitnessed && o.TestsGreen:
		return BucketSuccess
	default:
		return BucketStall
	}
}

// TierStats is the per-chosen-tier roll-up: how many joined pairs landed in each
// bucket, and OverTierWaste — successes where the chosen tier was more demanding
// than optimal (the cheaper optimal tier would likely have sufficed). N is the
// number of joined pairs at this chosen tier.
type TierStats struct {
	N             int            `json:"n"`
	Buckets       map[Bucket]int `json:"buckets"`
	OverTierWaste int            `json:"over_tier_waste"`
}

// Recommendation is one advisory, human-visible proposal for a chosen tier. Action
// is a closed vocabulary; AutoApply is ALWAYS false — this fold never retunes a
// live threshold (that needs a separate keep-bit and a reviewed diff).
type Recommendation struct {
	Tier      Tier   `json:"tier"`
	Action    string `json:"action"`
	Rationale string `json:"rationale"`
	Evidence  string `json:"evidence"`
	AutoApply bool   `json:"auto_apply"`
}

// Closed-vocabulary recommendation actions.
const (
	// ActionExpandCheaper: over-tier successes with no rework/escalation at this
	// tier — the cheaper optimal tier is a safe expansion candidate for this class.
	ActionExpandCheaper = "expand-cheaper"
	// ActionRaiseFloor: this tier saw rework or escalation — do NOT down-tier; the
	// floor is doing its job (or should rise).
	ActionRaiseFloor = "raise-floor"
	// ActionHold: not enough signal either way — keep the current threshold.
	ActionHold = "hold"
)

// minOverTierEvidence is how many over-tier successes at a chosen tier are needed
// before the fold will propose expanding a cheaper tier for that class. Risk is
// asymmetric: a single revert/escalation withholds a down-tier, but one lucky
// over-tier success is not enough to propose one.
const minOverTierEvidence = 2

// CalibrationReport is the operator-facing fold result. Decisions is how many
// decision rows were seen; Joined/Unjoined split them by whether a witnessed
// outcome was found; Buckets is the overall tally; PerTier is the per-chosen-tier
// roll-up; OverTierWaste is the total over-tier successes; WitnessSources names
// the evidence behind each bucket; Recommendations are the advisory proposals.
type CalibrationReport struct {
	Decisions       int                `json:"decisions"`
	Joined          int                `json:"joined"`
	Unjoined        int                `json:"unjoined"`
	Buckets         map[Bucket]int     `json:"buckets"`
	PerTier         map[Tier]TierStats `json:"per_tier"`
	OverTierWaste   int                `json:"over_tier_waste"`
	WitnessSources  map[Bucket]string  `json:"witness_sources"`
	Recommendations []Recommendation   `json:"recommendations"`
}

// Calibrate folds tier decisions against witnessed outcomes into a
// CalibrationReport. It joins each decision to an outcome by issue number; a
// decision with NO witnessed outcome is counted as Unjoined and never bucketed
// (there is nothing to calibrate against). When more than one outcome shares an
// issue number the LAST one wins, matching a durable ledger where a later witness
// supersedes an earlier one.
//
// It is pure: no wall clock, no I/O, inputs untouched. Buckets, PerTier, and
// WitnessSources are always non-nil so callers can index without a nil check.
func Calibrate(decisions []TierDecision, outcomes []WitnessedOutcome) CalibrationReport {
	byIssue := make(map[int]WitnessedOutcome, len(outcomes))
	for _, o := range outcomes {
		byIssue[o.Issue] = o
	}

	rep := CalibrationReport{
		Decisions:      len(decisions),
		Buckets:        map[Bucket]int{},
		PerTier:        map[Tier]TierStats{},
		WitnessSources: map[Bucket]string{},
	}
	for b, src := range witnessSource {
		rep.WitnessSources[b] = src
	}

	for _, d := range decisions {
		o, ok := byIssue[d.Issue]
		if !ok {
			rep.Unjoined++
			continue
		}
		rep.Joined++
		bucket := classify(o)
		rep.Buckets[bucket]++

		st := rep.PerTier[d.Chosen]
		if st.Buckets == nil {
			st.Buckets = map[Bucket]int{}
		}
		st.N++
		st.Buckets[bucket]++
		if bucket == BucketSuccess && d.Chosen.moreDemandingThan(d.Optimal) {
			st.OverTierWaste++
			rep.OverTierWaste++
		}
		rep.PerTier[d.Chosen] = st
	}

	rep.Recommendations = recommend(rep.PerTier)
	return rep
}

// recommend turns each active chosen tier's stats into exactly one advisory
// Recommendation, in the fixed tier order. The rule is asymmetric and
// risk-first: any rework or escalation at a tier proposes RAISE-FLOOR (never a
// down-tier); otherwise enough over-tier successes propose EXPAND-CHEAPER;
// otherwise HOLD. Correct refusals are ignored here — they are not demerits.
func recommend(perTier map[Tier]TierStats) []Recommendation {
	var recs []Recommendation
	for _, tier := range tierOrder {
		st, ok := perTier[tier]
		if !ok || st.N == 0 {
			continue
		}
		rework := st.Buckets[BucketReverted] + st.Buckets[BucketEscalation]
		switch {
		case rework > 0:
			recs = append(recs, Recommendation{
				Tier:      tier,
				Action:    ActionRaiseFloor,
				Rationale: "chosen tier saw rework or escalation; hold or raise the floor for this class rather than down-tiering",
				Evidence: fmt.Sprintf("reverts=%d escalations=%d over_n=%d",
					st.Buckets[BucketReverted], st.Buckets[BucketEscalation], st.N),
				AutoApply: false,
			})
		case st.OverTierWaste >= minOverTierEvidence:
			recs = append(recs, Recommendation{
				Tier:      tier,
				Action:    ActionExpandCheaper,
				Rationale: "over-tier successes with no rework or escalation; the cheaper optimal tier is a safe expansion candidate for this class",
				Evidence: fmt.Sprintf("over_tier_waste=%d success=%d over_n=%d",
					st.OverTierWaste, st.Buckets[BucketSuccess], st.N),
				AutoApply: false,
			})
		default:
			recs = append(recs, Recommendation{
				Tier:      tier,
				Action:    ActionHold,
				Rationale: "not enough signal to move the threshold either way",
				Evidence:  fmt.Sprintf("success=%d over_tier_waste=%d over_n=%d", st.Buckets[BucketSuccess], st.OverTierWaste, st.N),
				AutoApply: false,
			})
		}
	}
	return recs
}

// Render formats a CalibrationReport as a compact, deterministic operator readout:
// the join counts, the overall bucket tally (fixed order), per-tier over-tier
// waste, and the advisory recommendations. It is the shadow-mode surface — it
// proposes, it never mutates.
func (r CalibrationReport) Render() string {
	if r.Decisions == 0 {
		return "tier calibration (decisions=0): no rows"
	}
	out := fmt.Sprintf("tier calibration (decisions=%d): joined=%d unjoined=%d over_tier_waste=%d\n",
		r.Decisions, r.Joined, r.Unjoined, r.OverTierWaste)
	out += "  buckets:"
	for _, b := range bucketOrder {
		out += fmt.Sprintf(" %s=%d", b, r.Buckets[b])
	}
	out += "\n"
	for _, tier := range tierOrder {
		st, ok := r.PerTier[tier]
		if !ok {
			continue
		}
		out += fmt.Sprintf("  %s: n=%d success=%d stall=%d escalation=%d revert=%d refuse=%d over_tier_waste=%d\n",
			tier, st.N,
			st.Buckets[BucketSuccess], st.Buckets[BucketStall], st.Buckets[BucketEscalation],
			st.Buckets[BucketReverted], st.Buckets[BucketCorrectRefuse], st.OverTierWaste)
	}
	for _, rec := range r.Recommendations {
		out += fmt.Sprintf("  rec %s -> %s [advisory, auto_apply=%v] (%s)\n",
			rec.Tier, rec.Action, rec.AutoApply, rec.Evidence)
	}
	return out
}

// SortedRecommendations returns the recommendations in tier order (they already
// are, but this makes the guarantee explicit for a caller that reorders upstream).
func SortedRecommendations(recs []Recommendation) []Recommendation {
	out := make([]Recommendation, len(recs))
	copy(out, recs)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Tier.demand() < out[j].Tier.demand() })
	return out
}
