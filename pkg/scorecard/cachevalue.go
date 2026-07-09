package scorecard

import (
	"fmt"
	"math"
	"sort"
)

// This file defines the four cache-value sub-aspect scores of epic #2783's workstream D
// ("Score + RSI") and composes them into the D1 headline. Each sub-aspect is a pure 0..1
// score with a named formula; the four fold into one control-pane payload via Fold, so an
// RSI loop optimizes the honest *net* cache-value picture instead of reward-hacking the
// gross share (the ~7.7x over-valuation #2783 exists to correct).
//
// The facts each sub-aspect scores are SUPPLIED by the caller -- the cachevaluereport shell,
// which lands the fact-gathering via its own exclusive lease (issue #2815 NOTE). Keeping the
// formulas here, pure over inputs and importing nothing but fmt, makes them deterministic,
// unit-testable with fixtures, and reusable by that shell without this scoring core reaching
// up into cachevaluereport (which would red architest with an ARCH_LAYER_VIOLATION).
//
// Each sub-aspect is INDIVIDUALLY RETIRABLE: its 0..1 fraction is a standalone exported
// function, its KPI a standalone builder, and its defect is retired by adding the real thing
// (capture the missing witness / label the $ figure / wire the arm / converge net->gross) --
// never by weakening the formula. ComposeD1 folds all four into the D1 headline.

// AThemeWitnesses are the observe-at-event ground-truth captures epic #2783 workstream A
// ("Observe-at-event") adds per compaction fire. observation_completeness scores the fraction
// of these present per fire, so this named set is the completeness denominator -- a fixed
// denominator (never zero) keeps the fraction well-defined and the score ungameable by
// simply reporting fewer witnesses.
var AThemeWitnesses = []string{
	"counterfactual_read",   // the provider cache_read the shed span would have gotten
	"suffix_burst_creation", // the one-time cold suffix re-write the fire triggered downstream
	"breakpoint_layout",     // where the cache breakpoints sat at fire time
	"bail_opportunity_cost", // the value of the bail the fire declined
	"ttl_tier",              // the 1h/5m write-premium tier in force at the event
}

// CThemeArms are the ablation arms epic #2783 workstream C ("Ablate") wires on real traffic.
// ablation_coverage scores the fraction of these wired, over this fixed named set so the
// denominator is never zero and "coverage" cannot be faked by declaring fewer arms.
var CThemeArms = []string{
	"compaction_on_off",     // controlled ON/OFF on real traffic
	"valuation_basis_sweep", // gross vs cache-read-marginal basis sweep
	"hardlimit_vs_shaved",   // avoided-a-hard-limit separated from shaved-cached-tokens
	"burst_paysback",        // validate CacheBurstPaysBack (suffix-burst debit)
}

// DivergenceDefectThreshold is the gross-vs-net share gap (in absolute 0..1 share points)
// above which gross_net_divergence emits a defect. #2783 found the guard path booking a 26%
// gross share against a ~3.4% net -- a ~0.226 gap; a healthy report keeps gross ~= net, so a
// gap over 5 share points is the reward-hack alarm.
const DivergenceDefectThreshold = 0.05

// CacheValueFacts carries the raw, caller-supplied facts the four sub-aspects score. The
// caller (the leased cachevaluereport shell) fills it from the real Track-2 ledger; this
// scoring core never reads disk or git itself.
type CacheValueFacts struct {
	// A-theme: one entry per compaction fire giving how many AThemeWitnesses that fire
	// actually captured (0..len(AThemeWitnesses)). observation_completeness is the mean
	// per-fire fraction present.
	FireWitnessCounts []int

	// B3: how many fak $ figures the report emits, and how many carry an explicit
	// valuation_basis label. valuation_basis_honesty is the labelled fraction.
	DollarFigures          int
	DollarFiguresWithBasis int

	// C-theme: how many of the CThemeArms are wired live. ablation_coverage is wired over
	// len(CThemeArms).
	AblationArmsWired int

	// D: the gross and net fak-authored cache-value shares (0..1 fractions). Their
	// normalized absolute divergence is the reward-hack guardrail.
	FakShareGross float64
	FakShareNet   float64
}

// clamp01 keeps a fraction in [0,1]. A caller passing counts that imply a ratio above 1
// (present > expected) or a negative is clamped rather than propagating a nonsense ratio
// into the fold.
func clamp01(x float64) float64 {
	switch {
	case x < 0:
		return 0
	case x > 1:
		return 1
	default:
		return x
	}
}

// ObservationCompleteness is the observation_completeness sub-aspect (0..1): the mean, over
// the given fires, of the fraction of AThemeWitnesses that fire captured. No fires -> 1.0
// (no fire is observed incompletely when there are no fires).
func ObservationCompleteness(fireWitnessCounts []int) float64 {
	if len(fireWitnessCounts) == 0 || len(AThemeWitnesses) == 0 {
		return 1
	}
	denom := float64(len(AThemeWitnesses))
	var sum float64
	for _, present := range fireWitnessCounts {
		sum += clamp01(float64(present) / denom)
	}
	return sum / float64(len(fireWitnessCounts))
}

// ValuationBasisHonesty is the valuation_basis_honesty sub-aspect (0..1): the fraction of
// reported fak $ figures that carry an explicit valuation_basis label (B3). No figures ->
// 1.0 (nothing was claimed without a basis).
func ValuationBasisHonesty(figures, withBasis int) float64 {
	if figures <= 0 {
		return 1
	}
	return clamp01(float64(withBasis) / float64(figures))
}

// AblationCoverage is the ablation_coverage sub-aspect (0..1): the fraction of CThemeArms
// wired live. The denominator is the fixed named arm set, so it is never zero.
func AblationCoverage(armsWired int) float64 {
	if len(CThemeArms) == 0 {
		return 1
	}
	return clamp01(float64(armsWired) / float64(len(CThemeArms)))
}

// GrossNetDivergence is the gross_net_divergence sub-aspect (0..1): the normalized absolute
// gap between the gross and net fak-authored shares. Shares are 0..1 fractions, so the gap
// is already normalized; it is clamped defensively. 0 == gross and net agree (honest); 1 ==
// maximally diverged (the gross figure is inflated relative to the net headline).
func GrossNetDivergence(gross, net float64) float64 {
	d := gross - net
	if d < 0 {
		d = -d
	}
	return clamp01(d)
}

// ObservationCompletenessKPI builds the observation_completeness KPI. A defect is emitted
// (and retired by capturing the missing per-fire witnesses) whenever mean coverage < 1.0.
func ObservationCompletenessKPI(fireWitnessCounts []int) KPI {
	c := ObservationCompleteness(fireWitnessCounts)
	k := KPI{
		Key:    "observation_completeness",
		Group:  "A_observe",
		Score:  100 * c,
		Detail: fmt.Sprintf("mean A-theme witness coverage %.3f over %d fire(s) (of %d witnesses/fire)", c, len(fireWitnessCounts), len(AThemeWitnesses)),
	}
	if c < 1 {
		k.Defects = []string{fmt.Sprintf("observation_completeness: mean per-fire A-theme coverage %.3f < 1.0 across %d fire(s) -- capture the missing witnesses", c, len(fireWitnessCounts))}
	}
	return k
}

// ValuationBasisHonestyKPI builds the valuation_basis_honesty KPI. A defect is emitted (and
// retired by labelling every $ figure with its valuation_basis) whenever a figure lacks one.
func ValuationBasisHonestyKPI(figures, withBasis int) KPI {
	h := ValuationBasisHonesty(figures, withBasis)
	k := KPI{
		Key:    "valuation_basis_honesty",
		Group:  "B_basis",
		Score:  100 * h,
		Detail: fmt.Sprintf("%d of %d fak $ figures carry a valuation_basis (B3)", withBasis, figures),
	}
	if figures > 0 && withBasis < figures {
		k.Defects = []string{fmt.Sprintf("valuation_basis_honesty: %d of %d fak $ figures carry no valuation_basis (B3) -- label the basis", figures-withBasis, figures)}
	}
	return k
}

// AblationCoverageKPI builds the ablation_coverage KPI. A defect is emitted (and retired by
// wiring the arm on real traffic) for each CThemeArm not yet wired.
func AblationCoverageKPI(armsWired int) KPI {
	cov := AblationCoverage(armsWired)
	total := len(CThemeArms)
	k := KPI{
		Key:    "ablation_coverage",
		Group:  "C_ablate",
		Score:  100 * cov,
		Detail: fmt.Sprintf("%d of %d C-theme ablation arms wired", armsWired, total),
	}
	if armsWired < total {
		k.Defects = []string{fmt.Sprintf("ablation_coverage: %d of %d C-theme arms unwired -- wire the ablation arm on real traffic", total-armsWired, total)}
	}
	return k
}

// GrossNetDivergenceKPI builds the gross_net_divergence KPI. The KPI score is quality-
// oriented (higher == better, so it composes into a quality headline): 100*(1-divergence).
// A defect is emitted (and retired by converging net toward gross HONESTLY -- correcting the
// valuation, not inflating net) whenever the divergence exceeds DivergenceDefectThreshold.
func GrossNetDivergenceKPI(gross, net float64) KPI {
	d := GrossNetDivergence(gross, net)
	k := KPI{
		Key:    "gross_net_divergence",
		Group:  "D_divergence",
		Score:  100 * (1 - d),
		Detail: fmt.Sprintf("|gross %.4f - net %.4f| = %.4f divergence (threshold %.2f)", gross, net, d, DivergenceDefectThreshold),
	}
	if d > DivergenceDefectThreshold {
		k.Defects = []string{fmt.Sprintf("gross_net_divergence: gross share diverges from net by %.4f > %.2f -- reward-hack risk, converge net->gross honestly", d, DivergenceDefectThreshold)}
	}
	return k
}

// CacheValueSubAspectKeys is the canonical, ordered set of the four D1 sub-aspect KPI keys
// #2815 names -- the same exported-canonical-list treatment AThemeWitnesses / CThemeArms get,
// so the four sub-aspects are themselves an enumerable, addressable set (not four string
// literals scattered across the KPI builders). It is the "individually retirable" contract as
// data: a consumer folds D1 once, then iterates these keys to retire the heaviest sub-aspect
// defect worst-first, and D1SubAspectKPIs yields them in exactly this order.
var CacheValueSubAspectKeys = []string{
	"observation_completeness", // A-theme: fraction of witnesses present per fire
	"valuation_basis_honesty",  // B3: fraction of fak $ figures carrying a basis
	"ablation_coverage",        // C-theme: fraction of arms wired
	"gross_net_divergence",     // D: |fak_share_gross - fak_share_net| normalized
}

// D1SubAspectKPIs builds the four cache-value sub-aspect KPIs from the caller-supplied facts,
// in CacheValueSubAspectKeys order. Each KPI is INDIVIDUALLY RETIRABLE -- a consumer can score,
// inspect, or retire any one sub-aspect standalone without folding the whole card -- and
// ComposeD1 folds exactly this slice, so the enumeration and the D1 headline can never drift.
func D1SubAspectKPIs(f CacheValueFacts) []KPI {
	return []KPI{
		ObservationCompletenessKPI(f.FireWitnessCounts),
		ValuationBasisHonestyKPI(f.DollarFigures, f.DollarFiguresWithBasis),
		AblationCoverageKPI(f.AblationArmsWired),
		GrossNetDivergenceKPI(f.FakShareGross, f.FakShareNet),
	}
}

// ComposeD1 folds the four cache-value sub-aspects into the D1 headline payload. The
// composite corpus.value is the D1 headline an RSI loop trends; cachevalue_debt is the count
// of sub-aspect defects (each individually retirable). Each raw sub-aspect 0..1 score is also
// stamped into corpus under its exact name, so a consumer reads all four AND the D1 composite
// from one payload. The strict grade curve (GradeStrict) is used because this is a
// provenance-honesty card, like the conflation card it sits beside.
func ComposeD1(f CacheValueFacts) Payload {
	return Fold(CacheValueScoreSchema, D1SubAspectKPIs(f), "cachevalue_debt", nil, Messages{
		Finding:         "cache-value sub-aspects carry debt: the fak-authored share is not yet honestly observed / based / ablated / net-aligned",
		FindingClean:    "cache-value sub-aspects clean: observation, valuation basis, ablation coverage and gross/net divergence all pass",
		NextAction:      "retire the heaviest sub-aspect defect worst-first by adding the real thing (capture the witness / label the basis / wire the arm / converge net->gross)",
		NextActionClean: "hold the line; keep fak_share_net the headline and gross a labelled upper bound",
		Grade:           GradeStrict,
		ExtraCorpus: map[string]any{
			"observation_completeness": Round3(ObservationCompleteness(f.FireWitnessCounts)),
			"valuation_basis_honesty":  Round3(ValuationBasisHonesty(f.DollarFigures, f.DollarFiguresWithBasis)),
			"ablation_coverage":        Round3(AblationCoverage(f.AblationArmsWired)),
			"gross_net_divergence":     Round3(GrossNetDivergence(f.FakShareGross, f.FakShareNet)),
		},
	})
}

// --- D2: the net-regression RSI guardrail (issue #2820, epic #2783 workstream D) ---
//
// D1 (ComposeD1) SCORES the current cache-value picture. D2 GATES a change to it. An RSI
// loop optimizing fak_share must not be able to reward-hack by inflating the GROSS share
// (fire compaction to bust cache, book the shed at an unlabeled 1.0x-on-warm dollar) while
// the true NET stays flat or negative -- the exact ~7.7x over-valuation #2783 exists to
// correct. D2 pins a baseline (the last accepted honest state) and reds a merge/loop step
// that:
//
//   - regresses the net share below the pinned floor (net_nonregression),
//   - raises the gross share without a matching net gain (gross_up_without_net) -- the
//     literal "gross-up with flat/negative net" fence #2783's acceptance names,
//   - drops valuation-basis honesty (valuation_basis_nonregression) -- an unlabeled fak $
//     figure reappears, the exact 1.0x-on-warm slip (#2796), and
//   - lets the gross diverge from the net past the reward-hack alarm (divergence_ceiling).
//
// The gate BLOCKS iff any fence reds (cachevalue_gate_debt > 0 / ok == false). Like D1 it is
// pure over caller-supplied facts and imports nothing but fmt, so it stays deterministic,
// unit-testable with fixtures, and reusable by the leased cachevaluereport shell without this
// scoring core reaching up into cachevaluereport (which would red architest).

const (
	// CacheValueScoreSchema tags the D1 sub-aspect SCORE card (ComposeD1).
	CacheValueScoreSchema = "fak-cachevalue-scorecard/1"
	// CacheValueGateSchema tags the D2 GATE card (ComposeD2), distinct from D1's so a
	// roster/consumer reads the SCORE card and the GATE card apart.
	CacheValueGateSchema = "fak-cachevalue-gate/1"
)

// gateEps is the tolerance below which a share/honesty move is treated as flat, so float
// formatting noise (0.034 stored vs re-derived) can never spuriously red or clear a fence.
const gateEps = 1e-9

// CacheValueBaseline is the pinned "last accepted honest" cache-value floor the D2 gate
// ratchets a candidate against. It carries the three quantities the reward-hack moves: the
// gross and net fak-authored shares (0..1) and the valuation-basis honesty fraction (0..1).
type CacheValueBaseline struct {
	FakShareGross         float64
	FakShareNet           float64
	ValuationBasisHonesty float64
}

// BaselineFromFacts pins a D2 baseline from an accepted corpus of facts -- the "pin
// fak_share_net on the current corpus" step of #2783's regression fence. The gross/net
// shares and the basis-honesty fraction become the floors a later candidate may not regress
// below.
func BaselineFromFacts(f CacheValueFacts) CacheValueBaseline {
	return CacheValueBaseline{
		FakShareGross:         f.FakShareGross,
		FakShareNet:           f.FakShareNet,
		ValuationBasisHonesty: ValuationBasisHonesty(f.DollarFigures, f.DollarFiguresWithBasis),
	}
}

// netNonRegressionKPI reds when the candidate net share fell below the pinned floor -- the
// headline (fak_share_net) itself regressed.
func netNonRegressionKPI(b CacheValueBaseline, c CacheValueFacts) KPI {
	k := KPI{
		Key:    "net_nonregression",
		Group:  "D2_gate",
		Score:  100,
		Detail: fmt.Sprintf("candidate net %.4f vs pinned floor %.4f", c.FakShareNet, b.FakShareNet),
	}
	if c.FakShareNet+gateEps < b.FakShareNet {
		k.Score = 0
		k.Defects = []string{fmt.Sprintf("net_nonregression: candidate net share %.4f fell below the pinned floor %.4f -- the fak_share_net headline regressed", c.FakShareNet, b.FakShareNet)}
	}
	return k
}

// grossUpWithoutNetKPI is the reward-hack fence: it reds when gross rose vs the pinned floor
// but net did NOT (flat or negative) -- a gain that is all gross and no net, which is exactly
// what firing compaction to bust cache buys. This is the "gross-up with flat/negative net"
// #2783's acceptance names; the loop is gated on net, not gross.
func grossUpWithoutNetKPI(b CacheValueBaseline, c CacheValueFacts) KPI {
	k := KPI{
		Key:    "gross_up_without_net",
		Group:  "D2_gate",
		Score:  100,
		Detail: fmt.Sprintf("candidate gross %.4f (floor %.4f), net %.4f (floor %.4f)", c.FakShareGross, b.FakShareGross, c.FakShareNet, b.FakShareNet),
	}
	grossUp := c.FakShareGross > b.FakShareGross+gateEps
	netGained := c.FakShareNet > b.FakShareNet+gateEps
	if grossUp && !netGained {
		k.Score = 0
		k.Defects = []string{fmt.Sprintf("gross_up_without_net: gross rose %.4f->%.4f with net flat/negative (%.4f <= %.4f) -- the reward-hack #2783 fences; gate the loop on net, not gross", b.FakShareGross, c.FakShareGross, c.FakShareNet, b.FakShareNet)}
	}
	return k
}

// valuationBasisNonRegressionKPI reds when basis honesty fell below the pinned floor -- an
// unlabeled fak $ figure reappeared, the literal "unlabeled 1.0x-on-warm valuation" (#2796).
func valuationBasisNonRegressionKPI(floor, candidate float64) KPI {
	k := KPI{
		Key:    "valuation_basis_nonregression",
		Group:  "D2_gate",
		Score:  100,
		Detail: fmt.Sprintf("candidate valuation-basis honesty %.3f vs pinned floor %.3f", candidate, floor),
	}
	if candidate+gateEps < floor {
		k.Score = 0
		k.Defects = []string{fmt.Sprintf("valuation_basis_nonregression: basis honesty %.3f fell below the pinned floor %.3f -- an unlabeled fak $ figure (1.0x-on-warm) reappeared; label its valuation_basis", candidate, floor)}
	}
	return k
}

// divergenceCeilingKPI reds when the candidate's gross diverges from its net past the
// reward-hack alarm (DivergenceDefectThreshold) -- the 1.0x-on-warm gross-up signature, a
// hard ceiling independent of the baseline so a fresh inflation is caught even at a zero pin.
func divergenceCeilingKPI(divergence float64) KPI {
	k := KPI{
		Key:    "divergence_ceiling",
		Group:  "D2_gate",
		Score:  100,
		Detail: fmt.Sprintf("candidate gross/net divergence %.4f (ceiling %.2f)", divergence, DivergenceDefectThreshold),
	}
	if divergence > DivergenceDefectThreshold {
		k.Score = 0
		k.Defects = []string{fmt.Sprintf("divergence_ceiling: gross diverges from net by %.4f > %.2f -- the 1.0x-on-warm gross-up signature; converge net->gross honestly", divergence, DivergenceDefectThreshold)}
	}
	return k
}

// ComposeD2 folds the four D2 fences into the gate's control-pane payload. cachevalue_gate_debt
// is the count of red fences; ok == false (verdict ACTION) is the BLOCK signal. Each fence is
// individually retirable by making the CANDIDATE honest (raise net, converge gross->net, label
// the basis) -- never by weakening the fence. The strict grade curve matches D1.
func ComposeD2(baseline CacheValueBaseline, candidate CacheValueFacts) Payload {
	candHonesty := ValuationBasisHonesty(candidate.DollarFigures, candidate.DollarFiguresWithBasis)
	divergence := GrossNetDivergence(candidate.FakShareGross, candidate.FakShareNet)
	kpis := []KPI{
		netNonRegressionKPI(baseline, candidate),
		grossUpWithoutNetKPI(baseline, candidate),
		valuationBasisNonRegressionKPI(baseline.ValuationBasisHonesty, candHonesty),
		divergenceCeilingKPI(divergence),
	}
	return Fold(CacheValueGateSchema, kpis, "cachevalue_gate_debt", nil, Messages{
		Finding:         "cache-value gate BLOCK: a candidate reintroduces a gross-up / unlabeled-on-warm valuation without a matching net gain",
		FindingClean:    "cache-value gate PASS: net held or improved, gross tracks net, and every fak $ figure stays labelled",
		NextAction:      "reject the change: converge the reward-hack back to net (value shed at the cache-read marginal, label the basis) instead of grossing up",
		NextActionClean: "hold the line; fak_share_net stays the headline and gross a labelled upper bound",
		Grade:           GradeStrict,
		ExtraCorpus: map[string]any{
			"baseline_fak_share_gross":  Round3(baseline.FakShareGross),
			"baseline_fak_share_net":    Round3(baseline.FakShareNet),
			"candidate_fak_share_gross": Round3(candidate.FakShareGross),
			"candidate_fak_share_net":   Round3(candidate.FakShareNet),
			"gross_net_divergence":      Round3(divergence),
		},
	})
}

// CacheValueGateBlocks is the D2 merge/loop decision: it BLOCKS (true) iff any D2 fence reds
// against the pinned baseline, returning the joined fence reason. This is the verb a pre-merge
// gate or an RSI loop step calls to refuse a gross-up-without-net change; block == false is a
// clean pass.
func CacheValueGateBlocks(baseline CacheValueBaseline, candidate CacheValueFacts) (bool, string) {
	p := ComposeD2(baseline, candidate)
	if p.OK {
		return false, "clean"
	}
	return true, p.Reason
}

// CacheValueControlPaneMembers registers the two workstream-D cache-value scores as
// control-pane members: D1 (ComposeD1, the sub-aspect SCORE card) and D2 (ComposeD2, the
// net-regression GATE card). A roster/conceptbench consumer folds both from one call, so the
// cache-value surface contributes both its score AND its gate to the control pane, each with
// its own debt key (cachevalue_debt / cachevalue_gate_debt).
func CacheValueControlPaneMembers(baseline CacheValueBaseline, candidate CacheValueFacts) []Payload {
	return []Payload{ComposeD1(candidate), ComposeD2(baseline, candidate)}
}

// --- D3: the cache-value ACCURACY-debt scorer (issue #2814, epic #2783 workstream D) ---
//
// D1 (ComposeD1) SCORES each cache-value sub-aspect as a 0..1 quality fraction; D2 (ComposeD2)
// GATES a change against a pinned baseline. D3 is the third sibling: it AGGREGATES the accuracy
// signals into a single deterministic INTEGER debt plus a worst-first worklist, so the largest
// accuracy debt is triaged first. It answers "how untrue is the reported number, and which
// untruth is heaviest?" -- truth-of-the-number as a retirable debt count (issue #2814).
//
// The debt rises with exactly the three accuracy classes the issue names:
//   - gross_net_divergence:          |gross - net| past the reward-hack alarm, in whole
//                                     threshold-widths (a bigger gap is heavier debt),
//   - unlabeled_valuation_basis:      one unit per fak $ figure emitted with no valuation_basis,
//   - missing_time_of_event_witness:  one unit per A-theme (time-of-event) witness a fire failed
//                                     to capture.
//
// ablation_coverage (a C-theme WIRING signal, not an accuracy-of-the-number signal) is
// deliberately NOT a D3 accuracy class: D3 scores whether the reported number is TRUE, not
// whether every ablation arm is wired. Each unit is individually retirable by adding the real
// thing (converge net->gross honestly / label the basis / capture the witness); a clean corpus
// scores 0 with an empty worklist.
//
// Like D1/D2 this is pure over caller-supplied CacheValueFacts and reaches up into no shell, so it
// stays deterministic and unit-testable with fixtures (importing nothing but fmt/math/sort).
// Registering D3 into the scorecard/conceptbench control pane or gating merges on it is the
// SIBLING #2820's job -- D3 here is the scorer + its worklist only, and CacheValueControlPaneMembers
// stays the D1 score + D2 gate.

// CacheValueAccuracyDebtSchema tags the D3 ACCURACY-debt card (ComposeD3), distinct from the D1
// score schema and D2 gate schema so a roster/consumer reads all three cards apart.
const CacheValueAccuracyDebtSchema = "fak-cachevalue-accuracy/1"

// The three accuracy-debt classes #2814 names. A worklist row is keyed by one of these; the
// canonical order below also breaks ties when two classes carry equal integer debt, so the
// worst-first ordering is fully deterministic.
const (
	AccuracyClassGrossNetDivergence = "gross_net_divergence"
	AccuracyClassUnlabeledBasis     = "unlabeled_valuation_basis"
	AccuracyClassMissingWitness     = "missing_time_of_event_witness"
)

// CacheValueAccuracyClasses is the canonical, ordered accuracy-debt class set -- the same
// exported-canonical-list treatment CacheValueSubAspectKeys / AThemeWitnesses get. It is the
// deterministic tie-break order (used when two classes carry equal debt) AND the enumeration a
// consumer iterates to address each accuracy class standalone.
var CacheValueAccuracyClasses = []string{
	AccuracyClassGrossNetDivergence,
	AccuracyClassUnlabeledBasis,
	AccuracyClassMissingWitness,
}

// AccuracyDebtRow is one accuracy-debt class's contribution to the D3 worklist: the class key,
// its integer debt, and a human detail. The worklist is these rows with Debt > 0, sorted
// worst-first (largest Debt first, CacheValueAccuracyClasses order breaking ties).
type AccuracyDebtRow struct {
	Class  string `json:"class"`
	Debt   int    `json:"debt"`
	Detail string `json:"detail"`
}

// divergenceDebtUnits converts a gross-vs-net divergence into whole integer debt units: 0 while
// the divergence stays within the healthy band (<= DivergenceDefectThreshold, the same alarm D1
// uses), then one unit per whole threshold-width the divergence exceeds the alarm by. So a
// divergence at or below the alarm books 0 (clean), and the ~0.226 gap #2783 found books
// ceil((0.226-0.05)/0.05) = 4 units -- debt that RISES with the gap and retires to 0 when gross
// and net reconverge into the healthy band.
func divergenceDebtUnits(divergence float64) int {
	if divergence <= DivergenceDefectThreshold+gateEps {
		return 0
	}
	return int(math.Ceil((divergence - DivergenceDefectThreshold) / DivergenceDefectThreshold))
}

// unlabeledBasisUnits counts fak $ figures emitted with no valuation_basis label -- one debt unit
// per unlabeled figure, never negative if the caller over-reports labelled figures.
func unlabeledBasisUnits(figures, withBasis int) int {
	if u := figures - withBasis; u > 0 {
		return u
	}
	return 0
}

// missingWitnessUnits counts, over every fire, how many A-theme (time-of-event) witnesses that
// fire failed to capture -- one debt unit per missing capture. A fire that over-reports (present
// > len(AThemeWitnesses)) contributes 0, never negative debt.
func missingWitnessUnits(fireWitnessCounts []int) int {
	perFire := len(AThemeWitnesses)
	total := 0
	for _, present := range fireWitnessCounts {
		if missing := perFire - present; missing > 0 {
			total += missing
		}
	}
	return total
}

// CacheValueAccuracyDebt is the D3 core: it folds the caller-supplied facts into a single
// deterministic INTEGER accuracy debt (the weighted sum of the three classes) and a worst-first
// worklist (rows with debt > 0, largest first, CacheValueAccuracyClasses order breaking ties).
// A fully honest corpus -- gross ~= net, every figure labelled, every witness captured -- scores
// 0 with an empty worklist. This is the issue #2814 deliverable: truth-of-the-number as a
// retirable integer with a deterministic triage order.
func CacheValueAccuracyDebt(f CacheValueFacts) (int, []AccuracyDebtRow) {
	divergence := GrossNetDivergence(f.FakShareGross, f.FakShareNet)
	divUnits := divergenceDebtUnits(divergence)
	basisUnits := unlabeledBasisUnits(f.DollarFigures, f.DollarFiguresWithBasis)
	witnessUnits := missingWitnessUnits(f.FireWitnessCounts)

	byClass := map[string]AccuracyDebtRow{
		AccuracyClassGrossNetDivergence: {
			Class:  AccuracyClassGrossNetDivergence,
			Debt:   divUnits,
			Detail: fmt.Sprintf("|gross %.4f - net %.4f| = %.4f divergence, %d unit(s) past the %.2f alarm", f.FakShareGross, f.FakShareNet, divergence, divUnits, DivergenceDefectThreshold),
		},
		AccuracyClassUnlabeledBasis: {
			Class:  AccuracyClassUnlabeledBasis,
			Debt:   basisUnits,
			Detail: fmt.Sprintf("%d of %d fak $ figure(s) carry no valuation_basis label", basisUnits, f.DollarFigures),
		},
		AccuracyClassMissingWitness: {
			Class:  AccuracyClassMissingWitness,
			Debt:   witnessUnits,
			Detail: fmt.Sprintf("%d A-theme time-of-event witness capture(s) missing across %d fire(s) (of %d/fire)", witnessUnits, len(f.FireWitnessCounts), len(AThemeWitnesses)),
		},
	}

	total := divUnits + basisUnits + witnessUnits

	// worst-first worklist: only classes actually in debt, largest first, CacheValueAccuracyClasses
	// order breaking ties so the ordering is fully deterministic.
	rank := map[string]int{}
	for i, c := range CacheValueAccuracyClasses {
		rank[c] = i
	}
	worklist := make([]AccuracyDebtRow, 0, len(CacheValueAccuracyClasses))
	for _, c := range CacheValueAccuracyClasses {
		if r := byClass[c]; r.Debt > 0 {
			worklist = append(worklist, r)
		}
	}
	sort.SliceStable(worklist, func(i, j int) bool {
		if worklist[i].Debt != worklist[j].Debt {
			return worklist[i].Debt > worklist[j].Debt
		}
		return rank[worklist[i].Class] < rank[worklist[j].Class]
	})
	return total, worklist
}

// accuracyKPI builds the D3 KPI for one accuracy class. Its Score is the matching D1 quality
// fraction (so the D3 headline degrades with the same sub-aspect quality D1 reports); a class in
// debt owns exactly one Defect string so Fold's ok/verdict flips iff any class is in debt.
func accuracyKPI(class string, score float64, worklist []AccuracyDebtRow) KPI {
	k := KPI{Key: class, Group: "D3_accuracy", Score: score}
	for _, r := range worklist {
		if r.Class == class {
			k.Detail = r.Detail
			k.Defects = []string{fmt.Sprintf("%s: %s -- retire it (%d debt unit(s))", class, r.Detail, r.Debt)}
			return k
		}
	}
	k.Detail = fmt.Sprintf("%s: clean (0 accuracy debt)", class)
	return k
}

// ComposeD3 folds the D3 accuracy debt into the cache-value family's control-pane payload shape
// (symmetric with ComposeD1/ComposeD2) so the accuracy debt can be snapshot-pinned and read beside
// the score and gate cards. corpus["cachevalue_accuracy_debt"] is the deterministic WEIGHTED
// INTEGER debt (whole threshold-widths of divergence + unlabeled figures + missing witnesses),
// overriding Fold's raw defect count; corpus["accuracy_worklist"] is the worst-first triage order;
// ok == (debt == 0).
//
// NOTE: producing this payload is NOT registering D3 into the scorecard/conceptbench control pane
// or gating merges on it -- that wiring is the sibling #2820, so CacheValueControlPaneMembers stays
// the D1 score + D2 gate only (issue #2814 in-scope is the scorer + its worklist).
func ComposeD3(f CacheValueFacts) Payload {
	total, worklist := CacheValueAccuracyDebt(f)
	divergence := GrossNetDivergence(f.FakShareGross, f.FakShareNet)
	kpis := []KPI{
		accuracyKPI(AccuracyClassGrossNetDivergence, 100*(1-divergence), worklist),
		accuracyKPI(AccuracyClassUnlabeledBasis, 100*ValuationBasisHonesty(f.DollarFigures, f.DollarFiguresWithBasis), worklist),
		accuracyKPI(AccuracyClassMissingWitness, 100*ObservationCompleteness(f.FireWitnessCounts), worklist),
	}
	return Fold(CacheValueAccuracyDebtSchema, kpis, "cachevalue_accuracy_debt", nil, Messages{
		Finding:         "cache-value ACCURACY debt: the reported number is not yet true -- gross diverges from net, a figure is unlabeled, or a time-of-event witness is missing",
		FindingClean:    "cache-value accuracy clean: gross tracks net, every fak $ figure is labelled, and every time-of-event witness was captured",
		NextAction:      "retire the heaviest accuracy debt worst-first: converge net->gross honestly / label the valuation_basis / capture the missing A-theme witness",
		NextActionClean: "hold the line; the reported cache-value number is honest",
		Grade:           GradeStrict,
		ExtraCorpus: map[string]any{
			// the issue's weighted INTEGER debt, overriding Fold's raw defect-string count
			"cachevalue_accuracy_debt": total,
			"accuracy_worklist":        worklist,
			"gross_net_divergence":     Round3(divergence),
		},
	})
}
