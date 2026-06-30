package supportmaturity

import (
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// This file is the C2 keystone of the support-maturity epic (#1243, #1245). #1244
// (C1) lowered today's scattered support scales onto the closed M0–M7 ladder —
// VOCABULARY only: it gave each rung the value a cell CLAIMS. This file binds each
// rung to the non-author witness that PROVES that claim, and defines the two rules a
// cell's rung may move under:
//
//   - PROMOTE: a cell advances ONE rung only when a witness its own author did NOT
//     write confirms the next rung — gated by shipgate.Evaluate (KEEP/REVERT) and,
//     across a run, shipgate's consecutive-non-keep breaker (ESCALATE). A bench number
//     a cell reports about itself never advances it; the keep-bit is non-forgeable.
//   - DROP: a cell falls ONE rung when the witness BOUND to its current rung regresses
//     (a red CI oracle demotes M4Correct → M3Runs). A rung is an envelope with a
//     witness, never a promise — when the witness goes red the envelope shrinks.
//
// The promotion machinery is shipgate's, reused verbatim (internal/shipgate): the
// kernel adjudicating a maturity claim is never the cell asserting it.

// WitnessKind names the single non-author witness that PROVES a rung — the evidence a
// cell must carry to honestly stand there. The kinds track the witnesses the epic
// (#1243) names, one per rung:
//
//	M0 none        → WitnessNone           (UNDEFINED / unparseable — nothing proves it)
//	M1 fenced      → WitnessFence          (an honest refusal: covmatrix.Fenced / preflight REFUSE_*)
//	M2 loads       → WitnessPreflight      (the ggufload preflight READY verdict)
//	M3 runs        → WitnessProofPath      (covmatrix PROOF-PATH-ONLY — the cpu reference runs)
//	M4 correct     → WitnessOracleInCI     (a CI-runnable numeric oracle — covmatrix.OracleInCI)
//	M5 optimized   → WitnessBenchCommitted (a compute.Caps capability + a committed bench)
//	M6 parity      → WitnessSotaParity     (the turnbench sota-local-baseline parity class)
//	M7 beyond-sota → WitnessSotaBeyond     (beats the baseline — a BENCHMARK-AUTHORITY row)
type WitnessKind uint8

const (
	WitnessNone WitnessKind = iota
	WitnessFence
	WitnessPreflight
	WitnessProofPath
	WitnessOracleInCI
	WitnessBenchCommitted
	WitnessSotaParity
	WitnessSotaBeyond
)

// witnessKindName renders each witness kind as a stable token, indexed by WitnessKind.
var witnessKindName = []string{
	WitnessNone:           "none",
	WitnessFence:          "fence",
	WitnessPreflight:      "preflight",
	WitnessProofPath:      "proof-path",
	WitnessOracleInCI:     "oracle-in-ci",
	WitnessBenchCommitted: "bench-committed",
	WitnessSotaParity:     "sota-parity",
	WitnessSotaBeyond:     "sota-beyond",
}

// String renders the witness kind as its stable token.
func (k WitnessKind) String() string {
	if int(k) < len(witnessKindName) {
		return witnessKindName[k]
	}
	return "unknown"
}

// witnessForRung binds each rung to the witness that PROVES it, indexed by Rung. The
// binding is total and injective — each rung has its own distinct witness — so a rung
// is never a bare claim and no two rungs share a proof.
var witnessForRung = []WitnessKind{
	M0None:       WitnessNone,
	M1Fenced:     WitnessFence,
	M2Loads:      WitnessPreflight,
	M3Runs:       WitnessProofPath,
	M4Correct:    WitnessOracleInCI,
	M5Optimized:  WitnessBenchCommitted,
	M6Parity:     WitnessSotaParity,
	M7BeyondSOTA: WitnessSotaBeyond,
}

// WitnessFor returns the witness kind that PROVES rung r. An out-of-range rung binds to
// WitnessNone — the honest "nothing witnesses this" default the From* lowerings rely on.
func WitnessFor(r Rung) WitnessKind {
	if int(r) < len(witnessForRung) {
		return witnessForRung[r]
	}
	return WitnessNone
}

// Promote attempts to advance a cell ONE rung above current, gated by two independent
// rules — neither of which the cell's own author can forge:
//
//  1. BINDING: the evidence kind must be exactly the witness WitnessFor binds to the
//     TARGET rung (current+1). You cannot reach M5 'optimized' with a preflight verdict;
//     only a committed bench witnesses M5. A kind mismatch is REVERTed before shipgate
//     is even consulted.
//
//  2. NON-AUTHOR: shipgate.Evaluate(w) must KEEP — a strict gain (or, for a boolean
//     presence witness, Before=0,After=1) confirmed by the suite-green + truth-clean
//     signals the candidate's author did NOT write. An author-only bench-win — a real
//     Before<After but TruthClean=false — is REVERTed by shipgate, so the cell does NOT
//     promote. This is the keystone of #1245: a number a cell reports about itself can
//     never carry it up the ladder.
//
// breaker, when non-nil, folds the decision into shipgate's consecutive-non-keep gate so
// a run of refused promotions ESCALATEs to a human (the third arm of the reused
// KEEP/REVERT/ESCALATE vocabulary). The rung advances ONLY on KEEP; REVERT and ESCALATE
// both hold it at current. A cell already at M7BeyondSOTA has no higher witness, so any
// attempt there is REVERTed.
func Promote(current Rung, kind WitnessKind, w shipgate.Witness, breaker *shipgate.Gate) (Rung, shipgate.Decision) {
	rec := PromoteWithRecord(current, kind, w, breaker)
	return rec.Next, rec.Decision
}

// PromotionRecord is the auditable form of Promote. Promote keeps the compact API
// for existing callers; PromoteWithRecord returns this richer payload so RSI-like
// controls can journal the same score evidence they used to decide the rung move.
type PromotionRecord struct {
	Current      Rung
	Target       Rung
	Next         Rung
	WitnessKind  WitnessKind
	ExpectedKind WitnessKind
	Decision     shipgate.Decision
	Kept         bool
	Witness      shipgate.Witness
	Score        Scorecard
}

// Scorecard is the structured score payload for support-maturity promotion. It is
// evidence only: the promotion authority remains the witness-kind binding plus
// shipgate's keep-bit.
type Scorecard struct {
	Name       string           `json:"name,omitempty"`
	Value      float64          `json:"value"`
	Grade      string           `json:"grade,omitempty"`
	Components []ScoreComponent `json:"components,omitempty"`
}

// ScoreComponent is one named numeric axis of a Scorecard.
type ScoreComponent struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitempty"`
}

// PromoteWithRecord is Promote with structured score telemetry. It advances the
// rung only under the same rules as Promote, then records the reason shape.
func PromoteWithRecord(current Rung, kind WitnessKind, w shipgate.Witness, breaker *shipgate.Gate) PromotionRecord {
	record := func(d shipgate.Decision) shipgate.Decision {
		if breaker != nil {
			return breaker.Record(d)
		}
		return d
	}
	if !current.Less(M7BeyondSOTA) { // already at the top — nothing higher to witness
		dec := record(shipgate.REVERT)
		return promotionRecord(current, current, current, kind, WitnessNone, dec, false, w, "top")
	}
	target := current + 1
	if kind != WitnessFor(target) { // wrong witness for this rung — refuse before shipgate
		dec := record(shipgate.REVERT)
		return promotionRecord(current, target, current, kind, WitnessFor(target), dec, false, w, "wrong-witness")
	}
	dec, ev := shipgate.Evaluate(w)
	dec = record(dec)
	if dec == shipgate.KEEP {
		return promotionRecord(current, target, target, kind, WitnessFor(target), dec, ev.Kept(), ev, "promoted")
	}
	return promotionRecord(current, target, current, kind, WitnessFor(target), dec, ev.Kept(), ev, promotionGrade(dec, ev))
}

func promotionRecord(current, target, next Rung, kind, expected WitnessKind, decision shipgate.Decision, kept bool, w shipgate.Witness, grade string) PromotionRecord {
	rec := PromotionRecord{
		Current:      current,
		Target:       target,
		Next:         next,
		WitnessKind:  kind,
		ExpectedKind: expected,
		Decision:     decision,
		Kept:         kept,
		Witness:      w,
	}
	rec.Score = promotionScorecard(rec, grade)
	return rec
}

func promotionScorecard(rec PromotionRecord, grade string) Scorecard {
	bindingOK := 0.0
	if rec.ExpectedKind != WitnessNone && rec.WitnessKind == rec.ExpectedKind {
		bindingOK = 1
	}
	advanced := 0.0
	if rec.Next != rec.Current {
		advanced = 1
	}
	return Scorecard{
		Name:  "support_maturity_promotion",
		Value: float64(rec.Next),
		Grade: grade,
		Components: []ScoreComponent{
			{Name: "current_rung", Value: float64(rec.Current), Unit: "rung"},
			{Name: "target_rung", Value: float64(rec.Target), Unit: "rung"},
			{Name: "next_rung", Value: float64(rec.Next), Unit: "rung"},
			{Name: "binding_ok", Value: bindingOK, Unit: "bool"},
			{Name: "metric_before", Value: rec.Witness.Before, Unit: "metric"},
			{Name: "metric_after", Value: rec.Witness.After, Unit: "metric"},
			{Name: "metric_delta", Value: rec.Witness.After - rec.Witness.Before, Unit: "metric"},
			{Name: "suite_green", Value: boolFloat(rec.Witness.SuiteGreen), Unit: "bool"},
			{Name: "truth_clean", Value: boolFloat(rec.Witness.TruthClean), Unit: "bool"},
			{Name: "kept", Value: boolFloat(rec.Kept), Unit: "bool"},
			{Name: "advanced", Value: advanced, Unit: "bool"},
		},
	}
}

func promotionGrade(dec shipgate.Decision, w shipgate.Witness) string {
	if dec == shipgate.ESCALATE {
		return "escalated"
	}
	switch {
	case !w.TruthClean:
		return "truth-dirty"
	case !w.SuiteGreen && w.Class == shipgate.ClassFull:
		return "suite-red"
	default:
		return "reverted"
	}
}

func boolFloat(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

// Drop demotes a cell when the witness BOUND to its current rung regresses — the
// drop-on-regression half of #1245. A rung is an envelope with a witness; when
// WitnessFor(current) goes red the cell can no longer honestly stand there and falls to
// the next-lower rung, whose weaker witness still holds. The epic's golden case: a red CI
// oracle (the M4 witness, covmatrix.OracleInCI) demotes M4Correct → M3Runs — the cell
// still runs on the cpu reference (M3) but can no longer claim CI-witnessed correctness
// (M4). witnessRed is the measured regression of the bound witness; a clean witness holds
// the rung, and M0None is the floor.
func Drop(current Rung, witnessRed bool) Rung {
	if !witnessRed || current == M0None {
		return current
	}
	return current - 1
}
