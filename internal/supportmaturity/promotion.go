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
	record := func(d shipgate.Decision) shipgate.Decision {
		if breaker != nil {
			return breaker.Record(d)
		}
		return d
	}
	if !current.Less(M7BeyondSOTA) { // already at the top — nothing higher to witness
		return current, record(shipgate.REVERT)
	}
	target := current + 1
	if kind != WitnessFor(target) { // wrong witness for this rung — refuse before shipgate
		return current, record(shipgate.REVERT)
	}
	dec, _ := shipgate.Evaluate(w)
	dec = record(dec)
	if dec == shipgate.KEEP {
		return target, dec
	}
	return current, dec
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
