package callavoid

// session.go — the in-lane CONSOLIDATION half of the parent #815: a single
// AccountFromObservations entrypoint that composes the three pieces the children shipped
// — the Counters fold (#818), the witnessed productive-deny coverage fan-out (#820), and
// the observed per-class ProveMemo admit-set (#819) — into ONE honest session headline.
//
// Why this leaf exists. Each child shipped its own producer in isolation: TallyFromCounters
// (fold.go) maps the kernel tallies, WitnessesFromCoverage (coverage.go) turns fired deny
// rules into enumerated witnesses, and AdmittedFromObservations (classobserve.go) projects
// the measured-and-proving cache classes. But the PARENT #815 asks that the headline be
// honest END-TO-END — a caller should not have to hand-stitch
// `Account(TallyFromCountersWitnessed(c, WitnessesFromCoverage(fired)))` and separately
// `AdmittedFromObservations(obs)`, getting the netting order wrong (double-counting a deny
// as both hard and productive) or accidentally crediting an un-witnessed / un-measured path.
// AccountFromObservations is the one composed entrypoint that does the wiring correctly,
// once, so the no-tool-call (pure-avoid) economics read the same whether a caller folds by
// hand or through this seam. It is the headline a tier-4 guard/gateway caller renders.
//
// Honesty invariants the composition preserves (proven by session_test.go):
//   - NO DOUBLE-COUNT: a witnessed productive deny is MOVED out of HardDeny, never added on
//     top (TallyFromCountersWitnessed's netting), so RawTurns counts each deny exactly once.
//   - NO UNWITNESSED CREDIT: a deny with no enumerated coverage stays a HardDeny — symmetric,
//     credits zero — so only the witnessed fan-out reaches the graded EffectiveTurns.
//   - NO UNMEASURED CREDIT: the admitted-class set is the projection of measured-AND-proving
//     classes only (#819's calibrate-don't-assume); an abstained or refuted class is absent.
//   - SPECULATIVE STAYS EXCLUDED: any bare Redirects the caller passes remain on the
//     speculative axis and never move the grade (#816), exactly as for a hand-built Tally.
//
// Tier-1 discipline holds: this composes only same-package pure functions and imports
// nothing internal. The tier-4 caller still owns reading the live kernel.Counters into the
// mirror Counters, building the fired-deny coverage from live verdicts, and counting the
// per-class observation — it then hands those three already-shaped inputs here and gets back
// one self-describing SessionReport.

// SessionInput is the already-shaped, tier-1 input a tier-4 caller hands the composed
// headline: the session's kernel Counters (the #818 mirror), the productive denies it
// witnessed from structural coverage (#820), the per-class observations the vDSO recorded
// (#819), and any purely-speculative redirect fan-outs whose variants were NOT enumerated
// (#816 — reported, never graded). Every field is optional; an all-zero SessionInput yields
// an empty, break-even report (the honest "nothing happened" headline), not a credit.
type SessionInput struct {
	// Counters is the session's kernel call-path tallies, mirrored into the tier-1 Counters
	// shape by the caller (EngineCalls/VDSOHits/Transforms/Denies). It is folded by
	// TallyFromCounters; the WitnessedDenies below are netted OUT of its Denies so a deny is
	// never counted as both hard and productive.
	Counters Counters `json:"counters"`
	// WitnessedDenies are the productive denies whose pruned fan-out is ENUMERATED from a deny
	// rule's structural coverage (built by the caller via WitnessFromCoverage / FiredDeny). Each
	// is moved out of Counters.Denies and credited from its OWN variants — realized credit, not
	// an asserted count. An empty-variant witness credits nothing and is surfaced as
	// WitnessedEmpty; it can never inflate the headline.
	WitnessedDenies []WitnessedRedirect `json:"witnessed_denies,omitempty"`
	// ClassObservations are the per-tool-class RAW statistics the vDSO recorded over the window.
	// They drive ONLY the admitted-class advisory set (which classes the tier-2 cache should
	// warm); they do NOT touch the amplification grade — caching economics decide WHAT to cache,
	// the Counters decide what avoidance ALREADY happened. Folded by FoldClassObservations, so an
	// unmeasured class abstains and a write-churned class is declined.
	ClassObservations []ClassObservation `json:"class_observations,omitempty"`
	// SpeculativeRedirects are productive-deny fan-outs the caller could NOT enumerate (a count,
	// not a variant set). They are carried onto the speculative axis verbatim — bounded, reported,
	// and EXCLUDED from the grade (#816) — exactly as a hand-built Tally.Redirects would be.
	SpeculativeRedirects []int `json:"speculative_redirects,omitempty"`
	// ValidateCost / CaptureCost price the cache overhead and stale-miss bet, mirroring Tally.
	// Default 0 (the idealized-vDSO regime; Account floors validation at ValidateFloor regardless).
	ValidateCost float64 `json:"validate_cost"`
	CaptureCost  float64 `json:"capture_cost"`
	// MaxRedirectFanout caps each redirect/witness fan-out; 0 uses DefaultMaxRedirectFanout.
	MaxRedirectFanout int `json:"max_redirect_fanout"`
}

// SessionReport is the composed, self-describing session headline: the realized amplification
// TurnReport (from the folded Counters + witnessed fan-out) PLUS the observed admitted-class
// advisory (from the class observations). The two are reported together but kept distinct —
// the amplification is what avoidance ALREADY bought this window, the admitted classes are
// what the cache should warm NEXT — so a caller renders one honest line without conflating
// "what we saved" with "what to cache".
type SessionReport struct {
	Schema string `json:"schema"`
	// Turns is the realized amplification headline for the session (#818 fold + #820 witnessed
	// credit). Its grade is built from realized, Counter-backed dispositions only; the
	// speculative redirects ride its excluded speculative axis.
	Turns TurnReport `json:"turns"`
	// AdmittedClasses is the advisory tier-2 allow-set: the class names whose MEASURED economics
	// proved (#819). An abstained (unmeasured) or refuted (measured net-loss) class is absent.
	// It is a recommendation surface, never an input to the amplification grade.
	AdmittedClasses []string `json:"admitted_classes,omitempty"`
	// ObservedClasses is the full per-class verdict set the AdmittedClasses were projected from,
	// so a caller can show WHY a class was declined (abstained vs measured-refuted), not just the
	// survivors. Empty when no observations were supplied.
	ObservedClasses []ObservedClassGate `json:"observed_classes,omitempty"`
}

// AccountFromObservations is the single composed entrypoint #815 consolidates to. It folds a
// whole session's already-shaped inputs into ONE honest headline, doing the three children's
// wiring in the one order that stays honest:
//
//  1. Fold the Counters and NET the witnessed productive denies out of HardDeny
//     (TallyFromCountersWitnessed) — so a deny is counted once, never as both hard and
//     productive (no double-count).
//  2. Carry any non-enumerated SpeculativeRedirects onto the excluded speculative axis — they
//     are reported and bounded but never move the grade (no unwitnessed credit in the grade).
//  3. Account the composed Tally — the witnessed fan-out folds into the realized amplification,
//     the speculative redirects do not.
//  4. Project the admitted tier-2 classes from the per-class observations — measured-and-proving
//     only (no unmeasured credit), as an advisory set distinct from the grade.
//
// Pure and deterministic: same SessionInput, same SessionReport. The no-tool-call (pure-avoid)
// case — a window of only memo hits / repairs / witnessed denies with zero Execute — accounts
// correctly: the executed work is just the near-free validations, the effective turns tower over
// it, and the amplification is the realized rebate, not a trust claim.
func AccountFromObservations(in SessionInput) SessionReport {
	tally := TallyFromCountersWitnessed(in.Counters, in.WitnessedDenies)
	tally.Redirects = in.SpeculativeRedirects
	tally.ValidateCost = in.ValidateCost
	tally.CaptureCost = in.CaptureCost
	tally.MaxRedirectFanout = in.MaxRedirectFanout

	observed := FoldClassObservations(in.ClassObservations)
	var admitted []string
	for _, g := range observed {
		if g.Measured && g.Decision.Admit {
			admitted = append(admitted, g.Decision.Class)
		}
	}

	return SessionReport{
		Schema:          "fak.callavoid.session.v1",
		Turns:           Account(tally),
		AdmittedClasses: admitted,
		ObservedClasses: observed,
	}
}
