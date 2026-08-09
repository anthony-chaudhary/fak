package main

import (
	"fmt"
	"io"
	"strings"
)

// printHuman renders the full report as readable text: the provenance banner first (so nobody
// can miss it), the hypothesis, the MEASURED arms (#5851), then the MODELED fixture half -- the
// per-pair table, the aggregate modeled delta, and the sign test -- in that order, so a reader
// hits both the "this is modeled, not measured" label and the measured/modeled split before any
// number.
func printHuman(w io.Writer, r Report) {
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, " %s\n", r.Provenance)
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, "issue: %s\n", r.Issue)
	fmt.Fprintf(w, "hypothesis: %s\n\n", r.Hypothesis)

	printMeasured(w, r)

	fmt.Fprintln(w, "-- MODELED half: hand-authored fixture corpus, scored by the stated cost model ----")
	fmt.Fprintf(w, "%-24s %10s %10s %10s | %10s %10s %10s | %8s\n",
		"pair", "a-mech", "a-judge", "a-comply", "b-mech", "b-judge", "b-comply", "delta")
	for _, p := range r.Pairs {
		fmt.Fprintf(w, "%-24s %10d %10d %10.3f | %10d %10d %10.3f | %8.3f\n",
			p.ID, p.ArmAMechanical, p.ArmAJudgement, p.ArmACompliance,
			p.ArmBMechanical, p.ArmBJudgement, p.ArmBCompliance, p.Delta)
	}

	fmt.Fprintln(w)
	fmt.Fprintf(w, "mean modeled compliance -- arm A (negative-framed):    %.4f\n", r.MeanA)
	fmt.Fprintf(w, "mean modeled compliance -- arm B (affordance-first):   %.4f\n", r.MeanB)
	fmt.Fprintf(w, "mean modeled delta (B - A):                            %+.4f\n", r.MeanDelta)
	fmt.Fprintf(w, "paired sign test: %d/%d non-tied pairs favor arm B (ties=%d), one-sided p = %.6g\n",
		r.SignTest.Favoring_B, r.SignTest.NUsed, r.SignTest.Ties, r.SignTest.PValue)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "This is a MODELED proxy over a hand-picked fixture corpus: the delta above is what the")
	fmt.Fprintln(w, "stated linear cost model predicts from negframe's negation classification, not a measured")
	fmt.Fprintln(w, "agent compliance rate. See README.md 'Upgrading to a live witness' for what would promote")
	fmt.Fprintln(w, "this from modeled to observed.")
}

// printMeasured renders the #5851 half: the two arms the #3568 env lever produced over the real
// injected stream. It states, in the section header and again per column, exactly which numbers
// are MEASURED (the arm label and the negation counts, read off the run's journal) and which stay
// MODELED (the compliance rate the load maps to) -- the DoD's measured/modeled distinction.
func printMeasured(w io.Writer, r Report) {
	if r.Measured == nil {
		fmt.Fprintln(w, "-- MEASURED half (#5851): UNAVAILABLE ------------------------------------------")
		reason := r.MeasuredUnavailable
		if reason == "" {
			reason = "not attempted"
		}
		fmt.Fprintf(w, "  reason: %s\n", reason)
		fmt.Fprintln(w, "  Everything below is MODELED only -- no arm label in this report came from a run.")
		fmt.Fprintln(w)
		return
	}
	m := r.Measured
	fmt.Fprintln(w, "-- MEASURED half (#5851): arms produced by the #3568 env lever, not by fixtures --")
	fmt.Fprintf(w, "  provenance: %s\n", m.Provenance)
	fmt.Fprintf(w, "  binary:     %s (%s)\n", m.Binary, m.BinarySource)
	fmt.Fprintf(w, "  measured path: fak %s\n", strings.Join(m.Argv, " "))
	fmt.Fprintf(w, "  %-10s %-34s %-16s %8s %9s %10s | %18s\n",
		"arm", "selected by", "arm (MEASURED)", "applied", "residual", "fallbacks", "compliance (MODELED)")
	for _, a := range []MeasuredArm{m.Control, m.Treatment} {
		fmt.Fprintf(w, "  %-10s %-34s %-16s %8d %9d %10d | %18.3f\n",
			a.Label, a.Env, a.Arm, a.Applied, a.Residual, a.VerbatimFallback, a.Compliance)
	}
	fmt.Fprintf(w, "  measured residual delta (treatment - control): %+d\n", m.ResidualDelta)
	fmt.Fprintf(w, "  modeled compliance delta from that load:       %+.4f\n", m.ModeledDelta)
	fmt.Fprintf(w, "  %s\n", m.Note)
	fmt.Fprintf(w, "  arm labels read from: %s\n", m.Control.JournalPath)
	fmt.Fprintf(w, "                        %s\n", m.Treatment.JournalPath)
	fmt.Fprintln(w)
}

// selfCheckReport prints a concise PASS/FAIL self-check and returns whether it passed. Two
// independent checks:
//  1. Corpus integrity: every Arm A carries at least one negframe finding (it is genuinely
//     negative-framed) and every Arm B carries zero (the reframe genuinely removed the negation
//     the classifier detects) -- a broken fixture (e.g. a reframe that leaked a "don't") would
//     silently invalidate the whole experiment, so this is checked BEFORE trusting the delta.
//  2. Direction: the modeled mean delta is positive and the sign test favors Arm B on every
//     non-tied pair, matching the thesis direction.
func selfCheckReport(w io.Writer, r Report) bool {
	ok := true
	fmt.Fprintln(w, "negframe-steerability-ab selfcheck")
	fmt.Fprintf(w, "  provenance: %s\n", r.Provenance)

	// Measured-arm check (#5851). SKIP rather than FAIL when no binary was resolvable, so the
	// offline fixture spine stays runnable on a host with no fak build; a measured half that IS
	// present must carry the two distinct journal-sourced arm labels or the run is not an A/B.
	switch {
	case r.Measured == nil:
		reason := r.MeasuredUnavailable
		if reason == "" {
			reason = "not attempted"
		}
		fmt.Fprintf(w, "  SKIP measured arms: %s -- report is MODELED only\n", reason)
	case r.Measured.Control.Arm == measuredArmOff && r.Measured.Treatment.Arm == measuredArmOn:
		fmt.Fprintf(w, "  PASS measured arms: control=%s treatment=%s, both read from the run's journal\n",
			r.Measured.Control.Arm, r.Measured.Treatment.Arm)
	default:
		fmt.Fprintf(w, "  FAIL measured arms: control=%q treatment=%q, want %q/%q from the journal\n",
			r.Measured.Control.Arm, r.Measured.Treatment.Arm, measuredArmOff, measuredArmOn)
		ok = false
	}

	badCorpus := 0
	for _, p := range r.Pairs {
		armATotal := p.ArmAMechanical + p.ArmAJudgement
		armBTotal := p.ArmBMechanical + p.ArmBJudgement
		if armATotal < 1 {
			fmt.Fprintf(w, "  FAIL corpus: pair %q -- Arm A carries no negframe finding (%q)\n", p.ID, p.ArmAText)
			badCorpus++
		}
		if armBTotal != 0 {
			fmt.Fprintf(w, "  FAIL corpus: pair %q -- Arm B still carries %d negframe finding(s) (%q)\n",
				p.ID, armBTotal, p.ArmBText)
			badCorpus++
		}
	}
	if badCorpus == 0 {
		fmt.Fprintf(w, "  PASS corpus: all %d pairs well-formed (Arm A negative, Arm B clean)\n", len(r.Pairs))
	} else {
		ok = false
	}

	if r.MeanDelta > 0 {
		fmt.Fprintf(w, "  PASS direction: mean modeled delta is positive (%+.4f)\n", r.MeanDelta)
	} else {
		fmt.Fprintf(w, "  FAIL direction: mean modeled delta is not positive (%+.4f)\n", r.MeanDelta)
		ok = false
	}
	if r.SignTest.NUsed > 0 && r.SignTest.Favoring_B == r.SignTest.NUsed {
		fmt.Fprintf(w, "  PASS sign test: all %d non-tied pairs favor Arm B (one-sided p = %.6g)\n",
			r.SignTest.NUsed, r.SignTest.PValue)
	} else {
		fmt.Fprintf(w, "  FAIL sign test: only %d/%d non-tied pairs favor Arm B\n", r.SignTest.Favoring_B, r.SignTest.NUsed)
		ok = false
	}

	if ok {
		fmt.Fprintln(w, "SELFCHECK PASS -- modeled A/B direction matches the thesis; this is still a proxy, not a live-model result.")
	} else {
		fmt.Fprintln(w, "SELFCHECK FAIL")
	}
	return ok
}
