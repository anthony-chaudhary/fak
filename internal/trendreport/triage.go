package trendreport

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// This file applies the decenter-the-human doctrine at the shared report-gate
// seam. AdvisoryGate pages (exit 1) whenever a report is INCOMPLETE — a dimension
// could not be measured — and hands the operator a NextAction. But an incomplete
// report whose NextAction is a runnable rerun (`repair scores, then rerun
// fak cadence`) is not a person's call: it is obvious, fleet-routable work. Only
// an incomplete report whose next move names authority a person holds is a genuine
// page.
//
// AdvisoryGateTriaged folds the report's own page-vs-act decision through
// internal/choicetriage so the disposition is set AT THE SOURCE — before the
// operator brief ever folds the report — and every report that embeds Envelope
// (cadence, milestone, and any future consumer) inherits it. It soaks behind an
// enforce/warn switch: "warn"/"" returns the base gate byte-for-byte, so the
// change is observable before it stops paging.

// TriageEnforced reports whether the decenter-the-human fold is active for the
// given mode string. Only "enforce" (case-insensitive) flips paging; "warn", ""
// and anything else leave AdvisoryGate's finding-only paging unchanged so the
// change can soak. Mirrors operatorbrief's enforce/warn soak switch.
func TriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// TriageEnvelope folds a report envelope's own page-vs-act decision through
// choicetriage: the Finding is the surfaced question, the Reason the "why", and
// the NextAction the concrete move already in hand. NeedsHuman is the one bit a
// gate reads — true only when the report's next move names authority a person
// holds.
func TriageEnvelope(label string, e Envelope) choicetriage.Verdict {
	return choicetriage.Triage(choicetriage.Signal{
		Source:   label,
		Question: e.Finding,
		Detail:   e.Reason,
		Action:   e.NextAction,
	})
}

// AdvisoryGateTriaged is AdvisoryGate with the decenter-the-human fold applied.
// When the base gate would page (an INCOMPLETE report) but the report's NextAction
// routes to the fleet — a runnable rerun, a fresh-context evaluation, or a ticket,
// none of which a person holds authority over — the gate clears and names the
// route instead of paging. A report whose next move DOES name authority still
// pages. Soaks behind enforce; warn/"" returns the base gate unchanged.
func AdvisoryGateTriaged(label, finding, reason, nextAction, unmeasuredFinding string, enforce bool) GateVerdict {
	base := AdvisoryGate(label, finding, reason, unmeasuredFinding)
	if base.Exit == 0 || !enforce {
		return base
	}
	v := choicetriage.Triage(choicetriage.Signal{
		Source:   label,
		Question: finding,
		Detail:   reason,
		Action:   nextAction,
	})
	if v.NeedsHuman {
		return base
	}
	return GateVerdict{Exit: 0, Message: label + " ROUTED: " + reason + " — " + v.Resolve}
}

// TriageSelfcheck is the deterministic, no-I/O proof of the report-seam fold: an
// incomplete report with a runnable rerun routes to the fleet (no page) under
// enforce, still pages under warn (soak), and a measured report is unchanged.
func TriageSelfcheck() error {
	const (
		label  = "CADENCE"
		unmeas = "cadence_unmeasured"
		reason = "a dimension could not be measured"
		rerun  = "repair scores, then rerun `fak cadence`"
	)

	if g := AdvisoryGateTriaged(label, unmeas, reason, rerun, unmeas, true); g.Exit != 0 {
		return fmt.Errorf("a runnable incomplete report must route to the fleet under enforce, got exit %d (%s)", g.Exit, g.Message)
	}
	if g := AdvisoryGateTriaged(label, unmeas, reason, rerun, unmeas, false); g.Exit != 1 {
		return fmt.Errorf("warn mode must leave the incomplete page unchanged, got exit %d", g.Exit)
	}
	// An incomplete report whose next move names authority still pages.
	if g := AdvisoryGateTriaged("RELEASE", "release_unmeasured", "publish decision pending", "approve the tagged release before publish", "release_unmeasured", true); g.Exit != 1 {
		return fmt.Errorf("an authority-bearing incomplete report must still page under enforce, got exit %d", g.Exit)
	}
	// A measured report never paged; triage leaves it OK.
	if g := AdvisoryGateTriaged(label, "cadence_recorded", "recorded", "", unmeas, true); g.Exit != 0 {
		return fmt.Errorf("a measured report must stay OK, got exit %d", g.Exit)
	}
	if !TriageEnforced("enforce") || TriageEnforced("") || TriageEnforced("warn") {
		return fmt.Errorf("TriageEnforced must flip only on \"enforce\"")
	}
	return nil
}
