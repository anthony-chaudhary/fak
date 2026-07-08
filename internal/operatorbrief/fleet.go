package operatorbrief

// fleet.go folds the cross-ledger loop-liveness pane (internal/loopfleet) into the
// operator brief as one more optional source, exactly like heaviness. It is the
// answer to a blind spot the static report envelopes leave open: cadence/program/
// milestone/heaviness all describe WORK, but none of them notices when a LOOP that
// should be ticking has gone quiet. Without this, the brief can tell an operator to
// "stay out of the loop / monitor" while a scheduled loop has silently gone dark.
//
// The mapping is conservative on purpose — a dark loop is surfaced as a WATCH, not a
// human page. A loop being quiet is a "review and decide whether to revive or retire"
// signal, not an unmeasured-witness emergency; auto-paging on it would train operators
// to ignore the pane (some loops are intentionally off on a given host). So Fleet only
// ever shapes attention (Watch/Background), never flips the paging gate on its own.

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
)

// fleetState records the loop-fleet pane's presence and derived verdict as a source
// stamp. Like heaviness it carries no date/commit, so it never perturbs snapshot
// coherence. A fleet with no folded loops reads as "unmeasured" (no ledger present),
// a dark loop as "action", everything else as measured-ok.
func fleetState(r *loopfleet.Report) SourceState {
	if r == nil {
		return SourceState{Name: "fleet", Status: "missing"}
	}
	ru := r.Rollup
	verdict, finding := "OK", "fleet_live"
	switch {
	case ru.Loops == 0:
		verdict, finding = "OK", "fleet_unmeasured"
	case ru.Dark > 0:
		verdict, finding = "ACTION", "loops_dark"
	case ru.Stale > 0:
		verdict, finding = "OK", "loops_stale"
	}
	return SourceState{Name: "fleet", Schema: r.Schema, Status: reportStatus(ru.Dark == 0, finding), Verdict: verdict, Finding: finding}
}

// addFleet folds the loop-fleet rollup into a single worst-first item. Dark beats
// stale beats live; an absent ledger set is background (nothing to witness yet), so
// the fleet source can never manufacture a human page — only a review-worthy watch.
func addFleet(r *Report, f loopfleet.Report) {
	ru := f.Rollup
	detail := fleetDetail(f)
	switch {
	case ru.Loops == 0:
		r.addBackground("fleet", "no loop ledgers measured", detail,
			"run the loops (nightrun/dispatch/cadence) so fleet liveness is witnessed in the brief")
	case ru.Dark > 0:
		r.addWatch("fleet", fmt.Sprintf("%s dark", loopNoun(ru.Dark)), detail,
			"check each dark loop: revive it, or retire its ledger if the loop is intentionally stopped")
	case ru.Stale > 0:
		r.addWatch("fleet", fmt.Sprintf("%s slipping", loopNoun(ru.Stale)), detail,
			"watch the stale loop(s); if their age keeps rising past cadence they go dark")
	default:
		r.addBackground("fleet", "all loops live", detail,
			"no loop has gone quiet past its cadence; keep the scheduled cadence")
	}
}

// fleetDetail summarizes the rollup and names the loops behind any dark/stale verdict,
// so the operator sees which loop to look at without re-running `fak loop rollup`.
func fleetDetail(f loopfleet.Report) string {
	ru := f.Rollup
	parts := []string{fmt.Sprintf("%d loop(s): %d live, %d stale, %d dark, %d unknown",
		ru.Loops, ru.Live, ru.Stale, ru.Dark, ru.Unknown)}
	if dark := kindsInState(f.Loops, "dark"); dark != "" {
		parts = append(parts, "dark: "+dark)
	}
	if stale := kindsInState(f.Loops, "stale"); stale != "" {
		parts = append(parts, "stale: "+stale)
	}
	if ru.Skipped > 0 {
		parts = append(parts, fmt.Sprintf("%d ledger(s) absent/unreadable", ru.Skipped))
	}
	return strings.Join(parts, "; ")
}

// kindsInState lists the loop kinds currently in a given health state, comparing
// against loopmgr's stable state vocabulary as strings so operatorbrief need not
// import loopmgr just to name a bucket.
func kindsInState(loops []loopfleet.LoopHealth, state string) string {
	var names []string
	for _, l := range loops {
		if string(l.State) == state {
			names = append(names, l.Kind)
		}
	}
	return strings.Join(names, ", ")
}

func loopNoun(n int) string {
	if n == 1 {
		return "1 loop"
	}
	return fmt.Sprintf("%d loops", n)
}
