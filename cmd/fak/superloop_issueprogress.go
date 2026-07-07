package main

import (
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// nightIssueProgress measures the LIVE count of issues a run has progressed, reusing the
// dispatch-progress ledger's witnessed close count — the same closed_now fold that
// `fak dispatch progress` reports, which depends only on the ledger (not a gh open-count)
// so it survives a gh outage (#2639). This is the impure measurement the pure
// superloop.Walk cannot make for itself.
//
// The measured bit is load-bearing and fail-closed: when NO dispatch ledger exists yet
// (no run has happened), it returns measured=false so the walk stays surface-only rather
// than fabricating "0 progressed" for an intent that simply has not run. A ledger that is
// present but sums to zero is a REAL measured zero (measured=true) — the gate treats that
// as owing the whole headline, distinct from "unmeasured". This is the same
// declared-vs-measured honesty the pure fold keeps: a number is only gated when witnessed.
func nightIssueProgress(root string) (count int, measured bool) {
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	if _, err := os.Stat(filepath.Join(runsDir, dispatchProgressLogName)); err != nil {
		return 0, false // no dispatch ledger -> not measured, surface-only (never gates)
	}
	return dispatchProgressFoldClosedHistory(runsDir), true
}

// issueProgressWalkOpts is the shell seam that hands a declared-target intent its LIVE
// progress count so superloop.Walk folds the issue-target satisfaction gate. It only
// measures when the intent DECLARES a headline (IssueTarget > 0) — walking an intent with
// no target pays no measurement cost and never gates — and only binds a WithIssueProgress
// option when a dispatch ledger was actually found, so an intent that has not run stays
// surface-only. Returns nil (no opts) in every other case, which Walk treats as an
// ordinary un-gated walk.
func issueProgressWalkOpts(root string, s superloop.Super) []superloop.WalkOpt {
	if s.IssueTarget <= 0 {
		return nil
	}
	count, measured := nightIssueProgress(root)
	if !measured {
		return nil
	}
	return []superloop.WalkOpt{superloop.WithIssueProgress(count)}
}
