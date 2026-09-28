package main

// dispatch_status_throughput.go — the `fak dispatch status` HEADLINE block
// (gateshare/fak/lesson-1).
//
// # The defect this fixes
//
// `dispatch status` led with `live_worker_count`, which is `len(workers)` — a
// PURE LIVENESS count (cmd/fak/dispatch_status.go, dispatchStatusScan). A worker
// that died before it began work is not live, so a total outage and a quiet night
// render IDENTICALLY. internal/dispatchdoa's package doc records what that cost:
// 2026-07-28..08-03, 350 of 382 spawned workers died at argv parse, every one wrote
// a `.witness` carrying `claim: CLAIM_NO_COMMIT, reason: unknown`, and
// `fak dispatch status` — "the surface an operator actually reads" — showed
// "0 live worker(s)" for six days straight, which reads exactly like an idle fleet.
//
// # The fix
//
// Lead the card with what LANDED, and make the decision table explicitly separate
// "nothing ran" (idle) from "everything ran and nothing landed" (outage). The three
// inputs are the ones already folded: commits in a window (collectProgress), the
// finished-spawn census (dispatchdoa), and the live-worker count. Only their
// COMBINATION distinguishes the two states that a single liveness number cannot:
//
//	finished > 0, commits == 0, live == 0  ->  OUTAGE_SUSPECT  (the DOA shape)
//	commits > 0                            ->  LANDING         (work is landing)
//	live > 0, commits == 0                 ->  WORKING         (in flight, none due yet)
//	everything == 0                        ->  IDLE            (genuinely nothing ran)
//	commits unknown                        ->  UNKNOWN         (say so; never fall back)
//
// # Why UNKNOWN exists
//
// The pre-fix card degraded silently: an unavailable progress fold simply vanished
// and the reader was left with the liveness number alone, i.e. exactly the lie this
// file exists to stop. A card that cannot measure must SAY IT CANNOT MEASURE, so
// UNKNOWN is a first-class outcome and never renders as 0.

import (
	"fmt"
	"time"
)

// dispatchThroughputWindow is the headline's landing window. It is deliberately LONGER
// than defaultProgressWindow (10m): a ten-minute commit count is too noisy to make
// "0 commits" mean anything, and the whole point of this block is that a zero must
// be informative. One day is the shortest window over which a zero is evidence.
const dispatchThroughputWindow = 24 * time.Hour

// dispatchThroughputBaseline is the comparison window for the landing rate.
const dispatchThroughputBaseline = 24 * time.Hour

// Landing verdicts. Closed set — an unknown value is a bug, not a state.
const (
	dispatchLandingUnknown       = "unknown"
	dispatchLandingLanding       = "landing"
	dispatchLandingWorking       = "working"
	dispatchLandingIdle          = "idle"
	dispatchLandingOutageSuspect = "outage_suspect"
)

// dispatchThroughput is the fleet-dispatch-status/1 headline block: did work LAND.
type dispatchThroughput struct {
	WindowHours    int    `json:"window_hours"`
	Commits        int    `json:"commits"`
	LiveWorkers    int    `json:"live_workers"`
	FinishedSpawns int    `json:"finished_spawns"`
	DeadOnArrival  int    `json:"doa"`
	Landing        string `json:"landing"`
	Reason         string `json:"reason"`
	NextAction     string `json:"next_action,omitempty"`
	Error          string `json:"error,omitempty"`
}

// dispatchThroughputFold builds the headline from the three already-folded signals.
// It is a PURE function of its arguments so a test drives the whole decision table
// without touching git or the filesystem — which is the point: the outage case must
// be provable, not merely observed.
func dispatchThroughputFold(liveWorkers int, spawn *dispatchStatusSpawnHealth, prog *dispatchProgressSummary) *dispatchThroughput {
	t := &dispatchThroughput{
		WindowHours: int(dispatchThroughputWindow.Hours()),
		LiveWorkers: liveWorkers,
		Landing:     dispatchLandingUnknown,
	}
	if spawn != nil {
		t.FinishedSpawns = spawn.Runs
		t.DeadOnArrival = spawn.DOA
	}

	// The commit count is the headline's payload. If it cannot be read, the block is
	// UNKNOWN and says why — it never degrades to 0, because 0 is a claim.
	if prog == nil || !prog.Available {
		t.Landing = dispatchLandingUnknown
		if prog != nil && prog.Error != "" {
			t.Error = prog.Error
			t.Reason = "commit count unavailable: " + prog.Error
		} else {
			t.Reason = "commit count unavailable"
		}
		t.NextAction = "fix the commit-count fold; do NOT read liveness as landing"
		return t
	}
	t.Commits = prog.Commits

	switch {
	case prog.Commits > 0:
		t.Landing = dispatchLandingLanding
		t.Reason = fmt.Sprintf("%d commit(s) landed in the last %s", prog.Commits, dispatchThroughputWindow)
		t.NextAction = "no action — work is landing"
	case t.LiveWorkers > 0:
		t.Landing = dispatchLandingWorking
		t.Reason = fmt.Sprintf("%d live worker(s), no commit due in the last %s yet", t.LiveWorkers, dispatchThroughputWindow)
		t.NextAction = "give the live workers a tick; re-check the landing count next window"
	case t.FinishedSpawns > 0:
		// THE DOA SHAPE. Workers finished, nothing landed, nothing is running.
		// The old card rendered this as "0 live worker(s)" — indistinguishable from idle.
		t.Landing = dispatchLandingOutageSuspect
		t.Reason = fmt.Sprintf("%d spawn(s) finished, nothing landed, 0 still live — this is an outage, not an idle fleet", t.FinishedSpawns)
		if t.DeadOnArrival > 0 {
			t.NextAction = fmt.Sprintf("%d of %d spawns died before starting — check the launch argv against the binary on PATH, then the launch preflight", t.DeadOnArrival, t.FinishedSpawns)
		} else {
			t.NextAction = "spawns finished without landing — read the spawn-health causes and the first tick of the first dead worker's log"
		}
	default:
		t.Landing = dispatchLandingIdle
		t.Reason = "no spawns, no live workers, no commits — genuinely idle"
		t.NextAction = "nothing running; start a wave if work is expected"
	}
	return t
}

// dispatchThroughputFoldWindowed is dispatchThroughputFold over its OWN commit window.
// The progress summary on the card is a 10-minute stall detector; the headline needs a
// 24-hour landing count, and reusing the 10-minute number would make a busy repo look
// idle at the wrong hour. Failures degrade to UNKNOWN via the shared decision table.
func dispatchThroughputFoldWindowed(root string, liveWorkers int, spawn *dispatchStatusSpawnHealth) *dispatchThroughput {
	r, err := collectProgress(root, dispatchThroughputWindow, dispatchThroughputBaseline, defaultProgressStallAfter, time.Now())
	if err != nil {
		return dispatchThroughputFold(liveWorkers, spawn, &dispatchProgressSummary{Error: err.Error()})
	}
	return dispatchThroughputFold(liveWorkers, spawn, &dispatchProgressSummary{
		Available:       true,
		Verdict:         r.Verdict,
		Commits:         r.Commits,
		GitHubAvailable: r.GitHub.Available,
	})
}

// dispatchThroughputLine renders the headline as the operator's FIRST line, and
// deliberately labels the liveness count as secondary so a reader who stops after
// line one has read the number that can fail.
func dispatchThroughputLine(t *dispatchThroughput) string {
	if t == nil {
		return ""
	}
	return fmt.Sprintf("LANDING [%s] %d commit(s) in %s | live workers %d (secondary) | %s",
		t.Landing, t.Commits, dispatchThroughputWindow, t.LiveWorkers, t.Reason)
}
