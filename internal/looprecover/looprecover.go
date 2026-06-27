// Package looprecover is the deterministic recovery decision the fak dispatch fleet is
// missing: given the durable loop ledger's record of every dispatched run, which runs STARTED
// but never finished or were never witnessed — the work that should be re-dispatched or
// re-verified rather than left silently abandoned?
//
// # The gap it closes
//
// The loop ledger (internal/loopmgr) records fire/admit/start/heartbeat/end/witness events for
// every run, and loopmgr.Summarize folds them into a per-LOOP snapshot (counts + the last run).
// What no surface produces is the cross-run RECOVERY WORKLIST: the specific runs that are
// orphaned (a worker started, then went silent and never ended — a crash, a rate limit, a
// timeout) or unwitnessed (the run ended or claimed done, but no independent witness ever bound
// it). dispatch_status.py's silent-worker scan finds dead PIDs with empty logs, and
// resume_watch.py classifies resumed sessions, but neither answers "across the whole ledger,
// which dispatched units are incomplete and need to run again?". This package is that answer.
//
// # Recovery vs backfill
//
// This leaf covers RECOVERY — re-run/re-verify work that STARTED and failed. BACKFILL (fill
// work that was skipped or whose schedule window was missed) needs a schedule model the ledger
// alone does not carry, and is a separate, later rung. The two recovery dispositions here,
// orphaned and unwitnessed, are exactly the rows a re-dispatch driver would feed back as fresh
// candidates (composing with internal/dispatchorder, which would then collapse any duplicates).
//
// # Pure, total, and robust
//
// Plan takes the clock as data (NowUnix) and imports nothing internal — same facts, same
// Result. The orphan call prefers a CONFIRMED worker-liveness signal when the caller has one
// (a started run whose worker is known-dead is orphaned at once; a known-live worker is never
// orphaned however long it runs), and falls back to a conservative staleness window only when
// liveness is unknown — so a legitimately long-running worker is not falsely reclaimed. The
// impure half (read the ledger, optionally probe the worker pid) lives in the cmd/fak shell,
// the same leaf/shell split internal/resume and internal/dispatchorder use.
package looprecover

import "sort"

// DefaultStaleSeconds is how long a started-but-unfinished run with UNKNOWN worker liveness may
// stay silent before it is presumed orphaned. It is deliberately generous (longer than a
// dispatch fire interval plus a typical worker's runtime) so a slow-but-alive worker is not
// reclaimed; the precise signal is confirmed liveness, which overrides this entirely.
const DefaultStaleSeconds = 45 * 60

// Disposition is the recovery verdict for one run.
type Disposition string

const (
	// DispComplete: the run was witnessed — proven done, nothing to recover.
	DispComplete Disposition = "complete"
	// DispRunning: the run is in progress (a live worker, or recent activity within the stale
	// window) — leave it alone.
	DispRunning Disposition = "running"
	// DispOrphaned: the run started and then went silent (worker confirmed dead, or no activity
	// past the stale window) without ending or being witnessed — a re-dispatch candidate.
	DispOrphaned Disposition = "orphaned"
	// DispUnwitnessed: the run ended or claimed done but no witness ever bound it — a re-verify
	// (and, if still unproven, re-dispatch) candidate.
	DispUnwitnessed Disposition = "unwitnessed"
	// DispFailed: the run reached a terminal failure (failed or canceled) — reported, but
	// retrying is the operator's call, not an automatic recovery.
	DispFailed Disposition = "failed"
)

// The closed reason vocabulary for a Ranked.Reason.
const (
	ReasonWitnessed        = "witnessed"           // complete: an independent witness bound the run
	ReasonWorkerLive       = "worker_live"         // running: the worker is confirmed alive
	ReasonRecentActivity   = "recent_activity"     // running: activity within the stale window
	ReasonWorkerDead       = "worker_dead"         // orphaned: the worker is confirmed dead
	ReasonSilentPastStale  = "silent_past_stale"   // orphaned: no activity past the stale window (presumed)
	ReasonEndedUnwitness   = "ended_unwitnessed"   // unwitnessed: ended with no witness
	ReasonClaimedUnwitness = "claimed_unwitnessed" // unwitnessed: claimed done with no witness
	ReasonFailed           = "failed"              // failed: a terminal failure status
	ReasonCanceled         = "canceled"            // failed: canceled
)

// RunFact is the closed set of facts about ONE dispatched run that the recovery decision needs
// — folded by the shell from the ledger events sharing a run id, never the event stream itself.
type RunFact struct {
	// RunID identifies the run (the ledger's run_id); echoed in the worklist.
	RunID string `json:"run_id"`
	// LoopID is the loop the run belongs to (e.g. "issue-resolve-dispatch/...").
	LoopID string `json:"loop_id"`
	// Unit is an optional human label for the work (an issue #, a lane) lifted from the run's
	// event summaries — for the operator's eye only; the decision never depends on it.
	Unit string `json:"unit,omitempty"`
	// Started/Ended/Witnessed/Claimed/Failed/Canceled are whether the run reached each state
	// (folded from the ledger event kinds and run statuses).
	Started   bool `json:"started"`
	Ended     bool `json:"ended"`
	Witnessed bool `json:"witnessed"`
	Claimed   bool `json:"claimed"`
	Failed    bool `json:"failed"`
	Canceled  bool `json:"canceled"`
	// LastEventUnix is the time of the run's most recent ledger event (the silence clock).
	LastEventUnix int64 `json:"last_event_unix"`
	// WorkerKnown reports whether the caller probed the run's worker liveness at all; when false
	// the decision falls back to staleness. WorkerLive is meaningful only when WorkerKnown.
	WorkerKnown bool `json:"worker_known"`
	WorkerLive  bool `json:"worker_live"`
}

// Ranked is one run with the recovery verdict attached.
type Ranked struct {
	RunFact
	Disposition Disposition `json:"disposition"`
	Reason      string      `json:"reason"`
	// AgeSeconds is how long since the run's last event (NowUnix - LastEventUnix, floored at 0).
	AgeSeconds int64 `json:"age_seconds"`
}

// Input is everything Plan needs: the run facts, the clock as data, and the staleness window.
type Input struct {
	Runs []RunFact `json:"runs"`
	// NowUnix is the current time as data (the leaf never reads a clock).
	NowUnix int64 `json:"now_unix"`
	// StaleSeconds is the silence window for a worker-unknown run (0 => DefaultStaleSeconds;
	// negative disables staleness so only confirmed-dead workers are orphaned).
	StaleSeconds int64 `json:"stale_seconds"`
}

// Result is the full recovery verdict: every run's disposition plus the actionable worklist.
type Result struct {
	// Runs is every run, recovery candidates first (orphaned, then unwitnessed), each oldest-first.
	Runs []Ranked `json:"runs"`
	// Recover is the RunIDs that need recovery (orphaned ∪ unwitnessed), oldest-first — the worklist.
	Recover []string `json:"recover"`
	// Counts of each disposition.
	OrphanedCount    int `json:"orphaned_count"`
	UnwitnessedCount int `json:"unwitnessed_count"`
	RunningCount     int `json:"running_count"`
	CompleteCount    int `json:"complete_count"`
	FailedCount      int `json:"failed_count"`
}

// Plan is THE deterministic recovery decision: same Input in, same Result out — no clock, no
// I/O. It classifies every run and returns the orphaned-and-unwitnessed worklist, oldest-first.
// Total over any input (an empty run set yields an empty, defined Result).
func Plan(in Input) Result {
	stale := in.StaleSeconds
	if stale == 0 {
		stale = DefaultStaleSeconds
	}

	ranked := make([]Ranked, 0, len(in.Runs))
	for _, f := range in.Runs {
		age := in.NowUnix - f.LastEventUnix
		if age < 0 {
			age = 0
		}
		r := Ranked{RunFact: f, AgeSeconds: age}
		r.Disposition, r.Reason = classify(f, age, stale)
		ranked = append(ranked, r)
	}

	// Order: orphaned first, then unwitnessed (the worklist), each oldest-first; then running,
	// complete, failed. Deterministic and stable.
	sort.SliceStable(ranked, func(i, j int) bool {
		pi, pj := recoverRank(ranked[i].Disposition), recoverRank(ranked[j].Disposition)
		if pi != pj {
			return pi < pj
		}
		if ranked[i].AgeSeconds != ranked[j].AgeSeconds {
			return ranked[i].AgeSeconds > ranked[j].AgeSeconds // oldest (most stuck) first
		}
		return ranked[i].RunID > ranked[j].RunID // total order for determinism
	})

	out := Result{Runs: ranked}
	for _, r := range ranked {
		switch r.Disposition {
		case DispOrphaned:
			out.OrphanedCount++
			out.Recover = append(out.Recover, r.RunID)
		case DispUnwitnessed:
			out.UnwitnessedCount++
			out.Recover = append(out.Recover, r.RunID)
		case DispRunning:
			out.RunningCount++
		case DispComplete:
			out.CompleteCount++
		case DispFailed:
			out.FailedCount++
		}
	}
	return out
}

// classify applies the recovery rungs in precedence order. A witnessed run is complete; a
// terminal failure is failed; an ended-or-claimed run that was never witnessed is unwitnessed;
// a started run is orphaned when its worker is confirmed dead, or (worker unknown) when it has
// been silent past the stale window, else running.
func classify(f RunFact, age, stale int64) (Disposition, string) {
	switch {
	case f.Witnessed:
		return DispComplete, ReasonWitnessed
	case f.Canceled:
		return DispFailed, ReasonCanceled
	case f.Failed:
		return DispFailed, ReasonFailed
	case f.Ended:
		return DispUnwitnessed, ReasonEndedUnwitness
	case f.Claimed:
		return DispUnwitnessed, ReasonClaimedUnwitness
	case f.Started:
		switch {
		case f.WorkerKnown && !f.WorkerLive:
			return DispOrphaned, ReasonWorkerDead
		case f.WorkerKnown && f.WorkerLive:
			return DispRunning, ReasonWorkerLive
		case stale >= 0 && age >= stale:
			return DispOrphaned, ReasonSilentPastStale
		default:
			return DispRunning, ReasonRecentActivity
		}
	default:
		// Admitted/armed but never started — nothing has run yet; leave it (a backfill concern,
		// out of this leaf's recovery scope).
		return DispRunning, ReasonRecentActivity
	}
}

// recoverRank orders the dispositions for the worklist: orphaned, unwitnessed, then the rest.
func recoverRank(d Disposition) int {
	switch d {
	case DispOrphaned:
		return 0
	case DispUnwitnessed:
		return 1
	case DispRunning:
		return 2
	case DispComplete:
		return 3
	default: // DispFailed
		return 4
	}
}
