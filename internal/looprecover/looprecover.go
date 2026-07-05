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
// liveness is unknown — so a legitimately long-running worker is not falsely reclaimed. That
// confirmed signal comes from Probe, the pure pid-liveness primitive here: given a run's
// recorded worker identity and a caller-injected Liveness lookup, it returns DEAD / ALIVE /
// UNKNOWN, defeating pid reuse with start-time identity (a held-but-different-start pid is a
// new process, so the recorded worker is DEAD). Re-dispatch is thereby gated on DEAD: a
// confirmed-alive-but-silent worker classifies ALIVE_SILENT (a liveness-beat problem) and stays
// out of the worklist. The impure half (read the ledger, run the OS side of the Liveness probe)
// lives in the cmd/fak shell, the same leaf/shell split internal/resume and
// internal/dispatchorder use.
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
	ReasonWorkerLive       = "worker_live"         // running: the worker is confirmed alive and recently active
	ReasonAliveSilent      = "alive_silent"        // running: worker confirmed alive but ledger-silent past stale (a liveness-beat problem, never re-dispatched)
	ReasonRecentActivity   = "recent_activity"     // running: activity within the stale window
	ReasonWorkerDead       = "worker_dead"         // orphaned: the worker is confirmed dead (or its pid was reused)
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
	// WorkerPID/WorkerHost/WorkerStart are the run's worker identity as folded from the ledger:
	// the process id (loop.go stamps it into the start/end event Metrics), its host, and a
	// reuse-safe start-time fingerprint. They are what a caller probes for CONFIRMED liveness
	// (see Probe); all-zero means the run carries no probeable worker and staleness decides.
	WorkerPID   int    `json:"worker_pid,omitempty"`
	WorkerHost  string `json:"worker_host,omitempty"`
	WorkerStart string `json:"worker_start,omitempty"`
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
			// DEAD: the worker is confirmed gone (or its pid was reused) — the one signal that
			// gates re-dispatch. Orphaned however recent the last event.
			return DispOrphaned, ReasonWorkerDead
		case f.WorkerKnown && f.WorkerLive && stale >= 0 && age >= stale:
			// ALIVE_SILENT: the worker is confirmed alive but has gone ledger-silent past the
			// stale window — a liveness-beat problem, NOT an orphan. Never re-dispatched (a live
			// worker doing the work must not be double-run); surfaced distinctly for the beat rung.
			return DispRunning, ReasonAliveSilent
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

// ProbeVerdict is the pid-liveness verdict for one run's recorded worker — the precise signal
// that lets recovery distinguish a DEAD worker (safe to re-dispatch) from an ALIVE-but-silent
// one (a liveness-beat problem, never a re-dispatch target). It is stronger than the staleness
// window: a confirmed verdict overrides age entirely.
type ProbeVerdict string

const (
	// ProbeUnknown: the run carries no probeable pid, or the caller supplied no prober — liveness
	// is unknown and the decision falls back to the staleness window.
	ProbeUnknown ProbeVerdict = "unknown"
	// ProbeDead: the recorded worker process is gone — or a DIFFERENT process now holds its pid
	// (pid reuse, caught by start-time identity). Safe to re-dispatch / re-verify.
	ProbeDead ProbeVerdict = "dead"
	// ProbeAlive: the recorded worker process is still running — its pid is held by a process
	// whose start-time identity matches what the run recorded. Never a re-dispatch target.
	ProbeAlive ProbeVerdict = "alive"
)

// Liveness is the OS-facing lookup the impure shell injects as DATA, so Probe stays pure and
// testable with a decoy pid: given a pid, report whether a process currently holds it and, if
// so, that process's start-time identity (a reuse-safe fingerprint — e.g. the OS process
// creation time, the same idea as procguard.StreakKey's start field). A returned (,,false)
// means no process holds the pid.
type Liveness func(pid int) (start string, running bool)

// Probe is the pure, reuse-safe pid-liveness decision — the primitive the recovery shell calls
// per orphan candidate before trusting the staleness window. It defeats pid reuse with
// start-time identity: a pid that is HELD but whose current start identity differs from the one
// the run recorded is a DIFFERENT process, so the original worker is DEAD, not alive. When
// either side carries no start identity, Probe cannot prove reuse and reports the liveness it
// can see (an alive pid ⇒ ProbeAlive). No pid or no prober ⇒ ProbeUnknown (staleness decides).
func Probe(pid int, recordedStart string, live Liveness) ProbeVerdict {
	if pid <= 0 || live == nil {
		return ProbeUnknown
	}
	start, running := live(pid)
	if !running {
		return ProbeDead
	}
	if recordedStart != "" && start != "" && start != recordedStart {
		// pid reuse: a new process now holds the pid; the run's recorded worker is gone.
		return ProbeDead
	}
	return ProbeAlive
}

// ProbeRun probes a run's recorded worker identity (WorkerPID + WorkerStart) for liveness — the
// convenience form of Probe over a RunFact the shell has folded from the ledger.
func ProbeRun(f RunFact, live Liveness) ProbeVerdict {
	return Probe(f.WorkerPID, f.WorkerStart, live)
}

// ApplyProbe folds a probe verdict into a run's worker-liveness facts so Plan can consume it: a
// confirmed verdict (alive/dead) sets WorkerKnown and WorkerLive accordingly; ProbeUnknown
// leaves the fields untouched so the decision falls back to the staleness window. Returns the
// updated RunFact by value — the leaf never mutates its caller's slice in place.
func (f RunFact) ApplyProbe(v ProbeVerdict) RunFact {
	switch v {
	case ProbeAlive:
		f.WorkerKnown, f.WorkerLive = true, true
	case ProbeDead:
		f.WorkerKnown, f.WorkerLive = true, false
	}
	return f
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
