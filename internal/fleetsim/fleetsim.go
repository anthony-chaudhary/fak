// Package fleetsim is a deterministic synthetic-ledger model for the "safe 400
// GitHub issues/hour parallel-agent throughput" program (issue #1819,
// fleet-400iph). It answers a planning question the live fleet cannot answer
// safely: "does a fleet of N workers, each closing an issue every W seconds,
// actually sustain the target close rate?" — and it answers it by pure
// computation over a synthetic ledger, WITHOUT spawning, launching, counting, or
// observing any real worker.
//
// The model.
//
// A real dispatch fleet writes a witnessed-close ledger: one row per issue that a
// worker actually finished, carrying the worker's id, when the session started,
// how long it took, and how it ended (closed or failed). fleetsim generates that
// ledger synthetically. Given a concurrency (how many workers run in parallel), a
// median session duration, a total window, and a per-worker failure rate, a
// Fixture lays down deterministic CloseEvents: each of the N workers processes
// back-to-back sessions across the window, and every session that ends in the
// closed outcome is a witnessed close. Replay then folds those events into a
// witnessed closes-per-hour rate and reports whether the target was hit.
//
// Why it hits 400/hour. Little's law run forward: with C workers each turning a
// session every W seconds, the fleet completes C sessions every W seconds, i.e.
// C * (3600 / W) sessions per hour; scaled by the closed fraction (1 - failRate)
// that is the witnessed-close rate. Sizing C >= ceil(400 / ((3600/W) * closed))
// makes the synthetic ledger clear 400 witnessed closes/hour by construction. The
// DefaultFixture picks such a (concurrency, session) pair for the 400 target.
//
// Determinism. There is NO time.Now and NO rand anywhere in this package. Every
// event is a pure function of the fixed parameters passed in (concurrency,
// session duration, window, failure cadence, seed offsets), so a test that
// asserts ">= 400 witnessed closes/hour" is exactly reproducible run to run and
// box to box. Failures are assigned by a deterministic stride derived from the
// integer failure rate, not by sampling a random source.
//
// Scope. Pure model: it imports only the standard library (and, optionally,
// internal/fleetcap for the Little's-law worker-count cross-check). It is off the
// hot path, launches nothing, and has no CLI surface — a clean exported API only.
package fleetsim

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/fleetcap"
)

// Outcome is how a synthetic session ended. Only Closed sessions count as
// witnessed closes; Failed sessions occupy a worker but produce no close.
type Outcome int

const (
	// Closed is a witnessed close: a worker finished an issue. These are the
	// events Replay folds into the closes-per-hour rate.
	Closed Outcome = iota
	// Failed is a session that occupied a worker but did not close an issue
	// (crash, refusal, abandoned). It consumes wall-clock but is not a witness.
	Failed
)

// String renders an Outcome as its ledger token.
func (o Outcome) String() string {
	switch o {
	case Closed:
		return "closed"
	case Failed:
		return "failed"
	default:
		return "unknown"
	}
}

const (
	// secondsPerHour is the window a rate is normalized against.
	secondsPerHour = 3600.0
	// defaultTarget is the program's headline goal: 400 witnessed closes/hour.
	defaultTarget = 400.0
	// defaultSessionSeconds is the median synthetic session (10 minutes), the
	// same median fleetcap sizes its default table around.
	defaultSessionSeconds = 600.0
	// defaultFailRatePct is the synthetic per-session failure rate used when a
	// fixture does not override it: a conservative 5% so the closed fraction
	// (0.95) still clears the target at the sized concurrency.
	defaultFailRatePct = 5
)

// CloseEvent is one row of the synthetic ledger: a single worker session and how
// it ended. It is the atom Replay folds. StartSec/DurationSec are seconds from the
// window origin; a session is a witnessed close iff Outcome == Closed.
type CloseEvent struct {
	WorkerID    int     // 0-based synthetic worker index
	StartSec    float64 // session start, seconds from window origin
	DurationSec float64 // wall-clock the session occupied the worker
	Outcome     Outcome // Closed (a witness) or Failed
}

// EndSec is when the session finished, seconds from the window origin.
func (e CloseEvent) EndSec() float64 { return e.StartSec + e.DurationSec }

// Fixture is the deterministic recipe for a synthetic ledger. It carries no
// generated state — Events() derives the ledger purely from these parameters, so
// the same Fixture always yields the same ledger.
type Fixture struct {
	// Concurrency is the number of synthetic workers running in parallel. Each
	// works back-to-back sessions across the window.
	Concurrency int
	// SessionSeconds is the per-session duration (constant across the model, so
	// the fold is a clean multiple; a real median would jitter but the rate math
	// is identical). Must be > 0.
	SessionSeconds float64
	// WindowSeconds is the total wall-clock the ledger spans. Defaults to one
	// hour when non-positive.
	WindowSeconds float64
	// FailRatePct is the integer per-session failure rate in [0,100]. Failures
	// are assigned deterministically (every k-th session, k derived from the
	// rate), never sampled — so the closed count is exactly reproducible.
	FailRatePct int
}

// window returns the effective window in seconds, defaulting to one hour.
func (f Fixture) window() float64 {
	if f.WindowSeconds > 0 {
		return f.WindowSeconds
	}
	return secondsPerHour
}

// failStride returns the deterministic stride k for failure assignment: every
// k-th session (1-based) across the whole ledger fails. A rate of 0 (or anything
// non-positive) yields a stride of 0, meaning "never fail". The stride is
// round(100/pct), so 5% -> every 20th session, 50% -> every 2nd, 100% -> every
// session.
func (f Fixture) failStride() int {
	if f.FailRatePct <= 0 {
		return 0
	}
	if f.FailRatePct >= 100 {
		return 1
	}
	return int(math.Round(100.0 / float64(f.FailRatePct)))
}

// Events materializes the synthetic ledger deterministically. Each of the
// Concurrency workers processes back-to-back SessionSeconds sessions from the
// window origin until the next session would start at or after the window end;
// events are laid down worker-major (all of worker 0's sessions, then worker 1's,
// …) so the order is a pure function of the parameters. The global 1-based session
// index drives failure assignment via failStride, giving an exactly reproducible
// closed/failed split with no randomness.
func (f Fixture) Events() []CloseEvent {
	if f.Concurrency <= 0 || f.SessionSeconds <= 0 {
		return nil
	}
	win := f.window()
	stride := f.failStride()
	// Sessions each worker can start strictly before the window end.
	perWorker := int(math.Floor(win / f.SessionSeconds))
	if perWorker <= 0 {
		return nil
	}
	events := make([]CloseEvent, 0, f.Concurrency*perWorker)
	globalIdx := 0 // 1-based after the increment below; drives failure stride
	for w := 0; w < f.Concurrency; w++ {
		for s := 0; s < perWorker; s++ {
			globalIdx++
			outcome := Closed
			if stride > 0 && globalIdx%stride == 0 {
				outcome = Failed
			}
			events = append(events, CloseEvent{
				WorkerID:    w,
				StartSec:    float64(s) * f.SessionSeconds,
				DurationSec: f.SessionSeconds,
				Outcome:     outcome,
			})
		}
	}
	return events
}

// DefaultFixture returns a Fixture sized so its synthetic ledger clears the 400
// witnessed-closes/hour program target over a one-hour window, using the default
// 10-minute session and 5% failure rate. The concurrency is derived, not guessed:
// it is fleetcap.RequiredWorkers for the target grossed up by the closed fraction,
// so the closes (not just the sessions) clear 400/hour.
func DefaultFixture() Fixture {
	return FixtureForTarget(defaultTarget, defaultSessionSeconds, defaultFailRatePct)
}

// FixtureForTarget sizes a one-hour Fixture whose witnessed closes/hour is >=
// target, given a session duration and integer failure rate. It grosses the target
// up by 1/(closed fraction) so the closed events — not merely the sessions — clear
// the target, then asks fleetcap.RequiredWorkers for the Little's-law concurrency.
// A non-positive/absurd input falls back to the package defaults.
func FixtureForTarget(target, sessionSeconds float64, failRatePct int) Fixture {
	if target <= 0 {
		target = defaultTarget
	}
	if sessionSeconds <= 0 {
		sessionSeconds = defaultSessionSeconds
	}
	if failRatePct < 0 || failRatePct >= 100 {
		failRatePct = defaultFailRatePct
	}
	closedFrac := 1.0 - float64(failRatePct)/100.0
	// Sessions/hour we must sustain so that closed sessions >= target.
	grossedRate := target / closedFrac
	workers := fleetcap.RequiredWorkers(grossedRate, sessionSeconds/60.0)
	if workers <= 0 {
		workers = 1
	}
	return Fixture{
		Concurrency:    workers,
		SessionSeconds: sessionSeconds,
		WindowSeconds:  secondsPerHour,
		FailRatePct:    failRatePct,
	}
}

// Report is the folded result of a dry-run Replay: the witnessed-close counts and
// the window they span. It computes derived rates on demand so nothing is stored
// that could drift from the counts.
type Report struct {
	Workers       int     // distinct synthetic workers in the ledger
	TotalSessions int     // all sessions, closed + failed
	Closes        int     // witnessed closes (Outcome == Closed)
	Failures      int     // failed sessions
	WindowSeconds float64 // wall-clock the ledger spans
}

// ClosesPerHour is the witnessed-close rate: closes normalized to a one-hour
// window. A zero/negative window yields 0 (no basis to project a rate).
func (r Report) ClosesPerHour() float64 {
	if r.WindowSeconds <= 0 {
		return 0
	}
	return float64(r.Closes) * (secondsPerHour / r.WindowSeconds)
}

// SessionsPerHour is the total-session throughput (closed + failed) per hour, the
// pre-failure ceiling ClosesPerHour is a fraction of.
func (r Report) SessionsPerHour() float64 {
	if r.WindowSeconds <= 0 {
		return 0
	}
	return float64(r.TotalSessions) * (secondsPerHour / r.WindowSeconds)
}

// TargetAchieved reports whether the witnessed closes-per-hour meets or exceeds
// target. It is the ticket's done-condition predicate: true means the dry-run
// replay proved the target without any real worker running.
func (r Report) TargetAchieved(target float64) bool {
	return r.ClosesPerHour() >= target
}

// Replay folds a Fixture into a Report by pure computation. It materializes the
// synthetic ledger and counts witnessed closes — it spawns, launches, and observes
// NOTHING. Calling Replay has no side effects: no processes, no I/O, no clock, no
// randomness; the same Fixture always yields the same Report.
func Replay(f Fixture) Report {
	events := f.Events()
	rep := Report{WindowSeconds: f.window()}
	seen := make(map[int]struct{})
	for _, e := range events {
		rep.TotalSessions++
		seen[e.WorkerID] = struct{}{}
		switch e.Outcome {
		case Closed:
			rep.Closes++
		case Failed:
			rep.Failures++
		}
	}
	rep.Workers = len(seen)
	return rep
}
