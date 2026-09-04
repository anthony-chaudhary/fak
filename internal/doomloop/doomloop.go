package doomloop

import "fmt"

// Verdict is the closed classification vocabulary for one worker's trailing
// sample window. Exactly one verdict describes a worker at a sample boundary.
type Verdict string

const (
	// VerdictUnknown: too little evidence to decide. The fail-closed default -
	// a worker we cannot read is never called a doom loop and never corrected.
	VerdictUnknown Verdict = "UNKNOWN"
	// VerdictHealthy: the latest window advanced verified progress, or the doom
	// pattern has not yet tripped. Nothing to correct.
	VerdictHealthy Verdict = "HEALTHY"
	// VerdictIdle: alive but not spending - a parked or quiet worker. Not a doom
	// loop (a doom loop burns); left alone.
	VerdictIdle Verdict = "IDLE"
	// VerdictWedged: not spending AND not alive at the latest sample - a frozen
	// worker. A stall, not a doom loop; surfaced for a different rung (resume).
	VerdictWedged Verdict = "WEDGED"
	// VerdictDoomLoop: burning effort with flat verified progress for at least
	// TripWindows consecutive windows - the confirmed pattern this leaf exists
	// to catch.
	VerdictDoomLoop Verdict = "DOOM_LOOP"
)

// Correction is the closed, graduated recommendation vocabulary. The ladder is
// reversible-first: it climbs from no-op to a soft nudge to an operator
// escalation, and deliberately stops below any destructive rung.
type Correction string

const (
	// CorrectNone: take no action.
	CorrectNone Correction = "NONE"
	// CorrectObserve: record the sample; a sub-threshold burning-flat streak is
	// building but has not tripped. Watch, do not intervene.
	CorrectObserve Correction = "OBSERVE"
	// CorrectNudge: deliver a soft, reversible re-anchor packet to the session
	// steer channel - the first real intervention. Injects "re-read your
	// objective," nothing destructive.
	CorrectNudge Correction = "NUDGE"
	// CorrectEscalate: hand the persistent doom loop to an operator. Still not
	// destructive - a structured record + notification, never an auto-teardown.
	CorrectEscalate Correction = "ESCALATE"
)

// ReasonDoomLoop is the closed refusal-vocabulary token a caller stamps when it
// gates continued unproductive spend on a confirmed doom loop. Mirrors the
// dos.toml [reasons] entry so the emitted reason is verifiable, not prose.
const ReasonDoomLoop = "DOOM_LOOP"

// Sample is one observation of a worker at a sample boundary. The core reads
// only DELTAS between adjacent samples, so the absolute origin of each counter
// does not matter - only that Effort and Progress are monotone non-decreasing
// within one worker's stream.
type Sample struct {
	// UnixMillis is when this observation was taken. Injected by the caller -
	// the core carries no clock so it stays pure and deterministic under test.
	UnixMillis int64 `json:"unix_millis"`
	// Effort is a monotone counter of work SPENT: transcript lines, assistant
	// turns, tool-uses, or tokens. Its positive delta means "still burning."
	Effort int64 `json:"effort"`
	// Progress is a monotone counter of VERIFIED forward progress: commits landed
	// on the worker's region, witnessed intent-ledger steps. Never a self-report.
	// Its flat delta under burning effort is the doom-loop signal.
	Progress int64 `json:"progress"`
	// Alive reports whether the worker's process/heartbeat was live at this
	// sample. Distinguishes a frozen worker (WEDGED) from a merely quiet one
	// (IDLE) when effort is not being spent.
	Alive bool `json:"alive"`
}

// Config tunes the classifier. The zero value is not valid; callers should start
// from DefaultConfig and adjust.
type Config struct {
	// MinSamples is the fewest samples that permit any verdict but UNKNOWN. Below
	// it the core fails closed. Must be >= 2 (a delta needs two points).
	MinSamples int
	// TripWindows is K: the count of consecutive burning-flat windows that
	// confirms a DOOM_LOOP and recommends a NUDGE.
	TripWindows int
	// EscalateWindows is the longer streak at which the recommendation climbs
	// from NUDGE to ESCALATE. Must be >= TripWindows.
	EscalateWindows int
	// EffortEpsilon: an effort delta must be strictly greater than this to count
	// as "burning." Default 0 (any positive delta burns).
	EffortEpsilon int64
	// ProgressEpsilon: a progress delta must be strictly greater than this to
	// count as "advancing." Default 0 (any positive delta advances).
	ProgressEpsilon int64
}

// DefaultConfig is the tuned starting point: at least 3 samples to decide, trip
// a doom loop after 3 consecutive burning-flat windows, escalate after 6.
func DefaultConfig() Config {
	return Config{
		MinSamples:      3,
		TripWindows:     3,
		EscalateWindows: 6,
		EffortEpsilon:   0,
		ProgressEpsilon: 0,
	}
}

func (c Config) withDefaults() Config {
	if c.MinSamples < 2 {
		c.MinSamples = DefaultConfig().MinSamples
	}
	if c.TripWindows < 1 {
		c.TripWindows = DefaultConfig().TripWindows
	}
	if c.EscalateWindows < c.TripWindows {
		c.EscalateWindows = c.TripWindows
	}
	return c
}

// Result is the classified outcome for one worker's trailing window.
type Result struct {
	Verdict    Verdict    `json:"verdict"`
	Correction Correction `json:"correction"`
	Reason     string     `json:"reason"`
	// BurningFlatStreak is the length of the trailing run of consecutive
	// burning-flat windows - the evidence behind a DOOM_LOOP verdict and the
	// dial the correction ladder reads.
	BurningFlatStreak int `json:"burning_flat_streak"`
	// EffortDelta and ProgressDelta are the latest window's deltas, surfaced so a
	// caller can render "burned N, landed 0" without recomputing.
	EffortDelta   int64 `json:"effort_delta"`
	ProgressDelta int64 `json:"progress_delta"`
	// Windows is the number of adjacent-sample windows the verdict rested on.
	Windows int `json:"windows"`
}

// window is one adjacent-sample transition.
type window struct {
	effortDelta   int64
	progressDelta int64
}

// Classify folds one worker's ordered samples (oldest first) into a verdict and
// a graduated correction recommendation. Pure: it reads only the samples and the
// config, allocates nothing observable, and never acts.
//
// The decision, in order:
//
//  1. Too few samples -> UNKNOWN / NONE. Fail closed: we do not correct a worker
//     we cannot yet read.
//  2. The latest window advanced verified progress -> HEALTHY / NONE. Real
//     progress ends any doom episode immediately; the streak resets to zero.
//  3. The latest window is not burning (effort flat) -> IDLE if alive, WEDGED if
//     not. A worker that is not spending cannot be in a doom LOOP; whichever
//     stall it is belongs to another rung.
//  4. The latest window is burning AND flat -> measure the trailing burning-flat
//     streak and climb the ladder: below TripWindows it is HEALTHY / OBSERVE
//     (watching, not yet intervening); at TripWindows it is DOOM_LOOP / NUDGE;
//     at EscalateWindows it is DOOM_LOOP / ESCALATE.
//
// Invariant: doom loop detection is fail-closed and bounded. When sample evidence
// is insufficient, ambiguous, or unreadable, Classify defaults to VerdictUnknown and CorrectNone.
// Guard: requires at least two samples and at least TripWindows consecutive burning-flat
// windows before escalating to VerdictDoomLoop and recommending intervention.
func Classify(samples []Sample, cfg Config) Result {
	cfg = cfg.withDefaults()
	if len(samples) < cfg.MinSamples {
		return Result{
			Verdict:    VerdictUnknown,
			Correction: CorrectNone,
			Reason:     fmt.Sprintf("insufficient evidence: %d samples < MinSamples %d", len(samples), cfg.MinSamples),
		}
	}

	wins := windows(samples)
	last := wins[len(wins)-1]
	streak := trailingBurningFlatStreak(wins, cfg)

	res := Result{
		BurningFlatStreak: streak,
		EffortDelta:       last.effortDelta,
		ProgressDelta:     last.progressDelta,
		Windows:           len(wins),
	}

	switch {
	case last.progressDelta > cfg.ProgressEpsilon:
		res.Verdict = VerdictHealthy
		res.Correction = CorrectNone
		res.Reason = fmt.Sprintf("verified progress advanced (+%d) in the latest window", last.progressDelta)

	case last.effortDelta <= cfg.EffortEpsilon:
		if samples[len(samples)-1].Alive {
			res.Verdict = VerdictIdle
			res.Correction = CorrectNone
			res.Reason = "not spending effort and no progress: parked/quiet worker, alive"
		} else {
			res.Verdict = VerdictWedged
			res.Correction = CorrectNone
			res.Reason = "not spending effort, no progress, and not alive: frozen worker"
		}

	default: // burning AND flat
		switch {
		case streak >= cfg.EscalateWindows:
			res.Verdict = VerdictDoomLoop
			res.Correction = CorrectEscalate
			res.Reason = fmt.Sprintf("burning with flat verified progress for %d consecutive windows (>= escalate %d): persistent doom loop", streak, cfg.EscalateWindows)
		case streak >= cfg.TripWindows:
			res.Verdict = VerdictDoomLoop
			res.Correction = CorrectNudge
			res.Reason = fmt.Sprintf("burning with flat verified progress for %d consecutive windows (>= trip %d): doom loop confirmed", streak, cfg.TripWindows)
		default:
			res.Verdict = VerdictHealthy
			res.Correction = CorrectObserve
			res.Reason = fmt.Sprintf("burning with flat verified progress for %d window(s) (< trip %d): watching, not yet intervening", streak, cfg.TripWindows)
		}
	}
	return res
}

// windows builds the adjacent-sample transitions. A negative delta (a counter
// that went backwards, e.g. a transcript rotated) is clamped to zero so a reset
// never reads as progress and never masks a doom loop.
func windows(samples []Sample) []window {
	out := make([]window, 0, len(samples)-1)
	for i := 1; i < len(samples); i++ {
		ed := samples[i].Effort - samples[i-1].Effort
		pd := samples[i].Progress - samples[i-1].Progress
		if ed < 0 {
			ed = 0
		}
		if pd < 0 {
			pd = 0
		}
		out = append(out, window{effortDelta: ed, progressDelta: pd})
	}
	return out
}

// trailingBurningFlatStreak counts the run of burning-flat windows ending at the
// most recent one. A window breaks the streak the moment it either stops burning
// (effort delta not above epsilon) or lands progress (progress delta above
// epsilon) - so any real progress, however small, ends the doom episode.
func trailingBurningFlatStreak(wins []window, cfg Config) int {
	n := 0
	for i := len(wins) - 1; i >= 0; i-- {
		burning := wins[i].effortDelta > cfg.EffortEpsilon
		flat := wins[i].progressDelta <= cfg.ProgressEpsilon
		if burning && flat {
			n++
			continue
		}
		break
	}
	return n
}

// Interpretation returns a one-line, human-facing read of a result: what it
// means and what the caller should do next. Mirrors the dos_* servers' habit of
// never handing back a bare verdict.
func (r Result) Interpretation() string {
	switch r.Correction {
	case CorrectNudge:
		return "confirmed doom loop - deliver a soft re-anchor nudge to the session steer channel"
	case CorrectEscalate:
		return "persistent doom loop - escalate to an operator; a nudge did not recover it"
	case CorrectObserve:
		return "a burning-flat streak is building but has not tripped - record and keep watching"
	default:
		switch r.Verdict {
		case VerdictHealthy:
			return "making verified progress - leave it alone"
		case VerdictIdle:
			return "quiet but alive - not a doom loop; no action"
		case VerdictWedged:
			return "frozen - not a doom loop; route to the resume/liveness rung"
		default:
			return "insufficient evidence - take no action until more samples land"
		}
	}
}
