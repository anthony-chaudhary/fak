package ops

import "time"

// Liveness is the closed vocabulary for whether the ops loop is ticking (#14481).
type Liveness string

const (
	// LivenessLive means the last load-cycle row is younger than the staleness threshold.
	LivenessLive Liveness = "LIVE"
	// LivenessStale means ticks exist but the last one is older than the threshold.
	LivenessStale Liveness = "STALE"
	// LivenessNeverTicked means the load ledger has no rows at all.
	LivenessNeverTicked Liveness = "NEVER_TICKED"
)

// DefaultLivenessStaleAfter is how old the last load-cycle row may be before
// the loop is reported STALE. It is several multiples of the 60s base cadence
// plus the duty-cycle stretch headroom.
const DefaultLivenessStaleAfter = 10 * time.Minute

// LoadSummary is the pure rollup of fak-ops-load/1 rows that `fak ops status` surfaces.
type LoadSummary struct {
	Liveness           Liveness
	LastTickAt         time.Time
	LastTickAgeSeconds int64
	// SelfCPUSharePercent is sum(self_cpu_ms) over measured in-window ticks
	// divided by the wall time they span. Valid only when ShareMeasured.
	SelfCPUSharePercent float64
	ShareMeasured       bool
	Ticks               int
	Productive          int
	Idle                int
	Skipped             int
	Host                *HostLoad
}

// SummarizeLoad folds cycles into a LoadSummary over [now-window, now].
// Liveness is judged from the newest row regardless of window.
func SummarizeLoad(cycles []LoadCycle, now time.Time, window, staleAfter time.Duration) LoadSummary {
	if staleAfter <= 0 {
		staleAfter = DefaultLivenessStaleAfter
	}
	s := LoadSummary{Liveness: LivenessNeverTicked}
	if len(cycles) == 0 {
		return s
	}
	last := cycles[0]
	for _, c := range cycles {
		if c.Finished.After(last.Finished) {
			last = c
		}
	}
	s.LastTickAt = last.Finished
	age := now.Sub(last.Finished)
	if age < 0 {
		age = 0
	}
	s.LastTickAgeSeconds = int64(age / time.Second)
	s.Liveness = LivenessLive
	if age > staleAfter {
		s.Liveness = LivenessStale
	}
	s.Host = last.Host

	cutoff := now.Add(-window)
	var cpuMS, wallMS int64
	var first, end time.Time
	for _, c := range cycles {
		if c.Finished.Before(cutoff) {
			continue
		}
		s.Ticks++
		switch c.Verdict {
		case LoadProductive:
			s.Productive++
		case LoadIdle:
			s.Idle++
		case LoadSkipped:
			s.Skipped++
		}
		wallMS += c.WallMS
		if c.HaveSelfCPU {
			cpuMS += c.SelfCPUMillis
			s.ShareMeasured = true
		}
		if first.IsZero() || c.Started.Before(first) {
			first = c.Started
		}
		if c.Finished.After(end) {
			end = c.Finished
		}
	}
	if s.ShareMeasured {
		span := end.Sub(first).Milliseconds()
		if span < wallMS {
			span = wallMS
		}
		if span > 0 {
			s.SelfCPUSharePercent = float64(cpuMS) * 100 / float64(span)
		}
	}
	return s
}

// ResumeBy is when a pause force-resumes: paused_at + reactivate_when.max_pause.
// Zero when not paused or no parseable max_pause was declared.
func (s PauseState) ResumeBy() time.Time {
	if !s.Paused || s.PausedAt.IsZero() {
		return time.Time{}
	}
	d, err := time.ParseDuration(s.ReactivateWhen.MaxPause)
	if err != nil || d <= 0 {
		return time.Time{}
	}
	return s.PausedAt.Add(d)
}
