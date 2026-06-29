package nightrun

import (
	"fmt"
	"sort"
	"time"
)

// Scored is one Task with the selector's verdict: whether the box can run it,
// its blended priority score, the freshness facts that fed the score, and a
// human reason. plan renders the whole ranked slice; next() returns the top
// feasible one.
type Scored struct {
	Task          Task    `json:"task"`
	Feasible      bool    `json:"feasible"`
	Reason        string  `json:"reason"`
	Score         float64 `json:"score"`
	LastCollected string  `json:"last_collected,omitempty"` // YYYY-MM-DD or "" (never)
	AgeDays       float64 `json:"age_days"`                 // -1 when never collected on this box
	Novelty       float64 `json:"novelty"`                  // component scores, for transparency
	ValueWeight   float64 `json:"value_weight"`
	Staleness     float64 `json:"staleness"`
	// Saturated is true when this feasible, auto-runnable datum has already been
	// collected on this box AND has not aged past its re-check window (Staleness==0)
	// — i.e. re-running it tonight would just re-measure a settled number, gathering
	// no new information. It is the per-task input to the run loop's SATURATED stop
	// verdict: when EVERY feasible task is saturated, the only genuinely-new data the
	// box could gather is blocked on a capability it does not have yet, so the loop
	// should back off rather than re-fire a fresh measurement. A never-collected, an
	// overdue/aging, a Manual, or an infeasible task is NOT saturated.
	Saturated bool `json:"saturated,omitempty"`
}

// selector weights blend the three signals that decide collection priority into a
// single weighted SUM (they total 1.0). Novelty is the largest term — a first-ever
// datum on this box is the most valuable thing to collect — but it is a blend, NOT
// a strict tier: a long-stale high-value collected datum can edge ahead of a
// never-collected low-value one. The Value class and staleness reorder within that
// blend. Loosely mirrors tools/bench_plan.py's coverage-leaning, value-aware
// scoring, resolved for the LOCAL box.
const (
	wNovelty   = 0.45
	wValue     = 0.35
	wStaleness = 0.20
)

// Rank scores every Task against the box's Capabilities and the collection
// ledger at the injected now, returning them sorted feasible-first, then by
// score descending, then by Value rank, then by id — a total, deterministic
// order (a fixed now + caps + ledger yields byte-identical output, so a test
// pins it). Infeasible Tasks are kept (so plan can show WHY the box can't run
// them) but always sort after every feasible Task.
func Rank(tasks []Task, caps Capabilities, ledger []CollectRow, now time.Time) []Scored {
	out := make([]Scored, 0, len(tasks))
	for _, t := range tasks {
		out = append(out, score(t, caps, ledger, now))
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Feasible != b.Feasible {
			return a.Feasible // feasible first
		}
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if a.Task.Value.rank() != b.Task.Value.rank() {
			return a.Task.Value.rank() > b.Task.Value.rank()
		}
		return a.Task.ID < b.Task.ID
	})
	return out
}

// Next returns the single highest-priority FEASIBLE Scored task, and whether one
// exists. It is the literal answer to "what should I collect next on this box."
func Next(ranked []Scored) (Scored, bool) {
	for _, s := range ranked {
		if s.Feasible {
			return s, true
		}
	}
	return Scored{}, false
}

// score computes one Task's verdict. Novelty owns the never-collected case;
// staleness owns the re-measure case; the Value weight is always blended so a
// frontier datum outranks a smoke check at equal freshness.
func score(t Task, caps Capabilities, ledger []CollectRow, now time.Time) Scored {
	s := Scored{Task: t, ValueWeight: t.Value.weight(), AgeDays: -1}

	feasible, why := caps.Satisfies(t)
	s.Feasible = feasible

	last, ok := lastCollected(ledger, t.ID, caps.Box)
	if ok {
		s.LastCollected = last.Date
		if age, valid := ageDays(now, last.GeneratedAt, last.Date); valid {
			s.AgeDays = age
		}
	}

	// Novelty: a datum never collected on this box is maximally novel; once
	// collected it is no longer novel (staleness then governs re-collection).
	if !ok {
		s.Novelty = 1.0
	}
	// Staleness: only meaningful for an already-collected datum — how far past its
	// re-check interval it has drifted, clamped to [0,1].
	if ok && s.AgeDays >= 0 {
		s.Staleness = clamp01(s.AgeDays / float64(t.recheckDays()))
	}

	s.Score = wNovelty*s.Novelty + wValue*s.ValueWeight + wStaleness*s.Staleness
	// Saturated: a feasible, auto-runnable datum that is already collected on this box
	// and still fresh (Staleness==0) carries no new information if re-run tonight. A
	// Manual recipe is never an auto-collectable datum, so it is never "saturated"
	// (the loop skips it for a different reason); an infeasible or never-collected or
	// aging task is not saturated either.
	if s.Feasible && s.Task.autoRunnable() && ok && s.Staleness == 0 {
		s.Saturated = true
	}
	s.Reason = reasonFor(s, why)
	return s
}

// reasonFor renders the one-line human explanation for a Scored task.
func reasonFor(s Scored, infeasibleWhy string) string {
	if !s.Feasible {
		return "not feasible here — " + infeasibleWhy
	}
	// A Manual task's requirements are met, but its Run is an operator recipe the
	// unattended loop SKIPS (it needs a credential/GPU/browser the prober cannot
	// express). Say so up front so plan/next never present a skip-only task as the
	// next datum an --apply sweep will collect — the loop records OutcomeSkipped.
	if s.Task.Manual {
		return "operator recipe — run by hand; the unattended `run --apply` loop skips it (needs a setup the prober can't gate)"
	}
	switch {
	case s.LastCollected == "":
		return fmt.Sprintf("never collected on this box — a first-ever %s datum", s.Task.Value)
	case s.Staleness >= 1.0:
		return fmt.Sprintf("last collected %s (%.1fd ago, re-check %dd) — overdue, re-measure to catch drift",
			s.LastCollected, s.AgeDays, s.Task.recheckDays())
	case s.Staleness > 0:
		return fmt.Sprintf("last collected %s (%.1fd ago, re-check %dd) — aging toward a re-measure",
			s.LastCollected, s.AgeDays, s.Task.recheckDays())
	default:
		return fmt.Sprintf("last collected %s (%.1fd ago) — fresh, low priority", s.LastCollected, s.AgeDays)
	}
}

// ageDays returns the age in days of a collection, preferring the precise
// generated_at timestamp and falling back to the date. Returns valid=false when
// neither parses.
func ageDays(now time.Time, generatedAt, date string) (float64, bool) {
	if t, err := time.Parse(time.RFC3339, generatedAt); err == nil {
		return now.UTC().Sub(t.UTC()).Hours() / 24, true
	}
	if t, err := time.Parse("2006-01-02", date); err == nil {
		return now.UTC().Sub(t.UTC()).Hours() / 24, true
	}
	return -1, false
}

func clamp01(f float64) float64 {
	if f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}
