package scoreboard

import (
	"fmt"
	"math"
)

const WorktreeABSchema = "fak-worktree-ab/1"

type WorktreeABArm struct {
	Name            string  `json:"name"`
	Worktree        bool    `json:"worktree"`
	Resolved        int     `json:"resolved"`
	DurationSeconds float64 `json:"duration_seconds"`
	PoisonIncidents int     `json:"poison_incidents"`
	PeakConcurrency int     `json:"peak_concurrency"`
	WaveID          string  `json:"wave_id"`
	HostID          string  `json:"host_id,omitempty"`
}

func (a WorktreeABArm) IssuesPerHour() float64 {
	if a.DurationSeconds <= 0 {
		return 0
	}
	return float64(a.Resolved) * 3600 / a.DurationSeconds
}

type WorktreeABReport struct {
	Schema   string        `json:"schema"`
	Baseline WorktreeABArm `json:"baseline"`
	Isolated WorktreeABArm `json:"isolated"`
	Verdict  string        `json:"verdict"`
}

func FoldWorktreeAB(baseline, isolated WorktreeABArm) WorktreeABReport {
	baseline.Name, baseline.Worktree = "baseline", false
	isolated.Name, isolated.Worktree = "isolated", true
	verdict := "NOT_PROVEN"
	if baseline.DurationSeconds > 0 && isolated.DurationSeconds > 0 && isolated.PoisonIncidents == 0 {
		verdict = "ISOLATION_POISON_FREE"
	}
	return WorktreeABReport{Schema: WorktreeABSchema, Baseline: baseline, Isolated: isolated, Verdict: verdict}
}

func WorktreeABUpdate(r WorktreeABReport) Update {
	line := func(a WorktreeABArm) string {
		return fmt.Sprintf("%s: %.2f issues/h, %d poison, %.1fs, peak %d", a.Name, a.IssuesPerHour(), a.PoisonIncidents, a.DurationSeconds, a.PeakConcurrency)
	}
	return Update{Title: "Dispatch worktree A/B", Verdict: r.Verdict, Lines: []string{line(r.Baseline), line(r.Isolated)}}
}

func WorktreeABEquivalentWave(a, b WorktreeABArm) bool {
	return a.WaveID != "" && a.WaveID == b.WaveID && a.Resolved == b.Resolved && a.Resolved > 0 &&
		(a.HostID == "" || b.HostID == "" || a.HostID == b.HostID) && !math.IsNaN(a.DurationSeconds) && !math.IsNaN(b.DurationSeconds)
}
