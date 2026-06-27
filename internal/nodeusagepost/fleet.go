package nodeusagepost

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// ParseSnapshot decodes the JSON `fak lab status --json` emits into a fleet.Snapshot.
// It is the inverse of the render path: the CLI reads a snapshot file/stdin and folds
// it into a node-usage card, so the same pure fleet.Snapshot shape is the contract
// between `fak lab status` and `fak nodeusage post`.
func ParseSnapshot(raw []byte) (fleet.Snapshot, error) {
	var s fleet.Snapshot
	if err := json.Unmarshal(raw, &s); err != nil {
		return fleet.Snapshot{}, fmt.Errorf("parse fleet snapshot: %w", err)
	}
	return s, nil
}

// FromSnapshot folds a fleet.Snapshot (the node-usage signal) into a scoreboard.Update
// — the one card shape every Slack feeder posts. It carries the readiness score, the
// reachable fraction, the per-state and per-class node counts, and the top "what needs
// me now" attention item, so #node-usage scans the current compute-node usage at a
// glance.
//
// The verdict is OK unless the snapshot has a `crit` attention item (a box down or
// unreachable) — a degraded fleet is the operator's real node-usage problem, so it
// reads ACTION. The grade is a coarse map of the 0-100 readiness score so the channel
// colors consistently with the other feeders.
func FromSnapshot(snap fleet.Snapshot, source string) scoreboard.Update {
	up := scoreboard.Update{
		Title:   "node usage",
		Grade:   gradeOf(snap.Score),
		Score:   fmt.Sprintf("%d/%d reachable", snap.Reachable, snap.Total),
		Verdict: verdictOf(snap),
		Detail:  detailOf(snap),
		Lines:   linesOf(snap),
		Source:  source,
	}
	return up
}

// gradeOf maps the 0-100 fleet readiness score onto an A-F grade — the same coarse
// banding the capacity feed uses, so a node-usage card and a capacity card read the
// same way. An empty fleet scores 0 -> F (nothing ready).
func gradeOf(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 75:
		return "B"
	case score >= 50:
		return "C"
	case score >= 25:
		return "D"
	default:
		return "F"
	}
}

// verdictOf is ACTION when any attention item is crit (a box down or unreachable),
// else OK. A crit item is the only node-usage state worth pulling an operator in; a
// version-skew or stale warn is informational and stays OK.
func verdictOf(snap fleet.Snapshot) string {
	for _, it := range snap.Attention {
		if it.Level == "crit" {
			return "ACTION"
		}
	}
	return "OK"
}

// detailOf is the one-line headline: the top attention item's title (crit first, then
// warn, then the single healthy line), so the card leads with the thing that matters.
func detailOf(snap fleet.Snapshot) string {
	if len(snap.Attention) > 0 {
		return snap.Attention[0].Title
	}
	if snap.Total == 0 {
		return "no nodes in the roster"
	}
	return ""
}

// linesOf renders the at-a-glance node-usage breakdown: per-state node counts (live /
// idle / draining / down / unknown), then per-class counts, then the readiness score.
// States are listed in a fixed operational order so the line is stable across runs.
func linesOf(snap fleet.Snapshot) []string {
	var lines []string
	for _, st := range []fleet.State{
		fleet.StateLive, fleet.StateIdle, fleet.StateDraining, fleet.StateDown, fleet.StateUnknown,
	} {
		if n := snap.ByState[st]; n > 0 {
			lines = append(lines, fmt.Sprintf("%s: %d", st, n))
		}
	}
	// Per-class counts come pre-sorted (desc count, then key) from the fold; render as
	// class=N so node usage is visible per hardware class.
	cls := append([]string{}, classLines(snap)...)
	sort.Strings(cls)
	lines = append(lines, cls...)
	lines = append(lines, fmt.Sprintf("readiness: %d", snap.Score))
	return lines
}

// classLines renders the per-class node counts as "class=N" entries. A fleet.Snapshot
// already orders ByClass deterministically; we re-sort the rendered strings in linesOf
// for a stable, scannable line independent of count ties.
func classLines(snap fleet.Snapshot) []string {
	out := make([]string, 0, len(snap.ByClass))
	for _, c := range snap.ByClass {
		out = append(out, fmt.Sprintf("%s=%d", c.Key, c.Count))
	}
	return out
}
