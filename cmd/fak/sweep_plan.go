package main

import (
	"sort"
	"strings"
)

// sweepClass buckets a dirty path by how an automated drive-to-zero must treat it.
type sweepClass string

const (
	// classStampable: a change in a known lane — commit-by-path under a `(fak <lane>)` stamp.
	classStampable sweepClass = "stampable"
	// classNoLane: a change with no inferable lane (a root-level file) — needs a hand-chosen stamp.
	classNoLane sweepClass = "no-lane"
	// classJunk: stray scratch/log output that must be SURFACED, never committed.
	classJunk sweepClass = "junk"
)

// dirtyEntry is one `git status --porcelain` record: a path, its XY status, untracked-ness.
type dirtyEntry struct {
	Path      string `json:"path"`
	Status    string `json:"status"` // trimmed porcelain XY ("M", "A", "D", "??", ...)
	Untracked bool   `json:"untracked"`
}

// sweepEntry is a classified dirty path.
type sweepEntry struct {
	dirtyEntry
	Lane  string     `json:"lane,omitempty"`
	Class sweepClass `json:"class"`
}

// sweepGroup is the unit one commit would cover: every stampable path in a single lane.
type sweepGroup struct {
	Lane         string   `json:"lane"`
	Trailer      string   `json:"suggested_trailer"`
	Paths        []string `json:"paths"`
	Score        int      `json:"score"`                   // 0-100 apply-readiness score for this lane group
	ScoreReasons []string `json:"score_reasons,omitempty"` // why Score dropped below 100
}

// sweepPlan is the full grouped view of a dirty working tree.
type sweepPlan struct {
	TotalDirty int          `json:"total_dirty"`
	Groups     []sweepGroup `json:"groups"`
	NoLane     []sweepEntry `json:"no_lane,omitempty"`
	Junk       []sweepEntry `json:"junk,omitempty"`
}

// laneResolver maps a repo-relative path to its `(fak <lane>)` leaf, "" when none can be inferred.
type laneResolver func(path string) string

// classifyDirty buckets every dirty entry into a sweepPlan: stampable paths grouped by lane (each
// sorted, lanes sorted), plus the no-lane and junk residuals. Pure over (entries, resolver), so it
// is unit-testable with no git tree and no dos.toml.
func classifyDirty(entries []dirtyEntry, resolve laneResolver) sweepPlan {
	plan := sweepPlan{TotalDirty: len(entries)}
	byLane := map[string][]dirtyEntry{}
	for _, e := range entries {
		se := sweepEntry{dirtyEntry: e}
		switch {
		case isSweepJunk(e):
			se.Class = classJunk
			plan.Junk = append(plan.Junk, se)
		default:
			lane := resolve(e.Path)
			se.Lane = lane
			if lane == "" {
				se.Class = classNoLane
				plan.NoLane = append(plan.NoLane, se)
				continue
			}
			se.Class = classStampable
			byLane[lane] = append(byLane[lane], e)
		}
	}
	lanes := make([]string, 0, len(byLane))
	for lane := range byLane {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for _, lane := range lanes {
		entries := byLane[lane]
		sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
		paths := make([]string, len(entries))
		for i, e := range entries {
			paths[i] = e.Path
		}
		score, reasons := scoreSweepGroup(entries)
		plan.Groups = append(plan.Groups, sweepGroup{
			Lane:         lane,
			Trailer:      "(fak " + lane + ")",
			Paths:        paths,
			Score:        score,
			ScoreReasons: reasons,
		})
	}
	return plan
}

func scoreSweepGroup(entries []dirtyEntry) (int, []string) {
	score := 100
	var reasons []string

	switch {
	case len(entries) > 25:
		score -= 25
		reasons = append(reasons, "large lane group (>25 paths)")
	case len(entries) > 10:
		score -= 12
		reasons = append(reasons, "medium lane group (>10 paths)")
	}

	statuses := map[string]bool{}
	untracked, deletions := 0, 0
	for _, e := range entries {
		status := sweepStatusKind(e)
		statuses[status] = true
		if e.Untracked {
			untracked++
		}
		if strings.Contains(status, "D") {
			deletions++
		}
	}
	if len(statuses) > 1 {
		score -= 8
		reasons = append(reasons, "mixed git statuses")
	}
	if untracked > 0 {
		score -= 8
		reasons = append(reasons, "includes untracked source")
	}
	if deletions > 0 {
		score -= 10
		reasons = append(reasons, "includes deletions")
	}
	if score < 1 {
		return 1, reasons
	}
	return score, reasons
}

func sweepStatusKind(e dirtyEntry) string {
	if e.Untracked {
		return "??"
	}
	s := strings.TrimSpace(e.Status)
	if s == "" {
		return "M"
	}
	return s
}

// isSweepJunk reports whether an UNTRACKED path is stray scratch/log output an automated sweep
// must surface rather than commit. Deliberately conservative — it only flags shapes that are
// never source: a misdirected harness-scratchpad write whose separators were flattened into one
// repo-root filename, and captured per-run stdio logs. A tracked change is never junk.
func isSweepJunk(e dirtyEntry) bool {
	if !e.Untracked {
		return false
	}
	p := strings.ReplaceAll(e.Path, "\\", "/")
	lower := strings.ToLower(p)
	trimmed := strings.TrimSuffix(p, "/")
	base := trimmed[strings.LastIndexByte(trimmed, '/')+1:]
	rootLevel := !strings.Contains(trimmed, "/")
	// A misdirected scratch write: the harness scratchpad path got flattened (its separators
	// stripped) and landed as one long repo-root filename.
	if rootLevel && strings.Contains(lower, "scratchpad") && strings.Contains(lower, "temp") {
		return true
	}
	// Root coverage files are generated by local test runs; they are never source.
	if rootLevel && (base == "coverage" || base == "coverage.out" || strings.HasSuffix(base, ".coverprofile")) {
		return true
	}
	// A root path made only of Unicode private-use glyphs is a malformed artifact, not a
	// repo lane. One current failure mode is a private-use glyph directory containing an
	// accidental nested clone; surface it as junk instead of no-lane source work.
	if rootLevel && isPrivateUseOnly(base) {
		return true
	}
	// Captured per-run stdio logs left behind in an experiment dir.
	for _, suf := range []string{".run.err", ".run.out", ".run.log"} {
		if strings.HasSuffix(base, suf) {
			return true
		}
	}
	return false
}

func isPrivateUseOnly(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !isPrivateUseRune(r) {
			return false
		}
	}
	return true
}

func isPrivateUseRune(r rune) bool {
	return (r >= 0xE000 && r <= 0xF8FF) ||
		(r >= 0xF0000 && r <= 0xFFFFD) ||
		(r >= 0x100000 && r <= 0x10FFFD)
}

// intersectPaths returns the canonical (git-form) paths in have that the want set names,
// normalizing separators and a leading "./" on both sides so a Windows --path still matches.
func intersectPaths(have, want []string) []string {
	wantSet := map[string]bool{}
	for _, p := range want {
		wantSet[normSweepPath(p)] = true
	}
	var out []string
	for _, p := range have {
		if wantSet[normSweepPath(p)] {
			out = append(out, p)
		}
	}
	return out
}

func normSweepPath(p string) string {
	return strings.TrimPrefix(strings.ReplaceAll(p, "\\", "/"), "./")
}

// parsePorcelainZ parses NUL-terminated `git status --porcelain=v1 -z --no-renames` output. Each
// record is "XY PATH" (XY at [0:2], a space at [2], the path from [3:]); the trailing empty field
// after the final NUL is skipped.
func parsePorcelainZ(out string) []dirtyEntry {
	var entries []dirtyEntry
	for _, rec := range strings.Split(out, "\x00") {
		if len(rec) < 4 {
			continue
		}
		xy := rec[:2]
		entries = append(entries, dirtyEntry{
			Path:      rec[3:],
			Status:    strings.TrimSpace(xy),
			Untracked: xy == "??",
		})
	}
	return entries
}

func stampableCount(plan sweepPlan) int {
	n := 0
	for _, g := range plan.Groups {
		n += len(g.Paths)
	}
	return n
}
