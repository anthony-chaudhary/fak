package main

// One-pass runs-directory snapshot for a dispatch tick. A single tick used to scan
// the runs directory up to five times -- liveResolutionLanes, liveResolutionScopes
// (twice: once for issue details, once per pick for the tree-collision gate),
// recentlyAttemptedIssues and cooldownIssueRows (each globbing + statting the same
// sidecars again) -- so the discovery cost grew O(N)x(scan loops) in the number of
// resolve-*.log/.witness sidecars, and every fresh scan could also read a slightly
// different filesystem than the last. scanRunsSnapshot walks the runs directory ONCE
// (one resolve-*.log glob + one resolve-*.witness glob, one stat/read per sidecar),
// captures every fact the tick needs, and serves the live-lane / live-scope /
// cooldown / attempt-budget / tree-collision views as pure projections over that
// captured state -- so the read cost is one pass regardless of how many views read
// from it, and every view sees one consistent instant (#3593).

import (
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// runsLogEntry is the captured per-resolve-worker view for one resolve-*.log sidecar,
// computed once during the scan. live == its pid is alive AND its log is not a terminal
// banner no-op (#1275/#1398); lane/tree/leaseID are read only for a live worker, matching
// the original per-loop short-circuits so the projections stay byte-identical.
type runsLogEntry struct {
	log      string
	issue    int
	hasIssue bool
	live     bool
	pid      int
	lane     string
	tree     []string
	leaseID  string
	worker   string
}

// runsSnapshot is the single per-tick scan of the runs directory. entries preserves
// sorted-log order (so slice projections match the legacy scans); latest holds the
// newest attempt mtime per issue folded over BOTH .log and .witness sidecars (the
// cooldown evidence); attempts holds the distinct <issue>-<stamp> attempt keys per
// issue over .log sidecars only (the poison-cap evidence). now is the instant every
// cooldown projection is measured against.
type runsSnapshot struct {
	now      time.Time
	entries  []runsLogEntry
	latest   map[int]time.Time
	attempts map[int]map[string]bool
}

// scanRunsSnapshot performs the one runs-directory pass. All sidecar I/O routes through
// the fsGlob/fsStat/fsReadFile seam (see dispatch_tick_livescan.go) so the pass is
// countable in tests; the projection methods below touch only the captured maps/slices
// and do zero further I/O.
func scanRunsSnapshot(runsDir string, now time.Time) *runsSnapshot {
	s := &runsSnapshot{
		now:      now,
		latest:   map[int]time.Time{},
		attempts: map[int]map[string]bool{},
	}
	for _, log := range resolveLogs(runsDir) {
		base := filepath.Base(log)
		stem := strings.TrimSuffix(log, filepath.Ext(log))

		// Poison-cap attempt key (attemptExhaustedIssues parity): .log sidecars only,
		// keyed by the <issue>-<stamp> base with a legacy .out/.err split folded to one.
		if issue, ok := issueFromResolveLog(base); ok {
			key := strings.TrimSuffix(base, ".log")
			key = strings.TrimSuffix(key, ".out")
			key = strings.TrimSuffix(key, ".err")
			if s.attempts[issue] == nil {
				s.attempts[issue] = map[string]bool{}
			}
			s.attempts[issue][key] = true
		}

		// One stat serves BOTH the cooldown mtime fold and the banner-noop size gate.
		st, statErr := fsStat(log)
		if statErr == nil {
			// Cooldown fold over the .log side (cooldownIssueRowsAt parity): key via the
			// extension-agnostic attempt regex, keep the newest mtime per issue.
			s.touchResolveAttempt(base, st.ModTime())
		}

		entry := runsLogEntry{log: log, worker: filepath.Base(stem)}
		entry.issue, entry.hasIssue = issueFromResolveLog(base)
		if pid, ok := readPID(stem + ".pid"); ok && dispatchPIDAlive(pid) {
			entry.pid = pid
			// dispatchLogIsBannerNoop fails closed to false on a stat error, so an
			// unclassifiable log stays live -- reproduce that off the stat we already have.
			if !(statErr == nil && classifyBannerNoop(st, log)) {
				entry.live = true
				entry.lane = laneFromSpawnHeader(log)
				if entry.hasIssue {
					entry.tree = readResolveLeaseTree(stem + dispatchLeaseTreeSidecarSuffix)
					entry.leaseID = readResolveLeaseID(stem, entry.lane)
				}
			}
		}
		s.entries = append(s.entries, entry)
	}
	// Witness sidecars extend the cooldown fold: a witnessed dead slot cools its issue
	// even after its .log is swept (the .witness is the durable evidence, its post-mortem
	// mtime carries the most-recent attempt touch).
	for _, wit := range resolveWitnessFiles(runsDir) {
		st, err := fsStat(wit)
		if err != nil {
			continue
		}
		s.touchResolveAttempt(filepath.Base(wit), st.ModTime())
	}
	return s
}

// touchResolveAttempt folds one resolve-attempt artifact into the cooldown map, keeping the
// NEWEST touch per issue. name is the artifact's basename -- a .log or a .witness -- and an
// artifact that names no issue is ignored. Both sides of the scan cool an issue through this
// one rule, so a witness and a log that touch the same issue cannot fold differently.
func (s *runsSnapshot) touchResolveAttempt(name string, mod time.Time) {
	issue, ok := issueFromResolveAttempt(name)
	if !ok {
		return
	}
	if prev, exists := s.latest[issue]; !exists || mod.After(prev) {
		s.latest[issue] = mod
	}
}

// liveScopes returns the live resolution scopes in sorted-log order -- the projection
// liveResolutionScopes used to compute by re-scanning the runs directory.
func (s *runsSnapshot) liveScopes() []dispatchLiveScope {
	var out []dispatchLiveScope
	for _, e := range s.entries {
		if !e.live || !e.hasIssue {
			continue
		}
		out = append(out, dispatchLiveScope{
			Issue:   e.issue,
			Lane:    e.lane,
			Tree:    e.tree,
			Log:     e.log,
			PID:     e.pid,
			Worker:  e.worker,
			LeaseID: e.leaseID,
		})
	}
	return out
}

// liveLanes returns the lanes held by a live, non-noop worker. It does NOT require the
// log's issue to parse (parity with the legacy liveResolutionLanes, which keyed off the
// spawn-header lane alone), so it can hold a lane a scope would skip.
func (s *runsSnapshot) liveLanes() map[string]bool {
	out := map[string]bool{}
	for _, e := range s.entries {
		if e.live && e.lane != "" {
			out[e.lane] = true
		}
	}
	return out
}

// liveIssueDetails maps each live issue to its first (sorted-log order) live scope.
func (s *runsSnapshot) liveIssueDetails() map[int]dispatchLiveScope {
	out := map[int]dispatchLiveScope{}
	for _, live := range s.liveScopes() {
		if _, exists := out[live.Issue]; !exists {
			out[live.Issue] = live
		}
	}
	return out
}

// liveIssues is the set of issues with a live worker.
func (s *runsSnapshot) liveIssues() map[int]bool {
	out := map[int]bool{}
	for issue := range s.liveIssueDetails() {
		out[issue] = true
	}
	return out
}

// treeCollision reports the first live worker whose in-flight tree overlaps requested.
func (s *runsSnapshot) treeCollision(requested []string) (dispatchLiveScope, bool) {
	return treeCollisionFromScopes(s.liveScopes(), requested)
}

// treeCollisionFromScopes is the pure tree-overlap check shared by the snapshot method
// and the tick's per-pick collision gate, so a pick can be checked against an already
// captured scope set instead of re-scanning the runs directory for every candidate.
func treeCollisionFromScopes(scopes []dispatchLiveScope, requested []string) (dispatchLiveScope, bool) {
	requested = dispatchTrimTree(requested)
	if len(requested) == 0 {
		return dispatchLiveScope{}, false
	}
	for _, live := range scopes {
		if len(live.Tree) == 0 {
			continue
		}
		if dispatchorder.TreesOverlap(requested, live.Tree) {
			return live, true
		}
	}
	return dispatchLiveScope{}, false
}

// cooldownRows projects the per-issue cooldown rows off the captured attempt mtimes,
// measured against the snapshot's now (cooldownIssueRowsAt parity).
func (s *runsSnapshot) cooldownRows(cooldownMin int) []dispatchCooldownRow {
	if cooldownMin <= 0 {
		return nil
	}
	cooldown := time.Duration(cooldownMin) * time.Minute
	issues := make([]int, 0, len(s.latest))
	for issue := range s.latest {
		issues = append(issues, issue)
	}
	sort.Ints(issues)
	out := make([]dispatchCooldownRow, 0, len(issues))
	for _, issue := range issues {
		last := s.latest[issue]
		if last.After(s.now) {
			last = s.now
		}
		next := last.Add(cooldown)
		remaining := int(next.Sub(s.now).Seconds())
		if remaining < 0 {
			remaining = 0
		}
		age := int(s.now.Sub(last).Seconds())
		if age < 0 {
			age = 0
		}
		out = append(out, dispatchCooldownRow{
			Issue:                    issue,
			LastAttemptUnix:          last.Unix(),
			LastAttemptAgeSeconds:    age,
			CooldownRemainingSeconds: remaining,
			NextEligibleUnix:         next.Unix(),
			Cooling:                  remaining > 0,
		})
	}
	return out
}

// cooldownRowMaps is the JSON-payload shape of cooldownRows (cooldownIssueRows parity).
func (s *runsSnapshot) cooldownRowMaps(cooldownMin int) []map[string]any {
	rows := s.cooldownRows(cooldownMin)
	out := make([]map[string]any, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Map())
	}
	return out
}

// recentlyAttempted is the set of issues still inside their cooldown window.
func (s *runsSnapshot) recentlyAttempted(cooldownMin int) map[int]bool {
	out := map[int]bool{}
	if cooldownMin <= 0 {
		return out
	}
	for _, row := range s.cooldownRows(cooldownMin) {
		if row.Cooling {
			out[row.Issue] = true
		}
	}
	return out
}

// attemptExhausted is the poison-cap set: issues whose distinct recorded attempts have
// reached budget (attemptExhaustedIssues parity; budget <= 0 disables the cap).
func (s *runsSnapshot) attemptExhausted(budget int) map[int]bool {
	out := map[int]bool{}
	if budget <= 0 {
		return out
	}
	for issue, keys := range s.attempts {
		if len(keys) >= budget {
			out[issue] = true
		}
	}
	return out
}
