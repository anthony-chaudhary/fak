package stallscan

// growth.go — the TRAJECTORY axis for stallscan. Classify (single-sample) can
// only see a process that is ALREADY huge: the absolute HandleLeakProc (10k) /
// ThreadLeakProc (500) lines fire on a snapshot. But a leak is a slope, not a
// level — the question an operator actually wants answered is "is anything
// CLIMBING right now?", and a process climbing toward exhaustion is a leak while
// still well under the absolute line. That early-warning is exactly what the
// windows-handles research note (2026-07-08, item 3) asked for: "threshold on
// GROWTH, not absolute count … a single process passing ~10k and STILL CLIMBING
// is the signal worth alerting on."
//
// ClassifyWithBaseline adds that axis without disturbing Classify: it runs the
// full single-sample classification on the current sample, then overlays growth
// attribution computed from a baseline sample (the counts a caller captured for
// the same processes earlier — the --watch loop feeds the count each process had
// when first observed this session, so growth measures accretion-since-first-seen
// and is robust to per-interval noise). Growth keeps the same WARNING discipline
// as the absolute axes: it raises a calm box to elevated and names the culprit,
// but NEVER fabricates a stall and never downgrades a real one.

import "fmt"

// ClassifyWithBaseline is Classify plus cross-sample GROWTH detection. baseline
// carries earlier per-process handle/thread counts (matched to cur by PID+name);
// cur is the current sample. It is pure: same inputs, same output, no I/O, and it
// mutates neither sample's slices.
//
// A process is a growth suspect when its count climbed by at/above the axis's
// GrowthDelta AND its current count is at/above the axis's GrowthFloor. The floor
// sits below the absolute leak line, so growth warns EARLIER than the absolute
// axis; a high-but-stable process (climb 0) is never flagged by growth (only by
// the absolute axis), cleanly dividing "high" from "climbing".
func ClassifyWithBaseline(baseline, cur Sample, t Thresholds) Verdict {
	v := Classify(cur, t)

	hg, hgDelta, hasHG := worstHandleGrowth(baseline.TopHandles, cur.TopHandles, t.HandleGrowthDelta, t.HandleGrowthFloor)
	if hasHG {
		v.HandleGrowthProcess, v.HandleGrowthPID = hg.Name, hg.PID
		v.HandleGrowthCount, v.HandleGrowthDelta = hg.Handles, hgDelta
		v.Reasons = append(v.Reasons, fmt.Sprintf("%s (pid %d) handle count climbed +%d to %d since first seen (>= +%d growth, floor %d — leak trajectory)",
			hg.Name, hg.PID, hgDelta, hg.Handles, t.HandleGrowthDelta, t.HandleGrowthFloor))
	}

	tg, tgDelta, hasTG := worstThreadGrowth(baseline.TopThreads, cur.TopThreads, t.ThreadGrowthDelta, t.ThreadGrowthFloor)
	if hasTG {
		v.ThreadGrowthProcess, v.ThreadGrowthPID = tg.Name, tg.PID
		v.ThreadGrowthCount, v.ThreadGrowthDelta = tg.Threads, tgDelta
		v.Reasons = append(v.Reasons, fmt.Sprintf("%s (pid %d) thread count climbed +%d to %d since first seen (>= +%d growth, floor %d — thread-leak trajectory)",
			tg.Name, tg.PID, tgDelta, tg.Threads, t.ThreadGrowthDelta, t.ThreadGrowthFloor))
	}

	// Growth is a WARNING, mirroring the absolute axes: it raises a calm box to
	// elevated and owns the cause if nothing else claimed it (handle pressure wins
	// the label when both grow, as it is closer to pool exhaustion). It never sets
	// a stall and never touches an already-stall/elevated verdict's Level/Cause.
	if v.Level == LevelCalm && (hasHG || hasTG) {
		v.Level = LevelElevated
		if v.Cause == CauseNone {
			if hasHG {
				v.Cause = CauseHandleLeak
			} else {
				v.Cause = CauseThreadLeak
			}
		}
	}
	return v
}

// worstHandleGrowth returns the process whose handle count climbed the most
// between baseline and cur, among those matched by PID+name whose climb is
// at/above delta AND whose current count is at/above floor. Returns the current
// entry, the climb, and ok=false if none qualify (or delta<=0). Pure and
// non-mutating. Matching requires the name to agree so a reused PID (a different
// process that inherited the number) never reads as growth.
func worstHandleGrowth(baseline, cur []ProcHandles, delta, floor int) (ProcHandles, int, bool) {
	if delta <= 0 {
		return ProcHandles{}, 0, false
	}
	base := make(map[int]ProcHandles, len(baseline))
	for _, p := range baseline {
		base[p.PID] = p
	}
	worst := ProcHandles{}
	worstClimb := 0
	found := false
	for _, p := range cur {
		if p.Handles < floor {
			continue
		}
		b, ok := base[p.PID]
		if !ok || b.Name != p.Name {
			continue // first sight or PID reuse: no trajectory to measure
		}
		climb := p.Handles - b.Handles
		if climb < delta {
			continue
		}
		if !found || climb > worstClimb || (climb == worstClimb && p.PID < worst.PID) {
			worst, worstClimb, found = p, climb, true
		}
	}
	return worst, worstClimb, found
}

// worstThreadGrowth mirrors worstHandleGrowth for the thread census.
func worstThreadGrowth(baseline, cur []ProcThreads, delta, floor int) (ProcThreads, int, bool) {
	if delta <= 0 {
		return ProcThreads{}, 0, false
	}
	base := make(map[int]ProcThreads, len(baseline))
	for _, p := range baseline {
		base[p.PID] = p
	}
	worst := ProcThreads{}
	worstClimb := 0
	found := false
	for _, p := range cur {
		if p.Threads < floor {
			continue
		}
		b, ok := base[p.PID]
		if !ok || b.Name != p.Name {
			continue
		}
		climb := p.Threads - b.Threads
		if climb < delta {
			continue
		}
		if !found || climb > worstClimb || (climb == worstClimb && p.PID < worst.PID) {
			worst, worstClimb, found = p, climb, true
		}
	}
	return worst, worstClimb, found
}
