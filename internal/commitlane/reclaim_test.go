package commitlane

import "testing"

// TestDecideIndexLockReclaim pins the evidence bar for reclaiming a stale git
// index.lock: staleness ALONE reaps (index.lock has no owner pid, so a frozen mtime
// refutes a holder and neither an unrelated by-name writer nor a failed probe vetoes it,
// matching safecommit's age-only reap), while a FRESH lock is kept whenever it is
// un-probed or has a live writer, and an absent lock is nothing to reclaim.
func TestDecideIndexLockReclaim(t *testing.T) {
	const path = "/repo/.git/index.lock"
	writer := []ProcessFact{{PID: 4242, Match: "git_writer", Role: "writer"}}

	cases := []struct {
		name       string
		rep        Report
		wantReap   bool
		wantReason IndexLockReclaimReason
	}{
		{
			name:       "absent lock — nothing to reclaim",
			rep:        Report{ProcessProbe: "ok", IndexLock: IndexLock{Path: path, Present: false}},
			wantReap:   false,
			wantReason: ReclaimKeepAbsent,
		},
		{
			name: "probe error on a FRESH lock fails closed (no live-writer proof for a young lock)",
			rep: Report{
				ProcessProbe: "error",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: false},
			},
			wantReap:   false,
			wantReason: ReclaimKeepProbeFailed,
		},
		{
			name: "STALE lock reaps even with a probe error — staleness bypasses the probe, matching safecommit's age-only reap",
			rep: Report{
				ProcessProbe: "error",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: true},
			},
			wantReap:   true,
			wantReason: ReclaimReapStale,
		},
		{
			name: "probe not_run on a FRESH lock fails closed",
			rep: Report{
				ProcessProbe: "not_run",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: false},
			},
			wantReap:   false,
			wantReason: ReclaimKeepProbeFailed,
		},
		{
			name: "STALE lock reaps regardless of an unrelated by-name live writer (index.lock has no owner pid)",
			rep: Report{
				ProcessProbe: "ok",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: true},
				LiveWriters:  writer,
			},
			wantReap:   true,
			wantReason: ReclaimReapStale,
		},
		{
			name: "FRESH lock WITH a live writer is kept (writer may be mid-write)",
			rep: Report{
				ProcessProbe: "ok",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: false},
				LiveWriters:  writer,
			},
			wantReap:   false,
			wantReason: ReclaimKeepLiveWriter,
		},
		{
			name: "fresh lock with no writer is kept until it ages past the grace window",
			rep: Report{
				ProcessProbe: "ok",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: false},
			},
			wantReap:   false,
			wantReason: ReclaimKeepFresh,
		},
		{
			name: "stale + probe-ok + no live writer is the orphaned-lock signature — reap",
			rep: Report{
				ProcessProbe: "ok",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: true},
			},
			wantReap:   true,
			wantReason: ReclaimReapStale,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DecideIndexLockReclaim(tc.rep)
			if got.Reap != tc.wantReap {
				t.Errorf("Reap = %v, want %v (reason %q)", got.Reap, tc.wantReap, got.Reason)
			}
			if got.Reason != tc.wantReason {
				t.Errorf("Reason = %q, want %q", got.Reason, tc.wantReason)
			}
			if got.Path != path {
				t.Errorf("Path = %q, want %q", got.Path, path)
			}
		})
	}
}

// TestDecideIndexLockReclaimReapPillarsArePresenceAndStaleness guards the safety-critical
// property that a reap requires EXACTLY the two index.lock pillars — presence and
// staleness — and removing either flips to keep. It also pins the OTHER half of #5335
// item 3: an unrelated by-name live writer or a failed probe must NOT flip a stale reap
// to keep, because index.lock carries no owner pid so neither refutes an orphaned lock.
func TestDecideIndexLockReclaimReapPillarsArePresenceAndStaleness(t *testing.T) {
	reapable := Report{
		ProcessProbe: "ok",
		IndexLock:    IndexLock{Path: "/g/index.lock", Present: true, StaleHint: true},
	}
	if d := DecideIndexLockReclaim(reapable); !d.Reap {
		t.Fatalf("baseline should reap, got %+v", d)
	}
	// Removing either reap pillar (presence, staleness) must flip to keep.
	pillarRemovers := map[string]func(*Report){
		"not present": func(r *Report) { r.IndexLock.Present = false },
		"not stale":   func(r *Report) { r.IndexLock.StaleHint = false },
	}
	for name, mut := range pillarRemovers {
		r := reapable
		mut(&r)
		if d := DecideIndexLockReclaim(r); d.Reap {
			t.Errorf("%s: expected keep, but decision reaped (%+v)", name, d)
		}
	}
	// A by-name live writer or a failed probe is NOT a pillar: a STALE lock still reaps,
	// matching the age-only reap safecommit performs inside every fak commit.
	nonPillars := map[string]func(*Report){
		"by-name live writer": func(r *Report) { r.LiveWriters = []ProcessFact{{PID: 1}} },
		"probe failed":        func(r *Report) { r.ProcessProbe = "error" },
	}
	for name, mut := range nonPillars {
		r := reapable
		mut(&r)
		if d := DecideIndexLockReclaim(r); !d.Reap {
			t.Errorf("%s: a stale lock must still reap, but decision kept (%+v)", name, d)
		}
	}
}
