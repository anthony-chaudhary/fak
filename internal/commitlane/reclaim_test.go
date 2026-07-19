package commitlane

import "testing"

// TestDecideIndexLockReclaim pins the evidence bar for reclaiming a stale git
// index.lock: the reapable signature is present + probe-ok + no-live-writer + stale,
// and EVERY weaker state (absent, un-probed, live writer, or too fresh) keeps the lock.
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
			name: "probe error fails closed even when stale with no listed writer",
			rep: Report{
				ProcessProbe: "error",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: true},
			},
			wantReap:   false,
			wantReason: ReclaimKeepProbeFailed,
		},
		{
			name: "probe not_run fails closed",
			rep: Report{
				ProcessProbe: "not_run",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: true},
			},
			wantReap:   false,
			wantReason: ReclaimKeepProbeFailed,
		},
		{
			name: "live writer present keeps the lock (even if stale)",
			rep: Report{
				ProcessProbe: "ok",
				IndexLock:    IndexLock{Path: path, Present: true, StaleHint: true},
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

// TestDecideIndexLockReclaimReapIsSoleFallThrough guards the safety-critical property
// that reap is reached ONLY through the full-evidence default branch: flipping any one
// guard away from its reapable value must stop the reap.
func TestDecideIndexLockReclaimReapIsSoleFallThrough(t *testing.T) {
	reapable := Report{
		ProcessProbe: "ok",
		IndexLock:    IndexLock{Path: "/g/index.lock", Present: true, StaleHint: true},
	}
	if d := DecideIndexLockReclaim(reapable); !d.Reap {
		t.Fatalf("baseline should reap, got %+v", d)
	}
	// Each mutation removes exactly one pillar of the evidence and must flip to keep.
	mutators := map[string]func(*Report){
		"not present":  func(r *Report) { r.IndexLock.Present = false },
		"probe failed": func(r *Report) { r.ProcessProbe = "error" },
		"live writer":  func(r *Report) { r.LiveWriters = []ProcessFact{{PID: 1}} },
		"not stale":    func(r *Report) { r.IndexLock.StaleHint = false },
	}
	for name, mut := range mutators {
		r := reapable
		mut(&r)
		if d := DecideIndexLockReclaim(r); d.Reap {
			t.Errorf("%s: expected keep, but decision reaped (%+v)", name, d)
		}
	}
}
