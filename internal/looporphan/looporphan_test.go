package looporphan

import "testing"

// findPID returns the verdict for a pid, or a zero Verdict if absent.
func findPID(r Report, pid int) Verdict {
	for _, v := range r.Verdicts {
		if v.PID == pid {
			return v
		}
	}
	return Verdict{}
}

func TestPlan_KeepTheOneParentingLiveWork(t *testing.T) {
	// THE core scenario the leaf exists for: a lane with two detached supervisors,
	// one still parenting a live worker, one idle. Keep the live one; reap the idle
	// duplicate.
	census := []Supervisor{
		{PID: 10, Start: "s10", Lane: "auth", Parent: ParentDead, LiveWorkers: 1},
		{PID: 11, Start: "s11", Lane: "auth", Parent: ParentDead, LiveWorkers: 0},
	}
	r := Plan(census, DefaultConfig())

	if v := findPID(r, 10); v.Action != KEEP || v.Reason != ReasonKeepLiveWork {
		t.Fatalf("pid 10: want KEEP/%s, got %s/%s", ReasonKeepLiveWork, v.Action, v.Reason)
	}
	if v := findPID(r, 11); v.Action != REAP || v.Reason != ReasonDupIdle {
		t.Fatalf("pid 11: want REAP/%s, got %s/%s", ReasonDupIdle, v.Action, v.Reason)
	}
	if r.Keep != 1 || r.Reap != 1 {
		t.Fatalf("counts: want keep=1 reap=1, got keep=%d reap=%d", r.Keep, r.Reap)
	}
}

func TestPlan_TableScenarios(t *testing.T) {
	tests := []struct {
		name       string
		census     []Supervisor
		wantAction map[int]Action
		wantReason map[int]string
	}{
		{
			name:       "single orphan idle, fenced -> REAP",
			census:     []Supervisor{{PID: 1, Start: "a", Lane: "x", Parent: ParentDead}},
			wantAction: map[int]Action{1: REAP},
			wantReason: map[int]string{1: ReasonOrphanIdle},
		},
		{
			name:       "single attached idle -> KEEP",
			census:     []Supervisor{{PID: 2, Start: "b", Lane: "y", Parent: ParentAlive}},
			wantAction: map[int]Action{2: KEEP},
			wantReason: map[int]string{2: ReasonKeepAttached},
		},
		{
			name: "collision: two live one lane -> COLLISION both",
			census: []Supervisor{
				{PID: 3, Start: "c", Lane: "z", Parent: ParentDead, LiveWorkers: 1},
				{PID: 4, Start: "d", Lane: "z", Parent: ParentDead, LiveWorkers: 2},
			},
			wantAction: map[int]Action{3: COLLISION, 4: COLLISION},
			wantReason: map[int]string{3: ReasonCollisionLive, 4: ReasonCollisionLive},
		},
		{
			name: "collision plus idle dup -> two COLLISION, one REAP",
			census: []Supervisor{
				{PID: 5, Start: "e", Lane: "w", Parent: ParentDead, LiveWorkers: 1},
				{PID: 6, Start: "f", Lane: "w", Parent: ParentDead, LiveWorkers: 1},
				{PID: 7, Start: "g", Lane: "w", Parent: ParentDead, LiveWorkers: 0},
			},
			wantAction: map[int]Action{5: COLLISION, 6: COLLISION, 7: REAP},
			wantReason: map[int]string{7: ReasonDupIdle},
		},
		{
			name:       "orphan-eligible but unfenced -> UNKNOWN/NO_FENCE",
			census:     []Supervisor{{PID: 8, Start: "", Lane: "u", Parent: ParentDead}},
			wantAction: map[int]Action{8: UNKNOWN},
			wantReason: map[int]string{8: ReasonNoFence},
		},
		{
			name:       "no identity -> UNKNOWN/EVIDENCE_THIN",
			census:     []Supervisor{{PID: 9, Start: "h", Lane: "", Cmdline: "  "}},
			wantAction: map[int]Action{9: UNKNOWN},
			wantReason: map[int]string{9: ReasonEvidenceThin},
		},
		{
			name:       "parent unknown, no live work -> UNKNOWN/EVIDENCE_THIN",
			census:     []Supervisor{{PID: 12, Start: "i", Lane: "p", Parent: ParentUnknown}},
			wantAction: map[int]Action{12: UNKNOWN},
			wantReason: map[int]string{12: ReasonEvidenceThin},
		},
		{
			name: "group by cmdline when lane empty",
			census: []Supervisor{
				{PID: 13, Start: "j", Cmdline: "fak loop drive --region a", Parent: ParentDead, LiveWorkers: 1},
				{PID: 14, Start: "k", Cmdline: "fak loop drive --region a", Parent: ParentDead, LiveWorkers: 0},
			},
			wantAction: map[int]Action{13: KEEP, 14: REAP},
			wantReason: map[int]string{13: ReasonKeepLiveWork, 14: ReasonDupIdle},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := Plan(tc.census, DefaultConfig())
			for pid, want := range tc.wantAction {
				if got := findPID(r, pid).Action; got != want {
					t.Errorf("pid %d: action want %s, got %s", pid, want, got)
				}
			}
			for pid, want := range tc.wantReason {
				if got := findPID(r, pid).Reason; got != want {
					t.Errorf("pid %d: reason want %s, got %s", pid, want, got)
				}
			}
		})
	}
}

// TestPlan_NeverReapLiveWork is the load-bearing safety invariant: across a mixed
// census, no supervisor that parents live work is ever recommended for REAP.
func TestPlan_NeverReapLiveWork(t *testing.T) {
	census := []Supervisor{
		{PID: 1, Start: "a", Lane: "l1", Parent: ParentDead, LiveWorkers: 1},
		{PID: 2, Start: "b", Lane: "l1", Parent: ParentDead, LiveWorkers: 0},
		{PID: 3, Start: "c", Lane: "l2", Parent: ParentAlive, LiveWorkers: 3},
		{PID: 4, Start: "d", Lane: "l2", Parent: ParentDead, LiveWorkers: 2},
		{PID: 5, Start: "e", Lane: "l3", Parent: ParentDead, LiveWorkers: 0},
		{PID: 6, Start: "", Lane: "l4", Parent: ParentDead, LiveWorkers: 0},
	}
	r := Plan(census, DefaultConfig())
	live := map[int]bool{1: true, 3: true, 4: true}
	for _, v := range r.Verdicts {
		if v.Action == REAP && live[v.PID] {
			t.Fatalf("SAFETY VIOLATION: pid %d parents live work but was marked REAP (%s)", v.PID, v.Reason)
		}
	}
}

func TestPlan_AllowUnfencedReapOverride(t *testing.T) {
	census := []Supervisor{{PID: 1, Start: "", Lane: "x", Parent: ParentDead}}
	if v := findPID(Plan(census, Config{AllowUnfencedReap: true}), 1); v.Action != REAP || v.Reason != ReasonOrphanIdle {
		t.Fatalf("override: want REAP/%s, got %s/%s", ReasonOrphanIdle, v.Action, v.Reason)
	}
}

func TestReport_ReapPIDs(t *testing.T) {
	census := []Supervisor{
		{PID: 1, Start: "a", Lane: "x", Parent: ParentDead},                 // REAP
		{PID: 2, Start: "b", Lane: "y", Parent: ParentAlive},                // KEEP
		{PID: 3, Start: "c", Lane: "z", Parent: ParentDead, LiveWorkers: 1}, // KEEP (live)
	}
	got := Plan(census, DefaultConfig()).ReapPIDs()
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("ReapPIDs: want [1], got %v", got)
	}
}

func TestPlan_DeterministicOrder(t *testing.T) {
	census := []Supervisor{
		{PID: 30, Start: "a", Lane: "b"},
		{PID: 10, Start: "a", Lane: "a", Parent: ParentDead},
		{PID: 20, Start: "a", Lane: "a", Parent: ParentDead, LiveWorkers: 1},
	}
	r := Plan(census, DefaultConfig())
	// group "a" (pids 10,20) sorts before group "b" (pid 30); within a group, by PID.
	wantPIDs := []int{10, 20, 30}
	if len(r.Verdicts) != 3 {
		t.Fatalf("want 3 verdicts, got %d", len(r.Verdicts))
	}
	for i, want := range wantPIDs {
		if r.Verdicts[i].PID != want {
			t.Fatalf("verdict[%d]: want pid %d, got %d", i, want, r.Verdicts[i].PID)
		}
	}
}

// TestPlan_AttachedIdleInLiveLaneKept locks the HIGH-severity fix: an idle
// supervisor that shares a (fallback) lane with a live keeper must NOT be reaped
// when its own parent is alive - the lane key can collide across unrelated loops,
// so an attached idle loop is kept, and only a confirmed orphan idle is reaped.
func TestPlan_AttachedIdleInLiveLaneKept(t *testing.T) {
	census := []Supervisor{
		// the live keeper for lane "auth"
		{PID: 100, Start: "a", Lane: "auth", Parent: ParentAlive, LiveWorkers: 1},
		// idle, same lane, but its OWN parent is alive -> attached, must be KEPT
		{PID: 200, Start: "b", Lane: "auth", Parent: ParentAlive, LiveWorkers: 0},
		// idle, same lane, parent gone -> confirmed orphan duplicate, REAP
		{PID: 300, Start: "c", Lane: "auth", Parent: ParentDead, LiveWorkers: 0},
		// idle, same lane, parent unknown -> fail closed, UNKNOWN (never reap)
		{PID: 400, Start: "d", Lane: "auth", Parent: ParentUnknown, LiveWorkers: 0},
	}
	r := Plan(census, DefaultConfig())
	if v := findPID(r, 100); v.Action != KEEP || v.Reason != ReasonKeepLiveWork {
		t.Errorf("pid100 (live keeper): want KEEP/%s, got %s/%s", ReasonKeepLiveWork, v.Action, v.Reason)
	}
	if v := findPID(r, 200); v.Action != KEEP || v.Reason != ReasonKeepAttached {
		t.Errorf("pid200 (attached idle): want KEEP/%s, got %s/%s - would have been wrongly reaped", ReasonKeepAttached, v.Action, v.Reason)
	}
	if v := findPID(r, 300); v.Action != REAP || v.Reason != ReasonDupIdle {
		t.Errorf("pid300 (orphan idle dup): want REAP/%s, got %s/%s", ReasonDupIdle, v.Action, v.Reason)
	}
	if v := findPID(r, 400); v.Action != UNKNOWN {
		t.Errorf("pid400 (unknown-parent idle): want UNKNOWN (fail closed), got %s", v.Action)
	}
}

// TestPlan_TwoLiveOneAttachedIdle covers the len(live)>=2 (COLLISION) branch: the
// two live supervisors collide (operator decision), a same-lane attached idle is
// still kept, and an orphan idle is reaped.
func TestPlan_TwoLiveOneAttachedIdle(t *testing.T) {
	census := []Supervisor{
		{PID: 10, Start: "a", Lane: "web", Parent: ParentAlive, LiveWorkers: 1},
		{PID: 20, Start: "b", Lane: "web", Parent: ParentAlive, LiveWorkers: 1},
		{PID: 30, Start: "c", Lane: "web", Parent: ParentAlive, LiveWorkers: 0}, // attached idle -> KEEP
		{PID: 40, Start: "d", Lane: "web", Parent: ParentDead, LiveWorkers: 0},  // orphan idle -> REAP
	}
	r := Plan(census, DefaultConfig())
	if v := findPID(r, 10); v.Action != COLLISION {
		t.Errorf("pid10: want COLLISION, got %s", v.Action)
	}
	if v := findPID(r, 30); v.Action != KEEP || v.Reason != ReasonKeepAttached {
		t.Errorf("pid30 (attached idle): want KEEP/%s, got %s/%s", ReasonKeepAttached, v.Action, v.Reason)
	}
	if v := findPID(r, 40); v.Action != REAP {
		t.Errorf("pid40 (orphan idle): want REAP, got %s", v.Action)
	}
}
