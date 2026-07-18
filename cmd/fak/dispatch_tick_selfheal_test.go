package main

import (
	"reflect"
	"testing"
)

// Witness for #3109: dispatch preflight is otherwise refuse-only on unattributed_live -- it
// counts orphaned worker PIDs (a botched teardown's descendant still carrying the dispatch
// marker but holding NO seat lease) as pool depletion and wedges dispatch until a separately
// scheduled janitor clears them. The self-heal (dispatch_tick_preflight.go / dispatch_tick.go)
// surfaces those exact PIDs as a janitor worklist and tree-reaps them so the pool recovers on
// the next tick. These tests pin the three DoD clauses: (1) the worklist names exactly the
// unattributed marker-orphans and none of the leased/unrelated PIDs; (2) the reap routes every
// PID through the injectable tree-kill seam (procguard.KillPID), not a bare kill; (3) a
// follow-up tick, with the reaped PIDs gone from the worker set, shows unattributed_live back
// at zero -- the seat pool has self-healed without an operator or the scheduled janitor.

func TestDispatchUnattributedWorklistNamesOnlyMarkerOrphansWithNoLease(t *testing.T) {
	// The worker set as preflight COUNTS it: marker-matched PIDs only. dispatchCmdlineWorkerPIDs
	// is the real counting predicate -- feed it a marker orphan plus an unrelated (non-marker)
	// process to prove the unrelated one never even enters the worker set, let alone the worklist.
	rows := []dispatchCodexProcessRow{
		{PID: 202, Name: "claude.exe", Cmdline: `claude -p "resolve GitHub issue #3109 ..."`},
		{PID: 303, Name: "pwsh.exe", Cmdline: "dos-dispatch-loop --workspace ."},
		{PID: 404, Name: "chrome.exe", Cmdline: "chrome --some-unrelated-flag"},
	}
	workerPIDs := dispatchCmdlineWorkerPIDs("", rows)
	if workerPIDs[404] {
		t.Fatalf("unrelated non-marker PID 404 must never be counted as a worker: %v", workerPIDs)
	}
	// 101 is a marker worker too, but it holds a LIVE lease -- attributed, not an orphan.
	workerPIDs[101] = true
	leasedPIDs := map[int]bool{101: true}

	got := dispatchUnattributedWorklist(workerPIDs, leasedPIDs)
	want := []int{202, 303}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("worklist = %v, want exactly the marker orphans with no lease %v (leased 101 + unrelated 404 excluded)", got, want)
	}
}

func TestDispatchUnattributedWorklistSkipsNonPositivePIDs(t *testing.T) {
	got := dispatchUnattributedWorklist(map[int]bool{0: true, -1: true, 202: true}, map[int]bool{})
	if !reflect.DeepEqual(got, []int{202}) {
		t.Fatalf("worklist = %v, want only the positive orphan PID [202]", got)
	}
}

func TestDispatchReapWorklistRoutesEveryPIDThroughTreeKillSeam(t *testing.T) {
	orig := dispatchReapPID
	t.Cleanup(func() { dispatchReapPID = orig })

	var reaped []int
	dispatchReapPID = func(pid int) (bool, string) {
		reaped = append(reaped, pid)
		return true, ""
	}

	worklist := []int{202, 303}
	outcomes := dispatchReapWorklist(worklist)

	// Every worklist PID went through the injectable seam -- which defaults to procguard.KillPID,
	// a process-TREE kill. A bare kill would bypass this seam and leave the orphan's own
	// descendants behind to re-poison the count, so routing THROUGH the seam is the assertion.
	if !reflect.DeepEqual(reaped, worklist) {
		t.Fatalf("reaped via seam = %v, want every worklist PID %v tree-killed", reaped, worklist)
	}
	if len(outcomes) != len(worklist) {
		t.Fatalf("outcomes = %v, want one per worklist PID", outcomes)
	}
	for i, oc := range outcomes {
		if oc.PID != worklist[i] || !oc.OK {
			t.Fatalf("outcome[%d] = %+v, want PID %d reaped OK", i, oc, worklist[i])
		}
	}
}

func TestDispatchReapWorklistSkipsNonPositivePIDs(t *testing.T) {
	orig := dispatchReapPID
	t.Cleanup(func() { dispatchReapPID = orig })
	var reaped []int
	dispatchReapPID = func(pid int) (bool, string) {
		reaped = append(reaped, pid)
		return true, ""
	}
	dispatchReapWorklist([]int{0, -5, 202})
	if !reflect.DeepEqual(reaped, []int{202}) {
		t.Fatalf("reaped = %v, want the reaper to skip non-positive PIDs and kill only [202]", reaped)
	}
}

func TestDispatchSelfHealPoolRecoversOnNextTickAfterReap(t *testing.T) {
	orig := dispatchReapPID
	t.Cleanup(func() { dispatchReapPID = orig })
	killed := map[int]bool{}
	dispatchReapPID = func(pid int) (bool, string) {
		killed[pid] = true
		return true, ""
	}

	// Tick 1: one leased worker (101) plus two orphaned descendants (202, 303) that carry the
	// dispatch marker but hold no lease -- preflight counts them as unattributed_live=2 and,
	// refuse-only, would wedge here. The self-heal surfaces the exact orphan worklist.
	leased := map[int]bool{101: true}
	tick1Workers := map[int]bool{101: true, 202: true, 303: true}
	worklist := dispatchUnattributedWorklist(tick1Workers, leased)
	if !reflect.DeepEqual(worklist, []int{202, 303}) {
		t.Fatalf("tick-1 worklist = %v, want the two orphans [202 303]", worklist)
	}

	// Reap them via the tree-kill seam (refuse THIS tick, kill the poison for the next).
	dispatchReapWorklist(worklist)
	if !killed[202] || !killed[303] {
		t.Fatalf("reap did not tree-kill both orphans; killed=%v", killed)
	}

	// Tick 2: the reaped orphans are gone from the host worker set; only the leased worker
	// remains, so unattributed_live is back at zero and the seat pool has self-healed --
	// dispatch admits again instead of staying wedged until a scheduled janitor runs.
	tick2Workers := map[int]bool{}
	for pid := range tick1Workers {
		if !killed[pid] {
			tick2Workers[pid] = true
		}
	}
	if recovered := dispatchUnattributedWorklist(tick2Workers, leased); len(recovered) != 0 {
		t.Fatalf("tick-2 worklist = %v, want empty -- pool should have recovered after the reap", recovered)
	}
}
