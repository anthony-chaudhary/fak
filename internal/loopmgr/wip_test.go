package loopmgr

import "testing"

// wip_test.go -- the flow-limit signal. These tests pin the two properties a WIP
// measurement has to have to be safe to gate a spawn on: it must count units that
// OUTLIVED their worker process, and it must never charge queued inventory.

func TestStatusWIPCountsStartedMinusEnded(t *testing.T) {
	s := Status{Loops: []LoopSnapshot{
		{LoopID: "a", Admitted: 10, Started: 7, Ended: 3}, // 4 in hand, 3 queued
		{LoopID: "b", Admitted: 4, Started: 4, Ended: 2},  // 2 in hand, 0 queued
	}}
	wip, inv := s.WIP()
	if wip != 6 {
		t.Fatalf("wip = %d, want 6 (4+2 started-and-unfinished)", wip)
	}
	if inv != 3 {
		t.Fatalf("inventory = %d, want 3 (admitted but never begun)", inv)
	}
}

// TestStatusWIPSurvivesTheDeathOfEveryWorker is the load-bearing one. Dispatch admission
// measures live worker PROCESSES; this measures units. A ledger in which every run was
// started and none ended -- the exact shape a fleet leaves behind when its sessions are
// reaped or crash -- must still report the work as in hand, because that is the work a
// new start would be piled on top of and the work a process count cannot see.
func TestStatusWIPSurvivesTheDeathOfEveryWorker(t *testing.T) {
	s := Status{Loops: []LoopSnapshot{
		{LoopID: "reaped", Admitted: 40, Started: 40, Ended: 0},
	}}
	if wip, _ := s.WIP(); wip != 40 {
		t.Fatalf("wip = %d, want 40 -- a unit does not finish just because its process died", wip)
	}
}

// TestStatusWIPNeverChargesInventory is the inventory-vs-WIP contract at the producer,
// twinned with dispatchtick's at the gate. A loop with an enormous admitted backlog and
// nothing begun owns NO work in progress.
func TestStatusWIPNeverChargesInventory(t *testing.T) {
	s := Status{Loops: []LoopSnapshot{{LoopID: "backlog", Admitted: 1300, Started: 0, Ended: 0}}}
	wip, inv := s.WIP()
	if wip != 0 {
		t.Fatalf("wip = %d, want 0 -- 1300 admitted-but-unstarted units are inventory, not WIP", wip)
	}
	if inv != 1300 {
		t.Fatalf("inventory = %d, want 1300 carried separately so a consumer cannot add it in by accident", inv)
	}
}

// TestStatusWIPClampsPerLoopOnATruncatedLedger pins the rotation guard: a loop whose
// starts were sealed into an earlier segment can show more ends than starts, and that
// negative balance must not lend headroom to a genuinely over-limit sibling.
func TestStatusWIPClampsPerLoopOnATruncatedLedger(t *testing.T) {
	s := Status{Loops: []LoopSnapshot{
		{LoopID: "rotated", Admitted: 0, Started: 2, Ended: 9}, // -7 if not clamped
		{LoopID: "hot", Admitted: 6, Started: 6, Ended: 1},     // 5 in hand
	}}
	if wip, _ := s.WIP(); wip != 5 {
		t.Fatalf("wip = %d, want 5 -- a truncated loop must clamp at 0, not offset a live sibling", wip)
	}
}

func TestStatusWIPEmptyLedgerIsZero(t *testing.T) {
	wip, inv := Status{}.WIP()
	if wip != 0 || inv != 0 {
		t.Fatalf("empty status = %d/%d, want 0/0", wip, inv)
	}
}
