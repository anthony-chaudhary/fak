package agent

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// Test 1: Verify ClampWaitTimeout clamps requested timeouts under 60,000ms to the 60s minimum floor.
func TestSubagentSupervisor_ClampWaitTimeout(t *testing.T) {
	cases := []struct {
		name     string
		input    int
		expected int
	}{
		{"under 60s: 1000ms", 1000, MinSubagentTimeoutMs},
		{"under 60s: 10000ms", 10000, MinSubagentTimeoutMs},
		{"zero ms", 0, MinSubagentTimeoutMs},
		{"negative ms", -500, MinSubagentTimeoutMs},
		{"under 60s: 59999ms", 59999, MinSubagentTimeoutMs},
		{"exact floor: 60000ms", 60000, MinSubagentTimeoutMs},
		{"above floor: 120000ms", 120000, 120000},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClampWaitTimeout(tc.input)
			if got != tc.expected {
				t.Fatalf("ClampWaitTimeout(%d) = %d; want %d", tc.input, got, tc.expected)
			}
		})
	}
}

// Test 2: Spawn multiple subagents without calling explicit close, verify AutoReap / automatic thread
// reaping reclaims finished subagents and prevents thread pool exhaustion.
func TestSubagentSupervisor_AutoReap_ThreadExhaustion(t *testing.T) {
	sup := NewSubagentSupervisor(2)

	// Spawn up to capacity (2 threads)
	t1, err := sup.Spawn("parent-sess", "sub-1", nil)
	if err != nil || t1 == nil {
		t.Fatalf("Spawn(sub-1) failed: %v", err)
	}
	t2, err := sup.Spawn("parent-sess", "sub-2", nil)
	if err != nil || t2 == nil {
		t.Fatalf("Spawn(sub-2) failed: %v", err)
	}

	if sup.ActiveCount() != 2 {
		t.Fatalf("ActiveCount() = %d; want 2", sup.ActiveCount())
	}

	// Saturated: Attempting to spawn 3rd should fail
	_, err = sup.Spawn("parent-sess", "sub-3", nil)
	if err == nil {
		t.Fatalf("expected error on thread limit saturation, got nil")
	}
	expectedErr := "collab spawn failed: agent thread limit reached"
	if !strings.Contains(err.Error(), expectedErr) {
		t.Fatalf("unexpected error message: %v; want %q", err, expectedErr)
	}

	// Mark sub-1 finished via Complete (without calling any explicit close or manual removal)
	sup.Complete("sub-1", nil)

	// AutoReap manually to reclaim finished threads
	sup.AutoReap()
	if sup.ActiveCount() != 1 {
		t.Fatalf("after AutoReap, ActiveCount() = %d; want 1", sup.ActiveCount())
	}
	if sup.CompletedCount() != 1 {
		t.Fatalf("after AutoReap, CompletedCount() = %d; want 1", sup.CompletedCount())
	}

	// Now spawning sub-3 succeeds
	t3, err := sup.Spawn("parent-sess", "sub-3", nil)
	if err != nil || t3 == nil {
		t.Fatalf("Spawn(sub-3) failed after AutoReap: %v", err)
	}
	if sup.ActiveCount() != 2 {
		t.Fatalf("ActiveCount() = %d; want 2", sup.ActiveCount())
	}

	// Mark sub-2 finished
	sup.Complete("sub-2", nil)

	// Test proactive reaping inside Spawn: without calling AutoReap explicitly,
	// Spawn should proactively reap sub-2 and successfully allocate sub-4
	t4, err := sup.Spawn("parent-sess", "sub-4", nil)
	if err != nil || t4 == nil {
		t.Fatalf("Spawn(sub-4) with proactive reaping failed: %v", err)
	}
	if sup.ActiveCount() != 2 {
		t.Fatalf("ActiveCount() = %d; want 2", sup.ActiveCount())
	}

	// All active are running (sub-3, sub-4); spawn should fail again
	_, err = sup.Spawn("parent-sess", "sub-5", nil)
	if err == nil || !strings.Contains(err.Error(), expectedErr) {
		t.Fatalf("expected saturated error %q, got: %v", expectedErr, err)
	}
}

// Test 3: Call GetGoal from a spawned subagent and assert it receives the projected parent goal state
// (ReadOnly: true, correct Description) instead of null.
func TestSubagentSupervisor_GetGoal(t *testing.T) {
	sup := NewSubagentSupervisor(8)

	parentGoal := &GoalDescriptor{
		GoalID:          "goal-plan-auth",
		Description:     "Implement secure OAuth2 token exchange with refresh rotation",
		ReadOnly:        false, // parent goal may be mutable
		ParentSessionID: "sess-parent-root",
	}

	thread, err := sup.Spawn("sess-parent-root", "sub-auth-1", parentGoal)
	if err != nil || thread == nil {
		t.Fatalf("Spawn failed: %v", err)
	}

	// Retrieve goal via GetGoal
	goal, err := sup.GetGoal("sub-auth-1")
	if err != nil {
		t.Fatalf("GetGoal failed: %v", err)
	}
	if goal == nil {
		t.Fatalf("GetGoal returned nil goal descriptor")
	}

	// Assert projected snapshot has ReadOnly = true
	if !goal.ReadOnly {
		t.Errorf("expected goal.ReadOnly == true, got false")
	}
	if goal.Description != parentGoal.Description {
		t.Errorf("goal.Description = %q; want %q", goal.Description, parentGoal.Description)
	}
	if goal.GoalID != parentGoal.GoalID {
		t.Errorf("goal.GoalID = %q; want %q", goal.GoalID, parentGoal.GoalID)
	}
	if goal.ParentSessionID != "sess-parent-root" {
		t.Errorf("goal.ParentSessionID = %q; want %q", goal.ParentSessionID, "sess-parent-root")
	}
	if goal.ProjectedAt.IsZero() {
		t.Errorf("expected non-zero ProjectedAt timestamp")
	}

	// Test subagent spawned without goal
	_, err = sup.Spawn("sess-parent-root", "sub-no-goal", nil)
	if err != nil {
		t.Fatalf("Spawn without goal failed: %v", err)
	}
	goalNone, err := sup.GetGoal("sub-no-goal")
	if err == nil || goalNone != nil {
		t.Errorf("expected error for subagent without goal, got goal=%v, err=%v", goalNone, err)
	}

	// Test unknown subagent
	_, err = sup.GetGoal("non-existent-subagent")
	if err == nil {
		t.Errorf("expected error for non-existent subagent, got nil")
	}
}

// Test TeardownParent reaps all child subagents of the specified parent.
func TestSubagentSupervisor_TeardownParent(t *testing.T) {
	sup := NewSubagentSupervisor(8)

	_, err := sup.Spawn("parent-1", "sub-1a", nil)
	if err != nil {
		t.Fatalf("Spawn(sub-1a) failed: %v", err)
	}
	_, err = sup.Spawn("parent-1", "sub-1b", nil)
	if err != nil {
		t.Fatalf("Spawn(sub-1b) failed: %v", err)
	}
	_, err = sup.Spawn("parent-2", "sub-2a", nil)
	if err != nil {
		t.Fatalf("Spawn(sub-2a) failed: %v", err)
	}

	if sup.ActiveCount() != 3 {
		t.Fatalf("ActiveCount() = %d; want 3", sup.ActiveCount())
	}

	sup.TeardownParent("parent-1")

	if sup.ActiveCount() != 1 {
		t.Fatalf("ActiveCount() after TeardownParent = %d; want 1", sup.ActiveCount())
	}

	// sub-2a should still be active
	t2a, ok := sup.GetThread("sub-2a")
	if !ok || t2a.State != SubagentStateRunning {
		t.Fatalf("sub-2a should still be running, got %+v", t2a)
	}

	// sub-1a should be in completed pool
	t1a, ok := sup.GetThread("sub-1a")
	if !ok || t1a.State != SubagentStateCompleted {
		t.Fatalf("sub-1a should be completed in pool, got %+v", t1a)
	}
}

// Test Wait unblocks on Complete and returns correct status and exit error.
func TestSubagentSupervisor_Wait(t *testing.T) {
	sup := NewSubagentSupervisor(4)

	_, err := sup.Spawn("parent-sess", "sub-wait-1", nil)
	if err != nil {
		t.Fatalf("Spawn failed: %v", err)
	}

	go func() {
		time.Sleep(10 * time.Millisecond)
		sup.Complete("sub-wait-1", nil)
	}()

	thread, err := sup.Wait("sub-wait-1", 1000) // 1000 will be clamped to 60000ms floor
	if err != nil {
		t.Fatalf("Wait failed: %v", err)
	}
	if thread.State != SubagentStateCompleted {
		t.Fatalf("thread.State = %q; want %q", thread.State, SubagentStateCompleted)
	}

	// Test Wait with error exit
	customErr := errors.New("subagent synthetic failure")
	_, err = sup.Spawn("parent-sess", "sub-wait-err", nil)
	if err != nil {
		t.Fatalf("Spawn failed: %v", err)
	}

	go func() {
		time.Sleep(10 * time.Millisecond)
		sup.Complete("sub-wait-err", customErr)
	}()

	threadErr, err := sup.Wait("sub-wait-err", 5000)
	if err != customErr {
		t.Fatalf("expected error %v, got %v", customErr, err)
	}
	if threadErr.State != SubagentStateFailed {
		t.Fatalf("threadErr.State = %q; want %q", threadErr.State, SubagentStateFailed)
	}
}
