package laneadmit

import (
	"errors"
	"sync"
	"testing"
)

func TestTransitionBasicAcquireRelease(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	if err := mgr.Acquire("worker-1", "test-authoring"); err != nil {
		t.Fatalf("Acquire test-authoring failed: %v", err)
	}

	if !mgr.IsHeld("test-authoring") {
		t.Fatal("expected test-authoring to be held")
	}

	holder, ok := mgr.Holder("test-authoring")
	if !ok || holder != "worker-1" {
		t.Fatalf("expected holder worker-1, got %q (ok=%v)", holder, ok)
	}

	// Double acquire from same session should succeed idempotently.
	if err := mgr.Acquire("worker-1", "test-authoring"); err != nil {
		t.Fatalf("re-entrant Acquire failed: %v", err)
	}

	// Acquire by another worker on same lane should fail.
	if err := mgr.Acquire("worker-2", "test-authoring"); err == nil {
		t.Fatal("expected Acquire collision for worker-2 on test-authoring")
	} else if !errors.Is(err, ErrLaneOccupied) {
		t.Fatalf("expected ErrLaneOccupied, got %v", err)
	}

	// Release by worker-1.
	if err := mgr.Release("worker-1", "test-authoring"); err != nil {
		t.Fatalf("Release failed: %v", err)
	}

	if mgr.IsHeld("test-authoring") {
		t.Fatal("expected test-authoring to be released")
	}

	// Release again should return ErrLaneNotHeld.
	if err := mgr.Release("worker-1", "test-authoring"); err == nil {
		t.Fatal("expected ErrLaneNotHeld on double release")
	} else if !errors.Is(err, ErrLaneNotHeld) {
		t.Fatalf("expected ErrLaneNotHeld, got %v", err)
	}
}

func TestTransitionSuccess(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	if err := mgr.Acquire("session-1", "test-authoring"); err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}

	res, err := mgr.TransitionLane("session-1", "test-authoring", "code")
	if err != nil {
		t.Fatalf("TransitionLane unexpected error: %v", err)
	}

	if !res.Success {
		t.Fatalf("expected transition success, got failure with reason: %s", res.Reason)
	}
	if res.FromLane != "test-authoring" {
		t.Fatalf("expected FromLane test-authoring, got %q", res.FromLane)
	}
	if res.ToLane != "code" {
		t.Fatalf("expected ToLane code, got %q", res.ToLane)
	}
	if res.AcquiredAt.IsZero() {
		t.Fatal("expected AcquiredAt to be non-zero")
	}

	// Old lane must be released.
	if mgr.IsHeld("test-authoring") {
		t.Fatal("fromLane test-authoring must be released after transition")
	}

	// New lane must be held by session-1.
	holder, ok := mgr.Holder("code")
	if !ok || holder != "session-1" {
		t.Fatalf("expected holder session-1 for code lane, got %q (ok=%v)", holder, ok)
	}

	held := mgr.HeldLanes("session-1")
	if len(held) != 1 || held[0] != "code" {
		t.Fatalf("expected HeldLanes [code], got %v", held)
	}
}

func TestTransitionOccupiedRefusal(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	// session-1 holds "code".
	if err := mgr.Acquire("session-1", "code"); err != nil {
		t.Fatalf("Acquire session-1 code failed: %v", err)
	}
	// session-2 holds "test-authoring".
	if err := mgr.Acquire("session-2", "test-authoring"); err != nil {
		t.Fatalf("Acquire session-2 test-authoring failed: %v", err)
	}

	// session-2 attempts to transition to "code", which is occupied by session-1.
	res, err := mgr.TransitionLane("session-2", "test-authoring", "code")
	if err != nil {
		t.Fatalf("expected nil error on refusal, got: %v", err)
	}

	if res.Success {
		t.Fatal("expected transition to be refused because target lane is occupied")
	}
	if res.Reason != ReasonCollisionRisk {
		t.Fatalf("expected reason %q, got %q", ReasonCollisionRisk, res.Reason)
	}

	// Invariant: no drop. session-2 MUST still hold "test-authoring".
	holder2, ok2 := mgr.Holder("test-authoring")
	if !ok2 || holder2 != "session-2" {
		t.Fatalf("session-2 must still hold test-authoring (no drop), got holder=%q ok=%v", holder2, ok2)
	}

	// session-1 MUST still hold "code".
	holder1, ok1 := mgr.Holder("code")
	if !ok1 || holder1 != "session-1" {
		t.Fatalf("session-1 must still hold code, got holder=%q ok=%v", holder1, ok1)
	}
}

func TestTransitionUnheldLane(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	// session-1 does not hold "test-authoring".
	res, err := mgr.TransitionLane("session-1", "test-authoring", "code")
	if err == nil {
		t.Fatal("expected error when fromLane is not held")
	}
	if !errors.Is(err, ErrLaneNotHeld) {
		t.Fatalf("expected ErrLaneNotHeld, got %v", err)
	}
	if res.Success {
		t.Fatal("expected Success=false when fromLane is not held")
	}

	// "code" should NOT be held.
	if mgr.IsHeld("code") {
		t.Fatal("toLane should not be acquired when transition fails")
	}
}

func TestTransitionSameLane(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	if err := mgr.Acquire("session-1", "test-authoring"); err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}

	res, err := mgr.TransitionLane("session-1", "test-authoring", "test-authoring")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Success {
		t.Fatalf("expected success on same-lane transition, got: %v", res.Reason)
	}
	if !mgr.IsHeld("test-authoring") {
		t.Fatal("test-authoring should still be held")
	}
}

func TestTransitionHierarchicalConflict(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	// session-1 holds coarse lane "gateway".
	if err := mgr.Acquire("session-1", "gateway"); err != nil {
		t.Fatalf("Acquire gateway failed: %v", err)
	}
	// session-2 holds "test-authoring".
	if err := mgr.Acquire("session-2", "test-authoring"); err != nil {
		t.Fatalf("Acquire test-authoring failed: %v", err)
	}

	// session-2 attempts to transition to sub-lane "gateway/server".
	res, err := mgr.TransitionLane("session-2", "test-authoring", "gateway/server")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Success {
		t.Fatal("expected transition to be refused due to parent lane conflict")
	}
	if res.Reason != ReasonCollisionRisk {
		t.Fatalf("expected reason %q, got %q", ReasonCollisionRisk, res.Reason)
	}

	// session-2 must still hold "test-authoring".
	if holder, ok := mgr.Holder("test-authoring"); !ok || holder != "session-2" {
		t.Fatalf("session-2 must keep test-authoring, got %q", holder)
	}
}

func TestTransitionTaxonomyExclusiveAndTree(t *testing.T) {
	tax := Taxonomy{
		Loaded:    true,
		Exclusive: map[string]bool{"release": true},
		Trees: map[string][]string{
			"code":    {"internal/code/**"},
			"overlap": {"internal/code/**"},
			"docs":    {"docs/**"},
		},
	}

	mgr := NewDynamicLeaseManager().WithTaxonomy(tax)

	// session-1 holds docs.
	if err := mgr.Acquire("session-1", "docs"); err != nil {
		t.Fatalf("Acquire docs failed: %v", err)
	}
	// session-2 holds code.
	if err := mgr.Acquire("session-2", "code"); err != nil {
		t.Fatalf("Acquire code failed: %v", err)
	}

	// session-1 tries to transition to exclusive lane "release" while session-2 holds code.
	res, err := mgr.TransitionLane("session-1", "docs", "release")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Success {
		t.Fatal("expected refusal transitioning to exclusive lane with active peers")
	}
	if res.Reason != ReasonCollisionRisk {
		t.Fatalf("expected reason %q, got %q", ReasonCollisionRisk, res.Reason)
	}

	// session-1 tries to transition to "overlap" which shares tree with "code".
	res2, err2 := mgr.TransitionLane("session-1", "docs", "overlap")
	if err2 != nil {
		t.Fatalf("unexpected error: %v", err2)
	}
	if res2.Success {
		t.Fatal("expected refusal on tree overlap")
	}

	// session-1 still holds docs.
	if holder, ok := mgr.Holder("docs"); !ok || holder != "session-1" {
		t.Fatalf("session-1 must still hold docs, got %q", holder)
	}
}

func TestTransitionConcurrent(t *testing.T) {
	mgr := NewDynamicLeaseManager()

	const n = 10
	// Initialize n workers on their own test lanes.
	for i := 0; i < n; i++ {
		lane := "test-" + string(rune('a'+i))
		session := "worker-" + string(rune('a'+i))
		if err := mgr.Acquire(session, lane); err != nil {
			t.Fatalf("Acquire %s failed: %v", lane, err)
		}
	}

	// All n workers try to transition concurrently to the single "code" lane.
	var wg sync.WaitGroup
	results := make([]TransitionResult, n)
	errorsList := make([]error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			lane := "test-" + string(rune('a'+idx))
			session := "worker-" + string(rune('a'+idx))
			results[idx], errorsList[idx] = mgr.TransitionLane(session, lane, "code")
		}(i)
	}
	wg.Wait()

	successCount := 0
	var winnerSession string
	for i := 0; i < n; i++ {
		if errorsList[i] != nil {
			t.Fatalf("worker %d got unexpected error: %v", i, errorsList[i])
		}
		if results[i].Success {
			successCount++
			winnerSession = "worker-" + string(rune('a'+i))
		}
	}

	if successCount != 1 {
		t.Fatalf("expected exactly 1 worker to transition to code, got %d", successCount)
	}

	// Check code lane holder.
	codeHolder, ok := mgr.Holder("code")
	if !ok || codeHolder != winnerSession {
		t.Fatalf("expected code holder %s, got %s (ok=%v)", winnerSession, codeHolder, ok)
	}

	// Check that all other workers kept their original test lane.
	for i := 0; i < n; i++ {
		session := "worker-" + string(rune('a'+i))
		lane := "test-" + string(rune('a'+i))
		if session == winnerSession {
			if mgr.IsHeld(lane) {
				t.Fatalf("winner %s must have released %s", winnerSession, lane)
			}
		} else {
			h, held := mgr.Holder(lane)
			if !held || h != session {
				t.Fatalf("worker %s must have retained %s, got holder %s (held=%v)", session, lane, h, held)
			}
		}
	}
}
