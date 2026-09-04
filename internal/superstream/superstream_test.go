package superstream

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

func TestNormalizedSpecDefaults(t *testing.T) {
	raw := StreamSpec{
		ID:     "stream-test",
		Intent: "quality-hardening",
		Queue: []WorkItem{
			{ID: "task-1", Lane: "gateway"},
			{ID: "task-2", Lane: "docs", MaxTurns: 15},
		},
	}
	norm := raw.NormalizedSpec()

	if norm.MaxTurnsPerItem != DefaultMaxTurnsPerItem {
		t.Fatalf("MaxTurnsPerItem = %d, want %d", norm.MaxTurnsPerItem, DefaultMaxTurnsPerItem)
	}
	if norm.MaxTurnsTotal != DefaultMaxTurnsTotal {
		t.Fatalf("MaxTurnsTotal = %d, want %d", norm.MaxTurnsTotal, DefaultMaxTurnsTotal)
	}
	if norm.Queue[0].Status != ItemPending {
		t.Fatalf("Queue[0].Status = %s, want %s", norm.Queue[0].Status, ItemPending)
	}
	if norm.Queue[0].MaxTurns != DefaultMaxTurnsPerItem {
		t.Fatalf("Queue[0].MaxTurns = %d, want %d", norm.Queue[0].MaxTurns, DefaultMaxTurnsPerItem)
	}
	if norm.Queue[1].MaxTurns != 15 {
		t.Fatalf("Queue[1].MaxTurns = %d, want 15", norm.Queue[1].MaxTurns)
	}
}

func TestContextSafetyEvaluation(t *testing.T) {
	spec := StreamSpec{
		ID:              "stream-ctx-test",
		Intent:          "test",
		MaxTurnsTotal:   20,
		MaxTokensTotal:  100000,
		MaxTurnsPerItem: 10,
		Queue: []WorkItem{
			{ID: "item-1", Lane: "gateway", MaxTurns: 10, MaxTokens: 50000},
		},
	}.NormalizedSpec()

	state := NewStreamState(spec)

	// Step 1: Initial state is SAFE
	v1 := EvaluateContextSafety(spec, state)
	if v1.Status != StatusContextSafe {
		t.Fatalf("v1 status = %s, want %s", v1.Status, StatusContextSafe)
	}

	// Step 2: High turn usage triggers PRESSURE_WARN (7/10 = 70%)
	state.CurrentItemTurns = 7
	state.TotalTurnsSpent = 7
	v2 := EvaluateContextSafety(spec, state)
	if v2.Status != StatusContextPressureWarn {
		t.Fatalf("v2 status = %s, want %s", v2.Status, StatusContextPressureWarn)
	}

	// Step 3: Turn limit reached triggers RESET_REQUIRED (10/10)
	state.CurrentItemTurns = 10
	state.TotalTurnsSpent = 10
	v3 := EvaluateContextSafety(spec, state)
	if v3.Status != StatusContextResetRequired || !v3.RecommendReset {
		t.Fatalf("v3 status = %s (reset=%v), want RESET_REQUIRED", v3.Status, v3.RecommendReset)
	}

	// Step 4: Total stream budget exhausted triggers EXHAUSTED
	state.TotalTurnsSpent = 20
	v4 := EvaluateContextSafety(spec, state)
	if v4.Status != StatusContextExhausted {
		t.Fatalf("v4 status = %s, want %s", v4.Status, StatusContextExhausted)
	}
}

func TestStreamCarryoverSeedStructure(t *testing.T) {
	spec := StreamSpec{
		ID:       "stream-carryover",
		Intent:   "refactoring",
		BasePins: []string{"keep tests green", "commit by explicit path"},
		Queue: []WorkItem{
			{ID: "item-1", Title: "Task 1", Lane: "gateway", Status: ItemCompleted, CommitSHA: "abc1234", WitnessResult: "PASS"},
			{ID: "item-2", Title: "Task 2", Lane: "engine", Status: ItemExecuting},
			{ID: "item-3", Title: "Task 3", Lane: "docs", Status: ItemPending},
		},
	}
	state := StreamState{
		StreamID:    spec.ID,
		Intent:      spec.Intent,
		ActiveIndex: 1,
		Queue:       spec.Queue,
	}

	seed := BuildCarryoverSeed(spec, state)

	if seed.Schema != CarryoverSchema {
		t.Fatalf("seed.Schema = %q, want %q", seed.Schema, CarryoverSchema)
	}
	if len(seed.CompletedItems) != 1 {
		t.Fatalf("len(CompletedItems) = %d, want 1", len(seed.CompletedItems))
	}
	if seed.CompletedItems[0].CommitSHA != "abc1234" {
		t.Fatalf("CompletedItems[0].CommitSHA = %q, want abc1234", seed.CompletedItems[0].CommitSHA)
	}
	if seed.CurrentItem == nil || seed.CurrentItem.ID != "item-2" {
		t.Fatalf("seed.CurrentItem = %v, want item-2", seed.CurrentItem)
	}
	if seed.NextItem == nil || seed.NextItem.ID != "item-3" {
		t.Fatalf("seed.NextItem = %v, want item-3", seed.NextItem)
	}
	if len(seed.StreamPins) != 2 {
		t.Fatalf("len(StreamPins) = %d, want 2", len(seed.StreamPins))
	}
	if seed.Layout == nil || seed.Layout.Base.Precision != "exact" {
		t.Fatalf("seed.Layout not properly initialized: %+v", seed.Layout)
	}
}

func TestDecideStepProgressionHappyPath(t *testing.T) {
	spec := StreamSpec{
		ID:              "stream-happy",
		Intent:          "clean-run",
		MaxTurnsTotal:   30,
		MaxTurnsPerItem: 6,
		Queue: []WorkItem{
			{ID: "task-1", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Witness: "make test-gateway"},
		},
	}.NormalizedSpec()

	state := NewStreamState(spec)
	holder := "agent-worker-1"
	tax := laneadmit.Taxonomy{Loaded: true}
	now := time.Now()

	// Step 1: Item is pending, lane is free -> ACQUIRE_LEASE
	d1 := DecideStep(spec, state, holder, nil, tax)
	if d1.Action != ActionAcquireLease {
		t.Fatalf("d1.Action = %s, want %s", d1.Action, ActionAcquireLease)
	}
	ApplyStep(spec, &state, d1, StepResult{LeaseAcquired: true, FencingToken: "fence-123"}, now)
	if state.CurrentLease == nil || state.ActiveItem().Status != ItemLeaseAcquired {
		t.Fatalf("state after d1 invalid: lease=%v, status=%s", state.CurrentLease, state.ActiveItem().Status)
	}

	// Step 2: Lease acquired -> EXECUTE_ITEM
	d2 := DecideStep(spec, state, holder, nil, tax)
	if d2.Action != ActionExecuteItem {
		t.Fatalf("d2.Action = %s, want %s", d2.Action, ActionExecuteItem)
	}
	ApplyStep(spec, &state, d2, StepResult{ExecutedTurns: 2, ExecutedTokens: 5000}, now)
	if state.CurrentItemTurns != 2 || state.ActiveItem().Status != ItemExecuting {
		t.Fatalf("state after d2 invalid: turns=%d, status=%s", state.CurrentItemTurns, state.ActiveItem().Status)
	}

	// Step 3: Executing -> WITNESS_AND_COMMIT
	d3 := DecideStep(spec, state, holder, nil, tax)
	if d3.Action != ActionWitnessAndCommit {
		t.Fatalf("d3.Action = %s, want %s", d3.Action, ActionWitnessAndCommit)
	}
	ApplyStep(spec, &state, d3, StepResult{WitnessSuccess: true, WitnessOutput: "PASS", CommitSHA: "7a8b9c"}, now)
	if state.ActiveItem().Status != ItemCommitted || state.ActiveItem().CommitSHA != "7a8b9c" {
		t.Fatalf("state after d3 invalid: status=%s, sha=%s", state.ActiveItem().Status, state.ActiveItem().CommitSHA)
	}

	// Step 4: Committed, but lease still held -> RELEASE_LEASE
	d4 := DecideStep(spec, state, holder, nil, tax)
	if d4.Action != ActionReleaseLease {
		t.Fatalf("d4.Action = %s, want %s", d4.Action, ActionReleaseLease)
	}
	ApplyStep(spec, &state, d4, StepResult{LeaseReleased: true}, now)
	if state.CurrentLease != nil || state.ActiveItem().Status != ItemCompleted {
		t.Fatalf("state after d4 invalid: lease=%v, status=%s", state.CurrentLease, state.ActiveItem().Status)
	}

	// Step 5: All queue items completed -> STREAM_COMPLETE
	d5 := DecideStep(spec, state, holder, nil, tax)
	if d5.Action != ActionStreamComplete {
		t.Fatalf("d5.Action = %s, want %s", d5.Action, ActionStreamComplete)
	}
	ApplyStep(spec, &state, d5, StepResult{}, now)
	if !state.Closed {
		t.Fatalf("state.Closed = false, want true")
	}
}

func TestContendedHeadSkipsToDisjointItem(t *testing.T) {
	spec := StreamSpec{
		ID:     "stream-contended",
		Intent: "disjoint-skip-test",
		Queue: []WorkItem{
			{ID: "task-gateway", Lane: "gateway", Tree: []string{"internal/gateway/**"}},
			{ID: "task-docs", Lane: "docs", Tree: []string{"docs/**"}},
		},
	}.NormalizedSpec()

	state := NewStreamState(spec)
	holder := "stream-agent"
	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"docs":    {"docs/**"},
		},
	}

	// Peer holds a lease on gateway
	peerLeases := []laneadmit.Lease{
		{
			ID:     "peer-gw-lease",
			Lane:   "gateway",
			Tree:   []string{"internal/gateway/**"},
			Holder: "peer-session-42",
		},
	}

	// Item 0 ("task-gateway") is contended, but Item 1 ("task-docs") is disjoint.
	// DecideStep should detect the collision and recommend YIELD_CONTENDED with DisjointIndex=1.
	d := DecideStep(spec, state, holder, peerLeases, tax)
	if d.Action != ActionYieldContended {
		t.Fatalf("d.Action = %s, want %s", d.Action, ActionYieldContended)
	}
	if d.DisjointIndex != 1 {
		t.Fatalf("d.DisjointIndex = %d, want 1", d.DisjointIndex)
	}

	// Applying the step switches ActiveIndex to 1 (the disjoint item)
	ApplyStep(spec, &state, d, StepResult{}, time.Now())
	if state.ActiveIndex != 1 {
		t.Fatalf("state.ActiveIndex = %d, want 1", state.ActiveIndex)
	}
	if state.ActiveItem().ID != "task-docs" {
		t.Fatalf("ActiveItem = %s, want task-docs", state.ActiveItem().ID)
	}

	// Next step on task-docs is free to acquire its lease!
	dNext := DecideStep(spec, state, holder, peerLeases, tax)
	if dNext.Action != ActionAcquireLease {
		t.Fatalf("dNext.Action = %s, want %s", dNext.Action, ActionAcquireLease)
	}
}
