package superstream

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// scripted lifecycle results, mirroring BenchmarkFullStreamWorkflow.
func drainResultFor(dec StepDecision) StepResult {
	switch dec.Action {
	case ActionAcquireLease:
		return StepResult{LeaseAcquired: true, FencingToken: "fence-tok"}
	case ActionExecuteItem:
		return StepResult{ExecutedTurns: 2, ExecutedTokens: 1200}
	case ActionWitnessAndCommit:
		return StepResult{WitnessSuccess: true, WitnessOutput: "PASS", CommitSHA: "abc012"}
	case ActionReleaseLease:
		return StepResult{LeaseReleased: true}
	default:
		return StepResult{}
	}
}

// TestStepperMatchesDecideStep pins the semantic contract: stepping through
// a scripted multi-item lifecycle with a Stepper must produce decisions
// identical to DecideStep at every state.
func TestStepperMatchesDecideStep(t *testing.T) {
	spec := StreamSpec{
		ID:              "stepper-equiv",
		Intent:          "stepper-decides-equal",
		MaxTurnsTotal:   100,
		MaxTurnsPerItem: 10,
		BasePins:        []string{"tests-pass"},
		Queue: []WorkItem{
			{ID: "item-gw", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Witness: "test-gw"},
			{ID: "item-eng", Lane: "engine", Tree: []string{"internal/engine/**"}, Witness: "test-eng"},
			{ID: "item-docs", Lane: "docs", Tree: []string{"docs/**"}, Witness: "test-docs"},
		},
	}
	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"engine":  {"internal/engine/**"},
			"docs":    {"docs/**"},
		},
	}
	now := time.Now()

	norm := spec.NormalizedSpec()
	state := NewStreamState(norm)
	stepper := NewStepper(spec, "equiv-holder")

	steps := 0
	for !state.Closed && steps < 50 {
		steps++
		want := DecideStep(spec, state, "equiv-holder", nil, tax)
		got := stepper.Step(state, nil, tax)
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("step %d: stepper decision diverged from DecideStep:\n want %+v\n got  %+v", steps, want, got)
		}
		ApplyStep(spec, &state, want, drainResultFor(want), now)
	}
	if !state.Closed {
		t.Fatalf("stream did not close in %d steps", steps)
	}
	if state.CompletedCount != 3 {
		t.Fatalf("expected 3 completed items, got %d", state.CompletedCount)
	}
}

// TestStepperStepDoesNotCopyQueue pins the amortization contract: after
// construction, one Stepper.Step over a Q-item queue must not copy the
// queue, so per-step allocation stays O(1) in queue length. DecideStep at
// the same Q spends one queue copy (~Q*184 B) per call.
func TestStepperStepDoesNotCopyQueue(t *testing.T) {
	const items = 2000
	const runs = 20

	queue := make([]WorkItem, items)
	for i := range queue {
		queue[i] = WorkItem{
			ID:     fmt.Sprintf("item-%d", i),
			Lane:   fmt.Sprintf("lane-%d", i),
			Status: ItemPending,
		}
	}
	spec := StreamSpec{
		ID:              "stepper-no-copy",
		Intent:          "amortized-normalization",
		MaxTurnsTotal:   1 << 20,
		MaxTurnsPerItem: 8,
		// Complexity gate: deliberately beyond the default MaxItems cap.
		MaxItems: 1 << 20,
		Queue:    queue,
	}
	state := StreamState{StreamID: spec.ID, Queue: queue}
	tax := laneadmit.Taxonomy{Loaded: true}
	stepper := NewStepper(spec, "stepper-holder")

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < runs; i++ {
		if dec := stepper.Step(state, nil, tax); dec.Action != ActionAcquireLease {
			t.Fatalf("unexpected action: %s", dec.Action)
		}
	}
	runtime.ReadMemStats(&after)

	bytesPerOp := int64(after.TotalAlloc-before.TotalAlloc) / runs
	// An absolute bound: reason strings and the verdict are a few hundred
	// bytes; one queue copy at Q=2000 is ~368 KB. Anything approaching one
	// WorkItem per item means the normalized copy leaked back into Step.
	const bound = 4096
	if bytesPerOp > bound {
		t.Fatalf("Stepper.Step still scales with queue length: %d B/op > %d B bound at Q=%d (one queue copy would be ~%d B)",
			bytesPerOp, bound, items, items*184)
	}
}

// TestStepperDrainAllocationLinear pins the drain contract: draining a
// Q-item stream through Stepper + ApplyStep must allocate O(Q) total, not
// the O(Q^2) a per-step normalization would spend.
func TestStepperDrainAllocationLinear(t *testing.T) {
	const items = 1000

	queue := make([]WorkItem, items)
	for i := range queue {
		queue[i] = WorkItem{
			ID:     fmt.Sprintf("item-%d", i),
			Lane:   fmt.Sprintf("lane-%d", i),
			Status: ItemPending,
		}
	}
	spec := StreamSpec{
		ID:              "stepper-drain",
		Intent:          "linear-drain",
		MaxTurnsTotal:   1 << 24,
		MaxTokensTotal:  1 << 30,
		MaxTurnsPerItem: 8,
		// Complexity gate: deliberately beyond the default MaxItems cap.
		MaxItems: 1 << 24,
		Queue:    queue,
	}
	tax := laneadmit.Taxonomy{Loaded: true}
	now := time.Now()

	stepper := NewStepper(spec, "drain-holder")
	state := NewStreamState(spec)

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	steps := 0
	for !state.Closed && steps < 8*items {
		steps++
		dec := stepper.Step(state, nil, tax)
		if dec.Action == ActionStreamHalted {
			t.Fatalf("drain halted at step %d: %s", steps, dec.Reason)
		}
		ApplyStep(spec, &state, dec, drainResultFor(dec), now)
	}
	runtime.ReadMemStats(&after)

	itemSize := int64(reflect.TypeOf(WorkItem{}).Size())
	summarySize := int64(reflect.TypeOf(ItemSummary{}).Size())
	total := int64(after.TotalAlloc - before.TotalAlloc)
	// O(Q^2) normalization would spend ~items queue copies:
	// items * items * itemSize (~184 MB here). The linear contract is: per
	// item, the drain spends the small decision strings plus at most one
	// windowed carryover seed (two copies of at most `window` summaries).
	perItem := int64(DefaultCarryoverReceiptWindow)*summarySize*2 + 4096
	bound := int64(items) * perItem
	if total > bound {
		t.Fatalf("drain allocation is superlinear: %d B total > %d B bound over %d steps at Q=%d (per-step normalization would be ~%d B)",
			total, bound, steps, items, int64(items)*int64(items)*itemSize)
	}
	if state.CompletedCount != items {
		t.Fatalf("expected %d completed items, got %d", items, state.CompletedCount)
	}
}

// TestStreamSpecValidateCap pins the queue-length bound: an oversized queue
// fails Validate unless the spec states a higher cap explicitly.
func TestStreamSpecValidateCap(t *testing.T) {
	queue := make([]WorkItem, DefaultMaxItems+1)
	for i := range queue {
		queue[i] = WorkItem{ID: fmt.Sprintf("item-%d", i), Lane: fmt.Sprintf("lane-%d", i)}
	}

	oversized := StreamSpec{ID: "big", Queue: queue}
	if err := oversized.Validate(); err == nil {
		t.Fatalf("expected Validate to refuse %d-item queue under default cap %d", len(queue), DefaultMaxItems)
	}

	raised := StreamSpec{ID: "big-raised", MaxItems: len(queue) + 1, Queue: queue}
	if err := raised.Validate(); err != nil {
		t.Fatalf("explicit higher cap should pass: %v", err)
	}

	small := StreamSpec{ID: "small", Queue: queue[:3]}
	if err := small.Validate(); err != nil {
		t.Fatalf("bounded queue should pass: %v", err)
	}
}

// TestDecideStepRefusesOversizedQueue pins the navigation-side enforcement
// of the cap: a queue longer than the (defaulted) MaxItems halts the stream
// with a split-the-stream reason instead of being navigated.
func TestDecideStepRefusesOversizedQueue(t *testing.T) {
	queue := make([]WorkItem, DefaultMaxItems+1)
	for i := range queue {
		queue[i] = WorkItem{ID: fmt.Sprintf("item-%d", i), Lane: fmt.Sprintf("lane-%d", i), Status: ItemPending}
	}
	spec := StreamSpec{ID: "over-cap", Queue: queue}
	state := NewStreamState(spec)
	tax := laneadmit.Taxonomy{Loaded: true}

	dec := DecideStep(spec, state, "holder", nil, tax)
	if dec.Action != ActionStreamHalted {
		t.Fatalf("expected ActionStreamHalted for oversized queue, got %s (%s)", dec.Action, dec.Reason)
	}

	// A closed stream stays closed regardless of queue size.
	state.Closed = true
	state.CloseReason = "test-close"
	dec = DecideStep(spec, state, "holder", nil, tax)
	if dec.Action != ActionStreamHalted {
		t.Fatalf("expected closed-stream halt to keep precedence, got %s", dec.Action)
	}

	// The stepper enforces the same bound from its one normalized copy.
	state.Closed = false
	state.CloseReason = ""
	stepper := NewStepper(spec, "holder")
	if dec := stepper.Step(state, nil, tax); dec.Action != ActionStreamHalted {
		t.Fatalf("expected Stepper to refuse oversized queue too, got %s (%s)", dec.Action, dec.Reason)
	}
}
