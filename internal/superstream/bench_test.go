package superstream

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// BenchmarkDecideStep exercises the pure decision kernel across the primary lifecycle phases.
func BenchmarkDecideStep(b *testing.B) {
	spec := StreamSpec{
		ID:              "bench-stream",
		Intent:          "benchmark-lifecycle",
		MaxTurnsTotal:   50,
		MaxTurnsPerItem: 10,
		Queue: []WorkItem{
			{ID: "task-gw", Lane: "gateway", Tree: []string{"internal/gateway/**"}},
		},
	}.NormalizedSpec()

	holder := "bench-worker"
	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
		},
	}

	b.Run("Pending_AcquireLease", func(b *testing.B) {
		state := NewStreamState(spec)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dec := DecideStep(spec, state, holder, nil, tax)
			if dec.Action != ActionAcquireLease {
				b.Fatalf("unexpected action: %s", dec.Action)
			}
		}
	})

	b.Run("LeaseAcquired_ExecuteItem", func(b *testing.B) {
		state := NewStreamState(spec)
		held := MakeHeldLease(state.StreamID, state.Queue[0], holder, time.Now(), "token-1")
		state.CurrentLease = &held
		state.Queue[0].Status = ItemLeaseAcquired

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dec := DecideStep(spec, state, holder, nil, tax)
			if dec.Action != ActionExecuteItem {
				b.Fatalf("unexpected action: %s", dec.Action)
			}
		}
	})

	b.Run("Executing_WitnessAndCommit", func(b *testing.B) {
		state := NewStreamState(spec)
		held := MakeHeldLease(state.StreamID, state.Queue[0], holder, time.Now(), "token-1")
		state.CurrentLease = &held
		state.Queue[0].Status = ItemExecuting
		state.CurrentItemTurns = 3

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dec := DecideStep(spec, state, holder, nil, tax)
			if dec.Action != ActionWitnessAndCommit {
				b.Fatalf("unexpected action: %s", dec.Action)
			}
		}
	})

	b.Run("Committed_ReleaseLease", func(b *testing.B) {
		state := NewStreamState(spec)
		held := MakeHeldLease(state.StreamID, state.Queue[0], holder, time.Now(), "token-1")
		state.CurrentLease = &held
		state.Queue[0].Status = ItemCommitted
		state.Queue[0].CommitSHA = "abcdef123456"

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dec := DecideStep(spec, state, holder, nil, tax)
			if dec.Action != ActionReleaseLease {
				b.Fatalf("unexpected action: %s", dec.Action)
			}
		}
	})

	b.Run("AllTerminal_StreamComplete", func(b *testing.B) {
		state := NewStreamState(spec)
		state.Queue[0].Status = ItemCompleted
		state.ActiveIndex = 0

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dec := DecideStep(spec, state, holder, nil, tax)
			if dec.Action != ActionStreamComplete {
				b.Fatalf("unexpected action: %s", dec.Action)
			}
		}
	})
}

// BenchmarkApplyStep exercises state transitions for individual actions.
func BenchmarkApplyStep(b *testing.B) {
	spec := StreamSpec{
		ID:              "bench-apply",
		Intent:          "bench-transitions",
		MaxTurnsTotal:   50,
		MaxTurnsPerItem: 10,
		Queue: []WorkItem{
			{ID: "task-gw", Lane: "gateway", Tree: []string{"internal/gateway/**"}},
			{ID: "task-docs", Lane: "docs", Tree: []string{"docs/**"}},
		},
	}.NormalizedSpec()

	now := time.Now()

	b.Run("ActionAcquireLease", func(b *testing.B) {
		dec := StepDecision{Action: ActionAcquireLease}
		res := StepResult{LeaseAcquired: true, FencingToken: "fence-1"}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			state := NewStreamState(spec)
			ApplyStep(spec, &state, dec, res, now)
			if state.CurrentLease == nil {
				b.Fatal("lease was not acquired")
			}
		}
	})

	b.Run("ActionExecuteItem", func(b *testing.B) {
		dec := StepDecision{Action: ActionExecuteItem}
		res := StepResult{ExecutedTurns: 2, ExecutedTokens: 1500}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			state := NewStreamState(spec)
			state.Queue[0].Status = ItemLeaseAcquired
			ApplyStep(spec, &state, dec, res, now)
			if state.CurrentItemTurns != 2 {
				b.Fatalf("expected 2 turns, got %d", state.CurrentItemTurns)
			}
		}
	})

	b.Run("ActionWitnessAndCommit", func(b *testing.B) {
		dec := StepDecision{Action: ActionWitnessAndCommit}
		res := StepResult{WitnessSuccess: true, WitnessOutput: "OK", CommitSHA: "c0ffee"}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			state := NewStreamState(spec)
			state.Queue[0].Status = ItemExecuting
			ApplyStep(spec, &state, dec, res, now)
			if state.Queue[0].Status != ItemCommitted {
				b.Fatalf("expected ItemCommitted, got %s", state.Queue[0].Status)
			}
		}
	})

	b.Run("ActionReleaseLease", func(b *testing.B) {
		dec := StepDecision{Action: ActionReleaseLease}
		res := StepResult{LeaseReleased: true}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			state := NewStreamState(spec)
			held := MakeHeldLease(state.StreamID, state.Queue[0], "worker", now, "token")
			state.CurrentLease = &held
			state.Queue[0].Status = ItemCommitted
			ApplyStep(spec, &state, dec, res, now)
			if state.CurrentLease != nil || state.Queue[0].Status != ItemCompleted {
				b.Fatal("lease not released or item not completed")
			}
		}
	})

	b.Run("ActionAdvanceQueue", func(b *testing.B) {
		dec := StepDecision{Action: ActionAdvanceQueue}
		res := StepResult{}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			state := NewStreamState(spec)
			state.Queue[0].Status = ItemCompleted
			state.ActiveIndex = 0
			ApplyStep(spec, &state, dec, res, now)
			if state.ActiveIndex != 1 {
				b.Fatalf("expected ActiveIndex 1, got %d", state.ActiveIndex)
			}
		}
	})
}

// BenchmarkFullStreamWorkflow measures end-to-end progression of an entire multi-item workstream queue.
func BenchmarkFullStreamWorkflow(b *testing.B) {
	spec := StreamSpec{
		ID:              "bench-full-stream",
		Intent:          "full-stepping-simulation",
		MaxTurnsTotal:   100,
		MaxTurnsPerItem: 10,
		BasePins:        []string{"build-clean", "tests-pass"},
		Queue: []WorkItem{
			{ID: "item-gw", Title: "Gateway Task", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Witness: "test-gw"},
			{ID: "item-eng", Title: "Engine Task", Lane: "engine", Tree: []string{"internal/engine/**"}, Witness: "test-eng"},
			{ID: "item-docs", Title: "Docs Task", Lane: "docs", Tree: []string{"docs/**"}, Witness: "test-docs"},
		},
	}.NormalizedSpec()

	holder := "bench-stepper"
	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"engine":  {"internal/engine/**"},
			"docs":    {"docs/**"},
		},
	}
	now := time.Now()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		state := NewStreamState(spec)
		steps := 0
		maxSteps := 50

		for !state.Closed && steps < maxSteps {
			steps++
			dec := DecideStep(spec, state, holder, nil, tax)
			var res StepResult

			switch dec.Action {
			case ActionAcquireLease:
				res = StepResult{LeaseAcquired: true, FencingToken: "fence-tok"}
			case ActionExecuteItem:
				res = StepResult{ExecutedTurns: 2, ExecutedTokens: 1200}
			case ActionWitnessAndCommit:
				res = StepResult{WitnessSuccess: true, WitnessOutput: "PASS", CommitSHA: "abc012"}
			case ActionReleaseLease:
				res = StepResult{LeaseReleased: true}
			case ActionAdvanceQueue:
				res = StepResult{}
			case ActionStreamComplete:
				res = StepResult{}
			default:
				b.Fatalf("unexpected action during loop: %s", dec.Action)
			}

			ApplyStep(spec, &state, dec, res, now)
		}

		if !state.Closed {
			b.Fatalf("stream did not reach closed state in %d steps", steps)
		}
		if state.CompletedCount != 3 {
			b.Fatalf("expected 3 completed items, got %d", state.CompletedCount)
		}
	}
}

// BenchmarkEvaluateLeaseAdmission measures lane admission adjudication under various conditions.
func BenchmarkEvaluateLeaseAdmission(b *testing.B) {
	item := WorkItem{
		ID:   "task-model",
		Lane: "model",
		Tree: []string{"internal/model/**"},
	}
	holder := "agent-1"

	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"model":   {"internal/model/**"},
			"gateway": {"internal/gateway/**"},
			"engine":  {"internal/engine/**"},
			"docs":    {"docs/**"},
		},
	}

	b.Run("FreeLane_NoPeers", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verdict, _ := EvaluateLeaseAdmission("stream-1", item, holder, nil, tax)
			if !verdict.Admit {
				b.Fatalf("expected admission: %s", verdict.Reason)
			}
		}
	})

	b.Run("ContendedLane_SameLane", func(b *testing.B) {
		peerLeases := []laneadmit.Lease{
			{ID: "peer-1", Lane: "model", Tree: []string{"internal/model/**"}, Holder: "peer-worker"},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verdict, _ := EvaluateLeaseAdmission("stream-1", item, holder, peerLeases, tax)
			if verdict.Admit {
				b.Fatal("expected refusal on contended lane")
			}
		}
	})

	b.Run("ManyLiveLeases_Disjoint", func(b *testing.B) {
		peerLeases := make([]laneadmit.Lease, 20)
		for j := 0; j < 20; j++ {
			peerLeases[j] = laneadmit.Lease{
				ID:     fmt.Sprintf("peer-lease-%d", j),
				Lane:   fmt.Sprintf("peer-lane-%d", j),
				Tree:   []string{fmt.Sprintf("internal/pkg%d/**", j)},
				Holder: fmt.Sprintf("peer-%d", j),
			}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verdict, _ := EvaluateLeaseAdmission("stream-1", item, holder, peerLeases, tax)
			if !verdict.Admit {
				b.Fatalf("expected admission with disjoint peer leases: %s", verdict.Reason)
			}
		}
	})
}

// BenchmarkFindNextDisjointItem measures skipping contended items to find an admissible queue candidate.
func BenchmarkFindNextDisjointItem(b *testing.B) {
	queue := []WorkItem{
		{ID: "it-0", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Status: ItemPending},
		{ID: "it-1", Lane: "engine", Tree: []string{"internal/engine/**"}, Status: ItemPending},
		{ID: "it-2", Lane: "model", Tree: []string{"internal/model/**"}, Status: ItemPending},
		{ID: "it-3", Lane: "docs", Tree: []string{"docs/**"}, Status: ItemPending},
		{ID: "it-4", Lane: "bench", Tree: []string{"bench/**"}, Status: ItemPending},
	}

	peerLeases := []laneadmit.Lease{
		{ID: "p-0", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Holder: "peer-0"},
		{ID: "p-1", Lane: "engine", Tree: []string{"internal/engine/**"}, Holder: "peer-1"},
		{ID: "p-2", Lane: "model", Tree: []string{"internal/model/**"}, Holder: "peer-2"},
	}

	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"engine":  {"internal/engine/**"},
			"model":   {"internal/model/**"},
			"docs":    {"docs/**"},
			"bench":   {"bench/**"},
		},
	}

	holder := "stream-agent"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, found := FindNextDisjointItem("stream-skip", queue, 0, holder, peerLeases, tax)
		if !found || idx != 3 {
			b.Fatalf("expected disjoint index 3, got %d (found=%v)", idx, found)
		}
	}
}

// BenchmarkCanCoexistWithHeld measures checking coexistence between an existing held lease and a candidate item.
func BenchmarkCanCoexistWithHeld(b *testing.B) {
	held := &HeldLease{
		LeaseID: "stream:s1:item1:worker",
		Lane:    "gateway",
		Tree:    []string{"internal/gateway/**"},
		Holder:  "worker",
	}

	disjointItem := WorkItem{
		ID:   "item-docs",
		Lane: "docs",
		Tree: []string{"docs/**"},
	}

	overlappingItem := WorkItem{
		ID:   "item-gw-sub",
		Lane: "gateway",
		Tree: []string{"internal/gateway/server/**"},
	}

	tax := laneadmit.Taxonomy{
		Loaded: true,
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"docs":    {"docs/**"},
		},
	}

	b.Run("Disjoint_Coexists", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !CanCoexistWithHeld("s1", held, disjointItem, "worker", tax) {
				b.Fatal("expected items to coexist")
			}
		}
	})

	b.Run("Overlapping_Refuses", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if CanCoexistWithHeld("s1", held, overlappingItem, "worker", tax) {
				b.Fatal("expected overlapping items to conflict")
			}
		}
	})
}

// BenchmarkEvaluateContextSafety exercises context budget calculations across health boundaries.
func BenchmarkEvaluateContextSafety(b *testing.B) {
	spec := StreamSpec{
		ID:              "bench-ctx-safety",
		Intent:          "evaluate-safety",
		MaxTurnsTotal:   40,
		MaxTokensTotal:  200000,
		MaxTurnsPerItem: 10,
		Queue: []WorkItem{
			{ID: "task-1", Lane: "gateway", MaxTurns: 10, MaxTokens: 50000},
		},
	}.NormalizedSpec()

	b.Run("Status_Safe", func(b *testing.B) {
		state := NewStreamState(spec)
		state.CurrentItemTurns = 2
		state.TotalTurnsSpent = 5

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := EvaluateContextSafety(spec, state)
			if v.Status != StatusContextSafe {
				b.Fatalf("expected StatusContextSafe, got %s", v.Status)
			}
		}
	})

	b.Run("Status_PressureWarn", func(b *testing.B) {
		state := NewStreamState(spec)
		state.CurrentItemTurns = 7
		state.TotalTurnsSpent = 7

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := EvaluateContextSafety(spec, state)
			if v.Status != StatusContextPressureWarn {
				b.Fatalf("expected StatusContextPressureWarn, got %s", v.Status)
			}
		}
	})

	b.Run("Status_ResetRequired", func(b *testing.B) {
		state := NewStreamState(spec)
		state.CurrentItemTurns = 10
		state.TotalTurnsSpent = 10

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := EvaluateContextSafety(spec, state)
			if v.Status != StatusContextResetRequired {
				b.Fatalf("expected StatusContextResetRequired, got %s", v.Status)
			}
		}
	})

	b.Run("Status_Exhausted", func(b *testing.B) {
		state := NewStreamState(spec)
		state.TotalTurnsSpent = 40

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := EvaluateContextSafety(spec, state)
			if v.Status != StatusContextExhausted {
				b.Fatalf("expected StatusContextExhausted, got %s", v.Status)
			}
		}
	})
}

// BenchmarkBuildCarryoverSeed measures generation of the O(1) carryover context seed.
func BenchmarkBuildCarryoverSeed(b *testing.B) {
	cases := []struct {
		name      string
		numItems  int
		completed int
		pins      int
	}{
		{name: "1_completed_3_pins", numItems: 3, completed: 1, pins: 3},
		{name: "5_completed_5_pins", numItems: 8, completed: 5, pins: 5},
		{name: "10_completed_8_pins", numItems: 15, completed: 10, pins: 8},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			queue := make([]WorkItem, tc.numItems)
			for i := 0; i < tc.numItems; i++ {
				status := ItemPending
				sha := ""
				witness := ""
				if i < tc.completed {
					status = ItemCompleted
					sha = fmt.Sprintf("commit-%04d", i)
					witness = "PASS"
				}
				queue[i] = WorkItem{
					ID:            fmt.Sprintf("item-%d", i),
					Title:         fmt.Sprintf("Work Item #%d", i),
					Lane:          fmt.Sprintf("lane-%d", i),
					Status:        status,
					CommitSHA:     sha,
					WitnessResult: witness,
				}
			}

			pins := make([]string, tc.pins)
			for p := 0; p < tc.pins; p++ {
				pins[p] = fmt.Sprintf("stream-pin-%d", p)
			}

			spec := StreamSpec{
				ID:            "bench-seed",
				Intent:        "carryover-test",
				BasePins:      pins,
				MaxTurnsTotal: 100,
				Queue:         queue,
			}.NormalizedSpec()

			state := StreamState{
				StreamID:    spec.ID,
				Intent:      spec.Intent,
				ActiveIndex: tc.completed,
				Queue:       spec.Queue,
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				seed := BuildCarryoverSeed(spec, state)
				if len(seed.CompletedItems) != tc.completed {
					b.Fatalf("expected %d completed items, got %d", tc.completed, len(seed.CompletedItems))
				}
			}
		})
	}
}

// BenchmarkBuildStreamLayout measures constructing the ctxplan layout configuration.
func BenchmarkBuildStreamLayout(b *testing.B) {
	for _, pinCount := range []int{2, 6, 16} {
		b.Run(fmt.Sprintf("pins_%d", pinCount), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				layout := BuildStreamLayout(pinCount)
				if layout.Base.MaxSpans < 4 {
					b.Fatalf("expected >= 4 spans, got %d", layout.Base.MaxSpans)
				}
			}
		})
	}
}

// BenchmarkGenerateLeaseID measures string synthesis for deterministic lease identities.
func BenchmarkGenerateLeaseID(b *testing.B) {
	streamID := "stream:test:20260908"
	itemID := "task:feature:auth"
	holder := "worker:subagent:42"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := GenerateLeaseID(streamID, itemID, holder)
		if len(id) == 0 {
			b.Fatal("empty lease id")
		}
	}
}
