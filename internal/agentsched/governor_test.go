package agentsched

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestPriorityQueueOrdering(t *testing.T) {
	pq := NewPriorityQueue(512)

	// Submit in reverse priority order
	_ = pq.Enqueue(&Task{ID: "p3-speculative", Priority: abi.ThreadPriorityP3Speculative})
	_ = pq.Enqueue(&Task{ID: "p2-batch", Priority: abi.ThreadPriorityP2Batch})
	_ = pq.Enqueue(&Task{ID: "p1-interactive", Priority: abi.ThreadPriorityP1Interactive})
	_ = pq.Enqueue(&Task{ID: "p0-system", Priority: abi.ThreadPriorityP0System})

	if pq.Len() != 4 {
		t.Fatalf("expected len 4, got %d", pq.Len())
	}

	expectedOrder := []string{"p0-system", "p1-interactive", "p2-batch", "p3-speculative"}
	for _, expected := range expectedOrder {
		task, ok := pq.Dequeue(true)
		if !ok || task == nil {
			t.Fatalf("expected task %s, got none", expected)
		}
		if task.ID != expected {
			t.Fatalf("dequeue order error: got %s, want %s", task.ID, expected)
		}
	}

	// Verify FIFO ordering within the same tier
	pqFIFO := NewPriorityQueue(10)
	_ = pqFIFO.Enqueue(&Task{ID: "p1-first", Priority: abi.ThreadPriorityP1Interactive})
	_ = pqFIFO.Enqueue(&Task{ID: "p1-second", Priority: abi.ThreadPriorityP1Interactive})
	_ = pqFIFO.Enqueue(&Task{ID: "p1-third", Priority: abi.ThreadPriorityP1Interactive})
	for _, expected := range []string{"p1-first", "p1-second", "p1-third"} {
		task, ok := pqFIFO.Dequeue(true)
		if !ok || task == nil || task.ID != expected {
			t.Fatalf("expected FIFO order %s, got %v", expected, task)
		}
	}
}

func TestPriorityQueueCapacityBackpressure(t *testing.T) {
	const cap = 10
	pq := NewPriorityQueue(cap)

	for i := 0; i < cap; i++ {
		err := pq.Enqueue(&Task{ID: fmt.Sprintf("task-%d", i), Priority: abi.ThreadPriorityP2Batch})
		if err != nil {
			t.Fatalf("unexpected enqueue error at %d: %v", i, err)
		}
	}

	// 11th task must fail with ErrQueueFull
	err := pq.Enqueue(&Task{ID: "overflow", Priority: abi.ThreadPriorityP1Interactive})
	if err == nil {
		t.Fatalf("expected ErrQueueFull on saturated queue")
	}

	var admErr *abi.AdmissionError
	if !errors.As(err, &admErr) {
		t.Fatalf("expected *abi.AdmissionError, got %T: %v", err, err)
	}
	if admErr.Code != abi.AdmissionCodeQueueFull {
		t.Fatalf("expected code %q, got %q", abi.AdmissionCodeQueueFull, admErr.Code)
	}
	if admErr.RetryAfterMS <= 0 {
		t.Fatalf("expected positive retry_after_ms, got %d", admErr.RetryAfterMS)
	}

	// Capacity overflow preserves existing enqueued tasks
	if pq.Len() != cap {
		t.Fatalf("expected length %d after overflow, got %d", cap, pq.Len())
	}
	for i := 0; i < cap; i++ {
		task, ok := pq.Dequeue(true)
		if !ok || task == nil || task.ID != fmt.Sprintf("task-%d", i) {
			t.Fatalf("queue contents corrupted at %d: got %v", i, task)
		}
	}
}

func TestFourGateAdmissionGovernor(t *testing.T) {
	headroom := NewMemoryProviderHeadroom()
	headroom.SetBudget("acc-1", 5000)

	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 2,
		Headroom:        headroom,
	})

	// Task 1: Passes all 4 gates
	task1 := &Task{
		ID:           "task-1",
		Priority:     abi.ThreadPriorityP1Interactive,
		Lane:         "gateway",
		Tree:         []string{"internal/gateway/*"},
		AccountID:    "acc-1",
		TokensNeeded: 1000,
	}

	_ = gov.Submit(task1)
	admitted1, verdict1, err := gov.TryAdmit()
	if err != nil || admitted1 == nil || !verdict1.Admitted {
		t.Fatalf("task1 should be admitted: err=%v verdict=%+v", err, verdict1)
	}
	if gov.InFlight() != 1 {
		t.Fatalf("expected 1 in-flight, got %d", gov.InFlight())
	}

	// Gate 4 Test: Conflicting Lane Clearance
	taskConflictingLane := &Task{
		ID:        "task-conflict",
		Priority:  abi.ThreadPriorityP1Interactive,
		Lane:      "gateway",
		Tree:      []string{"internal/gateway/*"},
		AccountID: "acc-1",
	}
	_ = gov.Submit(taskConflictingLane)
	admittedConf, verdictConf, _ := gov.TryAdmit()
	if admittedConf != nil || verdictConf.Admitted {
		t.Fatalf("task with overlapping lane should be blocked by Gate 4")
	}
	if verdictConf.FailedGate != GateLaneClearance {
		t.Fatalf("expected GateLaneClearance, got %v", verdictConf.FailedGate)
	}
	if gov.Queue().Len() != 1 {
		t.Fatalf("expected unadmitted task to remain enqueued, got len %d", gov.Queue().Len())
	}

	// Remove conflicting task
	gov.Queue().Remove("task-conflict")

	// Task 2: Disjoint lane -> Passes
	task2 := &Task{
		ID:           "task-2",
		Priority:     abi.ThreadPriorityP2Batch,
		Lane:         "sessionjournal",
		Tree:         []string{"internal/sessionjournal/*"},
		AccountID:    "acc-1",
		TokensNeeded: 500,
	}
	_ = gov.Submit(task2)
	admitted2, verdict2, err := gov.TryAdmit()
	if err != nil || admitted2 == nil || !verdict2.Admitted {
		t.Fatalf("task2 should be admitted: err=%v verdict=%+v", err, verdict2)
	}
	if gov.InFlight() != 2 {
		t.Fatalf("expected 2 in-flight, got %d", gov.InFlight())
	}

	// Gate 1 Test: Worker Concurrency (K=2 is full)
	task3 := &Task{
		ID:       "task-3",
		Priority: abi.ThreadPriorityP0System,
		Lane:     "microagent",
		Tree:     []string{"internal/microagent/*"},
	}
	_ = gov.Submit(task3)
	admitted3, verdict3, _ := gov.TryAdmit()
	if admitted3 != nil || verdict3.Admitted {
		t.Fatalf("task3 should be blocked by Gate 1 worker concurrency")
	}
	if verdict3.FailedGate != GateWorkerConcurrency {
		t.Fatalf("expected GateWorkerConcurrency, got %v", verdict3.FailedGate)
	}
	if gov.Queue().Len() != 1 {
		t.Fatalf("expected task3 to remain enqueued on concurrency block, got len %d", gov.Queue().Len())
	}

	// Release task 1 -> inFlight drops to 1 -> task3 admitted
	gov.Release(admitted1)
	if gov.InFlight() != 1 {
		t.Fatalf("expected 1 in-flight after release, got %d", gov.InFlight())
	}

	admitted3, verdict3, err = gov.TryAdmit()
	if err != nil || admitted3 == nil || !verdict3.Admitted {
		t.Fatalf("task3 should now be admitted: err=%v verdict=%+v", err, verdict3)
	}

	gov.Release(admitted2)
	gov.Release(admitted3)
}

func TestGate2HostEnvelopeFailures(t *testing.T) {
	gov := NewGovernor(GovernorConfig{BaseConcurrency: 4})

	task := &Task{ID: "t1", Priority: abi.ThreadPriorityP1Interactive}
	_ = gov.Submit(task)

	// 1. High CPU
	gov.UpdateTelemetry(HostTelemetry{CPUPct: 89.0, FreeRAMBytes: 8 * 1024 * 1024 * 1024, OpenHandles: 5000})
	v := gov.CheckAdmission(task)
	if v.Admitted || v.FailedGate != GateHostEnvelope {
		t.Fatalf("expected GateHostEnvelope on 89%% CPU, got %+v", v)
	}

	// 2. Low RAM
	gov.UpdateTelemetry(HostTelemetry{CPUPct: 20.0, FreeRAMBytes: 2 * 1024 * 1024 * 1024, OpenHandles: 5000})
	v = gov.CheckAdmission(task)
	if v.Admitted || v.FailedGate != GateHostEnvelope {
		t.Fatalf("expected GateHostEnvelope on low RAM, got %+v", v)
	}

	// 3. High handles
	gov.UpdateTelemetry(HostTelemetry{CPUPct: 20.0, FreeRAMBytes: 8 * 1024 * 1024 * 1024, OpenHandles: 140000})
	v = gov.CheckAdmission(task)
	if v.Admitted || v.FailedGate != GateHostEnvelope {
		t.Fatalf("expected GateHostEnvelope on handle exhaustion, got %+v", v)
	}

	// 4. Thermal pressure
	gov.UpdateTelemetry(HostTelemetry{CPUPct: 30.0, FreeRAMBytes: 8 * 1024 * 1024 * 1024, OpenHandles: 5000, ThermalPressure: true})
	v = gov.CheckAdmission(task)
	if v.Admitted || v.FailedGate != GateHostEnvelope {
		t.Fatalf("expected GateHostEnvelope on thermal pressure, got %+v", v)
	}
}

func TestGate3ProviderHeadroomFailures(t *testing.T) {
	headroom := NewMemoryProviderHeadroom()
	headroom.SetBudget("acc-1", 1000)
	headroom.SetThrottled("acc-throttled", true)

	gov := NewGovernor(GovernorConfig{BaseConcurrency: 4, Headroom: headroom})

	// Throttled account
	taskThrottled := &Task{ID: "t-thr", Priority: abi.ThreadPriorityP1Interactive, AccountID: "acc-throttled"}
	v := gov.CheckAdmission(taskThrottled)
	if v.Admitted || v.FailedGate != GateProviderHeadroom {
		t.Fatalf("expected GateProviderHeadroom for throttled account, got %+v", v)
	}

	// Budget exceeded
	taskOverBudget := &Task{ID: "t-ob", Priority: abi.ThreadPriorityP1Interactive, AccountID: "acc-1", TokensNeeded: 2000}
	v = gov.CheckAdmission(taskOverBudget)
	if v.Admitted || v.FailedGate != GateProviderHeadroom {
		t.Fatalf("expected GateProviderHeadroom for over-budget account, got %+v", v)
	}

	// Budget within limit
	taskWithinBudget := &Task{ID: "t-ok", Priority: abi.ThreadPriorityP1Interactive, AccountID: "acc-1", TokensNeeded: 500}
	v = gov.CheckAdmission(taskWithinBudget)
	if !v.Admitted {
		t.Fatalf("expected task within budget to be admitted, got %+v", v)
	}
}

func TestDynamicThermalLoadSheddingAndRecovery(t *testing.T) {
	const baseK = 16
	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: baseK,
		DropP3OnStress:  true,
	})

	if gov.EffectiveConcurrency() != baseK {
		t.Fatalf("expected initial concurrency %d, got %d", baseK, gov.EffectiveConcurrency())
	}

	// Enqueue P1 and P3 tasks
	_ = gov.Submit(&Task{ID: "p1-task", Priority: abi.ThreadPriorityP1Interactive})
	_ = gov.Submit(&Task{ID: "p3-spec-1", Priority: abi.ThreadPriorityP3Speculative})
	_ = gov.Submit(&Task{ID: "p3-spec-2", Priority: abi.ThreadPriorityP3Speculative})

	// 1. Simulate thermal / CPU stress (CPU = 92%)
	gov.UpdateTelemetry(HostTelemetry{
		CPUPct:          92.0,
		ThermalPressure: true,
	})

	// Concurrency should dynamically downscale K -> max(1, K/2) = 8
	if gov.EffectiveConcurrency() != 8 {
		t.Fatalf("expected downscaled concurrency 8, got %d", gov.EffectiveConcurrency())
	}
	if !gov.IsP3Paused() {
		t.Fatalf("expected P3 to be paused under thermal stress")
	}
	if gov.PacingMS() < DefaultPacingHighMS {
		t.Fatalf("expected pacing >= %d ms, got %d ms", DefaultPacingHighMS, gov.PacingMS())
	}
	// P3 dropped while higher-priority tasks are preserved
	if gov.Queue().LenPriority(abi.ThreadPriorityP3Speculative) != 0 {
		t.Fatalf("expected speculative tasks to be dropped, got %d", gov.Queue().LenPriority(abi.ThreadPriorityP3Speculative))
	}
	if gov.Queue().LenPriority(abi.ThreadPriorityP1Interactive) != 1 {
		t.Fatalf("expected P1 task preserved during load shedding, got %d", gov.Queue().LenPriority(abi.ThreadPriorityP1Interactive))
	}

	// 2. Second severe tick cuts concurrency again: 8 -> 4
	gov.UpdateTelemetry(HostTelemetry{
		CPUPct:          95.0,
		ThermalPressure: true,
	})
	if gov.EffectiveConcurrency() != 4 {
		t.Fatalf("expected downscaled concurrency 4, got %d", gov.EffectiveConcurrency())
	}

	// 3. Telemetry recovers (CPU drops to 40%, thermal clear)
	for i := 0; i < 20; i++ {
		gov.UpdateTelemetry(HostTelemetry{
			CPUPct:          40.0,
			FreeRAMBytes:    8 * 1024 * 1024 * 1024,
			OpenHandles:     2000,
			ThermalPressure: false,
			PowerSag:        false,
		})
	}

	// Concurrency should be fully restored to baseK (16) without daemon restart
	if gov.EffectiveConcurrency() != baseK {
		t.Fatalf("expected restored concurrency %d, got %d", baseK, gov.EffectiveConcurrency())
	}
	if gov.IsP3Paused() {
		t.Fatalf("expected P3 to be unpaused after full recovery")
	}
	if gov.PacingMS() != 0 {
		t.Fatalf("expected pacing reset to 0, got %d", gov.PacingMS())
	}
}

func TestConcurrentQueueAdmissionRaceFreedom(t *testing.T) {
	gov := NewGovernor(GovernorConfig{BaseConcurrency: 8, QueueCapacity: 512})

	var wg sync.WaitGroup
	const numProducers = 10
	const tasksPerProducer = 40

	// Concurrent submissions
	for p := 0; p < numProducers; p++ {
		wg.Add(1)
		go func(prodID int) {
			defer wg.Done()
			for i := 0; i < tasksPerProducer; i++ {
				prio := abi.ThreadPriority(prodID % 4)
				_ = gov.Submit(&Task{
					ID:       fmt.Sprintf("prod-%d-task-%d", prodID, i),
					Priority: prio,
				})
			}
		}(p)
	}

	wg.Wait()
	if gov.Queue().Len() != numProducers*tasksPerProducer {
		t.Fatalf("expected %d queued tasks, got %d", numProducers*tasksPerProducer, gov.Queue().Len())
	}

	// Concurrent workers admitting and releasing
	var workersWg sync.WaitGroup
	const numWorkers = 8
	for w := 0; w < numWorkers; w++ {
		workersWg.Add(1)
		go func() {
			defer workersWg.Done()
			for {
				task, verdict, err := gov.TryAdmit()
				if err != nil || !verdict.Admitted || task == nil {
					if gov.Queue().Len() == 0 {
						return
					}
					time.Sleep(1 * time.Millisecond)
					continue
				}
				// Simulate worker turn
				time.Sleep(200 * time.Microsecond)
				gov.Release(task)
			}
		}()
	}

	workersWg.Wait()
	if gov.InFlight() != 0 {
		t.Fatalf("expected 0 in-flight at end, got %d", gov.InFlight())
	}
}

func BenchmarkQueueSaturationBackpressure(b *testing.B) {
	gov := NewGovernor(GovernorConfig{BaseConcurrency: 16, QueueCapacity: 512})

	// Fill queue to capacity
	for i := 0; i < 512; i++ {
		_ = gov.Submit(&Task{ID: fmt.Sprintf("warm-%d", i), Priority: abi.ThreadPriorityP2Batch})
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		err := gov.Submit(&Task{ID: "overflow", Priority: abi.ThreadPriorityP1Interactive})
		if err == nil {
			b.Fatalf("expected ErrQueueFull")
		}
	}
}

func TestGovernorEdgeCasesAndReset(t *testing.T) {
	// Defaults constructor
	gov := NewGovernor(GovernorConfig{})
	if gov.EffectiveConcurrency() != DefaultMaxWorkers {
		t.Fatalf("expected default %d, got %d", DefaultMaxWorkers, gov.EffectiveConcurrency())
	}
	if gov.Queue().Capacity() != abi.MaxQueueCapacity {
		t.Fatalf("expected capacity %d, got %d", abi.MaxQueueCapacity, gov.Queue().Capacity())
	}

	// String() on gates
	for _, g := range []AdmissionGate{GateWorkerConcurrency, GateHostEnvelope, GateProviderHeadroom, GateLaneClearance, AdmissionGate(99)} {
		if g.String() == "" {
			t.Fatalf("empty gate string")
		}
	}

	// Empty queue TryAdmit
	tEmpty, vEmpty, err := gov.TryAdmit()
	if err != nil || tEmpty != nil || vEmpty.Admitted {
		t.Fatalf("expected nil on empty queue")
	}

	// Priority queue edge cases
	pq := NewPriorityQueue(-1)
	if pq.Capacity() != abi.MaxQueueCapacity {
		t.Fatalf("expected capacity %d", abi.MaxQueueCapacity)
	}
	if err := pq.Enqueue(nil); err != nil {
		t.Fatalf("unexpected err on nil task")
	}
	_ = pq.Enqueue(&Task{ID: "inv", Priority: abi.ThreadPriority(99)})
	if pq.LenPriority(abi.ThreadPriority(99)) != 0 {
		t.Fatalf("expected 0 for invalid priority")
	}
	if pq.LenPriority(abi.ThreadPriorityP2Batch) != 1 {
		t.Fatalf("expected invalid priority to map to P2 batch")
	}
	if task, ok := pq.Peek(true); !ok || task.ID != "inv" {
		t.Fatalf("expected peek task 'inv'")
	}
	if pq.Remove("non-existent") {
		t.Fatalf("expected false on non-existent task")
	}
	if !pq.Remove("inv") {
		t.Fatalf("expected true on existing task")
	}
	if task, ok := pq.Dequeue(true); ok || task != nil {
		t.Fatalf("expected false on empty dequeue")
	}

	// Provider headroom throttled toggle
	headroom := NewMemoryProviderHeadroom()
	headroom.SetThrottled("acc", true)
	if !headroom.IsThrottled("acc") {
		t.Fatalf("expected throttled")
	}
	headroom.SetThrottled("acc", false)
	if headroom.IsThrottled("acc") {
		t.Fatalf("expected unthrottled")
	}
	if !headroom.HasTokenBudget("acc", 0) {
		t.Fatalf("expected true for 0 tokens")
	}
	if !headroom.HasTokenBudget("unregistered", 1000) {
		t.Fatalf("expected true for unregistered account")
	}

	// Early warning moderate CPU (76%)
	gov.UpdateTelemetry(HostTelemetry{CPUPct: 76.0})
	if !gov.IsP3Paused() {
		t.Fatalf("expected P3 paused under early warning CPU")
	}
	if gov.PacingMS() != DefaultPacingModerateMS {
		t.Fatalf("expected moderate pacing %d, got %d", DefaultPacingModerateMS, gov.PacingMS())
	}

	// Reset
	gov.Reset()
	if gov.EffectiveConcurrency() != DefaultMaxWorkers || gov.IsP3Paused() || gov.PacingMS() != 0 {
		t.Fatalf("reset failed to restore state")
	}
}

func TestGovernorReleaseClearsLeaseOnEmptyTaskID(t *testing.T) {
	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 2,
	})

	task := &Task{
		ID:       "",
		Priority: abi.ThreadPriorityP1Interactive,
		Lane:     "lane-test-empty-id",
	}

	if err := gov.Queue().Enqueue(task); err != nil {
		t.Fatalf("failed to enqueue task: %v", err)
	}

	admitted, verdict, err := gov.TryAdmit()
	if err != nil {
		t.Fatalf("unexpected error on TryAdmit: %v", err)
	}
	if !verdict.Admitted || admitted == nil {
		t.Fatalf("expected task to be admitted, got verdict: %+v", verdict)
	}
	if admitted.ID == "" {
		t.Fatalf("expected admitted task to have a non-empty ID assigned")
	}
	if len(gov.heldLeases) != 1 {
		t.Fatalf("expected 1 held lease, got %d", len(gov.heldLeases))
	}

	gov.Release(admitted)
	if len(gov.heldLeases) != 0 {
		t.Fatalf("expected 0 held leases after release, got %d (leak detected)", len(gov.heldLeases))
	}
}

func TestHeadOfLineBlockingCandidateAdmission(t *testing.T) {
	gov := NewGovernor(GovernorConfig{
		BaseConcurrency: 2,
	})

	// Task 1 holds lease on lane-a
	task1 := &Task{
		ID:       "task-1",
		Priority: abi.ThreadPriorityP1Interactive,
		Lane:     "lane-a",
		Tree:     []string{"internal/lane_a/*"},
	}
	if err := gov.Queue().Enqueue(task1); err != nil {
		t.Fatalf("failed to enqueue task1: %v", err)
	}
	admitted1, verdict1, err := gov.TryAdmit()
	if err != nil || !verdict1.Admitted || admitted1 == nil {
		t.Fatalf("failed to admit task1: %v, verdict: %+v", err, verdict1)
	}

	// Task 2 targets lane-a (blocked by Gate 4 conflict)
	task2 := &Task{
		ID:       "task-2-blocked",
		Priority: abi.ThreadPriorityP1Interactive,
		Lane:     "lane-a",
		Tree:     []string{"internal/lane_a/*"},
	}
	// Task 3 targets lane-b (disjoint lane, should be runnable and admitted despite task2 being head of queue)
	task3 := &Task{
		ID:       "task-3-runnable",
		Priority: abi.ThreadPriorityP1Interactive,
		Lane:     "lane-b",
		Tree:     []string{"internal/lane_b/*"},
	}

	if err := gov.Queue().Enqueue(task2); err != nil {
		t.Fatalf("failed to enqueue task2: %v", err)
	}
	if err := gov.Queue().Enqueue(task3); err != nil {
		t.Fatalf("failed to enqueue task3: %v", err)
	}

	// TryAdmit should skip blocked task2 and admit task3 without head-of-line blocking
	admitted3, verdict3, err := gov.TryAdmit()
	if err != nil {
		t.Fatalf("unexpected error on TryAdmit: %v", err)
	}
	if !verdict3.Admitted || admitted3 == nil {
		t.Fatalf("expected task3 to be admitted past blocked head task2, got verdict: %+v", verdict3)
	}
	if admitted3.ID != "task-3-runnable" {
		t.Fatalf("expected admitted task to be task-3-runnable, got %s", admitted3.ID)
	}

	// Blocked task2 should still be in the queue
	if gov.Queue().Len() != 1 {
		t.Fatalf("expected 1 task remaining in queue, got %d", gov.Queue().Len())
	}
	if peeked, ok := gov.Queue().Peek(true); !ok || peeked.ID != "task-2-blocked" {
		t.Fatalf("expected task-2-blocked to remain at head of queue, got %+v", peeked)
	}

	// Now release task1, freeing lane-a
	gov.Release(admitted1)

	// TryAdmit should now admit task2
	admitted2, verdict2, err := gov.TryAdmit()
	if err != nil {
		t.Fatalf("unexpected error on TryAdmit: %v", err)
	}
	if !verdict2.Admitted || admitted2 == nil || admitted2.ID != "task-2-blocked" {
		t.Fatalf("expected task-2-blocked to be admitted after release of task1, got %+v", verdict2)
	}
	if gov.Queue().Len() != 0 {
		t.Fatalf("expected queue to be empty, got %d", gov.Queue().Len())
	}
}
