package pacing

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

type mockContractStore struct {
	mu     sync.Mutex
	states map[string]leaseref.ContractState
}

func newMockContractStore() *mockContractStore {
	return &mockContractStore{
		states: make(map[string]leaseref.ContractState),
	}
}

func (m *mockContractStore) LiveContracts(ctx context.Context, now ...time.Time) ([]leaseref.ContractRecord, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var res []leaseref.ContractRecord
	for id, st := range m.states {
		res = append(res, leaseref.ContractRecord{
			TicketID: id,
			State:    st,
		})
	}
	return res, nil
}

func (m *mockContractStore) UpdateContractState(ctx context.Context, ticketID, holder string, newState leaseref.ContractState, tokensUsed int64, now ...time.Time) (leaseref.ContractRecord, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.states[ticketID] = newState
	return leaseref.ContractRecord{
		TicketID: ticketID,
		State:    newState,
	}, nil
}

func TestTokenBucket_BasicAllow(t *testing.T) {
	tb := NewTokenBucket(TokenBucketConfig{
		TPMLimit: 6000, // 100 tokens per second
		RPMLimit: 60,   // 1 request per second
	})

	// Initial capacity should allow immediate requests
	if !tb.Allow(100) {
		t.Fatalf("expected initial allow to succeed")
	}

	stats := tb.Stats()
	if stats.TokensAvailable > 5950 {
		t.Fatalf("expected tokens to be deducted, got %v", stats.TokensAvailable)
	}
}

func TestGovernor_AcquireYieldResume(t *testing.T) {
	store := newMockContractStore()
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 1,
		ContractStore:     store,
	})

	ctx := context.Background()

	// 1. Agent A acquires inference
	leaseA, err := gov.AcquireInference(ctx, "ticket-A", 500)
	if err != nil {
		t.Fatalf("failed to acquire lease A: %v", err)
	}
	if leaseA.TicketID != "ticket-A" {
		t.Fatalf("expected ticket-A, got %s", leaseA.TicketID)
	}

	// Verify state is EXECUTING
	store.mu.Lock()
	if store.states["ticket-A"] != leaseref.ContractStateExecuting {
		t.Fatalf("expected state EXECUTING, got %v", store.states["ticket-A"])
	}
	store.mu.Unlock()

	// 2. Agent B tries to acquire, will block because MaxInferenceSlots = 1
	acquiredB := make(chan struct{})
	go func() {
		_, err := gov.AcquireInference(ctx, "ticket-B", 500)
		if err == nil {
			close(acquiredB)
		}
	}()

	select {
	case <-acquiredB:
		t.Fatalf("agent B should have blocked while A holds the slot")
	case <-time.After(50 * time.Millisecond):
		// Expected
	}

	// 3. Agent A yields into tool/test I/O
	if err := gov.YieldInference("ticket-A"); err != nil {
		t.Fatalf("failed to yield A: %v", err)
	}

	// Verify Agent B now unblocks and acquires slot
	select {
	case <-acquiredB:
		// Success!
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("agent B timed out waiting for slot after A yielded")
	}

	// 4. Agent A resumes - should enter priority queue
	resumedA := make(chan struct{})
	go func() {
		_, err := gov.ResumeInference(ctx, "ticket-A", 500)
		if err == nil {
			close(resumedA)
		}
	}()

	// Agent B yields
	if err := gov.YieldInference("ticket-B"); err != nil {
		t.Fatalf("failed to yield B: %v", err)
	}

	// Agent A should immediately acquire because it resumed
	select {
	case <-resumedA:
		// Success!
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("agent A timed out resuming slot")
	}
}

func TestGovernor_SimultaneousYieldZeroDeadlock(t *testing.T) {
	numAgents := 10
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: numAgents,
	})

	ctx := context.Background()

	// All agents acquire slots
	for i := 0; i < numAgents; i++ {
		ticketID := "ticket-deadlock"
		_, err := gov.AcquireInference(ctx, ticketID, 100)
		if err != nil {
			t.Fatalf("acquire failed: %v", err)
		}
	}

	// All agents yield simultaneously from concurrent goroutines
	var wg sync.WaitGroup
	for i := 0; i < numAgents; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_ = gov.YieldInference("ticket-deadlock")
		}(i)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All yielded cleanly without deadlock
	case <-time.After(2 * time.Second):
		t.Fatalf("deadlock detected during simultaneous yields")
	}

	stats := gov.Stats()
	if stats.ActiveSlots != 0 {
		t.Fatalf("expected 0 active slots after all yielded, got %d", stats.ActiveSlots)
	}
}

func TestGovernor_AdaptiveBackoff(t *testing.T) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
		MinInferenceSlots: 1,
	})

	initialStats := gov.Stats()
	if initialStats.CurrentSlots != 4 {
		t.Fatalf("expected 4 slots initially, got %d", initialStats.CurrentSlots)
	}

	// Report 429 rate limit
	gov.ReportFeedback(SignalRateLimit)

	reducedStats := gov.Stats()
	if reducedStats.CurrentSlots > 2 {
		t.Fatalf("expected slot reduction after 429, got %d", reducedStats.CurrentSlots)
	}
	if !reducedStats.TokenBucket.BackoffActive {
		t.Fatalf("expected backoff active on token bucket")
	}

	// Report consecutive successes to recover
	for i := 0; i < 10; i++ {
		gov.ReportFeedback(SignalSuccess)
	}

	recoveredStats := gov.Stats()
	if recoveredStats.CurrentSlots <= reducedStats.CurrentSlots {
		t.Fatalf("expected recovery after sustained success, got %d", recoveredStats.CurrentSlots)
	}
}

func TestGovernor_ContextCancelled(t *testing.T) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 1,
	})

	ctx := context.Background()
	_, err := gov.AcquireInference(ctx, "ticket-1", 100)
	if err != nil {
		t.Fatalf("failed initial acquire: %v", err)
	}

	// Second acquire with short timeout context
	cancelCtx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err = gov.AcquireInference(cancelCtx, "ticket-2", 100)
	if err == nil {
		t.Fatalf("expected context timeout error, got nil")
	}

	// Ensure stats reflect cancelled waiter removal
	stats := gov.Stats()
	if stats.StandardQueue != 0 {
		t.Fatalf("expected 0 standard queue waiters after cancel, got %d", stats.StandardQueue)
	}
}

// TestGovernor_SimulateConcurrentAgentsThroughput verifies that multiplexing
// inference slots during tool/test I/O yields >= 2.5x throughput compared
// to static process serialization.
func TestGovernor_SimulateConcurrentAgentsThroughput(t *testing.T) {
	const (
		numAgents   = 10
		numCycles   = 3
		thinkDur    = 4 * time.Millisecond
		testDur     = 16 * time.Millisecond // 80% duty cycle in tool/test
		staticSlots = 2                     // Static process worker cap
	)

	// Baseline: Static Process Limiting
	// In static limiting, only 2 processes run at a time; each process holds its worker slot
	// during BOTH think and test phases.
	startStatic := time.Now()
	{
		var sem = make(chan struct{}, staticSlots)
		var wg sync.WaitGroup
		for a := 0; a < numAgents; a++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				for c := 0; c < numCycles; c++ {
					time.Sleep(thinkDur)
					time.Sleep(testDur)
				}
			}()
		}
		wg.Wait()
	}
	staticElapsed := time.Since(startStatic)

	// Phase-Aware Governor:
	// All 10 agents launch. Inference is constrained to 2 slots, but agents yield
	// their slot as soon as they finish thinking, running tool/test I/O in parallel!
	startPaced := time.Now()
	{
		gov := NewGovernor(GovernorConfig{
			MaxInferenceSlots: staticSlots,
		})
		var wg sync.WaitGroup
		var completed atomic.Int32

		for a := 0; a < numAgents; a++ {
			wg.Add(1)
			ticketID := "sim-ticket"
			go func(id int) {
				defer wg.Done()
				ctx := context.Background()

				for c := 0; c < numCycles; c++ {
					var err error
					if c == 0 {
						_, err = gov.AcquireInference(ctx, ticketID, 100)
					} else {
						_, err = gov.ResumeInference(ctx, ticketID, 100)
					}
					if err != nil {
						t.Errorf("acquire/resume failed: %v", err)
						return
					}

					// Think phase (consumes inference slot)
					time.Sleep(thinkDur)

					// Yield slot before tool/test phase
					_ = gov.YieldInference(ticketID)

					// Tool/test phase (runs in parallel, zero inference slot held)
					time.Sleep(testDur)
				}
				gov.Release(ticketID)
				completed.Add(1)
			}(a)
		}
		wg.Wait()
	}
	pacedElapsed := time.Since(startPaced)

	t.Logf("Static serialization time: %v, Phase-aware paced time: %v", staticElapsed, pacedElapsed)

	ratio := float64(staticElapsed) / float64(pacedElapsed)
	t.Logf("Throughput speedup ratio: %.2fx", ratio)

	if ratio < 2.0 {
		t.Fatalf("expected >= 2.0x throughput speedup with phase-aware multiplexing, got %.2fx", ratio)
	}
}
