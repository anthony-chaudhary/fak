package supervise

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

type mockSuperviseStore struct {
	mu        sync.Mutex
	contracts map[string]leaseref.ContractRecord
}

func newMockSuperviseStore() *mockSuperviseStore {
	return &mockSuperviseStore{
		contracts: make(map[string]leaseref.ContractRecord),
	}
}

func (m *mockSuperviseStore) LiveContracts(ctx context.Context, now ...time.Time) ([]leaseref.ContractRecord, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var list []leaseref.ContractRecord
	for _, rec := range m.contracts {
		list = append(list, rec)
	}
	return list, nil
}

func (m *mockSuperviseStore) UpdateContractState(ctx context.Context, ticketID, holder string, newState leaseref.ContractState, tokensUsed int64, now ...time.Time) (leaseref.ContractRecord, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.contracts[ticketID]
	if !ok {
		rec = leaseref.ContractRecord{TicketID: ticketID}
	}
	rec.State = newState
	if tokensUsed > 0 {
		rec.TokensUsed = tokensUsed
	}
	m.contracts[ticketID] = rec
	return rec, nil
}

type mockDeallocator struct {
	mu          sync.Mutex
	deallocated []string
}

func (m *mockDeallocator) Deallocate(ctx context.Context, ticketID string, keepBranch bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deallocated = append(m.deallocated, ticketID)
	return nil
}

type mockTreeReleaser struct {
	mu       sync.Mutex
	sessions []string
}

func (m *mockTreeReleaser) ReleaseSessionLeases(ctx context.Context, sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions = append(m.sessions, sessionID)
	return nil
}

// TestSupervisor_NeverReapsActiveHeartbeats verifies that a contract actively sending
// heartbeats (renewals) is never reaped.
func TestSupervisor_NeverReapsActiveHeartbeats(t *testing.T) {
	store := newMockSuperviseStore()
	dealloc := &mockDeallocator{}
	now := time.Now()

	// Contract acquired 2 hours ago, but renewed 10 seconds ago with 60s TTL
	store.contracts["ticket-heartbeat"] = leaseref.ContractRecord{
		TicketID:    "ticket-heartbeat",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Add(-2 * time.Hour).Unix(),
		RenewedAt:   now.Add(-10 * time.Second).Unix(),
		TTLSeconds:  60,
		TokenBudget: 10000,
		TokensUsed:  2000,
	}

	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: dealloc,
	})

	report, err := sup.Tick(context.Background(), now)
	if err != nil {
		t.Fatalf("tick failed: %v", err)
	}

	if report.ReapedCount != 0 {
		t.Fatalf("expected 0 reaped contracts for active heartbeat, got %d", report.ReapedCount)
	}
	if report.ActiveCount != 1 {
		t.Fatalf("expected 1 active contract, got %d", report.ActiveCount)
	}

	// Verify contract state remains EXECUTING
	store.mu.Lock()
	if store.contracts["ticket-heartbeat"].State != leaseref.ContractStateExecuting {
		t.Fatalf("expected state EXECUTING, got %v", store.contracts["ticket-heartbeat"].State)
	}
	store.mu.Unlock()
}

// TestSupervisor_ReapsExpiredContract verifies that an expired contract without renewal
// is reaped with reason TIMEOUT and a failure diagnosis is written.
func TestSupervisor_ReapsExpiredContract(t *testing.T) {
	store := newMockSuperviseStore()
	dealloc := &mockDeallocator{}
	releaser := &mockTreeReleaser{}
	tmpDir := t.TempDir()
	now := time.Now()

	// Contract acquired 100s ago with 60s TTL, never renewed
	store.contracts["ticket-expired"] = leaseref.ContractRecord{
		TicketID:    "ticket-expired",
		SessionID:   "session-xyz",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Add(-100 * time.Second).Unix(),
		RenewedAt:   0,
		TTLSeconds:  60,
		TokenBudget: 10000,
		TokensUsed:  500,
	}

	sup := NewSupervisor(SupervisorConfig{
		RepoRoot:            tmpDir,
		Store:               store,
		WorktreeDeallocator: dealloc,
		TreeLeaseReleaser:   releaser,
	})

	report, err := sup.Tick(context.Background(), now)
	if err != nil {
		t.Fatalf("tick failed: %v", err)
	}

	if report.ReapedCount != 1 {
		t.Fatalf("expected 1 reaped contract, got %d", report.ReapedCount)
	}
	if report.Reaped[0].Reason != ReasonTimeout {
		t.Fatalf("expected reason TIMEOUT, got %s", report.Reaped[0].Reason)
	}

	// Verify contract state became FAILED
	store.mu.Lock()
	if store.contracts["ticket-expired"].State != leaseref.ContractStateFailed {
		t.Fatalf("expected state FAILED, got %v", store.contracts["ticket-expired"].State)
	}
	store.mu.Unlock()

	// Verify deallocator was called
	dealloc.mu.Lock()
	if len(dealloc.deallocated) != 1 || dealloc.deallocated[0] != "ticket-expired" {
		t.Fatalf("expected deallocation of ticket-expired, got %+v", dealloc.deallocated)
	}
	dealloc.mu.Unlock()

	// Verify session tree leases were released
	releaser.mu.Lock()
	if len(releaser.sessions) != 1 || releaser.sessions[0] != "session-xyz" {
		t.Fatalf("expected tree lease release for session-xyz, got %+v", releaser.sessions)
	}
	releaser.mu.Unlock()

	// Verify diagnosis file was written
	diagFile := filepath.Join(tmpDir, ".dispatch-runs", "failures", "ticket-expired.json")
	data, err := os.ReadFile(diagFile)
	if err != nil {
		t.Fatalf("expected failure diagnosis file at %s: %v", diagFile, err)
	}
	var diag FailureDiagnosis
	if err := json.Unmarshal(data, &diag); err != nil {
		t.Fatalf("failed to parse diagnosis file: %v", err)
	}
	if diag.Reason != ReasonTimeout || diag.TicketID != "ticket-expired" {
		t.Fatalf("unexpected diagnosis content: %+v", diag)
	}
}

// TestSupervisor_ReapsBudgetExhausted verifies that exceeding token budget triggers reaping.
func TestSupervisor_ReapsBudgetExhausted(t *testing.T) {
	store := newMockSuperviseStore()
	dealloc := &mockDeallocator{}
	now := time.Now()

	store.contracts["ticket-overbudget"] = leaseref.ContractRecord{
		TicketID:    "ticket-overbudget",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Unix(),
		TTLSeconds:  3600,
		TokenBudget: 5000,
		TokensUsed:  5001, // Over budget!
	}

	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: dealloc,
	})

	report, err := sup.Tick(context.Background(), now)
	if err != nil {
		t.Fatalf("tick failed: %v", err)
	}

	if report.ReapedCount != 1 {
		t.Fatalf("expected 1 reaped contract, got %d", report.ReapedCount)
	}
	if report.Reaped[0].Reason != ReasonBudgetExhausted {
		t.Fatalf("expected reason BUDGET_EXHAUSTED, got %s", report.Reaped[0].Reason)
	}
}

// TestSupervisor_StuckLoopDetection verifies detecting >2 consecutive identical turns
// with zero file changes.
func TestSupervisor_StuckLoopDetection(t *testing.T) {
	store := newMockSuperviseStore()
	dealloc := &mockDeallocator{}
	now := time.Now()

	store.contracts["ticket-stuck"] = leaseref.ContractRecord{
		TicketID:    "ticket-stuck",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Unix(),
		TTLSeconds:  3600,
		TokenBudget: 50000,
		TokensUsed:  1000,
	}

	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: dealloc,
		StuckLoopThreshold:  3,
	})

	// Turn 1: tool call X, 0 file changes
	sup.RecordTurn("ticket-stuck", "bash:pytest tests/unit", 0)
	rep1, _ := sup.Tick(context.Background(), now)
	if rep1.ReapedCount != 0 {
		t.Fatalf("should not reap on 1 turn")
	}

	// Turn 2: identical tool call X, 0 file changes
	sup.RecordTurn("ticket-stuck", "bash:pytest tests/unit", 0)
	rep2, _ := sup.Tick(context.Background(), now)
	if rep2.ReapedCount != 0 {
		t.Fatalf("should not reap on 2 turns")
	}

	// Turn 3: identical tool call X, 0 file changes -> STUCK LOOP!
	sup.RecordTurn("ticket-stuck", "bash:pytest tests/unit", 0)
	rep3, _ := sup.Tick(context.Background(), now)
	if rep3.ReapedCount != 1 {
		t.Fatalf("expected 1 reaped contract on 3 identical turns, got %d", rep3.ReapedCount)
	}
	if rep3.Reaped[0].Reason != ReasonStuckLoop {
		t.Fatalf("expected reason STUCK_LOOP, got %s", rep3.Reaped[0].Reason)
	}
}

// TestSupervisor_LoopWithFileChangesNotStuck verifies that iterations with file changes
// are NOT flagged as stuck loops.
func TestSupervisor_LoopWithFileChangesNotStuck(t *testing.T) {
	store := newMockSuperviseStore()
	dealloc := &mockDeallocator{}
	now := time.Now()

	store.contracts["ticket-progressing"] = leaseref.ContractRecord{
		TicketID:    "ticket-progressing",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Unix(),
		TTLSeconds:  3600,
		TokenBudget: 50000,
		TokensUsed:  1000,
	}

	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: dealloc,
		StuckLoopThreshold:  3,
	})

	// 3 turns with same tool call, but files are changing each turn!
	sup.RecordTurn("ticket-progressing", "bash:go test ./...", 1)
	sup.RecordTurn("ticket-progressing", "bash:go test ./...", 2)
	sup.RecordTurn("ticket-progressing", "bash:go test ./...", 1)

	rep, _ := sup.Tick(context.Background(), now)
	if rep.ReapedCount != 0 {
		t.Fatalf("should not reap progressing worker with file changes")
	}
}
