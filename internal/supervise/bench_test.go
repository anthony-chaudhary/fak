package supervise

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

type benchContractStore struct {
	contracts []leaseref.ContractRecord
}

func (s *benchContractStore) LiveContracts(ctx context.Context, now ...time.Time) ([]leaseref.ContractRecord, error) {
	cp := make([]leaseref.ContractRecord, len(s.contracts))
	copy(cp, s.contracts)
	return cp, nil
}

func (s *benchContractStore) UpdateContractState(ctx context.Context, ticketID, holder string, newState leaseref.ContractState, tokensUsed int64, now ...time.Time) (leaseref.ContractRecord, error) {
	return leaseref.ContractRecord{
		TicketID:   ticketID,
		Holder:     holder,
		State:      newState,
		TokensUsed: tokensUsed,
	}, nil
}

type benchDeallocator struct{}

func (benchDeallocator) Deallocate(ctx context.Context, ticketID string, keepBranch bool) error {
	return nil
}

type benchTreeReleaser struct{}

func (benchTreeReleaser) ReleaseSessionLeases(ctx context.Context, sessionID string) error {
	return nil
}

func BenchmarkCleanTicketID(b *testing.B) {
	cases := []string{
		"ticket-1234",
		"  ticket-5678  ",
		"issue#9999-fix-bug",
		"lane/supervise/leaf-1",
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		res := CleanTicketID(cases[i%len(cases)])
		if res == "" {
			b.Fatalf("CleanTicketID returned empty string")
		}
	}
}

func BenchmarkSupervisor_RecordTurn(b *testing.B) {
	sup := NewSupervisor(SupervisorConfig{
		StuckLoopThreshold: 3,
	})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sup.RecordTurn("ticket-record-turn", "bash:pytest tests/unit", i%2)
	}
}

func BenchmarkSupervisor_ClearTicket(b *testing.B) {
	sup := NewSupervisor(SupervisorConfig{
		StuckLoopThreshold: 3,
	})
	sup.RecordTurn("ticket-clear", "cmd", 0)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sup.ClearTicket("ticket-clear")
		sup.RecordTurn("ticket-clear", "cmd", 0)
	}
}

func BenchmarkSupervisor_CheckReap(b *testing.B) {
	now := time.Now()
	var contracts []leaseref.ContractRecord

	// 10 active contracts with recent heartbeats
	for j := 0; j < 10; j++ {
		contracts = append(contracts, leaseref.ContractRecord{
			TicketID:    fmt.Sprintf("active-%d", j),
			State:       leaseref.ContractStateExecuting,
			AcquiredAt:  now.Add(-10 * time.Minute).Unix(),
			RenewedAt:   now.Add(-5 * time.Second).Unix(),
			TTLSeconds:  60,
			TokenBudget: 100000,
			TokensUsed:  5000,
		})
	}

	// 5 expired contracts
	for j := 0; j < 5; j++ {
		contracts = append(contracts, leaseref.ContractRecord{
			TicketID:    fmt.Sprintf("expired-%d", j),
			State:       leaseref.ContractStateExecuting,
			AcquiredAt:  now.Add(-10 * time.Minute).Unix(),
			RenewedAt:   0,
			TTLSeconds:  60,
			TokenBudget: 100000,
			TokensUsed:  5000,
		})
	}

	// 5 budget-exhausted contracts
	for j := 0; j < 5; j++ {
		contracts = append(contracts, leaseref.ContractRecord{
			TicketID:    fmt.Sprintf("overbudget-%d", j),
			State:       leaseref.ContractStateExecuting,
			AcquiredAt:  now.Add(-10 * time.Second).Unix(),
			RenewedAt:   now.Add(-2 * time.Second).Unix(),
			TTLSeconds:  60,
			TokenBudget: 10000,
			TokensUsed:  10001,
		})
	}

	store := &benchContractStore{contracts: contracts}
	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: benchDeallocator{},
		TreeLeaseReleaser:   benchTreeReleaser{},
	})
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		rep, err := sup.Tick(ctx, now)
		if err != nil {
			b.Fatalf("Tick failed: %v", err)
		}
		if rep.InspectedCount != 20 || rep.ActiveCount != 10 || rep.ReapedCount != 10 {
			b.Fatalf("unexpected supervise report: %+v", rep)
		}
	}
}

func BenchmarkSupervisor_Tick_AllActive_10(b *testing.B) {
	now := time.Now()
	var contracts []leaseref.ContractRecord

	for j := 0; j < 10; j++ {
		contracts = append(contracts, leaseref.ContractRecord{
			TicketID:    fmt.Sprintf("active-%d", j),
			State:       leaseref.ContractStateExecuting,
			AcquiredAt:  now.Add(-10 * time.Minute).Unix(),
			RenewedAt:   now.Add(-5 * time.Second).Unix(),
			TTLSeconds:  60,
			TokenBudget: 100000,
			TokensUsed:  5000,
		})
	}

	store := &benchContractStore{contracts: contracts}
	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: benchDeallocator{},
		TreeLeaseReleaser:   benchTreeReleaser{},
	})
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		rep, err := sup.Tick(ctx, now)
		if err != nil {
			b.Fatalf("Tick failed: %v", err)
		}
		if rep.ActiveCount != 10 || rep.ReapedCount != 0 {
			b.Fatalf("unexpected supervise report: %+v", rep)
		}
	}
}

func BenchmarkSupervisor_Tick_AllActive_100(b *testing.B) {
	now := time.Now()
	var contracts []leaseref.ContractRecord

	for j := 0; j < 100; j++ {
		contracts = append(contracts, leaseref.ContractRecord{
			TicketID:    fmt.Sprintf("active-%d", j),
			State:       leaseref.ContractStateExecuting,
			AcquiredAt:  now.Add(-10 * time.Minute).Unix(),
			RenewedAt:   now.Add(-5 * time.Second).Unix(),
			TTLSeconds:  60,
			TokenBudget: 100000,
			TokensUsed:  5000,
		})
	}

	store := &benchContractStore{contracts: contracts}
	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: benchDeallocator{},
		TreeLeaseReleaser:   benchTreeReleaser{},
	})
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		rep, err := sup.Tick(ctx, now)
		if err != nil {
			b.Fatalf("Tick failed: %v", err)
		}
		if rep.ActiveCount != 100 || rep.ReapedCount != 0 {
			b.Fatalf("unexpected supervise report: %+v", rep)
		}
	}
}

func BenchmarkSupervisor_StuckLoopDetection(b *testing.B) {
	now := time.Now()
	store := &benchContractStore{
		contracts: []leaseref.ContractRecord{
			{
				TicketID:    "ticket-stuck",
				State:       leaseref.ContractStateExecuting,
				AcquiredAt:  now.Unix(),
				TTLSeconds:  3600,
				TokenBudget: 50000,
				TokensUsed:  1000,
			},
		},
	}
	sup := NewSupervisor(SupervisorConfig{
		Store:               store,
		WorktreeDeallocator: benchDeallocator{},
		TreeLeaseReleaser:   benchTreeReleaser{},
		StuckLoopThreshold:  3,
	})
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		sup.RecordTurn("ticket-stuck", "bash:pytest tests/unit", 0)
		sup.RecordTurn("ticket-stuck", "bash:pytest tests/unit", 0)
		sup.RecordTurn("ticket-stuck", "bash:pytest tests/unit", 0)

		rep, err := sup.Tick(ctx, now)
		if err != nil {
			b.Fatalf("Tick failed: %v", err)
		}
		if rep.ReapedCount != 1 || len(rep.Reaped) == 0 || rep.Reaped[0].Reason != ReasonStuckLoop {
			b.Fatalf("expected stuck loop reap, got rep=%+v", rep)
		}
	}
}

func BenchmarkReaper_Evaluate(b *testing.B) {
	ctx := context.Background()
	now := time.Now()
	rec := leaseref.ContractRecord{
		TicketID:    "ticket-bench-reap",
		SessionID:   "session-bench",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Add(-100 * time.Second).Unix(),
		TTLSeconds:  60,
		TokenBudget: 5000,
		TokensUsed:  6000,
	}
	opts := ReaperOptions{
		Holder:              "bench-reaper",
		Store:               &benchContractStore{},
		WorktreeDeallocator: benchDeallocator{},
		TreeLeaseReleaser:   benchTreeReleaser{},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		diag, err := ReapContract(ctx, rec, ReasonBudgetExhausted, "tokens exceeded budget", opts, now)
		if err != nil || diag == nil || diag.Reason != ReasonBudgetExhausted {
			b.Fatalf("ReapContract failed: err=%v diag=%v", err, diag)
		}
	}
}

func BenchmarkReaper_WithDiskPersistence(b *testing.B) {
	tmpDir := b.TempDir()
	ctx := context.Background()
	now := time.Now()
	rec := leaseref.ContractRecord{
		TicketID:    "ticket-bench-disk",
		SessionID:   "session-bench-disk",
		State:       leaseref.ContractStateExecuting,
		AcquiredAt:  now.Add(-100 * time.Second).Unix(),
		TTLSeconds:  60,
		TokenBudget: 5000,
		TokensUsed:  6000,
	}
	opts := ReaperOptions{
		RepoRoot:            tmpDir,
		Holder:              "bench-reaper",
		Store:               &benchContractStore{},
		WorktreeDeallocator: benchDeallocator{},
		TreeLeaseReleaser:   benchTreeReleaser{},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		diag, err := ReapContract(ctx, rec, ReasonTimeout, "contract lease expired", opts, now)
		if err != nil || diag == nil || diag.Reason != ReasonTimeout {
			b.Fatalf("ReapContract failed: err=%v diag=%v", err, diag)
		}
	}
}
