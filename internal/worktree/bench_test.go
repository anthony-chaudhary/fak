package worktree

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

var benchSinkEnv []string

func BenchmarkWorktreeContextEnvList(b *testing.B) {
	wtCtx := &WorktreeContext{
		TicketID: "fak-12345",
		Path:     "C:/repo/.worktrees/fak-12345",
		Branch:   "worker/fak-12345",
		Env: map[string]string{
			"GIT_WORK_TREE": "C:/repo/.worktrees/fak-12345",
			"GIT_DIR":       "C:/repo/.git/worktrees/fak-12345",
			"FAK_LANE":      "worktree",
			"FAK_TICKET":    "fak-12345",
		},
		AllocatedAt: time.Now(),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEnv = wtCtx.EnvList()
	}
}

func BenchmarkWorktreeAllocateAndRelease(b *testing.B) {
	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		return "", "", nil
	}
	tmpDir := b.TempDir()
	mgr := NewManager(tmpDir, WithRunner(mockRunner))
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ticketID := fmt.Sprintf("bench-ticket-%d", i)
		wt, err := mgr.Allocate(ctx, ticketID, "main")
		if err != nil {
			b.Fatalf("allocate failed: %v", err)
		}
		if wt == nil {
			b.Fatalf("expected non-nil WorktreeContext")
		}
		if err := mgr.Release(ctx, ticketID); err != nil {
			b.Fatalf("release failed: %v", err)
		}
	}
}

func BenchmarkJanitorSweep(b *testing.B) {
	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		return "", "", nil
	}
	tmpDir := b.TempDir()
	worktreesDir := filepath.Join(tmpDir, ".worktrees")
	_ = os.MkdirAll(filepath.Join(worktreesDir, "ticket-active-1"), 0755)
	_ = os.MkdirAll(filepath.Join(worktreesDir, "ticket-active-2"), 0755)
	_ = os.MkdirAll(filepath.Join(worktreesDir, "ticket-orphan-1"), 0755)

	mgr := NewManager(tmpDir, WithRunner(mockRunner))
	ctx := context.Background()

	contracts := []leaseref.ContractRecord{
		{
			TicketID:   "active-1",
			State:      leaseref.ContractStateExecuting,
			AcquiredAt: time.Now().Unix(),
			TTLSeconds: 3600,
		},
		{
			TicketID:   "active-2",
			State:      leaseref.ContractStateExecuting,
			AcquiredAt: time.Now().Unix(),
			TTLSeconds: 3600,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := mgr.Sweep(ctx, contracts)
		if err != nil {
			b.Fatalf("sweep failed: %v", err)
		}
		if report == nil {
			b.Fatalf("expected non-nil report")
		}
	}
}

func BenchmarkAllocateThroughput_WithAutoSweep(b *testing.B) {
	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		if len(args) > 0 && args[0] == "rev-parse" {
			return "c001cafe\n", "", nil
		}
		return "", "", nil
	}
	tmpDir := b.TempDir()
	cfg := &AutoSweepConfig{
		Threshold:      10,
		DebounceWindow: 5 * time.Millisecond,
		ContractProvider: func(ctx context.Context) ([]leaseref.ContractRecord, error) {
			return nil, nil
		},
	}
	mgr := NewManager(tmpDir, WithRunner(mockRunner), WithAutoSweep(cfg))
	defer mgr.Close()
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ticketID := fmt.Sprintf("bench-auto-%d", i)
		wt, err := mgr.Allocate(ctx, ticketID, "c001cafe")
		if err != nil {
			b.Fatalf("allocate failed: %v", err)
		}
		if wt == nil {
			b.Fatalf("expected non-nil WorktreeContext")
		}
	}
}

func BenchmarkParallelAllocateAndAutoSweep(b *testing.B) {
	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		if len(args) > 0 && args[0] == "rev-parse" {
			return "c001cafe\n", "", nil
		}
		return "", "", nil
	}
	tmpDir := b.TempDir()
	cfg := &AutoSweepConfig{
		Threshold:      10,
		DebounceWindow: 5 * time.Millisecond,
		ContractProvider: func(ctx context.Context) ([]leaseref.ContractRecord, error) {
			return nil, nil
		},
	}
	mgr := NewManager(tmpDir, WithRunner(mockRunner), WithAutoSweep(cfg))
	defer mgr.Close()
	ctx := context.Background()

	var counter uint64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			id := atomic.AddUint64(&counter, 1)
			ticketID := fmt.Sprintf("par-%d", id)
			wt, err := mgr.Allocate(ctx, ticketID, "c001cafe")
			if err != nil {
				b.Errorf("allocate failed: %v", err)
				return
			}
			if wt == nil {
				b.Errorf("expected non-nil WorktreeContext")
				return
			}
			if err := mgr.Release(ctx, ticketID); err != nil {
				b.Errorf("release failed: %v", err)
				return
			}
		}
	})
}
