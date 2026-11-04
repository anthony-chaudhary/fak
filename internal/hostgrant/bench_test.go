// Benchmarks for the durable grant ledger's hot paths: every mutation
// serializes through Path+".lock" into a JSON load, mutate, and
// temp+fsync+rename save, while the idempotent retry returns the existing
// grant without rewriting state.
package hostgrant_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostgrant"
)

var (
	benchStartedAt = time.Date(2026, time.September, 23, 8, 0, 0, 0, time.UTC)
	benchCost      = hostgrant.Vector{AgentSteps: 4, Processes: 1, Compilers: 1, ModelCalls: 2}
)

func benchStore(b *testing.B) hostgrant.Store {
	b.Helper()
	return hostgrant.Store{
		Path:     filepath.Join(b.TempDir(), "grants.json"),
		Capacity: hostgrant.Vector{AgentSteps: 10_000, Processes: 64, Compilers: 8, ModelCalls: 1000},
	}
}

// Steady-state admission/teardown cycle: one fresh grant admitted and
// released per iteration, so the ledger stays empty between cycles.
func BenchmarkTryAcquireNewGrantCycle(b *testing.B) {
	b.ReportAllocs()
	store := benchStore(b)
	ctx := context.Background()
	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("bench-cycle-%d", i)
		grant, err := store.TryAcquire(ctx, hostgrant.Request{
			ID:    id,
			Owner: hostgrant.Owner{ID: "bench-agent", PID: 101, StartedAt: benchStartedAt},
			Cost:  benchCost,
			TTL:   time.Minute,
		})
		if err != nil {
			b.Fatalf("TryAcquire(%q): %v", id, err)
		}
		if err := store.Release(ctx, grant); err != nil {
			b.Fatalf("Release(%q): %v", id, err)
		}
	}
}

// Restart/idempotency fast path: re-issuing the identical request returns
// the existing grant without a state rewrite.
func BenchmarkTryAcquireIdempotentRetry(b *testing.B) {
	b.ReportAllocs()
	store := benchStore(b)
	ctx := context.Background()
	req := hostgrant.Request{
		ID:    "bench-retry",
		Owner: hostgrant.Owner{ID: "bench-agent", PID: 101, StartedAt: benchStartedAt},
		Cost:  benchCost,
		TTL:   time.Minute,
	}
	if _, err := store.TryAcquire(ctx, req); err != nil {
		b.Fatalf("seed TryAcquire: %v", err)
	}
	for i := 0; i < b.N; i++ {
		if _, err := store.TryAcquire(ctx, req); err != nil {
			b.Fatalf("idempotent retry: %v", err)
		}
	}
}

// Ownership handoff: each transfer moves the grant to a fresh owner and
// advances the fencing generation, forcing a full state rewrite.
func BenchmarkTransferToNewOwner(b *testing.B) {
	b.ReportAllocs()
	store := benchStore(b)
	ctx := context.Background()
	grant, err := store.TryAcquire(ctx, hostgrant.Request{
		ID:    "bench-transfer",
		Owner: hostgrant.Owner{ID: "bench-agent", PID: 101, StartedAt: benchStartedAt},
		Cost:  benchCost,
		TTL:   time.Minute,
	})
	if err != nil {
		b.Fatalf("seed TryAcquire: %v", err)
	}
	for i := 0; i < b.N; i++ {
		grant, err = store.Transfer(ctx, grant, hostgrant.Owner{ID: "bench-agent", PID: 1000 + i, StartedAt: benchStartedAt})
		if err != nil {
			b.Fatalf("Transfer iteration %d: %v", i, err)
		}
	}
}

// Churn across a populated ledger: eight active grants stay resident while
// one slot is released and re-acquired under the same ID each iteration.
func BenchmarkReleaseReacquireChurn(b *testing.B) {
	b.ReportAllocs()
	const slots = 8
	store := benchStore(b)
	ctx := context.Background()
	requests := make([]hostgrant.Request, slots)
	grants := make([]hostgrant.Grant, slots)
	for i := range requests {
		requests[i] = hostgrant.Request{
			ID:    fmt.Sprintf("bench-churn-%d", i),
			Owner: hostgrant.Owner{ID: "bench-agent", PID: 100 + i, StartedAt: benchStartedAt},
			Cost:  benchCost,
			TTL:   time.Minute,
		}
	}
	for i := range grants {
		var err error
		grants[i], err = store.TryAcquire(ctx, requests[i])
		if err != nil {
			b.Fatalf("seed TryAcquire(%q): %v", requests[i].ID, err)
		}
	}
	for i := 0; i < b.N; i++ {
		slot := i % slots
		if err := store.Release(ctx, grants[slot]); err != nil {
			b.Fatalf("Release(%q): %v", requests[slot].ID, err)
		}
		var err error
		grants[slot], err = store.TryAcquire(ctx, requests[slot])
		if err != nil {
			b.Fatalf("re-acquire(%q): %v", requests[slot].ID, err)
		}
	}
}
