package gym

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// BenchmarkGymReceiptVerification measures the performance of schema and integrity verification on simulation receipts.
func BenchmarkGymReceiptVerification(b *testing.B) {
	receipt := &GymReceipt{
		Schema:           GymReceiptSchema,
		ScenarioID:       "bench-scenario-1",
		Timestamp:        time.Now(),
		TurnsExecuted:    5,
		TotalToolCalls:   12,
		ElisionsObserved: 2,
		RestoresObserved: 1,
		MultiTurnPass:    true,
		Outcome:          OutcomePass,
		TranscriptDigest: "digest-sha256-abc12345",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ok, reason := receipt.VerifyReceipt("bench-scenario-1")
		if !ok {
			b.Fatalf("VerifyReceipt failed: %s", reason)
		}
	}
}

// BenchmarkGymArenaLifecycle measures the initialization, branching, and destruction of an ephemeral arena.
func BenchmarkGymArenaLifecycle(b *testing.B) {
	ctx := context.Background()
	baseDir := b.TempDir()
	anchor := filepath.Join(baseDir, "anchor.txt")
	if err := os.WriteFile(anchor, []byte("anchor content"), 0644); err != nil {
		b.Fatalf("failed to write anchor: %v", err)
	}

	cfg := Config{
		BaseDir:       baseDir,
		WorkspaceName: "bench-arena",
		PinnedPTY:     true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		arena, err := Create(ctx, cfg)
		if err != nil {
			b.Fatalf("Create failed: %v", err)
		}
		branch, err := arena.Fork(ctx, "bench-branch")
		if err != nil {
			_ = arena.Destroy()
			b.Fatalf("Fork failed: %v", err)
		}
		_ = branch.Destroy()
		_ = arena.Destroy()
	}
}
