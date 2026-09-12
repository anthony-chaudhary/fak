package ctxmmu_test

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestMetalKVCacheRestoration is the primary witness for issue #12308:
// feat(session): checkpointed Metal KV prefix cache restoration for resumed OpenCode sessions.
// Command: go test -v ./internal/ctxmmu -run TestMetalKVCacheRestoration
func TestMetalKVCacheRestoration(t *testing.T) {
	t.Run("RestorationLatencyUnder25ms", func(t *testing.T) {
		mmu := ctxmmu.New()
		pool := ctxmmu.NewSharedTokenPool()
		forkMgr := ctxmmu.NewForkManager()
		cowTable := ctxmmu.NewCOWPageTable()

		cm := ctxmmu.NewCheckpointManager(mmu, pool, forkMgr, cowTable)
		restorer := ctxmmu.NewMetalKVRestorer(cm, forkMgr, cowTable, pool, mmu)

		sessionID := "opencode-worker-session-001"
		const tokenCount = 1024

		// Set up session with 1024 committed tokens in the shared pool
		if err := pool.Reserve(sessionID, tokenCount); err != nil {
			t.Fatalf("pool.Reserve failed: %v", err)
		}
		if err := pool.Commit(sessionID, tokenCount); err != nil {
			t.Fatalf("pool.Commit failed: %v", err)
		}

		// Also register in COW table with physical blocks
		if _, err := cowTable.CreateSession(sessionID); err != nil {
			t.Fatalf("cowTable.CreateSession failed: %v", err)
		}
		cowTokens := make([]int, tokenCount)
		for i := 0; i < tokenCount; i++ {
			cowTokens[i] = 1000 + i
		}
		if err := cowTable.AppendTokens(sessionID, cowTokens); err != nil {
			t.Fatalf("cowTable.AppendTokens failed: %v", err)
		}

		// Step 1: Checkpoint the OpenCode session at a pause/tool boundary
		desc, err := restorer.CheckpointOpenCode(sessionID)
		if err != nil {
			t.Fatalf("CheckpointOpenCode failed: %v", err)
		}
		if desc == nil {
			t.Fatal("CheckpointOpenCode returned nil descriptor")
		}
		if desc.CommittedTokens != tokenCount {
			t.Fatalf("CommittedTokens = %d, want %d", desc.CommittedTokens, tokenCount)
		}

		// Step 2: Restore resident Metal KV cache in O(1) time
		result, err := restorer.RestoreMetalKV(sessionID)
		if err != nil {
			t.Fatalf("RestoreMetalKV failed: %v", err)
		}

		// Verify O(1) latency < 25ms threshold
		if result.ResumptionLatency > ctxmmu.MaxResumptionLatencyThreshold {
			t.Fatalf("ResumptionLatency %v exceeded threshold %v",
				result.ResumptionLatency, ctxmmu.MaxResumptionLatencyThreshold)
		}

		// Verify strictly zero physical bytes transferred on Apple Silicon UMA
		if result.PhysicalBytesTransferred != 0 {
			t.Fatalf("PhysicalBytesTransferred = %d, want 0 (zero-copy UMA)", result.PhysicalBytesTransferred)
		}

		// Verify restored token count matches
		if result.RestoredTokens != tokenCount {
			t.Fatalf("RestoredTokens = %d, want %d", result.RestoredTokens, tokenCount)
		}

		// Verify target architecture is set
		if result.TargetArch == "" {
			t.Fatal("TargetArch is empty")
		}

		t.Logf("Restoration succeeded in %v (<25ms deadline), speedup: %.2fx, tokens: %d",
			result.ResumptionLatency, result.SpeedupRatio, result.RestoredTokens)
	})

	t.Run("PerformanceAttributionStaysUnmeasuredWithoutBaseline", func(t *testing.T) {
		mmu := ctxmmu.New()
		pool := ctxmmu.NewSharedTokenPool()
		forkMgr := ctxmmu.NewForkManager()
		cowTable := ctxmmu.NewCOWPageTable()

		cm := ctxmmu.NewCheckpointManager(mmu, pool, forkMgr, cowTable)
		restorer := ctxmmu.NewMetalKVRestorer(cm, forkMgr, cowTable, pool, mmu)

		sessionID := "opencode-deep-context-session"
		const tokenCount = 4096 // 4k context tokens

		if err := pool.Reserve(sessionID, tokenCount); err != nil {
			t.Fatalf("pool.Reserve failed: %v", err)
		}
		if err := pool.Commit(sessionID, tokenCount); err != nil {
			t.Fatalf("pool.Commit failed: %v", err)
		}

		_, err := restorer.CheckpointOpenCode(sessionID)
		if err != nil {
			t.Fatalf("CheckpointOpenCode failed: %v", err)
		}

		result, err := restorer.RestoreMetalKV(sessionID)
		if err != nil {
			t.Fatalf("RestoreMetalKV failed: %v", err)
		}

		// No run-bound cold-prefill baseline was supplied, so the receipt must
		// stay explicitly unmeasured: never a fabricated 1.0x and never a ratio
		// derived from a historical constant.
		if result.SpeedupMeasured {
			t.Fatalf("SpeedupMeasured = true, want false without a run-bound baseline")
		}
		if result.SpeedupRatio != 0 || result.EstimatedColdPrefill != 0 {
			t.Fatalf("receipt = %+v, want explicitly unmeasured zero values", result)
		}

		// Functional restoration is independent of performance attribution.
		if result.RestoredTokens != tokenCount {
			t.Fatalf("RestoredTokens = %d, want %d", result.RestoredTokens, tokenCount)
		}
	})

	t.Run("CompactionCarryover", func(t *testing.T) {
		mmu := ctxmmu.New()
		pool := ctxmmu.NewSharedTokenPool()
		forkMgr := ctxmmu.NewForkManager()
		cowTable := ctxmmu.NewCOWPageTable()

		cm := ctxmmu.NewCheckpointManager(mmu, pool, forkMgr, cowTable)
		restorer := ctxmmu.NewMetalKVRestorer(cm, forkMgr, cowTable, pool, mmu)

		sessionID := "oc-compaction-boundary-42"
		const historicalTokens = 2048

		if _, err := cowTable.CreateSession(sessionID); err != nil {
			t.Fatalf("CreateSession failed: %v", err)
		}

		tokens := make([]int, historicalTokens)
		for i := range tokens {
			tokens[i] = 5000 + i
		}
		if err := cowTable.AppendTokens(sessionID, tokens); err != nil {
			t.Fatalf("AppendTokens failed: %v", err)
		}

		if err := pool.Reserve(sessionID, historicalTokens); err != nil {
			t.Fatalf("pool.Reserve failed: %v", err)
		}
		if err := pool.Commit(sessionID, historicalTokens); err != nil {
			t.Fatalf("pool.Commit failed: %v", err)
		}

		// Checkpoint historical context before compaction
		desc, err := restorer.CheckpointOpenCode(sessionID)
		if err != nil {
			t.Fatalf("CheckpointOpenCode failed: %v", err)
		}

		// Restore on new turn
		res, err := restorer.RestoreMetalKV(sessionID)
		if err != nil {
			t.Fatalf("RestoreMetalKV failed: %v", err)
		}

		if res.PrefixHashHex != desc.PrefixHashHex {
			t.Fatalf("PrefixHash mismatch: got %s, want %s", res.PrefixHashHex, desc.PrefixHashHex)
		}

		if res.RestoredTokens != historicalTokens {
			t.Fatalf("RestoredTokens = %d, want %d", res.RestoredTokens, historicalTokens)
		}
	})

	t.Run("FailClosedValidation", func(t *testing.T) {
		mmu := ctxmmu.New()
		pool := ctxmmu.NewSharedTokenPool()
		cm := ctxmmu.NewCheckpointManager(mmu, pool, nil, nil)
		restorer := ctxmmu.NewMetalKVRestorer(cm, nil, nil, pool, mmu)

		// 1. Empty session ID
		if _, err := restorer.RestoreMetalKV(""); !errors.Is(err, ctxmmu.ErrEmptySessionID) {
			t.Fatalf("expected ErrEmptySessionID, got %v", err)
		}
		if _, err := restorer.CheckpointOpenCode(""); !errors.Is(err, ctxmmu.ErrEmptySessionID) {
			t.Fatalf("expected ErrEmptySessionID, got %v", err)
		}

		// 2. Unknown session
		if _, err := restorer.RestoreMetalKV("non-existent-session"); !errors.Is(err, ctxmmu.ErrCheckpointNotFound) {
			t.Fatalf("expected ErrCheckpointNotFound, got %v", err)
		}

		// 3. Expired checkpoint
		expiredID := "opencode-expired-sess"
		_ = pool.Reserve(expiredID, 64)
		_ = pool.Commit(expiredID, 64)
		desc, err := restorer.CheckpointOpenCode(expiredID)
		if err != nil {
			t.Fatalf("CheckpointOpenCode failed: %v", err)
		}
		desc.SetExpiresAt(time.Now().Add(-1 * time.Minute)) // force expired

		if _, err := restorer.RestoreMetalKV(expiredID); !errors.Is(err, ctxmmu.ErrCheckpointExpired) {
			t.Fatalf("expected ErrCheckpointExpired, got %v", err)
		}

		// 4. Double restore
		freshID := "opencode-fresh-sess"
		_ = pool.Reserve(freshID, 64)
		_ = pool.Commit(freshID, 64)
		_, _ = restorer.CheckpointOpenCode(freshID)
		if _, err := restorer.RestoreMetalKV(freshID); err != nil {
			t.Fatalf("first restore failed: %v", err)
		}
		if _, err := restorer.RestoreMetalKV(freshID); !errors.Is(err, ctxmmu.ErrCheckpointAlreadyRestored) {
			t.Fatalf("expected ErrCheckpointAlreadyRestored on double restore, got %v", err)
		}
	})

	t.Run("ConcurrentSessions", func(t *testing.T) {
		mmu := ctxmmu.New()
		pool := ctxmmu.NewSharedTokenPool()
		cowTable := ctxmmu.NewCOWPageTable()
		cm := ctxmmu.NewCheckpointManager(mmu, pool, nil, cowTable)
		restorer := ctxmmu.NewMetalKVRestorer(cm, nil, cowTable, pool, mmu)

		const numWorkers = 8
		var wg sync.WaitGroup

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				sid := strings.ToLower(string(rune('a'+workerID))) + "-opencode-concurrent"
				tokCount := 128 * (workerID + 1)

				_ = pool.Reserve(sid, tokCount)
				_ = pool.Commit(sid, tokCount)

				_, err := restorer.CheckpointOpenCode(sid)
				if err != nil {
					t.Errorf("worker %d checkpoint failed: %v", workerID, err)
					return
				}

				res, err := restorer.RestoreMetalKV(sid)
				if err != nil {
					t.Errorf("worker %d restore failed: %v", workerID, err)
					return
				}

				if res.RestoredTokens != tokCount {
					t.Errorf("worker %d restored tokens = %d, want %d", workerID, res.RestoredTokens, tokCount)
				}
				if res.PhysicalBytesTransferred != 0 {
					t.Errorf("worker %d transferred bytes = %d, want 0", workerID, res.PhysicalBytesTransferred)
				}
			}(w)
		}

		wg.Wait()
	})

	t.Run("PrometheusMetricsExport", func(t *testing.T) {
		mmu := ctxmmu.New()
		pool := ctxmmu.NewSharedTokenPool()
		cm := ctxmmu.NewCheckpointManager(mmu, pool, nil, nil)
		restorer := ctxmmu.NewMetalKVRestorer(cm, nil, nil, pool, mmu)

		sid := "opencode-metrics-test"
		_ = pool.Reserve(sid, 512)
		_ = pool.Commit(sid, 512)

		_, _ = restorer.CheckpointOpenCode(sid)
		_, err := restorer.RestoreMetalKV(sid)
		if err != nil {
			t.Fatalf("RestoreMetalKV failed: %v", err)
		}

		metrics := restorer.Metrics()
		if metrics.RestorationsTotal != 1 {
			t.Fatalf("RestorationsTotal = %d, want 1", metrics.RestorationsTotal)
		}
		if metrics.TokensRestoredTotal != 512 {
			t.Fatalf("TokensRestoredTotal = %d, want 512", metrics.TokensRestoredTotal)
		}
		if metrics.EstimatedPrefillSecondsSaved != 0 {
			t.Fatalf("EstimatedPrefillSecondsSaved = %f, want 0 without a measured baseline", metrics.EstimatedPrefillSecondsSaved)
		}
		if metrics.MeasuredRestorationsTotal != 0 {
			t.Fatalf("MeasuredRestorationsTotal = %d, want 0 without a measured baseline", metrics.MeasuredRestorationsTotal)
		}

		prom := metrics.PrometheusMetrics()
		if !strings.Contains(prom, "fak_metal_kv_restorations_total 1") {
			t.Errorf("Prometheus output missing restorations total: %s", prom)
		}
		if !strings.Contains(prom, "fak_metal_kv_restoration_tokens_total 512") {
			t.Errorf("Prometheus output missing tokens total: %s", prom)
		}
		if !strings.Contains(prom, "fak_metal_kv_average_speedup_ratio") {
			t.Errorf("Prometheus output missing speedup ratio: %s", prom)
		}
	})

	t.Run("IsOpenCodeTarget", func(t *testing.T) {
		cases := []struct {
			id   string
			want bool
		}{
			{"opencode-session-123", true},
			{"oc-task-456", true},
			{"subagent-opencode-worker", true},
			{"OPENCODE-CAP", true},
			{"random-user-session", false},
			{"claude-code-session", false},
		}

		for _, tc := range cases {
			if got := ctxmmu.IsOpenCodeTarget(tc.id); got != tc.want {
				t.Errorf("IsOpenCodeTarget(%q) = %v, want %v", tc.id, got, tc.want)
			}
		}
	})
}
