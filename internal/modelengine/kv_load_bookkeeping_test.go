package modelengine

import (
	"errors"
	"fmt"
	"testing"
)

// These tests pin the O(1)-amortized KV-load bookkeeping contract adapted from
// vLLM PR #65831 (vllm/v1/core/block_pool.py and vllm/v1/core/kv_cache_manager.py
// at 518c624a1c731cfc6c73369ced6351cb03754ddf, Apache-2.0): N load transitions
// cost O(N) total pool touches, independent of pool size, and per-step
// aggregate reads cost zero pool touches.

func checkAggregates(t *testing.T, bk *KVLoadBookkeeper, want KVLoadAggregates) {
	t.Helper()
	if got := bk.Aggregates(); got != want {
		t.Fatalf("Aggregates() = %+v, want %+v", got, want)
	}
}

// TestKVLoadBookkeepingLifecycleValues drives one request through
// start -> promote -> finish -> release and pins the cached aggregate values
// and the exact pool-touch count at each phase.
func TestKVLoadBookkeepingLifecycleValues(t *testing.T) {
	bk, err := NewKVLoadBookkeeper(64)
	if err != nil {
		t.Fatalf("NewKVLoadBookkeeper(64) error = %v", err)
	}

	const req = "req-1"
	if err := bk.StartLoad(req, 16); err != nil {
		t.Fatalf("StartLoad error = %v", err)
	}
	checkAggregates(t, bk, KVLoadAggregates{LoadingRequests: 1, LoadingBlocks: 16, FreeBlocks: 48})
	if got := bk.PoolTouches(); got != 16 {
		t.Fatalf("PoolTouches after start = %d, want 16", got)
	}

	if err := bk.PromoteLoad(req); err != nil {
		t.Fatalf("PromoteLoad error = %v", err)
	}
	checkAggregates(t, bk, KVLoadAggregates{RunningRequests: 1, RunningBlocks: 16, FreeBlocks: 48})
	if got := bk.PoolTouches(); got != 16 {
		t.Fatalf("PoolTouches after promote = %d, want 16 (phase moves touch no pool slots)", got)
	}

	if err := bk.FinishLoad(req); err != nil {
		t.Fatalf("FinishLoad error = %v", err)
	}
	checkAggregates(t, bk, KVLoadAggregates{FinishedRequests: 1, FinishedBlocks: 16, FreeBlocks: 48})
	if got := bk.PoolTouches(); got != 16 {
		t.Fatalf("PoolTouches after finish = %d, want 16", got)
	}

	if err := bk.ReleaseLoad(req); err != nil {
		t.Fatalf("ReleaseLoad error = %v", err)
	}
	checkAggregates(t, bk, KVLoadAggregates{FreeBlocks: 64})
	if got := bk.PoolTouches(); got != 32 {
		t.Fatalf("PoolTouches after release = %d, want 32 (16 alloc + 16 free)", got)
	}
	if err := bk.VerifyConsistency(); err != nil {
		t.Fatalf("VerifyConsistency error = %v", err)
	}
	if got := bk.PoolTouches(); got != 32+64 {
		t.Fatalf("PoolTouches after audit = %d, want 96 (audit visits every slot once)", got)
	}
}

// TestKVLoadBookkeepingTransitionsCostLinearInTransitions pins the borrowed
// contract: N load lifecycles cost exactly 2*N*K pool touches (K alloc + K
// free per lifecycle), identically for pool sizes that differ by 128x, and
// interleaved per-step aggregate reads add zero touches. A scheduler that
// recomputed aggregates by walking the pool per transition or per step would
// cost O(N*P) and fail this test.
func TestKVLoadBookkeepingTransitionsCostLinearInTransitions(t *testing.T) {
	const (
		n     = 64 // load lifecycles
		k     = 8  // blocks per load
		steps = 256
	)
	wantTouches := int64(2 * n * k)
	for _, poolSize := range []int{n * k, 65536} {
		t.Run(fmt.Sprintf("pool-%d", poolSize), func(t *testing.T) {
			bk, err := NewKVLoadBookkeeper(poolSize)
			if err != nil {
				t.Fatalf("NewKVLoadBookkeeper(%d) error = %v", poolSize, err)
			}
			ids := make([]string, n)
			for i := range ids {
				ids[i] = fmt.Sprintf("req-%d", i)
			}

			// Start every load; the small pool is now saturated.
			for _, id := range ids {
				if err := bk.StartLoad(id, k); err != nil {
					t.Fatalf("StartLoad(%q) error = %v", id, err)
				}
			}
			checkAggregates(t, bk, KVLoadAggregates{
				LoadingRequests: n, LoadingBlocks: n * k, FreeBlocks: poolSize - n*k,
			})

			// Per-step schedule reads while saturated: zero pool touches.
			before := bk.PoolTouches()
			for s := 0; s < steps; s++ {
				bk.Aggregates()
			}
			if got := bk.PoolTouches(); got != before {
				t.Fatalf("per-step aggregate reads added %d pool touches, want 0", got-before)
			}

			for _, id := range ids {
				if err := bk.PromoteLoad(id); err != nil {
					t.Fatalf("PromoteLoad(%q) error = %v", id, err)
				}
			}
			checkAggregates(t, bk, KVLoadAggregates{
				RunningRequests: n, RunningBlocks: n * k, FreeBlocks: poolSize - n*k,
			})

			for _, id := range ids {
				if err := bk.FinishLoad(id); err != nil {
					t.Fatalf("FinishLoad(%q) error = %v", id, err)
				}
			}
			checkAggregates(t, bk, KVLoadAggregates{
				FinishedRequests: n, FinishedBlocks: n * k, FreeBlocks: poolSize - n*k,
			})

			for _, id := range ids {
				if err := bk.ReleaseLoad(id); err != nil {
					t.Fatalf("ReleaseLoad(%q) error = %v", id, err)
				}
			}
			checkAggregates(t, bk, KVLoadAggregates{FreeBlocks: poolSize})

			if got := bk.PoolTouches(); got != wantTouches {
				t.Fatalf("PoolTouches after %d lifecycles = %d, want %d (O(N) total, independent of pool size)",
					n, got, wantTouches)
			}

			// Control: the consistency audit is the only operation that visits
			// every pool slot, and it does so exactly once.
			if err := bk.VerifyConsistency(); err != nil {
				t.Fatalf("VerifyConsistency error = %v", err)
			}
			if got := bk.PoolTouches(); got != wantTouches+int64(poolSize) {
				t.Fatalf("PoolTouches after audit = %d, want %d", got, wantTouches+int64(poolSize))
			}
		})
	}
}

// TestKVLoadBookkeepingMixedPhasesAggregates pins the running-sum aggregates
// while distinct requests sit in different lifecycle phases at once.
func TestKVLoadBookkeepingMixedPhasesAggregates(t *testing.T) {
	bk, err := NewKVLoadBookkeeper(128)
	if err != nil {
		t.Fatalf("NewKVLoadBookkeeper(128) error = %v", err)
	}

	if err := bk.StartLoad("a", 32); err != nil {
		t.Fatalf("StartLoad(a) error = %v", err)
	}
	if err := bk.StartLoad("b", 16); err != nil {
		t.Fatalf("StartLoad(b) error = %v", err)
	}
	if err := bk.StartLoad("c", 8); err != nil {
		t.Fatalf("StartLoad(c) error = %v", err)
	}
	if err := bk.PromoteLoad("a"); err != nil {
		t.Fatalf("PromoteLoad(a) error = %v", err)
	}
	if err := bk.PromoteLoad("c"); err != nil {
		t.Fatalf("PromoteLoad(c) error = %v", err)
	}
	if err := bk.FinishLoad("c"); err != nil {
		t.Fatalf("FinishLoad(c) error = %v", err)
	}

	checkAggregates(t, bk, KVLoadAggregates{
		LoadingRequests:  1,
		RunningRequests:  1,
		FinishedRequests: 1,
		LoadingBlocks:    16,
		RunningBlocks:    32,
		FinishedBlocks:   8,
		FreeBlocks:       72,
	})
	if err := bk.VerifyConsistency(); err != nil {
		t.Fatalf("VerifyConsistency error = %v", err)
	}
}

// TestKVLoadBookkeepingSentinelErrors pins the closed error contract with
// errors.Is; no assertion relies on error prose.
func TestKVLoadBookkeepingSentinelErrors(t *testing.T) {
	t.Run("invalid pool size", func(t *testing.T) {
		if _, err := NewKVLoadBookkeeper(0); !errors.Is(err, ErrKVLoadInvalidPoolSize) {
			t.Fatalf("err = %v, want ErrKVLoadInvalidPoolSize", err)
		}
	})

	bk, err := NewKVLoadBookkeeper(16)
	if err != nil {
		t.Fatalf("NewKVLoadBookkeeper(16) error = %v", err)
	}

	t.Run("invalid block count", func(t *testing.T) {
		if err := bk.StartLoad("r", 0); !errors.Is(err, ErrKVLoadInvalidBlocks) {
			t.Fatalf("err = %v, want ErrKVLoadInvalidBlocks", err)
		}
	})

	t.Run("double start", func(t *testing.T) {
		if err := bk.StartLoad("r", 4); err != nil {
			t.Fatalf("first StartLoad error = %v", err)
		}
		if err := bk.StartLoad("r", 1); !errors.Is(err, ErrKVLoadAlreadyActive) {
			t.Fatalf("err = %v, want ErrKVLoadAlreadyActive", err)
		}
	})

	t.Run("promote unknown request", func(t *testing.T) {
		if err := bk.PromoteLoad("missing"); !errors.Is(err, ErrKVLoadNotFound) {
			t.Fatalf("err = %v, want ErrKVLoadNotFound", err)
		}
	})

	t.Run("finish before promote", func(t *testing.T) {
		if err := bk.FinishLoad("r"); !errors.Is(err, ErrKVLoadWrongPhase) {
			t.Fatalf("err = %v, want ErrKVLoadWrongPhase", err)
		}
	})

	t.Run("promote finished load", func(t *testing.T) {
		if err := bk.PromoteLoad("r"); err != nil {
			t.Fatalf("PromoteLoad error = %v", err)
		}
		if err := bk.FinishLoad("r"); err != nil {
			t.Fatalf("FinishLoad error = %v", err)
		}
		if err := bk.PromoteLoad("r"); !errors.Is(err, ErrKVLoadWrongPhase) {
			t.Fatalf("err = %v, want ErrKVLoadWrongPhase", err)
		}
	})

	t.Run("release unknown request", func(t *testing.T) {
		if err := bk.ReleaseLoad("missing"); !errors.Is(err, ErrKVLoadNotFound) {
			t.Fatalf("err = %v, want ErrKVLoadNotFound", err)
		}
	})

	t.Run("capacity exceeded", func(t *testing.T) {
		if err := bk.StartLoad("fill", 12); err != nil {
			t.Fatalf("StartLoad(fill) error = %v", err)
		}
		if err := bk.StartLoad("over", 1); !errors.Is(err, ErrKVLoadCapacityExceeded) {
			t.Fatalf("err = %v, want ErrKVLoadCapacityExceeded", err)
		}
	})
}
