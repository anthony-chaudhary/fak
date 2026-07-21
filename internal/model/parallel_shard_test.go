package model

import (
	"os"
	"sync/atomic"
	"testing"
)

// TestParForShardCoverageExactlyOnce pins the invariant the node-sharded chunk cursor must not
// break: every index in [0,n) is executed EXACTLY once, no matter how the queue is split across
// shards. A double-run would double-accumulate a row (silently wrong logits); a missed run would
// leave a row at zero. The existing TestParallelMatchesSerial catches both indirectly through the
// numerics — this catches them directly, and sweeps the shard counts that box topology can produce
// (including counts that do not divide the chunk count evenly, where the span math is at risk).
func TestParForShardCoverageExactlyOnce(t *testing.T) {
	if numWorkers <= 1 {
		t.Skip("pool disabled (numWorkers<=1); sharding is a no-op")
	}
	for _, shards := range []string{"1", "2", "3", "5", "7", "8", "64"} {
		t.Run("shards="+shards, func(t *testing.T) {
			t.Setenv("FAK_PAR_SHARDS", shards)
			for _, n := range []int{1, 2, 7, 63, 64, 65, 1000, 4096} {
				counts := make([]int32, n)
				parFor(n, numWorkers, func(lo, hi int) {
					for i := lo; i < hi; i++ {
						atomic.AddInt32(&counts[i], 1)
					}
				})
				for i, c := range counts {
					if c != 1 {
						t.Fatalf("shards=%s n=%d: index %d ran %d times, want exactly 1", shards, n, i, c)
					}
				}
			}
		})
	}
}

// TestParShardCountResolution pins the shard-count policy: FAK_PAR_SHARDS pins it, it never
// exceeds the participant count (a shard with no participant would never drain), and it clamps to
// 1 — the original single-cursor behaviour — whenever the host reports no multi-node topology.
func TestParShardCountResolution(t *testing.T) {
	orig, had := os.LookupEnv("FAK_PAR_SHARDS")
	defer func() {
		if had {
			os.Setenv("FAK_PAR_SHARDS", orig)
		} else {
			os.Unsetenv("FAK_PAR_SHARDS")
		}
	}()

	os.Setenv("FAK_PAR_SHARDS", "8")
	if got := parShardCount(64); got != 8 {
		t.Fatalf("pinned shards = %d, want 8", got)
	}
	// Never more shards than participants.
	if got := parShardCount(4); got != 4 {
		t.Fatalf("shards with 4 participants = %d, want clamp to 4", got)
	}
	os.Setenv("FAK_PAR_SHARDS", "1")
	if got := parShardCount(64); got != 1 {
		t.Fatalf("shards=1 must reproduce the single-cursor path, got %d", got)
	}
	// A single participant can only ever have one shard.
	if got := parShardCount(1); got != 1 {
		t.Fatalf("single participant shards = %d, want 1", got)
	}
}
