package gateway

import "testing"

// TestStickyPrefixRoute pins the slack-tolerant sticky prefix routing borrowed
// from microsoft/vidur's TolerantStickyLOPUncachedGlobalScheduler (ADAPT —
// re-expressed in Go, no copied code; source:
// vidur/scheduler/global_scheduler/tolerant_sticky_lop_uncached_global_scheduler.py
// @abae7f63aa857300f5cdc6f5e0d27860cd24721b, MIT): a prefix's home instance keeps
// its traffic while its load stays within tolerance x the fleet's minimum load,
// and sheds to the least-loaded instance past the bound, migrating the sticky
// record with it. The closed contracts are exact (instance, hit) pairs and load
// counts — no error prose, since the router reports placement, not errors.
func TestStickyPrefixRoute(t *testing.T) {
	t.Run("single_replica_precondition", func(t *testing.T) {
		// The slack only bites above one replica: with a single instance the
		// home IS the minimum load, so the bound can never be exceeded and no
		// shed is possible. Recorded here rather than claimed as a win.
		r := NewSlackTolerantFleetCacheRouter(1, 4, 1.5)
		if got, hit := r.Route("A"); got != 0 || hit {
			t.Fatalf("first Route = (%d, %v), want (0, false)", got, hit)
		}
		for i := 0; i < 4; i++ {
			if got, hit := r.Route("A"); got != 0 || !hit {
				t.Fatalf("repeat %d Route = (%d, %v), want (0, true) — a single-replica fleet never sheds", i+1, got, hit)
			}
		}
	})

	t.Run("sticks_within_slack_inclusive_bound", func(t *testing.T) {
		r := NewSlackTolerantFleetCacheRouter(3, 4, 1.5)
		if got, hit := r.Route("A"); got != 0 || hit {
			t.Fatalf("cold Route = (%d, %v), want (0, false)", got, hit)
		}
		// Balanced external in-flight pressure on all three instances.
		r.load = []int{4, 4, 4}
		if got, hit := r.Route("A"); got != 0 || !hit {
			t.Fatalf("balanced Route = (%d, %v), want (0, true) — stickiness within slack", got, hit)
		}
		// Exactly at the bound (6 <= 1.5 x 4): stickiness is preserved, mirroring
		// the upstream inclusive comparison.
		r.load = []int{6, 4, 4}
		if got, hit := r.Route("A"); got != 0 || !hit {
			t.Fatalf("at-bound Route = (%d, %v), want (0, true) — the bound is inclusive", got, hit)
		}
	})

	t.Run("sheds_beyond_slack_and_migrates_home", func(t *testing.T) {
		r := NewSlackTolerantFleetCacheRouter(3, 4, 1.5)
		if got, hit := r.Route("A"); got != 0 || hit {
			t.Fatalf("cold Route = (%d, %v), want (0, false)", got, hit)
		}
		// Home is instance 0; drive it past the slack bound.
		r.load = []int{7, 4, 4} // 7 > 1.5 x 4 -> shed
		if got, hit := r.Route("A"); got != 1 || hit {
			t.Fatalf("shed Route = (%d, %v), want (1, false) — least-loaded other instance, cold", got, hit)
		}
		if got := r.home["A"]; got != 1 {
			t.Fatalf("sticky record after shed = %d, want 1 (home migrates with the session)", got)
		}
		// The migrated home is now within slack and sticks again.
		r.load = []int{7, 5, 4} // 5 <= 1.5 x 4 -> stick
		if got, hit := r.Route("A"); got != 1 || !hit {
			t.Fatalf("post-migration Route = (%d, %v), want (1, true)", got, hit)
		}
		// Past the bound again: the session follows the cool side to instance 2.
		r.load = []int{7, 7, 4} // 7 > 1.5 x 4 -> shed off instance 1
		if got, hit := r.Route("A"); got != 2 || hit {
			t.Fatalf("second shed Route = (%d, %v), want (2, false)", got, hit)
		}
		if got := r.home["A"]; got != 2 {
			t.Fatalf("sticky record after second shed = %d, want 2", got)
		}
	})

	t.Run("shed_to_holder_reports_hit", func(t *testing.T) {
		// Ping-pong: shedding back onto an instance that still holds the prefix
		// is a cache hit even though the sticky record moved.
		r := NewSlackTolerantFleetCacheRouter(2, 4, 1.5)
		if got, hit := r.Route("A"); got != 0 || hit {
			t.Fatalf("cold Route = (%d, %v), want (0, false)", got, hit)
		}
		r.load = []int{5, 0} // 5 > 1.5 x 0 -> shed to instance 1
		if got, hit := r.Route("A"); got != 1 || hit {
			t.Fatalf("shed Route = (%d, %v), want (1, false)", got, hit)
		}
		r.load = []int{1, 5} // 5 > 1.5 x 1 -> shed back to instance 0, which still holds A
		if got, hit := r.Route("A"); got != 0 || !hit {
			t.Fatalf("ping-pong Route = (%d, %v), want (0, true) — residency survives the migration", got, hit)
		}
	})

	t.Run("rehomes_when_home_evicts_prefix", func(t *testing.T) {
		// Capacity 1 per instance: admitting C evicts A from its home, so the
		// stale sticky record must not keep routing to an empty slot.
		r := NewSlackTolerantFleetCacheRouter(2, 1, 1.5)
		if got, hit := r.Route("A"); got != 0 || hit {
			t.Fatalf("A Route = (%d, %v), want (0, false)", got, hit)
		}
		if got, hit := r.Route("B"); got != 1 || hit {
			t.Fatalf("B Route = (%d, %v), want (1, false)", got, hit)
		}
		if got, hit := r.Route("C"); got != 0 || hit {
			t.Fatalf("C Route = (%d, %v), want (0, false) — evicts A from instance 0", got, hit)
		}
		if got, hit := r.Route("A"); got != 1 || hit {
			t.Fatalf("re-homed A Route = (%d, %v), want (1, false) — evicted home loses stickiness", got, hit)
		}
		if got := r.home["A"]; got != 1 {
			t.Fatalf("sticky record after eviction = %d, want 1", got)
		}
	})

	t.Run("rigid_default_unchanged", func(t *testing.T) {
		// The default constructor stays rigid: an imbalanced holder still
		// absorbs every turn of its prefix, which is what the committed
		// hit-rate witness (TestKVAwareFleetRoutingHitRate) pins.
		r := NewFleetCacheRouter(3, 4)
		if got, hit := r.Route("A"); got != 0 || hit {
			t.Fatalf("cold Route = (%d, %v), want (0, false)", got, hit)
		}
		r.load = []int{9, 0, 0}
		if got, hit := r.Route("A"); got != 0 || !hit {
			t.Fatalf("rigid Route = (%d, %v), want (0, true) — no slack armed by default", got, hit)
		}
	})
}
