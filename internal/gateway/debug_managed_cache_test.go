package gateway

import "testing"

// The /debug/vars managed_cache block is the posture sibling of the #1849 cache_attribution
// owner split (#2190): it must make the managed-cache 1h TTL-upgrade lever legible LIVE, and
// crucially keep an ACTIVE-but-inert session (lever on, zero upgrades) VISIBLE rather than
// collapsing it into the same "no block" a passive session emits.
func TestManagedCacheVars(t *testing.T) {
	t.Run("active with upgrades reports posture and count", func(t *testing.T) {
		got := managedCacheVars(true, AdjudicationSummary{
			CacheTTLUpgraded:       3,
			CacheTTLUpgradeReasons: map[string]uint64{"volatile_head": 2},
		})
		if got == nil {
			t.Fatal("active session must emit a block")
		}
		if !got.Active || got.Inert {
			t.Fatalf("active-with-upgrades: Active=%t Inert=%t, want Active=true Inert=false", got.Active, got.Inert)
		}
		if got.Upgraded != 3 {
			t.Fatalf("Upgraded = %d, want 3", got.Upgraded)
		}
		if got.Reasons["volatile_head"] != 2 {
			t.Fatalf("Reasons did not carry through: %v", got.Reasons)
		}
	})

	t.Run("active but zero upgrades is a visible inert block", func(t *testing.T) {
		got := managedCacheVars(true, AdjudicationSummary{
			CacheTTLUpgradeReasons: map[string]uint64{"no_stable_breakpoint": 5},
		})
		if got == nil {
			t.Fatal("ACTIVE-but-inert must stay visible, not collapse to a nil block")
		}
		if !got.Active || !got.Inert {
			t.Fatalf("active-zero: Active=%t Inert=%t, want both true", got.Active, got.Inert)
		}
		if got.Upgraded != 0 {
			t.Fatalf("Upgraded = %d, want 0", got.Upgraded)
		}
	})

	t.Run("passive cold session stays quiet", func(t *testing.T) {
		if got := managedCacheVars(false, AdjudicationSummary{}); got != nil {
			t.Fatalf("passive cold session should emit no block, got %+v", got)
		}
	})

	t.Run("passive but observed still reports (defensive)", func(t *testing.T) {
		got := managedCacheVars(false, AdjudicationSummary{CacheTTLUpgraded: 1})
		if got == nil {
			t.Fatal("observed activity must not be dropped even when the lever reads off")
		}
		if got.Active || got.Inert {
			t.Fatalf("passive-observed: Active=%t Inert=%t, want both false", got.Active, got.Inert)
		}
	})
}
