package gateway

import (
	"fmt"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestReplicaPendingPrefixDisabledDefault(t *testing.T) {
	def := DefaultPendingPrefixConfig()
	if def.TTL != 0 {
		t.Fatalf("DefaultPendingPrefixConfig().TTL = %s, want 0 (disabled by default)", def.TTL)
	}

	pending, err := NewPendingPrefixIndex(def)
	if err != nil {
		t.Fatalf("NewPendingPrefixIndex(default): %v", err)
	}
	if pending.Enabled() {
		t.Fatal("default PendingPrefixIndex must report Enabled() == false")
	}

	prefix := []string{"sys", "cold-burst"}
	pending.Record("w0", prefix)
	if pending.Len() != 0 {
		t.Fatalf("disabled index recorded stake: Len=%d", pending.Len())
	}
	if got := pending.PendingOverlap("w0", prefix, 0); got != 0 {
		t.Fatalf("disabled PendingOverlap = %d, want 0", got)
	}

	confirmed := NewPrefixResidencyIndex(8)
	confirmed.Observe("w1", []string{"sys"})
	if got := pending.CombinedOverlap(confirmed, "w1", prefix, 0); got != 1 {
		t.Fatalf("CombinedOverlap with disabled pending = %d, want confirmed 1", got)
	}

	// With TTL=0, a cold burst spreads across idle replicas instead of herding onto w0.
	base := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
	pol := NewPendingPrefixPolicy(base, pending)
	candidates := []PlannerReplica{{Name: "w0"}, {Name: "w1"}, {Name: "w2"}}
	loads := map[string]int{}
	loadFn := func(name string) int { return loads[name] }

	seen := map[string]int{}
	for i := 0; i < 6; i++ {
		chosen, ok := pol.Pick(candidates, prefix, loadFn)
		if !ok {
			t.Fatalf("Pick(%d) failed", i)
		}
		loads[chosen.Name]++
		seen[chosen.Name]++
	}
	if len(seen) != 3 {
		t.Fatalf("TTL=0 disabled default should spread cold burst across all 3 replicas, got %v", seen)
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPendingPrefixBurstConvergesUpToMaxFollowers(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	pending, err := NewPendingPrefixIndex(PendingPrefixConfig{
		TTL:          5 * time.Second,
		MaxFollowers: 3,
		ConsultBelow: 10,
		MaxRecords:   64,
		Now:          func() time.Time { return now },
	})
	if err != nil {
		t.Fatalf("NewPendingPrefixIndex: %v", err)
	}
	if !pending.Enabled() {
		t.Fatal("expected Enabled() == true for TTL > 0")
	}

	base := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), SkewThreshold{AbsLoad: 100, RelLoad: 100})
	pol := NewPendingPrefixPolicy(base, pending)
	candidates := []PlannerReplica{{Name: "w0"}, {Name: "w1"}, {Name: "w2"}}
	prefix := []string{"agent-scaffold", "tools", "cold"}

	loads := map[string]int{}
	loadFn := func(name string) int { return loads[name] }

	// 1st request places the cold stake; next MaxFollowers (3) requests follow that
	// same replica -> total 4 consecutive picks on the first staker before spilling.
	var picks []string
	for i := 0; i < 5; i++ {
		chosen, ok := pol.Pick(candidates, prefix, loadFn)
		if !ok {
			t.Fatalf("Pick(%d) failed", i)
		}
		loads[chosen.Name]++
		picks = append(picks, chosen.Name)
	}

	firstOwner := picks[0]
	for i := 1; i <= 3; i++ {
		if picks[i] != firstOwner {
			t.Fatalf("follower %d routed to %q, want stake owner %q (picks=%v)", i, picks[i], firstOwner, picks)
		}
	}
	if picks[4] == firstOwner {
		t.Fatalf("5th request exceeded MaxFollowers=3 and still routed to %q (picks=%v)", firstOwner, picks)
	}
	if loads[firstOwner] != 4 {
		t.Fatalf("first owner load = %d, want 4 (1 staker + 3 followers)", loads[firstOwner])
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPendingPrefixTTLExpiryReleasesStake(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	pending, err := NewPendingPrefixIndex(PendingPrefixConfig{
		TTL:          2 * time.Second,
		MaxFollowers: 5,
		ConsultBelow: 10,
		Now:          func() time.Time { return now },
	})
	if err != nil {
		t.Fatalf("NewPendingPrefixIndex: %v", err)
	}

	prefix := []string{"b1", "b2", "b3"}
	pending.Record("w0", prefix)
	if got := pending.PendingOverlap("w0", prefix, 0); got != 3 {
		t.Fatalf("active PendingOverlap = %d, want 3", got)
	}
	if got := pending.Len(); got != 1 {
		t.Fatalf("active Len = %d, want 1", got)
	}

	// Just before expiry: stake remains active.
	now = now.Add(1900 * time.Millisecond)
	if got := pending.PendingOverlap("w0", []string{"b1", "b2", "other"}, 0); got != 2 {
		t.Fatalf("prefix partial match before expiry = %d, want 2", got)
	}

	// Past TTL: stake expires and is released cleanly.
	now = now.Add(200 * time.Millisecond)
	if got := pending.PendingOverlap("w0", prefix, 0); got != 0 {
		t.Fatalf("expired PendingOverlap = %d, want 0", got)
	}
	if got := pending.Len(); got != 0 {
		t.Fatalf("expired Len = %d, want 0", got)
	}

	// Once expired, a cold request places on the least-loaded idle worker (w1) instead of busy w0.
	base := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
	pol := NewPendingPrefixPolicy(base, pending)
	candidates := []PlannerReplica{{Name: "w0"}, {Name: "w1"}}
	chosen, ok := pol.Pick(candidates, prefix, func(name string) int {
		if name == "w0" {
			return 2
		}
		return 0
	})
	if !ok || chosen.Name != "w1" {
		t.Fatalf("after TTL expiry expected idle w1, got %q (ok=%v)", chosen.Name, ok)
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPendingPrefixOverloadedOwnerSkipped(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	pending, err := NewPendingPrefixIndex(PendingPrefixConfig{
		TTL:          5 * time.Second,
		MaxFollowers: 8,
		ConsultBelow: 4,
		Now:          func() time.Time { return now },
	})
	if err != nil {
		t.Fatalf("NewPendingPrefixIndex: %v", err)
	}

	prefix := []string{"hot-prefix", "turn1"}
	pending.Record("w0", prefix)

	if got := pending.PendingOverlap("w0", prefix, 3); got != 2 {
		t.Fatalf("below ConsultBelow (3 < 4): PendingOverlap = %d, want 2", got)
	}
	if got := pending.PendingOverlap("w0", prefix, 4); got != 0 {
		t.Fatalf("at ConsultBelow (4 >= 4): PendingOverlap = %d, want 0 (skipped)", got)
	}

	base := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), SkewThreshold{AbsLoad: 100, RelLoad: 100})
	pol := NewPendingPrefixPolicy(base, pending)
	candidates := []PlannerReplica{{Name: "w0"}, {Name: "w1"}}

	// With w0 at load=3 (<4), pending stake keeps request on w0 even though w1 has load=0.
	chosen, ok := pol.Pick(candidates, prefix, func(name string) int {
		if name == "w0" {
			return 3
		}
		return 0
	})
	if !ok || chosen.Name != "w0" {
		t.Fatalf("owner below ConsultBelow should keep follower, got %q (ok=%v)", chosen.Name, ok)
	}

	// Once w0 reaches load=4 (>= ConsultBelow), pending is skipped and request goes to w1.
	chosen, ok = pol.Pick(candidates, prefix, func(name string) int {
		if name == "w0" {
			return 4
		}
		return 0
	})
	if !ok || chosen.Name != "w1" {
		t.Fatalf("overloaded owner (>= ConsultBelow) must be skipped for w1, got %q (ok=%v)", chosen.Name, ok)
	}
}

// fak-test:runtime fast est=1s
func TestReplicaPendingPrefixCombinesMaxAndBoundsAndValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  PendingPrefixConfig
	}{
		{name: "negative TTL", cfg: PendingPrefixConfig{TTL: -time.Second}},
		{name: "negative MaxFollowers", cfg: PendingPrefixConfig{MaxFollowers: -1}},
		{name: "negative ConsultBelow", cfg: PendingPrefixConfig{ConsultBelow: -1}},
		{name: "negative MaxRecords", cfg: PendingPrefixConfig{MaxRecords: -1}},
	} {
		if _, err := NewPendingPrefixIndex(tc.cfg); err == nil {
			t.Fatalf("%s: expected validation error", tc.name)
		}
	}

	now := time.Unix(1_800_000_000, 0)
	pending, err := NewPendingPrefixIndex(PendingPrefixConfig{
		TTL:          5 * time.Second,
		MaxFollowers: 2,
		ConsultBelow: 10,
		MaxRecords:   3,
		Now:          func() time.Time { return now },
	})
	if err != nil {
		t.Fatalf("NewPendingPrefixIndex: %v", err)
	}

	confirmed := NewPrefixResidencyIndex(8)
	confirmed.Observe("w0", []string{"p1", "p2", "p3"})
	pending.Record("w0", []string{"p1"})
	if got := pending.CombinedOverlap(confirmed, "w0", []string{"p1", "p2", "p3"}, 0); got != 3 {
		t.Fatalf("CombinedOverlap should take max(confirmed=3, pending=1) = %d", got)
	}

	pending.Record("w1", []string{"q1", "q2", "q3", "q4"})
	if got := pending.CombinedOverlap(confirmed, "w1", []string{"q1", "q2", "q3", "q4"}, 0); got != 4 {
		t.Fatalf("CombinedOverlap should take max(confirmed=0, pending=4) = %d", got)
	}

	// MaxRecords=3 bounds total stakes cleanly.
	for i := 0; i < 10; i++ {
		pending.Record("w0", []string{fmt.Sprintf("k-%d", i)})
	}
	if got := pending.Len(); got > 3 {
		t.Fatalf("pending stakes exceeded MaxRecords=3: len=%d", got)
	}
}
