package radixkv

import (
	"fmt"
	"math"
	"testing"
)

// This covers expiry reads of valid signed-domain descriptors at unsigned tree
// clocks. It does not qualify admitting new anchors beyond MaxInt64 or wrapping
// the tree clock from MaxUint64 back to zero.
// fak-test:justify why=regression when=changed:internal/radixkv/**
// fak-test:runtime fast est=10ms lane=default
func TestRetentionUnsignedTreeClock(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name          string
		admitted, ttl int64
		clock         uint64
		expired       bool
	}{
		{"signed_maximum", math.MaxInt64 - 1, 2, math.MaxInt64, false},
		{"unsigned_endpoint", math.MaxInt64 - 1, 2, uint64(math.MaxInt64) + 1, false},
		{"unsigned_after_endpoint", math.MaxInt64 - 1, 2, uint64(math.MaxInt64) + 2, true},
		{"expired_before_signed_boundary", 0, 1, uint64(math.MaxInt64) + 1, true},
		{"largest_sum_endpoint", math.MaxInt64, math.MaxInt64, math.MaxUint64 - 1, false},
		{"largest_sum_expired", math.MaxInt64, math.MaxInt64, math.MaxUint64, true},
		{"forever", 0, RetainForever, math.MaxUint64, false},
	} {
		for _, priority := range []int{75, 100} {
			t.Run(fmt.Sprintf("%s/%d", tc.name, priority), func(t *testing.T) {
				tree := New(0)
				add := func(token int) *node {
					boundary, matched := tree.Lookup([]int{token})
					leaf := tree.Insert(boundary, []int{token}[matched:], nil)
					tree.Done(leaf)
					return leaf
				}
				leaf, weaker := add(1), add(2)
				req := RetentionRequest{Priority: priority, Admitted: tc.admitted, TTL: tc.ttl}
				if err := tree.SetNodeRetention(leaf, req); err != nil {
					t.Fatalf("valid descriptor rejected: %v", err)
				}
				if err := tree.SetNodeRetention(weaker, RetentionRequest{Priority: DefaultRetentionPriority}); err != nil {
					t.Fatal(err)
				}
				tree.clock = tc.clock
				if got := req.expiredAtClock(tree.clock); got != tc.expired {
					t.Fatalf("unsigned expiry=%v want %v at %d", got, tc.expired, tree.clock)
				}
				if tree.clock <= math.MaxInt64 && req.Expired(int64(tree.clock)) != tc.expired {
					t.Fatal("signed public API disagrees within its representable domain")
				}
				wantTier := TierFromRetentionPriority(priority)
				wantVictim := weaker
				if tc.expired {
					wantTier, wantVictim = Tier3Probationary, leaf
				}
				if got := tree.NodeTier(leaf); got != wantTier {
					t.Fatalf("NodeTier=%v want %v", got, wantTier)
				}
				if got := tree.IsNodePinned(leaf); got != (wantTier == Tier0PinnedRoot) {
					t.Fatalf("IsNodePinned=%v at clock %d", got, tree.clock)
				}
				if seg, ok := tree.nodeTierSeg(leaf); !ok || seg != wantTier.Seg() {
					t.Fatalf("tree segment=%d present=%v want %d", seg, ok, wantTier.Seg())
				}
				if seg := nodeEvictionSeg(leaf, tree.clock); seg != wantTier.Seg() {
					t.Fatalf("strategy segment=%d want %d", seg, wantTier.Seg())
				}
				if got := tree.selectVictimLeaf(false); got != wantVictim {
					t.Fatalf("victim=%p want %p at clock %d", got, wantVictim, tree.clock)
				}
				if tc.expired {
					// Explicit permanent tier assignment remains authoritative.
					tree.clock = 0
					tree.SetNodeTier(leaf, Tier0PinnedRoot)
					tree.clock = tc.clock
					if !tree.IsNodePinned(leaf) || tree.NodeTier(leaf) != Tier0PinnedRoot || tree.selectVictimLeaf(false) != weaker {
						t.Fatal("unsigned expiry weakened an explicit permanent tier")
					}
				}
			})
		}
	}
}
