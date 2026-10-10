package radixkv

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// This witnesses the snapshot eligibility phase for cost-aware admission with
// no token pressure. Structural strategy preparation and token-plane evictions
// are separate prediction contracts.
// fak-test:justify why=regression when=changed:internal/radixkv/**
// fak-test:runtime fast est=1s lane=default
func TestSnapshotAdmissionProjectedPhase(t *testing.T) {
	newTree := func(t *testing.T, budget int64) (*Tree, func([]int) *node) {
		t.Helper()
		tree := NewWithTierBudgetsAndEvictionPolicy(1000, budget, 0, EvictionCostAware)
		t.Cleanup(func() { tree.forEachNode(tree.releaseHotSnapshot) })
		seed := func(tokens []int) *node {
			t.Helper()
			boundary, matched := tree.Lookup(tokens)
			snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
			leaf, err := tree.InsertSnapshot(boundary, tokens[matched:], snap, make([]float32, 16))
			if err != nil {
				snap.Close()
				t.Fatal(err)
			}
			tree.Done(leaf)
			return leaf
		}
		return tree, seed
	}
	predict := func(t *testing.T, tree *Tree, boundary, resident *node, suffix []int, bytes int64) []admissionComparison {
		t.Helper()
		before, clock, refs := tree.Stats(), tree.clock, boundary.refs
		owner, generation := resident.snapshot, resident.recordGen
		comparisons := tree.snapshotAdmissionComparisons(boundary, suffix, bytes)
		if !reflect.DeepEqual(tree.Stats(), before) || tree.clock != clock || boundary.refs != refs ||
			resident.snapshot != owner || resident.recordGen != generation {
			t.Fatal("snapshot prediction mutated clock, lease, topology, accounting or ownership")
		}
		return comparisons
	}
	for _, mode := range []string{"released-boundary", "extra-lease", "permanent-pin", "computing"} {
		t.Run(mode, func(t *testing.T) {
			tree, seed := newTree(t, 64)
			resident := seed([]int{1})
			if mode == "permanent-pin" {
				tree.SetNodeTier(resident, Tier0PinnedRoot)
			}
			boundary, matched := tree.Lookup([]int{1, 2})
			if boundary != resident || matched != 1 || boundary.refs != 1 {
				t.Fatal("fixture did not lease the resident prefix")
			}
			if mode == "extra-lease" {
				second, _ := tree.Lookup([]int{1, 2})
				if second != boundary || boundary.refs != 2 {
					t.Fatal("fixture did not retain the independent second lease")
				}
			}
			if mode == "computing" {
				resident.SetState(NodeComputingPrefill)
			}
			// Isolate physical capacity from the independent frequency/value gate.
			tree.admissionSketch = nil
			comparisons := predict(t, tree, boundary, resident, []int{2}, 64)
			allowed := mode == "released-boundary"
			if len(comparisons) != 1 || (comparisons[0].forcedReason == "") != allowed {
				t.Fatalf("projected capacity=%+v allowed=%v", comparisons, allowed)
			}
			if allowed && (comparisons[0].victim.Leased || comparisons[0].victim.Bytes != 64 || comparisons[0].victim.Tokens != 1) {
				t.Fatal("prediction did not describe the released prefix snapshot")
			}
			clock, refs, owner := tree.clock, boundary.refs, resident.snapshot
			snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
			n, err := tree.InsertSnapshot(boundary, []int{2}, snap, make([]float32, 16))
			if err != nil {
				snap.Close()
				t.Fatal(err)
			}
			if tree.clock != clock+1 || tree.snapshotBytes != 64 {
				t.Fatal("insertion tick or bounded snapshot bytes changed")
			}
			if allowed {
				if n == boundary || n.snapshot != snap || resident.snapshot != nil || boundary.refs != 0 || n.refs != 1 {
					t.Fatal("actual insertion did not reclaim its released boundary")
				}
			} else if n != boundary || snap.Cache != nil || resident.snapshot != owner || boundary.refs != refs {
				t.Fatal("protected capacity bypass changed a lease or snapshot owner")
			}
			tree.Done(n)
			if mode == "extra-lease" {
				tree.Done(boundary)
			}
		})
	}
	for _, ttl := range []int64{1, 2} {
		name := "expires-on-insertion-tick"
		if ttl == 2 {
			name = "inclusive-insertion-endpoint"
		}
		t.Run(name, func(t *testing.T) {
			tree, seed := newTree(t, 64)
			resident := seed([]int{1})
			boundary, _ := tree.Lookup([]int{2})
			if err := tree.SetNodeRetention(resident, RetentionRequest{Priority: 100, TTL: ttl, Admitted: int64(tree.clock) - 1}); err != nil {
				t.Fatal(err)
			}
			if !tree.IsNodePinned(resident) {
				t.Fatal("fixture retention already expired before prediction")
			}
			tree.admissionSketch = nil
			comparisons := predict(t, tree, boundary, resident, []int{2}, 64)
			allowed := ttl == 1
			if len(comparisons) != 1 || (comparisons[0].forcedReason == "") != allowed {
				t.Fatalf("projected TTL capacity=%+v allowed=%v", comparisons, allowed)
			}
			snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
			n, err := tree.InsertSnapshot(boundary, []int{2}, snap, make([]float32, 16))
			if err != nil {
				snap.Close()
				t.Fatal(err)
			}
			if allowed {
				if n == boundary || n.snapshot != snap || resident.snapshot != nil {
					t.Fatal("insertion-tick expiry remained absent from actual capacity")
				}
			} else if n != boundary || snap.Cache != nil || resident.snapshot == nil {
				t.Fatal("inclusive endpoint lost its protection")
			}
			tree.Done(n)
		})
	}
	t.Run("empty-suffix-does-not-tick-or-release", func(t *testing.T) {
		tree, seed := newTree(t, 128)
		resident := seed([]int{1})
		other := seed([]int{2})
		boundary, _ := tree.Lookup([]int{1})
		if err := tree.SetNodeRetention(other, RetentionRequest{Priority: 100, TTL: 1, Admitted: int64(tree.clock) - 1}); err != nil {
			t.Fatal(err)
		}
		tree.admissionSketch = nil
		comparisons := predict(t, tree, boundary, resident, nil, 128)
		if len(comparisons) != 1 || comparisons[0].forcedReason != admissionReasonInsufficientCapacity {
			t.Fatal("replacement advanced TTL or counted its excluded boundary")
		}
		clock, owner := tree.clock, resident.snapshot
		snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
		n, err := tree.InsertSnapshot(boundary, nil, snap, make([]float32, 32))
		if err != nil {
			snap.Close()
			t.Fatal(err)
		}
		if n != boundary || snap.Cache != nil || resident.snapshot != owner || boundary.refs != 1 || tree.clock != clock || other.snapshot == nil {
			t.Fatal("replacement bypass changed insertion phase or ownership")
		}
		tree.Done(n)
	})
	t.Run("zero-budget-remains-unbounded", func(t *testing.T) {
		tree, seed := newTree(t, 0)
		resident := seed([]int{1})
		boundary, _ := tree.Lookup([]int{1, 2})
		if comparisons := predict(t, tree, boundary, resident, []int{2}, 64); len(comparisons) != 0 {
			t.Fatal("unbounded byte budget introduced an admission comparison")
		}
		snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
		n, err := tree.InsertSnapshot(boundary, []int{2}, snap, make([]float32, 16))
		if err != nil {
			snap.Close()
			t.Fatal(err)
		}
		if n == boundary || n.snapshot != snap || resident.snapshot == nil || tree.snapshotBytes != 128 {
			t.Fatal("unbounded insert evicted its ancestor snapshot")
		}
		tree.Done(n)
	})
}
