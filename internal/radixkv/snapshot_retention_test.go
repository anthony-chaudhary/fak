package radixkv

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s
func TestSnapshotRetentionAdmissionEvictionParity(t *testing.T) {
	newTree := func(t *testing.T, budget int64) (*Tree, func([]int) *node) {
		t.Helper()
		tree := NewWithBudgets(1000, budget)
		t.Cleanup(func() { tree.forEachNode(tree.releaseHotSnapshot) })
		admit := func(tokens []int) *node {
			t.Helper()
			b, m := tree.Lookup(tokens)
			snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
			n, err := tree.InsertSnapshot(b, tokens[m:], snap, make([]float32, 16))
			if err != nil {
				snap.Close()
				t.Fatalf("admit %v: %v", tokens, err)
			}
			tree.Done(n)
			return n
		}
		return tree, admit
	}

	t.Run("priority_pin_lease_and_computing", func(t *testing.T) {
		tree, admit := newTree(t, 6*64)
		pinned := admit([]int{1})
		active := admit([]int{2, 2})
		idle := admit([]int{3, 3, 3})
		probation := admit([]int{4, 4, 4, 4})
		leased := admit([]int{5})
		computing := admit([]int{6})
		tree.SetNodeTier(pinned, Tier0PinnedRoot)
		tree.SetNodeTier(active, Tier1ActiveSubagent)
		tree.SetNodeTier(idle, Tier2IdleSubagent)
		tree.SetNodeTier(probation, Tier3Probationary)
		leased.refs++
		computing.SetState(NodeComputingPrefill)
		before := tree.snapshotBytes
		victims, enough := tree.prospectiveSnapshotVictims(3*64, nil)
		want := []*node{probation, idle, active}
		var wantStats []compute.KVSpanStats
		for _, n := range want {
			wantStats = append(wantStats, snapshotKVSpanStats(n))
		}
		if !enough || !reflect.DeepEqual(victims, wantStats) || tree.snapshotBytes != before {
			t.Fatalf("dry run = %+v enough=%v bytes=%d, want %+v and unchanged bytes=%d", victims, enough, tree.snapshotBytes, wantStats, before)
		}
		if victims, enough := tree.prospectiveSnapshotVictims(4*64, nil); enough || !reflect.DeepEqual(victims, wantStats) {
			t.Fatalf("mixed protected capacity dry run=%+v enough=%v, want only %+v and no fit", victims, enough, wantStats)
		}
		for _, n := range want {
			if n.snapshot == nil {
				t.Fatal("dry run released a snapshot")
			}
			if got := tree.snapshotVictim(nil); got != n {
				t.Fatalf("runtime victim=%p, want %p", got, n)
			}
			tree.releaseHotSnapshot(n)
		}
		if victims, enough := tree.prospectiveSnapshotVictims(1, nil); enough || len(victims) != 0 {
			t.Fatalf("protected residency counted as reclaimable: %+v enough=%v", victims, enough)
		}
		tree.maxSnapshotBytes = tree.snapshotBytes
		if tree.makeSnapshotRoom(64, nil) || pinned.snapshot == nil || leased.snapshot == nil || computing.snapshot == nil {
			t.Fatal("byte pressure reclaimed protected snapshots to force a fit")
		}
	})

	t.Run("descendants_ttl_exclusion_and_ties", func(t *testing.T) {
		tree, admit := newTree(t, 4*64)
		ancestor := admit([]int{1})
		child := admit([]int{1, 2})
		idle := admit([]int{3, 3, 3})
		excluded := admit([]int{4})
		tree.SetNodeTier(idle, Tier2IdleSubagent)
		if err := tree.SetNodeRetention(child, RetentionRequest{Priority: 75, TTL: 1, Admitted: 0}); err != nil {
			t.Fatal(err)
		}
		// The parent's older LRU stamp must not make admission compare against it
		// before the expired, deeper snapshot the runtime would really evict.
		victims, enough := tree.prospectiveSnapshotVictims(2*64, excluded)
		want := []compute.KVSpanStats{snapshotKVSpanStats(child), snapshotKVSpanStats(ancestor)}
		if !enough || !reflect.DeepEqual(victims, want) {
			t.Fatalf("deepest-first dry run=%+v enough=%v, want %+v", victims, enough, want)
		}
		for _, n := range []*node{child, ancestor} {
			if got := tree.snapshotVictim(excluded); got != n {
				t.Fatalf("runtime victim=%p, want %p", got, n)
			}
			tree.releaseHotSnapshot(n)
		}
		// Existing records provide a stable tie key even when logical recency is
		// identical. The same choice is used by prediction and real eviction.
		tree.SetNodeTier(excluded, Tier2IdleSubagent)
		idle.lastUsed, excluded.lastUsed = 10, 10
		for i := 0; i < 32; i++ {
			if got := tree.snapshotVictim(nil); got != idle {
				t.Fatalf("equal-priority victim=%p, want first record %p", got, idle)
			}
		}
		victims, enough = tree.prospectiveSnapshotVictims(64, nil)
		if !enough || !reflect.DeepEqual(victims, []compute.KVSpanStats{snapshotKVSpanStats(idle)}) {
			t.Fatalf("equal-priority dry run=%+v enough=%v", victims, enough)
		}
	})

	t.Run("insert_refuses_pinned_capacity", func(t *testing.T) {
		tree, admit := newTree(t, 64)
		pinned := admit([]int{1})
		tree.SetNodeTier(pinned, Tier0PinnedRoot)
		boundary, _ := tree.Lookup([]int{2})
		snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
		defer snap.Close()
		n, err := tree.InsertSnapshot(boundary, []int{2}, snap, make([]float32, 16))
		tree.Done(n)
		if !errors.Is(err, ErrSnapshotByteBudget) || pinned.snapshot == nil || tree.snapshotBytes != 64 {
			t.Fatalf("insert err=%v pinned=%v bytes=%d", err, pinned.snapshot != nil, tree.snapshotBytes)
		}
		// Byte-budget rejection retains caller ownership, including a usable snapshot.
		clone, err := snap.Clone()
		if err != nil || clone == nil || clone.Cache == nil {
			t.Fatalf("rejected caller-owned snapshot was closed: clone=%v err=%v", clone, err)
		}
		clone.Close()
	})

	t.Run("zero_budget_and_zero_byte_snapshot", func(t *testing.T) {
		tree, admit := newTree(t, 0)
		resident := admit([]int{1})
		boundary, _ := tree.Lookup([]int{2})
		snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
		zero, err := tree.InsertSnapshot(boundary, []int{2}, snap, nil)
		if err != nil {
			snap.Close()
			t.Fatal(err)
		}
		tree.Done(zero)
		zero.lastUsed = 0
		before := tree.snapshotBytes
		if !tree.makeSnapshotRoom(1<<20, nil) || tree.snapshotBytes != before {
			t.Fatal("zero byte budget stopped being unbounded")
		}
		victims, enough := tree.prospectiveSnapshotVictims(64, nil)
		want := []compute.KVSpanStats{snapshotKVSpanStats(zero), snapshotKVSpanStats(resident)}
		if !enough || !reflect.DeepEqual(victims, want) || victims[0].Bytes != 0 {
			t.Fatalf("zero-byte victim dry run=%+v enough=%v, want %+v", victims, enough, want)
		}
		if tree.snapshotBytes != before || zero.snapshot == nil || resident.snapshot == nil {
			t.Fatal("zero-byte dry run changed resident snapshots")
		}
	})
}
