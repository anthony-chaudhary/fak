package radixkv

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s
func TestFiniteHighPriorityRetentionExpires(t *testing.T) {
	for _, priority := range []int{90, 100} {
		t.Run(fmt.Sprint(priority), func(t *testing.T) {
			tree := NewWithBudgets(1000, 64)
			t.Cleanup(func() { tree.forEachNode(tree.releaseHotSnapshot) })
			tokens := []int{1, 2, 3}
			boundary, _ := tree.Lookup(tokens[:1])
			ancestor := tree.Insert(boundary, tokens[:1], nil)
			tree.Done(ancestor)
			boundary, matched := tree.Lookup(tokens)
			snap := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
			leaf, err := tree.InsertSnapshot(boundary, tokens[matched:], snap, make([]float32, 16))
			if err != nil {
				snap.Close()
				t.Fatal(err)
			}
			tree.Done(leaf)
			req := RetentionRequest{Priority: priority, TTL: 3, Admitted: int64(tree.clock)}
			if err := tree.SetNodeRetention(leaf, req); err != nil {
				t.Fatal(err)
			}
			if got, ok := tree.NodeRetention(leaf); !ok || got != req {
				t.Fatalf("retention=%+v present=%v, want exact finite request %+v", got, ok, req)
			}
			if tree.IsNodePinned(ancestor) || ancestor.retention != nil {
				t.Fatal("finite retention permanently pinned or overwrote an existing ancestor")
			}
			// A subsequent radix split must copy the bounded descriptor rather
			// than leave a permanent pin behind on the new ancestor.
			mid, matched := tree.Lookup(tokens[:2])
			tree.Done(mid)
			if matched != 2 || mid == leaf {
				t.Fatalf("split boundary=%p leaf=%p matched=%d", mid, leaf, matched)
			}
			if got, ok := tree.NodeRetention(mid); !ok || got != req {
				t.Fatalf("split retention=%+v present=%v, want %+v", got, ok, req)
			}
			tree.clock = uint64(req.Admitted + req.TTL)
			for _, n := range []*node{mid, leaf} {
				if !tree.IsNodePinned(n) || tree.NodeTier(n) != Tier0PinnedRoot {
					t.Fatal("high-priority retention expired at the inclusive window boundary")
				}
			}
			if tree.snapshotVictim(nil) != nil || tree.selectVictimLeaf(false) != nil {
				t.Fatal("live finite retention was selected for eviction")
			}
			if victims, enough := tree.prospectiveSnapshotVictims(64, nil); enough || len(victims) != 0 {
				t.Fatalf("live finite retention counted as capacity: %+v enough=%v", victims, enough)
			}
			tree.clock++
			for _, n := range []*node{mid, leaf} {
				if tree.IsNodePinned(n) || tree.NodeTier(n) != Tier3Probationary {
					t.Fatal("finite retention remained pinned after its window")
				}
			}
			if tree.snapshotVictim(nil) != leaf || tree.selectVictimLeaf(false) != leaf {
				t.Fatal("expired finite retention remained absent from snapshot/token victims")
			}
			if victims, enough := tree.prospectiveSnapshotVictims(64, nil); !enough || len(victims) != 1 || victims[0].Tokens != len(tokens) {
				t.Fatalf("expired retention dry run=%+v enough=%v", victims, enough)
			}
			if !tree.makeSnapshotRoom(64, nil) || leaf.snapshot != nil {
				t.Fatal("expired snapshot was not reclaimed under byte pressure")
			}
			tree.SetRetention(0)
			if tree.MatchLen(tokens) != 0 || tree.PinnedNodes() != 0 || tree.Stats().SnapshotBytes != 0 {
				t.Fatal("expired retention left an immortal prefix or snapshot")
			}
		})
	}

	t.Run("explicit_permanent_paths", func(t *testing.T) {
		for _, mode := range []string{"pin-prefix", "set-tier", "retain-forever"} {
			t.Run(mode, func(t *testing.T) {
				tree := New(100)
				tokens := []int{1, 2, 3}
				boundary, _ := tree.Lookup(tokens)
				leaf := tree.Insert(boundary, tokens, nil)
				tree.Done(leaf)
				switch mode {
				case "pin-prefix":
					tree.PinPrefix(tokens)
				case "set-tier":
					tree.SetNodeTier(leaf, Tier0PinnedRoot)
				case "retain-forever":
					if err := tree.SetNodeRetention(leaf, RetentionRequest{Priority: 100, TTL: RetainForever}); err != nil {
						t.Fatal(err)
					}
				}
				mid, _ := tree.Lookup(tokens[:1])
				tree.Done(mid)
				tree.clock += 1000
				tree.SetRetention(0)
				if !tree.IsNodePinned(leaf) || !tree.IsNodePinned(mid) || tree.MatchLen(tokens) != len(tokens) {
					t.Fatal("explicit permanent path lost its pin")
				}
				boundary, _ = tree.Lookup(tokens)
				child := tree.Insert(boundary, []int{4}, nil)
				req := RetentionRequest{Priority: 100, TTL: 1, Admitted: int64(tree.clock)}
				if err := tree.SetNodeRetention(child, req); err != nil {
					t.Fatal(err)
				}
				tree.Done(child)
				tree.clock += 2
				tree.SetRetention(0)
				if tree.IsNodePinned(child) || !tree.IsNodePinned(leaf) || !tree.IsNodePinned(mid) || tree.MatchLen([]int{1, 2, 3, 4}) != len(tokens) {
					t.Fatal("expiring a finite child changed an independent permanent ancestor pin")
				}
			})
		}
	})
}
