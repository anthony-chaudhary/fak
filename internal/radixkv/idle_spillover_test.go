package radixkv

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// insertIdleTestSnapshot inserts a synthetic prefix snapshot into tree and releases its lease.
func insertIdleTestSnapshot(t *testing.T, tree *Tree, m *model.Model, be compute.Backend, ids []int) string {
	t.Helper()
	sess, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	sess.Prefill(ids)
	snap, err := sess.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	root, _ := tree.Lookup(nil)
	n, err := tree.InsertSnapshot(root, ids, snap, nil)
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	tree.Done(n)
	return tree.snapshotDigest("", n)
}

// fak-test:runtime fast est=50ms lane=default
func TestIdleSlotSpilloverGPUToRAMToLocalNVMe(t *testing.T) {
	t.Run("disabled_is_noop", func(t *testing.T) {
		m := model.NewSynthetic(remoteL3TestConfig())
		be := &deviceCapsBackend{Backend: compute.Default()}
		tree := NewWithTierBudgetsAndEvictionPolicy(0, 1<<30, 1<<30, EvictionLRU)
		store := &ssdSpillCountingStore{}
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-idle-test", be, m.Cfg, 1<<30); err != nil {
			t.Fatal(err)
		}
		insertIdleTestSnapshot(t, tree, m, be, []int{1, 2, 3, 4})
		tree.clock += 1000

		rep := tree.TriggerIdleSpillover(IdleSpilloverConfig{Enabled: false, GPUIdleTicks: 1, RAMIdleTicks: 1})
		if rep != (IdleSpilloverReport{}) {
			t.Fatalf("disabled report=%+v, want zero", rep)
		}
		if store.puts != 0 || tree.hostSnapshotBytes != 0 {
			t.Fatalf("disabled spillover mutated tiers: puts=%d hostBytes=%d", store.puts, tree.hostSnapshotBytes)
		}
	})

	t.Run("gpu_to_ram_to_nvme_and_restore", func(t *testing.T) {
		m := model.NewSynthetic(remoteL3TestConfig())
		be := &deviceCapsBackend{Backend: compute.Default()}
		tree := NewWithTierBudgetsAndEvictionPolicy(0, 1<<30, 1<<30, EvictionLRU)
		store := &ssdSpillCountingStore{}
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-idle-test", be, m.Cfg, 1<<30); err != nil {
			t.Fatal(err)
		}

		idsIdle := []int{3, 7, 11, 13}
		idsPinned := []int{5, 9, 2, 4}
		insertIdleTestSnapshot(t, tree, m, be, idsIdle)
		pinnedDigest := insertIdleTestSnapshot(t, tree, m, be, idsPinned)
		pinnedNode := tree.findSnapshotByDigest(pinnedDigest)
		if pinnedNode == nil {
			t.Fatal("missing pinned snapshot node")
		}
		pinnedNode.pinned = true

		// Advance clock past GPUIdleTicks (10) but below RAMIdleTicks (50).
		tree.clock += 20
		cfg := IdleSpilloverConfig{
			Enabled:          true,
			GPUIdleTicks:     10,
			RAMIdleTicks:     50,
			MaxSpillsPerTick: 4,
		}
		rep1 := tree.TriggerIdleSpillover(cfg)
		if rep1.StagedToRAM != 1 || rep1.EvictedFromGPU != 1 || rep1.FreedGPUBytes <= 0 {
			t.Fatalf("phase 1 GPU->RAM report=%+v, want 1 staged & evicted", rep1)
		}
		if rep1.SkippedPinned != 1 {
			t.Fatalf("SkippedPinned=%d, want 1", rep1.SkippedPinned)
		}
		if rep1.SpilledToNVMe != 0 || store.puts != 0 {
			t.Fatalf("unexpected NVMe spill before RAMIdleTicks: rep=%+v puts=%d", rep1, store.puts)
		}
		if tree.hostSnapshotBytes <= 0 {
			t.Fatal("expected host DRAM L2 bytes > 0 after GPU->RAM spill")
		}

		// Advance clock past RAMIdleTicks (50) and trigger again: RAM -> local NVMe.
		tree.clock += 40
		rep2 := tree.TriggerIdleSpillover(cfg)
		if rep2.SpilledToNVMe != 1 || rep2.EvictedFromRAM != 1 || rep2.FreedRAMBytes <= 0 {
			t.Fatalf("phase 2 RAM->NVMe report=%+v, want 1 spilled & evicted", rep2)
		}
		if store.puts != 1 || tree.hostSnapshotBytes != 0 || tree.LocalSSDReferencedBytes() <= 0 {
			t.Fatalf("post-NVMe state: puts=%d hostBytes=%d ssdBytes=%d", store.puts, tree.hostSnapshotBytes, tree.LocalSSDReferencedBytes())
		}

		// Verify the spilled prefix restores bit-identically from SnapshotTierLocalL3.
		n, recovered, matched, tier, err := tree.LookupSnapshotTieredContext(context.Background(), idsIdle)
		if err != nil {
			t.Fatal(err)
		}
		defer tree.Done(n)
		if recovered == nil || matched != len(idsIdle) || tier != SnapshotTierLocalL3 {
			t.Fatalf("NVMe restore snapshot=%v matched=%d tier=%q, want SnapshotTierLocalL3", recovered, matched, tier)
		}
		recovered.Close()
	})

	t.Run("bounded_max_spills_per_tick", func(t *testing.T) {
		m := model.NewSynthetic(remoteL3TestConfig())
		be := &deviceCapsBackend{Backend: compute.Default()}
		tree := NewWithTierBudgetsAndEvictionPolicy(0, 1<<30, 1<<30, EvictionLRU)
		insertIdleTestSnapshot(t, tree, m, be, []int{1, 2, 3, 4})
		insertIdleTestSnapshot(t, tree, m, be, []int{5, 6, 7, 8})
		tree.clock += 100

		rep := tree.TriggerIdleSpillover(IdleSpilloverConfig{
			Enabled:          true,
			GPUIdleTicks:     10,
			RAMIdleTicks:     500,
			MaxSpillsPerTick: 1,
		})
		if rep.EvictedFromGPU != 1 {
			t.Fatalf("EvictedFromGPU=%d, want 1 (bounded by MaxSpillsPerTick)", rep.EvictedFromGPU)
		}
	})
}
