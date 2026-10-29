package radixkv

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ssdSpillCountingStore records how many spills reached the byte owner.
type ssdSpillCountingStore struct {
	memorySnapshotStore
	puts int
}

func (s *ssdSpillCountingStore) Put(ctx context.Context, key string, payload []byte) error {
	s.puts++
	return s.memorySnapshotStore.Put(ctx, key, payload)
}

// localSSDFixture builds a tree whose host L2 fits exactly one prefix image, stages
// prefix A to L2 (device copy dropped), then stages prefix B so L2 pressure must evict A.
// configure runs before any staging so the local-SSD tier (or its absence) is in place
// when the eviction happens.
func localSSDFixture(t *testing.T, configure func(*Tree, *model.Model, compute.Backend)) (*Tree, *model.Model, compute.Backend, []int) {
	t.Helper()
	m := model.NewSynthetic(remoteL3TestConfig())
	be := &deviceCapsBackend{Backend: compute.Default()}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	configure(tree, m, be)
	idsA := []int{3, 7, 11, 13}
	idsB := []int{5, 9, 2, 4}

	digestA := insertRemoteL3Snapshot(t, tree, m, be, idsA)
	if got := tree.StageSnapshotToHost(digestA); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage A to host: %+v", got)
	}
	if tree.EvictHotSnapshot(digestA) != len(idsA) {
		t.Fatal("device L1 copy of A was not removed")
	}
	// Shrink L2 to exactly A's resident image so staging B must evict A.
	tree.maxHostSnapshotBytes = tree.hostSnapshotBytes

	digestB := insertRemoteL3Snapshot(t, tree, m, be, idsB)
	if got := tree.StageSnapshotToHost(digestB); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage B to host: %+v", got)
	}
	if got := tree.Stats().L2Evictions; got != 1 {
		t.Fatalf("L2 evictions=%d, want 1 (A displaced by B)", got)
	}
	return tree, m, be, idsA
}

// fak-test:runtime fast est=50ms lane=default
func TestHostL2PressureSpillsVictimToLocalSSD(t *testing.T) {
	store := &ssdSpillCountingStore{}
	tree, m, be, idsA := localSSDFixture(t, func(tree *Tree, m *model.Model, be compute.Backend) {
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1<<30); err != nil {
			t.Fatal(err)
		}
	})
	if store.puts != 1 {
		t.Fatalf("spills=%d, want exactly one per L2 victim", store.puts)
	}
	if got := tree.LocalSSDReferencedBytes(); got <= 0 {
		t.Fatalf("referenced SSD bytes=%d, want the spilled envelope charged", got)
	}

	n, recovered, matched, tier, err := tree.LookupSnapshotTieredContext(context.Background(), idsA)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Done(n)
	if recovered == nil || matched != len(idsA) || tier != SnapshotTierLocalL3 {
		t.Fatalf("SSD lookup snapshot=%v matched=%d tier=%q", recovered, matched, tier)
	}
	restored, err := m.NewBackendSessionChecked(be)
	if err != nil {
		recovered.Close()
		t.Fatal(err)
	}
	defer restored.Close()
	if err := recovered.Restore(restored); err != nil {
		recovered.Close()
		t.Fatal(err)
	}
	recovered.Close()
	fresh, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	fresh.Prefill(idsA)
	const next = 17
	if diff := maxAbsDiff(restored.Step(next), fresh.Step(next)); diff != 0 {
		t.Fatalf("SSD-restored continuation drift max_abs_diff=%g", diff)
	}
	stats := tree.Stats()
	if !stats.LocalL3Enabled || stats.LocalL3Hits != 1 || stats.LocalL3HitTokens != len(idsA) ||
		stats.LocalL3StageBytes <= 0 || stats.LocalL3RestoreBytes <= 0 {
		t.Fatalf("local SSD telemetry = %+v", stats)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestHostL2PressureWithoutSSDSpillDropsVictim(t *testing.T) {
	tree, _, _, idsA := localSSDFixture(t, func(*Tree, *model.Model, compute.Backend) {})
	n, recovered, _, tier, err := tree.LookupSnapshotTieredContext(context.Background(), idsA)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Done(n)
	if recovered != nil || tier != SnapshotTierMiss {
		t.Fatalf("nil spiller lookup snapshot=%v tier=%q, want today's miss", recovered, tier)
	}
	if stats := tree.Stats(); stats.LocalL3Enabled || stats.LocalL3Hits != 0 || stats.LocalL3StageBytes != 0 {
		t.Fatalf("off tier telemetry = %+v", stats)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestHostL2SpillRespectsLocalSSDBudget(t *testing.T) {
	store := &ssdSpillCountingStore{}
	tree, _, _, idsA := localSSDFixture(t, func(tree *Tree, m *model.Model, be compute.Backend) {
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1); err != nil {
			t.Fatal(err)
		}
	})
	if store.puts != 0 || tree.LocalSSDReferencedBytes() != 0 {
		t.Fatalf("over-budget spill puts=%d referenced=%d, want dropped", store.puts, tree.LocalSSDReferencedBytes())
	}
	n, recovered, _, _, err := tree.LookupSnapshotTieredContext(context.Background(), idsA)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Done(n)
	if recovered != nil {
		t.Fatal("over-budget victim restored, want a miss")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestHostL2SpillPrunedRecordReadsAsCleanMiss(t *testing.T) {
	store := &ssdSpillCountingStore{}
	tree, _, _, idsA := localSSDFixture(t, func(tree *Tree, m *model.Model, be compute.Backend) {
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1<<30); err != nil {
			t.Fatal(err)
		}
	})
	store.data = map[string][]byte{} // the store owner pruned the record
	n, recovered, _, _, err := tree.LookupSnapshotTieredContext(context.Background(), idsA)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Done(n)
	if recovered != nil {
		t.Fatal("pruned SSD record restored, want a clean miss")
	}
	if got := tree.LocalSSDReferencedBytes(); got != 0 {
		t.Fatalf("referenced SSD bytes=%d after pruned miss, want 0", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestConfigureLocalSSDSnapshotStoreRefusesNonPositiveBudget(t *testing.T) {
	m := model.NewSynthetic(remoteL3TestConfig())
	be := &deviceCapsBackend{Backend: compute.Default()}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	if err := tree.ConfigureLocalSSDSnapshotStore(&memorySnapshotStore{}, "m", be, m.Cfg, 0); err == nil {
		t.Fatal("zero budget accepted, want refusal (0 = OFF arm installs no tier)")
	}
	if tree.LocalSSDSnapshotEnabled() {
		t.Fatal("tier armed after refused configuration")
	}
}
