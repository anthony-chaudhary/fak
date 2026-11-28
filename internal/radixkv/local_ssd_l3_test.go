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
	puts    int
	touches [][]string
	deletes [][]string
}

func (s *ssdSpillCountingStore) Put(ctx context.Context, key string, payload []byte) error {
	s.puts++
	return s.memorySnapshotStore.Put(ctx, key, payload)
}

func (s *ssdSpillCountingStore) Touch(_ context.Context, keys []string) error {
	s.touches = append(s.touches, append([]string(nil), keys...))
	return nil
}

func (s *ssdSpillCountingStore) Delete(_ context.Context, keys []string) error {
	s.deletes = append(s.deletes, append([]string(nil), keys...))
	for _, k := range keys {
		delete(s.data, k)
	}
	return nil
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

// fak-test:runtime fast est=20ms lane=default
func TestUpperTierHitTouchesSSDCopy(t *testing.T) {
	store := &ssdSpillCountingStore{}
	tree, m, be, idsA := localSSDFixture(t, func(tree *Tree, m *model.Model, be compute.Backend) {
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1<<30); err != nil {
			t.Fatal(err)
		}
	})
	// Re-insert A into L1 while its ssdSnapshot ref is still attached, then hit L1.
	nA, _ := tree.Lookup(idsA)
	if nA == nil || nA.ssdSnapshot == nil {
		tree.Done(nA)
		t.Fatal("expected A to retain ssdSnapshot ref")
	}
	wantDigest := nA.ssdSnapshot.digest
	tree.Done(nA)

	// Stage a longer extension of A into L1 while A retains its ssdSnapshot ref, then hit L1.
	idsLong := append(append([]int(nil), idsA...), 15, 17)
	tree.maxSnapshotBytes = 1 << 30
	sess, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	sess.Prefill(idsLong)
	snap, err := sess.PrefixSnapshot()
	sess.Close()
	if err != nil {
		t.Fatal(err)
	}
	nA, _ = tree.Lookup(idsA)
	leaf, err := tree.InsertSnapshot(nA, []int{15, 17}, snap, nil)
	tree.Done(nA)
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	tree.Done(leaf)

	n, recovered, matched, tier, err := tree.LookupSnapshotTieredContext(context.Background(), idsLong)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Done(n)
	if recovered == nil || matched != len(idsLong) || tier != SnapshotTierDeviceL1 {
		t.Fatalf("expected L1 hit, got tier=%q matched=%d", tier, matched)
	}
	recovered.Close()

	if len(store.touches) != 1 || len(store.touches[0]) != 1 || store.touches[0][0] != wantDigest {
		t.Fatalf("upper-tier hit touches=%v, want [[%s]]", store.touches, wantDigest)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestLocalSSDSpillEvictsColdestToLowWatermark(t *testing.T) {
	store := &ssdSpillCountingStore{}
	m := model.NewSynthetic(remoteL3TestConfig())
	be := &deviceCapsBackend{Backend: compute.Default()}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)

	idsA := []int{3, 7, 11, 13}
	idsB := []int{5, 9, 2, 4}
	idsC := []int{6, 8, 10, 12}
	idsD := []int{14, 15, 16, 17}

	// Arm with large budget first to measure one spilled envelope's size.
	if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1<<30); err != nil {
		t.Fatal(err)
	}

	stageAndDropL1 := func(ids []int) {
		t.Helper()
		d := insertRemoteL3Snapshot(t, tree, m, be, ids)
		if got := tree.StageSnapshotToHost(d); got.Outcome != SnapshotTransferOK {
			t.Fatalf("stage %v to host: %+v", ids, got)
		}
		if tree.EvictHotSnapshot(d) != len(ids) {
			t.Fatalf("evict hot %v", ids)
		}
	}

	// Stage A to L2, then shrink L2 to 1 entry so each subsequent stage spills the L2 victim to SSD.
	stageAndDropL1(idsA)
	tree.maxHostSnapshotBytes = tree.hostSnapshotBytes

	stageAndDropL1(idsB) // spills A to SSD
	entryBytes := tree.LocalSSDReferencedBytes()
	if entryBytes <= 0 {
		t.Fatalf("entryBytes=%d, want >0", entryBytes)
	}
	// Budget = 3 entries: A, B, C fit; spilling D exceeds high watermark and must evict down to <= 80% (<=2 entries).
	tree.localSSDBudget = entryBytes * 3

	stageAndDropL1(idsC) // spills B to SSD
	// Touch A via an SSD restore so B becomes the coldest SSD entry.
	nA, snapA, _, tierA, err := tree.LookupSnapshotTieredContext(context.Background(), idsA)
	if err != nil || snapA == nil || tierA != SnapshotTierLocalL3 {
		t.Fatalf("restore A before spill: snap=%v tier=%q err=%v", snapA, tierA, err)
	}
	snapA.Close()
	tree.Done(nA)

	// Stage D into L2 -> spills C to SSD (now A, B, C = 3*entryBytes == budget).
	stageAndDropL1(idsD)
	if tree.LocalSSDReferencedBytes() != entryBytes*3 {
		t.Fatalf("SSD bytes before overflow=%d, want %d", tree.LocalSSDReferencedBytes(), entryBytes*3)
	}

	// One more entry E forces D out of L2 -> SSD overflow! Should evict coldest (B) down to low watermark, NOT drop D.
	idsE := []int{1, 2, 3, 4}
	stageAndDropL1(idsE)

	if tree.LocalSSDReferencedBytes() > tree.localSSDBudget {
		t.Fatalf("SSD bytes=%d exceeds budget=%d", tree.LocalSSDReferencedBytes(), tree.localSSDBudget)
	}
	if len(store.deletes) == 0 {
		t.Fatal("expected coldest SSD entry to be deleted from store on watermark eviction")
	}

	// D (incoming spill) must be restorable from SSD (not dropped).
	nD, snapD, _, tierD, err := tree.LookupSnapshotTieredContext(context.Background(), idsD)
	if err != nil || snapD == nil || tierD != SnapshotTierLocalL3 {
		t.Fatalf("incoming spill D lookup: snap=%v tier=%q err=%v, want LocalL3 hit", snapD, tierD, err)
	}
	snapD.Close()
	tree.Done(nD)

	// A (touched) must survive; B (coldest) must be evicted.
	nA2, snapA2, _, tierA2, err := tree.LookupSnapshotTieredContext(context.Background(), idsA)
	if err != nil || snapA2 == nil || tierA2 != SnapshotTierLocalL3 {
		t.Fatalf("touched A lookup: snap=%v tier=%q err=%v, want LocalL3 hit", snapA2, tierA2, err)
	}
	snapA2.Close()
	tree.Done(nA2)

	nB, snapB, _, tierB, err := tree.LookupSnapshotTieredContext(context.Background(), idsB)
	if err != nil || snapB != nil || tierB != SnapshotTierMiss {
		t.Fatalf("coldest B lookup: snap=%v tier=%q err=%v, want miss after eviction", snapB, tierB, err)
	}
	tree.Done(nB)
}

type gatedSSDStore struct {
	ssdSpillCountingStore
	gate    chan struct{}
	failGet error
}

func (s *gatedSSDStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	if s.gate != nil {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return nil, false, ctx.Err()
		}
	}
	if s.failGet != nil {
		return nil, false, s.failGet
	}
	return s.ssdSpillCountingStore.Get(ctx, key)
}

// fak-test:runtime fast est=50ms lane=default
func TestLocalSSDRestoreParksRequestAndReadmits(t *testing.T) {
	store := &gatedSSDStore{gate: make(chan struct{})}
	tree, m, be, idsA := localSSDFixture(t, func(tree *Tree, m *model.Model, be compute.Backend) {
		if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1<<30); err != nil {
			t.Fatal(err)
		}
	})

	// 1. Park request for A while disk Get is blocked on store.gate.
	job, parked, err := tree.ParkLocalSSDRestore(context.Background(), idsA)
	if err != nil || !parked || job == nil {
		t.Fatalf("ParkLocalSSDRestore parked=%v job=%v err=%v", parked, job, err)
	}
	if job.State() != LocalSSDRestoreParked {
		t.Fatalf("initial state=%v, want LocalSSDRestoreParked", job.State())
	}
	if got := tree.Stats().ProtectedTokens; got == 0 {
		t.Fatalf("ProtectedTokens=%d while parked, want >0 (node leased)", got)
	}

	// While A is parked, another request (B) can still be scheduled on the tree.
	idsB := []int{5, 9, 2, 4}
	nB, snapB, _, tierB, err := tree.LookupSnapshotTieredContext(context.Background(), idsB)
	if err != nil || snapB == nil || tierB != SnapshotTierDeviceL1 {
		t.Fatalf("concurrent B lookup while A parked: snap=%v tier=%q err=%v", snapB, tierB, err)
	}
	snapB.Close()
	tree.Done(nB)

	// 2. Unblock disk read and re-admit A.
	close(store.gate)
	<-job.Done()
	if job.State() != LocalSSDRestoreReady {
		t.Fatalf("completed state=%v, want LocalSSDRestoreReady", job.State())
	}
	nA, snapA, matchedA, tierA, err := job.Readmit()
	if err != nil || snapA == nil || matchedA != len(idsA) || tierA != SnapshotTierLocalL3 {
		t.Fatalf("Readmit: snap=%v matched=%d tier=%q err=%v", snapA, matchedA, tierA, err)
	}
	restored, err := m.NewBackendSessionChecked(be)
	if err != nil {
		snapA.Close()
		tree.Done(nA)
		t.Fatal(err)
	}
	if err := snapA.Restore(restored); err != nil {
		restored.Close()
		snapA.Close()
		tree.Done(nA)
		t.Fatal(err)
	}
	snapA.Close()
	tree.Done(nA)

	fresh, err := m.NewBackendSessionChecked(be)
	if err != nil {
		restored.Close()
		t.Fatal(err)
	}
	fresh.Prefill(idsA)
	if diff := maxAbsDiff(restored.Step(17), fresh.Step(17)); diff != 0 {
		restored.Close()
		fresh.Close()
		t.Fatalf("parked-restore continuation drift=%g", diff)
	}
	restored.Close()
	fresh.Close()
	if got := tree.Stats().ProtectedTokens; got != 0 {
		t.Fatalf("ProtectedTokens=%d after Done, want 0", got)
	}

	// 3. Failed load rewinds the tree lease cleanly.
	store.failGet = context.DeadlineExceeded
	job2, parked2, err := tree.ParkLocalSSDRestore(context.Background(), idsA)
	if err != nil || !parked2 || job2 == nil {
		t.Fatalf("ParkLocalSSDRestore (fail) parked=%v err=%v", parked2, err)
	}
	<-job2.Done()
	n2, snap2, _, _, err2 := job2.Readmit()
	if err2 == nil || snap2 != nil || n2 != nil {
		t.Fatalf("failed Readmit got n=%v snap=%v err=%v, want error + nil", n2, snap2, err2)
	}
	if got := tree.Stats().ProtectedTokens; got != 0 {
		t.Fatalf("ProtectedTokens=%d after failed-load rewind, want 0", got)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestDemotionSkipsSupersededRecurrentCheckpoint(t *testing.T) {
	store := &ssdSpillCountingStore{}
	m := model.NewSynthetic(remoteL3TestConfig())
	be := &deviceCapsBackend{Backend: compute.Default()}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	if err := tree.ConfigureLocalSSDSnapshotStore(store, "synthetic-ssd-test", be, m.Cfg, 1<<30); err != nil {
		t.Fatal(err)
	}
	tree.ConfigureRecurrentRetention(8) // interval [0,8)

	// Stage two recurrent checkpoints in the same interval [0,8):
	// prefix1 (len 2) is superseded by prefix2 (len 4) on the same linear chain.
	ids1 := []int{3, 7}
	ids2 := []int{3, 7, 11, 13}
	d1 := insertRemoteL3Snapshot(t, tree, m, be, ids1)
	if got := tree.StageSnapshotToHost(d1); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage 1: %+v", got)
	}
	tree.EvictHotSnapshot(d1)
	n1, _ := tree.Lookup(ids1)
	tree.MarkRecurrentCheckpoint(n1)
	tree.Done(n1)

	d2 := insertRemoteL3Snapshot(t, tree, m, be, ids2)
	if got := tree.StageSnapshotToHost(d2); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage 2: %+v", got)
	}
	tree.EvictHotSnapshot(d2)
	n2, _ := tree.Lookup(ids2)
	tree.MarkRecurrentCheckpoint(n2)
	tree.Done(n2)

	// Shrink L2 to current (n1+n2) size so staging a 3rd 4-token entry evicts n1 (superseded) and n2 (latest).
	tree.maxHostSnapshotBytes = tree.hostSnapshotBytes
	ids3 := []int{5, 9, 2, 4}
	d3 := insertRemoteL3Snapshot(t, tree, m, be, ids3)
	if got := tree.StageSnapshotToHost(d3); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage 3: %+v", got)
	}
	tree.EvictHotSnapshot(d3)

	// Superseded n1 must NOT have been spilled; only n2 should be on SSD (1 put, not 2).
	if store.puts != 1 {
		t.Fatalf("store.puts=%d, want 1 (superseded checkpoint n1 skipped)", store.puts)
	}
	nCheck, snapCheck, _, tierCheck, err := tree.LookupSnapshotTieredContext(context.Background(), ids1)
	if err != nil {
		t.Fatal(err)
	}
	tree.Done(nCheck)
	if snapCheck != nil || tierCheck == SnapshotTierLocalL3 {
		if snapCheck != nil {
			snapCheck.Close()
		}
		t.Fatalf("superseded checkpoint ids1 was spilled to SSD (tier=%q), want skipped", tierCheck)
	}
}
