package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type deviceCapsBackend struct{ compute.Backend }

func (b *deviceCapsBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = true
	return caps
}

func TestHostDRAML2StagesEvictsAndRestoresCompletePrefix(t *testing.T) {
	cfg := model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        64,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       63,
	}
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	ids := []int{3, 7, 11, 13}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	s.Prefill(ids)
	snap, err := s.PrefixSnapshot()
	s.Close()
	if err != nil {
		t.Fatal(err)
	}

	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	root, _ := tree.Lookup(nil)
	leaf, err := tree.InsertSnapshot(root, ids, snap, []float32{1, 2, 3})
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	tree.Done(leaf)

	n, l1, matched, tier, err := tree.LookupSnapshotTiered(ids)
	if err != nil {
		t.Fatal(err)
	}
	if l1 == nil || matched != len(ids) || tier != SnapshotTierDeviceL1 {
		t.Fatalf("first lookup snapshot=%v matched=%d tier=%q", l1, matched, tier)
	}
	l1.Close()
	tree.Done(n)
	n = nil

	resident, candidates := tree.PressuredSnapshotCandidates()
	if resident <= 0 || len(candidates) != 1 || candidates[0].DeviceBytes != resident {
		t.Fatalf("pressure candidates resident=%d candidates=%+v", resident, candidates)
	}
	staged := tree.StageSnapshotToHost(candidates[0].Digest)
	if staged.Outcome != SnapshotTransferOK || staged.BytesMoved <= 0 || staged.Positions != len(ids) {
		t.Fatalf("stage = %+v", staged)
	}
	stagedNode := tree.findSnapshotByDigest(candidates[0].Digest)
	if stagedNode == nil || stagedNode.hostSnapshot == nil {
		t.Fatal("stage did not install a physical host snapshot")
	}
	wantHostBytes := stagedNode.hostSnapshot.ResidentBytes()
	if evicted := tree.EvictHotSnapshot(candidates[0].Digest); evicted != len(ids) {
		t.Fatalf("hot eviction positions=%d, want %d", evicted, len(ids))
	}

	n, l2, matched, tier, err := tree.LookupSnapshotTiered(ids)
	if err != nil {
		t.Fatal(err)
	}
	if l2 == nil || matched != len(ids) || tier != SnapshotTierHostL2 {
		t.Fatalf("second lookup snapshot=%v matched=%d tier=%q", l2, matched, tier)
	}
	tree.Done(n)
	restored, err := m.NewBackendSessionChecked(be)
	if err != nil {
		l2.Close()
		t.Fatal(err)
	}
	if err := l2.Restore(restored); err != nil {
		l2.Close()
		restored.Close()
		t.Fatal(err)
	}
	l2.Close()
	defer restored.Close()
	fresh, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	fresh.Prefill(ids)
	const next = 17
	got := restored.Step(next)
	want := fresh.Step(next)
	if diff := maxAbsDiff(got, want); diff != 0 {
		t.Fatalf("host-L2 continuation drift max_abs_diff=%g", diff)
	}

	stats := tree.Stats()
	if stats.DeviceSnapshotBytes != 0 || stats.HostSnapshotBytes != wantHostBytes {
		t.Fatalf("physical tier residency = %+v", stats)
	}
	if stats.L1Hits != 1 || stats.L1Misses != 1 || stats.L2Hits != 1 || stats.L2Misses != 0 {
		t.Fatalf("tier lookup counters = %+v", stats)
	}
	if stats.L2StageBytes != staged.BytesMoved || stats.L2RestoreBytes != staged.BytesMoved {
		t.Fatalf("transfer bytes stage/restore=%d/%d want %d", stats.L2StageBytes, stats.L2RestoreBytes, staged.BytesMoved)
	}
}

func TestHostDRAML2StageFaultRetainsHotOwner(t *testing.T) {
	cfg := model.Config{
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       1,
		HeadDim:          8,
		IntermediateSize: 32,
		VocabSize:        32,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       31,
	}
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	s.Prefill([]int{1, 2})
	snap, err := s.PrefixSnapshot()
	s.Close()
	if err != nil {
		t.Fatal(err)
	}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1, EvictionLRU)
	root, _ := tree.Lookup(nil)
	leaf, err := tree.InsertSnapshot(root, []int{1, 2}, snap, nil)
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	tree.Done(leaf)
	_, candidates := tree.PressuredSnapshotCandidates()
	if len(candidates) != 1 {
		t.Fatalf("candidates=%+v", candidates)
	}
	staged := tree.StageSnapshotToHost(candidates[0].Digest)
	if staged.Outcome != SnapshotTransferFault {
		t.Fatalf("stage=%+v, want typed fault", staged)
	}
	if evicted := tree.EvictHotSnapshot(candidates[0].Digest); evicted != 0 {
		t.Fatalf("faulted stage evicted %d hot positions", evicted)
	}
	n, got, _, tier, err := tree.LookupSnapshotTiered([]int{1, 2})
	if err != nil || got == nil || tier != SnapshotTierDeviceL1 {
		t.Fatalf("hot owner not retained after stage fault: snapshot=%v tier=%q err=%v", got, tier, err)
	}
	tree.Done(n)
	got.Close()
}

func TestHostDRAML2DisabledDoesNotReportLookups(t *testing.T) {
	tree := NewWithBudgets(0, 0)
	node, snapshot, _, tier, err := tree.LookupSnapshotTiered([]int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	tree.Done(node)
	if snapshot != nil || tier != SnapshotTierMiss {
		t.Fatalf("snapshot=%v tier=%q, want miss", snapshot, tier)
	}
	stats := tree.Stats()
	if stats.L1Misses != 1 {
		t.Fatalf("L1 misses=%d, want 1", stats.L1Misses)
	}
	if stats.L2Misses != 0 || stats.L2Faults != 0 {
		t.Fatalf("disabled L2 reported lookups: misses=%d faults=%d", stats.L2Misses, stats.L2Faults)
	}
}

func TestHostDRAML2DigestIncludesNamespace(t *testing.T) {
	cfg := model.Config{
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       1,
		HeadDim:          8,
		IntermediateSize: 32,
		VocabSize:        32,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       31,
	}
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	ids := []int{5, 8}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	s.Prefill(ids)
	first, err := s.PrefixSnapshot()
	s.Close()
	if err != nil {
		t.Fatal(err)
	}
	second, err := first.Clone()
	if err != nil {
		first.Close()
		t.Fatal(err)
	}

	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	for i, item := range []struct {
		ns   string
		snap *model.PrefixSnapshot
	}{
		{ns: "tenant:a", snap: first},
		{ns: "tenant:b", snap: second},
	} {
		boundary, matched := tree.LookupNS(item.ns, ids)
		leaf, err := tree.InsertSnapshot(boundary, ids[matched:], item.snap, nil)
		if err != nil {
			item.snap.Close()
			t.Fatalf("insert namespace %d: %v", i, err)
		}
		tree.Done(leaf)
	}

	_, candidates := tree.PressuredSnapshotCandidates()
	if len(candidates) != 2 {
		t.Fatalf("candidates=%+v, want one per namespace", candidates)
	}
	if candidates[0].Digest == candidates[1].Digest {
		t.Fatalf("identical token paths aliased across namespaces: %+v", candidates)
	}
	if got := tree.StageSnapshotToHost(candidates[0].Digest); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage first namespace: %+v", got)
	}
	firstNode := tree.findSnapshotByDigest(candidates[0].Digest)
	secondNode := tree.findSnapshotByDigest(candidates[1].Digest)
	if firstNode == nil || firstNode.hostSnapshot == nil {
		t.Fatal("selected namespace was not staged")
	}
	if secondNode == nil || secondNode.hostSnapshot != nil {
		t.Fatal("staging one namespace mutated the identical-token peer")
	}
}
