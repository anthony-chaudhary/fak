package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestSegmentRecordLifecycle is the witness for issue #12848: a node-owned
// incarnation handle (RecordHandle) is minted only when a complete local
// snapshot record is admitted, survives L1->L2 ownership transfer, is replaced
// by a fresh incarnation on re-admission even with identical tokens, dies when
// the last complete local copy is logically evicted, and survives an edge split.
func TestSegmentRecordLifecycle(t *testing.T) {
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

	makeSnapshot := func(ids []int) *model.PrefixSnapshot {
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatalf("new session: %v", err)
		}
		s.Prefill(ids)
		snap, err := s.PrefixSnapshot()
		s.Close()
		if err != nil {
			t.Fatalf("prefix snapshot: %v", err)
		}
		return snap
	}

	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)

	// (2) A CPU KV-only node (structural intermediate, snapshot == nil) has NO
	// valid handle; a plain Insert must never mint one.
	kvOnlyIDs := []int{101, 102, 103}
	root, _ := tree.Lookup(kvOnlyIDs)
	kvLeaf := tree.Insert(root, kvOnlyIDs, nil)
	if kvLeaf == nil {
		t.Fatal("kv-only Insert returned nil leaf")
	}
	if _, ok := kvLeaf.RecordHandle(); ok {
		t.Fatalf("kv-only node %d minted a record handle", kvLeaf.plen)
	}
	if kvLeaf.hasRecord || kvLeaf.recordGen != 0 {
		t.Fatalf("kv-only node carries record state: hasRecord=%v gen=%d", kvLeaf.hasRecord, kvLeaf.recordGen)
	}
	tree.Done(kvLeaf)

	// (1) InsertSnapshot admits a complete snapshot and mints a record.
	ids := []int{3, 7, 11, 13}
	snap1 := makeSnapshot(ids)
	b1, _ := tree.Lookup(ids)
	node1, err := tree.InsertSnapshot(b1, ids, snap1, nil)
	if err != nil {
		snap1.Close()
		t.Fatalf("InsertSnapshot: %v", err)
	}
	h1, ok := node1.RecordHandle()
	if !ok {
		t.Fatal("InsertSnapshot did not mint a record handle")
	}
	if h1.Generation() == 0 {
		t.Fatalf("minted handle has zero generation")
	}
	if !tree.ValidateRecord(h1) {
		t.Fatalf("freshly minted handle fails validation")
	}
	tree.Done(node1)

	// (5) The record follows the node across L1 -> L2 -> L1.
	resident, candidates := tree.PressuredSnapshotCandidates()
	if resident <= 0 || len(candidates) != 1 {
		t.Fatalf("pressure candidates resident=%d candidates=%+v", resident, candidates)
	}
	if got := tree.StageSnapshotToHost(candidates[0].Digest); got.Outcome != SnapshotTransferOK {
		snap1.Close()
		t.Fatalf("stage to host = %+v", got)
	}
	if evicted := tree.EvictHotSnapshot(candidates[0].Digest); evicted != len(ids) {
		snap1.Close()
		t.Fatalf("hot eviction positions=%d, want %d", evicted, len(ids))
	}
	if !tree.ValidateRecord(h1) {
		snap1.Close()
		t.Fatal("record died on L1->L2 demotion though a host copy remains")
	}
	if got := tree.RestoreSnapshotFromHost(candidates[0].Digest); got.Outcome != SnapshotTransferOK {
		snap1.Close()
		t.Fatalf("restore from host = %+v", got)
	}
	if !tree.ValidateRecord(h1) {
		snap1.Close()
		t.Fatal("record did not survive L2->L1 restore")
	}
	if node1.snapshot == nil {
		snap1.Close()
		t.Fatal("restore did not reinstall the hot owner")
	}

	// (3) Replacing the snapshot on the same node with IDENTICAL tokens mints a
	// NEW incarnation and invalidates the old handle.
	snap2 := makeSnapshot(ids)
	node2, err := tree.InsertSnapshot(node1, nil, snap2, nil)
	if err != nil {
		snap2.Close()
		t.Fatalf("replacement InsertSnapshot: %v", err)
	}
	if node2 != node1 {
		snap2.Close()
		t.Fatalf("replacement landed on a different node: %p != %p", node2, node1)
	}
	h2, ok := node2.RecordHandle()
	if !ok {
		snap2.Close()
		t.Fatal("replacement did not mint a record handle")
	}
	if h2.Generation() == h1.Generation() {
		snap2.Close()
		t.Fatalf("replacement reused generation %d", h1.Generation())
	}
	if tree.ValidateRecord(h1) {
		snap2.Close()
		t.Fatal("stale handle validated after replacement")
	}
	if !tree.ValidateRecord(h2) {
		snap2.Close()
		t.Fatal("new handle failed validation after replacement")
	}

	// (4) Logical eviction: dropping the last complete local copy kills the
	// record. EvictNode -> releaseSnapshotPayload releases every local tier.
	tree.EvictNode(node2)
	if node2.hasRecord || node2.recordGen != 0 {
		t.Fatalf("record survived logical eviction: hasRecord=%v gen=%d", node2.hasRecord, node2.recordGen)
	}
	if tree.ValidateRecord(h2) {
		t.Fatal("handle validated after its node's record was logically evicted")
	}

	// (6) An edge split preserves the record of the snapshot-bearing node and
	// does not transfer it onto the fresh structural intermediate.
	longIDs := []int{3, 7, 11, 13, 21}
	snapLong := makeSnapshot(longIDs)
	bl, _ := tree.Lookup(longIDs)
	nLong, err := tree.InsertSnapshot(bl, longIDs, snapLong, nil)
	if err != nil {
		snapLong.Close()
		t.Fatalf("long InsertSnapshot: %v", err)
	}
	hLong, ok := nLong.RecordHandle()
	if !ok {
		snapLong.Close()
		t.Fatal("long snapshot admission did not mint a record")
	}
	tree.Done(nLong)

	before := tree.Stats().Splits
	// A request sharing only the first three tokens diverges mid-edge, forcing
	// the long node's edge to split.
	div, _ := tree.Lookup([]int{3, 7, 11, 99})
	if tree.Stats().Splits != before+1 {
		snapLong.Close()
		t.Fatalf("expected exactly one split, got %d -> %d", before, tree.Stats().Splits)
	}
	tree.Done(div)
	mid := nLong.parent
	if mid == nil {
		snapLong.Close()
		t.Fatal("split did not leave the snapshot node attached to an intermediate")
	}
	if mid.hasRecord || mid.recordGen != 0 {
		snapLong.Close()
		t.Fatalf("split transferred a record onto the new intermediate: hasRecord=%v gen=%d", mid.hasRecord, mid.recordGen)
	}
	if !tree.ValidateRecord(hLong) {
		snapLong.Close()
		t.Fatal("record invalidated by split though its snapshot-bearing node still owns it")
	}
	if nLong.snapshot == nil {
		snapLong.Close()
		t.Fatal("split dropped the snapshot from its owning node")
	}

	// (7) The multi-tier aggregator releaseSnapshotPayload must invalidate a node
	// whose ONLY remaining copy is a remote L3 reference: releaseHot and
	// releaseHost mint nothing once remote is the sole tier, so the aggregator's
	// final invalidateRecordIfNoLocalCopy is the checkpoint that kills it. This
	// exercises the full teardown path EvictNode -> closeSubtreeSnapshots ->
	// releaseSnapshotPayload with a real remoteSnapshot, not a faked hasRecord.
	remoteStore := &memorySnapshotStore{}
	rTree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	if err := rTree.ConfigureRemoteSnapshotStore(remoteStore, "synthetic-l3-test", be, m.Cfg); err != nil {
		t.Fatalf("configure remote store: %v", err)
	}
	rIDs := []int{3, 7, 11, 13}
	rSnap := makeSnapshot(rIDs)
	rb, _ := rTree.Lookup(rIDs)
	rNode, err := rTree.InsertSnapshot(rb, rIDs, rSnap, nil)
	if err != nil {
		rSnap.Close()
		t.Fatalf("remote InsertSnapshot: %v", err)
	}
	hRemote, ok := rNode.RecordHandle()
	if !ok {
		rSnap.Close()
		t.Fatal("remote-only admission did not mint a record")
	}
	rTree.Done(rNode)
	_, rCandidates := rTree.PressuredSnapshotCandidates()
	if len(rCandidates) != 1 {
		rSnap.Close()
		t.Fatalf("remote pressure candidates=%+v, want one", rCandidates)
	}
	rDigest := rCandidates[0].Digest
	if got := rTree.StageSnapshotToRemote(t.Context(), rDigest); got.Outcome != SnapshotTransferOK {
		rSnap.Close()
		t.Fatalf("stage remote: %+v", got)
	}
	if evicted := rTree.EvictHotSnapshot(rDigest); evicted != len(rIDs) {
		rSnap.Close()
		t.Fatalf("remote hot eviction positions=%d, want %d", evicted, len(rIDs))
	}
	// Remote is now the sole copy: the record must follow the tiers and survive.
	if rNode.snapshot != nil || rNode.hostSnapshot != nil || rNode.remoteSnapshot == nil {
		rSnap.Close()
		t.Fatalf("remote-only state not reached: hot=%v host=%v remote=%v",
			rNode.snapshot, rNode.hostSnapshot, rNode.remoteSnapshot)
	}
	if !rTree.ValidateRecord(hRemote) {
		rSnap.Close()
		t.Fatal("record died while a remote L3 copy remained")
	}
	// Full teardown drops the remote reference and must invalidate the record.
	rTree.EvictNode(rNode)
	if rNode.hasRecord || rNode.recordGen != 0 {
		rSnap.Close()
		t.Fatalf("remote-only record survived releaseSnapshotPayload: hasRecord=%v gen=%d",
			rNode.hasRecord, rNode.recordGen)
	}
	if rTree.ValidateRecord(hRemote) {
		rSnap.Close()
		t.Fatal("handle validated after a remote-only node was logically evicted")
	}
}

// TestRemoteL3MissInvalidatesRecord is the witness for fak#12949: when a node has
// been hot- and host-evicted so its sole remaining copy is a remote L3 reference,
// and the L3 store then misses on restore, the last complete copy is gone and the
// node's record incarnation must die with it. Before the fix, restoreSnapshotFromRemote
// nil'd n.remoteSnapshot without calling invalidateRecordIfNoLocalCopy, leaving
// hasRecord=true / recordGen!=0 and a RecordHandle that still reported ok.
func TestRemoteL3MissInvalidatesRecord(t *testing.T) {
	cfg := remoteL3TestConfig()
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	store := &memorySnapshotStore{}

	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	if err := tree.ConfigureRemoteSnapshotStore(store, "synthetic-l3-test", be, m.Cfg); err != nil {
		t.Fatalf("configure remote store: %v", err)
	}

	ids := []int{3, 7, 11, 13}
	digest := insertRemoteL3Snapshot(t, tree, m, be, ids)
	_, n := tree.findSnapshotByDigestNS(digest)
	if n == nil {
		t.Fatal("inserted snapshot node not found")
	}
	h, ok := n.RecordHandle()
	if !ok {
		t.Fatal("admission did not mint a record handle")
	}

	if got := tree.StageSnapshotToRemote(t.Context(), digest); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage remote: %+v", got)
	}
	if evicted := tree.EvictHotSnapshot(digest); evicted != len(ids) {
		t.Fatalf("hot eviction positions=%d, want %d", evicted, len(ids))
	}
	// Remote is now the sole complete copy; the record must still be live.
	if n.snapshot != nil || n.hostSnapshot != nil || n.remoteSnapshot == nil {
		t.Fatalf("remote-only state not reached: hot=%v host=%v remote=%v",
			n.snapshot, n.hostSnapshot, n.remoteSnapshot)
	}
	if !tree.ValidateRecord(h) {
		t.Fatal("record died while the remote L3 copy was still present")
	}

	// Evict the L3 object so the next restore observes a genuine !found miss.
	delete(store.data, digest)

	snap, found, err := tree.restoreSnapshotFromRemote(t.Context(), "", n)
	if err != nil {
		t.Fatalf("restore on L3 miss returned error: %v", err)
	}
	if found || snap != nil {
		t.Fatalf("expected clean miss, got found=%v snap=%v", found, snap)
	}
	if n.remoteSnapshot != nil {
		t.Fatalf("L3 miss did not clear the remote reference: %v", n.remoteSnapshot)
	}
	if n.hasRecord || n.recordGen != 0 {
		t.Fatalf("remote-only record survived an L3 miss: hasRecord=%v gen=%d", n.hasRecord, n.recordGen)
	}
	if _, ok := n.RecordHandle(); ok {
		t.Fatal("RecordHandle still reported ok after the last copy was dropped")
	}
	if tree.ValidateRecord(h) {
		t.Fatal("stale handle validated after the last complete copy was dropped by an L3 miss")
	}
}
