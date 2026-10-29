package radixkv

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// local_ssd_l3.go — the durable LOCAL-SSD overflow tier below host DRAM L2
// (perf-parity T-06 / umem TICKET-04). When L2 pressure evicts a staged prefix
// image, makeHostSnapshotRoom demotes it to local disk instead of dropping it for
// a cold prefill; a later lookup that misses both L1 and L2 restores it from the
// store. The tier reuses the remote-L3 envelope format and SnapshotStore contract
// verbatim, so any l3kv.Store (the production DiskStore) is a valid byte owner.
//
// It is deliberately eviction-driven: nothing is staged proactively. A node only
// reaches SSD after it has already been admitted to L2 and then lost the L2
// budget race, so the tier can never hold a prefix the hot/host tiers would have
// served anyway, and its whole footprint is bounded by localSSDBudget.

// ssdSnapshotRef is a verified reference to a complete serialized HostPrefixSnapshot
// in the local-SSD overflow tier. Like a remote reference it owns no local payload
// bytes; its on-disk size counts against the tree's SSD budget while attached.
type ssdSnapshotRef struct {
	digest string
	scope  string
	tokens int
	bytes  int64
}

// localSSDDigestPrefix separates this tier's content addressing from the remote
// tier's ("fak-prefix-l3/v1") even though both serialize into the same envelope
// shape: a record written by one tier must never be read back as the other's.
const localSSDDigestPrefix = "fak-prefix-ssd/v1\x00"

// ConfigureLocalSSDSnapshotStore installs the local-SSD overflow byte owner on a
// tree. modelID and cfg are the scope/geometry gates checked before recovered
// bytes can become a live PrefixSnapshot; backend is the physical install target;
// budget caps the total on-disk bytes the tree will reference.
func (t *Tree) ConfigureLocalSSDSnapshotStore(store SnapshotStore, modelID string, backend compute.Backend, cfg model.Config, budget int64) error {
	if t == nil || store == nil || backend == nil || modelID == "" {
		return errors.New("radixkv: incomplete local SSD prefix snapshot configuration")
	}
	if budget <= 0 {
		return fmt.Errorf("radixkv: local SSD prefix snapshot budget must be positive, got %d", budget)
	}
	t.localSSDStore = store
	t.localSSDModelID = modelID
	t.localSSDBackend = backend
	t.localSSDConfig = cfg
	t.localSSDBudget = budget
	return nil
}

// DetachLocalSSDSnapshotStore removes the local-SSD overflow tier at runtime.
// Attached references stay with their nodes until released through the normal
// lifecycle; the tier simply stops admitting new spills and serving restores.
func (t *Tree) DetachLocalSSDSnapshotStore() {
	if t == nil {
		return
	}
	t.localSSDStore = nil
	t.localSSDModelID = ""
	t.localSSDBackend = nil
	t.localSSDConfig = model.Config{}
	t.localSSDBudget = 0
}

// LocalSSDSnapshotEnabled reports whether this tree owns an armed local-SSD tier.
func (t *Tree) LocalSSDSnapshotEnabled() bool {
	return t != nil && t.localSSDStore != nil && t.localSSDBackend != nil && t.localSSDModelID != "" && t.localSSDBudget > 0
}

// LocalSSDBudgetBytes returns the configured on-disk byte cap (0 when unarmed).
func (t *Tree) LocalSSDBudgetBytes() int64 {
	if t == nil {
		return 0
	}
	return t.localSSDBudget
}

// LocalSSDReferencedBytes returns the on-disk bytes currently referenced by attached
// ssdSnapshot refs (the tree-side view of the tier's footprint).
func (t *Tree) LocalSSDReferencedBytes() int64 {
	if t == nil {
		return 0
	}
	return t.localSSDSnapshotBytes
}

func (t *Tree) localSSDScope(ns string) string { return t.localSSDModelID + "\x00" + ns }

func (t *Tree) localSSDDigest(ns string, n *node) string {
	tokenDigest := cachemeta.DigestTokenIDs(n.tokens())
	return cachemeta.DigestBytes([]byte(localSSDDigestPrefix + t.localSSDScope(ns) + "\x00" + tokenDigest))
}

// namespaceOf finds which namespace root owns n by DFS over every namespace subtree.
// The spill site knows the victim's namespace directly, but the lookup path does not
// carry it; the walk runs only on the rare L1+L2-missed restore probe over a handful of
// namespaces, so per-root DFS is cheap enough there.
func (t *Tree) namespaceOf(n *node) string {
	if t == nil || n == nil {
		return ""
	}
	found, foundFlag := "", false
	t.forEachRootNS(func(ns string, r *node) {
		if foundFlag {
			return
		}
		var walk func(m *node)
		walk = func(m *node) {
			if foundFlag || m == nil {
				return
			}
			if m == n {
				found, foundFlag = ns, true
				return
			}
			for _, child := range m.children {
				walk(child)
			}
		}
		walk(r)
	})
	return found
}

// spillHostSnapshotToSSD demotes one host-DRAM L2 image to the local-SSD overflow
// tier just before L2 pressure releases it. Best-effort by contract: every failure
// path (serialization refused, budget full, disk error) leaves the node without an
// SSD reference and returns silently so the caller's eviction proceeds unchanged.
func (t *Tree) spillHostSnapshotToSSD(ns string, n *node) {
	if !t.LocalSSDSnapshotEnabled() || n == nil || n.hostSnapshot == nil {
		return
	}
	if n.ssdSnapshot != nil {
		return // already demoted under this digest
	}
	if n.remoteSnapshot != nil {
		return // the remote tier already preserves this prefix; no duplicate copy on disk
	}
	payload, err := n.hostSnapshot.MarshalBinary()
	if err != nil {
		t.l3LocalFaults++
		return
	}
	digest := t.localSSDDigest(ns, n)
	scope := t.localSSDScope(ns)
	envelope := encodeRemoteSnapshotEnvelope(digest, scope, n.plen, payload)
	if t.localSSDSnapshotBytes+int64(len(envelope)) > t.localSSDBudget {
		return // over budget: drop rather than grow the tier past its cap
	}
	if err := t.localSSDStore.Put(context.Background(), digest, envelope); err != nil {
		t.l3LocalFaults++
		return
	}
	n.ssdSnapshot = &ssdSnapshotRef{digest: digest, scope: scope, tokens: n.plen, bytes: int64(len(envelope))}
	t.localSSDSnapshotBytes += int64(len(envelope))
	t.l3LocalStageBytes += int64(len(envelope))
}

// restoreSnapshotFromLocalSSD recovers one demoted prefix from the local-SSD overflow
// tier. It verifies the stored envelope against the node's CURRENT digest/scope/token
// length before decoding, so a stale reference (the node was split or re-keyed after
// the spill) reads as a clean miss instead of restoring foreign bytes.
func (t *Tree) restoreSnapshotFromLocalSSD(ctx context.Context, n *node) (*model.PrefixSnapshot, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	ref := n.ssdSnapshot
	if ref == nil {
		return nil, false, nil
	}
	ns := t.namespaceOf(n)
	wantDigest := t.localSSDDigest(ns, n)
	wantScope := t.localSSDScope(ns)
	if ref.digest != wantDigest || ref.scope != wantScope || ref.tokens != n.plen {
		// Stale reference: the node's key changed after the spill. Drop it cleanly.
		t.releaseSSDSnapshot(n)
		return nil, false, nil
	}
	envelope, found, err := t.localSSDStore.Get(ctx, wantDigest)
	if err != nil {
		t.l3LocalFaults++
		return nil, false, fmt.Errorf("radixkv: local SSD L3 get: %w", err)
	}
	if !found {
		t.releaseSSDSnapshot(n)
		if n.snapshot == nil && n.hostSnapshot == nil && n.remoteSnapshot == nil {
			n.cachedLogits = nil
		}
		t.invalidateRecordIfNoLocalCopy(n)
		return nil, false, nil
	}
	t.l3LocalRestoreBytes += int64(len(envelope))
	payload, err := decodeRemoteSnapshotEnvelope(envelope, wantDigest, wantScope, n.plen)
	if err != nil {
		t.l3LocalFaults++
		return nil, false, err
	}
	host, err := model.DecodeHostPrefixSnapshot(payload, t.localSSDBackend, t.localSSDConfig)
	if err != nil {
		t.l3LocalFaults++
		return nil, false, err
	}
	if err := ctx.Err(); err != nil {
		host.Close()
		return nil, false, err
	}
	snap, err := host.Restore()
	host.Close()
	if err != nil {
		t.l3LocalFaults++
		return nil, false, err
	}
	// Native restoration may finish after cancellation. Retire its acquired
	// owner before either request lookup or explicit page-in can publish it.
	if err := ctx.Err(); err != nil {
		snap.Close()
		return nil, false, err
	}
	t.l3LocalHits++
	t.l3LocalHitTokens += n.plen
	return snap, true, nil
}

// releaseSSDSnapshot detaches the node's SSD reference and uncharges its on-disk
// bytes. The on-disk record itself remains until the store owner prunes it; the
// reference is what bounds the tree's accounting.
func (t *Tree) releaseSSDSnapshot(n *node) {
	if n == nil || n.ssdSnapshot == nil {
		return
	}
	t.localSSDSnapshotBytes -= n.ssdSnapshot.bytes
	n.ssdSnapshot = nil
}
