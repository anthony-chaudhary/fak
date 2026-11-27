package radixkv

import (
	"context"
	"errors"
	"fmt"
	"sync"

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

// SnapshotToucher is an optional extension of SnapshotStore that renews the
// recency of already-staged keys without transferring payload bytes.
type SnapshotToucher interface {
	Touch(ctx context.Context, keys []string) error
}

// SnapshotDeleter is an optional extension of SnapshotStore that removes
// evicted records from the underlying store.
type SnapshotDeleter interface {
	Delete(ctx context.Context, keys []string) error
}

// SnapshotAsyncGetter is an optional extension of SnapshotStore that dispatches
// an asynchronous Get and invokes onComplete off the caller's critical path.
type SnapshotAsyncGetter interface {
	GetAsync(ctx context.Context, key string, onComplete func(payload []byte, found bool, err error))
}

// localSSDLowWatermarkRatio is the target occupancy fraction (80%) that
// watermark eviction drains down to when an incoming spill would exceed
// localSSDBudget.
const localSSDLowWatermarkRatio = 0.80

// touchLocalSSDAncestors batches a data-free Touch for every attached SSD
// snapshot along n's ancestor chain so an upper-tier hit keeps hot SSD entries
// from being pruned first. Best-effort: never fails an already-served hit.
func (t *Tree) touchLocalSSDAncestors(ctx context.Context, n *node) {
	if t == nil || n == nil || t.localSSDStore == nil {
		return
	}
	var digests []string
	for cur := n; cur != nil; cur = cur.parent {
		if cur.ssdSnapshot != nil && cur.ssdSnapshot.digest != "" {
			t.clock++
			cur.ssdSnapshot.lastTouch = t.clock
			digests = append(digests, cur.ssdSnapshot.digest)
		}
	}
	if len(digests) == 0 {
		return
	}
	if toucher, ok := t.localSSDStore.(SnapshotToucher); ok {
		_ = toucher.Touch(ctx, digests)
	}
}

// ssdSnapshotRef is a verified reference to a complete serialized HostPrefixSnapshot
// in the local-SSD overflow tier. Like a remote reference it owns no local payload
// bytes; its on-disk size counts against the tree's SSD budget while attached.
type ssdSnapshotRef struct {
	digest    string
	scope     string
	tokens    int
	bytes     int64
	lastTouch uint64
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
	incoming := int64(len(envelope))
	if incoming > t.localSSDBudget {
		return // single entry exceeds total SSD budget: cannot fit even after eviction
	}
	if t.localSSDSnapshotBytes+incoming > t.localSSDBudget {
		t.evictLocalSSDToLowWatermark(incoming)
	}
	if t.localSSDSnapshotBytes+incoming > t.localSSDBudget {
		return // still over budget after evicting all evictable entries
	}
	if err := t.localSSDStore.Put(context.Background(), digest, envelope); err != nil {
		t.l3LocalFaults++
		return
	}
	t.clock++
	n.ssdSnapshot = &ssdSnapshotRef{digest: digest, scope: scope, tokens: n.plen, bytes: incoming, lastTouch: t.clock}
	t.localSSDSnapshotBytes += incoming
	t.l3LocalStageBytes += incoming
}

// evictLocalSSDToLowWatermark evicts the coldest SSD-resident entries until
// referenced bytes are at or below both (localSSDBudget * lowWatermarkRatio) and
// (localSSDBudget - incoming), so the incoming spill fits without dropping.
func (t *Tree) evictLocalSSDToLowWatermark(incoming int64) {
	target := int64(float64(t.localSSDBudget) * localSSDLowWatermarkRatio)
	if maxAllowed := t.localSSDBudget - incoming; maxAllowed < target {
		target = maxAllowed
	}
	if target < 0 {
		target = 0
	}
	var evictedDigests []string
	for t.localSSDSnapshotBytes > target {
		var victim *node
		t.forEachRoot(func(r *node) {
			var walk func(m *node)
			walk = func(m *node) {
				if m == nil {
					return
				}
				if m.ssdSnapshot != nil {
					if victim == nil || m.ssdSnapshot.lastTouch < victim.ssdSnapshot.lastTouch ||
						(m.ssdSnapshot.lastTouch == victim.ssdSnapshot.lastTouch && m.ssdSnapshot.digest < victim.ssdSnapshot.digest) {
						victim = m
					}
				}
				for _, child := range m.children {
					walk(child)
				}
			}
			walk(r)
		})
		if victim == nil {
			break
		}
		evictedDigests = append(evictedDigests, victim.ssdSnapshot.digest)
		t.releaseSSDSnapshot(victim)
		t.l3LocalEvictions++
		if victim.snapshot == nil && victim.hostSnapshot == nil && victim.remoteSnapshot == nil {
			victim.cachedLogits = nil
		}
		t.invalidateRecordIfNoLocalCopy(victim)
	}
	if len(evictedDigests) > 0 {
		if deleter, ok := t.localSSDStore.(SnapshotDeleter); ok {
			_ = deleter.Delete(context.Background(), evictedDigests)
		}
	}
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
	t.touchLocalSSDAncestors(ctx, n)
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

// LocalSSDRestoreState reports the lifecycle state of a parked async SSD restore.
type LocalSSDRestoreState int

const (
	LocalSSDRestoreParked LocalSSDRestoreState = iota
	LocalSSDRestoreReady
	LocalSSDRestoreFailed
	LocalSSDRestoreRewound
)

var (
	ErrLocalSSDRestoreStillParked    = errors.New("radixkv: local SSD restore is still parked")
	ErrLocalSSDRestoreAlreadySettled = errors.New("radixkv: local SSD restore already settled")
)

// ParkedLocalSSDRestore represents an in-flight asynchronous SSD prefix restore
// whose request is parked (holding a tree lease on its lookup node) so other
// requests can continue scheduling while disk I/O runs.
type ParkedLocalSSDRestore struct {
	tree       *Tree
	lookupNode *node
	candidate  *node
	wantDigest string
	wantScope  string
	matched    int
	cancel     context.CancelFunc
	readyCh    chan struct{}

	mu            sync.Mutex
	state         LocalSSDRestoreState
	snap          *model.PrefixSnapshot
	found         bool
	err           error
	envelopeBytes int64
	settled       bool
}

// Done returns a channel that closes when the background SSD restore finishes
// (whether with a restored snapshot, a clean miss, or an error).
func (p *ParkedLocalSSDRestore) Done() <-chan struct{} { return p.readyCh }

// State returns the current lifecycle state of the parked restore.
func (p *ParkedLocalSSDRestore) State() LocalSSDRestoreState {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.state
}

// ParkLocalSSDRestore looks up tokens, pins the longest SSD-resident ancestor
// with a tree lease, and launches an asynchronous SSD restore so the caller can
// park the request and keep scheduling other work until Done() closes. Returns
// (nil, false, nil) when no valid SSD candidate exists.
func (t *Tree) ParkLocalSSDRestore(ctx context.Context, tokens []int) (*ParkedLocalSSDRestore, bool, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if !t.LocalSSDSnapshotEnabled() {
		return nil, false, nil
	}
	n, _ := t.Lookup(tokens)
	for candidate := n; candidate != nil; candidate = candidate.parent {
		ref := candidate.ssdSnapshot
		if ref == nil {
			continue
		}
		ns := t.namespaceOf(candidate)
		wantDigest := t.localSSDDigest(ns, candidate)
		wantScope := t.localSSDScope(ns)
		if ref.digest != wantDigest || ref.scope != wantScope || ref.tokens != candidate.plen {
			t.releaseSSDSnapshot(candidate)
			continue
		}
		jobCtx, cancel := context.WithCancel(ctx)
		p := &ParkedLocalSSDRestore{
			tree:       t,
			lookupNode: n,
			candidate:  candidate,
			wantDigest: wantDigest,
			wantScope:  wantScope,
			matched:    candidate.plen,
			cancel:     cancel,
			readyCh:    make(chan struct{}),
			state:      LocalSSDRestoreParked,
		}
		t.l3LocalParked++
		store := t.localSSDStore
		backend := t.localSSDBackend
		cfg := t.localSSDConfig
		plen := candidate.plen

		finish := func(envelope []byte, found bool, getErr error) {
			var (
				snap *model.PrefixSnapshot
				err  error
			)
			if getErr != nil {
				err = fmt.Errorf("radixkv: local SSD L3 get: %w", getErr)
			} else if found {
				var payload []byte
				payload, err = decodeRemoteSnapshotEnvelope(envelope, wantDigest, wantScope, plen)
				if err == nil {
					var host *model.HostPrefixSnapshot
					host, err = model.DecodeHostPrefixSnapshot(payload, backend, cfg)
					if err == nil {
						if cerr := jobCtx.Err(); cerr != nil {
							host.Close()
							err = cerr
						} else {
							snap, err = host.Restore()
							host.Close()
							if err == nil {
								if cerr := jobCtx.Err(); cerr != nil {
									snap.Close()
									snap = nil
									err = cerr
								}
							}
						}
					}
				}
			}
			p.mu.Lock()
			if p.settled {
				p.mu.Unlock()
				if snap != nil {
					snap.Close()
				}
				return
			}
			p.snap = snap
			p.found = found
			p.err = err
			p.envelopeBytes = int64(len(envelope))
			if err == nil && found && snap != nil {
				p.state = LocalSSDRestoreReady
			} else {
				p.state = LocalSSDRestoreFailed
			}
			p.mu.Unlock()
			close(p.readyCh)
		}

		if asyncGetter, ok := store.(SnapshotAsyncGetter); ok {
			asyncGetter.GetAsync(jobCtx, wantDigest, finish)
		} else {
			go func() {
				b, ok, err := store.Get(jobCtx, wantDigest)
				finish(b, ok, err)
			}()
		}
		return p, true, nil
	}
	t.Done(n)
	return nil, false, nil
}

// Readmit completes a finished parked restore on the tree goroutine. On
// success it returns (lookupNode, snap, matched, SnapshotTierLocalL3, nil) and
// transfers ownership of snap and the lookupNode lease to the caller. On
// failure or miss it rewinds the tree lease cleanly and returns the error.
func (p *ParkedLocalSSDRestore) Readmit() (*node, *model.PrefixSnapshot, int, SnapshotTier, error) {
	if p == nil {
		return nil, nil, 0, SnapshotTierMiss, ErrLocalSSDRestoreAlreadySettled
	}
	p.mu.Lock()
	if p.state == LocalSSDRestoreParked {
		p.mu.Unlock()
		return nil, nil, 0, SnapshotTierMiss, ErrLocalSSDRestoreStillParked
	}
	if p.settled {
		p.mu.Unlock()
		return nil, nil, 0, SnapshotTierMiss, ErrLocalSSDRestoreAlreadySettled
	}
	p.settled = true
	snap := p.snap
	p.snap = nil
	found := p.found
	err := p.err
	envBytes := p.envelopeBytes
	if err != nil || !found || snap == nil {
		p.state = LocalSSDRestoreRewound
	}
	p.mu.Unlock()

	p.cancel()
	t := p.tree
	if t.l3LocalParked > 0 {
		t.l3LocalParked--
	}
	if err != nil {
		t.l3LocalFaults++
		t.Done(p.lookupNode)
		return nil, nil, p.matched, SnapshotTierLocalL3, err
	}
	if !found || snap == nil {
		t.releaseSSDSnapshot(p.candidate)
		if p.candidate.snapshot == nil && p.candidate.hostSnapshot == nil && p.candidate.remoteSnapshot == nil {
			p.candidate.cachedLogits = nil
		}
		t.invalidateRecordIfNoLocalCopy(p.candidate)
		t.l3LocalMisses++
		t.Done(p.lookupNode)
		return nil, nil, 0, SnapshotTierMiss, nil
	}
	t.l3LocalRestoreBytes += envBytes
	t.l3LocalHits++
	t.l3LocalHitTokens += p.matched
	t.touchLocalSSDAncestors(context.Background(), p.candidate)
	return p.lookupNode, snap, p.matched, SnapshotTierLocalL3, nil
}

// Rewind aborts a parked restore and releases its tree lease and any unclaimed
// snapshot cleanly. Safe to call multiple times or after Readmit.
func (p *ParkedLocalSSDRestore) Rewind() {
	if p == nil {
		return
	}
	p.cancel()
	p.mu.Lock()
	if p.settled {
		p.mu.Unlock()
		return
	}
	p.settled = true
	snap := p.snap
	p.snap = nil
	p.state = LocalSSDRestoreRewound
	p.mu.Unlock()

	if snap != nil {
		snap.Close()
	}
	if p.tree.l3LocalParked > 0 {
		p.tree.l3LocalParked--
	}
	p.tree.Done(p.lookupNode)
}
