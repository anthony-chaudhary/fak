package radixkv

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// SnapshotPressureCandidate is one hot complete-prefix owner that can be
// staged into the physical host-DRAM L2 and then released from the device.
type SnapshotPressureCandidate struct {
	Digest      string
	Tokens      int
	DeviceBytes int64
	LastUsed    uint64
}

// SnapshotTransfer is the typed result of a host L2 stage/restore operation.
type SnapshotTransfer struct {
	Outcome    string
	Digest     string
	Positions  int
	BytesMoved int64
	Reason     string
}

const (
	SnapshotTransferOK    = "ok"
	SnapshotTransferMiss  = "miss"
	SnapshotTransferFault = "fault"
)

// HostL2Enabled reports whether this tree owns a bounded host-DRAM tier.
func (t *Tree) HostL2Enabled() bool {
	return t != nil && t.maxHostSnapshotBytes > 0
}

// PressuredSnapshotCandidates enumerates hot device payloads without copying
// them. The byte count is the physical device residency reclaimed by demotion.
func (t *Tree) PressuredSnapshotCandidates() (int64, []SnapshotPressureCandidate) {
	if t == nil || (!t.HostL2Enabled() && !t.RemoteSnapshotEnabled()) {
		return 0, nil
	}
	var resident int64
	var out []SnapshotPressureCandidate
	t.forEachNodeNS(func(ns string, n *node) {
		if n.snapshot == nil {
			return
		}
		_, device := n.snapshot.ResidencyBytes()
		if device <= 0 {
			return
		}
		resident += device
		out = append(out, SnapshotPressureCandidate{
			Digest:      t.snapshotDigest(ns, n),
			Tokens:      n.plen,
			DeviceBytes: device,
			LastUsed:    n.lastUsed,
		})
	})
	return resident, out
}

// StageSnapshotToHost copies a complete hot owner into process-local host DRAM.
// It deliberately leaves the source untouched; the capacity adapter evicts only
// after this returns OK.
func (t *Tree) StageSnapshotToHost(digest string) SnapshotTransfer {
	if t == nil || !t.HostL2Enabled() {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "host DRAM L2 disabled"}
	}
	n := t.findSnapshotByDigest(digest)
	if n == nil || n.snapshot == nil {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "hot prefix snapshot absent"}
	}
	if n.hostSnapshot != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferOK, Digest: digest, Positions: n.plen}
	}
	host, err := n.snapshot.CloneToHost()
	if err != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Positions: n.plen, Reason: err.Error()}
	}
	incoming := hostSnapshotResidentBytes(host)
	if incoming > t.maxHostSnapshotBytes || !t.makeHostSnapshotRoom(incoming, n) {
		host.Close()
		return SnapshotTransfer{
			Outcome:   SnapshotTransferFault,
			Digest:    digest,
			Positions: n.plen,
			Reason:    ErrHostSnapshotByteBudget.Error(),
		}
	}
	n.hostSnapshot = host
	t.hostSnapshotBytes += incoming
	moved := host.TransferBytes()
	t.l2StageBytes += moved
	return SnapshotTransfer{
		Outcome:    SnapshotTransferOK,
		Digest:     digest,
		Positions:  n.plen,
		BytesMoved: moved,
	}
}

// EvictHotSnapshot releases one digest-addressed hot owner while preserving a
// staged host image. It refuses to drop the sole complete copy.
func (t *Tree) EvictHotSnapshot(digest string) int {
	if t == nil {
		return 0
	}
	n := t.findSnapshotByDigest(digest)
	if n == nil || n.snapshot == nil || (n.hostSnapshot == nil && n.remoteSnapshot == nil) {
		return 0
	}
	positions := n.plen
	t.releaseHotSnapshot(n)
	return positions
}

// RestoreSnapshotFromHost materializes a new hot owner at the node. Normal
// request lookup uses the same HostPrefixSnapshot.Restore path but keeps the
// restored owner request-local; this method is the explicit capacity page-in.
func (t *Tree) RestoreSnapshotFromHost(digest string) SnapshotTransfer {
	if t == nil {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "prefix tree absent"}
	}
	n := t.findSnapshotByDigest(digest)
	if n == nil || n.hostSnapshot == nil {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "host prefix snapshot absent"}
	}
	if n.snapshot != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferOK, Digest: digest, Positions: n.plen}
	}
	snap, err := n.hostSnapshot.Restore()
	if err != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Positions: n.plen, Reason: err.Error()}
	}
	if err := t.installHotSnapshot(n, snap); err != nil {
		return SnapshotTransfer{
			Outcome:   SnapshotTransferFault,
			Digest:    digest,
			Positions: n.plen,
			Reason:    err.Error(),
		}
	}
	moved := n.hostSnapshot.TransferBytes()
	t.l2RestoreBytes += moved
	return SnapshotTransfer{
		Outcome:    SnapshotTransferOK,
		Digest:     digest,
		Positions:  n.plen,
		BytesMoved: moved,
	}
}

func (t *Tree) installHotSnapshot(n *node, snap *model.PrefixSnapshot) error {
	incoming := snapshotResidentBytes(snap, n.cachedLogits)
	if !t.makeSnapshotRoom(incoming, n) {
		snap.Close()
		return ErrSnapshotByteBudget
	}
	n.snapshot = snap
	t.snapshotBytes += incoming
	return nil
}

func hostSnapshotResidentBytes(snap *model.HostPrefixSnapshot) int64 {
	if snap == nil {
		return 0
	}
	return snap.ResidentBytes()
}

func (t *Tree) makeHostSnapshotRoom(delta int64, exclude *node) bool {
	if delta <= 0 {
		return true
	}
	for t.hostSnapshotBytes+delta > t.maxHostSnapshotBytes {
		victim := t.hostSnapshotVictim(exclude)
		if victim == nil {
			return false
		}
		t.releaseHostSnapshot(victim)
		t.l2Evictions++
	}
	return true
}

func (t *Tree) hostSnapshotVictim(exclude *node) *node {
	strat := t.evictionStrategy()
	if prep, ok := strat.(TreePreparer); ok {
		prep.PrepareTree(t)
	}
	var victim *node
	t.forEachNode(func(n *node) {
		if n == exclude || n.refs > 0 || n.hostSnapshot == nil {
			return
		}
		if victim == nil || strat.Priority(n).less(strat.Priority(victim)) {
			victim = n
		}
	})
	return victim
}

func (t *Tree) releaseHostSnapshot(n *node) {
	if n == nil || n.hostSnapshot == nil {
		return
	}
	t.hostSnapshotBytes -= hostSnapshotResidentBytes(n.hostSnapshot)
	n.hostSnapshot.Close()
	n.hostSnapshot = nil
	if n.snapshot == nil && n.remoteSnapshot == nil {
		n.cachedLogits = nil
	}
}

func (t *Tree) releaseSnapshotPayload(n *node) {
	t.releaseHotSnapshot(n)
	t.releaseHostSnapshot(n)
	t.releaseRemoteSnapshot(n)
}

func (t *Tree) findSnapshotByDigest(digest string) *node {
	_, n := t.findSnapshotByDigestNS(digest)
	return n
}

func (t *Tree) findSnapshotByDigestNS(digest string) (string, *node) {
	if digest == "" {
		return "", nil
	}
	var foundNS string
	var found *node
	t.forEachNodeNS(func(ns string, n *node) {
		if found != nil || (n.snapshot == nil && n.hostSnapshot == nil && n.remoteSnapshot == nil) {
			return
		}
		if t.snapshotDigest(ns, n) == digest {
			foundNS = ns
			found = n
		}
	})
	return foundNS, found
}

func (t *Tree) forEachNode(fn func(*node)) {
	t.forEachNodeNS(func(_ string, n *node) {
		fn(n)
	})
}

func (t *Tree) forEachNodeNS(fn func(ns string, n *node)) {
	if t == nil || fn == nil {
		return
	}
	var visit func(string, *node)
	visit = func(ns string, n *node) {
		if n == nil {
			return
		}
		if n.parent != nil {
			fn(ns, n)
		}
		for _, child := range n.children {
			visit(ns, child)
		}
	}
	t.forEachRootNS(visit)
}

func snapshotNodeDigest(ns string, n *node) string {
	tokenDigest := cachemeta.DigestTokenIDs(n.tokens())
	return cachemeta.DigestBytes([]byte(ns + "\x00" + tokenDigest))
}
