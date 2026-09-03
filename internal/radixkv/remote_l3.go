package radixkv

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	remoteSnapshotMagic   = "FKL3"
	remoteSnapshotVersion = uint16(1)
)

var (
	ErrRemoteSnapshotIntegrity = errors.New("radixkv: remote prefix snapshot integrity check failed")
	ErrRemoteSnapshotVersion   = errors.New("radixkv: unsupported remote prefix snapshot version")
	ErrRemoteSnapshotScope     = errors.New("radixkv: remote prefix snapshot scope mismatch")
	ErrRemoteSnapshotDigest    = errors.New("radixkv: remote prefix snapshot digest mismatch")
)

// SnapshotStore is the span-keyed durable K/V contract implemented by l3kv.Store.
// It lives here to keep radixkv independent of the concrete transport package.
type SnapshotStore interface {
	Put(context.Context, string, []byte) error
	Get(context.Context, string) ([]byte, bool, error)
}

type remoteSnapshotRef struct {
	digest string
	scope  string
	tokens int
	bytes  int64
}

// ConfigureRemoteSnapshotStore installs the production L3 byte owner on a tree.
// modelID and cfg are the scope/geometry gates checked before recovered bytes can
// become a live PrefixSnapshot; backend is the physical install target.
func (t *Tree) ConfigureRemoteSnapshotStore(store SnapshotStore, modelID string, backend compute.Backend, cfg model.Config) error {
	if t == nil || store == nil || backend == nil || modelID == "" {
		return errors.New("radixkv: incomplete remote prefix snapshot configuration")
	}
	t.remoteSnapshotStore = store
	t.remoteModelID = modelID
	t.remoteBackend = backend
	t.remoteConfig = cfg
	if t.remoteBreaker == nil {
		t.remoteBreaker = NewRemoteL3Breaker(DefaultBreakerConfig())
	}
	return nil
}

// RemoteL3Breaker returns the circuit breaker guarding remote L3 snapshot reads.
func (t *Tree) RemoteL3Breaker() *RemoteL3Breaker {
	if t == nil {
		return nil
	}
	if t.remoteBreaker == nil {
		t.remoteBreaker = NewRemoteL3Breaker(DefaultBreakerConfig())
	}
	return t.remoteBreaker
}

// SetRemoteL3Breaker installs an explicit circuit breaker on the tree.
func (t *Tree) SetRemoteL3Breaker(b *RemoteL3Breaker) {
	if t != nil {
		t.remoteBreaker = b
	}
}

// ConfigureRemoteL3Breaker configures the tree's circuit breaker with cfg.
func (t *Tree) ConfigureRemoteL3Breaker(cfg BreakerConfig) {
	if t != nil {
		t.remoteBreaker = NewRemoteL3Breaker(cfg)
	}
}

func (t *Tree) RemoteSnapshotEnabled() bool {
	return t != nil && t.remoteSnapshotStore != nil && t.remoteBackend != nil && t.remoteModelID != ""
}

func (t *Tree) snapshotScope(ns string) string { return t.remoteModelID + "\x00" + ns }

func (t *Tree) snapshotDigest(ns string, n *node) string {
	if t.RemoteSnapshotEnabled() {
		tokenDigest := cachemeta.DigestTokenIDs(n.tokens())
		return cachemeta.DigestBytes([]byte("fak-prefix-l3/v1\x00" + t.snapshotScope(ns) + "\x00" + tokenDigest))
	}
	return snapshotNodeDigest(ns, n)
}

// StageSnapshotToRemote copies/uses the existing physical host image, encodes it,
// and confirms the l3kv Put before publishing the node's remote reference.
func (t *Tree) StageSnapshotToRemote(ctx context.Context, digest string) SnapshotTransfer {
	if t == nil || !t.RemoteSnapshotEnabled() {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "remote HTTP L3 disabled"}
	}
	ns, n := t.findSnapshotByDigestNS(digest)
	if n == nil || (n.snapshot == nil && n.hostSnapshot == nil) {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "local prefix snapshot absent"}
	}
	if err := ctx.Err(); err != nil {
		return t.remoteStageFault(digest, n.plen, err, 0)
	}
	host := n.hostSnapshot
	temporary := false
	if host == nil {
		var err error
		host, err = n.snapshot.CloneToHost()
		if err != nil {
			return t.remoteStageFault(digest, n.plen, err, 0)
		}
		temporary = true
	}
	if temporary {
		defer host.Close()
	}
	payload, err := host.MarshalBinary()
	if err != nil {
		return t.remoteStageFault(digest, n.plen, err, 0)
	}
	scope := t.snapshotScope(ns)
	envelope := encodeRemoteSnapshotEnvelope(digest, scope, n.plen, payload)
	started := time.Now()
	err = t.remoteSnapshotStore.Put(ctx, digest, envelope)
	elapsed := time.Since(started).Nanoseconds()
	t.l3StageNanos += elapsed
	if err != nil {
		return t.remoteStageFault(digest, n.plen, err, 0)
	}
	if n.remoteSnapshot != nil {
		t.remoteSnapshotBytes -= n.remoteSnapshot.bytes
	}
	n.remoteSnapshot = &remoteSnapshotRef{digest: digest, scope: scope, tokens: n.plen, bytes: int64(len(envelope))}
	t.remoteSnapshotBytes += int64(len(envelope))
	t.l3StageBytes += int64(len(envelope))
	return SnapshotTransfer{Outcome: SnapshotTransferOK, Digest: digest, Positions: n.plen, BytesMoved: int64(len(envelope))}
}

func (t *Tree) remoteStageFault(digest string, tokens int, err error, elapsed int64) SnapshotTransfer {
	t.l3Faults++
	t.l3StageFaults++
	t.l3StageNanos += elapsed
	return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Positions: tokens, Reason: "remote L3 stage: " + err.Error()}
}

func (t *Tree) restoreSnapshotFromRemote(ctx context.Context, ns string, n *node) (*model.PrefixSnapshot, bool, error) {
	ref := n.remoteSnapshot
	if ref == nil {
		return nil, false, nil
	}
	wantDigest := t.snapshotDigest(ns, n)
	wantScope := t.snapshotScope(ns)
	if ref.digest != wantDigest || ref.scope != wantScope || ref.tokens != n.plen {
		t.l3Faults++
		t.l3RestoreFaults++
		return nil, false, ErrRemoteSnapshotScope
	}
	breaker := t.RemoteL3Breaker()
	allowed, isProbe := breaker.Allow()
	if !allowed {
		return nil, false, nil
	}
	started := time.Now()
	envelope, found, err := t.remoteSnapshotStore.Get(ctx, wantDigest)
	t.l3RestoreNanos += time.Since(started).Nanoseconds()
	breaker.RecordResult(err, isProbe)
	if err != nil {
		t.l3Faults++
		t.l3RestoreFaults++
		return nil, false, fmt.Errorf("radixkv: remote L3 get: %w", err)
	}
	if !found {
		t.remoteSnapshotBytes -= ref.bytes
		n.remoteSnapshot = nil
		if n.snapshot == nil && n.hostSnapshot == nil {
			n.cachedLogits = nil
		}
		return nil, false, nil
	}
	t.l3RestoreBytes += int64(len(envelope))
	payload, err := decodeRemoteSnapshotEnvelope(envelope, wantDigest, wantScope, n.plen)
	if err != nil {
		t.l3Faults++
		t.l3RestoreFaults++
		return nil, false, err
	}
	host, err := model.DecodeHostPrefixSnapshot(payload, t.remoteBackend, t.remoteConfig)
	if err != nil {
		t.l3Faults++
		t.l3RestoreFaults++
		return nil, false, err
	}
	snap, err := host.Restore()
	host.Close()
	if err != nil {
		t.l3Faults++
		t.l3RestoreFaults++
		return nil, false, err
	}
	t.l3Hits++
	t.l3HitTokens += n.plen
	return snap, true, nil
}

// RestoreSnapshotFromRemote is the explicit capacity page-in twin of normal L3
// lookup. It installs the recovered PrefixSnapshot as the node's hot owner.
func (t *Tree) RestoreSnapshotFromRemote(ctx context.Context, digest string) SnapshotTransfer {
	if t == nil || !t.RemoteSnapshotEnabled() {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "remote HTTP L3 disabled"}
	}
	ns, n := t.findSnapshotByDigestNS(digest)
	if n == nil || n.remoteSnapshot == nil {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "remote prefix snapshot absent"}
	}
	if n.snapshot != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferOK, Digest: digest, Positions: n.plen}
	}
	snap, found, err := t.restoreSnapshotFromRemote(ctx, ns, n)
	if err != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Positions: n.plen, Reason: err.Error()}
	}
	if !found {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Positions: n.plen, Reason: "remote prefix snapshot absent"}
	}
	if err := t.installHotSnapshot(n, snap); err != nil {
		return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Positions: n.plen, Reason: err.Error()}
	}
	return SnapshotTransfer{Outcome: SnapshotTransferOK, Digest: digest, Positions: n.plen, BytesMoved: n.remoteSnapshot.bytes}
}

// EvictHostSnapshot releases the process-local L2 image while retaining any
// verified L3 reference. It is the explicit local-loss/reclaim operation used by
// normal L2 pressure and by the remote-restore witness.
func (t *Tree) EvictHostSnapshot(digest string) int {
	if t == nil {
		return 0
	}
	n := t.findSnapshotByDigest(digest)
	if n == nil || n.hostSnapshot == nil || (n.snapshot == nil && n.remoteSnapshot == nil) {
		return 0
	}
	positions := n.plen
	t.releaseHostSnapshot(n)
	t.l2Evictions++
	return positions
}

func (t *Tree) releaseRemoteSnapshot(n *node) {
	if n == nil || n.remoteSnapshot == nil {
		return
	}
	t.remoteSnapshotBytes -= n.remoteSnapshot.bytes
	n.remoteSnapshot = nil
}

func encodeRemoteSnapshotEnvelope(digest, scope string, tokens int, payload []byte) []byte {
	var b bytes.Buffer
	b.WriteString(remoteSnapshotMagic)
	_ = binary.Write(&b, binary.BigEndian, remoteSnapshotVersion)
	writeEnvelopeString(&b, digest)
	writeEnvelopeString(&b, scope)
	_ = binary.Write(&b, binary.BigEndian, uint64(tokens))
	sum := sha256.Sum256(payload)
	b.Write(sum[:])
	_ = binary.Write(&b, binary.BigEndian, uint64(len(payload)))
	b.Write(payload)
	return b.Bytes()
}

func writeEnvelopeString(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.BigEndian, uint32(len(s)))
	b.WriteString(s)
}

func decodeRemoteSnapshotEnvelope(data []byte, digest, scope string, tokens int) ([]byte, error) {
	if len(data) < len(remoteSnapshotMagic)+2 || string(data[:len(remoteSnapshotMagic)]) != remoteSnapshotMagic {
		return nil, fmt.Errorf("%w: unknown magic", ErrRemoteSnapshotIntegrity)
	}
	off := len(remoteSnapshotMagic)
	version := binary.BigEndian.Uint16(data[off : off+2])
	off += 2
	if version != remoteSnapshotVersion {
		return nil, fmt.Errorf("%w %d (this build reads v%d)", ErrRemoteSnapshotVersion, version, remoteSnapshotVersion)
	}
	readString := func() (string, error) {
		if len(data)-off < 4 {
			return "", ErrRemoteSnapshotIntegrity
		}
		n := int(binary.BigEndian.Uint32(data[off : off+4]))
		off += 4
		if n > len(data)-off {
			return "", ErrRemoteSnapshotIntegrity
		}
		s := string(data[off : off+n])
		off += n
		return s, nil
	}
	gotDigest, err := readString()
	if err != nil {
		return nil, err
	}
	if gotDigest != digest {
		return nil, ErrRemoteSnapshotDigest
	}
	gotScope, err := readString()
	if err != nil {
		return nil, err
	}
	if gotScope != scope {
		return nil, ErrRemoteSnapshotScope
	}
	if len(data)-off < 8+sha256.Size+8 {
		return nil, ErrRemoteSnapshotIntegrity
	}
	gotTokens := binary.BigEndian.Uint64(data[off : off+8])
	off += 8
	if gotTokens != uint64(tokens) {
		return nil, ErrRemoteSnapshotScope
	}
	wantSum := data[off : off+sha256.Size]
	off += sha256.Size
	n := binary.BigEndian.Uint64(data[off : off+8])
	off += 8
	if n > uint64(len(data)-off) || n != uint64(len(data)-off) {
		return nil, ErrRemoteSnapshotIntegrity
	}
	payload := data[off:]
	gotSum := sha256.Sum256(payload)
	if !bytes.Equal(wantSum, gotSum[:]) {
		return nil, ErrRemoteSnapshotIntegrity
	}
	return payload, nil
}
