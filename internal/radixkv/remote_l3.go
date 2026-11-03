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
// cfg, including its nested metadata, is immutable caller-owned configuration,
// not a deep copy. Keep the supplied store/backend and configuration usable for
// every admitted job until it actually retires, including after detach or
// reconfiguration. Normal serve boot supplies the model's static Model.Cfg.
func (t *Tree) ConfigureRemoteSnapshotStore(store SnapshotStore, modelID string, backend compute.Backend, cfg model.Config) error {
	if t == nil || store == nil || backend == nil || modelID == "" {
		return errors.New("radixkv: incomplete remote prefix snapshot configuration")
	}
	if t.remoteConfigGeneration == ^uint64(0) {
		return ErrRemoteConfigExhausted
	}
	t.remoteConfigGeneration++
	t.remoteSnapshotStore = store
	t.remoteModelID = modelID
	t.remoteBackend = backend
	t.remoteConfig = cfg
	if t.remoteBreaker == nil {
		t.remoteBreaker = NewRemoteL3Breaker(DefaultBreakerConfig())
	}
	return nil
}

// AttachRemoteSnapshotStore attaches or reconfigures the remote snapshot store at runtime.
func (t *Tree) AttachRemoteSnapshotStore(store SnapshotStore, modelID string, backend compute.Backend, cfg model.Config) error {
	return t.ConfigureRemoteSnapshotStore(store, modelID, backend, cfg)
}

// DetachRemoteSnapshotStore detaches the remote snapshot store at runtime.
func (t *Tree) DetachRemoteSnapshotStore(ctx context.Context) error {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	if t == nil {
		return nil
	}
	if t.remoteConfigGeneration == ^uint64(0) {
		return ErrRemoteConfigExhausted
	}
	t.remoteConfigGeneration++
	t.remoteSnapshotStore = nil
	t.remoteModelID = ""
	t.remoteBackend = nil
	t.remoteConfig = model.Config{}
	return nil
}

// HasRemoteSnapshotStore returns true if a remote snapshot store is attached.
func (t *Tree) HasRemoteSnapshotStore() bool {
	return t != nil && t.remoteSnapshotStore != nil
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
	return t.stageSnapshotToRemote(ctx, digest, nil)
}

func (t *Tree) stageSnapshotToRemote(ctx context.Context, digest string, lock *remoteTransferLock) SnapshotTransfer {
	if t == nil || !t.RemoteSnapshotEnabled() {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "remote HTTP L3 disabled"}
	}
	ns, n := t.findSnapshotByDigestNS(digest)
	if n == nil || (n.snapshot == nil && n.hostSnapshot == nil) {
		return SnapshotTransfer{Outcome: SnapshotTransferMiss, Digest: digest, Reason: "local prefix snapshot absent"}
	}
	job, err := t.beginRemoteTransfer(ctx, ns, n, lock)
	if err != nil {
		return t.remoteStageFault(digest, n.plen, err, 0)
	}
	defer job.finish()
	host := n.hostSnapshot
	temporary := false
	if host == nil {
		host, err = n.snapshot.CloneToHost()
		if err != nil {
			return t.remoteStageFault(digest, job.tokens, err, 0)
		}
		temporary = true
	}
	if temporary {
		defer host.Close()
	}
	// Locked preparation copies bytes into an independent envelope. Neither a
	// node lease nor the borrowed host image is used as an unlocked byte owner.
	payload, err := host.MarshalBinary()
	if err != nil {
		return t.remoteStageFault(digest, job.tokens, err, 0)
	}
	envelope := encodeRemoteSnapshotEnvelope(job.digest, job.scope, job.tokens, payload)
	var elapsed int64
	err = job.execute(func() error {
		started := time.Now()
		defer func() { elapsed = time.Since(started).Nanoseconds() }()
		return job.store.Put(ctx, job.digest, envelope)
	})
	t.l3StageNanos += elapsed
	if err == nil {
		err = job.validate(ctx)
	}
	if err != nil {
		return t.remoteStageFault(digest, job.tokens, err, 0)
	}
	if n.remoteSnapshot != nil {
		t.remoteSnapshotBytes -= n.remoteSnapshot.bytes
	}
	n.remoteSnapshot = &remoteSnapshotRef{digest: job.digest, scope: job.scope, tokens: job.tokens, bytes: int64(len(envelope))}
	t.remoteSnapshotBytes += int64(len(envelope))
	t.l3StageBytes += int64(len(envelope))
	return SnapshotTransfer{Outcome: SnapshotTransferOK, Digest: digest, Positions: job.tokens, BytesMoved: int64(len(envelope))}
}

func (t *Tree) remoteStageFault(digest string, tokens int, err error, elapsed int64) SnapshotTransfer {
	t.l3Faults++
	t.l3StageFaults++
	t.l3StageNanos += elapsed
	return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Positions: tokens, Reason: "remote L3 stage: " + err.Error()}
}

func (t *Tree) restoreSnapshotFromRemote(ctx context.Context, ns string, n *node) (*model.PrefixSnapshot, bool, error) {
	return t.restoreSnapshotFromRemoteWithLock(ctx, ns, n, nil)
}

func (t *Tree) restoreSnapshotFromRemoteWithLock(ctx context.Context, ns string, n *node, lock *remoteTransferLock) (*model.PrefixSnapshot, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	ref := n.remoteSnapshot
	if ref == nil {
		return nil, false, nil
	}
	wantDigest, wantScope := t.snapshotDigest(ns, n), t.snapshotScope(ns)
	if ref.digest != wantDigest || ref.scope != wantScope || ref.tokens != n.plen {
		t.l3Faults++
		t.l3RestoreFaults++
		return nil, false, ErrRemoteSnapshotScope
	}
	job, err := t.beginRemoteTransfer(ctx, ns, n, lock)
	if errors.Is(err, ErrRemoteTransferBusy) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	defer job.finish()
	breaker := t.RemoteL3Breaker()
	allowed, isProbe := breaker.Allow()
	if !allowed {
		return nil, false, nil
	}
	var snap *model.PrefixSnapshot
	var found bool
	var envelope []byte
	var transportErr error
	var transportReturned bool
	var elapsed int64
	err = job.execute(func() error {
		func() {
			started := time.Now()
			defer func() { elapsed = time.Since(started).Nanoseconds() }()
			envelope, found, transportErr = job.store.Get(ctx, job.digest)
			transportReturned = true
		}()
		if transportErr == nil {
			transportErr = ctx.Err()
		}
		if transportErr != nil {
			return transportErr
		}
		if !found {
			return nil
		}
		payload, decodeErr := decodeRemoteSnapshotEnvelope(envelope, job.digest, job.scope, job.tokens)
		if decodeErr != nil {
			return decodeErr
		}
		host, decodeErr := model.DecodeHostPrefixSnapshot(payload, job.backend, job.config)
		if decodeErr != nil {
			return decodeErr
		}
		defer host.Close()
		if cancelErr := ctx.Err(); cancelErr != nil {
			return cancelErr
		}
		snap, decodeErr = host.Restore()
		if decodeErr != nil {
			return decodeErr
		}
		return ctx.Err()
	})
	t.l3RestoreNanos += elapsed
	if !transportReturned {
		transportErr = err
	}
	// A captured probe belongs to this job even if configuration changed while
	// unlocked. Retire that exact probe once, never a replacement breaker's probe.
	breaker.RecordResult(transportErr, isProbe)
	if transportReturned && transportErr == nil && found {
		t.l3RestoreBytes += int64(len(envelope))
	}
	if err == nil {
		err = job.validate(ctx)
	}
	if err != nil {
		job.discard(snap)
		// Reconfiguration or replacement is a cache miss, not permission to use
		// the old image and not a backend fault. The request can recompute safely.
		if errors.Is(err, ErrRemoteTransferStale) {
			return nil, false, nil
		}
		t.l3Faults++
		t.l3RestoreFaults++
		if transportErr != nil {
			return nil, false, fmt.Errorf("radixkv: remote L3 get: %w", err)
		}
		return nil, false, err
	}
	if !found {
		t.remoteSnapshotBytes -= ref.bytes
		n.remoteSnapshot = nil
		if n.snapshot == nil && n.hostSnapshot == nil {
			n.cachedLogits = nil
		}
		t.invalidateRecordIfNoLocalCopy(n)
		return nil, false, nil
	}
	t.l3Hits++
	t.l3HitTokens += job.tokens
	return snap, true, nil
}

// RestoreSnapshotFromRemote is the explicit capacity page-in twin of normal L3
// lookup. It installs the recovered PrefixSnapshot as the node's hot owner.
func (t *Tree) RestoreSnapshotFromRemote(ctx context.Context, digest string) SnapshotTransfer {
	return t.restoreSnapshotFromRemoteIntoTree(ctx, digest, nil)
}

func (t *Tree) restoreSnapshotFromRemoteIntoTree(ctx context.Context, digest string, lock *remoteTransferLock) SnapshotTransfer {
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
	snap, found, err := t.restoreSnapshotFromRemoteWithLock(ctx, ns, n, lock)
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
	if n == nil || n.hostSnapshot == nil || (n.snapshot == nil && n.remoteSnapshot == nil && n.ssdSnapshot == nil) {
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
