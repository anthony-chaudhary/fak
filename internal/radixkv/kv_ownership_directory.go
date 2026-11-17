// SPDX-License-Identifier: Apache-2.0
// KV ownership directory contract. Clean-room adaptation of the Redis CAS
// ownership directory mechanism described in basetenlabs/deepseek-ultrafast
// crates/kv-transfer/src/directory.rs:43-121 at
// 76766a97308bfd1658fb055e835d0f92de737dc3 (Elastic License 2.0,
// source-available; idea only). No upstream code is copied: the mechanism -
// one owner record per KV content digest carrying {epoch, seq, peer}, a
// mint-once epoch generation that fences late writers, and stale-view
// refusal - is re-expressed here as a pure host-side contract over an
// injectable store and clock, testable with no RDMA NIC.
package radixkv

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

// KVOwnershipDigestLen is the closed digest contract: a lowercase hex
// SHA-256, the same shape as the remote-transfer tier's content digests.
const KVOwnershipDigestLen = 64

// KVOwnershipMaxCheck bounds one Check batch, the host-side twin of the
// directory's bounded check script.
const KVOwnershipMaxCheck = 1024

var (
	// ErrKVOwnershipStale is returned when a claim or check presents a
	// revision that no longer matches the current record: ownership
	// changed, the record expired, or the directory restarted and its
	// epoch rolled. A producer that observes it has lost ownership and
	// stops publishing.
	ErrKVOwnershipStale = errors.New("radixkv: KV ownership revision is stale")
	// ErrKVOwnershipDigest is returned for a digest outside the closed
	// 64-lowercase-hex contract.
	ErrKVOwnershipDigest = errors.New("radixkv: invalid KV ownership digest")
	// ErrKVOwnershipPeer is returned when a claim names no owner. The
	// empty peer is reserved for unowned records.
	ErrKVOwnershipPeer = errors.New("radixkv: invalid KV ownership peer")
	// ErrKVOwnershipCheck is returned for a Check batch outside the
	// closed contract (oversized or carrying an invalid digest).
	ErrKVOwnershipCheck = errors.New("radixkv: invalid KV ownership check batch")
	// ErrKVOwnershipState is returned when the injected store holds
	// directory state that no longer decodes.
	ErrKVOwnershipState = errors.New("radixkv: corrupted KV ownership directory state")
	// ErrKVOwnershipConfig is returned when the directory is
	// constructed with an incomplete store, clock, or TTL.
	ErrKVOwnershipConfig = errors.New("radixkv: incomplete KV ownership directory configuration")
)

// KVOwnershipRevision is the CAS view of one owner record: the fencing epoch
// generation and the monotonically increasing claim sequence within it.
type KVOwnershipRevision struct {
	Epoch string
	Seq   uint64
}

// KVOwnershipRecord is the owner record for one KV content digest. Peer is
// "" while the record is unowned.
type KVOwnershipRecord struct {
	Revision KVOwnershipRevision
	Peer     string
}

// KVOwnershipStore is the injected durable K/V backend for directory state.
// The directory is the only writer and serializes every read-modify-write
// under its own lock; a distributed backend must preserve the same closed
// semantics.
type KVOwnershipStore interface {
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Set(ctx context.Context, key string, value []byte) error
}

// KVOwnershipClock is the injected time source for record expiry.
type KVOwnershipClock interface {
	Now() time.Time
}

// KVOwnershipCheckEntry is one (digest, expected revision) pair probed by
// Check.
type KVOwnershipCheckEntry struct {
	Digest   string
	Revision KVOwnershipRevision
}

// KVOwnershipCheckResult reports the current epoch and the indices of the
// entries whose revisions no longer match their records.
type KVOwnershipCheckResult struct {
	Epoch   string
	Invalid []int
}

type kvOwnershipEntry struct {
	revision KVOwnershipRevision
	peer     string
	updated  time.Time
}

// KVOwnershipDirectory is a pure host-side ownership contract for the
// remote-transfer tier. It keeps one owner record per KV content digest
// carrying {epoch, seq, peer}; the epoch is minted once per directory
// lifetime and fences every late writer still holding a rolled generation;
// a claim or check presenting a revision that no longer matches its record
// is refused as stale. The onboarded event channel of the borrowed
// mechanism is a transport concern and stays with the RDMA tier; this
// contract is the host-side authority a producer consults before publishing.
type KVOwnershipDirectory struct {
	store KVOwnershipStore
	clock KVOwnershipClock
	ttl   time.Duration
	mu    sync.Mutex
}

// NewKVOwnershipDirectory constructs a directory over the injected store and
// clock. ttl is the owner-record lifetime: a record not refreshed within ttl
// is re-initializable and its views are stale. store and clock must be
// non-nil and ttl positive.
func NewKVOwnershipDirectory(store KVOwnershipStore, clock KVOwnershipClock, ttl time.Duration) (*KVOwnershipDirectory, error) {
	if store == nil || clock == nil || ttl <= 0 {
		return nil, ErrKVOwnershipConfig
	}
	return &KVOwnershipDirectory{store: store, clock: clock, ttl: ttl}, nil
}

// Lookup returns the owner record for digest, initializing it at the current
// epoch when it is absent, expired, or still carries a rolled epoch. It never
// reports a record from an older generation: a late reader sees the current
// epoch with a fresh sequence and no owner. A current record is refreshed so
// continuous access keeps it alive.
func (d *KVOwnershipDirectory) Lookup(ctx context.Context, digest string) (KVOwnershipRecord, error) {
	if d == nil {
		return KVOwnershipRecord{}, ErrKVOwnershipConfig
	}
	if !validKVOwnershipDigest(digest) {
		return KVOwnershipRecord{}, ErrKVOwnershipDigest
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	epoch, err := d.ensureEpoch(ctx)
	if err != nil {
		return KVOwnershipRecord{}, err
	}
	entry, ok, err := d.readRecord(ctx, digest)
	if err != nil {
		return KVOwnershipRecord{}, err
	}
	if !ok || entry.revision.Epoch != epoch || d.expired(entry.updated) {
		seq, err := d.nextSeq(ctx)
		if err != nil {
			return KVOwnershipRecord{}, err
		}
		entry = kvOwnershipEntry{revision: KVOwnershipRevision{Epoch: epoch, Seq: seq}, updated: d.clock.Now()}
		if err := d.writeRecord(ctx, digest, entry); err != nil {
			return KVOwnershipRecord{}, err
		}
	} else {
		entry.updated = d.clock.Now()
		if err := d.writeRecord(ctx, digest, entry); err != nil {
			return KVOwnershipRecord{}, err
		}
	}
	return KVOwnershipRecord{Revision: entry.revision, Peer: entry.peer}, nil
}

// Claim takes or refreshes ownership of digest for peer, CAS-ing on
// expected: it is refused with ErrKVOwnershipStale unless the current record
// exists, is unexpired, carries the current epoch, and matches expected
// exactly. A successful claim advances the sequence and records the new
// owner; a peer that still holds a superseded revision has lost ownership.
func (d *KVOwnershipDirectory) Claim(ctx context.Context, digest string, expected KVOwnershipRevision, peer string) (KVOwnershipRecord, error) {
	if d == nil {
		return KVOwnershipRecord{}, ErrKVOwnershipConfig
	}
	if !validKVOwnershipDigest(digest) {
		return KVOwnershipRecord{}, ErrKVOwnershipDigest
	}
	if peer == "" {
		return KVOwnershipRecord{}, ErrKVOwnershipPeer
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	epoch, err := d.ensureEpoch(ctx)
	if err != nil {
		return KVOwnershipRecord{}, err
	}
	entry, ok, err := d.readRecord(ctx, digest)
	if err != nil {
		return KVOwnershipRecord{}, err
	}
	if !ok || d.expired(entry.updated) || entry.revision.Epoch != epoch || entry.revision != expected {
		return KVOwnershipRecord{}, ErrKVOwnershipStale
	}
	seq, err := d.nextSeq(ctx)
	if err != nil {
		return KVOwnershipRecord{}, err
	}
	entry.revision = KVOwnershipRevision{Epoch: epoch, Seq: seq}
	entry.peer = peer
	entry.updated = d.clock.Now()
	if err := d.writeRecord(ctx, digest, entry); err != nil {
		return KVOwnershipRecord{}, err
	}
	return KVOwnershipRecord{Revision: entry.revision, Peer: entry.peer}, nil
}

// Check probes a batch of (digest, expected revision) pairs against the
// current records and returns the current epoch plus the indices of the
// stale views. A valid entry refreshes its record's expiry; a stale entry is
// the signal for its producer to stop publishing. Missing records are stale
// without being initialized: Check probes, Lookup initializes.
func (d *KVOwnershipDirectory) Check(ctx context.Context, entries []KVOwnershipCheckEntry) (KVOwnershipCheckResult, error) {
	if d == nil {
		return KVOwnershipCheckResult{}, ErrKVOwnershipConfig
	}
	if len(entries) > KVOwnershipMaxCheck {
		return KVOwnershipCheckResult{}, ErrKVOwnershipCheck
	}
	for _, entry := range entries {
		if !validKVOwnershipDigest(entry.Digest) {
			return KVOwnershipCheckResult{}, ErrKVOwnershipCheck
		}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	epoch, err := d.ensureEpoch(ctx)
	if err != nil {
		return KVOwnershipCheckResult{}, err
	}
	invalid := []int{}
	for i, entry := range entries {
		rec, ok, err := d.readRecord(ctx, entry.Digest)
		if err != nil {
			return KVOwnershipCheckResult{}, err
		}
		if !ok || d.expired(rec.updated) || rec.revision.Epoch != epoch || rec.revision != entry.Revision {
			invalid = append(invalid, i)
			continue
		}
		rec.updated = d.clock.Now()
		if err := d.writeRecord(ctx, entry.Digest, rec); err != nil {
			return KVOwnershipCheckResult{}, err
		}
	}
	return KVOwnershipCheckResult{Epoch: epoch, Invalid: invalid}, nil
}

func validKVOwnershipDigest(digest string) bool {
	if len(digest) != KVOwnershipDigestLen {
		return false
	}
	for i := 0; i < len(digest); i++ {
		c := digest[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

const (
	kvOwnershipEpochKey = "fak-kv-ownership/v1:epoch"
	kvOwnershipSeqKey   = "fak-kv-ownership/v1:sequence"
	kvOwnershipOwnerKey = "fak-kv-ownership/v1:owner:"
)

func kvOwnershipRecordKey(digest string) string { return kvOwnershipOwnerKey + digest }

// ensureEpoch returns the current fencing generation, minting and persisting
// one when the directory state is absent (a fresh or restarted directory).
// A rolled epoch is what fences late writers: every record still carrying
// the old generation is stale until re-initialized. Every caller holds d.mu.
func (d *KVOwnershipDirectory) ensureEpoch(ctx context.Context) (string, error) {
	raw, ok, err := d.store.Get(ctx, kvOwnershipEpochKey)
	if err != nil {
		return "", err
	}
	if ok && len(raw) > 0 {
		return string(raw), nil
	}
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	epoch := hex.EncodeToString(b[:])
	if err := d.store.Set(ctx, kvOwnershipEpochKey, []byte(epoch)); err != nil {
		return "", err
	}
	return epoch, nil
}

// nextSeq advances the directory-wide claim counter. Every caller holds
// d.mu.
func (d *KVOwnershipDirectory) nextSeq(ctx context.Context) (uint64, error) {
	var seq uint64
	raw, ok, err := d.store.Get(ctx, kvOwnershipSeqKey)
	if err != nil {
		return 0, err
	}
	if ok {
		if len(raw) != 8 {
			return 0, ErrKVOwnershipState
		}
		seq = binary.BigEndian.Uint64(raw)
	}
	seq++
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], seq)
	if err := d.store.Set(ctx, kvOwnershipSeqKey, b[:]); err != nil {
		return 0, err
	}
	return seq, nil
}

// expired reports whether entry's record has outlived the directory TTL.
// Every caller holds d.mu.
func (d *KVOwnershipDirectory) expired(updated time.Time) bool {
	return d.clock.Now().Sub(updated) > d.ttl
}

func (d *KVOwnershipDirectory) readRecord(ctx context.Context, digest string) (kvOwnershipEntry, bool, error) {
	raw, ok, err := d.store.Get(ctx, kvOwnershipRecordKey(digest))
	if err != nil || !ok {
		return kvOwnershipEntry{}, ok, err
	}
	entry, err := decodeKVOwnershipEntry(raw)
	if err != nil {
		return kvOwnershipEntry{}, false, err
	}
	return entry, true, nil
}

func (d *KVOwnershipDirectory) writeRecord(ctx context.Context, digest string, entry kvOwnershipEntry) error {
	return d.store.Set(ctx, kvOwnershipRecordKey(digest), encodeKVOwnershipEntry(entry))
}

const (
	kvOwnershipMagic   = "FKOV"
	kvOwnershipVersion = uint16(1)
)

func encodeKVOwnershipEntry(entry kvOwnershipEntry) []byte {
	var b bytes.Buffer
	b.WriteString(kvOwnershipMagic)
	_ = binary.Write(&b, binary.BigEndian, kvOwnershipVersion)
	_ = binary.Write(&b, binary.BigEndian, uint32(len(entry.revision.Epoch)))
	b.WriteString(entry.revision.Epoch)
	_ = binary.Write(&b, binary.BigEndian, entry.revision.Seq)
	_ = binary.Write(&b, binary.BigEndian, uint32(len(entry.peer)))
	b.WriteString(entry.peer)
	_ = binary.Write(&b, binary.BigEndian, int64(entry.updated.UnixNano()))
	return b.Bytes()
}

func decodeKVOwnershipEntry(data []byte) (kvOwnershipEntry, error) {
	var entry kvOwnershipEntry
	if len(data) < len(kvOwnershipMagic)+2+4 {
		return entry, ErrKVOwnershipState
	}
	if string(data[:len(kvOwnershipMagic)]) != kvOwnershipMagic {
		return entry, ErrKVOwnershipState
	}
	off := len(kvOwnershipMagic)
	if binary.BigEndian.Uint16(data[off:off+2]) != kvOwnershipVersion {
		return entry, ErrKVOwnershipState
	}
	off += 2
	readLen := func() (int, error) {
		if len(data)-off < 4 {
			return 0, ErrKVOwnershipState
		}
		n := int(binary.BigEndian.Uint32(data[off : off+4]))
		off += 4
		if n > len(data)-off {
			return 0, ErrKVOwnershipState
		}
		return n, nil
	}
	epochLen, err := readLen()
	if err != nil {
		return entry, err
	}
	entry.revision.Epoch = string(data[off : off+epochLen])
	off += epochLen
	if len(data)-off < 8+4 {
		return entry, ErrKVOwnershipState
	}
	entry.revision.Seq = binary.BigEndian.Uint64(data[off : off+8])
	off += 8
	peerLen, err := readLen()
	if err != nil {
		return entry, err
	}
	entry.peer = string(data[off : off+peerLen])
	off += peerLen
	if len(data)-off < 8 {
		return entry, ErrKVOwnershipState
	}
	entry.updated = time.Unix(0, int64(binary.BigEndian.Uint64(data[off:off+8]))).UTC()
	if off+8 != len(data) {
		return entry, ErrKVOwnershipState
	}
	return entry, nil
}
