// SPDX-License-Identifier: Apache-2.0
// Closed-contract tests for the KV ownership directory: owner records
// keyed by content digest carrying {epoch, seq, peer}, epoch fencing of
// late writers, and stale-view refusal. Software evidence only: an
// in-memory store and a fake clock stand in for Redis and the wall clock;
// no RDMA NIC, no hardware claim.
package radixkv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"testing"
	"time"
)

type kvOwnershipFakeStore struct {
	mu   sync.Mutex
	data map[string][]byte
}

func newKVOwnershipFakeStore() *kvOwnershipFakeStore {
	return &kvOwnershipFakeStore{data: map[string][]byte{}}
}

func (s *kvOwnershipFakeStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.data[key]
	if !ok {
		return nil, false, nil
	}
	out := make([]byte, len(v))
	copy(out, v)
	return out, true, nil
}

func (s *kvOwnershipFakeStore) Set(_ context.Context, key string, value []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]byte, len(value))
	copy(out, value)
	s.data[key] = out
	return nil
}

func (s *kvOwnershipFakeStore) clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = map[string][]byte{}
}

type kvOwnershipFakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *kvOwnershipFakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *kvOwnershipFakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func kvOwnershipDigest(t *testing.T, seed string) string {
	t.Helper()
	sum := sha256.Sum256([]byte(seed))
	return hex.EncodeToString(sum[:])
}

func TestKVOwnershipDirectoryLookupInitializesUnownedRecord(t *testing.T) {
	store := newKVOwnershipFakeStore()
	clock := &kvOwnershipFakeClock{now: time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)}
	dir, err := NewKVOwnershipDirectory(store, clock, time.Hour)
	if err != nil {
		t.Fatalf("NewKVOwnershipDirectory: %v", err)
	}
	ctx := context.Background()
	digest := kvOwnershipDigest(t, "prefix-a")

	rec, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if rec.Peer != "" {
		t.Fatalf("fresh record must be unowned, got peer %q", rec.Peer)
	}
	if rec.Revision.Seq != 1 {
		t.Fatalf("first initialization must carry sequence 1, got %d", rec.Revision.Seq)
	}
	if rec.Revision.Epoch == "" {
		t.Fatal("record must carry the fencing epoch")
	}

	again, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("second Lookup: %v", err)
	}
	if again != rec {
		t.Fatalf("current record must be stable across Lookups: %+v vs %+v", again, rec)
	}

	// A new directory object over the same store resumes the same epoch:
	// the generation is minted once per directory lifetime, not per object.
	resumed, err := NewKVOwnershipDirectory(store, clock, time.Hour)
	if err != nil {
		t.Fatalf("resumed directory: %v", err)
	}
	rec3, err := resumed.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("resumed Lookup: %v", err)
	}
	if rec3.Revision.Epoch != rec.Revision.Epoch {
		t.Fatalf("epoch rolled over the same store: %q vs %q", rec3.Revision.Epoch, rec.Revision.Epoch)
	}
	if rec3 != rec {
		t.Fatalf("resumed record must match: %+v vs %+v", rec3, rec)
	}
}

func TestKVOwnershipDirectoryClaimCASAndStaleRefusal(t *testing.T) {
	store := newKVOwnershipFakeStore()
	clock := &kvOwnershipFakeClock{now: time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)}
	dir, err := NewKVOwnershipDirectory(store, clock, time.Hour)
	if err != nil {
		t.Fatalf("NewKVOwnershipDirectory: %v", err)
	}
	ctx := context.Background()
	digest := kvOwnershipDigest(t, "prefix-b")

	rev0, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	owned, err := dir.Claim(ctx, digest, rev0.Revision, "peer-a")
	if err != nil {
		t.Fatalf("Claim by current owner: %v", err)
	}
	if owned.Peer != "peer-a" {
		t.Fatalf("claim must record the owner, got %q", owned.Peer)
	}
	if owned.Revision.Epoch != rev0.Revision.Epoch {
		t.Fatalf("claim must keep the current epoch: %q vs %q", owned.Revision.Epoch, rev0.Revision.Epoch)
	}
	if owned.Revision.Seq != rev0.Revision.Seq+1 {
		t.Fatalf("claim must advance the sequence once: %d vs %d", owned.Revision.Seq, rev0.Revision.Seq)
	}

	// The superseded revision is a stale view: the CAS refuses it and the
	// record is unchanged.
	if _, err := dir.Claim(ctx, digest, rev0.Revision, "peer-b"); err != ErrKVOwnershipStale {
		t.Fatalf("stale claim must refuse with the stale sentinel, got %v", err)
	}
	seen, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup after refused claim: %v", err)
	}
	if seen != owned {
		t.Fatalf("refused claim must not change the record: %+v vs %+v", seen, owned)
	}

	// A fresh view lets the other peer take ownership; the old owner's view
	// is now stale.
	rev1, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("fresh Lookup: %v", err)
	}
	taken, err := dir.Claim(ctx, digest, rev1.Revision, "peer-b")
	if err != nil {
		t.Fatalf("Claim by fresh view: %v", err)
	}
	if taken.Peer != "peer-b" || taken.Revision.Seq != rev1.Revision.Seq+1 {
		t.Fatalf("takeover must record the new owner at the next sequence: %+v", taken)
	}
	if _, err := dir.Claim(ctx, digest, rev0.Revision, "peer-a"); err != ErrKVOwnershipStale {
		t.Fatalf("superseded owner must be refused, got %v", err)
	}
	res, err := dir.Check(ctx, []KVOwnershipCheckEntry{{Digest: digest, Revision: rev0.Revision}})
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if len(res.Invalid) != 1 || res.Invalid[0] != 0 {
		t.Fatalf("superseded view must be the only stale entry, got %v", res.Invalid)
	}
}

func TestKVOwnershipDirectoryEpochFencesLateWriters(t *testing.T) {
	store := newKVOwnershipFakeStore()
	clock := &kvOwnershipFakeClock{now: time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)}
	dir, err := NewKVOwnershipDirectory(store, clock, time.Hour)
	if err != nil {
		t.Fatalf("NewKVOwnershipDirectory: %v", err)
	}
	ctx := context.Background()
	digest := kvOwnershipDigest(t, "prefix-c")

	rev0, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	owned, err := dir.Claim(ctx, digest, rev0.Revision, "peer-a")
	if err != nil {
		t.Fatalf("Claim: %v", err)
	}

	// A directory restart clears its state and mints a new epoch; every
	// view still carrying the old generation is fenced.
	store.clear()
	rolled, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup after restart: %v", err)
	}
	if rolled.Revision.Epoch == owned.Revision.Epoch {
		t.Fatal("restart must mint a new epoch generation")
	}
	if rolled.Peer != "" || rolled.Revision.Seq != 1 {
		t.Fatalf("rolled record must be unowned at sequence 1: %+v", rolled)
	}

	// The late writer's claim is fenced even though its revision matched
	// the old record exactly.
	if _, err := dir.Claim(ctx, digest, owned.Revision, "peer-a"); err != ErrKVOwnershipStale {
		t.Fatalf("late writer must be fenced, got %v", err)
	}
	res, err := dir.Check(ctx, []KVOwnershipCheckEntry{{Digest: digest, Revision: owned.Revision}})
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if res.Epoch != rolled.Revision.Epoch {
		t.Fatalf("Check must report the current epoch: %q vs %q", res.Epoch, rolled.Revision.Epoch)
	}
	if len(res.Invalid) != 1 || res.Invalid[0] != 0 {
		t.Fatalf("old-epoch view must be stale, got %v", res.Invalid)
	}

	// The current generation works normally after the roll.
	fresh, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("fresh Lookup: %v", err)
	}
	claimed, err := dir.Claim(ctx, digest, fresh.Revision, "peer-b")
	if err != nil {
		t.Fatalf("Claim at the new epoch: %v", err)
	}
	if claimed.Peer != "peer-b" || claimed.Revision.Epoch != rolled.Revision.Epoch {
		t.Fatalf("post-roll claim must land at the new epoch: %+v", claimed)
	}
}

func TestKVOwnershipDirectoryCheckReportsStaleViews(t *testing.T) {
	store := newKVOwnershipFakeStore()
	clock := &kvOwnershipFakeClock{now: time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)}
	dir, err := NewKVOwnershipDirectory(store, clock, time.Hour)
	if err != nil {
		t.Fatalf("NewKVOwnershipDirectory: %v", err)
	}
	ctx := context.Background()
	digestA := kvOwnershipDigest(t, "prefix-d")
	digestB := kvOwnershipDigest(t, "prefix-e")
	revA, err := dir.Lookup(ctx, digestA)
	if err != nil {
		t.Fatalf("Lookup A: %v", err)
	}
	revB, err := dir.Lookup(ctx, digestB)
	if err != nil {
		t.Fatalf("Lookup B: %v", err)
	}
	ownedA, err := dir.Claim(ctx, digestA, revA.Revision, "peer-a")
	if err != nil {
		t.Fatalf("Claim A: %v", err)
	}
	ownedB, err := dir.Claim(ctx, digestB, revB.Revision, "peer-b")
	if err != nil {
		t.Fatalf("Claim B: %v", err)
	}

	res, err := dir.Check(ctx, []KVOwnershipCheckEntry{
		{Digest: digestA, Revision: ownedA.Revision},
		{Digest: digestB, Revision: ownedB.Revision},
	})
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if len(res.Invalid) != 0 {
		t.Fatalf("current views must not be stale, got %v", res.Invalid)
	}

	// Supersede digestA's record; only its view goes stale.
	revA2, err := dir.Lookup(ctx, digestA)
	if err != nil {
		t.Fatalf("Lookup A2: %v", err)
	}
	if _, err := dir.Claim(ctx, digestA, revA2.Revision, "peer-c"); err != nil {
		t.Fatalf("Claim A2: %v", err)
	}
	res, err = dir.Check(ctx, []KVOwnershipCheckEntry{
		{Digest: digestA, Revision: ownedA.Revision},
		{Digest: digestB, Revision: ownedB.Revision},
	})
	if err != nil {
		t.Fatalf("Check after takeover: %v", err)
	}
	if len(res.Invalid) != 1 || res.Invalid[0] != 0 {
		t.Fatalf("only the superseded view may be stale, got %v", res.Invalid)
	}

	// A missing record is stale without being initialized by Check.
	empty := kvOwnershipDigest(t, "prefix-f")
	res, err = dir.Check(ctx, []KVOwnershipCheckEntry{{Digest: empty, Revision: revB.Revision}})
	if err != nil {
		t.Fatalf("Check missing: %v", err)
	}
	if len(res.Invalid) != 1 || res.Invalid[0] != 0 {
		t.Fatalf("missing record must be stale, got %v", res.Invalid)
	}

	// Closed batch contract.
	if _, err := dir.Check(ctx, nil); err != nil {
		t.Fatalf("empty Check: %v", err)
	}
	oversized := make([]KVOwnershipCheckEntry, KVOwnershipMaxCheck+1)
	for i := range oversized {
		oversized[i] = KVOwnershipCheckEntry{Digest: digestA, Revision: revA.Revision}
	}
	if _, err := dir.Check(ctx, oversized); err != ErrKVOwnershipCheck {
		t.Fatalf("oversized Check must refuse with the batch sentinel, got %v", err)
	}
	if _, err := dir.Check(ctx, []KVOwnershipCheckEntry{{Digest: "not-a-digest"}}); err != ErrKVOwnershipCheck {
		t.Fatalf("invalid digest in Check must refuse with the batch sentinel, got %v", err)
	}
}

func TestKVOwnershipDirectoryTTLExpiry(t *testing.T) {
	store := newKVOwnershipFakeStore()
	clock := &kvOwnershipFakeClock{now: time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)}
	ttl := 2 * time.Hour
	dir, err := NewKVOwnershipDirectory(store, clock, ttl)
	if err != nil {
		t.Fatalf("NewKVOwnershipDirectory: %v", err)
	}
	ctx := context.Background()
	digest := kvOwnershipDigest(t, "prefix-g")

	rev0, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	owned, err := dir.Claim(ctx, digest, rev0.Revision, "peer-a")
	if err != nil {
		t.Fatalf("Claim: %v", err)
	}

	// Past the TTL the record is re-initializable and every old view is
	// stale; the epoch generation itself survives.
	clock.advance(3 * time.Hour)
	rolled, err := dir.Lookup(ctx, digest)
	if err != nil {
		t.Fatalf("Lookup after TTL: %v", err)
	}
	if rolled.Revision.Epoch != owned.Revision.Epoch {
		t.Fatalf("TTL expiry must not roll the epoch: %q vs %q", rolled.Revision.Epoch, owned.Revision.Epoch)
	}
	if rolled.Peer != "" || rolled.Revision.Seq != owned.Revision.Seq+1 {
		t.Fatalf("expired record must re-initialize unowned at the next sequence: %+v", rolled)
	}
	if _, err := dir.Claim(ctx, digest, owned.Revision, "peer-a"); err != ErrKVOwnershipStale {
		t.Fatalf("expired owner must be refused, got %v", err)
	}
	res, err := dir.Check(ctx, []KVOwnershipCheckEntry{{Digest: digest, Revision: owned.Revision}})
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if len(res.Invalid) != 1 || res.Invalid[0] != 0 {
		t.Fatalf("expired view must be stale, got %v", res.Invalid)
	}
}

func TestKVOwnershipDirectoryClosedInputContract(t *testing.T) {
	store := newKVOwnershipFakeStore()
	clock := &kvOwnershipFakeClock{now: time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)}
	ctx := context.Background()

	if _, err := NewKVOwnershipDirectory(nil, clock, time.Hour); err != ErrKVOwnershipConfig {
		t.Fatalf("nil store must refuse with the config sentinel, got %v", err)
	}
	if _, err := NewKVOwnershipDirectory(store, nil, time.Hour); err != ErrKVOwnershipConfig {
		t.Fatalf("nil clock must refuse with the config sentinel, got %v", err)
	}
	if _, err := NewKVOwnershipDirectory(store, clock, 0); err != ErrKVOwnershipConfig {
		t.Fatalf("non-positive TTL must refuse with the config sentinel, got %v", err)
	}

	dir, err := NewKVOwnershipDirectory(store, clock, time.Hour)
	if err != nil {
		t.Fatalf("NewKVOwnershipDirectory: %v", err)
	}
	good := kvOwnershipDigest(t, "prefix-h")
	for _, bad := range []string{"", "abc", good[:63], good + "0", "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789"} {
		if _, err := dir.Lookup(ctx, bad); err != ErrKVOwnershipDigest {
			t.Fatalf("Lookup(%q) must refuse with the digest sentinel, got %v", bad, err)
		}
		if _, err := dir.Claim(ctx, bad, KVOwnershipRevision{}, "peer-a"); err != ErrKVOwnershipDigest {
			t.Fatalf("Claim(%q) must refuse with the digest sentinel, got %v", bad, err)
		}
	}
	rev, err := dir.Lookup(ctx, good)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if _, err := dir.Claim(ctx, good, rev.Revision, ""); err != ErrKVOwnershipPeer {
		t.Fatalf("empty peer must refuse with the peer sentinel, got %v", err)
	}

	// The record blob is a closed binary contract: corrupt state is a
	// state error, never a silent re-initialization.
	if err := store.Set(ctx, kvOwnershipRecordKey(good), []byte("garbage")); err != nil {
		t.Fatalf("seed corrupted state: %v", err)
	}
	if _, err := dir.Lookup(ctx, good); err != ErrKVOwnershipState {
		t.Fatalf("corrupted record must surface the state sentinel, got %v", err)
	}
}
