// Package blob is the v0.1 default backend behind every abi.Ref: a
// content-addressed (sha256) in-memory blob store. It provides the three things
// the frozen ABI leaves open as seams:
//
//   - abi.Resolver        — materialize bytes from a Ref / Put bytes -> Ref
//   - abi.RegionBackend   — the registered provider of that Resolver (the
//     zero-copy seam; v0.1 is a copy-CAS, a shared arena is a later swap)
//   - abi.PageOutBackend  — the context-MMU's page-out codec (cold/quarantined
//     results page out to a handle Ref and back)
//
// Content addressing is the load-bearing property: the digest IS the identity,
// so the vDSO (tier-2 cache) and the context-MMU (paged-out results) share one
// store and a byte-identical payload is stored once. Small payloads stay inline
// (RefInline) to avoid a store round-trip on the hot path.
package blob

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// InlineMax is the largest payload kept inline on the Ref itself. Anything larger
// is stored in the CAS and the Ref carries only the digest+handle.
const InlineMax = 256

// DefaultMaxBytes bounds the resident CAS so a long-running gateway cannot accrete
// distinct payloads without limit. It is generous — far above any single call's
// working set, so eviction only ever fires in the pathological unbounded-growth
// case — and only ever drops UNPINNED digests (transient call args/results no live
// holder still references). Override with FAK_BLOB_MAX_BYTES (0 = unbounded).
const DefaultMaxBytes = 1 << 30 // 1 GiB

// Store is a concurrency-safe, byte-bounded content-addressed blob store.
//
// Bounding is pin-aware: a digest a live holder will resolve LATER (a vDSO tier-2
// cache entry, a context-MMU held quarantine handle) is Pin'd, which protects it
// from eviction; everything else (transient gateway call args/results that are
// resolved only within their producing call) is evictable LRU/FIFO once the
// resident footprint exceeds maxBytes. Eviction never drops a pinned digest, so it
// can never break the vDSO soundness invariant ("a cache hit equals a fresh call")
// or a gated page-in. recall Sessions carry their OWN CAS (recall.Load), so they
// are unaffected by global eviction. maxBytes <= 0 disables eviction entirely
// (legacy append-only behavior).
type Store struct {
	mu       sync.RWMutex
	blobs    map[string][]byte        // digest -> bytes
	bytes    int64                    // total bytes resident in blobs (O(1) footprint tap)
	maxBytes int64                    // 0 => unbounded (no eviction)
	pins     map[string]int           // digest -> pin count (>0 => protected, kept OUT of lru)
	lru      *list.List               // evictable (unpinned) digests; front = most-recently inserted
	lruIndex map[string]*list.Element // digest -> its lru element (only unpinned, resident digests)
	puts     int64
	hits     int64 // Put of an already-present digest (dedup)
	// resolv is bumped from Resolve under the READ lock (Resolve is intentionally
	// concurrent — it only reads the blob map), so the counter itself must be atomic:
	// two readers holding the RLock would otherwise race on this plain increment. A
	// concurrent K-arm replay (turnbench.RunPolicyReplay) resolves the same shared
	// payload from several goroutines at once, which is exactly the path that trips it.
	resolv  int64
	evicted int64 // digests dropped by the byte bound
}

// New returns an empty store bounded by FAK_BLOB_MAX_BYTES (default DefaultMaxBytes).
func New() *Store {
	return newStore(maxBytesFromEnv())
}

// NewWithBudget returns an empty store whose resident CAS holds at most maxBytes of
// blob payload (a non-positive maxBytes falls back to DefaultMaxBytes). It is the seam
// the leak-regression test uses to exercise eviction with a small budget; the bound is
// still pin-aware, so a pinned digest is never the thing evicted.
func NewWithBudget(maxBytes int64) *Store {
	if maxBytes < 1 {
		maxBytes = DefaultMaxBytes
	}
	return newStore(maxBytes)
}

func newStore(maxBytes int64) *Store {
	return &Store{
		blobs:    map[string][]byte{},
		maxBytes: maxBytes,
		pins:     map[string]int{},
		lru:      list.New(),
		lruIndex: map[string]*list.Element{},
	}
}

func maxBytesFromEnv() int64 {
	if v, ok := os.LookupEnv("FAK_BLOB_MAX_BYTES"); ok {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
			return n
		}
	}
	return DefaultMaxBytes
}

// storeLocked inserts a NEW digest's bytes (caller holds s.mu and has confirmed the
// digest is absent), tracking the footprint, the evictable LRU, and then enforcing
// the byte bound. A digest pinned before it was stored stays out of the LRU.
func (s *Store) storeLocked(d string, b []byte) {
	s.blobs[d] = append([]byte(nil), b...)
	s.bytes += int64(len(b))
	if s.pins[d] == 0 {
		s.lruIndex[d] = s.lru.PushFront(d)
	}
	s.evictLocked()
}

// evictLocked drops unpinned digests (oldest first) until the resident footprint is
// within maxBytes. Pinned digests are never in lru, so they are never evicted; if
// only pinned digests remain, the footprint legitimately exceeds the bound and the
// loop stops (it bounds the leak, not the live working set). Caller holds s.mu.
func (s *Store) evictLocked() {
	if s.maxBytes <= 0 {
		return
	}
	for s.bytes > s.maxBytes {
		el := s.lru.Back()
		if el == nil {
			return // everything resident is pinned (live) — nothing safe to drop
		}
		d := el.Value.(string)
		s.lru.Remove(el)
		delete(s.lruIndex, d)
		if b, ok := s.blobs[d]; ok {
			s.bytes -= int64(len(b))
			delete(s.blobs, d)
		}
		s.evicted++
	}
}

// Pin protects a digest from eviction for as long as a live holder will resolve it
// later (a vDSO cache entry, a held quarantine handle). It is refcounted, so a
// digest shared by several holders (content dedup) survives until the LAST Unpin.
// A no-op for the empty digest. Safe to call before or after the bytes are stored.
func (s *Store) Pin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pins[digest]++
	if s.pins[digest] == 1 {
		if el, ok := s.lruIndex[digest]; ok { // was evictable -> protect it
			s.lru.Remove(el)
			delete(s.lruIndex, digest)
		}
	}
}

// Unpin releases one Pin. When the last holder unpins, the digest becomes evictable
// again (re-entered at the LRU front if still resident). A no-op if not pinned.
func (s *Store) Unpin(digest string) {
	if digest == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := s.pins[digest]
	if n <= 0 {
		return
	}
	if n == 1 {
		delete(s.pins, digest)
		if _, ok := s.blobs[digest]; ok {
			s.lruIndex[digest] = s.lru.PushFront(digest)
		}
		s.evictLocked() // it may now be the thing that puts us over budget
		return
	}
	s.pins[digest] = n - 1
}

// Digest is the canonical content address of b.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// PreparePut builds the addressable Ref header every content-addressed Store's Put
// shares — the digest, length, default taint/scope, and the InlineMax split. A
// payload at or below InlineMax rides inline on the Ref (no store round-trip) and
// is returned with inline=true and Kind=RefInline; a larger payload is returned
// with inline=false and Kind=RefBlob, leaving the caller to persist the bytes in
// its own backing store (memory map, disk, remote object endpoint) before handing
// the Ref back. It is the shared prologue hoisted out of blob/blobfs/blobhttp Put.
func PreparePut(b []byte) (r abi.Ref, inline bool) {
	r = abi.Ref{Digest: Digest(b), Len: int64(len(b)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}
	if len(b) <= InlineMax {
		r.Kind = abi.RefInline
		r.Inline = append([]byte(nil), b...)
		return r, true
	}
	r.Kind = abi.RefBlob
	return r, false
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref by resolving its
// bytes through res (the backend the handle belongs to). It is the byte-identical
// PageIn shared by every content-addressed Store (memory/disk/remote): the only
// per-backend difference is which Resolver reads the bytes, so callers pass their
// own Store as res. The returned Ref carries the handle's identity (digest, taint,
// scope) with the resolved bytes inline.
func PageIn(ctx context.Context, res abi.Resolver, handle abi.Ref) (abi.Ref, error) {
	b, err := res.Resolve(ctx, handle)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefInline, Digest: handle.Digest, Inline: b, Len: int64(len(b)), Taint: handle.Taint, Scope: handle.Scope}, nil
}

// Put stores b and returns an addressable Ref. Small payloads are returned
// inline; larger ones are stored in the CAS. A byte-identical payload is stored
// exactly once (content-addressed dedup) — the property the vDSO and MMU rely on.
func (s *Store) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	r, inline := PreparePut(b)
	if inline {
		return r, nil
	}
	d := r.Digest
	s.mu.Lock()
	defer s.mu.Unlock()
	s.puts++
	if _, ok := s.blobs[d]; ok {
		s.hits++
	} else {
		s.storeLocked(d, b)
	}
	return r, nil
}

// Resolve materializes the bytes a Ref points at.
func (s *Store) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefBlob, abi.RefRegion:
		s.mu.RLock()
		defer s.mu.RUnlock()
		atomic.AddInt64(&s.resolv, 1) // RLock allows concurrent Resolvers; the counter must be atomic
		b, ok := s.blobs[r.Digest]
		if !ok {
			return nil, fmt.Errorf("blob: unknown digest %s", r.Digest)
		}
		return append([]byte(nil), b...), nil
	default:
		return nil, fmt.Errorf("blob: unknown RefKind %d", r.Kind)
	}
}

// PageOut moves a (possibly inline) Ref's bytes into the CAS and returns a handle
// Ref that carries no inline bytes — the MMU's "evict to a pointer" primitive.
func (s *Store) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, r)
	if err != nil {
		return abi.Ref{}, err
	}
	d := Digest(b)
	s.mu.Lock()
	if _, ok := s.blobs[d]; !ok {
		s.storeLocked(d, b)
	}
	s.mu.Unlock()
	return abi.Ref{Kind: abi.RefBlob, Digest: d, Len: int64(len(b)), Taint: r.Taint, Scope: r.Scope}, nil
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref.
func (s *Store) PageIn(ctx context.Context, handle abi.Ref) (abi.Ref, error) {
	return PageIn(ctx, s, handle)
}

// Stats reports store activity (puts, dedup hits, resolves) for KPI taps.
func (s *Store) Stats() (puts, dedupHits, resolves int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// resolv is bumped atomically under the READ lock by concurrent Resolvers, so it
	// is read atomically here too (the RLock alone does not order it against them).
	return s.puts, s.hits, atomic.LoadInt64(&s.resolv)
}

// Len reports the number of distinct blobs resident in the CAS.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.blobs)
}

// Bytes reports the total bytes resident in the CAS — the live footprint of the
// append-only store, the surface a footprint KPI / the proc-resource guard's
// memory analogue alarms on when it grows without bound.
func (s *Store) Bytes() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.bytes
}

// Reset drops every stored blob, reclaiming all CAS memory. SAFE ONLY at a
// quiescent lifecycle boundary: every outstanding RefBlob (a vDSO cache entry, a
// held quarantine handle, a paged-out pointer) becomes unresolvable afterward, so
// the caller must guarantee no in-flight Ref will be resolved across the Reset.
// It is the lifecycle leaf's reclaim hook — the CAS dual of vDSO.BumpWorld — for a
// long-running gateway that would otherwise accrete distinct payloads for its
// whole lifetime. Counters are left intact (they are cumulative activity taps).
func (s *Store) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.blobs = map[string][]byte{}
	s.bytes = 0
	s.pins = map[string]int{}
	s.lru = list.New()
	s.lruIndex = map[string]*list.Element{}
}

// Evicted reports how many digests the byte bound has dropped (a KPI tap; a rising
// count means real working pressure or a leak the bound is now absorbing).
func (s *Store) Evicted() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.evicted
}

// Resident reports the current resident CAS size (blob count, total bytes) and the
// lifetime count of blobs dropped by the byte budget — the leak-sweep observability
// triple (blobCount/bytes plateau while evicted climbs). A convenience view over
// Len/Bytes/Evicted under a single lock.
func (s *Store) Resident() (blobCount int, bytes, evicted int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.blobs), s.bytes, s.evicted
}

// MaxBytes reports the configured resident-footprint ceiling (0 = unbounded).
func (s *Store) MaxBytes() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxBytes
}

// SetMaxBytes changes the resident-footprint ceiling and immediately enforces it
// (0 = unbounded). Pinned digests are never evicted, so a tighter bound only ever
// drops unpinned transient entries.
func (s *Store) SetMaxBytes(n int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.maxBytes = n
	s.evictLocked()
}

// ----------------------------------------------------------------------------
// ABI registration: the blob store is the default RegionBackend + PageOutBackend.
// ----------------------------------------------------------------------------

// backend adapts *Store to abi.RegionBackend.
type backend struct{ s *Store }

// Resolver returns the underlying Store as the abi.Resolver for this RegionBackend.
func (b backend) Resolver() abi.Resolver { return b.s }
func (b backend) Caps() []abi.Capability { return nil }
func (b backend) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return b.s.PageOut(ctx, r)
}

// PageIn re-materializes a paged-out handle into an inline Ref via the underlying Store.
func (b backend) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return b.s.PageIn(ctx, h)
}

// Default is the process-wide store backing the registered RegionBackend, so the
// vDSO tier-2 cache and the context-MMU page-out share one CAS (content-addressed
// dedup across both).
var Default = New()

func init() {
	b := backend{Default}
	abi.RegisterRegionBackend(b)
	abi.RegisterPageOutBackend("blob", b)
}
