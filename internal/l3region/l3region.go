// Package l3region ships Stage 1 of child B of the L3 disaggregated-cache epic
// (#77 / epic #504; study docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md §4
// Option B): an L3RegionBackend behind fak's already-frozen Resolver seam
// (internal/abi.RegionBackend, registered via abi.RegisterRegionBackend).
//
// WHAT THE SEAM IS. An external L3 KV store (CAMA is the reference target) holds
// content-addressed PAGES — fixed-size blocks reached over RDMA mget/mset. fak's
// abi.Ref already carries a Digest (a content address) at the syscall boundary. So
// a Ref.Digest resolves to a SET of L3 page keys, and paging a region in/out IS an
// L3 mget/mset. The KV arena fak governs and the L3 pool the store holds become the
// same cells at two tiers, so fak's primitives (Evict, the DeletionCertificate mint)
// apply natively to the tier. This package is the resolver that maps span <-> page-key
// set behind the frozen Ref/Resolver interface — a backend swap, not an ABI change.
//
// WHAT STAGE 1 SHIPS (and ONLY Stage 1). Against a FAKE in-memory L3 (L3Store, a
// page-keyed Go map standing in for the RDMA pool — no CAMA, no network), it proves
// the two load-bearing properties the seam needs:
//   - a Ref resolves to a page-key SET: Put chunks a region into fixed-size pages,
//     msets each under its content-address page key, and records the ordered key set
//     in a manifest; PageKeys(ref) returns exactly that set (the control-path handle
//     Stage 2's Evict will invalidate and Stage 3's mget_rdma will fetch);
//   - a region round-trips BIT-EXACT through it: Resolve mgets the page-key set,
//     VERIFIES each fetched page hashes to its claimed key and the reassembled region
//     hashes to Ref.Digest (the "verify, don't trust" thesis — G1), then returns the
//     bytes with max|Δ|=0.
//
// CONTENT ADDRESSING is the property that makes the two tiers the same cells: a page
// key is sha256(page) and a region digest is sha256(region) — byte-identical to
// blob.Digest, so a Ref minted here resolves the same way the in-memory blob store
// would address it. Identical content dedups (same keys, same manifest entry).
//
// HONEST SCOPE / WHAT IS DEFERRED (per the epic's "control path only" constraint):
//   - Stage 2 (evict over the tier): L3Store.Mdel is the invalidation mechanism, but
//     wiring KVCache.Evict to invalidate a span's backing page keys is NOT done here.
//   - Stage 3 (real store): the CAMA connector, mget_rdma loopback, and the
//     data-path-bypass (bulk bytes flow client-direct while the resolver computes
//     page-key sets and invalidations out of band) are NOT done here. The in-memory
//     L3Store moves bytes through its map; that is a Stage-1 stand-in, not the
//     line-rate data path the production split keeps fak off of.
//   - This backend is NOT registered as the process-wide singleton Resolver: it is a
//     library a future stage wires (abi.RegisterRegionBackend is a last-wins setter
//     guarded by the architest single-registrant gate). Default builds are unchanged.
//
// Tier: foundation (1) — see internal/architest. Imports only abi + stdlib; an
// upward import fails the architest gate.
package l3region

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// CapL3Tier is the capability this backend advertises: a Ref resolves to a SET of
// L3 page keys in a disaggregated tier (mget/mset), NOT a local copy and NOT a
// zero-copy view. A caller negotiates it (Kernel.Negotiate) before assuming a
// resolved region is paged from L3.
const CapL3Tier abi.Capability = "l3.tier"

// PageBytes is the fixed page size a region is chunked into before it lands in L3.
// A real L3/CAMA pages KV in fixed blocks reached by one mget/mset key; 4 KiB is a
// representative block. A real deployment sizes this to the store's own block size.
const PageBytes = 4096

// digest is the content address (sha256 hex) of a byte span — byte-identical to
// blob.Digest, so a page key / region digest minted here is the same identity the
// in-memory blob tier would use (the "same cells at two tiers" property).
func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// ErrMiss is the typed L3 miss: a page key in a Ref's set is not resident (never
// stored, or invalidated by a future Mdel/Evict). Resolve returns it wrapped so a
// caller distinguishes "the tier does not hold this span" from a verification fault.
var ErrMiss = errors.New("l3region: page key not resident in L3")

// page is one content-addressed block: the key (sha256 of data) and the bytes. It is
// the unit of an L3 mget/mset.
type page struct {
	key  string
	data []byte
}

// L3Store is the FAKE in-memory L3 KV pool: a page-keyed byte map standing in for the
// external store reached over RDMA mget/mset in production (no CAMA, no network). It
// is content-addressed — the key IS sha256(page) — so Mset is idempotent and two
// identical pages share one slot. Concurrency-safe.
type L3Store struct {
	mu    sync.RWMutex
	pages map[string][]byte

	sets, gets, dels int64 // mget/mset/mdel counters (prove the round-trip hit the store)
}

// NewL3Store allocates an empty fake L3 pool.
func NewL3Store() *L3Store {
	return &L3Store{pages: map[string][]byte{}}
}

// Mset stores each page under its content-address key (idempotent: an identical page
// already resident is left as-is). The store keeps its OWN copy, as a real L3 pool
// holds the bytes independently of the caller's buffer.
func (s *L3Store) Mset(pages []page) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range pages {
		atomic.AddInt64(&s.sets, 1)
		if _, ok := s.pages[p.key]; ok {
			continue // content-addressed: identical key => identical bytes already resident
		}
		s.pages[p.key] = append([]byte(nil), p.data...)
	}
}

// Mget fetches the pages for an ordered key set. It returns the bytes in key order
// and ok=false (with the first missing key) if ANY key is absent — an L3 mget over a
// partially-evicted page set is a miss, not a half-region.
func (s *L3Store) Mget(keys []string) (out [][]byte, missing string, ok bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out = make([][]byte, 0, len(keys))
	for _, k := range keys {
		atomic.AddInt64(&s.gets, 1)
		b, ok := s.pages[k]
		if !ok {
			return nil, k, false
		}
		out = append(out, b)
	}
	return out, "", true
}

// Mdel invalidates page keys (the Stage-2 eviction mechanism: KVCache.Evict will call
// this to invalidate the L3 pages that backed an evicted span). It returns the count
// actually removed; a key already absent is not an error (idempotent invalidation).
func (s *L3Store) Mdel(keys []string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, k := range keys {
		atomic.AddInt64(&s.dels, 1)
		if _, ok := s.pages[k]; ok {
			delete(s.pages, k)
			n++
		}
	}
	return n
}

// Stats reports mset/mget/mdel op counts for KPI taps and tests.
func (s *L3Store) Stats() (sets, gets, dels int64) {
	return atomic.LoadInt64(&s.sets), atomic.LoadInt64(&s.gets), atomic.LoadInt64(&s.dels)
}

// L3RegionBackend implements abi.RegionBackend over an L3Store: its Resolver maps a
// Ref.Digest to a SET of L3 page keys (the manifest), Put = chunk+mset, Resolve =
// mget+verify+reassemble. It is the active abi.Resolver behind every RefRegion it
// issues (Resolver() returns the backend itself, like storedrv.Router).
type L3RegionBackend struct {
	store *L3Store

	mu       sync.RWMutex
	manifest map[string][]string // region digest -> ordered page-key set
	seq      uint64              // monotonic region id for the opaque Ref.Handle
}

// New builds an L3RegionBackend over the given fake L3 store.
func New(store *L3Store) *L3RegionBackend {
	return &L3RegionBackend{store: store, manifest: map[string][]string{}}
}

// chunk splits b into PageBytes pages, each content-addressed by its own bytes. A
// zero-length region is zero pages (its manifest entry is the empty set).
func chunk(b []byte) (pages []page, keys []string) {
	for off := 0; off < len(b); off += PageBytes {
		end := off + PageBytes
		if end > len(b) {
			end = len(b)
		}
		data := b[off:end]
		k := digest(data)
		pages = append(pages, page{key: k, data: data})
		keys = append(keys, k)
	}
	return pages, keys
}

// Put chunks b into content-addressed pages, msets them into L3, records the region's
// ordered page-key set in the manifest, and returns a RefRegion handle whose Digest is
// the whole-region content address. Resolution is by Digest (content identity), so two
// Puts of identical bytes dedup to one manifest entry and one set of L3 pages.
func (b *L3RegionBackend) Put(ctx context.Context, payload []byte) (abi.Ref, error) {
	pages, keys := chunk(payload)
	b.store.Mset(pages)

	d := digest(payload)
	b.mu.Lock()
	b.seq++
	handle := b.seq
	b.manifest[d] = keys
	b.mu.Unlock()

	return abi.Ref{
		Kind:   abi.RefRegion,
		Handle: handle,
		Digest: d,
		Len:    int64(len(payload)),
		Taint:  abi.TaintTainted, // fail-closed default, mirroring blob.Put / xenginekv.Put
		Scope:  abi.ScopeAgent,
	}, nil
}

// Resolve materializes the bytes a Ref points at. For a RefRegion it looks up the
// Ref.Digest's page-key set, mgets the pages from L3, VERIFIES the fetched bytes
// (each page must hash to its claimed key AND the reassembled region must hash to
// Ref.Digest — the "verify, don't trust" admission step, so a lying connector cannot
// pass off a substituted page), and returns the reassembled region bit-exact. An
// inline Ref returns its own bytes (a copy); a region this backend never minted, or
// one whose pages were invalidated, returns a typed miss.
func (b *L3RegionBackend) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefRegion:
		keys, err := b.PageKeys(r)
		if err != nil {
			return nil, err
		}
		pages, missing, ok := b.store.Mget(keys)
		if !ok {
			return nil, fmt.Errorf("%w: digest %s page %s", ErrMiss, r.Digest, missing)
		}
		out := make([]byte, 0, r.Len)
		for i, pb := range pages {
			if got := digest(pb); got != keys[i] {
				// G1: the page the store returned is NOT the page we asked for.
				return nil, fmt.Errorf("l3region: page %d verify failed: key %s but bytes hash %s", i, keys[i], got)
			}
			out = append(out, pb...)
		}
		if got := digest(out); got != r.Digest {
			return nil, fmt.Errorf("l3region: region verify failed: ref digest %s but reassembled %s", r.Digest, got)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("l3region: unsupported RefKind %d (this backend issues RefRegion)", r.Kind)
	}
}

// PageKeys returns the ordered L3 page-key SET a Ref resolves to — the explicit
// witness that "a Ref resolves to a page-key set", and the control-path handle the
// later stages act on (Stage 2 Mdel-invalidates it, Stage 3 mget_rdma-fetches it). It
// errors if the Ref is not a region this backend minted.
func (b *L3RegionBackend) PageKeys(r abi.Ref) ([]string, error) {
	if r.Kind != abi.RefRegion {
		return nil, fmt.Errorf("l3region: PageKeys needs a RefRegion handle, got RefKind %d", r.Kind)
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	keys, ok := b.manifest[r.Digest]
	if !ok {
		return nil, fmt.Errorf("%w: no manifest for digest %s (not minted by this backend)", ErrMiss, r.Digest)
	}
	return append([]string(nil), keys...), nil
}

// Resolver implements abi.RegionBackend: the backend IS the active Resolver.
func (b *L3RegionBackend) Resolver() abi.Resolver { return b }

// Caps implements abi.RegionBackend: it advertises the L3-tier capability.
func (b *L3RegionBackend) Caps() []abi.Capability { return []abi.Capability{CapL3Tier} }

// Compile-time proof the backend satisfies the frozen ABI seams.
var (
	_ abi.Resolver      = (*L3RegionBackend)(nil)
	_ abi.RegionBackend = (*L3RegionBackend)(nil)
)
