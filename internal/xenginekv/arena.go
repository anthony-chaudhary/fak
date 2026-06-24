// Package xenginekv ships the cross-engine zero-copy KV co-residence seam (#448):
// a RegionBackend whose Resolver hands out RefRegion handles into ONE addressable
// arena where an EXTERNAL engine's KV cache and fak's tool args/results CO-RESIDE.
//
// WHY IT EXISTS. fak's hard-to-copy differentiator — provable per-agent KV
// Evict/Clone (internal/model.KVCache.Evict quarantines a poisoned span so the
// model physically cannot attend to it; Clone splices a reusable prefix) — is real
// only where fak OWNS the KV (its own in-kernel model). Against an engine fak does
// NOT run (vLLM/SGLang owns the KV behind a process boundary), those primitives are
// a stub: there is no addressable seam to the engine's KV, so "evict a poisoned span
// so the model cannot attend to it" cannot be enforced. fak then keeps only the
// easily-replicated half (an adjudicator in front of the engine) and loses the
// hard-to-replicate half exactly where it would matter. This package is the missing
// seam — the in-address-space RegionBackend internal/abi's Ref doc already promised
// ("a later in-address-space impl — co-residing tool args/results with the KV cache —
// is a Resolver/RegionBackend swap behind Capability \"zerocopy\"").
//
// WHAT THE SEAM GUARANTEES, expressed on the frozen ABI:
//   - Resolve returns a VIEW into the arena (a sub-slice that aliases the backing
//     bytes) — no copy, no allocation. That is the "zero-copy" in the capability.
//   - Evict UNMAPS a span and zeroes its bytes: after it, the handle no longer
//     resolves and the bytes are physically gone from the shared region — the
//     cross-engine KV quarantine, holding whether or not fak runs the model.
//   - Clone duplicates a resident span to a fresh handle — the cross-engine prefix
//     reuse, the region-addressed dual of KVCache.Clone.
//   - PageOut hands the HANDLE across without moving bytes (the region stays
//     resident); PageIn returns the still-resolvable handle. Zero movement is the
//     property an external engine's pinned KV pages need.
//
// SCOPE / HONESTY. The arena here is a Go []byte: the in-process stand-in for what
// is, in production, a shared-memory / CUDA-IPC-imported handle onto the external
// engine's KV pages (AttachArena takes exactly such a buffer). The SEAM — the ABI
// boundary, the zero-copy Resolve, the Evict/Clone region primitives, the advertised
// "zerocopy" capability, the opt-in singleton-backend swap — is shipped and tested
// here. What remains is the engine-specific TRANSPORT that maps a real vLLM/SGLang KV
// region into an Arena; that plugs in behind this exact ABI with no further ABI change.
//
// Default-OFF: the package is inert unless FAK_XENGINE_KV opts in (see register.go),
// so the content-addressed blob store stays the live RegionBackend in every default
// build. This mirrors the storedrv router's opt-in last-wins swap.
package xenginekv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// CapZeroCopy is the capability this backend advertises: a Ref resolves to a VIEW
// into a co-resident arena, never a copy. A caller negotiates it (Kernel.Negotiate)
// before assuming a resolved []byte aliases live region bytes.
const CapZeroCopy abi.Capability = "zerocopy"

// span is one live allocation in the arena: the byte range [off, off+n) that a
// RefRegion handle (Handle == off) addresses. Evict deletes the entry; a deleted
// handle no longer resolves.
type span struct {
	off, n int64
}

// Arena is a co-resident byte region whose bytes fak does NOT own a private copy of:
// in production it is a shared-memory / CUDA-IPC-imported buffer onto an external
// engine's KV; here a Go []byte stands in. Put places fak's bytes INTO the arena (so
// tool args/results co-reside with the engine's KV) and returns a RefRegion handle;
// Resolve returns a zero-copy VIEW; Evict unmaps a span; Clone duplicates one.
//
// It is the active abi.Resolver behind every RefRegion this backend issues. A bump
// allocator hands out monotonically increasing offsets; an evicted span's bytes are
// NOT reclaimed for reuse (a freelist would let a later Put alias an offset a stale
// handle still names — the kind of confusion the quarantine exists to prevent).
type Arena struct {
	mu   sync.RWMutex
	buf  []byte         // the co-resident region (shared-memory stand-in)
	used int64          // bump pointer: next free offset
	live map[int64]span // off -> live span; an evicted span is removed
}

// NewArena allocates a fresh co-resident region of the given size in bytes. Use this
// when fak provisions the shared region itself (the in-process / test path).
func NewArena(size int) *Arena {
	if size < 0 {
		size = 0
	}
	return &Arena{buf: make([]byte, size), live: map[int64]span{}}
}

// AttachArena attaches to an EXTERNALLY provided region — the real co-residence path:
// buf is a shared-memory / IPC-imported buffer onto the external engine's KV. The
// arena does not copy buf; it writes fak's co-resident bytes into the tail of it.
func AttachArena(buf []byte) *Arena {
	return &Arena{buf: buf, live: map[int64]span{}}
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Put places b into the arena and returns a RefRegion handle addressing it. The
// handle's Handle is the byte offset; Resolve later returns a view of exactly these
// bytes. Returns an error if the arena cannot fit b (a real shared region is bounded).
// An inline-sized payload still lands in the arena: co-residence is the whole point,
// so even small tool args share the region with the KV rather than escaping to a copy.
func (a *Arena) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	n := int64(len(b))
	off := a.used
	if off+n > int64(len(a.buf)) {
		return abi.Ref{}, fmt.Errorf("xenginekv: arena full — need %d bytes at offset %d, region is %d", n, off, len(a.buf))
	}
	copy(a.buf[off:off+n], b)
	a.used = off + n
	a.live[off] = span{off: off, n: n}
	return abi.Ref{
		Kind:   abi.RefRegion,
		Handle: uint64(off),
		Digest: digest(b),
		Len:    n,
		Taint:  abi.TaintTainted, // fail-closed default, mirroring blob.Put
		Scope:  abi.ScopeAgent,
	}, nil
}

// Resolve materializes the bytes a Ref points at. For a RefRegion it returns a VIEW
// that ALIASES the arena's backing bytes — zero copy, zero allocation — which is the
// "zerocopy" capability's whole content. An inline Ref returns its own bytes (a copy,
// as those bytes never entered the arena). A handle whose span was Evicted no longer
// resolves: that is the cross-engine quarantine, enforced at the resolve seam.
//
// CONTRACT: the returned slice is only valid until the span is Evicted (which zeroes
// it) or the arena is concurrently mutated. A holder that keeps bytes across such a
// boundary must copy them. This matches a real pinned KV region: the view is live
// memory, not an owned snapshot.
func (a *Arena) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefRegion:
		a.mu.RLock()
		defer a.mu.RUnlock()
		s, ok := a.live[int64(r.Handle)]
		if !ok {
			return nil, fmt.Errorf("xenginekv: region handle %d is not resident (evicted or never allocated)", r.Handle)
		}
		return a.buf[s.off : s.off+s.n : s.off+s.n], nil // zero-copy view (cap-bounded so an append never clobbers a neighbour)
	default:
		return nil, fmt.Errorf("xenginekv: unsupported RefKind %d (this backend issues RefRegion)", r.Kind)
	}
}

// Evict is the cross-engine KV quarantine primitive: it unmaps a span and zeroes its
// bytes, so a poisoned tool result (or a KV span fak adjudicated as poisoned) is
// physically gone from the co-resident region — neither fak nor the external engine
// can attend to or resolve it again. It is the region-addressed dual of
// internal/model.KVCache.Evict, the half that previously held only where fak owned
// the KV. Returns an error if the handle is not resident (already evicted / unknown).
func (a *Arena) Evict(r abi.Ref) error {
	if r.Kind != abi.RefRegion {
		return fmt.Errorf("xenginekv: Evict needs a RefRegion handle, got RefKind %d", r.Kind)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off := int64(r.Handle)
	s, ok := a.live[off]
	if !ok {
		return fmt.Errorf("xenginekv: region handle %d is not resident (already evicted or unknown)", r.Handle)
	}
	for i := s.off; i < s.off+s.n; i++ {
		a.buf[i] = 0 // physically clear: a dangling view reads zeros, not the poisoned span
	}
	delete(a.live, off)
	return nil
}

// Clone duplicates a resident span to a FRESH handle (a copy within the arena), the
// region-addressed dual of KVCache.Clone: a computed prefix's bytes are reused by a
// later session without re-deriving them. Evicting one handle never affects the other,
// because the bytes are distinct allocations. Returns an error if src is not resident.
func (a *Arena) Clone(src abi.Ref) (abi.Ref, error) {
	if src.Kind != abi.RefRegion {
		return abi.Ref{}, fmt.Errorf("xenginekv: Clone needs a RefRegion handle, got RefKind %d", src.Kind)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	s, ok := a.live[int64(src.Handle)]
	if !ok {
		return abi.Ref{}, fmt.Errorf("xenginekv: region handle %d is not resident", src.Handle)
	}
	off := a.used
	if off+s.n > int64(len(a.buf)) {
		return abi.Ref{}, fmt.Errorf("xenginekv: arena full — cannot clone %d bytes at offset %d, region is %d", s.n, off, len(a.buf))
	}
	copy(a.buf[off:off+s.n], a.buf[s.off:s.off+s.n])
	a.used = off + s.n
	a.live[off] = span{off: off, n: s.n}
	out := src
	out.Handle = uint64(off)
	return out, nil
}

// PageOut hands a Ref's HANDLE across the context-MMU's page-out seam WITHOUT moving
// its bytes: a co-resident region stays resident, so paging a cold/quarantined result
// out is just relabelling the handle (inline bytes are admitted into the arena first,
// so the result co-resides on its way out). Zero movement is the property an external
// engine's pinned KV pages need — the bytes never leave the shared region.
func (a *Arena) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	if r.Kind == abi.RefInline {
		return a.Put(ctx, r.Inline) // admit into the arena so it co-resides
	}
	return r, nil // already region-resident: the handle IS the paged-out pointer
}

// PageIn returns the still-resolvable region handle (the bytes never left, so there is
// nothing to fetch back). It is the page-in dual of PageOut's zero-movement promise.
func (a *Arena) PageIn(ctx context.Context, handle abi.Ref) (abi.Ref, error) {
	return handle, nil
}

// backend adapts an Arena to the abi RegionBackend + PageOutBackend seams.
type backend struct{ a *Arena }

// Resolver returns the arena as the abi.Resolver behind every RefRegion this backend issues.
func (b backend) Resolver() abi.Resolver { return b.a }
func (b backend) Caps() []abi.Capability { return []abi.Capability{CapZeroCopy} }
func (b backend) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return b.a.PageOut(ctx, r)
}

// PageIn delegates to the arena's PageIn, returning the still-resolvable region handle
// (the bytes never left the shared region).
func (b backend) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return b.a.PageIn(ctx, h)
}

// Compile-time proof the arena/backend satisfy the frozen ABI seams.
var (
	_ abi.Resolver       = (*Arena)(nil)
	_ abi.RegionBackend  = backend{}
	_ abi.PageOutBackend = backend{}
)
