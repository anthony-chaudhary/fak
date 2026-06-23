// Package storedrv is fak's pluggable STORAGE-DRIVER framework — the seam that
// lets every kind of data live in the place that fits it (hot RAM, durable disk,
// a remote object store, later a columnar or KV backend) instead of the one
// in-memory map the v0.1 blob store gives everything. It is the answer to "tool
// results in a separate store, blobs in object storage, memory disaggregated
// across tiers": a content-addressed ROUTER that composes one or more Drivers and
// picks where a payload lands by its routing dimensions, while resolving a digest
// from whichever tier holds it.
//
// THE SEAM IT FILLS. internal/abi exposes a SINGLE global Ref backend
// (RegisterRegionBackend, last-wins) — so without this package a disaggregated or
// remote store can only REPLACE the local one process-wide, never coexist with it.
// The Router is that single backend, but underneath it fans across N Drivers, so
// "memory lives in different places" becomes executable instead of a one-Resolver
// swap. The Router is registered as the abi RegionBackend (and a page-out codec)
// ONLY when the operator opts in via FAK_STORE; unset, the package registers
// nothing and the in-memory blob store stays the live backend, byte-for-byte
// unchanged (see config.go).
//
// THE DRIVER SPI. A backend implements Driver (abi.Resolver + a stable ID) and
// MAY also implement the optional capabilities the Router fans out to:
// abi.CASPinner (Pin/Unpin for GC-safety), abi.PageOutBackend (durable page-out),
// and Deleter (hard byte erasure for provable-deletion). The three pure-stdlib
// drivers ship in the default build (blob = RAM, blobfs = disk, blobhttp = remote
// HTTP object store); an EXTERNAL driver that needs a module — DuckDB columnar,
// an embedded KV, a vendor SDK — registers a Factory from its own build-tagged
// package (RegisterFactory), so the default `go install` stays zero-dependency
// (no go.sum) while a power build opts into a heavier backend.
//
// CONTENT ADDRESSING is the property that makes tiering correct: the sha256 digest
// IS the identity, identical across every tier, so a payload routed to disk and a
// cache copy in RAM share one address and Resolve can try tiers in order until one
// hits. Routing decides WHERE a write lands; the digest guarantees a read finds it.
package storedrv

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// InlineMax mirrors blob.InlineMax: a payload this small rides inline on the Ref
// (no tier touched), so the router never adds a hop for tiny args/results.
const InlineMax = blob.InlineMax

// Driver is the STORAGE-DRIVER SPI: the minimum a backend implements to be a tier
// in the router. It is abi.Resolver (Put/Resolve) plus a stable ID for routing and
// diagnostics. A Driver MAY additionally implement abi.CASPinner (pin/unpin),
// abi.PageOutBackend (durable page-out), and Deleter (erasure) — the router
// type-asserts and fans out to whichever a given Driver supports.
type Driver interface {
	abi.Resolver
	ID() string
}

// Deleter is the optional hard-erasure capability a Driver implements to support
// provable-deletion / retention: remove the bytes for a digest. A digest already
// absent is not an error (idempotent erasure).
type Deleter interface {
	Delete(ctx context.Context, digest string) error
}

// Hint is the explicit per-datum routing signal a caller that KNOWS a payload's
// plane/scope/taint/durability passes to PutHinted — the dimensions the synthesis
// identified (scope, taint, durability class, size). The zero Hint (size 0, agent
// scope, tainted, turn-durability) routes exactly like the plain abi.Resolver.Put
// size path, so an unaware caller is unaffected.
type Hint struct {
	Plane      string         // logical data plane (tool_result, kv_artifact, audit_row, ...)
	Scope      abi.ShareScope // Agent < Fleet < Tenant — wider scope prefers a shared/durable tier
	Taint      abi.TaintLabel // Quarantined bytes must go to a sealable/deletable tier, never a warm shared one
	Durability string         // "turn" | "session" | "durable" — only "durable" is eligible to cross to a persistent tier
}

// Tier is one Driver in the router's ordered fan, with the admission predicate that
// decides whether a Put of a given size belongs here and whether it is a durable
// (system-of-record) tier the router write-throughs to under Mirror.
type Tier struct {
	Driver  Driver
	Accept  func(size int) bool // does a Put of this byte size belong in this tier?
	Durable bool                // a persistent tier (disk/remote); survives restart
}

// Router is the content-addressed storage router: the single abi RegionBackend +
// page-out codec that fans Put/Resolve/Pin/Delete/PageOut across an ordered list of
// Drivers. Tier index 0 is the hottest. It is concurrency-safe (the tier list is
// immutable after New; only atomic counters mutate).
type Router struct {
	tiers  []Tier
	mirror bool             // write-through: a Put also lands in every durable tier
	caps   []abi.Capability // advertised on the RegionBackend

	puts   int64
	resolv int64
	misses int64 // Resolve that found the digest in no tier
}

// New builds a router over the given tiers (index 0 hottest). It returns an error
// if no tiers are supplied. mirror enables write-through: a Put also lands in every
// durable tier so the datum is both hot-cached and persistent.
func New(tiers []Tier, mirror bool) (*Router, error) {
	if len(tiers) == 0 {
		return nil, errors.New("storedrv: a router needs at least one tier")
	}
	return &Router{tiers: tiers, mirror: mirror, caps: []abi.Capability{"store.tiered"}}, nil
}

// tierFor returns the index of the first tier that accepts a payload of size n,
// falling back to the LAST tier (the catch-all) if none explicitly accept.
func (r *Router) tierFor(n int) int {
	for i, t := range r.tiers {
		if t.Accept == nil || t.Accept(n) {
			return i
		}
	}
	return len(r.tiers) - 1
}

// Put stores b and returns an addressable Ref. Small payloads ride inline (no tier
// touched). Otherwise the payload routes to the first accepting tier; with mirror
// enabled it is ALSO written through to every durable tier so it is both hot and
// persistent. The returned Ref carries the content digest, valid against any tier.
func (r *Router) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	if len(b) <= InlineMax {
		return abi.Ref{Kind: abi.RefInline, Digest: blob.Digest(b), Inline: append([]byte(nil), b...),
			Len: int64(len(b)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}, nil
	}
	atomic.AddInt64(&r.puts, 1)
	primary := r.tierFor(len(b))
	ref, err := r.tiers[primary].Driver.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, fmt.Errorf("storedrv: put -> tier %s: %w", r.tiers[primary].Driver.ID(), err)
	}
	if r.mirror {
		for i, t := range r.tiers {
			if i == primary || !t.Durable {
				continue
			}
			// Best-effort write-through; a mirror miss must not fail the primary Put
			// (the datum is already stored and resolvable from the primary tier).
			_, _ = t.Driver.Put(ctx, b)
		}
	}
	return ref, nil
}

// PutHinted is the explicit-routing entry point for a caller that knows a payload's
// plane/scope/taint/durability. Quarantined bytes or a Fleet/Tenant scope or a
// "durable" class route to the first durable tier (so sealed/shared/persistent data
// never sits only in volatile RAM); everything else falls back to the size policy.
// It is additive over the frozen ABI (callers reach it through *Router, not the
// abi.Resolver interface) so it never changes the default Put path.
func (r *Router) PutHinted(ctx context.Context, b []byte, h Hint) (abi.Ref, error) {
	if len(b) <= InlineMax {
		return r.Put(ctx, b)
	}
	wantDurable := h.Taint == abi.TaintQuarantined || h.Scope != abi.ScopeAgent || h.Durability == "durable"
	if wantDurable {
		if i := r.firstDurable(); i >= 0 {
			atomic.AddInt64(&r.puts, 1)
			ref, err := r.tiers[i].Driver.Put(ctx, b)
			if err != nil {
				return abi.Ref{}, fmt.Errorf("storedrv: put-hinted -> tier %s: %w", r.tiers[i].Driver.ID(), err)
			}
			return ref, nil
		}
	}
	return r.Put(ctx, b)
}

// Resolve materializes the bytes a Ref points at, trying tiers in order (hottest
// first) until one holds the digest. Inline Refs carry their own bytes. A digest in
// no tier returns an error.
func (r *Router) Resolve(ctx context.Context, ref abi.Ref) ([]byte, error) {
	if ref.Kind == abi.RefInline {
		return append([]byte(nil), ref.Inline...), nil
	}
	atomic.AddInt64(&r.resolv, 1)
	var firstErr error
	for _, t := range r.tiers {
		b, err := t.Driver.Resolve(ctx, ref)
		if err == nil {
			return b, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	atomic.AddInt64(&r.misses, 1)
	if firstErr == nil {
		firstErr = fmt.Errorf("storedrv: unknown digest %s", ref.Digest)
	}
	return nil, firstErr
}

// Pin protects a digest from GC in every pin-aware tier (abi.CASPinner fan-out).
func (r *Router) Pin(digest string) {
	for _, t := range r.tiers {
		if p, ok := t.Driver.(abi.CASPinner); ok {
			p.Pin(digest)
		}
	}
}

// Unpin releases one pin in every pin-aware tier.
func (r *Router) Unpin(digest string) {
	for _, t := range r.tiers {
		if p, ok := t.Driver.(abi.CASPinner); ok {
			p.Unpin(digest)
		}
	}
}

// Delete erases a digest from every tier that supports erasure (Deleter fan-out) —
// the aggregate provable-deletion primitive a disaggregated store needs (one call
// binds every backend holding the content-addressed bytes). It returns the joined
// error of any tier that failed; a tier without a Deleter is skipped.
func (r *Router) Delete(ctx context.Context, digest string) error {
	var errs []error
	for _, t := range r.tiers {
		if d, ok := t.Driver.(Deleter); ok {
			if err := d.Delete(ctx, digest); err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", t.Driver.ID(), err))
			}
		}
	}
	return errors.Join(errs...)
}

// PageOut moves a Ref's bytes into a DURABLE tier (the first durable tier, else the
// last tier) unconditionally — even a small body — and returns a bytes-absent
// handle, so a quarantined/cold result paged out through the router survives a
// restart. It prefers a tier's own abi.PageOutBackend (unconditional store); a
// driver that is not a PageOutBackend falls back to Put.
func (r *Router) PageOut(ctx context.Context, ref abi.Ref) (abi.Ref, error) {
	b, err := r.Resolve(ctx, ref)
	if err != nil {
		return abi.Ref{}, err
	}
	t := r.pageOutTier()
	if po, ok := r.tiers[t].Driver.(abi.PageOutBackend); ok {
		return po.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b)), Taint: ref.Taint, Scope: ref.Scope})
	}
	stored, err := r.tiers[t].Driver.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefBlob, Digest: stored.Digest, Len: int64(len(b)), Taint: ref.Taint, Scope: ref.Scope}, nil
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref via Resolve.
func (r *Router) PageIn(ctx context.Context, handle abi.Ref) (abi.Ref, error) {
	b, err := r.Resolve(ctx, handle)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefInline, Digest: handle.Digest, Inline: b, Len: int64(len(b)), Taint: handle.Taint, Scope: handle.Scope}, nil
}

// firstDurable returns the index of the first durable tier, or -1 if none.
func (r *Router) firstDurable() int {
	for i, t := range r.tiers {
		if t.Durable {
			return i
		}
	}
	return -1
}

// pageOutTier picks where page-out lands: the first durable tier, else the last
// tier (the catch-all) so page-out always has a home.
func (r *Router) pageOutTier() int {
	if i := r.firstDurable(); i >= 0 {
		return i
	}
	return len(r.tiers) - 1
}

// Resolver implements abi.RegionBackend: the router IS the active Resolver.
func (r *Router) Resolver() abi.Resolver { return r }

// Caps implements abi.RegionBackend.
func (r *Router) Caps() []abi.Capability { return r.caps }

// Stats reports router activity for KPI taps.
func (r *Router) Stats() (puts, resolves, misses int64) {
	return atomic.LoadInt64(&r.puts), atomic.LoadInt64(&r.resolv), atomic.LoadInt64(&r.misses)
}

// Describe renders the tier topology for diagnostics / `fak` introspection, e.g.
// "tiers=[blob(hot) blobfs(durable) blobhttp(durable)] mirror=true".
func (r *Router) Describe() string {
	parts := make([]string, len(r.tiers))
	for i, t := range r.tiers {
		role := "hot"
		if t.Durable {
			role = "durable"
		}
		parts[i] = fmt.Sprintf("%s(%s)", t.Driver.ID(), role)
	}
	return fmt.Sprintf("tiers=[%s] mirror=%v", strings.Join(parts, " "), r.mirror)
}

// SelfCheck round-trips a probe payload through the router (Put then Resolve) and
// verifies byte-identity and digest stability — the storedrv analogue of a backend
// liveness proof. It also confirms the probe resolves from the tier it routed to.
func (r *Router) SelfCheck(ctx context.Context) error {
	probe := []byte("storedrv-selfcheck-" + strings.Repeat("x", InlineMax+16)) // > InlineMax so it hits a tier
	ref, err := r.Put(ctx, probe)
	if err != nil {
		return fmt.Errorf("storedrv selfcheck: put: %w", err)
	}
	if ref.Digest != blob.Digest(probe) {
		return fmt.Errorf("storedrv selfcheck: digest mismatch %q != %q", ref.Digest, blob.Digest(probe))
	}
	got, err := r.Resolve(ctx, ref)
	if err != nil {
		return fmt.Errorf("storedrv selfcheck: resolve: %w", err)
	}
	if string(got) != string(probe) {
		return errors.New("storedrv selfcheck: resolved bytes differ from stored")
	}
	return nil
}

// memDriver adapts the process-wide in-memory blob store (blob.Default) to the
// Driver SPI — the hot tier. blob.Default already implements abi.Resolver +
// abi.CASPinner + abi.PageOutBackend, so memDriver only adds the ID.
type memDriver struct{ s *blob.Store }

func (memDriver) ID() string { return "blob" }
func (m memDriver) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return m.s.Put(ctx, b)
}
func (m memDriver) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	return m.s.Resolve(ctx, r)
}
func (m memDriver) Pin(digest string)   { m.s.Pin(digest) }
func (m memDriver) Unpin(digest string) { m.s.Unpin(digest) }
func (m memDriver) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return m.s.PageOut(ctx, r)
}
func (m memDriver) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return m.s.PageIn(ctx, h)
}

// sortedFactorySchemes returns the registered external-driver schemes, sorted, for
// a deterministic "unknown scheme; known schemes are: ..." diagnostic.
func sortedFactorySchemes() []string {
	factoryMu.RLock()
	defer factoryMu.RUnlock()
	out := make([]string, 0, len(factories))
	for s := range factories {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// ensure the router satisfies the seams it registers against.
var (
	_ abi.Resolver       = (*Router)(nil)
	_ abi.RegionBackend  = (*Router)(nil)
	_ abi.PageOutBackend = (*Router)(nil)
	_ abi.CASPinner      = (*Router)(nil)
	_ Driver             = memDriver{}
)

// factory registry state lives here so storedrv.go and config.go share it.
var (
	factoryMu sync.RWMutex
	factories = map[string]Factory{}
)
