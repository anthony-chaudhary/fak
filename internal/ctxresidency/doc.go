// Package ctxresidency is the context-residency query (issue #521): a first-class,
// witnessable READ over the span ledger that composes the three layers that
// already maintain the context's coherence state — kvmmu (the KV-level span
// ledger Admit/Evict maintains), ctxmmu (the byte-level quarantine/clearance
// ledger), and cachemeta (the residency tiers + the dependent-entry graph the
// eviction blast radius is read from). It is the read side of that state; the
// write side (kvmmu.Append / AdmitResult / Quarantine / evict, ctxmmu.Admit /
// Clear / PageIn) is unchanged.
//
// Tier: composer (3) — see internal/architest. It imports kvmmu (3), ctxmmu (2),
// cachemeta (1), and abi (0); every edge satisfies the layered-DAG rule
// (importer tier >= imported tier). This package may import only packages whose
// tier is <= 3; an upward import fails the architest gate. See AGENTS.md and
// internal/architest for the layering contract.
//
// # Why a leaf, not an addition to kvmmu
//
// kvmmu is the span ledger's OWNER and sits on the write path (Append / evict);
// a query is a pure read that composes THREE layers. Putting it in kvmmu would
// make kvmmu import ctxmmu's counters and cachemeta's tier graph directly on the
// write path, and would collide with the kvmmu leaf's own in-flight work. A
// separate composer leaf keeps the read isolated, off the write path, and free
// to compose. It registers nothing with the frozen ABI (it is a library type a
// caller constructs, exactly like internal/polymodel) — so it reaches a live
// request path only when a caller wires Query into an operator/observability
// surface; the pure helpers here are deterministic and safe to call directly.
//
// # What it returns
//
// Query(c, mmu) returns a Snapshot: one Span per kvmmu segment, each classified
// resident | evictable | held with its cachemeta residency tier and an
// EvictBlastRadius (the K/V positions + the live cachemeta dependents an Evict
// would drop — the read-only projection of kvmmu.evict's invalidation walk),
// plus the byte-level totals from ctxmmu. The witness tests assert the snapshot
// RECONCILES with both kernels' own counters (ResidentTokens == kvmmu.CacheLen,
// HeldSpans == kvmmu.Evicted, ByteHeld == ctxmmu.HeldLen, ByteCleared ==
// len(ctxmmu.Cleared)) — so the query can never miscount vs the kernel.
//
// # C6: witness + audit surface (issue #1109)
//
// LoaderJournal reads the durable audit journal and reconciles all capability
// lifecycle events (CAP_FAULT, CAP_EVICT, CAP_VERSION_BIND) against the kernel's
// authoritative counters. It is the read side of the trust floor for the
// capability loader: every fault, eviction, and version-bind is a journal row,
// and LoaderJournal proves the loader's derived view matches the kernel's ledger.
// A LoaderSnapshot with Reconciled=true is verified; a mismatch surfaces a
// discrepancy the auditor must investigate.
//
// # The honest boundary
//
// The issue's target per-span shape includes {reason, bytes}; those are
// layer-specific and live at the ctxmmu BYTE tier, keyed by the ctxmmu
// quarantine id (q<n>), which the kvmmu span ledger (keyed by tool-call id) does
// not carry. Exposing them per-span would require enriching the kvmmu Segment;
// the byte-level totals are reported on the Snapshot instead, and reconcile with
// ctxmmu. The query is a pure read: it mutates nothing, so it cannot launder a
// poisoned span back into context — re-admission still re-screens through
// ctxmmu.PageIn's witness-clear requirement, which this query never satisfies on
// a caller's behalf. See docs/notes/RESEARCH-ultra-long-context-levels-and-
// naming-2026-06-22.md §3 (the "context-residency query" naming verdict).
package ctxresidency
