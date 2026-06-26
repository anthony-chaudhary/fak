// Package memview is the typed virtual-view contract over canonical raw memory
// cells (issue #904): a memory/context cell is CANONICAL (the raw bytes), and
// every summary / QA / graph / prompt-prefix / KV-prefix projection of it is a
// DERIVED view that carries provenance and an admission gate. No view is itself
// canonical, and no view executes a tool effect — a materialized view must
// re-enter adjudication before any effect.
//
// This is the "do not make memory one lossy summary blob" stance (#904): exact
// prompt/KV-cache reuse (LOSSLESS, byte-identical — internal/promptmmu and the
// vcachechain replay decision) is a DIFFERENT thing from a lossy semantic view
// (this package), and the two must never be conflated. Exact reuse answers "is
// this the same bytes"; a view answers "here is a derived projection, here is the
// source it came from, here is why it is (or is not) admissible."
//
// The contract binds a view to its source by a content digest + byte span. When
// the raw source bytes change, the digest changes, and every view bound to the
// old digest is INVALID — the same cache-line-invalidation semantics an OS MMU
// applies to a page table entry after a write. A view's admissibility is also
// gated on the SOURCE taint: a tainted/quarantined source taints the view, so a
// poisoned page can never back an admissible projection.
//
// The minimum record (MemoryViewRecord) is the design-note-made-type: raw input
// digest, view kind, producer, source span, inherited taint, freshness, optional
// witness, and an invalidation rule. It is deliberately small and non-invasive —
// it adds a NEW typed seam rather than refactoring recall/ctxmmu at once, so the
// graph-selector / KV-prefix / skill-manifest views named in #904 can land as
// later children behind the SAME contract. It imports only the frozen abi for the
// taint lattice + verdict; it defines a RawPage interface so a recall.Page (tier
// 3) adapts to it WITHOUT memview importing recall.
//
// Tier: mechanism (2) — see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate. It
// imports only the frozen abi (tier 0) + stdlib, registers nothing with the ABI,
// and is off the request path. See AGENTS.md and internal/architest for the
// layering contract; see docs/notes/MEMORY-VIEW-CONTRACT-2026-06-26.md for the
// SOTA readout + the mapping of current prompt-MMU/recall/cache work onto this
// table.
package memview
