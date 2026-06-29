// Package syspromptmmu emits fak's ordered base-context plan — the fak-first head of
// the context window (Rung 1 of the system-prompt MMU, epic #1258, issue #1259).
//
// Today no fak-authored system prompt exists; the serve path only preserves a
// passthrough prefix byte-identically. This package is the one genuinely new
// authorship surface: it authors fak's irreducible spine (the gate, the journal, what
// a capability is) as an immutable, version-stamped SegStable tier, followed by a
// versioned policy floor, and emits them as an ordered []cachemeta.PromptSegment — a
// PLAN, never wire bytes. Each segment carries a content-derived Witness (a blob hash)
// so a later turn can prove the spine is unchanged (invariant 1).
//
// Rung 1 (syspromptmmu.go) produces a plan only: no wire mutation, no provider call.
// Rung 2 (splice.go, #1260) is the cache-safe SPLICER that realizes the plan into the
// Anthropic `system` block (BuildSystemValue) and, on every later turn, swaps ONLY the
// after-breakpoint overlay while copying the resident spine+policy prefix verbatim
// (SpliceSystemOverlay) — proving bytes.Equal(prefix) e2e (invariants 1+2), fail-safe
// identity on a mutated spine. It is the system-block twin of promptmmu.CompactInboundTools
// and anchors on the same cached-prefix boundary via promptmmu.ArraySplicePoints. Rung 3
// (overlay.go, #1261) fills the overlay by QUERY — the first live caller of the
// skill-loader keystone (capindex.Catalog): SelectOverlay ranks the at-rest cards for a
// turn intent, faults winners up to a token budget, and emits the overlay segments the
// splice consumes; OverlayCache serves a HIT on an unchanged rank digest with no
// re-fault. The spine/policy non-evictable flags are the substrate Rung 4 (#1262) enforces.
// Rung 5 (edit.go, #1263) is the witness-gated base-edit core: GateEdit/ApplyEdit admit a
// self-proposed delta only when an INJECTED witness passes (the agent never grades its own
// edit), hard-refuse any edit to the spine or policy floor (the agent never rewrites its
// own meta-rules), and apply append-mostly over a copy so the prior plan is a bit-for-bit
// rollback. Rung 6 (audit.go, #1264) is the observability witness: AuditRealizedPrefix re-derives a
// wire body's resident prefix and proves it equals the planned spine — divergence is a
// loud alarm (an accidental head mutation caught before a cache miss), a harness-authored
// body is a neutral AuditAbsent. It consumes the context-safety doctrine (#1217); it does
// not mint parallel numbers.
//
// Tier: mechanism (2) — see internal/architest. This package may import only packages
// whose tier is <= 2; it imports cachemeta(1) + promptmmu(1) + capindex(2) + stdlib
// (capindex is a permitted same-tier edge). An upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package syspromptmmu
