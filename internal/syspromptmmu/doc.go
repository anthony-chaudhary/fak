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
// and anchors on the same cached-prefix boundary via promptmmu.ArraySplicePoints. The
// queried harness overlay (Rung 3, #1261) fills the overlay; the spine/policy
// non-evictable flags are the substrate Rung 4 (#1262) enforces.
//
// Tier: mechanism (2) — see internal/architest. This package may import only packages
// whose tier is <= 2; it imports cachemeta(1) + promptmmu(1) + stdlib. An upward import
// fails the architest gate. See AGENTS.md and internal/architest for the layering contract.
package syspromptmmu
