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
// This rung produces a plan only: no wire mutation, no provider call. It does not
// splice the plan into wire bytes (Rung 2, #1260) or query the harness overlay
// (Rung 3, #1261); the spine/policy non-evictable flags it carries are the substrate
// Rung 4 (#1262) enforces.
//
// Tier: mechanism (2) — see internal/architest. This package may import only packages
// whose tier is <= 2; it imports only cachemeta(1) + stdlib. An upward import fails the
// architest gate. See AGENTS.md and internal/architest for the layering contract.
package syspromptmmu
