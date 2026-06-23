// Package memq is the agent-facing MEMORY-OPERATION ALGEBRA — the substrate that
// lets an agent (or a plugin, a driver, or an operator) author its OWN memory
// strategy instead of the kernel hard-coding one. The slogan is "build SQL, not a
// specific query": fak already ships several SPECIFIC memory operations welded into
// Go — recall.Recall (retrieve a working set), recall.Dream (consolidate/clean a
// core image), recall.RequestContextChange (tombstone one page), contextq.Query
// (materialize a render plan). Each is one fixed pipeline. memq is the small,
// composable language those five pipelines are five SENTENCES of, plus the planner
// and executor that run an authored sentence — so a sixth strategy nobody anticipated
// is a data value an agent emits, not a kernel edit.
//
// # The model
//
//   - A Cell is one addressable unit of memory (a recall.Page, an in-memory note, a
//     derived disposition) with TYPED attributes: role, kind, durability class,
//     sealed/tombstoned flags, size, digest, references, and an OPEN Attrs bag (the
//     same forward-compatible posture as abi.Verdict.Meta). Backends supply cells.
//   - A Pred is a SERIALIZABLE predicate expression (and/or/not/eq/ne/lt/le/gt/ge/
//     match) over a cell's fields — the WHERE clause, authorable as JSON, never a Go
//     closure, so an agent or an MCP client can write one.
//   - A Query is an ordered pipeline of Ops: scan | filter | rank | limit | budget
//     (the pure SELECT side) and render | tombstone | consolidate | reclassify |
//     prune (the EFFECT side). The pipeline threads a working set of cells.
//   - Plan/Explain renders the pipeline (and its per-step cell counts) WITHOUT
//     executing — the "step through this before you run it" surface.
//   - Run executes the pipeline against a Backend.
//
// # The safety posture (inherited from the rest of the kernel, enforced + tested)
//
//   - There is NO hard-delete operator. The strongest forgetting is `tombstone`
//     (negative-only: future recall skips the cell, the bytes and the row survive for
//     audit — recall.RequestContextChange) and `prune` (unreferenced storage GC only).
//     This mirrors fak's negative-only/expire-by-default stance (CONTEXT-IS-NOT-MEMORY.md).
//   - Effects default to PROPOSED, not applied. Run mutates a backend only when the
//     caller grants an explicit Caps for that effect; without caps every mutation is
//     reported as a proposal the operator can inspect first. Fail-closed: the default
//     is don't-touch.
//   - render and consolidate NEVER read a sealed cell's bytes — Materialize routes
//     every page-in through the backend's trust gate (recall's quarantine re-screen),
//     so a poisoned span can never be rendered into context or folded into a summary.
//   - reclassify can never PROMOTE a cell to `durable`; it may only hold or lower a
//     class (expire-by-default; promotion must be earned, which this rung does not
//     grant). Same fail-closed instinct as recall's promotion gate.
//   - The whole pipeline is deterministic: a fixed (query, backend cells) yields a
//     byte-identical plan and result (no RNG, clock, or map-iteration dependence) —
//     witnessed in proofs_witness_test.go.
//
// # Honest scope (rung 1)
//
// render, tombstone, and prune are REAL: render pages bytes in through the gate;
// tombstone applies through recall.RequestContextChange (with caps) and persists;
// prune reclaims unreferenced storage on a backend that supports it. consolidate
// produces a REAL derived artifact (a deterministic extractive summary the agent can
// render into context) but does NOT yet write that disposition back to durable store,
// and reclassify is proposal-only — the durable write-back of a derived/­reclassified
// cell to a recall core image is the named rung-2 follow-on. memq COMPLEMENTS the
// existing recall.Dream (which still owns the trust-gate reseal half of a sleep pass);
// it does not replace it.
package memq
