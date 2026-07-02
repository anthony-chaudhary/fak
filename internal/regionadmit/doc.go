// Package regionadmit is the shared region-admission decision every execution
// surface consults before mutating a file tree: may THIS actor act on THIS
// (lane, tree) right now, given the live lease set and the workspace lane
// taxonomy?
//
// Before it, each surface answered differently: the dispatch tick ran its own
// geometric overlap scan, `fak loop drive` ran none (two loops, or a loop and a
// dispatch worker, could edit the same tree with no mutual awareness), and a
// manual session had no decision to consult at all. This leaf is the ONE
// admission contract — the in-binary twin of the `dos arbitrate` rule, fed by
// whatever live-lease projection the caller holds (usually internal/leaseref's
// refs/fak/locks/* records):
//
//   - geometry: a requested tree overlapping a live lease's tree refuses
//     (dispatchorder.TreesOverlap — one algebra, reused, never re-derived);
//   - lane semantics: a named lane serializes even on disjoint trees, and an
//     exclusive lane (dos.toml [lanes].exclusive) runs alone;
//   - closed vocabulary: every refusal is COLLISION_RISK with the rung that
//     fired and the conflicting lease as evidence — never free prose.
//
// It is pure (state in, verdict out); LoadTaxonomy is the one I/O helper (a
// tolerant dos.toml read, no subprocess). It does NOT acquire, renew, or
// release anything — holding a lease stays with internal/leaseref, and the
// same honest boundary applies: cross-machine this is visibility, not atomic
// acquisition (see internal/leaseref).
//
// Consumers: the dispatch tick's lane-lease acquire, `fak loop drive`'s
// region hold, and `fak loop region` (the decision verb a manual session or a
// super-loop enter path calls before touching a region).
//
// Tier: mechanism (2) — see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package regionadmit
