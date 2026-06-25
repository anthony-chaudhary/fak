// Package callavoid is the economics and effective-turn accounting for NOT making
// a local tool call — the principle that the cheapest, fastest, most reliable tool
// call is the one the kernel never has to dispatch.
//
// # Where it sits
//
// fak already AVOIDS calls in three live ways, all counted in internal/kernel's
// Counters:
//
//   - the vDSO (internal/vdso) serves a repeated read-only call from a local cache
//     — tier-1 pure elision, tier-2 content-addressed LRU keyed on
//     (tool, args-sha256, world-version), tier-3 static answers — so a duplicate
//     Read/Grep/Glob never reaches the engine (Counters.VDSOHits);
//   - the adjudicator DENIES a call at the capability floor before it runs, the
//     "fast reject" (Counters.Denies);
//   - the grammar layer REPAIRS a malformed call in-syscall instead of bouncing it
//     back for the model to retry (Counters.Transforms), the turn the TURN-TAX
//     benchmark (internal/turnbench) calls a FORCED save.
//
// The vDSO's contract is CORRECTNESS — "a cache hit equals a fresh call", enforced
// by binding every key to a world-version and bumping it on any write. What no leaf
// answers is the two questions this package exists for:
//
//  1. ECONOMICS — a cache that is correct is not automatically WORTH it. Validating
//     an entry, capturing a fingerprint, and eating the occasional stale miss all
//     cost something; in a write-heavy session the vDSO's GLOBAL world-version
//     invalidates every read entry on any write, so even a stable file can make
//     tier-2 a net loss. ProveMemo is the skeptical gate that says when avoidance
//     pays — the local-tool-call dual of vcachechain.ProveRecall's §11.0 cost gate.
//  2. AMPLIFICATION — Counters has raw tallies and turnbench has a TurnsSaved
//     difference, but neither expresses the thing that actually matters: how much
//     further the agent got per unit of real work. Account folds a window of
//     dispositions into effective-productive-turns and an amplification ratio, and
//     it credits the one avoidance the others miss — a PRODUCTIVE deny that prunes a
//     whole futile sub-tree the agent would otherwise have walked. One free deny
//     standing in for a hundred naive round-trips is the regime where an avoiding
//     kernel reaches a state a naive call-everything agent cannot reach in the same
//     budget, or would reach far slower.
//
// # The correctness law (inherited from vcache)
//
// Cost is always budgeted at the UNCACHED price; an avoided call is a realized
// rebate, never a trust claim. Every proof here sets CorrectnessDependsOn=false: the
// arithmetic decides whether to KEEP a cache, never whether a result is valid — that
// stays the vDSO's world-version invariant. A memo we cannot prove still valid must
// re-execute (fail toward the real call), so the worst case of being wrong about the
// economics is doing the naive thing, never returning a stale answer.
//
// # Layering
//
// This is a pure economics primitive: it imports nothing internal (math + the
// standard library only), so it can be folded by any layer above it. The live
// counters it models live in internal/kernel (tier 2, mechanism); a tier-4 caller
// (cmd/fak or a bench) maps kernel.Counters onto callavoid.Tally rather than this
// foundation leaf reaching upward for them. The Tally field names mirror the Counter
// names on purpose so that mapping is one obvious line.
//
// # Next milestone (not yet wired)
//
// Like the vcache proof leaves, this ships as a proven decision layer ahead of its
// live loop. The wiring is a tier-4 caller that reads a guard session's
// kernel.Counters (or gateway.AdjudicationSummary) into Account for the exit summary,
// and a per-tool-class ProveMemo gate at the vDSO tier-2 admission seam
// (kernel.Reap, between EvDispatch and eng.Complete) that declines to cache a class
// whose measured mutation rate refutes the economics. Both are off the hot path; this
// leaf computes, it does not adjudicate.
//
// Tier: foundation (1) — a pure metric/economics primitive, stdlib-only, importing
// nothing internal, alongside answershape, simhash, and modelroute.
package callavoid
