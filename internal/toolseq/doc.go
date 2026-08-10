// Package toolseq turns ordered per-session tool-call sequences into a
// tool-transition graph and its most common contiguous sequence variants —
// the "when do tools run, and in what order" view over a trajectory corpus.
//
// It is a pure, deterministic, stdlib-only leaf: sessions in, a sorted edge
// graph, ranked n-grams, and explainable workflow concepts out. Concepts bridge
// fleet aggregates to exemplar sessions for operator inspection and steering. Transitions never span a session boundary, so
// unrelated sessions cannot manufacture an adjacency. It imports nothing
// internal and sits off the hot path, so a report — or the `fak traj` front
// door (#2827) — can fold a corpus without pulling in trajectory machinery.
// Companion to internal/toolrollup's per-tool-TYPE aggregate; together they are
// session-analytics C2/C3 (#2824 / #2825).
//
// Tier: foundation (1) — see internal/architest. Imports only stdlib.
package toolseq
