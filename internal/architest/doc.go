// Package architest is the kernel's machine-checked architecture contract.
//
// fak's safety and modularity story rests on a small number of structural
// invariants that, until now, lived only in prose (ARCHITECTURE.md, DIRECTION.md,
// PARTITION.md). Prose drifts; a green CI gate does not. This package is the same
// move the product makes — "the kernel doesn't believe the agents" — applied to
// the kernel's own source: CI doesn't believe the docs, it re-derives the
// invariants from the tree on every run.
//
// It enforces these load-bearing contracts (see architest_test.go):
//
//  1. The LAYERED-DAG import rule. Every internal package has a declared tier; a
//     package may import only packages whose tier is <= its own. Upward imports
//     (a foundation lib reaching up into an integrator) are the layer inversion
//     that turns the dependency DAG into spaghetti and silently voids the
//     "two fleet workers editing two leaves cannot collide" guarantee. Go already
//     forbids cycles at compile time; this adds the missing layering check.
//
//  2. The hot path imports no os/exec. The packages on a live tool-call decision
//     (the adjudicator chain, kernel, vDSO, grammar, pre-flight, context-MMU) must
//     not import os/exec — the per-decide subprocess boundary fak exists to remove
//     (DIRECTION.md). This is DIRECTION's "reviewer's grep #1" turned into a gate.
//
//  3. Every package declares its tier. A new leaf that forgets to take a position
//     in the layering fails the suite, so growth cannot silently erode the contract.
//
//  4. The whole request path is interpreter-free. Stronger than (2): no package
//     transitively reachable from internal/registrations (the live request-path
//     closure, not just the seven hot-path leaves) may EXEC a script interpreter
//     (python/node/sh/...). Execing a compiled binary (git, the fak binary) is fine;
//     re-introducing an untyped runtime on the decision path is the DIRECTION.md
//     thesis violation — "if the binary needs an interpreter to adjudicate, the
//     direction is broken." A non-literal program arg fails closed.
//
//  5. The Python oracle seam stays off the path. The off-path ML-ecosystem oracle
//     (export_oracle.py and friends) may be named only from _test.go / off-path
//     commands; a live string reference on the registrations closure would put an
//     untyped seam on the request path even without an exec. The DIRECTION.md seam
//     table turned into a gate.
//
// The package has no runtime behavior and is intentionally NOT registered into the
// kernel (internal/registrations) — it is an off-path test harness. It uses only
// the standard library (go/parser, go/token) so it never adds a module dependency,
// preserving the repo's zero-external-deps property that CI relies on.
package architest
