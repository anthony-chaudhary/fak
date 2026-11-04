// Package architest is the kernel's machine-checked architecture gate: an off-path,
// stdlib-only CI harness that re-derives ARCHITECTURE.md/DIRECTION.md's structural rules
// from the tree on every run, because prose drifts and a green gate does not.
//
// Five load-bearing rules, each pinned by a named test in this package:
//
//  1. LAYERED-DAG IMPORTS: every internal package declares a tier and may import only
//     tiers <= its own; upward imports are the layer inversion that turns the DAG into
//     spaghetti and voids the "two fleet workers cannot collide" guarantee. Go forbids
//     cycles; this adds the missing layering check.
//  2. HOT PATH IMPORTS NO os/exec: the live tool-call decision packages (adjudicator
//     chain, kernel, vDSO, grammar, pre-flight, context-MMU) must not import os/exec —
//     the per-decide subprocess boundary fak exists to remove (DIRECTION.md grep #1).
//  3. EVERY PACKAGE DECLARES ITS TIER, and the declaration carries behavior: tier-1
//     primitives may import only abi. A new leaf that forgets to take a position fails
//     the suite, so growth cannot silently erode the layering.
//  4. INTERPRETER-FREE REQUEST PATH: stronger than (2) — nothing reachable from
//     internal/registrations may EXEC a script interpreter; execing a compiled binary
//     (git, fak) is fine. "If the binary needs an interpreter to adjudicate, the
//     direction is broken." A non-literal program arg fails closed.
//  5. PYTHON ORACLE SEAM OFF PATH: export_oracle.py and friends may be named only from
//     _test.go / off-path commands; a live string reference on the registrations closure
//     puts an untyped seam on the request path even without an exec.
//
// Every check fails closed: an undeclared tier, unmapped leaf, or upward import reds the
// gate. No runtime behavior; intentionally NOT registered into internal/registrations;
// stdlib-only, holding the module's external set at the two golang.org/x pins.
package architest
