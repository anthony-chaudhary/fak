// Package opttarget is the declarative target layer of the RSI optimization
// fuser (epic #1279). An OptTarget is DATA — a site, a bounded candidate
// grammar, a measurer binding, a metric direction, and guards — that Compile
// lowers into a rsiloop.Harness, so an optimization target is DECLARED, not
// hand-coded. The non-forgeable keep-bit (shipgate.Evaluate, driven by
// rsiloop.Run) is reused VERBATIM: the fuser changes only how many targets ride
// the oracle and how cheaply each is added, never the oracle itself.
//
// The carry-over is the DSL auto-fuser pattern (TVM/Ansor, Triton, Mojo,
// Mirage): you do not author each fused kernel by hand; you declare the op in a
// compact language and a compiler lowers it. Here the "kernel" is a Harness, the
// "oracle" is the keep-bit, and Compile is the lowering — the same constrained
// schedule space, the same fixed correctness oracle, scaled past the dozens of
// hand-wired harnesses the program caps out at today.
//
// Tier: integrator (4) — see internal/architest. This package imports rsiloop(4)
// (Compile returns a rsiloop.Harness), so it is forced to the integrator layer;
// it may import only packages whose tier is <= 4. See AGENTS.md and
// internal/architest for the layering contract.
package opttarget
