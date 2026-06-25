// Package vcachechain is the vCache chains & recall engine — milestone M4 of the
// vCache epic (issue #719). The full design lives in
// docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md (§8 Recall, §11.0 the headline
// correction, §13-M4). Where the M5 Governor (internal/vcachegov) decides the
// STEADY STATE of one warm prefix, M4 decides how to RECALL a unit from a CHAIN of
// prefixes: whether to replay the chain (rebuild) or send the unit cold, and in
// what order.
//
// This package implements the four M4 acceptance criteria as a PURE, deterministic,
// off-path decision layer:
//
//   - Prefix DAG (§12 net-new #1) — ChainNode / PrefixDAG carry the parent chain per
//     vBlock (the recall plan), not just identity. PrefixDAG.ChainTo reconstructs the
//     root→node path; PrefixDAG.TopologicalReplay orders multi-target recall into
//     fan levels with the send-one-then-fan barrier (§8 + Rule C2) between every level.
//   - 20-block lookback (§8 + Rule C3) — PlaceBreakpoints drops an intermediate
//     breakpoint every ~15 content blocks, capped at 4 per request, so a long
//     agentic chain never silently misses past Anthropic's ≤20-block walk-back.
//   - Partial warmth (§8) — PlanRecall replays only from the first cold node
//     (WarmDepth), the depth the live loop derives from cachemeta.Diverge /
//     FirstDivergeTokenOffset.
//   - Cost-gated rebuild (§8 + §11.0) — the load-bearing gate. Replaying a warm
//     prefix of P tokens at read multiplier r costs P·r token-equivalents to save a
//     fresh prefill of the recalled unit's U tokens. Rebuild is net-negative whenever
//     P·r ≥ U (§11.0: a 30k-token prefix at r=0.1 to recall one 10-token unit is a
//     3000→10, i.e. 300× LOSS), so the gate REFUSES almost every single-unit chain
//     rebuild and allows rebuild ONLY for amortized fan-out — a large warm prefix
//     shared by S sibling units recalled together, where P·r < S·U.
//
// GATED OFF BY DEFAULT (the issue title). PlanRecall takes the enable flag as a
// parameter (DefaultEnabled = false); with it off, every recall is DecisionGatedOff
// and the caller sends the unit cold. The live provider recall loop is not in this
// package: like the M5 Governor, it is an off-path decision engine the future M1–M3
// calibration and warm set wire into a live loop, so it adds zero rungs to the
// request path.
//
// Correctness never depends on the outcome (Law A2): whatever PlanRecall returns,
// the caller must always be able to re-send the full prefix; a rebuild is only ever
// a cost/latency win, never a license to elide resent context.
//
// Tier: mechanism (2) — see internal/architest. This package imports only cachemeta
// (tier 1), vcachegov (tier 2, for the Law-D4 SecretClassification), and the
// standard library; an upward import fails the architest gate. It is deliberately
// NOT registered into the kernel (internal/registrations).
package vcachechain
