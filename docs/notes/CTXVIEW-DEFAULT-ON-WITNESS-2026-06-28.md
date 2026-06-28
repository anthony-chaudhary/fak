---
title: "ctxview default-on witness: the O(1) planner is safe to wire live (2026-06-28)"
description: "The GPU-free, re-runnable evidence that the ctxplan O(1) context planner (the --ctx-view-budget gateway wiring) is correct and fail-open enough to flip from off-by-default to a conservative default-on: a synthetic known-answer selfcheck, the end-to-end on-the-wire gateway test (including the Anthropic passthrough prefix-byte-identity), and the fleet's realized-reuse dogfood."
slug: ctxview-default-on-witness
date: 2026-06-28
---

# ctxview default-on witness (2026-06-28)

The O(1) context planner (`internal/ctxplan` + the gateway `--ctx-view-budget` wiring in
`maybePlanMessages`/`maybeElideKVResidency`) ships **off by default** because, the design
note says, it rewrites in-flight turn history and should be "gate[d] until you have watched
a real session." This note is that watch: the GPU-free, re-runnable evidence that the
planner is correct and **fail-open** enough to flip to a conservative default-on.

## What was run on this host (win32, no GPU), all green

1. **Synthetic known-answer selfcheck** — `go run ./cmd/ctxplanbench -selfcheck` (exit 0):

   ```
   selfcheck replay: 5 turns, peak-planned 16/20, faithful 5/5, 5 ref → 3 faults (3 served), oldest-recall 0.020
   planning cost:    full-scan 15 cand, bounded-probe 15 cand (peak 5/128), plan-agree 5/5 turns
   ```

   The O(1) invariants hold: resident peak ≤ budget (16/20), **exact recall** every turn
   (faithful 5/5), every forecast-miss **served** (3 ref → 3 faults, 3 served, 0 refused,
   0 lost), and the bounded probe is **identical** to the full scan (plan-agree 5/5).

2. **End-to-end on-the-wire gateway test** — `go test ./internal/gateway -run TestCtxViewHTTP`
   (`ok ... 34.750s`). Reads the bytes that actually reach a mock upstream and asserts:
   OFF forwards the full history; ON forwards strictly fewer, bounded ≤ budget, never empty;
   the **Anthropic passthrough OFF byte-equals the inbound**; and ON stubs the off-topic
   middle (`[fak] ctxview-elided`) while keeping the **cached system prefix byte-identical**,
   preserving message count, with **exact recall** of the elided span (0 lost facts).

## What the fleet already dogfooded

The realized-reuse half landed and was immediately dogfooded: the #1066-honest cache-value
gate (`fak nightrun score`, commit `d7548e81`) reports the WITNESSED realized KV-prefix
reuse ratio over multi-turn sessions, and a follow-up (`94fbbbda`, #1114) tuned its
regression floor to the **measured 75.1% realized reuse** from a real run. So the planner's
*value* (turn-over-turn KV-prefix reuse) is observed on real traffic, not just modelled.

## Why this clears the flip gate

- **Fail-open by construction.** `maybePlanMessages` returns the full history on any planner
  error, empty render, or untraced one-shot — it "only ever shortens, and on doubt shortens
  nothing."
- **The flagship wire stays cache-safe.** Under budget>0 the `fak guard -- claude` passthrough
  applies the in-place same-role stub transform (`agent.CompactAnthropicHistoryToView`),
  proven prefix-byte-identical by `TestCtxViewHTTPAnthropicPassthroughPlansView` and bailing
  to identity on any ambiguity. The flip changes the flagship wire (it is no longer pure
  byte-for-byte forwarding) — that is the one behavior change a reviewer accepts here.
- **Narrow blast radius.** The flip is the gateway buffered/passthrough planner only;
  `FAK_CTXPLAN_SEAM` (agent-loop seam) and `FAK_INKERNEL_KVMMU` (in-kernel KV eviction) stay
  off.

## Honest residuals

- The heaviest **real-transcript** residency-shrink replay
  (`go run ./cmd/ctxplanbench -heaviest 5 -budget 8000`, the ~13.3× fewer-resident-tokens
  headline) was **not re-run on this host this session** — transcript discovery hung under
  heavy build-cache contention. The prior committed measurement
  ([CTXPLAN-REAL-TRANSCRIPT-MEASUREMENT-2026-06-23](CTXPLAN-REAL-TRANSCRIPT-MEASUREMENT-2026-06-23.md):
  13.3× fewer resident tokens, 100% of misses served, exact recall 715/715) stands as that
  evidence.
- The flip itself (`--ctx-view-budget` 0 → 8000 across `serve.go`/`guard.go` plus the
  `token_defaults.go`/`servewiring.go` parity locks and the regenerated scorecards) is the
  named next step. It is currently blocked on `cmd/fak` building — peer WIP in
  `cmd/fak/egress.go`/`accounts.go` (a `mustJSON` redeclaration) breaks the package, so the
  `TokenDefault` tests and the `fak`-binary doc regeneration cannot be witnessed green yet.
