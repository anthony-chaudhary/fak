---
title: "The fak roadmap: what's shipped and what's next"
description: "A reader-facing view of fak's direction on the now/next/future taxonomy: what is shipped and git-witnessed today, what is planned next, and the longer bets."
slug: roadmap
keywords:
  - fak roadmap
  - what we're building next
  - agent kernel roadmap
  - shipped vs planned
  - now next future
  - generation taxonomy
  - durable sessions
  - cache default-on
  - native agent harness
date: 2026-07-06
---

# The fak roadmap: what's shipped and what's next

> **What this is:** an honest, reader-facing view of where fak is heading. Every
> item in **Now** is already on the trunk and git-witnessed. Every item in
> **Next** and **Later** is labeled **planned** — a direction we are building
> toward, not a delivery promise. Nothing here claims market adoption, and every
> number is a witnessed number.

fak is one static Go binary that treats every agent tool call like a syscall —
the model proposes, the kernel disposes. This page tells you which parts of that
idea you can run today, which are the near-term foundation, and which are
longer-horizon bets. It is the reader-facing cut of the internal
[6-workstreams roadmap note](../notes/ROADMAP-6-WORKSTREAMS-2026-06-29.md) and
this artifact is dimension **E — Social proof & community** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).

## The horizons (and how to read them)

The horizons below follow the [generation contract](../generation.md), which
partitions work into four streams by *which horizon owns the evidence*, not by
priority:

| Horizon here | Generation stream | Meaning |
|---|---|---|
| **Now** | `gen/now` | Shipped on the trunk with a git witness you can check. |
| **Next** | `gen/next`, `gen/second-next` | Near-term foundation; needs a gate, dogfood run, or default-exposure proof before it is done. |
| **Later** | `gen/future` | A longer-horizon bet kept visible without pretending it is on the current release train. |

A horizon is not a completion percentage and not a promise. Items move between
horizons by the contract's own verbs — `promote` when evidence retires a
blocker, `demote` when an assumption fails, `retire` when it is superseded,
`park` when it is true-but-not-active.

## Now — shipped and git-witnessed

These are on the trunk today. Each traces to a witness in
[`CLAIMS.md`](../../CLAIMS.md) / [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md)
or a linked explainer — not to a self-report.

- **The default-deny tool-call gate.** `fak manage -- <agent>` and `fak preflight`
  put a capability floor in front of any agent: a dangerous tool call is refused
  by default (`verdict=DENY reason=DEFAULT_DENY`), at roughly `362 ns` per
  decision. This is the [tool-call-as-a-syscall](../explainers/tool-call-is-a-syscall.md)
  boundary, and it is the security floor that does not depend on *catching* an
  attack.
- **Verify, don't trust (DOS).** `dos commit-audit` refuses a false "done" from
  git evidence rather than the agent's word, and refusals carry a structured
  reason from a closed vocabulary. This is the trust substrate under a fleet of
  agents.
- **The addressable, bit-exact KV cache.** Evict one span from the middle of a
  kept run and leave the cache bit-for-bit identical (`max|Δ| = 0`), so long
  sessions stop getting expensive. On a tuned warm-cache session the fused kernel
  does the honest **~4.1×** less work described in the
  [bench-story](stories/the-real-4x.md) — never the flattering naive multiplier.
- **One static Go binary, drop-in.** Repoint one base URL; the same artifact a
  dev runs on a laptop is what a platform team hardens for a fleet.
- **fak owns its native agent loop.** The send → get-tool-calls → execute →
  feed-back iteration lives inside fak's own kernel seam, not a third-party SDK.
  Epic [#1315](https://github.com/anthony-chaudhary/fak/issues/1315) and its eight
  build-children (#1316–#1323) are closed and diff-witnessed on disk.

## Next — planned, near-term foundation

Labeled **planned** (`gen/next` / `gen/second-next`): the pieces we are building
now that still need a gate, a dogfood run, or a default-exposure proof before
they are done. The live status of each is tracked as GitHub issues.

- **Durable sessions (planned).** A SQLite-backed session and turn store so a
  long agent run survives a restart — the strongest host seam borrowed from the
  established proxy harnesses. Milestone **M#1 Durable sessions**; task handoff
  at session completion is [#1434](https://github.com/anthony-chaudhary/fak/issues/1434),
  WAL-class turn persistence is [#1363](https://github.com/anthony-chaudhary/fak/issues/1363).
  Demonstrated in the `fak manage -- claude` front door; **not yet** milestone-landed.
- **Cache default-on (planned).** Make the addressable KV cache the default
  across the productized `fak manage` / `fak serve` path, not an opt-in. Milestone
  **M#2 KV cache value**; the bit-exact mechanism above already ships — this is
  about turning it on by default with a witnessed value run.
- **Native harness host seams (planned).** With the loop kernel already owned
  (see **Now**), the next steps are the host surfaces around it — streaming and a
  richer permission surface — layered on the kernel seam rather than replacing it.

## Later — longer-horizon bets

Labeled **planned**, `gen/future`: kept visible on purpose, honest that they are
bets rather than the current release train.

- **Neo-silicon / neo-cloud binding (planned bet).** fak as a vendor-neutral
  binding layer across new accelerators and clouds, so the same capability floor
  and cache travel to hardware beyond today's GPUs — open epic
  [#1678](https://github.com/anthony-chaudhary/fak/issues/1678). The hardware-abstraction
  fences are in [hardware-portability](../explainers/hardware-portability.md).
- **Datacenter-GPU pure-fak kernel, end-to-end (planned research).** Serve
  GLM-5.2 / Qwen-3.6 through fak's *own* CUDA kernel and cache with a live
  correctness and tok/s witness — no llama.cpp, no vLLM in the path. Epic
  [#1010](https://github.com/anthony-chaudhary/fak/issues/1010); primitives have
  landed, but the live end-to-end witness is **not yet** captured, so the
  CPU-offload baseline stands until it is.

## Honest-scope fence

This roadmap follows the same *verify, don't trust* discipline the project turns
on its own claims:

- **No market-adoption claim.** A roadmap signals direction, not traction. Real
  adopter stories land only when they are real — see [voices](voices.md).
- **Witnessed numbers only.** The `~4.1×` warm-cache result, the `~362 ns`
  per-decision guard tax, and `max|Δ| = 0` are witnessed. The naive `~60×`
  multiplier, "agent city", and power-per-dollar figures are **simulated /
  design-target** and are labeled as such wherever they appear.
- **No false novelty.** The 0/29 prior-art audit says fak assembles known parts
  at a new seam; it does not claim an invented primitive. The injection detector
  is evadable by design — the floor is the default-deny gate plus structural
  quarantine, not the detector.
- **Horizons move.** An item's horizon is planning data, not a date. It can be
  promoted, demoted, retired, or parked as evidence changes.

## Where to go next

- Who fak is for, one door each: [personas](personas.md).
- The pitch at three zoom levels: [pitch ladder](pitch-ladder.md).
- The maturity climb (model × backend grid): [milestone status](../milestones/STATUS.md).
- The full internal reconciliation this page summarizes:
  [6-workstreams roadmap](../notes/ROADMAP-6-WORKSTREAMS-2026-06-29.md).
