---
title: "vCache gate-enablement tickets (2026-06-30): the issues that turn the gates ON"
description: "The GitHub issues filed from the top-50 default-on-cache plan: a keystone epic plus children for attribution, gate enablement (M1/M2/M5), cross-path integration, and the QA + dogfood contract every gate must pass before it flips default-on."
---

# vCache gate-enablement tickets (2026-06-30)

Filed from [`VCACHE-DEFAULT-ON-TOP50-2026-06-30.md`](VCACHE-DEFAULT-ON-TOP50-2026-06-30.md).
The vCache milestones #715–#720 were closed as *built*; the audit on 2026-06-30
showed they are **built-and-gated-OFF**. These issues are the *enablement* layer:
each gate flips default-ON only after a paired **QA** (honesty test + non-forgeable
witness) and **dogfood** (it ran on our own guard/serve traffic) gate is green.

## The tree

| # | Title | Track | Turns on |
|---|-------|-------|----------|
| **#1490** | epic: turn the vCache gates ON + honest per-mechanism attribution | epic | the whole loop |
| #1491 | per-mechanism + per-owner saving attribution | A — attribution | the "provider P% + fak F%" headline |
| #1492 | register the M5 governor into a live loop | B — gates | decision witness live; pin/lazy/evict/warm actions still open |
| #1493 | M2 star-anchor canonicalization as a default pre-flight gate | B — gates | fak's authored slice `F` > 0 |
| #1497 | M1 warmth-belief estimator + per-provider calibration, live | B — gates | steering (not just scoring) |
| #1498 | push the vBlock/anchor abstraction across every serving path | C — cross-path | pure-fak, sglang, vllm, llama, API |
| #1495 | QA harness — the honesty-test + witness contract | QA | the gate before every default-on |
| #1496 | dogfood loop — run the live loop on our own sessions + post the P&L | dogfood | the proof on our own traffic |

## Consumes (open, related)

- **#1303** — persist the Track-2 OBSERVED-$ ledger per guard session. #1491's
  ledger fields and #1496's dogfood append depend on it.
- **#1407** — compaction fires 0× on real Claude Code traffic (anchor-starved).
  #1493 (the M2 anchor gate) is the fix class; #1496 (dogfood) must catch the
  pathology.

## The contract (why every gate has QA + dogfood)

A gate can be *built and tested* and still be worthless if it never fires on real
traffic — that is exactly what happened to M1–M5. So default-on is **earned**:

1. **QA** proves the gate is *correct and honest* — correctness never depends on
   warmth (Law A2), a false-warm is demoted not trusted, every number is labeled
   by owner (OBSERVED vs WITNESSED), and the verdict is deterministic and witnessed
   by a hash-chained journal row, not a self-report.
2. **Dogfood** proves the gate *actually fires on our own usage* — it ran on a real
   `fak guard`/`fak serve` session, wrote a ledger/journal row, and moved fak's
   authored slice `F` above zero on a multi-turn session.

Only then does the default-on lever (`--vcache-governor`, `--vcache-anchor`, …)
flip to on.
