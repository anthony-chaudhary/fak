---
title: "DeepSeek V4 Pro pure-fak native-GPU witness — bridge read-back + open-gate record (2026-07-16)"
description: "Scrubbed, public-safe status for issue #4781: a sanctioned private-bridge read-back attempt against the lab GPU control channel, its result, and the two concrete gates that still block a pure-fak NATIVE DeepSeek-V4-Pro witness. This is a liveness/blocker record, NOT the required model witness — #4781 stays OPEN until a real native run produces parity + the four perf metrics + a scrubbed artifact + a DOS witness."
---

# DeepSeek V4 Pro pure-fak native-GPU witness — status (2026-07-16)

> **Status: NOT the required witness. Issue [#4781](https://github.com/anthony-chaudhary/fak/issues/4781) stays OPEN.**
> This file records only a *bridge read-back attempt* and the exact remaining gates.
> It claims **no** model result: no parity, no TTFT/TPOT, no throughput, no peak
> memory, no concurrency number. Those exist only after a real native run, per the
> issue's closure binding ("Close only when the captured artifact exists and is
> independently readable; not from transcript narration").

Parent program: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006).
Placement routes: [#4801](https://github.com/anthony-chaudhary/fak/issues/4801) /
[#4806](https://github.com/anthony-chaudhary/fak/issues/4806) (sanctioned lab/cloud
topology); [#4807](https://github.com/anthony-chaudhary/fak/issues/4807) (pure-fak
bounded expert-streaming spine on the free 80 GB-class A100 ranks). Coordinates with
[#4367](https://github.com/anthony-chaudhary/fak/issues/4367). Sibling evidence
classes are the hosted-API scorecard ([`DEEPSEEK-V4-PERF-SCORECARD.md`](../benchmarks/DEEPSEEK-V4-PERF-SCORECARD.md),
#3014) and the self-host vLLM/SGLang baseline
([`DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`](../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md),
#3013) — kept distinct, never conflated with the native class.

## Evidence class discipline (why this is not a shortcut)

The issue's own confusion-risk note is load-bearing: **hosted V4 Pro, NIM,
vLLM/SGLang, and pure-fak native kernels are distinct evidence classes and must be
labelled separately.** #4781 asks specifically for the **pure-fak NATIVE** class —
fak's own in-kernel MoE execution on the node's GPUs — not a hosted number and not a
tuned-engine number fronted by `fak serve`. This record therefore cannot be satisfied
by any of the already-shipped hosted/self-host artifacts; it needs a native run.

## Bridge read-back attempt (2026-07-16, scrubbed)

The sanctioned private control bridge (`dgxbridge`, from the `fak-private` companion
repo — never vendored here) was driven against the lab GPU control channel from this
session, which now carries a working `SLACK_BOT_TOKEN` (the credential the issue was
filed without on 2026-07-14). No secret and no host/channel identifier is reproduced
here.

- **`... -dgx-host <node> -probe status`** → a live control thread was discovered
  (host resolved to the sanctioned 80 GB-class node), but the control session replied
  **`STALE (no control reply within timeout)`**.
- **`... sessions`** → **`no !sessions reply within timeout`**.

**Read-back verdict:** the bridge reaches the lab GPU control channel at the Slack
transport layer, but **no live `default-N` control session is currently answering** —
consistent with the node being operator-gated (a live session must be brought up by
the operator before jobs can be dispatched). This is a liveness blocker, not a fak
defect.

## Target node (from the sanctioned node inventory, generic)

The sanctioned target is a node with **eight 80 GB-class A100 GPUs (compute 8.0),
~640 GB aggregate VRAM**, CUDA 12.x, Go preinstalled — of which up to six ranks are documented
as safely free for the #4807 bounded expert-streaming spine. Exact host/session/channel
identifiers are intentionally omitted (public-leak discipline).

## Two concrete gates still open

1. **Liveness gate (operator).** A live control session must be answering the bridge
   (`sessions` returns a running `default-N`; `status` returns a live session) before
   any native job can be dispatched and read back.
2. **Native-loader gate (engineering).** DeepSeek-V4-Pro is a **1.6T-total / 49B-active
   MoE with 1M context**. Native in-kernel 1.6T-weight loading was explicitly deferred
   in the #3013 self-host runbook ("a native fak MoE follow-on … may be filed only if
   this baseline surfaces a concrete fak-owned gap"). The bounded pure-fak
   expert-streaming path that would run the model on the free 80 GB-class A100 ranks is
   tracked by **#4807** and is not yet landed. Until #4807 (or a #4801/#4806 topology)
   produces a runnable native path, there is no pure-fak native execution to measure.

## What "done" still requires (unchanged from #4781)

- [ ] A live control session on the sanctioned node (gate 1).
- [ ] A runnable pure-fak native DeepSeek-V4-Pro path on the node (gate 2 / #4807).
- [ ] Exact tested model/revision, fak module revisions, GPU class/count, command, env.
- [ ] Correctness/parity gate pass (or a preserved failure witness).
- [ ] TTFT, TPOT/tokens-per-second, peak memory, and concurrency vs a tuned non-fak
      baseline.
- [ ] A scrubbed artifact copied here and passing public-leak checks.
- [ ] Any newly discovered optimization filed as its own DoD-complete issue.

This file satisfies **none** of those model-measurement boxes; it only records the
read-back attempt and pins the blockers so the next operator/engineering step is
checkable.
