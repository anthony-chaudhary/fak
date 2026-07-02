---
title: "Ultracode dogfood — a 9-agent fleet advancing Qwen3.6 parity (2026-06-28)"
description: "A measured session record: running fak's ultracode (multi-agent) mode as two concurrent waves to advance Qwen3.6-27B-vs-llama.cpp parity, with the honest agent-orchestration concurrency numbers and an explicit fence keeping them apart from fak's separate inference 5-10x."
date: 2026-06-28
---

# Ultracode dogfood — a 9-agent fleet advancing Qwen3.6 parity

This is a **session record**, not a hero benchmark. It documents one run of fak's
*ultracode* (multi-agent) operational mode — a concurrent fleet of coding agents on
disjoint file lanes of one live trunk — pointed at the Qwen3.6-27B-vs-llama.cpp parity
goal. It reports the **measured** agent-orchestration numbers honestly, and it keeps that
multiple strictly separate from fak's **inference** 5-10x (a different axis, cited below,
never blended). The metric is defined in
[`../explainers/ultracode-multi-agent-dogfood.md`](../explainers/ultracode-multi-agent-dogfood.md);
the machine-readable witness is
[`../../experiments/agent-live/ultracode-dogfood-witness-20260628.json`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/agent-live/ultracode-dogfood-witness-20260628.json).

> **Honesty (`../proofs/00-METHOD.md`).** Assembled on a `windows/amd64` orchestrator with
> **no Apple Silicon, no NVIDIA GPU, no 27B artifact**. Every speed/GPU/27B figure cited
> here is a **recorded prior Mac-node witness**, never re-measured here. A SKIP is not a
> PASS. Agent durations are each subagent's self-reported wall time.

---

## 1 — What the fleet did (two concurrent waves)

**Wave 1 — production (4 concurrent agents, window opened 15:16:14Z).** Four disjoint
lanes, each producing one reviewed artifact toward the parity goal:

| Lane | Deliverable | Agent wall-time |
|---|---|---:|
| docs | [`benchmarks/QWEN36-PARITY-ROLLUP-2026-06-28.md`](../benchmarks/QWEN36-PARITY-ROLLUP-2026-06-28.md) — single-page parity rollup (proven vs `not yet`, with one-command Mac repros) | 165 s |
| experiments | [`qwen36/token3-drift-investigation-2026-06-28.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/qwen36/token3-drift-investigation-2026-06-28.md) — root-cause investigation of the token-3 correctness drift + a per-layer divergence-probe design | 290 s |
| tools | [`qwen36_mac_parity_gate.sh`](https://github.com/anthony-chaudhary/fak/blob/main/tools/qwen36_mac_parity_gate.sh) — one-command Mac gate emitting a gradeable witness JSON | 647 s |
| docs | [`explainers/ultracode-multi-agent-dogfood.md`](../explainers/ultracode-multi-agent-dogfood.md) — the ultracode-mode definition + the value metric | 150 s |

**Wave 2 — independent verification (5 concurrent agents, window opened 15:35:04Z).** A
second fleet that *independently re-checked* wave-1 before ship — citations, links, test
names, flags, numbers — and was empowered to fix hard errors:

| Lane | Verdict | Agent wall-time |
|---|---|---:|
| verify rollup | **PASS** — 13/13 test names resolve, 18/18 links resolve, no stray `6.12`, commit hashes consistent; no edits | 559 s |
| verify drift citations | **PASS** — every `qwen35.go` symbol + line range accurate, mRoPE-refutation test+assertion real, phenomenon facts match source; no edits | 202 s |
| verify gate-script flags | **PASS** — `bash -n` clean, every CLI flag confirmed defined, `SKIP!=PASS` holds, the text→ids gap documented (not papered over); no edits | 519 s |
| verify explainer links | **PASS** — 7/7 links + 5/5 packages resolve, the separation law intact, no fan-out contradiction; no edits | 339 s |
| build-sanity | **SKIP** — subagent Bash refused (`POLICY_BLOCK/TERMINAL`); no build verdict produced (counts as 0 deliverables) | — |

Every verifier that could run **corroborated** wave-1 with specific evidence rather than
rubber-stamping it — the point of the second wave. The one build-sanity lane was
policy-blocked in the subagent sandbox; recorded as a SKIP, not rounded to a pass. (The
host-independent green `TestQwen35HybridViaMMMatchesCPUTemplate` is a prior committed
witness, cited in the rollup; the four shipped artifacts touch **no Go code**, so a fresh
build is not a ship prerequisite for them.)

---

## 2 — The measured value (agent-orchestration axis)

Using the metric from the explainer (concurrency factor = sum of successful agent
durations ÷ the window's longest successful agent; deliverable-count = independent reviewed
non-stub artifacts landed vs one-at-a-time):

| | Wave 1 | Wave 2 | Session |
|---|---:|---:|---:|
| Productive deliverables | 4 | 4 | **8** |
| Policy-blocked SKIPs | 0 | 1 | 1 |
| Agent labor (sum of durations) | 20.9 min | 27.0 min | **47.9 min** |
| Agent-phase wall-clock | 10.8 min | 9.3 min | **20.1 min** |
| Wall-clock concurrency factor | **1.93×** | **2.89×** | **2.38×** |
| Deliverable-count throughput | 4× | 4× | **8×** |

**The honest reading.** ~48 minutes of agent labor was compressed into ~20 minutes of
agent-phase wall-clock — a **2.38× wall-clock** speedup — and **8 independent reviewed
deliverables** landed that a single agent would have produced one after another. The
wall-clock multiple is **below** the deliverable count because the lanes were *imbalanced*
(one 647 s lane vs one 150 s lane) — exactly the Amdahl ceiling the explainer warns about:
the orchestrator waits on the slowest lane. Wave 2's tighter spread lifted it to 2.89×.
**Balanced lanes drive the factor toward N**; reaching a 5-10× wall-clock needs ~5-10
similarly-sized lanes, which is the lever to pull next, not a number to claim here.

---

## 3 — This is NOT fak's inference 5-10x (the separation law)

fak's **inference** value — the 5-10x family — is a *different axis* measured by a
different method, and is **never** multiplied or added to the orchestration factor above.
Those numbers live in
[`../../BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md),
e.g. session value-add **7.2×→10.0×** (Qwen2.5-1.5B, T=8→16), the README headline
**60.3× vs naive / 4.1× vs tuned** (50-turn × 5-agent reuse), and RadixAttention
**4.58×→6.95×**. This session *used* the agent fleet to advance the work that makes those
inference claims true and well-documented; it does not restate them as an orchestration
result.

---

## 4 — Parity outcome (what the fleet actually moved)

- **Correctness parity:** PROVEN at the architecture level (the tiny `qwen3_5` fixture is
  bit-exact vs HF, cosine 1.000000), **REFUTED at 27B scale** (token-3 near-tie argmax
  flip: fak `[248068,198,8160]` vs llama.cpp `[248068,198,90700]`). The fleet's
  investigation localizes the prime suspect to the **GDN delta-rule recurrent scan** (the
  only stateful op — matching the "agree 2 tokens, diverge on 3" signature) and *refutes*
  an mRoPE-section hypothesis (the text forward collapses mRoPE to plain partial-RoPE,
  guarded by a test). Next step: the per-layer divergence probe (design shipped; the 27B
  run is Mac/artifact-gated).
- **Speed parity:** `not yet` — fak M3 Pro single-stream decode 0.1→0.9→1.2 tok/s vs the
  **7.29 tok/s** llama.cpp-Metal bar (~6× under); the wall is command-buffer launch
  overhead (kernels are bit-correct), and the closing levers are Apple-Silicon-gated.
- **Gated residual:** the Mac M3 Pro `fakmetal` GPU-numerics + on-device tok/s witnesses
  need an Apple-Silicon host this orchestrator does not have. The one-command gate that
  produces a gradeable witness on that host is
  [`../../tools/qwen36_mac_parity_gate.sh`](https://github.com/anthony-chaudhary/fak/blob/main/tools/qwen36_mac_parity_gate.sh).

Full reconciliation:
[`benchmarks/QWEN36-PARITY-ROLLUP-2026-06-28.md`](../benchmarks/QWEN36-PARITY-ROLLUP-2026-06-28.md).
