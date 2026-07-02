---
title: "fak executive roll-up — the few things leadership needs to know"
description: "An aggregated executive snapshot of fak: the flagship wins, the live strategic goal, the real risks and the one open decision — every number carrying an honest provenance label (witnessed vs observed vs simulated vs unverified). Synthesized from PRODUCT-STATUS, BENCHMARK-AUTHORITY, the AgentDojo red-team, the dispatch audit, and the industry scorecard."
---

# fak executive roll-up — 2026-06-27

_The aggregation of the most important items across fak's reports and concept docs:
[PRODUCT-STATUS](PRODUCT-STATUS.md), [BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md),
the [AgentDojo red-team](https://github.com/anthony-chaudhary/fak/blob/main/experiments/agent-live/agentdojo-fak-fullstack-20260627.md),
[dispatch-status](dispatch-status.md), the [GLM-5.2 cache-value packet](benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md),
and the industry scorecard. Synthesized by a 7-agent aggregation+critic workflow; every
number cross-checked against the tree and carries a provenance label._

> **How to read the labels.** **WITNESSED** = fak authored and proved it (tests / a
> committed artifact). **OBSERVED** = a reading relayed from an external party (a box, a
> provider). **SIMULATED / PROJECTED** = a modeled stand-in, not measured. **UNVERIFIED** =
> claimed, no witness yet. The discipline is the point: a labeled gap counts as honest; an
> overclaim counts as a defect.

## The one-paragraph version

fak is one ~13 MB dependency-free Go binary that drops in front of any agent (Claude Code,
OpenAI, MCP) and turns every tool call into a permission check. Its load-bearing guarantee
is **structural**: a dangerous lever that isn't on the allow-list cannot be called no matter
what the model is told — refusal by construction, not by classifier. On its own AgentDojo-style
red-team it drives attack-success from **0.76 (detection alone) to 0.000 (full stack)** while
keeping benign tasks working. The product surface is mature and honestly mapped (11 laptop-runnable
products, grade-A scorecard). Two things to watch: a **closure-honesty problem** in the issue
backlog (only ~21% of "closed" issues are genuinely resolved), and **one open decision** —
the provider-side cost-savings claim is not yet verified on real traffic and should not be led
with until it is. The live strategic goal (GLM-5.2 in fak's own kernel) is on-track but
**box-gated**, not yet delivered.

## Flagship wins (WITNESSED)

- **Structural injection floor, proven end-to-end.** AgentDojo-style red-team: full-stack fak
  **ASR = 0.000 (0/38 attacks land)** vs **0.763** for a detection-only baseline; benign
  controls **1.0**; gate **PASS**. 9 attacks stopped at the parse floor, 29 at the IFC
  sink-gate. Commit `bf015c3d`, clean tree, corpus hash pinned. **Boundary:** this is fak's
  structural floor over a *fixed, fak-authored* corpus — not a public leaderboard and not a
  model-capability score. Reproduces with no key, no GPU, no network.
- **One binary is the whole governed surface.** Capability floor + result quarantine +
  trace-correlated audit + OpenAI/Anthropic/MCP wires, in a single static Go binary — no
  Python/CUDA toolchain, no sidecar fleet. The same artifact a dev runs on a laptop is the
  one you harden for a fleet.
- **Bit-exact mid-run KV eviction.** fak can cut a poisoned tool result out of the *middle* of
  a live KV cache and the model behaves as if it never saw it (`max|delta| = 0.000`, with a
  non-vacuity control). No shipped serving engine does mid-run eviction. _Honest fence: the
  v1 deletion certificate is self-attesting (integrity, not third-party independence)._
- **Mature, honestly-mapped product surface.** 11 durable laptop-runnable products, 13 more
  usable, 21 witnessed subsystems with no direct command yet; 100% concept coverage,
  product-debt 0, grade A. _Read it right: the score grades how complete and honest the product
  **map** is, not how much fak wins — a labeled honest stub counts as accurate._

## Performance — the honest headline

- **Lead with 4.1×, not 60×.** On a 50-turn × 5-agent run (Qwen2.5-1.5B, Apple M3 Pro,
  commit `2bbda6f`): **4.1× vs a tuned warm-cache baseline** is the number that survives
  procurement. The **60.3×** figure is *only* vs a naive re-send-everything-every-turn loop
  nobody runs in production — it must never appear as the standalone competitive number.
  [WITNESSED on one machine + a deterministic model.]
- **Prefix reuse scales predictably:** 4.58×–6.95× wall-clock across the small-model ladder,
  a hardware-independent **7.50× token reduction**, 86.7% hit rate, reproducing bit-identically
  on Mac arm64 and Windows x86_64. _Fence: both arms run on fak's own kernel — this is the
  self-reuse multiplier, not a measured margin over a live SGLang/vLLM._ [WITNESSED]
- **WebBench 8.8×–9.7× is a MODELED prefill-work floor** over the real 643-task set — derived
  geometry, not wall-clock or cost. The doc stamps it "not measured." [SIMULATED]
- **Power / energy numbers are SIMULATED placeholders** (no power meter on the box). Easiest
  line to misread as measured — keep it fenced.

## Trust & security

- The structural guarantee is the **capability floor**; the result-quarantine detector is
  **best-effort and ~100% evadable by design** — it is explicitly *not* the load-bearing
  defense. Any summary that frames the detector as the guarantee overstates it.
- **Prior-art honesty: 0/29 primitives are novel.** The contribution is the *assembly* — a
  fused, fail-open, witness-gated kernel with the tool call promoted to an in-process syscall,
  co-resident with the KV cache. The honesty ledger itself is the one moat competitors
  structurally cannot copy.
- **Scope:** one injection vector demonstrated live; generalization to a full attack matrix ×
  the model ladder is open forward work, not shipped coverage.

## Live strategic goal — GLM-5.2 in fak's own kernel (epic #1010)

- **On-track, box-gated — not yet a result.** The goal: serve GLM-5.2 from fak's pure-Go kernel
  and prove the cache-value lever (`kv_prefix.reused_tokens`) end-to-end on a real solved
  SWE-bench ticket. The observation seam (`fak swebench cache-witness`) is **shipped and
  tested** (`52dfea0d`); the live cache number is **PENDING — `not yet`**, residual is
  datacenter GPU access (#1012). [UNVERIFIED/pending — *not* simulated; there is simply no
  number yet.]
- **Why route around, not through:** GLM-5.2 in-kernel decodes at **~0.03–0.17 tok/s** under
  CPU-offload — a real, witnessed MoE expert-GEMM bandwidth wall (#996/#971), ~1000× slower
  than a resident path. This is a host cost, never attributed to a fak action. The proof
  routes around it via cache value on an already-solved ticket. [OBSERVED — box reading.]
- **Open caveat:** GLM-5.2 *output correctness* on current code is unverified (the strong
  HF-oracle tests skip — no torch on the box). The cache-value witness only matters if output
  is correct. Don't ship GLM as "proven" until the oracle runs or a real ticket returns sane
  output.

## Positioning

- **fak does not compete on raw token throughput.** As a gateway fronting SGLang it trails
  raw SGLang **0.75× at peak** (1085.6 vs 1451.6 tok/s, Qwen3.6-27B, 64-concurrent) — the
  gateway/adjudication tax, converging to ~3% at saturation. Naming this gap pre-empts the
  "your benchmark is naive" attack; fak's field is governance/containment that *fronts*
  vLLM/SGLang/llama.cpp, not a faster engine. [WITNESSED]
- The genuine moat: in-kernel capability floor + bit-exact KV reuse + the honesty ledger —
  the assembly, not any single invented primitive.

## Risks & the one open decision

1. **DECISION — provider cost realization is UNVERIFIED.** fak's *own* accounting reports
   **85.98% token-shed** on replayed real Codex usage (WITNESSED, internal). But whether the
   provider's billing actually cascades the cache hit (vs re-billing the dropped middle as
   fresh input) is **not** witnessed by provider telemetry — epic #745 needs one credentialed
   real-traffic session scraped to settle. **Do not lead marketing with provider-side cost
   savings until then.** Frame as "mechanism proven, cost realization unverified."
2. **RISK — closure honesty is 0.207.** Of 808 closed issues only **167 are TRUE_RESOLVED**;
   641 are CLAIMED_CLOSED. Real open work is ~808, not the 166 the backlog shows. This is the
   credibility bottleneck for any velocity story — commits ship (~118/6h) far faster than
   issues genuinely close (39/6h). [WITNESSED — dispatch audit.]
3. **RISK — the dispatch loop is cold,** not capacity-bound. 0/2 workers live (headroom 2),
   but last attributed close was ~231 min ago; 6h completion rate **6.5/h vs 10/h target**.
   The block is per-worker completion, not compute or worker headroom — adding workers won't
   help until the loop's real block clears.
4. **RISK — KV-quarantine not yet in the live loop (#579).** The flagship mid-run isolation is
   proven on a synthetic model but the live serve/agent loop doesn't call it yet — the keystone
   before fak can claim live KV isolation on a real turn.

## Provenance warnings the critic flagged

- The **60.3×** must always travel paired with the **4.1×-vs-tuned** figure; standalone it
  reads as a competitive number it is not.
- The **85.98% Codex save** is fak's internal shed accounting, *not* provider-confirmed billing
  — keep it separate from the (unverified) provider cost-cascade claim.
- The **GLM cache-value** is pending/unverified, *not* simulated — no modeled number stands in.
- A **grade-A 100/100** scorecard grades map completeness + honesty, not product maturity or
  "fak wins."

---

_Regenerate: this roll-up is synthesized from the linked source docs; each is independently
regenerable by its own tool (`tools/product_scorecard.py`, `tools/dispatch_status.py`,
`go run ./cmd/agentdojoredteam -json`). Numbers are bound to commits in
[BENCHMARK-AUTHORITY.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)._
