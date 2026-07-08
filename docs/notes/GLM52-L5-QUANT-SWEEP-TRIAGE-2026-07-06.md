---
title: "GLM-5.2 L5 decode quant sweep (#3077): generation triage + GPU server-2 witness gate (2026-07-06)"
description: "Triage for issue #3077 (epic #3073, Lane A / L5): classifies the UD-Q4_K_M vs UD-Q4_K_S vs Q4_K-pure vs Q3 decode quant sweep as gen/now, records the arm matrix + result-table schema so a GPU server-2 operator can run it turnkey, and names the exact acceptance gate this GPU-less worker cannot reach. NO number here is a served measurement; the only WITNESSED cell is the UD-Q4_K_M baseline carried from the 07-01 note."
---

# GLM-5.2 L5 quant sweep (#3077): triage + witness gate

> **What this is.** A generation triage + turnkey harness spec for issue
> [#3077](https://github.com/anthony-chaudhary/fak/issues/3077) — the Lane A / L5 decode
> quant sweep under epic [#3073](https://github.com/anthony-chaudhary/fak/issues/3073).
> It classifies the horizon from issue evidence and hands a GPU server-2 operator everything
> needed to produce the WITNESSED artifact the issue actually accepts.
>
> **What this is NOT.** Not the benchmark. The sweep requires staging/serving four
> ~400+ GiB quant checkpoints resident on a GPU server-2 8-GPU datacenter server (sm_80) lab node — hardware this
> Windows worker cannot reach (see §4). The only cell below that is a served measurement is
> the **UD-Q4_K_M baseline**, carried WITNESSED from
> [GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md).
> Every other row is **PENDING** until an operator runs it; nothing here is fabricated.

## 1. The ask (verbatim intent)

Lever **L5** (~1.1–1.5× decode, the bytes/token lever in the
[ceiling map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) §5). The gcp-glm demo notes that
**UD-Q4_K_S** hits the CUDA resident-Q4_K fast path and streams fewer/faster bytes than the
mixed **UD-Q4_K_M**. Trade decode speed against quality across four arms; record decode tok/s
+ resident GiB + a small quality probe per arm so the speed/quality pick is **evidenced, not
asserted**.

## 2. Generation triage — `gen/now`

Classified from issue evidence per [docs/generation.md](../generation.md); the horizon is
clear, so this is **not** kept `needs-triage`.

- **Stream: `gen/now`** (milestone `Generation G0 - Now / Immediate`). #3077 improves the
  **current** product — GLM-5.2 as served today on the lab nodes at 23.2 tok/s WITNESSED —
  has a clear witness path (one benchmark run), and carries no dependency on a future
  architecture bet. That is the `gen/now` definition exactly.
- **Promotion evidence** (what makes it now-horizon): it is a child of epic #3073, the active
  "drive 23.2→~150 tok/s **in a day**" program; L5 is one measurable lever on the current
  llama.cpp resident serve, not a research option.
- **Demotion / retirement evidence** (what would move it off now): if Lane F's GGUF-header
  ground-truth shows the ~1.1–1.5× L5 gain is subsumed by L1 (8-GPU tensor split, ~3–6×) so
  the quant choice no longer changes the serving pick → **demote / park**. If a peer lands the
  sweep first on the shared trunk → **retire as duplicate**. If UD-Q4_K_S will not load
  resident or the fast path does not engage on sm_80 (see §3) → **retire the arm**.
- **Invalidating assumption** (the one the sweep exists to test): that UD-Q4_K_S actually hits
  a "CUDA resident-Q4_K fast path" and streams fewer/faster bytes than UD-Q4_K_M. That claim
  is asserted from the gcp-glm demo notes, **not yet witnessed on the GPU server-2 sm_80 (Ampere)
  nodes** — which sit below the sm_90 DSA kernel floor and have no FP8 (ceiling doc §1). If the
  fast path does not materialize on Ampere, or Q3's quality delta is unacceptable, the
  1.1–1.5× estimate is void and L5 collapses to a quality-loss-only arm.

## 3. The arm matrix (what to stage/serve)

Node **GPU server 2 · Lane A · L5**. Engine = llama.cpp sm_80 CUDA build (engine-honest baseline;
keep any pure-fak kernel numbers in a separate artifact). Serve config mirrors the 07-01
baseline: `--n-gpu-layers 999`, no `--n-cpu-moe`, `--ctx-size 8192`, `/v1` alias `glm-5.2`,
temperature 0, single-stream decode of a fixed 256-token continuation.

| Arm | Quant | Hypothesis | Resident GiB | Decode tok/s | Quality-probe Δ |
|---|---|---|--:|--:|--:|
| A0 | **UD-Q4_K_M** (baseline) | mixed dynamic; reference | **433.82** WITNESSED | **23.2** WITNESSED | 0 (reference) |
| A1 | **UD-Q4_K_S** | hits CUDA resident-Q4_K fast path | PENDING | PENDING | PENDING |
| A2 | **Q4_K-pure** | pure Q4_K, no dynamic mix | PENDING | PENDING | PENDING |
| A3 | **Q3** (e.g. Q3_K_M) | fewest bytes, largest quality risk | PENDING | PENDING | PENDING |

A0 row provenance: 07-01 resident-serve note (≈434.6 GiB incl. KV/buffers; 433.82 GiB weight
bulk; 23.23 / 23.22 tok/s two-run `print_timing`).

**Quality probe** (`Δ` column): one fixed cheap probe run identically per arm, reported as a
delta vs A0 — a swebench-smoke pass-rate delta, or a canary-completion diff (edit distance /
token-agreement) on a fixed prompt set at temperature 0. Pick one and hold it fixed across all
four arms; a speed win that silently loses quality is not a win (net-true-value standard).

## 4. The acceptance gate this worker cannot reach

**Acceptance** = a recorded `experiments/benchmark/runs` artifact, `claim_class: WITNESSED`
(or OBSERVED), containing the §3 table filled for arms A1–A3.

**Blocker (host capability).** This worker runs on the Windows dev box, which has **no GPU**;
native `go test` is even OS-blocked here (AGENTS.md build/test notes). The four arms are
~400+ GiB checkpoints that must be staged from NVMe and served resident across a GPU server-2 8-GPU datacenter server
(640 GiB VRAM) node reached only through the operator-gated `fak-private` control bridge
([docs/private-comms-channel.md](../private-comms-channel.md)). Producing the A1–A3 rows is an
**operator hardware action**, not something this worker can run — and inventing the numbers
would violate the WITNESSED bar and the "no fabricated pass" rule. So the rows stay PENDING.

**Smallest next step.** An operator on **GPU server 2, Lane A** stages each of UD-Q4_K_S / Q4_K-pure /
Q3 resident, runs the fixed 256-token single-stream decode + the chosen quality probe per arm,
and writes the artifact under
`experiments/benchmark/runs/by-machine/<dgx-2-node-id>/<UTC-timestamp>-glm52-l5-quant-sweep/`
(`manifest.json` with `$schema: benchmark/run-manifest.v1`, `claim_class: WITNESSED`;
`result.json`; a `RESULTS.md` with the §3 table). A `(fak docs)` commit citing `#3077` that
adds that artifact path is what closes #3077 — **not this triage note**.

> **Do not auto-close #3077 on this note.** This is a triage + harness increment; the
> benchmark acceptance (§4) is unmet. If a close-resolved / close-batch arm binds a #3077
> reference here, treat it as a false close and reopen until the WITNESSED artifact lands.

## 5. Cohort note

The whole #3073 lever cohort (#3076 L4, #3077 L5, #3078 L3, #3079 L2, #3080 KV-paging) is
currently unclassified on GitHub. Each is the same shape — one lever, one WITNESSED
`experiments/benchmark/runs` artifact — and all read `gen/now` under the same reasoning as §2.
Classifying the cohort (and the epic) in one pass is the operator follow-up; this note triages
the assigned leaf, #3077.

*Companions:* [ceiling + lever map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) ·
[8-GPU resident serve (23.2 tok/s WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) ·
[generation contract](../generation.md).
