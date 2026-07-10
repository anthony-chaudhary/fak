---
title: "Triage: continuous-batching concurrency sweep for GLM-5.2 (GPU server 2 / Lane B / L2, #3079)"
description: "Generation-horizon classification and blocker record for issue #3079 — the L2 continuous-batching aggregate-throughput sweep. Classifies the work gen/now, names why the benchmark-curve artifact cannot be produced from the dev box, and pins the smallest next step plus the artifact contract the GPU-node worker must emit. Contains NO served numbers."
---

# Triage — #3079: continuous batching for GLM-5.2, concurrency sweep 1→128 (GPU server 2 · Lane B · L2)

> **What this is.** A *triage / classification* record for issue
> [#3079](https://github.com/anthony-chaudhary/fak/issues/3079), a child of epic
> [#3073](https://github.com/anthony-chaudhary/fak/issues/3073). It classifies the
> work's generation horizon, records the host blocker that keeps the benchmark artifact
> open, and pins the artifact contract the GPU-node worker must emit.
>
> **What this is NOT.** It is **not** the benchmark artifact. No served throughput or
> latency number appears here. The only figures cited are already-published labels:
> the **WITNESSED** 23.2 tok/s single-stream baseline and the **COMPUTED** roofline
> ceilings from
> [`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md).
> The Accept clause of #3079 — a recorded `experiments/benchmark/runs` curve labelled
> WITNESSED/OBSERVED — remains **open** (see §3).

## 1. The ask (verbatim intent)

Launch the resident serve with `--parallel N --cont-batching`; sweep concurrency
`{1,2,4,8,16,32,64,128}`; at each point record aggregate decode tok/s, per-stream tok/s,
and p50/p95 TTFT; watch KV residency against the ~206 GiB free VRAM. Accept = a
concurrency→{aggregate tok/s, per-stream tok/s, p50/p95 TTFT, KV GiB} curve artifact;
headline = peak aggregate tok/s and the knee.

## 2. Generation classification — `gen/now`

Horizon: **`gen/now`** (milestone *Generation G0 — Now / Immediate*). The issue carried
no generation label or milestone at intake; this record closes that classification gap
(the label/milestone binding is applied on the issue itself).

Basis, against the `docs/generation.md` test ("improves the current product, operator
loop, or trunk hygiene with a clear witness and no dependency on a future architecture
bet"):

- **Current-product improvement.** Standing up continuous batching raises aggregate
  throughput of the *current* resident serve on the *current* 8-GPU datacenter server (sm_80) lab iron —
  no new architecture is required.
- **Existing engine lever.** L2 rides `llama.cpp` `--parallel` + continuous batching, an
  engine capability that already exists; this is a serve-config + measurement stand-up,
  not a research bet.
- **Concrete witness.** The witness is the recorded curve artifact under
  `experiments/benchmark/runs`, not a self-report — a current-product measurement, which
  is exactly what `gen/now` requires.

This is not priority laundering: `gen/now` reflects the horizon (current-product,
existing-engine, concrete witness), independent of the issue's B-track priority.

### Closing evidence (generation contract)

- **Promotion evidence.** Parent epic #3073 is an active day-scale drive on current
  hardware; lever L2 uses an already-shipped engine path; the acceptance witness is a
  concrete recorded artifact. All three point *toward* now, so the item is promoted from
  unclassified to `gen/now`.
- **Demotion / retirement evidence.** None found. The work is neither superseded nor
  stale: no sibling has recorded a GPU server 2 Lane-B aggregate curve
  (`experiments/benchmark/runs/by-machine/` has no `gpu-server-2` node dir as of this note), and
  the epic remains OPEN. If a future `gen/second-next` native-EP path (#1482, Lane E)
  supersedes the llama.cpp batching lever, revisit — but that is a *separate* engine, kept
  separate by the epic's rules, and does not retire this measurement.
- **Invalidating assumption (recheck).** The classification assumes the aggregate ceiling
  stays a *current-product* target. It rests on two **ESTIMATED** ceiling inputs
  (active-params ~32B, active-bytes ~13 GiB/token); Lane F replaces those with GGUF-header
  truth. If Lane F re-derives the practical ceiling low enough that batching yields no
  fleet-value gain over single-stream, the L2 lever — not this classification's horizon —
  is what would demote.

## 3. Blocker — why the benchmark artifact is not produced here

The Accept clause requires a **live** resident serve and a full concurrency sweep on
**GPU server 2 / Lane B**, on 8-GPU datacenter-server (sm_80) iron, labelled
WITNESSED/OBSERVED. This dispatch ran on the Windows dev box, which:

- has **no GPU** and cannot host the resident GLM-5.2 serve (433.82 GiB resident);
- reaches the lab hardware only through the private control bridge
  (`private-comms-channel.md` → `../fak-private`), and running
  a resident serve plus an eight-point sweep is a major live operation, **outside this
  dispatch's declared "triage only" risk envelope**;
- must not fabricate the curve — a self-authored number is not a witness, and the repo's
  acceptance bar is an engine-honest recorded artifact.

So the numbers gate is **unreached by design on this host**, not failed. The honest state
is `not yet`: the classification landed; the measurement is owed by a GPU-node worker.

## 4. Smallest next step + the artifact contract

Execute on **GPU server 2 / Lane B** (a worker resident on the node, or an operator driving the
private bridge per `private-comms-channel.md`), then record the artifact under:

```
experiments/benchmark/runs/by-machine/gpu-server-2/<UTCstamp>-glm52-l2-contbatch-sweep/
  manifest.json     # benchmark/run-manifest.v1 (machine_id: gpu-server-2, claim_class: WITNESSED, git rev/dirty)
  result.json       # the curve: one row per concurrency point
```

Each `result.json` row, one per concurrency N in `{1,2,4,8,16,32,64,128}`:

| field | meaning |
|---|---|
| `concurrency` | N (the `--parallel N` degree) |
| `aggregate_decode_toks` | summed decode tok/s across all streams |
| `per_stream_decode_toks` | aggregate ÷ N |
| `ttft_p50_ms`, `ttft_p95_ms` | time-to-first-token percentiles |
| `kv_gib` | KV-cache residency at that point (watch vs ~206 GiB free VRAM) |

Headline to report on the issue: **peak aggregate tok/s** and the **knee** (the N past
which aggregate flattens or TTFT p95 breaks the SLO). Keep the llama.cpp engine-honest
baseline separate from any pure-fak kernel number (epic rule; #1482 lane stays separate).

## 5. Links

- Issue: [#3079](https://github.com/anthony-chaudhary/fak/issues/3079) · Epic:
  [#3073](https://github.com/anthony-chaudhary/fak/issues/3073)
- Ceiling: [`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md)
  (Ceiling B — aggregate: practical ~11–14k tok/s COMPUTED; 80% target ~9–11k)
- Reaching the hardware: `private-comms-channel.md`
