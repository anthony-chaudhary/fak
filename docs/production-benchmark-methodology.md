---
title: "fak production benchmark methodology: reproducible LLM throughput"
description: "How to reproduce fak production benchmarks from a clean checkout: parity and agent-fleet throughput, raw artifacts, clean-box rule, and tuned-reference policy."
---

# fak production benchmark methodology

This is the locked Phase 0 reproduction contract for `docs/production-readiness.md`.
The goal is that a skeptic can run the parity and fleet-throughput claims from a
clean checkout and keep the raw evidence.

## One command

From the repo root on the benchmark node:

```bash
bash tools/fak_node_bench.sh --pull
```

Smoke check, for wiring only:

```bash
bash tools/fak_node_bench.sh --short --host=phase0-smoke
```

Artifacts are written under `fak/experiments/fleet-nodes/<host>/` unless `--out=DIR`
is supplied. The manifest is `production-readiness-manifest.json`.

## Required raw artifacts

Each node run must preserve:

- `node-info.json` — host, OS, arch, CPU, core count, Go version, git rev.
- `production-readiness-manifest.json` — artifact list and workload caps used.
- `modelbench-q8.json` — Q8 parity lane: prefill/decode, plus recorded-workload
  prefill/decode sections when the workload file is present.
- `batchbench-q8.json` — multi-user batched decode throughput curve. Full canonical
  runs use the benchmark's standard prompt setup by default; recorded-workload
  prompt replay for batchbench is an opt-in diagnostic via `FAK_BATCH_WORKLOAD=1`
  because long recorded contexts multiplied by high batch sizes can exceed memory.
- `fleetbench.json` and `fleetbench.csv` — fleet turn-tax surface.
- `modelprof.json` and `q8kernel.txt` — diagnostic roofline/kernel context.

## Workload

The real-workload shape is pinned in:

```text
fak/experiments/agent-live/production-workload.json
```

It is derived from recorded `fak agent` A/B reports under `fak/experiments/agent-live`.
Because the model benchmark consumes token IDs directly and has no tokenizer on the
proof path, it replays deterministic token IDs at the recorded prompt/completion
lengths. Token identity is not treated as a semantic workload claim; the measured
compute cost is shape-driven. If a future tokenizer leaf lands, this methodology must
be revised to replay actual recorded token IDs.

For a full Phase 0 run, leave modelbench workload caps unset or set them to `0`:

```bash
FAK_WORKLOAD_PREFILL_CAP=0 FAK_WORKLOAD_PROMPT_CAP=0 bash tools/fak_node_bench.sh --pull
```

`FAK_WORKLOAD_PREFILL_CAP` controls the recorded prompt lengths replayed by
`modelbench`. `FAK_WORKLOAD_PROMPT_CAP` is recorded in the manifest for the optional
batchbench workload diagnostic, but the full canonical throughput curve does not use
that diagnostic unless `FAK_BATCH_WORKLOAD=1`.

For smoke runs, caps are allowed but the manifest must be quoted with the results.
Capped smoke data cannot close the clean-box Phase 0 gate.

## Clean-box rule

The canonical Phase 0 numbers must come from a node that is not running the live
multi-session fleet load. Before quoting a result:

- run from a clean checkout at the manifest git rev;
- record the node metadata in `node-info.json`;
- run the full, uncapped command above;
- keep the raw JSON/CSV, not just pasted tables;
- compare nodes with `python tools/fak_node_compare.py --json`.
- close the gate with:

```bash
python tools/fak_phase0_gate.py fak/experiments/fleet-nodes/<host> --clean-node
```

`--clean-node` is a provenance assertion: use it only for an artifact directory
captured on a node that was not running the live multi-session fleet load. The
checker validates the raw artifact set, uncapped workload replay, full batch
curve, and the 45x batched-decode threshold.

The repo-local `--short` run proves only that the harness works end to end.

## Tuned Reference Policy

Phase 0 node artifacts measure the in-repo Go kernel path and its internal
ablations. They are not, by themselves, a fair external serving comparison. A
headline performance claim must default to a tuned reference stack: llama.cpp,
SGLang, vLLM, TGI, or another incumbent launched with the cache, batching,
precision, scheduler, and parallelism settings a competent operator would use.

Keep these rows separate:

- **Naive/no-reuse:** diagnostic only. It shows the redundant work the lever can
  remove, and is useful for sanity-checking the ceiling.
- **Tuned reference:** the default denominator for external performance claims.
- **FAK current:** measured current artifact from this checkout.
- **FAK tuned:** measured after a specific FAK tuning change, or an explicitly
  labeled projection over FAK current.

The important question is not "how much faster are we than an untuned path?"
It is "how much of the remaining gap to the tuned reference can our own tuning
close?" If a current run shows 2x or 3x against a tuned reference, record that
as the current point. If a fan-out or prefix-reuse ablation suggests a much
higher internal ceiling, record it as candidate headroom until a tuned FAK
artifact proves it.

<!-- The "GPU server endpoint-load extension" section (lab-GPU server endpoint-load workflow,
     tools/dgx_*, docs/dgx-benchmark-methodology.md, and the private Benchmark
     harness) is excluded from the public copy -- operator-private lab infra. See
     PUBLIC-SCRUB-POLICY.md PRIVATE-ONLY list. -->
