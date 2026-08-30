---
title: "Issue #8821 — exact-Qwen3.8 CUDA decode-lever HOLD"
description: "Typed HOLD receipt for the unavailable quality-valid native CUDA profile, with the scrubbed sanctioned rerun route."
---

# Issue 8821: exact-Qwen3.8 CUDA decode-lever HOLD

This directory is a deterministic public **HOLD** receipt, not a CUDA profile
or a performance claim. It binds issue 8821, epic 10193 row 1, to repository base
`97ab1e4e3b34b26fd9f901c0a7d12f55b6bd3722`.

## Frozen envelope

- Artifact: `Qwen3.8-27B-Q4_K_M.gguf` from `unsloth/Qwen3.8-27B-GGUF`, revision
  `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Quantization: `Q4_K_M`.
- Target: NVIDIA A100 40GB, `sm_80`, CUDA.
- Workload: prompt/context/output token lengths `22/22/6`, batch 1,
  deterministic greedy decode.
- Required execution identity: `engine=fak-native`, `runtime=inkernel`,
  `path=cuda/qwen35-gdn-ssm-decode-v1`, `fallback=0`. llama.cpp is a
  comparator/reference only and cannot execute as a fallback.
- Quality gate: cosine similarity at least `0.995`, exact argmax equality, and
  maximum absolute logit error strictly less than `0.02`.
- Accounting: one wall-clock-bound receipt must include setup, recovery,
  prefill, first-token, steady-decode, verification, and teardown time.

## Why this is HOLD

The earlier issue 8819 witness failed its quality gate. Its zero CUDA device
counters are documented placeholders, so they cannot attribute a bottleneck or
support a lever.

The scrubbed sanctioned-route probe records the August 30, 2026 observation:

- the private bridge was `NOT_READY` and produced no qualifying CUDA session;
- the GCP probe was `STALE_AUTH`, returned no tiers, and therefore did not
  establish A100 eligibility;
- the pinned A100/fak-CUDA provisioning dry-run rendered successfully, but it
  is provisioning evidence only and not a qualifying model witness.

No account, project, hostname, private path, credential, or raw internal log is
included. The decision is `HOLD_NO_QUALIFYING_CUDA_EVIDENCE`, and
`selected_lever` is empty.

Current public tooling also lacks a full-model GDN capture that combines
per-stage CUDA-event attribution with the strict cosine/argmax/max-absolute
logit triad. Until one receipt supplies both, no decode lever may be selected.
No A/B was performed; any later A/B is limited to exactly one changed lever.

## Sanctioned rerun

Set `GCP_PROJECT` to the sanctioned project identifier and `GGUF_PATH` to the
absolute path of the exact named GGUF, then run:

```sh
GCP_PROJECT="$GCP_PROJECT" GGUF_PATH="$GGUF_PATH" \
  ./docs/_witnesses/issue-8821-qwen38-decode-lever/sanctioned-rerun.sh
```

The script verifies the artifact digest, runs the committed all-tier GCP probe,
requires the pinned A100 tier to be provisionable, runs the committed dry-run,
and prints the exact on-node CUDA acceptance/build/serve sequence. It exits
nonzero after printing the sequence because the missing full-model attribution
and quality-capture step must not be silently skipped.

Validate this receipt with:

```sh
cd docs/_witnesses/issue-8821-qwen38-decode-lever
go test ./...
bash -n sanctioned-rerun.sh
```
