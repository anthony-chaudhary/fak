---
name: serve-receipt
description: On-demand research protocol + deterministic receipt for serving a Hugging Face model on a concrete GPU target — fan out research lanes (model facts/files, engine support matrix, serving recipes, KV/state memory math), compute the tiered feasibility verdict with `fak serve-receipt` (never narrated math), and emit a witnessed launch receipt. Use when asked for an "initial config", "serving receipt", or "how do I serve <model> on <GPUs>".
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, WebFetch, WebSearch, Agent
argument-hint: "<hf-model-id> <gpu-or-rig> [--engine vllm|sglang] [--max-model-len N]"
metadata:
  opencode: claude-only
---

# /serve-receipt — model → GPU serving-config receipt, on demand

Turn "give me a great initial config for <model> on <hardware>" into a repeatable,
witnessed artifact. Two layers: **agent lanes do the research; the `fak serve-receipt`
Go leaf owns every number.** The kernel does not trust narrated arithmetic.

## Inputs

- `model` — HF repo id (e.g. `nvidia/Kimi-K3.1-Pro-NVFP4`).
- `gpu` — concrete target (e.g. "RTX PRO 6000s", "8×B200"). Resolve ambiguity: if the
  name spans generations, ASK or state the assumption explicitly (e.g. Ada 48GB vs
  Blackwell 96GB differ by FP4 support and 2× memory).

## Lane protocol (fan out in parallel; each lane returns a ≤45-line receipt with URLs)

1. **Facts/files lane** — model card + `config.json` + exact shard bytes
   (`https://huggingface.co/api/models/<id>?blobs=true`): total download bytes (exact,
   not estimated), per-dtype census, layers (full-attn vs linear/SSM split), MLA dims
   (`kv_lora_rank`, `qk_rope_head_dim`, indexer dims), expert counts, attention-head
   counts (ALL head groups — TP must divide every one), vocab, context, vision budget,
   `quantization_config`, recommended sampling.
2. **Support-matrix lane** — engine kernel support for the quant format on the target
   SM (e.g. NVFP4: native Blackwell CUTLASS/FlashInfer vs trtllm-gen datacenter-only vs
   Marlin fallback), image tags + CUDA/driver requirements, known-good/broken issues,
   interconnect reality (NVLink vs PCIe), version gates.
3. **Recipes lane** — known-good launch manifests for the model family on ANY hardware
   (recipes pages, vendor yamls, docker runs): verbatim args + env, measured tok/s,
   speculative-decoding configs, GB300-only vs portable knobs, pitfalls (timeouts,
   cudagraph capture memory, load format).
4. **Memory-math lane** — KV bytes/token (per layer × (latent + rope + indexer) with
   dtype bytes; note TP-replicated vs TP-sharded), per-SEQUENCE recurrent/SSM state,
   crossover math; verify against a published recipe number where one exists.
5. **Host-conventions lane** (local, read-only) — how the host repo fronts this engine
   (in fak: `fak serve --engine vllm`, campaign adapter schema, `examples/` conventions)
   so the receipt drops into local tooling.

## Synthesis rules (non-negotiable)

- **Tiered honesty**: produce a rig-size table (infeasible / tight / comfortable).
  An infeasible rig is STATED, never shipped as an OOM-on-load config. "Best effort"
  means the honest tiers, not a forced config.
- **Units discipline**: exact shard bytes from the API; label GB (10^9) vs GiB (2^30)
  everywhere; GPU VRAM is binary (96GB card = 96 GiB in nvidia-smi).
- **Replication vs sharding**: MLA latent/indexer KV is often TP-replicated (per-GPU
  KV does not shrink with TP); recurrent/SSM state usually is sharded. Getting this
  backwards flips verdicts.
- **Head divisibility**: TP must divide every attention head group (main heads AND
  sparse-indexer heads). Prefer PP-heavy layouts when interconnect is PCIe-only.
- Label every unconfirmed claim `UNVERIFIED` with what would confirm it.

## Deterministic core (the leaf — all arithmetic goes through it)

```bash
fak serve-receipt profiles
fak serve-receipt plan --facts examples/serve-receipt/<model>.facts.json \
  --gpu rtx-pro-6000-blackwell-96gb --tp 8 --pp 3 \
  --max-model-len 1048576 --max-num-seqs 32 --format md
# KV-bound rig: attach a host pool instead of failing the rig
fak serve-receipt plan --facts <facts.json> --gpu rtx-pro-6000-blackwell-96gb \
  --tp 16 --max-model-len 1048576 --host-kv-gib 32 --dcp 16 --curve
fak serve-receipt validate --receipt <receipt.json>   # re-derives every number; refuses mismatches
```

- Author a facts file (`fak.serve-receipt.facts/1`, see `examples/serve-receipt/README.md`)
  from lane 1+4 outputs with `observed_at` + `source_urls`; exact bytes only.
- Verdicts: `OK` / `TIGHT` / `OFFLOAD_REQUIRED` (exit 0, marked — device KV headroom does
  not kill a rig; attach a host pool and price the miss curve) / `INFEASIBLE` (exit 3 —
  weights; the expert-offload rejection arithmetic prints with the verdict). Offload
  flags: `--host-kv-gib`, `--host-pool-usability` (0.5 default, S8-anchor calibrated),
  `--dcp`, `--kv-variant fp8|nvfp4|nvfp4_mxfp4keys`, `--host-bandwidth-gbs`, `--curve`.
  Weight/expert offload is never a route — KV offload only. `validate` is the
  anti-narration gate — run it on the final receipt before presenting it.

## Receipt artifact & witnesses

- Save `examples/serve-receipt/<model>.<gpu>.receipt.{md,json}` (or scratch for one-offs).
- The receipt is NOT done until its witness checklist runs on the rig: server-load
  witness (kernel-selection log lines), 3-prompt accuracy spot-check at recommended
  sampling, `vllm bench serve` concurrency sweep, KV-headroom log cross-check vs the
  plan's `max_cached_tokens`. Until then the receipt says `not witnessed yet`.
- Link the tracking issue; land facts + receipt + leaf changes by explicit paths in one
  commit with the `(fak serve-receipt)` stamp.

## Worked example

Kimi-K3.1-Pro-NVFP4 × RTX PRO 6000 Blackwell (2026-11-12, issues #14362 + #14438):
facts at `examples/serve-receipt/kimi-k31-pro-nvfp4.facts.json`; verdicts — 8× INFEASIBLE
(expert-offload rejection printed), 16× OFFLOAD_REQUIRED (~1.01M tokens with a 32 GiB host
pool, miss curve attached), 24× TP8×PP3 OK (~1.29M-token KV pool). First end-to-end run of
this protocol; see that receipt for the lane-output shape.
