---
title: "Wave-2 RTX 5080 Release Witness: Inkling-2-Mini-UD-Q3_K_XL (C1-C4 <= 16K)"
description: "Scrubbed HW-WITNESSED RTX 5080 (sm_120) receipt for fak v0.60.0: 122.34 tok/s plain decode, 294-326 tok/s spec oracle, 0.002912 mean KL at 16K f16, and C1-C4 <= 16K HTTP serving vs llama.cpp."
---

# Wave-2 RTX 5080 Release Witness: Inkling-2-Mini (`C1-C4 <= 16K`)

**[HW-WITNESSED]** NVIDIA GeForce RTX 5080 16 GB (`sm_120`, `sm_120f` build). Machine-readable receipt: [`perf-receipt-v2.json`](perf-receipt-v2.json).

**Scope:** `Inkling-2-Mini-UD-Q3_K_XL.gguf`, `C1-C4` concurrency, `<= 16K` context (`f16` KV cache). Apple Silicon Metal `resident` and `> 16K` coding context (`fp8` KV auto-selection) are tracked separately and are not claimed by this receipt.

## 1. In-Process Decode & Speculative Oracle

| Witness | Config | Result |
|---|---|---|
| Plain decode (`TestInkling2CUDADecodeBench`) | `kv=f16`, auto ctx `29,184` | **122.34 tok/s** decode (127 steps), **602.04 tok/s** prefill (19 tok), `fnv64 d51b043b08677ea1` |
| Attend split (`TestCUDAInk2AttendSplit*`) | `sm_120` reference parity | **3/3 PASS** (`0.33s` / `0.58s` / `0.75s`) |
| Speculative oracle (`TestInkling2CUDASpecOracle`) | `kv=f32` (`deaebd6e43d4`) | **10/10 PASS** (`18.79s`); **294.28 tok/s** (depth 4), **326.21 tok/s** (depth 6), **307.74 tok/s** (depth 8) |
| Spec near-tie oracle (`TestInkling2CUDASpecNearTieOracle`) | `kv=f32`, depth 4 (`deaebd6e43d4`) | **4/4 PASS** (`12.75s`: `fibonacci` & `chat-template` 128/128 exact; `capitals` near-tie `0.0041 nats`; `passphrase-recall-1946` near-tie `0.0908 nats`) |
| `f32` batch/cohort parity (`TestInkling2CUDABatchDecodeMatchesOracle` / `CohortPrefillMatchesScalar`) | `kv=f32`, `C=4` / `C=8` (`deaebd6e43d4`) | **PASS** — `C=4` **224.95 tok/s** (`3.42s`), `C=8` **309.50 tok/s** (`7.33s`; 7/8 exact, row 3 near-tie `0.0211 nats`) |

## 2. 16K `f16` Batched Prefill KL Parity

| Test | Envelope | Result |
|---|---|---|
| `TestInkling2CUDABatchedPrefillKL` | `kv=f16`, 165 positions (5 prompts, up to `pp1946`) | **PASS** — mean KL **0.002912 nats**, max `0.083805`, same-top `164/165 (99.39%)` (`0.002389` mean KL with `AttendTree`) |
| `TestInkling2CUDAPrefillOracleForced` | `passphrase-recall-1946`, 32 forced steps | **PASS** — **32/32** batched top-1 match vs `llama.cpp` CPU oracle (`llama-gap 0.000`) |

## 3. Served HTTP Sweep (`C1-C4 <= 16K`) vs `llama.cpp`

`fak serve --engine inkernel --backend cuda` (`16K` ctx, `13,244 MiB` VRAM loaded) compared against `llama.cpp` stock (`0e4e925`, `-np 4 -c 16384 -fa on`) and the `realugbun/llama.cpp` fork (`-np 4 -c 16384 -fa on`) on the same RTX 5080:

| Cell | `fak` TTFT (ms) | `realugbun` TTFT (ms) | `llama-stock` TTFT (ms) | `fak` Agg E2E (tok/s) | `realugbun` Agg E2E (tok/s) | `llama-stock` Agg E2E (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| `C1_p512_d128` | 274.35 | 267.81 | 311.37 | 84.54 | 95.62 | 88.51 |
| `C1_p2048_d256` | 638.20 | 723.77 | 1,171.01 | 75.89 | 87.61 | 72.42 |
| `C2_p512_d128` | 351.33 | 530.63 | 559.93 | 121.16 | 139.58 | 135.42 |
| `C4_p512_d128` | **598.89** | 981.30 | 1,019.52 | 168.76 | 166.93 | 180.81 |
| `C4_p2048_d256` | **2,017.61** | 3,718.34 | 4,683.83 | **130.44** | 122.47 | 115.69 |

- **Where `fak` leads:** At `C4_p2048_d256`, `fak` cuts median TTFT by **1.84×** vs `realugbun` (`2,017.61` vs `3,718.34 ms`) and **2.32×** vs `llama-stock` (`4,683.83 ms`), while leading aggregate E2E throughput (`130.44` vs `122.47` and `115.69 tok/s`).
- **Where `fak` trails:** Single-stream served decode (`C1_p512_d128` `102.45` vs `118.63 tok/s`; `C1_p2048_d256` `93.23` vs `116.03 tok/s`) and `C2` aggregate E2E (`121.16` vs `139.58 tok/s`) still trail `realugbun`.

## 4. Non-Greedy Sampling & Greedy Fast-Path (`eb68a8cf8a`)

Live RTX 5080 (`sm_120f`) HTTP witness (`fak serve --engine inkernel --backend cuda`, `0` HTTP 500s across all 6 requests):

- **Seed determinism (`temp=0.7, top_p=0.9, seed=42` ×2):** HTTP 200, byte-identical (`sha256:ed3bb8d3f37c...`, 221 tokens).
- **Seed diversity (`seed=42` / `99` / `1234`):** 3 distinct outputs (`221` / `219` / `241` tokens).
- **Greedy fast-path (`temp=0`):** HTTP 200, **336.37 tok/s** stream decode (`109.05 tok/s` e2e including `1.774s` TTFT/setup, 283 tokens).

## 5. Honest Fences

- **16K `f16` KV ceiling on 16 GB GPUs:** In a 5-task `fak-deveval-outcome` smoke at `19,456` `f16` KV positions, 4/5 tasks passed their local checks before hitting `exceeds the 19456-token global cache` (`HTTP 500`); `> 16K` coding workloads require `fp8` KV auto-selection.
- **Speculative decode:** Off by default on `f16` KV; `passphrase-recall-1946` has one known near-tie divergence at step 25 (`0.3601 nats`).
- **Bare `fak serve` auto-fit:** An earlier `--fit auto` startup exit (`rc=1`) was fixed in `6e1a6b696d` and verified on bare `fak serve` (`HTTP 200`, `121.31 tok/s` greedy, `123.83 tok/s` sampled).
