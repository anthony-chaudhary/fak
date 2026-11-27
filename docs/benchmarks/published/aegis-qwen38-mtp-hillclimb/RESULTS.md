---
title: "Campaign record: RTX 5080 host RTX 5080 Qwen3.8-27B: latest llama.cpp + MTP hill climb"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-qwen38-mtp-hillclimb (2026-11-20): RTX 5080 host RTX 5080 Qwen3.8-27B: latest llama.cpp + MTP hill climb."
---

# RESULTS — RTX 5080 host RTX 5080 Qwen3.8-27B: latest llama.cpp + MTP hill climb (2026-11-20)

**[HW-WITNESSED]** rtx5080-host (RTX 5080 16 GB, sm_120, driver 615.78.08), measure lease
`hwl-ec045f3e07e983b1` (released). Five short exclusive windows (23:35Z–00:04Z); the serving
unit `fak-llama-qwen27b.service` was stopped per window and restored by an EXIT trap.
Machine-readable: [`perf-receipt-v2.json`](perf-receipt-v2.json), per-request rows
[`receipts/rows.jsonl`](receipts/rows.jsonl), layer placement [`receipts/load-all.txt`](receipts/load-all.txt), harness `receipts/qbench.sh` + `receipts/qwindows.sh.txt`.

## Verdict

Built-in MTP self-speculation (`--spec-type draft-mtp --spec-draft-n-max 2`) on llama.cpp
29b52d39 decodes at **104.6 tok/s vs 68.7 tok/s** for the served b11870 config (**+52%**, C1,
temp 0, 61% draft acceptance). The catch is memory: the speedup exists only when every layer stays on
the GPU, which caps context at **192k** (196608) on 16 GB. The router pin and `opencode.json` declare
262144, so the config is **staged, not promoted**.

| arm | engine | ctx | spec | layers on GPU | decode tok/s |
| --- | --- | --- | --- | --- | --- |
| A1/A2 (served) | b11870 | 262k | none | 66/66 | 68.66 / 68.66 |
| B1/B2 | 29b52d39 | 262k | none | 66/66 | 68.86 / 68.94 |
| H1/H2 | 29b52d39 | 32k | none | 66/66 | 68.92 / 68.86 |
| G1/G2 | 29b52d39 | 32k | MTP n=2 | 66/66 | 102.46 / 102.49 |
| N1 | 29b52d39 | 32k | MTP n=3 | 66/66 | 101.51 (acc 51%) |
| **L1** | 29b52d39 | **192k** | MTP n=2 | 66/66 | **104.56** |
| P3 | 29b52d39 | 224k | MTP n=2 | 62/66 | 53.94 |
| C/I | 29b52d39 | 262k | MTP n=2 | spilled | 31.8–32.0 |
| E1 | b11870 | 262k | MTP n=2 | spilled | 26.90 |

## What was learned

1. **The latest engine alone is flat** (+0.3%, within spread). The 466 upstream commits since b11870 bring
   no decode gain for this dense-hybrid Q2 model on sm_120.
2. **The MTP head is the lever** (the GGUF ships `blk.64.nextn.*`, `nextn_predict_layers=1`). This is consistent
   with PHL-008: n=2 beats n=3.
3. **MTP at 262k does not fit on 16 GB in any variant tried.** The draft context needs a 1 GiB f16 KV plus a larger target
   compute buffer (1.41 GiB). `-fit` then spills 8–12 layers to CPU and decode *halves* (27–39 tok/s).
   q4_0 draft KV cuts KV to 288 MiB but inflates the draft compute buffer to 1029 MiB, so it is a net loss;
   `-ub 64`, `--spec-draft-n-ubatch 8` and `-fitt 256` don't close the gap either (shortfall ≥ 1.0 GB).
4. Greedy MTP output diverges from non-MTP output after 211–430 chars at near-tie tokens (batch-verify
   numerics). Both are coherent. No eval-suite quality run was done.

## Promotion (open)

Staged unit: `rtx5080-host:/etc/fak/fak-llama-qwen27b.service.mtp192k-candidate` (29b52d39 build,
`-c 196608 ... --spec-type draft-mtp --spec-draft-n-max 2`). Applying it requires lowering the
advertised context 262144→196608 in one change, in three places: the router pin (`backends[rtx5080-host].engine.slot_context_tokens`,
`routes[*].context_window`; the pin needs elevation), `opencode.json` `limit.context` for
`rtx5080-host/qwen3.8-27b` + `qwen/qwen3.8-27b`, and the operator overlay. Otherwise requests between 192k and 262k
fail at the server instead of compacting.

## Context: r/LocalLLM "RTX 5080 PC vs GB10 workstation" (operator-supplied)

The post's 5080 ran Qwen3.8-Flash-Next IQ3_S (83.6 GB) through Strata with MTP4 and a hybrid GPU/RAM
placement, reporting 40–67 tok/s decode. RTX 5080 host holds the Strata source and a 63 GB Flash-Next Q2_0, but has only
30 GB RAM + 16 GB VRAM, so that configuration would page from NVMe. It is not reproducible here without more RAM.
The post's dense 8B row (130 tok/s at 512 input tokens) agrees with the bandwidth-bound picture.

## Lesson (pending append)

`fak-sync lessons add` refuses right now because the store reports UNKNOWN: there are 7 `ID_DUPLICATE` rows in
`docs/lessons/perf-hillclimb/k31-climb.jsonl` (K31C-00..09 twins, union-merge resurrection). The validated row is in
[`lesson-row.pending.json`](lesson-row.pending.json). Append it with
`fak-run fak-sync lessons add --store perf --as-of 2026-11-20 --json "$(cat lesson-row.pending.json)"` once the k31 store is de-duplicated.
