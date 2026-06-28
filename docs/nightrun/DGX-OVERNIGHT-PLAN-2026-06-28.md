---
title: "DGX overnight run plan — 2026-06-28"
description: "Per-box overnight data-collection plan for the DGX/da33 fleet driven over the Slack control bridge: what each box collects tonight, the exact runbook, and the honesty boundary."
---

# DGX overnight run plan — 2026-06-28

The fleet-scale companion to [`README.md`](README.md) (`fak nightrun`, which is
*local*-box aware). `nightrun` answers "what can the box I'm sitting on collect?";
this doc answers the same question for the **remote DGX/da33 boxes reached only
through the Slack control bridge** (`fak-private/tools/dgxsh.py`) — the boxes
where the project's frontier GLM-5.2 witnesses actually live.

It is the durable plan: the next operator or agent reads this, confirms the
bridge is live, and resumes collection where this night left off. The witnessed
numbers land in [`collected.jsonl`](collected.jsonl) (durable trunk evidence) and
the per-box raw logs stay on the box under `/tmp/fakgpu/<tag>.log`.

## Live fleet state (witnessed 2026-06-28 ~05:48Z via the bridge)

| box | channel | hardware | GLM-5.2 state tonight | overnight target |
|---|---|---|---|---|
| **dgx3** | `dgx3-control` | 8×A100-SXM4-**80GB**, 886 GiB RAM | **2 fak-kernel serves UP** (`/tmp/fakdgx` :8000, `/tmp/fakdgx_q5q6` :8001) running ~1.5 d; GPU0 ~37 GB, **GPU1-7 fully free** | **live decode tok/s witness** against :8000 (read-only chat completion — the #971 cpu-offload wall) |
| **da33** | `da33-control` | CPU-only high-RAM host (256 CPU, ~1 TB RAM, **~454 GiB free**) | no serve up; **fak-bin 0.34.0 staged on NVMe** (`/mnt/nvme-glm/fak-bin`); full UD-Q4_K_M (434 GB) staged on NVMe | **llama.cpp mmap CPU throughput baseline** (memory-safe — fak-native resident needs ~458 GB > free, would OOM-wedge the shared host) |
| **dgx2** | `dgx2-control` | 8×A100-SXM4-**40GB** | GLM-5.2 serve + a peer agent mid-flight (RC2 poll loop) | leave to peer; revisit when its session drains |
| **dgx1** | `dgx1-control` | 8×A100-SXM4-**40GB** | OCCUPIED — sglang `gpt-oss-120b` TP-8, VRAM fully committed | **not a GLM target** (do not evict) |

The symbolic `*-control` keys resolve to real Slack channel IDs in `fak-private`'s
`tools/dgxsh.py` `_CHANNELS` map — the IDs never reach this public tree by policy
(see [`docs/fak/scrubbing-real-values.md`](../fak/scrubbing-real-values.md)).

## What each box collects tonight

### dgx3 — GLM-5.2 live GPU decode tok/s (`witness-glm52-cpu-throughput`, GPU arm)

The two serves are peer-owned and have been up ~1.5 days; **do not restart them**.
Measure a single-stream timed completion against the live :8000 endpoint — that is
the non-forgeable `gateway_inference_turn` witness (the #971 `--cpu-offload-experts`
wall, historically ~190 s for a 5-tok turn ≈ 0.026 tok/s wall). Keep `max_tokens`
small (≈16) so the turn returns inside one poll window.

```bash
# from fak-private, in a FRESH session (newsess) to avoid peer-stdin collision:
CHANNEL=dgx3-control python tools/dgxsh.py bg <sess> \
  <scratch>/dgx3_glm_decode_witness.sh glmdec1
CHANNEL=dgx3-control python tools/dgxsh.py poll <sess> glmdec1 40
```

### da33 — GLM-5.2 UD-Q4_K_M CPU throughput baseline (`witness-glm52-cpu-throughput`, CPU arm)

`llama-bench` mmaps the 434 GB GGUF from NVMe (only touched pages resident,
page-cache reclaimable) so it **cannot OOM-wedge** the shared host (it also runs
`cama-server` ~16 GB + a 45-day SWE-bench `bench` eval — neither may die). This is
the documented memory-safe baseline arm.

```bash
LB=/projects/llama.cpp/build-cpu/bin/llama-bench
GGUF=/mnt/nvme-glm/glm52-q4/UD-Q4_K_M/GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf
"$LB" -m "$GGUF" -ngl 0 -t 128 -p 512 -n 64 -r 2   # mmap default ON
```

**Do NOT run fak-native resident on da33 unattended** until free RAM clears
~1.1× model size (~480 GB) above the live `cama-server`/`bench` working sets — the
all-resident `AddResidentQ4K` path wedged a 1 TB host once (#974). The fix in
flight is the in-`fak serve` storage-tier + RAM-headroom preflight gate
(`refuseSlowLoadPath` / `EstimateF32LoadMemoryPlan`); until that gate ships, the
resident path is operator-gated on this shared box.

## How the loop closes

1. **Probe** — `dgxsh.py auth` (bridge live?), then per-box `nvidia-smi` / `free -g`
   / `ss -ltnp` to confirm free GPUs and RAM headroom.
2. **Launch** — one `bg <sess> <runner> <tag>` per box (fresh `newsess` per box;
   never share a session with a live peer agent).
3. **Watch** — poll `/tmp/fakgpu/<tag>.done` on a slow cadence (the bridge is
   rate-limited; `cmd_sync` rc is unreliable on busy boxes → read the transcript
   tail directly).
4. **Record** — fold each completed datum into [`collected.jsonl`](collected.jsonl)
   as one `fak-nightrun-collect/1` row (box, task_id, value=`frontier`, the exact
   command, `outcome`, the captured number, the artifact path). Commit by path.

## Honesty boundary

- A box that can't run a task safely is **never launched** for it — da33 fak-native
  resident is gated on RAM headroom, so the loop can never claim a resident-path
  number it didn't actually produce.
- The dgx3 number is a **timed live-serve completion** (`completion_tokens` over
  wall, prefill included) labelled WITNESSED on fak's own kernel; it is the wall
  rate including the #971 offload tax, not a synthetic kernel microbench.
- The da33 number is **llama.cpp** (the mmap baseline), labelled OBSERVED against a
  third-party engine — it is the baseline the fak-native arm is measured against,
  never reported as fak's own throughput.
