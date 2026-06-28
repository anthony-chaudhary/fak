---
title: "GPU server overnight run plan — 2026-06-28"
description: "Per-box overnight data-collection plan for the GPU server/da33 fleet driven over the Slack control bridge: what each box collects tonight, the exact runbook, and the honesty boundary."
---

# GPU server overnight run plan — 2026-06-28

The fleet-scale companion to [`README.md`](README.md) (`fak nightrun`, which is
*local*-box aware). `nightrun` answers "what can the box I'm sitting on collect?";
this doc answers the same question for the **remote GPU server/da33 boxes reached only
through the Slack control bridge** (`fak-private/tools/dgxsh.py`) — the boxes
where the project's frontier GLM-5.2 witnesses actually live.

It is the durable plan: the next operator or agent reads this, confirms the
bridge is live, and resumes collection where this night left off. The witnessed
numbers land in [`collected.jsonl`](collected.jsonl) (durable trunk evidence) and
the per-box raw logs stay on the box under `/tmp/fakgpu/<tag>.log`.

## Live fleet state (witnessed 2026-06-28 ~05:48Z via the bridge)

| box | channel | hardware | GLM-5.2 state tonight | overnight target |
|---|---|---|---|---|
| **GPU server** | `dgx3-control` | 8-GPU datacenter server, 886 GiB RAM | **2 fak-kernel serves UP** (`/tmp/fakdgx` :8000, `/tmp/fakdgx_q5q6` :8001) running ~1.5 d; GPU0 ~37 GB, **GPU1-7 fully free** | **live decode tok/s witness** against :8000 (read-only chat completion — the #971 cpu-offload wall) |
| **da33** | `da33-control` | CPU-only high-RAM host (256 CPU, ~1 TB RAM, **~454 GiB free**) | no serve up; **fak-bin 0.34.0 staged on NVMe** (`/mnt/nvme-glm/fak-bin`); full UD-Q4_K_M (434 GB) staged on NVMe | **llama.cpp mmap CPU throughput baseline** (memory-safe — fak-native resident needs ~458 GB > free, would OOM-wedge the shared host) |
| **GPU server** | `dgx2-control` | 8-GPU datacenter server | GLM-5.2 serve + a peer agent mid-flight (RC2 poll loop) | leave to peer; revisit when its session drains |
| **GPU server** | `dgx1-control` | 8-GPU datacenter server | OCCUPIED — sglang `gpt-oss-120b` TP-8, VRAM fully committed | **not a GLM target** (do not evict) |

The symbolic `*-control` keys resolve to real Slack channel IDs in `fak-private`'s
`tools/dgxsh.py` `_CHANNELS` map — the IDs never reach this public tree by policy
(see [`docs/fak/scrubbing-real-values.md`](../fak/scrubbing-real-values.md)).

## What each box collects tonight

### GPU server — GLM-5.2 live GPU decode tok/s (`witness-glm52-cpu-throughput`, GPU arm)

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
- The GPU server number is a **timed live-serve completion** (`completion_tokens` over
  wall, prefill included) labelled WITNESSED on fak's own kernel; it is the wall
  rate including the #971 offload tax, not a synthetic kernel microbench.
- The da33 number is **llama.cpp** (the mmap baseline), labelled OBSERVED against a
  third-party engine — it is the baseline the fak-native arm is measured against,
  never reported as fak's own throughput.

## Results (collected 2026-06-28, in [`collected.jsonl`](collected.jsonl))

All three GPU server numbers are WITNESSED on fak's own CUDA kernel — read-only timed
completions against the live `--cpu-offload-experts` :8000 serve (the peer serve was
never restarted).

| datum | number | what it isolates |
|---|---|---|
| live decode 16-tok (prefill-conflated) | **0.0694 / 0.0698 tok/s** (replicated) | 16 tok / ~230 s wall incl. a 39-tok prefill; stable across two idle runs |
| live decode 32-tok (amortization point) | **0.1074 tok/s** | 32 tok / 298 s — the rate rises with generation length as prefill amortizes |
| **steady-state TPOT** (prefill-isolated) | **0.2324 tok/s** (4.30 s/tok) | (t₃₆ − t₄)/(36 − 4); the 16→32→∞-tok curve (0.069 → 0.107 → 0.2324) cleanly traces this asymptote |
| q5q6-config serve (:8001) decode | **0.0601 tok/s** (~14 % slower than Q4_K :8000) | the heavier Q5_K/Q6_K expert dequant path costs throughput |
| 2-way concurrency | **0.0639 tok/s agg (0.27×)** | two 12-tok streams *serialized* (A 375 s ≈ 2× B 187 s) |
| 200-tok prefill | **> 900 s** (timed out) | prefill is host-bound too, not just decode |

**The finding worth keeping:** concurrency makes GLM-5.2 decode on this serve
*worse*, not better (0.27× of single-stream). The two streams contended instead of
batching — so the #971 wall is a **shared host-resource bottleneck** (the CPU
expert-GEMM under `--cpu-offload-experts`), **not a per-stream GPU limit**. The 7
idle datacenter GPU on GPU server cannot be put to work by batching as the serve is configured;
closing the wall means moving the expert GEMM off the host (resident experts, or a
GPU expert path), not adding concurrency. This is the concrete data behind the
"1/8 GPUs used is first-class" utilization thesis: the waste is host expert-offload,
not GPU count.

**da33 CPU baseline — COLLECTED (the headline comparison).** The first two launches
were squeezed out by a **peer's 446 GiB fak-native resident serve** (driving da33 to
`avail≈14 GiB`); a later tick caught the box freed (peer serve gone, `avail≈461 GiB`)
and the mmap llama-bench completed safely (rc=0, 969 s, `free_after=252 GiB`, no wedge):

| llama.cpp CPU (96-thread, mmap, NVMe) | GLM-5.2 UD-Q4_K_M (433.82 GiB / 753.86 B) |
|---|---|
| prefill (pp64) | **3.34 tok/s** |
| decode (tg16) | **0.89 tok/s** |

**The headline:** llama.cpp CPU decode (**0.89 tok/s**, OBSERVED) is **~3.8× faster**
than fak's GPU + `--cpu-offload-experts` steady-state (**0.2324 tok/s**, WITNESSED) on
the *same* model. fak's cpu-offload path is currently *slower than pure-CPU llama.cpp*
— which sharply quantifies the #971 optimization opportunity and reinforces the
finding above: the experts must move off the host (resident / GPU expert path), not
stay CPU-offloaded. The baseline is labelled OBSERVED (third-party engine), never
reported as fak's own throughput.

**da33 NVMe load rate — 1.549 GiB/s.** A cold sequential read of the 434 GB GGUF from
the NVMe mount took 280 s = **1.549 GiB/s** (≈4.7 min of pure I/O), vs NFS at
0.063 GB/s (≈115 min, the prior load wall) — **~25× faster**. This settles the
operator's "~10 min load" target *for the storage tier* (achieved). But the full
load+bench wall was 969 s, so **CPU-bound mixed-quant dequant adds ~11 min on top of
the ~5 min I/O** — NVMe staging fixes the secondary I/O waste; the residual gap to
10 min is dequant. Actionable for the #1062 preflight gate: classify the load-path
tier *and* surface the dequant cost separately.

## The night's conclusion

The accessible frontier dimensions are now **saturated** (decode amortization curve,
TPOT, concurrency, prefill, quant-config, CPU baseline, NVMe load — all collected and
in places replicated). Every measurement points one way: fak's `--cpu-offload-experts`
path is **host-expert-GEMM-bound** — slower than pure-CPU llama.cpp (~3.8×), it does
not batch (concurrency is 0.27×), it penalizes prefill *and* decode, and Q5_K/Q6_K
experts cost more than Q4_K. The mandate is unambiguous: **move the expert GEMM off the
host** (resident experts or a GPU expert path); that, not GPU count or batching, is
what closes the #971 wall. NVMe staging independently removes the I/O load tax (~25×),
leaving CPU dequant as the residual load cost.

The hourly overnight tick keeps the loop alive: it re-attempts da33 only when
`avail ≥ 440 GiB` with no peer resident serve, collects one read-only GPU-server decode
when the serve is idle (a 900 s timeout, never overlapping witnesses — the serve
degrades under contention), records `skipped`/`failed` whenever a box can't safely
produce a datum, and — now that the accessible dimensions are saturated — favours a
genuinely *new* condition (a freed GPU server, a changed serve config) over re-running a
settled measurement.
