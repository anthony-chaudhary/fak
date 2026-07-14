# Qwen3.6-27B Q4_K_M CPU decode — real-weights int8/f32 + NUMA worker witness (2026-07-13)

Campaign: epic #4623 (Qwen3.6-27B CPU decode → 10 tok/s). This is the first
**real-weights** witness set, taken on the lab **CPU server** (dual EPYC-7742,
2 socket × 64c/128t = **256 threads, 8 NUMA nodes**, ~1 TB RAM, AVX2, no GPU)
against the resident `Qwen3.6-27B-Q4_K_M.gguf` (16.5 GB). Tool: `cmd/q4kdiag
-decode N` (built on-box, go1.26); 40 greedy decode steps after 2 warmup, timed
per `Step()`. All numbers **OBSERVED**, not modelled.

## 1. Correctness — int8 Q4_K reducer vs exact f32 Q4_K (C1 #4624 blocker)

The 22-token "Say OK." oracle prefill, first-token top-8 (oracle argmax = 248068
`<think>` @ ~28.3). Run under `FAK_KQ_INT8=0` (exact f32 Q4_K) and `=1` (int8
SDOT), default placement:

| rank | int8=0 (f32 Q4_K) id/logit | int8=1 (int8 Q4_K) id/logit |
|---|---|---|
| 1 | **248068** / 28.2392 | **248068** / 28.2415 |
| 2 | 3793 / 23.3626 | 3793 / 23.2394 |
| 3 | 31248 / 17.9527 | 31248 / 17.8851 |
| 4 | 11245 / 16.7981 | 11245 / 16.7408 |
| 5 | 547 / 14.7255 | 547 / 14.7525 |
| 6 | 248069 / 14.4601 | 248069 / 14.4229 |
| 7 | 10092 / 14.2613 | 10092 / 14.2311 |
| 8 | 220 / 13.9766 | 220 / 13.9140 |

**The int8 Q4_K path reproduces the f32 argmax AND the full top-8 rank order**
on real weights; logits agree to ~0.01–0.12. The #4624 first-token-agreement
blocker is witnessed. (Every decode cell below also holds first_token_id=248068,
so the argmax is stable across placement/worker changes too.)

## 2. Decode tok/s — placement × int8 × worker sweep (C1/C2 #4625)

`numactl <placement> env FAK_Q4K=1 FAK_KQ_INT8=<i> FAK_WORKERS=<w> q4kdiag -gguf … -decode 40`:

| placement | int8 | FAK_WORKERS | decode tok/s | first-tok | vs default |
|---|---|---|---|---|---|
| default | 1 | (256) | 0.572 | 248068 | 1.00× |
| interleave=all | 1 | (256) | 0.628 | 248068 | 1.10× |
| interleave=all | 1 | 16 | 0.571 | 248068 | 1.00× |
| interleave=all | 1 | 32 | 0.983 | 248068 | 1.72× |
| **interleave=all** | **1** | **64** | **1.418** | 248068 | **2.48×** |
| interleave=all | 1 | 128 | 1.333 | 248068 | 2.33× |
| cpunodebind0+membind0 | 1 | 32 | 0.832 | 248068 | 1.45× |
| interleave=all | 0 (f32) | 32 | 1.214 | 248068 | 2.12× |

## 3. Findings

1. **The default all-core (256-worker) decode is in the barrier-collapse regime.**
   The worker sweep peaks at **64 workers (1.418 tok/s)** and regresses by 256
   (0.628) — a **2.48× no-code win from `FAK_WORKERS=64 numactl --interleave=all`
   alone**. This directly confirms #4625: the resident Q4_K decode path runs
   uncapped at GOMAXPROCS=256 and the shared-cursor `parFor` barrier dominates
   at high worker counts. The knee (~64) is far below 256.
2. **The exact f32 Q4_K path is competitive with — here faster than — int8** at the
   same 32 workers (1.214 vs 0.983). At these worker counts decode is
   bandwidth/barrier-bound, not dequant-bound, so the ~8× *isolated-kernel* int8
   advantage does not translate end-to-end; the bit-exact f32 path is a viable
   (and quality-free) default. Worth an A/B at the w64 knee (f32 not yet measured
   there) before choosing the C1 default.
3. **Single-node (node0-local, 32 cores) < interleaved (0.832 vs 0.983)** — spreading
   weight pages across nodes beats confining to one node's ~2 channels, confirming
   aggregate NUMA bandwidth is the lever. But interleave still pays ~7/8 remote
   accesses; **per-node weight *replication* (C2 #4625) — full node-local copies —
   is the remaining lever toward the roofline.**
4. Gap to target: best witnessed **1.418 tok/s** vs the 10 tok/s goal. Env tuning
   delivered 2.48×; the rest needs the C2 replication + barrier-free schedule (and
   the roofline in §4 says how much headroom the box actually has).

## 4. Memory-bandwidth roofline (C3 #4626)

STREAM-Triad (24 B/elem, 1 GiB float64 arrays), on-box go, all-core:

| placement | all-core GB/s | note |
|---|---|---|
| default (first-touch) | **17.7** | node-0 confinement — the same pathology as default decode |
| `interleave=all` | **84.8** | pages striped over 8 nodes; 4.8× the default |
| single node (node0, 32c) | **26.9** | one NUMA node's ~2 channels |
| **implied 8-node-local ceiling** | **~215** | 8 × 26.9 — what per-node *replication* (C2) unlocks |

**Roofline read (bytes/token = 27e9 × 0.5625 = 15.2 GB/tok):**
- Best witnessed decode 1.418 tok/s ⇒ **21.5 GB/s effective** weight stream.
- That is **25% of the interleave wall (84.8)** and **10% of the replicated ceiling (~215)**
  → decode is **barrier-bound, not bandwidth-bound**; the C2 barrier-free schedule has real headroom.
- 10 tok/s needs 152 GB/s effective = **~71% of the ~215 GB/s replicated ceiling**.
  **So 10 tok/s is within this box's physical roofline** — but only via per-node weight
  **replication** (C2 #4625): `interleave` tops out at 84.8 GB/s (≈5.6 tok/s ceiling) because
  7/8 of accesses are remote; only full node-local replicas reach ~215 GB/s.
- The default-placement STREAM (17.7) mirrors the default-decode collapse (0.572): both are
  strangled by node-0 first-touch. Fixing placement is the whole game.

## 5. Provenance / reproduce

- Box: CPU server, `nproc=256`, 8 NUMA nodes (~128 GiB each), go1.26.1 at build.
- Model: `Qwen3.6-27B-Q4_K_M.gguf` (16,547,398,784 bytes) resident on NVMe.
- Tool: `cmd/q4kdiag -decode` (this repo, commit adding the flag).
- Turnkey re-run: `experiments/qwen36/witness-cpu-decode.sh <gguf>` reproduces §2–§4.
- All runs single-stream, greedy (argmax), oracle "Say OK." prefill.
