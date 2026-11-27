---
title: "Campaign record: how the realugbun/llama.cpp Inkling 2 speedup divides among its changes, RTX 5080 host RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak rtx5080-host-inkling2-realugbun-fusion-ablation (2026-11-22): how the realugbun/llama.cpp Inkling 2 speedup divides among its changes, RTX 5080 host RTX 5080."
---

# RESULTS — how the realugbun/llama.cpp Inkling 2 speedup divides among its changes, RTX 5080 host RTX 5080 (2026-11-22)

**[HW-WITNESSED]** upstream fork only. No fak engine ran and this is not a fak performance claim.
Study: [STUDY-realugbun-llama-cpp-2026-11-22](../../research/STUDY-realugbun-llama-cpp-2026-11-22.md), section "Addendum: measured split of the gain".
Parent receipt (the compound A/B this divides): inkling2-realugbun-fork-ab (RTX 5080 host).

## Verdict

The fused kernels carry 88% of the decode gain and 95% to 97% of the prefill gain. The two graph-side commits carry 10% to 11% of decode and 3% to 5% of prefill. The narrow-row pad change carries about 1% of decode and nothing measurable on prefill.

Decode is spread across four kernel groups of similar size: MIX, CLIP and the CTRL family each save 0.82 to 0.93 ms per token across both depths and both readings (0.84 to 0.90 ms for leave-one-out at depth 0), the CONV family 0.55 to 0.60 ms. Prefill is concentrated in two: MIX and RELBIAS together are 95% to 97% of the fusion saving, and RELBIAS alone is two thirds of it at depth 8,192. SITU is the smallest group on decode; on prefill CONV is the smallest and CLIP and SITU are both under 4 us per token.

The group contributions add up. At both depths, leave-one-out contributions sum to 99.6% to 103.4% of the all-fusions saving and only-one-on contributions to 98.9% to 99.7%. No interaction between groups is resolved; the 2.3% to 3.4% leave-one-out excess at depth 0 comes from noisy runs in window 2 (see "Additivity").

All eleven fusion types fire on the shipped model: eight on a batch-1 decode step, and `CONV`, `CTRL_NORM` and `CTRL_TAIL` on multi-token batches in place of `CONV_STATE`/`CONV_DYN`, `CTRL_DOWN` and `CTRL_OUT`.

## Decode: where the 3.97 ms per token goes (depth 0)

Per-token time is the additive unit; tokens per second are shown beside it because they are what the parent receipt reports.

| step | tok/s | ms/token | saved vs previous | share of total |
|---|---:|---:|---:|---:|
| base `e50c254a` | 80.98 | 12.349 | | |
| fork, fusions off, old pad path (graph-side commits only) | 83.90 | 11.920 | 0.429 | 10.8% |
| fork, fusions off (adds the pad change) | 84.28 | 11.866 | 0.054 | 1.4% |
| fork, all fusions on | 119.32 | 8.381 | 3.485 | 87.8% |
| total | | | 3.968 | 100% |

Fusion groups, depth 0. "Leave-one-out" is the time added when only that group is switched off; "only-one-on" is the time removed when only that group is switched on.

| group (fusion types) | leave-one-out ms | share of fusions | tok/s without it | only-one-on ms | share | tok/s with only it |
|---|---:|---:|---:|---:|---:|---:|
| MIX | 0.899 | 25.8% | 107.76 | 0.913 | 26.2% | 91.30 |
| CLIP | 0.842 | 24.2% | 108.43 | 0.836 | 24.0% | 90.66 |
| CTRL (`CTRL_DOWN`, `CTRL_OUT`, `CTRL_NORM`, `CTRL_TAIL`) | 0.838 | 24.0% | 108.48 | 0.819 | 23.5% | 90.52 |
| CONV (`CONV`, `CONV_STATE`, `CONV_DYN`) | 0.603 | 17.3% | 111.31 | 0.550 | 15.8% | 88.37 |
| RELBIAS | 0.283 | 8.1% | 115.42 | 0.272 | 7.8% | 86.25 |
| SITU | 0.100 | 2.9% | 117.92 | 0.058 | 1.7% | 84.69 |
| sum | 3.564 | 102.3% | | 3.447 | 98.9% | |

Depth 8,192 decode: base 76.86 tok/s (13.011 ms), graph-side only 79.45 (12.587), fusions off 79.75 (12.539), all on 112.50 (8.889). Total saved 4.122 ms: graph-side 0.424 (10.3%), pad 0.048 (1.2%), fusions 3.650 (88.6%).

| group | leave-one-out ms | share of fusions | tok/s without it | only-one-on ms | share |
|---|---:|---:|---:|---:|---:|
| MIX | 0.897 | 24.6% | 102.19 | 0.925 | 25.3% |
| CLIP | 0.832 | 22.8% | 102.88 | 0.838 | 23.0% |
| CTRL | 0.827 | 22.7% | 102.93 | 0.824 | 22.6% |
| CONV | 0.561 | 15.4% | 105.83 | 0.581 | 15.9% |
| RELBIAS | 0.455 | 12.5% | 107.02 | 0.455 | 12.5% |
| SITU | 0.090 | 2.5% | 111.38 | 0.001 | 0.0% |
| sum | 3.660 | 100.3% | | 3.624 | 99.3% |

Only RELBIAS grows with depth on decode (0.28 to 0.45 ms). The other groups are flat within 0.05 ms on leave-one-out (SITU's only-one-on reading moves 0.057 ms), which fits a per-launch cost that does not depend on context length. No profile was taken to confirm that.

## Prefill: pp512, microseconds per token

| step | depth 0 tok/s | us/token | saved | share | depth 8,192 tok/s | us/token | saved | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 2355.4 | 424.6 | | | 1334.2 | 749.5 | | |
| graph-side commits only | 2416.8 | 413.8 | 10.8 | 4.7% | 1364.7 | 732.8 | 16.7 | 3.3% |
| adds the pad change | 2420.8 | 413.1 | 0.7 | 0.3% | 1364.4 | 733.0 | -0.2 | 0.0% |
| all fusions on | 5106.8 | 195.8 | 217.3 | 95.0% | 4012.4 | 249.2 | 483.7 | 96.7% |
| total | | | 228.7 | | | | 500.3 | |

| group | d0 leave-one-out us | share | d0 tok/s without it | d0 only-one-on us | d8192 leave-one-out us | share | d8192 tok/s without it | d8192 only-one-on us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MIX | 151.9 | 69.9% | 2875.8 | 147.9 | 148.1 | 30.6% | 2516.7 | 148.3 |
| RELBIAS | 55.7 | 25.6% | 3976.0 | 55.3 | 322.0 | 66.6% | 1750.5 | 322.2 |
| CTRL | 9.9 | 4.5% | 4862.0 | 8.1 | 8.4 | 1.7% | 3881.9 | 7.7 |
| SITU | 3.7 | 1.7% | 5012.7 | 3.1 | 2.4 | 0.5% | 3974.3 | 3.2 |
| CLIP | 3.2 | 1.5% | 5025.1 | 2.7 | 2.8 | 0.6% | 3967.1 | 2.7 |
| CONV | 0.4 | 0.2% | 5096.0 | -2.0 | -1.8 | -0.4% | 4041.1 | -1.8 |
| sum | 224.7 | 103.4% | | 215.2 | 482.0 | 99.6% | | 482.4 |

The pad change and the CONV group are inside the noise on prefill (standard error of a six-sample mean: 0.3 to 1.4 us at depth 0 apart from `loo-MIX` at 3.5 us, and 3.0 to 3.8 us at depth 8,192). CLIP, SITU and CTRL are resolved at depth 0 and are small.

## Which fusions fire, and what each dispatch saves

Counted with CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`), so the fused dispatch runs on every evaluation. Decode figures are per token from 64 batch-1 steps at depth 0; prefill figures are per 512-token batch. Each figure is fused dispatches, with graph nodes replaced in brackets.

| type | per decode token | per 512-token prefill batch |
|---|---:|---:|
| CLIP | 144 (1,248) | 144 (1,248) |
| CONV | 0 | 176 (792) |
| CONV_STATE | 176 (1,496) | 0 |
| CONV_DYN | 16 (656) | 0 |
| SITU | 10 (140) | 10 (140) |
| MIX | 120 (960) | 120 (960) |
| CTRL_DOWN | 96 (384) | 0 |
| CTRL_OUT | 96 (768) | 0 |
| CTRL_NORM | 0 | 96 (288) |
| CTRL_TAIL | 0 | 96 (576) |
| RELBIAS | 48 (480) | 48 (480) |
| total | 706 (6,132) | 690 (4,484) |

On a decode step 706 fused launches stand in for 6,132 graph nodes. Dividing each group's depth-0 leave-one-out saving by its counts:

| group | dispatches per token | nodes replaced | ms saved | us per dispatch | us per node removed |
|---|---:|---:|---:|---:|---:|
| MIX | 120 | 960 | 0.899 | 7.5 | 1.07 |
| CLIP | 144 | 1,248 | 0.842 | 5.8 | 0.76 |
| CTRL | 192 | 1,152 | 0.838 | 4.4 | 0.87 |
| CONV | 192 | 2,152 | 0.603 | 3.1 | 0.31 |
| RELBIAS | 48 | 480 | 0.283 | 5.9 | 0.66 |
| SITU | 10 | 140 | 0.100 | 10.0 | 0.77 |

"Nodes removed" is nodes replaced minus dispatches. Many replaced nodes are views and reshapes that launch nothing, so the last column is a lower bound on the saving per removed launch; the CONV family replaces the most nodes per unit of time saved.

Prefill is different in kind. A 512-token batch runs 690 fused dispatches and saves 111 ms at depth 0, so the saving is work, not launch overhead: MIX saves 0.63 to 0.65 ms per dispatch at both depths, and RELBIAS saves 0.59 ms per dispatch at depth 0 and 3.4 ms at depth 8,192, where the unfused chain materializes a tensor that grows with the context.

## Additivity

- Leave-one-out sums: 102.3% and 100.3% of the all-fusions saving on decode (depth 0 and 8,192), 103.4% and 99.6% on prefill.
- Only-one-on sums: 98.9% and 99.3% on decode, 99.1% and 99.7% on prefill.
- Window 1 alone, decode depth 0: leave-one-out 100.0%, only-one-on 99.8%.
- Window 1 alone, prefill depth 0: leave-one-out 100.6%.
- The decode depth-0 excess comes from window 2: `rev-loo-CONV` read 110.82 tok/s against 111.80 in window 1, and `rev-loo-SITU` had one repetition at 115.73 against five at 118.06 to 118.75. `only-SITU` did not move between windows; `only-CONV` also slowed in window 2 (88.61 to 88.14 tok/s, about 0.06 ms per token), so the CONV figure is the least certain of the six.
- The prefill depth-0 excess also comes from window 2: `rev-loo-MIX` had one repetition at 2,733 tok/s against about 2,900, and `rev-loo-CONV` and `rev-loo-CTRL` each had a slow repetition.
- So the decode saving is a sum of independent per-group terms to within about 0.08 ms per token (2.3% of 3.485 ms). SITU's decode contribution is 0.06 to 0.10 ms depending on which arm is read, and CONV's is 0.55 to 0.60 ms.
- The step table sums to the total by construction (each row is a difference of adjacent arms).

## Drift anchors

Decode tok/s at depth 0, one value per run of three repetitions:

| arm | window 1 start | window 1 end | window 2 start | window 2 end |
|---|---:|---:|---:|---:|
| base | 78.21 | 80.95 | 81.06 | 80.92 |
| fork, patched build, all on | 119.36 | 119.34 | 119.34 | 119.27 |
| fork, original build | 119.33 (run 3 of 20) | | | 119.23 (run 18 of 20) |

- The fork arms did not drift: four runs within 0.08%.
- The patched build matches the original fork build (119.32 against 119.28 pooled), so the switch and the counters cost nothing measurable.
- Both match the parent receipt (80.98 and 119.28).
- The first timed base run of window 1 (`fwd-base-1`) is 3.4% slow on decode and about 4% slow on prefill at depth 0. It followed one discarded warm-up run. The other three base runs agree within 0.2% and equal the parent receipt, so `fwd-base-1` is excluded from the base figure. The cause was not identified. With it pooled, base is 80.29 tok/s and the graph-side term becomes 0.536 ms instead of 0.429 ms; no fusion figure depends on the base arm.

## Envelope

- Node: rtx5080-host, RTX 5080 16 GB, sm_120, driver 615.78.08, kernel 7.2.11-arch1-1, CUDA 13.4 nvcc, GCC 16.2.1, CMake 4.4.5.
- Lease: `hwl-37937785843f1b39`, class `owned-rtx5080`, kind measure, holder `claude-realugbun-ablation-20261122`, granted 2026-11-23T03:36:45Z, released 03:52:31Z. Two windows ran under it: 03:37:54Z to 03:43:50Z and 03:45:57Z to 03:51:33Z.
- Base: `ggml-org/llama.cpp` `e50c254a22c590c9d2a60db86690d48d42497e2b`, existing build, untouched.
- Fork, original: `realugbun/llama.cpp` `2eb2530ec4da95453e8ea7d68f937da174dfe799`, existing build, untouched.
- Fork, patched: a scratch copy of the fork tree at `~/oss-engines/llama.cpp-realugbun-2eb2530e-ablate-src` on RTX 5080 host, same commit plus the switch below, built with the same flags. It stays on the node.
- Model: Unsloth `Inkling-2-Mini-UD-Q3_K_XL`, `Mainline_Support` layout, sha256 `e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639`, all layers on the GPU.
- Serving: `fak-llama-qwen27b.service` was `active` before each window, stopped by the window script and restarted by it. It was `active` with `/health` ok and 15,110 MiB loaded before the lease was released. A peer lease (`hwl-55bd485da2a77b73`) was granted 32 seconds after the release and stopped the unit again for its own window.

## The switch

Three environment variables, read once, added to the scratch copy only (`receipts/harness.txt` holds the full diff, sha256 `ae9a8c6bd3f6c6261dc005cc26f665b235549059013e99f284ae6df280603e60`):

- `GGML_CUDA_INKLING2_SKIP=<comma list of fusion names, or ALL>`: `ggml_cuda_inkling2_match` treats a match of a listed type as no match. The next Inkling 2 matcher for the same first op then gets a turn, and failing that the nodes take upstream's ordinary path, including its generic fusions. An unknown name aborts.
- `GGML_CUDA_INKLING2_STATS=1`: prints fused dispatches, nodes replaced and turned-away matches per type at exit.
- `GGML_CUDA_PAD_NO_FLAT=1`: `pad_f32_cuda` takes the pre-`0f6ac0c9c` path.

Checks that the switch does what it says:

- `test-backend-ops test -o INKLING2_CLIP,INKLING2_CONV,INKLING2_SITU,INKLING2_MIX,INKLING2_CTRL,INKLING2_RELBIAS,INKLING2_CONV_STATE,INKLING2_CONV_DYN`: 82 of 82 pass with everything on (all eleven types fired during the run) and 82 of 82 with `SKIP=ALL` and the old pad path.
- Every timed patched-build run carries its counters (the two original-fork runs have none). A switched-off type shows zero dispatches in every run, and the types left on show the same dispatch counts as the all-on run (for example SITU stays at 340 when CLIP, which shares its first op, is off).
- All-on on the patched build equals the original fork build.
- Fusions off lands at 84.28 tok/s, 4.1% above base, which is the graph-side and pad contribution and nothing else.

## Commands

Every timed run, with the binary directory and environment varied per arm:

```
env GGML_CUDA_INKLING2_STATS=1 [GGML_CUDA_INKLING2_SKIP=<list>] [GGML_CUDA_PAD_NO_FLAT=1] \
  <bindir>/llama-bench -m <gguf> -ngl 99 -fa 1 -p 512 -n 128 -d 0,8192 -r 3 -o json
```

Count runs: `GGML_CUDA_DISABLE_GRAPHS=1 ... llama-bench -m <gguf> -ngl 99 -fa 1 -p 0 -n 64 -d 0 -r 1 --no-warmup -o json` for decode and `-p 512 -n 0` for prefill.

Windows: `OUT=~/inkling-bench/ablate-20261122-w1 ORDER=fwd EXTRAS=1 bash ~/inkling-bench/awindow.sh`, then `OUT=~/inkling-bench/ablate-20261122-w2 ORDER=rev bash ~/inkling-bench/awindow.sh`. Each window is one discarded base warm-up, base and all-on anchors, the 16 ablation runs (original fork, fusions off, fusions off with old pad, all on with old pad, six leave-one-out, six only-one-on), then the anchors again. Window 2 runs the 16 in reverse order. Between runs the script waits for 0 compute processes and under 1,000 MiB of GPU memory. The exact argv of each run is in `receipts/arms.jsonl`; for the `count-nograph-*` runs the script exports `GGML_CUDA_DISABLE_GRAPHS=1` beforehand, so it is in `awindow.sh` and not in their recorded argv.

Each arm is 6 samples (2 runs of 3 repetitions); the all-on arm is 12 and the base arm 9.

## Limits

- One model file, one quantization, one node, two depths. Depth 32,768 was not run.
- Groups, not types: the CONV and CTRL families were switched as wholes. Within each family the decode and prefill members are different types, so the decode figure belongs to `CONV_STATE` + `CONV_DYN` and `CTRL_DOWN` + `CTRL_OUT`, and the prefill figure to `CONV` and `CTRL_NORM` + `CTRL_TAIL`.
- The two graph-side commits (`7ec7714fa` normalized depth states built once, `2673946` chains kept together) were not separated. They are source changes in the model graph builder and would need a third build.
- "Fusions off" runs the fork's reordered graph without fusing it. It is not upstream's graph.
- Decode ran with CUDA graphs on, the default. The decode figures are what fusion saves under graph replay in llama.cpp. They rank where an unfused composition costs most; they do not predict what a native engine that never builds the unfused form would gain.
- Prefill at depth 8,192 is noisy (1% to 3.5% per sample), so effects under about 8 us per token are unresolved there.
- No per-arm quality check beyond the backend test. The parent receipt holds the KLD comparison for the all-on build.
- The counters in the patched build are atomic increments on the host dispatch path. They are present in every patched arm, including all-on.

## Receipts

Under [`receipts/`](receipts/): one row per `llama-bench` result with its argv, keyed by window and run, in [`arms.jsonl`](receipts/arms.jsonl); window logs, per-run fusion counters, backend-test tails and the two summary tables in [`windows.txt`](receipts/windows.txt); the window script, the patch script, the resulting diff and the summary scripts in [`harness.txt`](receipts/harness.txt); binary and model digests, the scratch-copy state, the lease record and the build command in [`digests.txt`](receipts/digests.txt).
