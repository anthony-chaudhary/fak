---
title: "Campaign record: realugbun/llama.cpp fork vs its upstream base, Inkling-2-Mini on Aegis RTX 5080"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak aegis-inkling2-realugbun-fork-ab (2026-11-22): realugbun/llama.cpp fork vs its upstream base, Inkling-2-Mini on Aegis RTX 5080."
---

# RESULTS — realugbun/llama.cpp fork vs its upstream base, Inkling-2-Mini on Aegis RTX 5080 (2026-11-22)

**[HW-WITNESSED]** upstream reproduction only. No fak engine ran and this is not a fak performance claim.
Study: [STUDY-realugbun-llama-cpp-2026-11-22](../../research/STUDY-realugbun-llama-cpp-2026-11-22.md).
Prior baseline on the same node and file: [aegis-inkling2-mini-oss-baseline-20261122](../aegis-inkling2-mini-oss-baseline-20261122/RESULTS.md).

## Verdict

The fork's speedup reproduces on the RTX 5080. The decode gain is larger than its README reports for the RTX 5060 Ti at every depth tested, and the prefill gain is larger at depths 0 and 8,192. Decode rises 45% to 47% at every depth tested and prefill rises 2.2x to 3.0x. The fastest OSS decode measured on this card for Inkling-2-Mini is now **119.3 tok/s**, up from 81.0.

| context depth | physical batch | pp512 base | pp512 fork | ratio | tg128 base | tg128 fork | gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 2360.7 | 5106.2 | 2.16x | 80.98 | 119.28 | +47.3% |
| 8,192 | 512 | 1346.5 | 4027.8 | 2.99x | 76.88 | 112.50 | +46.3% |
| 32,768 | 128 | 693.2 | 1949.6 | 2.81x | 69.97 | 101.58 | +45.2% |

Values are tokens per second, the mean of the two ABBA runs per arm. Each run is 3 repetitions at depths 0 and 8,192 and 2 repetitions at 32,768. The two runs of each arm agree within 0.1% on decode. The base arm (`e50c254a`) matches the earlier baseline on this node, which used llama.cpp `ad50351f5` (80.99 decode, 2362 pp512).

## Envelope

- Node: aegis-fak, RTX 5080 16 GB, sm_120, driver 615.78.08, kernel 7.2.11-arch1-1, CUDA 13.4 nvcc, GCC 16.2.1, CMake 4.4.5.
- Lease: `hwl-ab8fb540831414e9`, class `owned-rtx5080-aegis`, kind measure, holder `claude-study-realugbun-20261122`, granted 18:41:05Z, released 18:54:36Z (transcribed from `fak-sync nvlease history` into `receipts/digests.txt`). Three windows ran under it, about 4 minutes of GPU time in total.
- Arm A (base): `ggml-org/llama.cpp` `e50c254a22c590c9d2a60db86690d48d42497e2b`, the fork point.
- Arm B (fork): `realugbun/llama.cpp` `2eb2530ec4da95453e8ea7d68f937da174dfe799`.
- Build, both arms, as issued: `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DLLAMA_CURL=OFF`. The build log confirms Release and `120a-real`; it does not echo the command line. Binary digests are in `receipts/digests.txt`.
- Model: Unsloth `Inkling-2-Mini-UD-Q3_K_XL` in its `Mainline_Support` layout, sha256 `e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639`, all layers on the GPU.
- Command: `llama-bench -m <gguf> -ngl 99 -fa 1 -p 512 -n 128 -d <depth> -r <reps> -o json`, with `-b 512 -ub 128` added at depth 32,768.
- Order: A B B A in each window. One discarded base warm-up precedes the depth 0 and 8,192 window; the 32,768 window had none. Between runs the script waited for 0 compute processes and under 1,000 MiB of GPU memory.
- Serving: `fak-llama-qwen27b.service` was stopped for each window and restarted by the window script. The unit was `active` at the end of every window. A manual `/health` probe after the last window returned `{"status":"ok"}`; it is transcribed in `receipts/digests.txt`, not captured by the window script.

## Correctness and quality

- `test-backend-ops test -o INKLING2_CLIP,INKLING2_CONV,INKLING2_SITU,INKLING2_MIX,INKLING2_CTRL,INKLING2_RELBIAS,INKLING2_CONV_STATE,INKLING2_CONV_DYN` on the fork build, which is all eight chain names the fork defines: 82 of 82 cases pass on CUDA0 against the CPU backend.
- Perplexity KLD with `llama-perplexity -c 512 --chunks 8 -b 512`, base at physical batch 512 as the reference:

| arm | physical batch | mean KLD | max KLD | same top token |
|---|---:|---:|---:|---:|
| base, repeated | 512 | 0.000000 | 0.000050 | 100.000% |
| base | 64 | 0.009511 ± 0.000376 | 0.304 | 96.18% |
| fork | 512 | 0.007707 ± 0.000262 | 0.107 | 96.32% |
| fork | 64 | 0.010342 ± 0.000440 | 0.355 | 96.13% |

The base build is deterministic at a fixed batch. Changing only the physical batch on the base build moves the output by KLD 0.0095. The fork at the reference batch sits below that (0.0077). The fork at batch 64 sits slightly above it (0.0103), within about 1.4 combined standard errors. So the fork's numeric difference from upstream is the same size as upstream's own batch-size sensitivity, and its output is not bit-identical to upstream at the model level.

## Limits

- The sample is small: one model file, one quantization, one node, 8 perplexity chunks. The KLD corpus is the llama.cpp `docs/` markdown (sha256 in `receipts/digests.txt`), not a standard dataset.
- Depth 32,768 does not fit at the default physical batch on this card with this file: both builds fail to allocate a 3.3 GiB compute buffer (`realugbun-20261122T1845Z/deepA.log` in `receipts/windows.txt`). The 32K row therefore uses `-ub 128` and is not comparable to the other rows on prefill.
- A peer download (16 connections) was writing to the same NVMe during both windows. Decode standard deviation stayed at 0.35% of the mean or less and the two runs of each arm agree within 0.1%, so it is not a visible confounder for decode. It may affect prefill slightly.
- The first window attempt (`realugbun-20261122T1841Z` in `receipts/windows.txt`) pointed at the Unsloth root-layout file, which mainline cannot load. Only its backend-test result is used.
- No task-quality evaluation was run. Q3_K_XL is not quality-equivalent to the native NVFP4 and MXFP8 weights.
- No per-fusion ablation. The fork has no Inkling-only switch, so the gain cannot be divided among the kernels without more builds.

## Receipts

Under [`receipts/`](receipts/): every `llama-bench` JSON row keyed by window and label in [`probes.jsonl`](receipts/probes.jsonl); argv files, bench and perplexity logs, window logs and the build configure lines in [`windows.txt`](receipts/windows.txt); the two window scripts in [`harness.txt`](receipts/harness.txt); binary, model and corpus digests, the lease record and the build command in [`digests.txt`](receipts/digests.txt).
