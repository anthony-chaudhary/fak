# Hardware-agent request: DS4.1 Flash on Raider 5090

Request ID: `D41-HW-20261028-0310`. Prepared 2026-10-28 UTC. Status: **posted for pickup; unclaimed and unrun; not hardware-verified**.

## Pickup refresh — 2026-10-31

The existing request was [posted on issue #12640](https://github.com/anthony-chaudhary/fak/issues/12640#issuecomment-6538245444) for independently authorized hardware executors. The posted refresh is a delivery receipt, not an executor claim or execution result. Campaign target: 2026-10-31 10:43 UTC. It adds changed-path CUDA acceptance for published ports and retains the six-expert DS4.1 inventory/admission protocol below. Prepared, unpublished candidates must wait for verified public source and native build identities before execution.

Use only existing authorized hardware and exclusive ownership. This refresh grants no new machine access, purchases, driver/security changes, foreign-lease takeover, or revival of withdrawn Mac/Halo operations. Record an explicit claim and target admission before running; report actual receipts rather than treating a posted request as progress on hardware.

## Published CUDA source update — 2026-10-31

The 63-path opt-in CUDA MMQ/checked-terminal/routed-MMID composition is now published at [5211e8d3c5dd030a20d3806fbbef2581fc51866f](https://github.com/anthony-chaudhary/fak/commit/5211e8d3c5dd030a20d3806fbbef2581fc51866f), independently read back on remote main. Defaults remain opt-in. Scoped host controls, including current-production routed-provider scope, passed. Genuine CUDA 13.1 SM80 ptxas/fatbinary/GCC stages produced object SHA256 `f25c5e269b5f0e23864f7fb219642d36eb469847239698dfc193a1769c32411b`; the original nvcc invocation was interrupted, so this is a recovered object-stage result, not a clean full build or hardware qualification.

Executors must build and identify their complete target-compatible native archive/binary and run the published required-device witnesses. Archive/cgo linking, device lifetime/numerics, model parity, throughput, and native DS4.1 generation remain unverified by this update. At the 2026-10-31 03:59 UTC response check, no operator claim or run had been verified. The original posted request and all authority, ownership, resource, and stop limits above remain unchanged.

## Routing instruction — do not wait for the coordinator's connection

All hardware verification is to be requested in writing and performed by other independently authorized hardware agents. The hardware executor uses its own authorized Raider/fabric environment; a computer connection to the coordinator is **not** a prerequisite. This supersedes that prerequisite in older handoff notes. The executor owns machine admission and serialization; an independent reviewer audits its receipts. This request neither names an assigned agent nor grants access to new machines.

Return the inventory/admission receipt promptly, then incremental results and a terminal evidence bundle. If target access is unavailable to the assigned executor, report the exact unavailable target or permission and handoff needed; do not report that the coordinator must connect first. No hardware run has been established by the source research below.

## Outcome and ownership

1. **Hardware executor:** inventory Raider; qualify artifacts/runtime; obtain a genuine, coherent six-expert GSQ completion; measure local RAM/SSD and GPU-cache behavior; compare Unsloth under matched settings if resources admit it.
2. **Fabric executor:** after a valid local reference, use an existing authorized protected fabric route and qualified loader to measure the same workload. Local results must not wait for this optional arm. Report the actual mechanism: remote file storage, llama RPC, or Fak-native expert execution; these are not interchangeable.
3. **Independent evidence reviewer:** check identities, byte sums, configuration parity, raw timings/pressure and actual teardown; distinguish measured, inferred, blocked and unrun.

One owner of each GPU/model process; coordinate downloads/builds and serialize heavy work. Read-only inventory/review can proceed independently. Existing foreign `ggufload` ownership blocks conflicting live Fak edits, not an independently admitted llama.cpp reference run. Do not clear or bypass a foreign lease to execute this request.

## 1. Inventory and admission receipt

Record UTC time and target identity; OS/kernel and native Windows versus WSL/Linux; CPU/NUMA; physical and available RAM, commit/pagefile/swap and container/WSL limits; exact NVIDIA product (Laptop versus desktop), driver and dedicated VRAM total/free/used **bytes**, existing consumers, shared GPU memory, PCIe generation/width, temperature/clocks/power. The user's “24 GB” is not verified hardware inventory; distinguish GB from GiB.

Record model-volume identity, filesystem/quota/free bytes, SSD model if readable, and already-present exact models/partial downloads in known directories. Inventory existing runtime version, commit, binary SHA-256, build flags, help and CUDA device enumeration. Use installed read-only tools and supported platform equivalents; do not install monitors just to fill a field. Inventory only existing authorized fabric peers/routes, without scanning arbitrary hosts or changing exposure.

Admit GSQ first. Both complete model sets consume **583,627,802,815 bytes (543.546 GiB)** before staging, runtime or reserve; do not automatically download both or duplicate Engram to satisfy a filename. Record resources and approved run budget before acquisition or execution. Full local size/SHA verification is still required: the appendix identities are published LFS pointer evidence, **not proof the model payload has been downloaded or verified locally**.

## 2. Artifact and runtime contract

- GSQ: [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/a419d74cd28d5b9f8c5e2b9077d11a609c17bc32), revision `a419d74cd28d5b9f8c5e2b9077d11a609c17bc32`, IQ2_XXS native six-shard layout. Backbone **138,238,967,104 B**, IQ4_NL Engram **110,595,290,816 B**, total **248,834,257,920 B (231.745 GiB)**.
- Unsloth: [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/60ad3651641b5c97679c807e064ab1d7a3ca1455/UD-Q2_K_XL), revision `60ad3651641b5c97679c807e064ab1d7a3ca1455`, UD-Q2_K_XL seven shards. Backbone **199,621,523,103 B**, Q5_K Engram **135,172,021,792 B**, total **334,793,544,895 B (311.801 GiB)**. Different Engram formats are not deduplicable by assumption.
- Reference runtime: [llama.cpp b11936](https://github.com/ggml-org/llama.cpp/tree/27b7a8a009de2f5f75d8111b6171381c920df764), commit `27b7a8a009de2f5f75d8111b6171381c920df764`. This is the researched reproducibility pin, not a claim to be newest at execution time. Reuse an admitted compatible binary; record deviations. A newer runtime is a separate pinned arm, never a silent replacement. Any build needs its own capacity admission.

Open validated shard 1. GSQ shard 1 is metadata-only; present its shared `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` as `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf` according to pinned loader/model instructions. Verify identity and supported link semantics; do not overwrite an existing file or copy 110 GB unnecessarily. Reject incomplete/wrong-layout/hash-mismatched artifacts.

## 3. Matched execution matrix

Freeze and record text-only, **six routed experts**, context 2048, concurrency 1, prompt batch 128/microbatch 32, fixed threads/affinity, GPU layer placement, KV type, attention and operation-offload policy, template/tokenizer, seed, prompt and output limits. No DSpark/drafter, speculative decoding, reduced expert count or hidden prompt/KV reuse. A five-expert Strata result is a separate quality-changing experiment, not this baseline.

Use a loopback-only `llama-server` and fixed request harness. Check actual binary help before using `--cpu-moe --moe-cache-mib <MiB> --load-mode mmap --lazy-mode auto`. Pinned llama-bench lacks the expert-cache option; do not substitute an unsupported benchmark. Keep exact full command/environment in the receipt, excluding secrets.

| Arm | Requested outcome |
|---|---|
| G0 | GSQ cache 0: first token and coherent completion, then controlled timed requests |
| G4 | GSQ cache 4096 MiB, only if measured headroom admits; identical workload |
| G+ | At most one higher bounded cache budget, based on G0/G4 peaks rather than assumed free VRAM |
| U0/U4 | Admitted Unsloth equivalents, same core settings, enumerate every difference |
| F | Existing protected fabric, same reference workload and explicit implementation/placement/transport; otherwise exact blocked record |

Begin with a short smoke prompt and fixed maximum 32 generated tokens; then a fixed, nonprivate prompt of about 128 tokens and maximum 128 generated tokens for matched timing. Preserve actual token counts and prompt bytes/hash. Obtain at least three warm repetitions per admitted comparison; label first-observed/cold and warm separately. Do not globally flush OS caches to manufacture “cold.” A larger/context-scaling workload needs a new admission ledger.

`--cpu-moe` describes host weight placement, not proof of CPU matmul execution. Keep operation-offload policy constant; a CPU-executed control requires separate flags and backend evidence. The GPU expert cache is reactive SLRU, not predictive prefetch. It can retain CPU backing while adding GPU copies. Small active parameter count does not mean the full backbone fits VRAM. Lazy Engram mappings still consume resident pages and SSD I/O; file size, virtual mapping, RSS/PSS and page cache differ. Do not double-count one file-backed physical page as RSS plus page cache.

## 4. Measurements and receipts

Collect timestamped startup and request logs, actual output, load-to-ready seconds, TTFT, prompt-processing and decode token/s, input/output counts, per-token latency p50/p95 if available. State whether model loading is included in TTFT. Capture sampling interval and measurement scope.

Capture GPU peak dedicated allocation/reservation/driver usage and free bytes, shared-memory spill, utilization and thermal/throttle state; host peak working set/RSS/PSS, anonymous/file-backed memory, mapped bytes, available RAM/commit/swap/pagefile, major/minor faults and actual memory-pressure counters where available. Capture SSD read bytes/operations, throughput, latency/queue where available; attribute expert versus Engram traffic only with counters, otherwise label inference. Cache hits/misses/evictions/upload bytes/time require actual runtime counters; unavailable is **not zero**. GPU utilization alone proves none of these.

For fabric, capture per-peer placement and resources, actual transport/route, per-peer traffic during loading versus decoding, latency/errors and equivalent workload. Do not claim Fak-native execution from llama.cpp logs or a control heartbeat. Use only already-protected authorized endpoints; no firewall, credentials, RDMA, or internet-exposed RPC changes.

Return a compact bundle with:
- Manifest: request ID, executor and target identity, UTC start/end, model revision and every local filename/size/hash plus link identity, runtime commit/binary SHA/build/device list.
- Full sanitized invocation/configuration; prompt bytes/hash, template and actual token IDs or counts; raw outputs and timestamped performance/resource logs with units and sample period.
- Admission decision, process identity, termination/exit receipt, any exceptions and per-arm `PASS / FAIL / BLOCKED / UNRUN` table. A coherent completion is first-run evidence, not proof of full model quality.
- SHA-256 manifest for delivered evidence and an accessible handoff location; independent review recomputes sizes/timing and verifies six-expert controls. Send inventory first, incremental findings next, terminal evidence last. Preserve useful partial evidence after a stop.

## 5. Safety and stopping conditions

The executor must declare locally appropriate floors before starting; suggested conservative defaults are available RAM ≥ max(8 GiB, 10% physical RAM), dedicated GPU free ≥ 1 GiB in steady state, and model-volume free ≥ max(20 GiB, 10% capacity). These are proposed experiment guards, not measured machine properties. Record any owner-approved adjustment before the run.

Stop the owned workload on OOM/allocation failure, new OOM kill, floor breach, uncontrolled shared-GPU spill, sustained thrashing, thermal guard breach, invalid/NaN/garbled output, or loss of monitoring/control. Bound initial load to 10 minutes with observable progress, smoke to 120 seconds, a timed request to 5 minutes, and initial arm to 30 minutes. A necessary extension requires a documented new resource/time decision, not blind retries. Do not kill unknown processes or modify drivers, power/security settings, swap, global caches or source/worktrees. No destructive SSD test or reboot. Teardown only the workload this executor owns; verify exit.

If RAM, disk, compatible runtime, artifact identity, permission or peer availability fails admission, return the exact deficiency and next handoff. An unrun Unsloth/fabric arm does not invalidate a measured GSQ local result. Completion requires actual receipts or an explicit blocker for each requested arm—not a running label.

## Appendix: immutable published payload identities

GSQ filenames below use prefix `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-` except the named shared Engram file. Unsloth filenames use prefix `DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-` inside `UD-Q2_K_XL/`. Values are copied from already-reviewed source receipts, not newly fetched payloads.

| Set / suffix | Bytes | SHA-256 |
|---|---:|---|
| GSQ `00001-of-00006.gguf` | 5,780,704 | `f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9` |
| GSQ `00002-of-00006.gguf` | 39,535,781,184 | `25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3` |
| GSQ `00003-of-00006.gguf` | 39,736,712,000 | `8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48` |
| GSQ `00004-of-00006.gguf` | 39,036,908,384 | `9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0` |
| GSQ `00005-of-00006.gguf` | 19,923,784,832 | `7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1` |
| GSQ `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110,595,290,816 | `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3` |
| Unsloth `00001-of-00007.gguf` | 5,780,927 | `0c2a2031bd93ebd60aa6a4f3912775ba3990b5f39b24c6edf9ea029d2f387a28` |
| Unsloth `00002-of-00007.gguf` | 49,290,279,808 | `605972fece1b43277b9ff86a2cd79c9002ff424caefc507f20b6ccd47ae4ab39` |
| Unsloth `00003-of-00007.gguf` | 49,745,070,208 | `1cb58786245bd8c38478174eecc648b12691bb79c2bb2f446c135ab00263580e` |
| Unsloth `00004-of-00007.gguf` | 48,680,128,800 | `7b753295dd32534dfea52b2e9b5c4741b8007a955e38702ee37034271df7526e` |
| Unsloth `00005-of-00007.gguf` | 49,970,937,696 | `336be674937fe29683e7aeee7c6af32f896b864931a6c916907b78e0f3692de4` |
| Unsloth `00006-of-00007.gguf` | 1,929,325,664 | `192694fb09612e1200cce9ae35f3645ad39cf8aa43e94417070af9c4f4ecb43b` |
| Unsloth `00007-of-00007.gguf` | 135,172,021,792 | `a653c5309b7c74a6c41580aaa8c2d2bbdae0342253bca2cd3fc9e69d05f3d0eb` |
