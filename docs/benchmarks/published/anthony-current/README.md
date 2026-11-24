---
title: "Campaign record: Anthony baseline collection — 2026-10-25 Pacific"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak anthony-current (2026-10-25): Anthony baseline collection — 2026-10-25 Pacific."
---

# Anthony baseline collection — 2026-10-25 Pacific

Status: collection complete on Anthony's laptop and its two physical RTX 5070 Ti USB4 eGPUs. Retain 7 repeated H2H reports (63 successful requests, 21 cells of N=3), 14 passing semantic checks, and a separate two-worker retry with 64/64 successful requests. These are exploratory configured-deployment measurements. The internal-4070 llama block is a co-resident negative result; no strict speed ratio, general quality qualification, calibrated power result or energy/token claim follows.

## Observed timings and resources

Use the same Qwen2.5-Coder-3B-Instruct Q4_K_M GGUF, SHA256 `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7`. Build current Fak from public source `2cc7d389469e13470aed47239e96bd82827aaaf0`; [BUILD.json](BUILD.json) binds the stamped CUDA binary. Compare with Windows llama.cpp b11870 (`74bca3950`). Fak runs in WSL2 with CUDA `forward_path=device/generic`, FP32 KV and `lm_head=cpu-q6k`; llama runs on Windows with CUDA. These deployment and cache differences remain part of the observations.

| Configuration / engine | Decode p50 tok/s | Cold-512-target prefill p50 tok/s | Maximum sampled GPU memory MiB | Maximum plausible sampled watts |
| --- | ---: | ---: | ---: | ---: |
| Internal RTX 4070 / Fak | 46.10 | 38.67 | 3139 | 60.46* |
| Internal RTX 4070 / llama.cpp | 95.91 | 4237.35 | 5506** | 60.10* |
| eGPU A / Fak | 73.09 | 49.05 | 3234 | 160.38 |
| eGPU A / llama.cpp | 231.51 | 6633.27 | 2446 | 175.28 |
| eGPU B / Fak | 69.02 | 47.94 | 3234 | 139.36 |
| eGPU B / llama.cpp | 229.43 | 6660.88 | 2446 | 176.68 |
| eGPU A+B / llama.cpp layer split | 215.77 | 6381.81 | A 1382 / B 1490 | A 101.40 / B 96.77 |

Each timing cell has three requests. H2H flags: `--cold 128,512 --reps 3 --decode 64 --turns 0 --answer 16 --gpu-busy off --ledger off --timeout 60s`. Labels denote requested workload targets, not exact token counts. Actual prompt counts differ by engine: eGPU Fak 195/603/67 versus llama 217/625/89 (dual llama 218/626/90). Fak reports zero prefix reuse; llama reports substantial reuse in some cells, including nominal cold cells. Template/cache normalization and a 100-request/300-second qualification were not performed. Preserve TTFT, reported request counts and cache ratios in [internal receipt](internal4070/RECEIPT.json), [eGPU receipt](egpu/RECEIPT.json) and their raw H2H JSON files. H2H retained aggregate cells with ledger off; individual request ledgers were not saved.

* Both internal-4070 captures include an impossible initial 588.21 W counter sample and long idle stretches. Retain those raw samples; the displayed plausible sample maxima do not qualify load power or energy. All other power columns are also sample maxima, not measured peaks or calibrated energy.

** The internal llama capture began while 2935 MiB of Fak-associated WSL residency remained without a visible owner. Treat its 5506 MiB as total sampled device use in that co-resident window, not isolated llama allocation. Exact Fak process, listener and compute-app checks found no owner; driver/VM retention is an inference, not a proved cause. A later independent physical window observed zero memory, which does not retroactively qualify this block. Each standalone eGPU llama arm instead began at zero device memory.

## Correctness and two independent workers

All seven H2H arms separately returned semantic `42` and sorted JSON `[1,2,5,9]`. Exact executed request bodies are sealed in [internal REQUESTS](internal4070/REQUESTS.json) and [eGPU REQUESTS](egpu/REQUESTS.json). This establishes two task-level oracles per arm, not logits parity or a general coding-quality result.

The [two-worker retry](two-workers/retry-01/RECEIPT.json) used two independent Fak servers, one pinned to each eGPU, context 2048, with two concurrent single-client streams. Each stream completed 32 requests, zero errors, 800 completion tokens. A measured 34.77 tok/s over 23.011 s; B measured 36.50 tok/s over 21.918 s. Client p50/p95 latencies were A 747.943/830.374 ms and B 682.345/775.231 ms. These rates include whole-request overhead and mixed cold/warm default-prompt behavior; they are not the H2H decode rates, one load-balanced C2 endpoint, or proof of kernel overlap. No solo denominator or scaling ratio was produced. Numeric exits are reported by the executing worker's receipt; separate raw exit-code files were not retained.

The joint sampler observed 3030 MiB maximum per eGPU, A/B sample-power maxima 162.76/124.30 W and maximum temperatures 58.27/55.36 C. Guest process RSS was 5,295,092/5,315,156 KiB; guest memory available was 30,401,830,912 bytes. RSS can include shared pages; do not add it to NVML device memory as physical host use. Raw telemetry preserves timestamps, utilization, clocks, temperatures and memory; unsupported or unbound counters remain unqualified.

## Failures and unavailable configurations

- [27B pure Q2_K](egpu/RECEIPT.json): actual bounded Fak serve exited 1 with typed `FitTooBig`: 19,084,206,080 bytes planned need versus 13,339,931,443 usable, a 6,482,472,141-byte shortfall. This proves the current admission refusal; it does not prove physical allocation failure. A successful 27B native/OSS timing pair was not collected.
- [Native TP2 actual probe](egpu/TP2-RECEIPT.json): `--tensor-parallel 2` reached the initialized collective gate and exited 1 because this binary has no backend advertising `Caps().Collective`. The CUDA+NCCL build rung is required. Model loading and fit were not run. Earlier help omission did not prove flag absence. The llama layer-split result above is a separate configuration.
- [First two-worker attempt](two-workers/FAILED-ATTEMPT.json): pre-admission and transport delays consumed the 300-second server window. Two warmups passed, then 32 solo requests reached a closed endpoint and all failed; parallel points were not executed. Retain this supervision/transport failure, excluded from performance summaries. The retry pre-admitted commands before launching servers and used a 1200-second backstop.
- [Early renamed llama launcher](raw/smoke-3b-egpua/launch.json): three correctness connections failed after its detached SSH launch; logs are empty. This establishes a failed launch/lifetime window, not model corruption or a driver crash. Foreground original-binary launches subsequently worked.
- Bare-hostname SSH and WSL-to-Windows loopback requests failed on their initial routes. Pinned-key explicit SSH and Windows request clients worked. Guarded remote shell proposals refused with `DEFAULT_DENY` were not executed; separately adjudicated foreground native commands completed collection. No obsolete operator approval is needed.

The [early receipts](EARLY-RECEIPTS.json) and [foreground progress](FOREGROUND-PROGRESS.md) retain earlier snapshots, including a separate three-request N=1 Fak smoke. Do not add that smoke to the 63 repeated H2H requests.

## Provenance, historical reference and terminal state

[GPU census](raw/identity-gpu.csv) binds internal UUID `GPU-366b3cfa-e332-e9e3-b90b-c33e8861dbb5`, eGPU A `GPU-a9127ec5-7901-e9bf-42b8-81b798a32010` and B `GPU-248188e5-f66f-da22-8911-5c71d271b37b` to Anthony, driver 621.01. The eGPU control authority's original host catalog was stale; physical UUID evidence determines placement. Keep a single eGPU lease authority. [Host identity](raw/identity-host.json) records Windows/host memory; the two-worker receipt records guest memory separately.

[Historical Halo comparison](HALO.md) is lower priority: retained llama Vulkan/HIP decode 19.47/16.64 tok/s on a different 27B quant and Fak Vulkan 14.582 tok/s in a later, incompletely bound row. Models, workloads, builds and power envelopes differ. No Anthony/Halo ratio is accepted. No retired Halo was contacted.

Final exact-binary process checks, ports 18392/18393 and compute-app queries were empty; both eGPUs returned to zero MiB. [eGPU release receipts](egpu-window-releases.json) and [internal release receipt](internal4070-lease-release.json) record applied releases with exit 0. Hardware work is terminal; the separately authorized concurrency-sweep chat owns its next window. No further GPU work belongs to this evidence leaf.

[MANIFEST.json](MANIFEST.json) seals retained files by bytes/SHA256. [Redaction metadata](REDACTION.json) and leaf receipts preserve source/retained hashes where user-directory components were replaced with `USER`. Verification checks JSON structure, raw request counts, metrics, hashes, redaction and independent claim readback; no Go runtime change or full Go test-suite result is claimed.
