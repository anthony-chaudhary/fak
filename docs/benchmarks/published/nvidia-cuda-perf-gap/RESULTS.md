# Wave-2 step 1+2: H2H re-measure at public trunk HEAD `c7caeb22d` + prefill kernel profile

**[HW-WITNESSED]** 2026-10-31T02:05Z-02:52Z. Host `raider-fak`, internal RTX 5090 Laptop
`GPU-2a2a837f-c82d-0f22-4538-e3a063903380` (driver 615.78.08) pinned via
`CUDA_VISIBLE_DEVICES`, `Current Power Limit 95.00 W`. Measure lease
`hwl-0900772181a96b06` (holder `pi-goal-nvidia-cuda-perf-gap`), released 02:51:34Z and
verified in `.fak/nvlease/lease-history.jsonl`. `fak.service` stopped for the window,
restored active+READY (22858 MiB) at 02:51Z. Node idle otherwise.

Model `/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf` (sha256 `7e78da5d…`), ctx 2048,
kv fp32. Binary built on-node from `origin/main c7caeb22d6f3b3ba1d463025f557cb1d59792338`
(sha256 `2d860a1e…cc61a29e`, v0.57.0, SASS sm_120, 45 s; [raw/build.log](raw/build.log)).
llama.cpp b11870 denominator **reused from committed**
[h2h-head-20261029](../raider-5090/h2h-head-20261029/RESULTS.md) numbers per task
instruction — not re-run this leaf.

## Headline

**Trunk HEAD regressed this lane first: it cannot serve the baseline config at all.**
Bisect isolates the first bad commit to
`217d2d0f1` ("fix(compute): advertise implemented CUDA i-quant weight formats", Ref #12640).
With the documented revert knobs pinned, HEAD measures **+37% decode vs the daa1178b
baseline** (the mmvq default flip) and cold-512 now completes where daa1178b OOM'd.
L3 bar still not met: decode 0.745x llama (need ≥0.8x), prefill 0.284x (need ≥0.5x).

| Scenario (p50, n=3) | fak daa1178b (committed) | fak c7caeb22d + revert knobs | delta | llama b11870 (committed) | fak/llama |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode128: decode tok/s | 13.31 | **18.19** (18.15-18.27) | **+36.7%** | 24.41 | **0.745x** (was 0.545x) |
| cold 128: prefill tok/s | 171.5 | 160.85 (146.9-167.2) | -6.2% (flat within spread) | 566.6 | 0.284x |
| cold 128: TTFT ms | 1219 | 1293.1 | +6.1% | 364 | 3.55x slower |
| cold 128: decode tok/s | 12.21 | 16.24 | +33.0% | 27.25 | 0.596x |
| cold 512: prefill tok/s / TTFT ms | OOM (0/3 valid) | **174.08 / 3676.4 (3/3 valid)** | new capability | 955.7 / 668 | 0.182x |
| cold 512: decode tok/s | OOM | 14.35 (13.72-14.51) | new capability | 19.08 | 0.752x |

Peak VRAM during Run A: 21186 MiB ([raw/gpu-a.csv](raw/gpu-a.csv)) vs 23.8-24.1 GB
resident under daa1178b — mmvq skips the Q4 coefficient ring, which is also why
cold-512 no longer OOMs.

## Regression witness (the blocker found before any measurement was possible)

HEAD unknobbed fails warmup on the exact baseline config:

```
fak serve: backend startup warmup failed after 4.832s; readiness held as warmup_failed:
... tensor-preflight: gdn_in_qkv must be resident f32 or q8_0 for the existing GDN
sequence primitive; session closed, no CPU retry          (raw/serve-a2.log)
```

- Control: the old `daa1178b` binary reaches READY in 10.97 s on the same box/config
  ([raw/serve-old.log](raw/serve-old.log)); HEAD's fit plan flipped device weights
  8207661056 → 5653723136 (GDN projections dropped off the device-resident set).
- Bisect `daa1178b..c7caeb22d` (7 probes, build+serve+[READY]): GOOD at 49e730e76,
  dfb2fa205, 3a750b0fa, 7741e2725, 6a54cb30b, 4a675abca; **first BAD = 217d2d0f1**
  ([raw/bisect/](raw/bisect/)).
- Mechanism: `217d2d0f1` adds IQ2_XXS/IQ2_XS/IQ1_M/IQ4_NL to the CUDA matmul dtype
  declarations. `cmd/fak/serve_load_helpers.go` then appends
  `ggufload.WithDenseIQResident(IQ2_XXS, IQ2_XS)` to every CUDA serve load, and the
  loader gate `qwen35DeviceQKNativeRow` (`internal/ggufload/qwen35_native_rows.go:146`,
  fires when `len(residentDenseIQ)>0`) admits GDN in_proj_qkv/z + full-attention q/k
  rows to raw Q4_K residency on CUDA. That route was designed for Vulkan's composed
  MatMul path; the CUDA GDN sequence primitive has no raw-row consumer, so its
  tensor-preflight fails closed. The commit's own witness line says "device execution
  ... not claimed" — full-model acceptance was deferred, and this lane is where it
  broke.
- Operator workaround used for all timed arms: `FAK_QWEN35_NATIVE_QK_ROWS=0
  FAK_QWEN35_NATIVE_GDN_INPROJ_ROWS=0 FAK_QWEN35_NATIVE_SSM_OUT=0` (documented
  per-class revert knobs); they restore the historical Q8 route byte-for-byte
  (fit_confirm identical to baseline).
- Fix direction: gate `qwen35DeviceQKNativeRow`/`qwen35TiledOutProjResident` on a real
  raw-row execution capability of the consuming forward, not on "backend advertises
  any native i-quant dtype". **Open follow-on: file a ticket against 217d2d0f1.**

## Prefill kernel profile (Run B, one run, `FAK_CUDA_TIMESTAMP_PROFILE=1`)

560 `fak-cuda ts-stage` lines over cold-128 ×3 reps, ~1125 ms total stage time
([raw/ts-stage-all.txt](raw/ts-stage-all.txt), totals [raw/ts-stage-totals.tsv](raw/ts-stage-totals.tsv)):

| Kernel | ms | share |
| --- | ---: | ---: |
| k_qwen35_gdn_fused_in_proj | 276.7 | 24.6% |
| k_q4k_mmvq_gemv | 188.8 | 16.8% |
| k_q6k_gemm_panel | 145.9 | 13.0% |
| k_qwen35_gdn_out_proj | 106.5 | 9.5% |
| k_q6k_gemm | 101.3 | 9.0% |
| k_flash_attention | 56.6 | 5.0% |

GDN projections (in_proj + out_proj) = **34.1% of profiled prefill stage time** —
consistent with [ATTRIBUTION.md](ATTRIBUTION.md) §4 naming the GDN layout the prefill
hot spot. Next hill for the prefill gap: GDN fused in-proj first.

## Engines, argv, env

See [raw/argv.txt](raw/argv.txt) for timestamped argv. Client: pinned h2h
`~/oss-engines/fak-h2h-c9870f151d` (sha256 `fbcab944…`). Serve env: service drop-in
env plus the three revert knobs (and `FAK_CUDA_TIMESTAMP_PROFILE=1` for Run B only);
mmvq decode is now the trunk default, so no `FAK_CUDA_Q4K_DECODE` pin was needed.

## Status language

- Re-measure at trunk HEAD: **done, with deviation** — numbers are real HW-witnessed
  c7caeb22d results but require the documented revert knobs because HEAD regressed
  the lane; without them the receipt would be empty.
- Trunk regression: **witnessed and isolated** (commit + mechanism + bisect trail);
  **not yet fixed** — no fix landed, no ticket filed yet.
- L3 maturity bar: **not met yet** — decode 0.745x (gap 0.055x), prefill 0.284x
  (gap 0.216x).

## Next checkable steps

1. File a ticket against `217d2d0f1` (capability-gate mismatch, Ref #12640 lineage)
   with this receipt as witness; land the gate fix.
2. After the fix lands: re-run this exact argv without the revert knobs to confirm
   warmup passes and numbers hold (~18 tok/s decode expected).
3. Prefill hill: target `k_qwen35_gdn_fused_in_proj` (24.6% of prefill stage time) —
   candidate directions already sketched in ATTRIBUTION.md §4.

## Receipts

[perf-receipt-v2.json](perf-receipt-v2.json) · `raw/`: serve logs (a/a2/old/b), client
stdout/stderr, ledgers, gpu-a.csv, ts-stage artifacts, config.toml, build.log,
bisect/{probe.sh,build-*.log,serve-*.log}. Full logs also at
`raider-fak:~/h2h-head-recheck-20261031-runs`; build worktree kept at
`raider-fak:~/h2h-head-recheck-20261031` (registered in fak-public `git worktree list`);
bisect worktree removed.
