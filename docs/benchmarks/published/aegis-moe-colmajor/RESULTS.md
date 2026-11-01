# MOE-CM-HW-20261031-1856Z — hardware witness of the published col-major MoE GEMM tile-mapping defect on aegis-fak (2026-10-31)

Ticket `moe-colmajor-hw-witness-20261031` (lane `nvidia-test-leasing`). Node: **aegis-fak**,
RTX 5080 16 GiB, UUID `GPU-3e9fda0a-061a-6eb7-26c3-541bf8489090`, sm_120 (CC 12.0),
driver `615.78.08`, Arch Linux, CUDA toolkit 13.4 at `/opt/cuda`. Share lease
`hwl-4d3ce83e299a4a4a` (`owned-rtx5080-aegis`, epoch 120). Pre-existing consumer
`/usr/local/bin/fak` serving service (~3.2 GiB) untouched throughout; raider-fak not
attempted. Concurrency 1 on the GPU for all experiment launches; inter-test cooldowns.
Machine-readable pins: [manifest.json](manifest.json); raw streams in [raw/](raw/)
(`env.jsonl`, `selftest.jsonl`, `testA.jsonl`, `testB.jsonl`, `diag_write_set.json`,
`run.log`, `runA.log`, `diffs/`).

## Verdict

**[HW-WITNESSED] The pinned published kernel silently drops most of its output.** On real
sm_120 silicon, the pinned `v2_moe_fused.py` (published col-major mapping) writes only
**5.2 % / 12.3 % / 36.2 %** of output elements at EM = 124 / 1144 / 4600; every unwritten
element stays exactly zero while the reference is O(1) (max_abs_err up to 5.82) — silent
corruption, no error raised. The blog-order fix (`pid_m = pid % grid_m;
pid_n = pid // grid_m`) achieves **100 % coverage with all elements within tolerance** at
all three sizes. Moment-in-time perf data for the blog's two techniques collected below.

## Source pin and transfer

Pinned source: `github.com/meta-pytorch/applied-ai @ 2391954b19988bd76cf3c2ea84c1ce74b68d568b`,
subtree `kernels/triton/inference/col_major_moe_gemm`. Transferred verbatim to
`aegis-fak:~/moe-cm/20261031T1856Z/` and sha256-verified remote-vs-local:

| file | sha256 |
| --- | --- |
| `v0_moe_fused.py` | `9c86cc16c13c8392fe489767e3eb01e6899e8795c637d394f3346e3395a242ec` |
| `v1_moe_fused.py` | `d8d8d1fd06438c6c1e260d31e17589c716638ccd18396a7175b34ebcee0ab3ed` |
| `v2_moe_fused.py` | `bb14ebf3df3bb50463b2b726ca73e99b03f99ccc50edd6fd206f62b27213325f` |

Derived variants (single-hunk diffs in `raw/diffs/`): `v2b` = v2 + blog-order mapping;
`v2rm` = v2 + row-major mapping; `v1cm`/`v1rm` = v1 SplitK kernel with col-major /
row-major mapping. Runner `run_check.py` (sha256 `f33e34553507ecf911d92e7b0f32827ee4e207b3def5051339c13487b07207db`)
implements a pure-torch `moe_align_block_size` equivalent, self-tested against the pinned
source's docstring example before any experiment (`raw/selftest.jsonl`: PASS).

Environment (from `raw/env.jsonl`): Python 3.13.18, torch `2.15.0+cu132`, triton `3.9.0`,
CUDA runtime 13.2, venv `~/moe-cm-venv`; seed 1234; geometry E=8, K=4096, N=14336,
topk=1, fp16, input scale 1/8; correctness tolerance atol=0.01 absolute, rtol=0;
v2 config block_m=64/block_n=64/block_k=32; TEST B config block_m=32/block_n=64/
block_k=64 (+ split_k=2, group_m=8 for the SplitK arms). Reference = dense per-expert
torch GEMM gathered back to token order.

## TEST A — correctness / coverage (defect witness)

Per-variant, per-EM records in `raw/testA.jsonl`. `elem_cov` = fraction of C[:M, :N]
elements within tolerance of the reference; `tile_cov` = fraction of 64×64 geometric
output tiles fully within tolerance.

| EM | ntp | P_published elem_cov | P tile_cov | B_blogfix elem_cov | B tile_cov | max_abs_err (P / B) |
| --- | --- | --- | --- | --- | --- | --- |
| 124 | 512 | **0.0523** | 8/448 (0.018) | 1.0 | 1.0 | 5.21 / 0.0 |
| 1144 | 1472 | **0.1231** | 106/4032 (0.026) | 1.0 | 1.0 | 5.31 / 0.0 |
| 4600 | 4736 | **0.3623** | 1152/16128 (0.071) | 1.0 | 1.0 | 5.82 / 0.0 |

SplitK-on + col-major spot check (v1cm, split_k=2): elem_cov = tile_cov = 1.0 at all three
EMs (max_abs_err 0.003906, fp16 storage rounding; matching is atol-based, so
`max_rel_err_matched` spikes on near-zero reference values are expected and immaterial).

### Mechanism (verified, not just observed)

The pinned mapping is `pid_m = pid % grid_n; pid_n = pid // grid_m` with the early-return
guard `if pid_m * block_m >= num_tokens_post_padded: return`. Two details make the damage
large and easy to mispredict:

1. **The kernel's `grid_m` is computed from `EM = len(sorted_token_ids)`**, which is
   `total + E*(block_m - 1)` (the sentinel tail), not M and not ntp. For topk=1,
   block_m=64, E=8: EM_arg = M + 504 → grid_m_kernel = cdiv(M + 504, 64).
2. Only pids with `pid_m < cdiv(ntp, block_m)` proceed, and each such row-band receives
   exactly `G / grid_n = grid_m_kernel` column-tiles out of `grid_n = 224`. So the
   written-column fraction per band ≈ `grid_m_kernel / 224`:

| EM | grid_m_kernel | predicted fraction | measured elem_cov |
| --- | --- | --- | --- |
| 124 | cdiv(628,64)=10 | 4.5 % | 5.2 % |
| 1144 | cdiv(1648,64)=26 | 11.6 % | 12.3 % |
| 4600 | cdiv(5104,64)=80 | 35.7 % | 36.2 % |

Because `sorted_token_ids` groups tokens by expert (not by output row-band), even the
written stripes scatter across many geometric 64-row bands — hence the geometric-tile
full-match rate (2.6 % / 7.1 %) sits well below element coverage (12.3 % / 36.2 %).
Unwritten elements remain exactly zero: **silent corruption**.

Compiled-behavior verification (so the witness is not an artifact of stale cache or
misread source): TTGIR inspection shows the exact published lowering
(`pid_m = remsi(pid, grid_n)`, `pid_n = divsi(pid, grid_m)`, guard `cmpi sge ntp`), and an
on-device probe kernel replicating the prologue recorded `proceed = 598 = 23 × 26`
distinct `(pid_m, pid_n)` pairs at EM=1144 — matching the algebra exactly
(`raw/diag_write_set.json`). Reproduced with a clean Triton cache dir.

The blog-order fix is a bijection over the grid (every `(pid_m, pid_n)` tile written
exactly once) → 100 % coverage, all elements within tolerance, at all three EMs.

## TEST B — moment-in-time performance (blog's two techniques)

Single-launch kernel time, warmup 3 + reps 11, all arms at TEST B config
(block_m=32, block_n=64, block_k=64; SplitK arms add split_k=2, group_m=8).
Records in `raw/testB.jsonl`. **Moment-in-time**: shared node with the pre-existing
consumer present; concurrency 1 for our launches; not an isolated throughput receipt.

| EM | arm | mapping | SplitK | p50 ms | min / max ms | vs baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 124 | rm_off | row-major | off | 0.3064 | 0.3051 / 0.3173 | — |
| 124 | cm_off | col-major (published order) | off | 1.0812 | 1.0766 / 1.0875 | **+253 %** vs rm_off |
| 124 | rm_on | row-major | on | 0.3010 | 0.2994 / 0.3120 | −1.8 % vs rm_off |
| 124 | cm_on | col-major | on | 1.0862 | 1.0811 / 1.1019 | +0.5 % vs cm_off |
| 4600 | rm_off | row-major | off | 8.1318 | 8.0078 / 8.3215 | — |
| 4600 | cm_off | col-major (published order) | off | 5.7592 | 5.7529 / 5.7738 | **−29.2 %** vs rm_off |
| 4600 | rm_on | row-major | on | 6.2704 | 6.2560 / 6.2841 | −22.9 % vs rm_off |
| 4600 | cm_on | col-major | on | 5.8749 | 5.8661 / 5.8841 | +2.0 % vs cm_off |

Read: at decode-like M (124) the col-major ordering costs ~3.5×; at prefill-like M (4600)
it is ~29 % faster than row-major. SplitK helps row-major ~2 % at both sizes and is
neutral-to-slightly-negative for col-major. Caveat: the col-major arms at the pinned
mapping produce incorrect output (TEST A) — timings are valid execution-cost data,
correctness is not.

## Teardown

- Owned experiment processes killed on aegis-fak (`pgrep -af "[m]oe-cm-venv"` → none
  remaining; no foreign processes touched).
- VRAM readback post-teardown: **used 3234 MiB / free 12692 MiB** — identical to the
  pre-run baseline (0 % drift, within the ±2 % gate); util 0 %, 36 °C, 16.4 W idle;
  only the pre-existing `/usr/local/bin/fak` consumer remains.
- Lease `hwl-4d3ce83e299a4a4a` released via `fak-sync nvlease release --id <id> --holder <h> --apply`
  → verdict `ok`; release event recorded in the lease journal at 2026-10-31T19:21:46Z
  (verified with `nvlease history --id hwl-4d3ce83e299a4a4a`).
- No quarantine items; run directory retained on-node at `~/moe-cm/20261031T1856Z/`
  for re-audit.
