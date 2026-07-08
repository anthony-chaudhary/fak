---
title: "GLM-5.2 L4 flash-attention + CUDA-graph decode (#3076): generation triage + GPU server-3 witness gate (2026-07-06)"
description: "Triage for issue #3076 (epic #3073, GPU server 3 / Lane A / L4): classifies the -fa + CUDA-graph-decode lever as gen/now, hands a GPU server-3 operator the turnkey 2x2 (fa on/off × graph on/off) decode tok/s + TTFT protocol and greedy first-token parity check with the exact llama.cpp flags per cell, and names the WITNESSED acceptance gate this GPU-less Windows worker cannot reach. NO number here is a served measurement; the only WITNESSED cell is the UD-Q4_K_M baseline carried from the 07-01 resident-serve note."
---

# GLM-5.2 L4 flash-attention + CUDA-graph decode (#3076): triage + witness gate

> **What this is.** A generation triage + turnkey harness spec for issue
> [#3076](https://github.com/anthony-chaudhary/fak/issues/3076) — the **GPU server 3 / Lane A /
> L4** lever under epic [#3073](https://github.com/anthony-chaudhary/fak/issues/3073).
> It classifies the horizon from issue evidence and hands a GPU server-3 operator the exact
> 2x2 A/B protocol (flags, parity check, result schema) needed to produce the WITNESSED
> artifact the issue actually accepts.
>
> **What this is NOT.** Not the benchmark. The 2x2 requires a live 433.82 GiB resident
> serve on a GPU server-3 8-GPU datacenter server (sm_80) lab node — hardware this Windows
> worker cannot reach (see §4). The only cell below that is a served measurement is the
> **UD-Q4_K_M baseline**, carried WITNESSED from
> [GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md).
> Every other cell is **PENDING** until an operator runs it; nothing here is fabricated.

## 1. The ask (verbatim intent)

Lever **L4** (~1.2–1.8×, the launch/attention-overhead lever in the
[ceiling map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) §5). Add llama.cpp `-fa`
(flash-attention) and enable CUDA-graph decode capture on the resident serve; A/B decode
tok/s and TTFT with each on/off; verify output parity (greedy first-token match). The
issue's own premise: this lever "matters more once L1 removes the layer-split bubble"
([#3075](https://github.com/anthony-chaudhary/fak/issues/3075), the `-sm layer` → tensor/row
split). **Accept** = a 2x2 (fa on/off × graph on/off) decode tok/s table on the 8-GPU serve,
plus the parity check, recorded under `experiments/benchmark/runs` labelled WITNESSED/OBSERVED.

## 2. Generation triage — `gen/now`

Classified from issue evidence per [docs/generation.md](../generation.md); the horizon is
clear, so this is **not** kept `needs-triage`. (The label/milestone binding is applied on the
issue itself; #3076 already carries `gen/now` + `Generation G0 - Now / Immediate`.)

- **Stream: `gen/now`.** #3076 improves the **current** product — GLM-5.2 as served today on
  the lab nodes at 23.2 tok/s WITNESSED — with a clear one-run witness path and no dependency
  on a future architecture bet. `-fa` and CUDA-graph capture are already-shipped llama.cpp
  capabilities, not a research option. That is the `gen/now` definition exactly.
- **Promotion evidence** (what makes it now-horizon): it is a child of epic #3073, the active
  "drive 23.2→~150 tok/s **in a day**" program on current 8-GPU iron; L4 is one measurable
  serve-config lever on the current llama.cpp resident serve.
- **Demotion / retirement evidence** (what would move it off now): if the post-L1 A/B shows
  **<~1.05×** on both flags → retire as *measured-no-win* (the issue's own demotion clause).
  If the resident-serve engine pins a llama.cpp build where `-fa` / graph capture is broken
  for this model's attention shape → demote to *blocked-on-engine*. If a peer lands the 2x2
  first on the shared trunk → retire as *duplicate*.
- **Invalidating assumption** (the one the A/B exists to test): that the graph axis is even
  *live* on the **current** serve. llama.cpp uses CUDA-graph capture on a narrow path and skips
  it in dynamic / multi-device-split cases; the resident serve runs `-sm layer` across 8 GPUs
  today, so graph capture may be **inert** (graph-on ≈ graph-off) until L1 (#3075) switches to
  a tensor/row split. If the load log shows graphs disabled under the current split, the
  graph-on column is expected to read ~1.0× — that is a **real recorded finding**, not a
  failure, and it is exactly why the issue orders L4 after L1. Confirm the actual state from the
  llama-server load log (§3), do not assume it.

## 3. The 2x2 matrix (what to run) + the exact flags

Node **GPU server 3 · Lane A · L4**. Engine = llama.cpp sm_80 CUDA build (engine-honest
baseline; keep any pure-fak kernel numbers in a separate artifact — epic rule, #1482 lane
stays separate). Serve config mirrors the 07-01 baseline: `--n-gpu-layers 999`, no
`--n-cpu-moe`, `--ctx-size 8192`, `/v1` alias `glm-5.2`, temperature 0, single-stream decode
of a fixed 256-token continuation from a fixed prompt.

**The two axes (confirm the exact spellings against the pinned build's `--help` — flags drift
across llama.cpp versions, same version-pin discipline as `tools/glm52_serve.sh`):**

- **flash-attention axis** — `-fa` on the launch line (recent builds: `--flash-attn on` /
  `--flash-attn off`; older builds: bare `-fa` = on, absent = off). Baseline launch did **not**
  pass `-fa`, so the WITNESSED 23.2 tok/s is the **fa-off** column.
- **CUDA-graph axis** — there is no CLI flag; the documented toggle is the env var
  **`GGML_CUDA_DISABLE_GRAPHS`**. `GGML_CUDA_DISABLE_GRAPHS=1` in the server's environment =
  **graph off**; unset (build default) = **graph on/attempted**. Because capture is skipped
  under `-sm layer` multi-device split (see §2), record what the **llama-server load log
  actually reports** for flash-attn state and CUDA-graph enablement — that log line is the
  ground truth for which cell each run occupies, not the flag you passed.

| Cell | fa | graph (`GGML_CUDA_DISABLE_GRAPHS`) | decode tok/s | TTFT ms | greedy 1st-token id |
|---|---|---|--:|--:|---|
| C0 (baseline) | **off** | default/unset (state at capture **unrecorded** in 07-01 note) | **23.2** WITNESSED | PENDING | PENDING |
| C1 | **off** | off (`=1`) | PENDING | PENDING | PENDING |
| C2 | **on** (`-fa`) | off (`=1`) | PENDING | PENDING | PENDING |
| C3 | **on** (`-fa`) | on (unset) | PENDING | PENDING | PENDING |

C0 provenance: the 07-01 resident-serve note (23.23 / 23.22 tok/s two-run `print_timing`,
fa not passed). Its CUDA-graph state was **not recorded** at capture — a clean re-run of C1
(fa off × graph off) plus C0's flag set pins the baseline's true cell from the load log.
Headline to report on the issue: the best decode tok/s cell and its multiplier over C0.

**Greedy first-token parity (the `Accept` parity check).** With temperature 0 and a fixed
prompt, capture the **first generated token id** in every cell (the `1st-token id` column).
The parity gate: **all four cells must emit the same first token id** (and match C0). A
flag/graph change that alters the greedy first token is a correctness regression, not a
speedup — fail the cell and report it, do not average it away (net-true-value standard,
[docs/standards/net-true-value.md](../standards/net-true-value.md)). Optionally extend to the
first-N greedy tokens for a stronger parity witness.

## 4. The acceptance gate this worker cannot reach

**Acceptance** = a recorded `experiments/benchmark/runs` artifact, `claim_class: WITNESSED`
(or OBSERVED), containing the §3 2x2 table filled for cells C1–C3 plus the greedy first-token
parity verdict.

**Blocker (host capability).** This worker runs on the Windows dev box, which has **no GPU**;
native `go test` is even OS-blocked here (AGENTS.md build/test notes). The A/B needs the
433.82 GiB GLM-5.2 checkpoint staged from NVMe and served resident across a GPU server-3
8-GPU datacenter server (640 GiB VRAM) node reached only through the operator-gated
`fak-private` control bridge ([docs/private-comms-channel.md](../private-comms-channel.md)).
Standing up the resident serve and re-launching it four ways is an **operator hardware
action**, not something this worker can run — and inventing the tok/s / TTFT / token-id cells
would violate the WITNESSED bar and the "no fabricated pass" rule. So the cells stay PENDING.

**Harness (now wired).** The four-way launch + bench + parity fold is automated by
[`tools/glm52_l4_fa_cudagraph_ab.sh`](../../tools/glm52_l4_fa_cudagraph_ab.sh): it stands the
resident serve up once per cell via the L4 toggles now in `glm52_mgpu_serve.sh` (`FLASH_ATTN=on|off`
adds/omits `-fa`; `CUDA_GRAPHS=off` sets `GGML_CUDA_DISABLE_GRAPHS=1` while `=on` **unsets** it — ggml
presence-checks that var, so any value including `0` disables capture and graphs-on must be unset),
benches each through
the L8 harness (`glm52_bench_lever.sh`), captures the greedy first-token per cell, and folds a
`fak.glm52-l4-fa-cudagraph-ab.v1` verdict (best-cell speedup over C1 + the cross-cell parity check).
The default `SPLIT_MODE=layer` re-pins the current serve's cells; `SPLIT_MODE=row` is the real
post-L1 test (§2/§5). The unset serve invocation stays byte-identical to the 23.2 tok/s baseline
(fa omitted, graphs at build default), so this adds the knobs without moving the WITNESSED number.

**Smallest next step.** An operator on **GPU server 3, Lane A** runs
`SPLIT_MODE=row DEVICES=0,1,2,3,4,5,6,7 bash tools/glm52_l4_fa_cudagraph_ab.sh` on the resident
node (four cold loads, ~30-45 min), confirms the parity verdict is `all_cells_match: true`, and
writes the artifact under:

```
experiments/benchmark/runs/by-machine/<gpu-server-3-node-id>/<UTCstamp>-glm52-l4-fa-cudagraph/
  manifest.json   # $schema: benchmark/run-manifest.v1; machine_id, timestamp, git rev/branch/dirty,
                  #   model {name: GLM-5.2 UD-Q4_K_M, precision: served}, config.claim_class: WITNESSED, scrubbed: true
  result.json     # the 2x2: one row per cell {fa, graph, decode_tok_s, ttft_ms, first_token_id, parity_ok}
                  #   + a `parity` block: {all_cells_match: bool, first_token_id, baseline_cell: "C0"}
  RESULTS.md      # the §3 table filled + the parity verdict + the load-log flash-attn/graph lines per cell
```

A `(fak docs)` (or `(fak <run-leaf>)`) commit citing `#3076` that adds that artifact path is
what closes #3076 — **not this triage note**.

> **Do not auto-close #3076 on this note.** This is a triage + harness increment; the
> benchmark acceptance (§4) is unmet. If a close-resolved / close-batch arm binds a #3076
> reference here, treat it as a false close and reopen until the WITNESSED 2x2 + parity
> artifact lands.

## 5. Cohort note

The whole #3073 lever cohort (#3076 L4, #3077 L5, #3078 L3, #3079 L2, #3080 KV-paging) is the
same shape — one lever, one WITNESSED `experiments/benchmark/runs` artifact. Siblings #3077
(L5 quant), #3079 (L2 cont-batch), and #3080 (KV-paging) already carry triage notes under
`docs/notes/`; this note triages the assigned leaf, **#3076 (L4)**, and shares their GPU server node
witness blocker. L4's result is **ordered after L1** (#3075): the graph axis is expected to be
inert until the layer-split bubble is gone, so a near-1.0× 2x2 on the *current* serve is an
informative negative, and the lever's real test is a re-run post-L1.

*Companions:* [ceiling + lever map](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md) ·
[8-GPU resident serve (23.2 tok/s WITNESSED)](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md) ·
[L5 quant sweep triage (#3077)](GLM52-L5-QUANT-SWEEP-TRIAGE-2026-07-06.md) ·
[generation contract](../generation.md).
