# LiveCodeBench — GPU-server next steps — 2026-07-13

**Objective:** determine the concrete next action to advance the LiveCodeBench
campaign (#2085 / #3060) using the available GPU-server classes, and state honestly what is
executable now versus what is blocked.

**Operator:** autonomous benchmark loop. **Session limitation:** this session ran
from the Windows dev box without a ready private-control session, so no GPU-server
command was driven from here. Everything below is a verified-state synthesis plus a
transport-agnostic recipe for an authorized on-box operator.

## Verified state (not fabricated — read from the repo + today's sibling artifacts)

| Fact | Value | Source |
|---|---|---|
| LCB CLI arms shipped | `fetch`, `raw`, `fak`, `ab`, `export`, `contract`, `report`, `preflight` | `cmd/livecodebench/{main,raw,fak}.go` |
| Official grader runs end-to-end over **real** `release_v2` problems | pass@1 = 0.5 on 2 correct / 2 wrong hand-authored solutions | `experiments/livecodebench/glm52-run/RUN-LOG.md` (#3060) |
| The gap that grader run left open | generation half used **hand-authored** code — no model endpoint was reachable | same RUN-LOG, "honest boundary" |
| Real GLM-5.2 Q4 in-kernel serve **works on the 40 GB GPU server** | `planner=inkernel`, GLM-5.2 Q4_K_M, `:8090`, witnessed | `issue-1012-...-glm52-gpu-server-20260713.md` |
| 40 GB GPU-server decode speed | **~0.2 tok/s** (154-tok prefill = 720 s) under `--cpu-offload-experts` | #1012 per-turn table |
| Why offload is forced on that class | model resident ≈ 427 GB > 320 GB aggregate VRAM (8× **40** GB GPUs) | #1012 (resident 437,273 MiB) |
| 80 GB GPU-server alternative | 8× **80** GB GPUs = **640 GB**, powered up and **idle**, no serve resident | `GPU-SERVER-BENCH-RECON-2026-07-13.md` |

**Bottom line:** the LCB pipeline is fully wired and the official grader is proven
over real problems. The **only** missing evidence is *generations produced by a
real served model on real LCB problems, then officially graded*. The submission
gate (`LIVECODEBENCH-SUBMISSION-PACKET.md`, #2115) stays `BLOCKED_PRECREDENTIAL`
until that exists.

## The fork

Producing real served-model generations needs a serve fast enough for the target
problem count. That splits into two distinct next actions:

### Option A — bounded wiring-witness (cheap, closes the #3060 gap)

Prove the **one** unproven seam — *real served-model generations → official
grade* — without paying the GLM offload wall. Point the `fak` arm at **any fast
on-GPU serve that fits in a single 40 GB card** (a small model, decode at normal
speed) over a **tiny real release-pinned slice**, then grade with the official
evaluator.

```bash
# 1. tiny real slice (network fetch of a pinned release), e.g. 3 problems
fak livecodebench fetch --release-version release_v6 --scenario codegeneration \
    --fetch --limit 3 --out suite.json

# 2. fak arm against a fast on-GPU serve (endpoint = whatever is served)
fak livecodebench fak --suite suite.json --model <served-id> \
    --endpoint http://127.0.0.1:8090/v1 --n 1 --temperature 0 \
    --max-tokens 512 --out fak-report.json

# 3. grade via the official custom-evaluator (handoff pinned by `contract`)
fak livecodebench export --format custom-evaluator --out custom_output.json ...
python -m lcb_runner.runner.custom_evaluator \
    --custom_output_file custom_output.json --release_version release_v6
```

This is **not** a publishable pass@1 (too few problems; `result_claim_allowed`
stays false). It is the honest next datum: *the full fak→official-grade pipe run
with a real model on a real GPU.* **Fastest way to retire the last unproven seam.**

> Note the export seam: `livecodebench export` currently reads a fixture-shaped
> file, not the `fak`-arm report directly. Confirm/adapt the report→custom-evaluator
> binding on-box before trusting step 3; the exact grading command is pinned by
> `fak livecodebench contract`.

### Option B — publishable pure-kernel GLM-5.2 number (target the 80 GB class)

A leaderboard-grade pure-kernel run needs hundreds of problems × n samples. On
the **40 GB GPU server** that is impractical: at ~0.2 tok/s a single ~600-token LCB prompt is
~45 min of prefill alone, so a real sweep is many days.

The offload wall is a **VRAM** problem, and the **80 GB GPU server does not have it**: 640 GB
aggregate holds the ~427 GB Q4 model fully on-GPU, so experts need not offload to
host RAM and decode should be far faster. That class is idle today. **Hypothesis to
verify on-box:** that the in-kernel CUDA backend can shard GLM-5.2 Q4 experts
across 8× 80 GB with no `--cpu-offload-experts` — confirm resident placement and
measured tok/s before committing a full sweep.

**Recommendation:** use the 40 GB class for the Option-A wiring witness and staging; the
publishable pure-kernel GLM-5.2 run belongs on the **80 GB class**.

## 40 GB GPU-server recipe (proven working in #1012)

```
fak serve --gguf <model-path>/GLM-5.2-Q4_K_M-00001-of-00008.gguf \
    --backend cuda --cpu-offload-experts --addr :8090
# gate before trusting anything: /healthz planner must == "inkernel"
```

fak v0.40.0 @ `471c3d9`, `go build -tags cuda` (libfakcuda.a sm_80 prebuilt).
Private transport commands, channel routing, and raw readback stay in `fak-private`;
the public witness begins at this generic on-box serve command and the `/healthz` gate.

## Honest boundary

No GPU-server command was executed this session (private readiness was unavailable). The results
scaffold cells (`pass@1`, `pass@5`, `Engine (pure-kernel arm)`) stay
`pending run` / `pending GPU run` and this artifact promotes nothing. It records
the verified pipeline state, the two-way fork, and ready-to-run recipes.
