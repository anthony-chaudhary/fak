---
title: "Ablate: Self-Ablation Feature Sweep (Deterministic Core)"
description: "The deterministic, $0, no-model half of the self-ablation benchmark harness (epic #607): one frozen tool-call trace replayed under N feature configs, deltas read straight off the kernel counters."
---

# ABLATE-RESULTS — the deterministic self-ablation sweep (`fak ablate`)

> This is **Regime A** of the self-ablation benchmark harness (epic
> [#607](https://github.com/anthony-chaudhary/fak/issues/607)): a *kernel-feature*
> ablation. It asks "what does feature X cost/save?" by varying **one `FAK_*`
> knob** while holding the model, the task, **and the tool calls** constant.
> That makes it the **exact-workload** twin of the cross-agent ablation (Regime B
> — pure fak vs Claude Code / ultracode), which varies the whole agent + model
> and can only be scored distributionally. The `Trace.WorkloadHash` equality guard
> that makes this regime ironclad is *exactly wrong* for Regime B; the two are
> **two harnesses sharing one record schema**, and Regime B is not claimed here.

## What shipped — both sweep rungs of Regime A

`fak ablate` generalizes the 2-arm `fak bench` (vDSO on/off) into an N-ARM
matrix: replay ONE frozen tool-call trace through each feature config and emit
one `AblationRun` per arm, every arm bound to the trace's single workload hash
by an N-arm identical-workload guard (`ablate.Report.Validate`, generalizing
`metrics.Report.Validate` from a fixed pair to N). The deltas are
apples-to-apples by construction — every arm ran the same work.

Both feature rungs are wired. **Rung 1** flips the one runtime-settable knob
(`vdso`) IN-PROCESS. **Rung 2** sweeps the env-gated levers — each read once at
process start, so every arm re-execs a child carrying that arm's `FAK_*` env
(`ablate.SweepViaSubprocess`, driven by the hidden `ablate-arm` child verb). A
mixed sweep routes wholly through rung 2, so the `vdso` arm and the env arms land
in one report under the same workload-hash guard. `BuildSweep` accepts every
token in the closed catalog (a typo still fails loud); the CLI picks the rung
from whether any swept feature is env-gated.

## The cache-lever catalog — the eight ablatable cache things

The sweepable **cache** levers are a closed set of eight `FeatureCard`s in
[`internal/ablate/catalog.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ablate/catalog.go)
— the single source of truth a live arm and the menu share, so "what is this
lever" is byte-identical to the classification a run reports. Print it with
**`fak ablate --list`** (`--list --json` for the raw cards):

| lever | preset | plane | component | fidelity | rung · env gate |
|---|---|---|---|---|---|
| `vdso` | — | `kernel_tool_cache` | vdso | lossless | 1 · in-process runtime knob |
| `radix` | `@local` | `local_kv` | inkernel_radix | lossless | 2 · `FAK_INKERNEL_RADIX` |
| `ctxplan_seam` | — | `context_view` | ctxplan_seam | recoverable | 2 · `FAK_CTXPLAN_SEAM` |
| `compressor` | `@context` | `context_compression` | headroom_compressor | recoverable | 2 · `FAK_COMPRESSOR` |
| `bp_plan` | `@wire-cache` | `provider_prompt_cache_control` | breakpoint_planner | lossless | 2 · `FAK_ABLATE_BP_PLAN` |
| `prefix_guard` | `@wire-cache` | `provider_prompt_cache_control` | prefix_guard | lossless | 2 · `FAK_ABLATE_PREFIX_GUARD` |
| `ttl_1h` | `@wire-cache` | `provider_prompt_cache_control` | ttl_1h | passive | 2 · `FAK_ABLATE_TTL_1H` |
| `uncached_trim` | `@wire-cache` | `provider_prompt_cache_control` | uncached_trim | lossy | 2 · `FAK_ABLATE_UNCACHED_TRIM` |

What each lever does to the cache:

- **`vdso`** — serves a repeated tool call from the in-kernel vDSO fast path without re-adjudicating or re-calling the engine.
- **`radix`** — reuses a shared KV prefix across turns via in-kernel RadixAttention (needs `engine=inkernel`).
- **`ctxplan_seam`** — serves cache-safe materialized context views through the ctxplan seam (needs `engine=inkernel`).
- **`compressor`** — sheds recoverable context through the headroom compressor to hold the cacheable prefix under budget.
- **`bp_plan`** — places provider cache breakpoints without changing model-visible prefix bytes.
- **`prefix_guard`** — guards prefix stability before relying on provider-cache economics.
- **`ttl_1h`** — changes provider cache retention only; model-visible prefix bytes are unchanged.
- **`uncached_trim`** — sheds or rewrites uncached context; prefix integrity covers only the guarded cacheable prefix.

**Presets** sweep a whole cache plane in one flag: `@wire-cache` → the four
`provider_prompt_cache_control` levers (`bp_plan,prefix_guard,ttl_1h,uncached_trim`),
`@local` → `radix`, `@context` → `compressor`. A new lever on an existing plane
joins its preset automatically (the preset is derived from `Plane`, not a hand list).

Not every sweepable feature is a cache lever: `normgate` / `ifc` / `gitgate` /
`wire_screen` / `wire_redact` / `toon_wire` are guard/wire knobs and deliberately
carry **no** card, so `--list` prints the cache subset, not all of `KnownFeatures`.

## The committed artifact — `tau2-airline-smoke`, vDSO on/off

The minimal first experiment the epic names: a deterministic feature-sweep over
one frozen `tau2` trace, committed as a reproducible artifact at zero cost.

**Artifact:** [`experiments/ablate/tau2-smoke-vdso-ablation.json`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/ablate/tau2-smoke-vdso-ablation.json)
**Reproduce:** `go run ./cmd/fak ablate --trace testdata/tau2/tau2-smoke.json --sweep vdso`

| arm | features | calls | vdso_hits | engine_calls | denies | quar | tokens | Δ tokens |
|---|---|---|---|---|---|---|---|---|
| all-off (baseline) | vdso=off | 12 | 0 | 12 | 0 | 0 | 937 | — |
| vdso | vdso=on | 12 | 7 | 5 | 0 | 0 | 417 | **−520** |

The vDSO fast path serves **7 of 12** repeated calls from cache, so only **5**
reach the engine (12 → 5 `engine_calls`); the 7 cached decisions carry a bounded
verdict instead of the full result, cutting **520 tokens** (937 → 417) from the
model context on this 12-call trace. `denies`/`quarantines` are 0 on both arms —
this read-heavy airline trace triggers neither the deny floor nor a quarantine,
so those counters are an honest zero (the vDSO is the only feature this trace
exercises).

## Reproducibility — what is and is not machine-bound

The **counter fields reproduce byte-identical** on any host:
`workload_hash`, `calls`, `vdso_hits`, `engine_calls`, `denies`,
`quarantines`, `input_tokens`, `output_tokens` — they are read from the kernel's
own deterministic event counters on a frozen trace, with no model and no decode.

The **timing fields are single-box and NOT cited as a headline**:
`p50_ns`, `p99_ns`, `mean_ns`, the latency `buckets`, and `wall_seconds` are
clock measurements on the build host (Windows, go1.26.3 here). They are
committed for completeness and fenced as illustrative, exactly the way the
model-ladder wall-clocks are single-box while the deterministic token/hit-rate
metrics reproduce exactly (see
[BENCHMARK-AUTHORITY.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md) §"The model-ladder thesis").

## Honesty fences (what this rung does NOT measure)

- **Provider-cache levers are harness gates, not live gateway flips (yet).** The
  four `@wire-cache` levers (`bp_plan` / `ttl_1h` / `prefix_guard` / `uncached_trim`)
  sweep through `FAK_ABLATE_*` envs that gate the arm's *measured shape*; a future
  live runner maps these tokens to real gateway setters, but the sweep vocabulary is
  stable now. The in-kernel levers (`radix` / `ctxplan_seam`) require
  `engine=inkernel` to move a counter — on the offline mock engine their arm reports
  an honest zero delta.
- **`--from-session` is in-process only.** A session-backed cassette replays the
  runtime `vdso` sweep; env-gated arms need a bench trace — `--from-session` refuses
  an env-gated sweep (usage exit 2) rather than silently measuring an unspawned child.
- **Cross-agent arm — now shipped (rung 3).** Regime B (pure fak vs Claude Code /
  ultracode) needs a live model + API tokens and a distributional validity contract
  (success-gate, N-run variance, model-named baseline, decomposed cache). That
  controller now exists — see [§Rung 3](#rung-3--the-cross-agent-controller-regime-b)
  below — as a SEPARATE harness (`tools/cross_agent_ablate.py`): it shares this
  record schema's decomposed-token discipline but deliberately does NOT use the
  `WorkloadHash` guard, because an external model emits different tool calls each run.
- **Two regimes, two numbers.** This document reports only kernel-efficiency
  (Regime A). It never blends into an agent-capability claim.

## Witness

`go test ./internal/ablate ./cmd/fak` — `TestSweep_VDSO_NArmGuardAndIsolatedDelta`,
`TestValidate_RefusesMismatchedWorkloadHash`, `TestBuildSweep_UnknownAndDuplicate`,
`TestAblateJSONReport`, `TestAblateUnknownFeatureUsageError`, and the catalog-visual
golden `TestAblateListHuman` / `TestAblateListJSON` (which pin `--list` to the closed
catalog, so the table above cannot drift from `catalog.go`). On a native-Windows
host run the suite under WSL (`./test.ps1`); `go build` / `go vet` are native.

# Rung 3 — the cross-agent controller (Regime B)

> **Regime B** of epic [#607](https://github.com/anthony-chaudhary/fak/issues/607)
> ([#623](https://github.com/anthony-chaudhary/fak/issues/623)): the *agent* ablation.
> It asks "what does the kernel cost/save in front of a real agent?" by running the
> SAME task through `claude_code` (bare `claude -p`) vs `claude_code+fak`
> (`fak manage -- claude -p`). The external model emits different tool calls each run,
> so the `WorkloadHash` equality guard does NOT apply — validity here is
> **distributional**, not exact-workload. The controller (`tools/cross_agent_ablate.py`)
> enforces a four-part contract: a **success-gate** (no "saved" number unless both arms
> completed AND succeeded), **N-run variance** (mean ± CI95 over K≥5 reps), a
> **model-named** baseline (kernel-efficiency is *refused* unless the model is held
> constant across arms), and a **decompose** into two numbers that are never blended.

**Artifact:** [`experiments/ablate/cross-agent-pong-opus.json`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/ablate/cross-agent-pong-opus.json)
**Reproduce (offline, from the embedded raw reps):** `python tools/cross_agent_ablate.py report --reps <reps.json>`
**Re-measure (live, costs API tokens):** `python tools/cross_agent_ablate.py run --task pong --k 5 --fak ./fak.exe`

## The measured run — `pong` task, K=5 reps/arm, `claude-opus-4-8`

The task: *create `RESULT.txt` containing exactly `PONG`* — a one-tool-call task with a
deterministic success check (`RESULT.txt.strip() == "PONG"`). Same OAuth account both arms,
single Windows host, single-shot sessions. Tokens are **decomposed, never summed**:

| arm | success | turns | output tok | fresh input | cache-read | cache-create | total ingested | adjudication |
|---|---|---|---|---|---|---|---|---|
| `claude_code` (baseline) | 5/5 | 2.0 | 126.8 ±13.97 | 2503 | 42 619 | 6 955 | 52 078 | — |
| `claude_code+fak` | 5/5 | 2.0 | 124.0 ±10.46 | 2144 | 72 506 | 6 413 | 81 063 | 5 allowed · 0 denied · 0 repaired · 0 quarantined |

*(± is CI95 over the 5 successful reps; tokens are per-rep means.)*

## The two numbers (never one)

- **kernel-efficiency (model held constant — `claude-opus-4-8` both arms).** The kernel
  hop is **output-token and turn neutral**: output ratio **0.98×** (saved ≈ 3 tok),
  turns ratio **1.00×**. But it is **not** total-token-neutral: total ingested context
  ratio **1.56×** (`−28 986` tok, a NEGATIVE "saved" = overhead, reported with its real
  sign). The decomposition shows where it goes — fresh input *falls* (2503 → 2144) while
  provider-cache-read *rises* (42.6k → 72.5k): the gateway hop reshapes the prompt-cache
  split, pushing more context into the (cheapest) cache-read tier. The `+fak` arm's five
  `Write` calls each crossed the capability floor and were **ALLOWED** (the hash-chained
  journal carries the witness).
- **agent-capability (model varies).** Success rate `1.0` on both arms. With both arms on
  one model the capability axis is ~held here, so kernel-efficiency is the live number;
  this axis becomes load-bearing once a `pure_fak` (different planner/model) arm is added.

## Honesty fences (what this run does NOT establish)

- **One tiny, tool-light task.** A single `Write` call ⇒ only ALLOW decisions; the deny /
  repair / quarantine counters are an honest **zero** because this task never proposes a
  blocked call. A denial-inducing and a multi-tool task are follow-on work.
- **Single host, single account, single-shot sessions.** The cache-split numbers reflect
  cold-prefix caching on one Windows box, not steady-state reuse across a long session;
  they are illustrative of the *shape* of the kernel hop's effect, not a fleet SLA.
- **Two regimes, two numbers — never blended.** The `1.56×` total-token figure is a
  resource (kernel) number with the model held constant; it is never reported as an
  agent-capability or quality multiple.

## Witness

`python tools/cross_agent_ablate_test.py` — 17 hermetic tests (no network, no `claude`,
no `fak`): the `session_audit` token decomposition + de-dup, the journal verdict counter
(ALLOW/DENY/TRANSFORM/QUARANTINE, VDSO_HIT separated), the CI95 math, the success-gate
(no saved number off a failed arm), the model-named refusal, and the two-number decompose.
Auto-gated as HERMETIC by `tools/gated_tool_tests.py`. The committed artifact regenerates
byte-identical offline via `report --reps` (the raw reps are embedded).
