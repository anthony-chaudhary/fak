---
title: "LiveCodeBench runbook"
description: "The runbook for running LiveCodeBench against fak: environment, adapter status, and the steps needed to produce a scored, citable submission."
---

# LiveCodeBench Runbook

Status: **runbook assembled; native fak adapter pending child issues**. This document is
the command path and evidence contract for running LiveCodeBench through fak without
turning a dry run into a score claim.

Upstream harness: [LiveCodeBench](https://github.com/livecodebench/livecodebench).
Epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085).
Results page: [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md).

## Shipped Vs Residual

| Piece | State | Evidence / residual |
|---|---|---|
| OpenAI-compatible fak gateway | shipped | `fak serve` exposes `/v1/chat/completions`; LCB can target an OpenAI-compatible endpoint once model-style plumbing is configured. |
| In-kernel serving path | shipped for serving; LCB run residual | `fak serve --gguf --engine inkernel --backend <backend>` is the fak-owned model path; the pure-kernel codegen arm (§3a) drives LCB codegen through it with no external engine, but a graded pass rate stays `pending GPU run`. |
| LiveCodeBench native suite/report schema | pending | #2087 through #2095. |
| All four LCB scenario adapters | pending | #2096 through #2099. |
| Official custom-evaluator export | shipped | #2102; `go run ./cmd/livecodebench export --format custom-evaluator` (backed by `internal/livecodebench.WriteCustomEvaluatorInput`; not yet wired through the `fak` front door, tracked with the pending CLI wrapper below), pinned by `TestCustomEvaluatorItemsFixtureRoundTrip` and `TestRunExportCustomEvaluatorWritesGradeableInput`. |
| fak-native official-run contract | shipped | #2110; `go run ./cmd/livecodebench contract` emits the result-claim-gated two-arm run contract (raw `lcb_runner` vs fak-native), pinning constants + the official grading handoff, `result_claim_allowed` always false. Backed by `internal/livecodebench.BuildOfficialRunContract`, pinned by `TestOfficialRunContract*` and `TestRunContractWritesGatedOfficialRunContract`. |
| fak-native CLI wrapper (generate) | pending | #2109, #2111 (the `fak livecodebench generate` arm the contract references). |
| Evidence class + promotion requirements | shipped | #2114; every report artifact stamps `evidence_class` plus the `promotion_requirements` checklist (`internal/livecodebench/promotion.go`), pinned by `TestNewReportEnumeratesPromotionRequirements` and `TestMarkOfficiallyGraded`. See "Evidence Classes And Promotion Requirements" below. |
| Honesty gates and authority promotion | pending | #2113, #2115. |
| Results scaffold | shipped | [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md); all score cells remain `pending run`. |

The honest residual: until #2102 and #2113 land, a local LCB smoke can prove wiring but
cannot promote a fak result. A reportable result requires official LiveCodeBench grading
of saved generations plus the release/date-window identity recorded on the results page.

## Constants For A Run

Choose these before generating anything and keep them identical for raw and fak arms:

```bash
export LCB_RELEASE=release_v6
export LCB_START_DATE=YYYY-MM-DD
export LCB_END_DATE=YYYY-MM-DD
export LCB_MODEL=<model-name>
export LCB_OUT=experiments/livecodebench/<run-id>
```

`LCB_RELEASE` must be explicit. Do not rely on `release_latest` in a published result.
The start/end dates are the contamination window; if the model training cutoff is unknown,
carry that as a residual instead of weakening the window.

## 1. Install The Official Harness

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git external/LiveCodeBench
cd external/LiveCodeBench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

The official runner is the scoring authority. fak may generate or route completions, but
the promoted score comes from `lcb_runner`.

## 2. Start A fak Gateway

Proxy a hosted or separately served OpenAI-compatible model:

```bash
fak serve \
  --provider openai \
  --base-url "$UPSTREAM_OPENAI_BASE_URL" \
  --model "$LCB_MODEL" \
  --addr 127.0.0.1:8080
```

Or serve a local GGUF through the in-kernel path:

```bash
fak serve \
  --gguf /srv/models/<model>.gguf \
  --engine inkernel \
  --backend cuda \
  --addr 127.0.0.1:8080
```

The in-kernel command is the fak-owned serving path. A result from the proxy command is
still useful, but it must be labeled as a gateway/adjudication run, not as native model
throughput or native model quality.

## 3. Generate For Each Scenario

Upstream scenarios and commands:

```bash
# Code generation
python -m lcb_runner.runner.main \
  --model "$LCB_MODEL" \
  --scenario codegeneration \
  --release_version "$LCB_RELEASE"

# Self-repair; requires prior code-generation samples.
python -m lcb_runner.runner.main \
  --model "$LCB_MODEL" \
  --scenario selfrepair \
  --codegen_n <num-codes-from-codegeneration> \
  --n 1 \
  --release_version "$LCB_RELEASE"

# Test output prediction
python -m lcb_runner.runner.main \
  --model "$LCB_MODEL" \
  --scenario testoutputprediction \
  --release_version "$LCB_RELEASE"

# Code execution
python -m lcb_runner.runner.main \
  --model "$LCB_MODEL" \
  --scenario codeexecution \
  --release_version "$LCB_RELEASE"

# Optional code-execution chain-of-thought mode
python -m lcb_runner.runner.main \
  --model "$LCB_MODEL" \
  --scenario codeexecution \
  --cot_code_execution \
  --release_version "$LCB_RELEASE"
```

For the fak arm, the pending native wrapper should preserve the same scenario names and
emit saved generations without grading them:

```bash
fak livecodebench generate \
  --gateway http://127.0.0.1:8080/v1 \
  --model "$LCB_MODEL" \
  --release-version "$LCB_RELEASE" \
  --scenario <codegeneration|selfrepair|testoutputprediction|codeexecution> \
  --start-date "$LCB_START_DATE" \
  --end-date "$LCB_END_DATE" \
  --out "$LCB_OUT/fak/<scenario>"
```

That command is the target CLI contract for #2109 through #2112. Until it exists, any
manual OpenAI-wire run must record the exact adapter/model-style patch used to aim
LiveCodeBench at `http://127.0.0.1:8080/v1`.

## 3a. Pure-Kernel Codegen Arm (No External Engine In The Path)

The pure-kernel story — the LCB analogue of the
[SWE-bench pure-kernel runbook](SWEBENCH-PURE-KERNEL-RUNBOOK.md): the `codegeneration`
scenario is served by fak's **own** decode, with **no** SGLang / vLLM / hosted proxy in
the path. What makes it pure-kernel is the **absence of `--base-url`** on `fak serve` —
the model streams from fak's native forward pass, not a relayed upstream endpoint.

```bash
# Serve codegen from fak's own decode (pure-kernel: no --base-url, no external engine).
FAK_Q4K=1 fak serve \
  --gguf /srv/models/<coder-model>-q4_k_m.gguf \
  --engine inkernel \
  --backend cuda \
  --addr 127.0.0.1:8080

# Generate the codegeneration scenario against the in-kernel gateway (fak arm).
fak livecodebench generate \
  --gateway http://127.0.0.1:8080/v1 \
  --model "$LCB_MODEL" \
  --release-version "$LCB_RELEASE" \
  --scenario codegeneration \
  --start-date "$LCB_START_DATE" \
  --end-date "$LCB_END_DATE" \
  --out "$LCB_OUT/fak-inkernel/codegeneration"
```

Pin the arm in the machine-readable run contract, which records the engine alongside the
backend and marks the arm pure-kernel:

```bash
go run ./cmd/livecodebench contract \
  --release-version "$LCB_RELEASE" \
  --scenario codegeneration \
  --start-date "$LCB_START_DATE" \
  --end-date "$LCB_END_DATE" \
  --model "$LCB_MODEL" \
  --engine inkernel \
  --serving-backend "cuda q4_k_m" \
  --out "$LCB_OUT/fak-inkernel/contract.json"
```

That emits `constants.engine = "inkernel"`, `pure_kernel: true`, and
`pure_kernel_result_status: "pending GPU run"`, with `result_claim_allowed: false`.

The saved generations then flow through the same official grading handoff (§4–§5). Record
`engine=inkernel` and the serving `backend` on the [results page](LIVECODEBENCH-RESULTS.md)
alongside the generation-artifact digest.

The honest fence, identical to the SWE-bench pure-kernel arm: the device kernels are
argmax-exact against the CPU reference and `fak serve --engine inkernel` is a landed,
tested serving path. A bounded 8-problem CPU-reference run through the pure-kernel arm
was officially graded on 2026-07-14 (pass@1 `0.0`; see the results ledger), proving the
live generation-to-official-evaluator wiring. The full-window `pass@1` / `pass@5` cells
stay pending until the complete pinned release/date window runs over the intended device
backend. Do not promote a bounded wiring witness as a full benchmark result.

## 4. Export Custom-Evaluator Input

The official custom evaluator expects one row per benchmark problem:

```json
[
  {
    "question_id": "example-id",
    "code_list": ["candidate 1", "candidate 2"]
  }
]
```

The fak export step produces that shape from a fixture holding the saved generations
(`question_id` + `code_list` per item, order preserved):

```bash
go run ./cmd/livecodebench export --format custom-evaluator \
  --fixture "$LCB_OUT/fak-codegeneration-fixture.json" \
  --out "$LCB_OUT/fak-codegeneration-custom.json"
```

Shipped in #2102 (`internal/livecodebench.WriteCustomEvaluatorInput`). The promotion rule is
already fixed: the exported JSON digest must be recorded before grading, and the same file
must be the one handed to `lcb_runner`.

## 5. Grade With The Official Evaluator

For direct upstream runs, add `--evaluate`:

```bash
python -m lcb_runner.runner.main \
  --model "$LCB_MODEL" \
  --scenario codegeneration \
  --evaluate \
  --release_version "$LCB_RELEASE"
```

For fak-saved generations, use the custom evaluator:

```bash
python -m lcb_runner.runner.custom_evaluator \
  --custom_output_file "$LCB_OUT/fak-codegeneration-custom.json"
```

Then compute the date-windowed score from the saved evaluation artifact:

```bash
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file "$LCB_OUT/<official-eval-all-file>" \
  --start_date "$LCB_START_DATE" \
  --end_date "$LCB_END_DATE"
```

Only this official grading handoff can fill the `pass@1` and `pass@5` cells in
[LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md).

## 6. Record The Result

Update [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md) with:

- `release_version`, scenario, start date, end date.
- model identity, serving backend, and model training-cutoff statement or residual.
- raw-arm generation artifact digest.
- fak-arm generation artifact digest.
- official grading command and output artifact digest.
- evidence class from #2114 and authority status from #2115.

Do not copy a score into another doc until #2113's `result_claim_allowed` gate agrees
that official grading happened over the recorded generations.

## Evidence Classes And Promotion Requirements (#2114)

Every LiveCodeBench report artifact stamps an `evidence_class` and the
`promotion_requirements` checklist, copying the terminalbench honesty pattern
(`internal/livecodebench/promotion.go`):

- `fixture-smoke` — a committed-fixture run; LCB's spelling of terminalbench's
  `SIMULATED_LOCAL_FIXTURE`. Stamped by the fixture smoke and `run` reports.
- `local-ungraded` — real generations, no official grading; pass-like numbers are
  local signals only. Stamped by `NewReport` and the scenario scorers.
- `official-lcb-runner-graded` — the only class that can carry
  `result_claim_allowed: true`. Stamped exclusively by `MarkOfficiallyGraded`,
  which refuses a report with no graded arm results and re-validates on promotion.

The promotion requirements enumerate what must be recorded before a local number
may be promoted to a claimable score, and ride on every report verbatim:

- `problem-ids-pinned-and-identical-across-arms`
- `release-version-and-date-window-recorded`
- `both-arms-generations-saved-with-digest`
- `official-lcb-runner-grader-output-recorded`
- `same-config-across-arms`

`Report.Validate` enforces the fence: pass-rate fields require a named evidence
class, and `result_claim_allowed` requires `official-lcb-runner-graded`.

## Honesty Links

- Epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085)
- Result gate: [#2113](https://github.com/anthony-chaudhary/fak/issues/2113)
- Evidence class and promotion requirements: [#2114](https://github.com/anthony-chaudhary/fak/issues/2114)
- Authority row and submission gate: [#2115](https://github.com/anthony-chaudhary/fak/issues/2115)
- Results scaffold: [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md)


## Deterministic arm-report export

A completed `raw` or `fak` generation arm is already the authoritative source of
`question_id` plus ordered completions. Convert that report directly to the
upstream custom-evaluator input without hand-editing or another model call:

```bash
go run ./cmd/livecodebench export \
  --from-report experiments/livecodebench/glm52-run/raw-report.json \
  --format custom-evaluator \
  --out custom-output.json
python -m lcb_runner.runner.custom_evaluator \
  --custom_output_file custom-output.json \
  --release_version release_v6
```

The export fails closed if the report is not a `raw`/`fak` arm, omits its pinned
release, contains duplicate or empty question IDs, or has missing/empty
completions. Export does not itself permit a score claim; the official evaluator
result remains the grading witness.
