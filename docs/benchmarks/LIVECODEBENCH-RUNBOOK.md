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
| In-kernel serving path | shipped for serving; LCB run residual | `fak serve --gguf --engine inkernel --backend <backend>` is the fak-owned model path; a full LCB run on it is still pending. |
| LiveCodeBench native suite/report schema | pending | #2087 through #2095. |
| All four LCB scenario adapters | pending | #2096 through #2099. |
| Official custom-evaluator export | pending | #2102; this runbook names the required JSON shape now. |
| fak-native CLI wrapper | pending | #2109 through #2111. |
| Honesty gates and authority promotion | pending | #2113, #2114, #2115. |
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

The fak export step must produce that shape for the exact saved generations:

```bash
fak livecodebench export-custom \
  --input "$LCB_OUT/fak/codegeneration" \
  --out "$LCB_OUT/fak-codegeneration-custom.json"
```

This is pending #2102. The promotion rule is already fixed: the exported JSON digest must
be recorded before grading, and the same file must be the one handed to `lcb_runner`.

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

## Honesty Links

- Epic: [#2085](https://github.com/anthony-chaudhary/fak/issues/2085)
- Result gate: [#2113](https://github.com/anthony-chaudhary/fak/issues/2113)
- Evidence class and promotion requirements: [#2114](https://github.com/anthony-chaudhary/fak/issues/2114)
- Authority row and submission gate: [#2115](https://github.com/anthony-chaudhary/fak/issues/2115)
- Results scaffold: [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md)
