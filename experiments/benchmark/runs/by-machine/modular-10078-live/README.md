# Qwen3.8 native capacity capture - invalid qualification

Issue: https://github.com/anthony-chaudhary/fak/issues/10078

This is a **hardware-witnessed negative result**, captured on 2026-09-10. It does
not certify a throughput peak, saturation point, SLA knee, or repeatable latency
capability. Read `qualification.json` before consuming `receipt.json`.

## What was witnessed

The service executed the fixed workload with Qwen3.8-27B-Q4_K_M through fak's
native Vulkan path. Non-streaming identity probes before and after the sweep
reported no fallback. The executable, model file, service process/start time, and
kernel-selection digest remained unchanged.

- Native source: `fak@d3dd8447de3707afbc05bc450812cbce3858b66b`.
- Capture library source: `fak@6e1c5eb7c4c998211ba6764b7314e3eff0682594`.
- Model and executable SHA-256s, clocks, governor mode, and temperatures are in
  `snapshot-before.json` and `snapshot-after.json`.
- One fixed workload of 32 independent one-turn requests was submitted at each
  concurrency: 1, 2, 4, 8, 16, and 32. The maximum output allowance was 16 tokens.
- The first five points each returned 32 successful, exact-output responses.
  Concurrency 32 returned 16 exact-output HTTP 200 responses and 16 HTTP 429
  refusals. No automatic retry, session reset, policy change, or server deployment
  was performed.

## Why this does not qualify

The pre-run manifest described the downstream native scheduler's capacity as
256. A source and live control-plane readback found an **earlier per-arm limit
of 16**. The shared task prefix maps these otherwise independent sessions into
one research arm. The 32-request point exceeded that effective envelope and
must not support a peak or knee. The unchanged native scheduler refusal
counters do not describe refusals at the earlier gate.

`effective-capacity.json` contains only this task's scrubbed arm observation.
The immutable pre-run snapshot also omitted message role bytes in its reservation
estimate. `capacity-readback.json` corrects that estimate from 32 to 35 tokens
per request; the downstream scheduler capacity calculation remains 256.

Total elapsed setup from the goal's start exceeded the measurement duration.
`launch.json` and `qualification.json` report both, including audit, historical
evidence correction, resource/trust investigation, worktree preparation, and
recipe construction. This is observed task setup cost, not a claimed minimum
cost of a future rerun.

The original sweep library labeled the partially refused point valid because it
had positive throughput from successful responses. The accompanying software fix
rejects any point containing failed requests with `measurement_incomplete`.
The original captured receipt is retained as diagnostic evidence; only its
absolute local artifact path was removed. Its original maximum is right-censored,
and the claim gate refused it. The software readback is in
`software-validation.json`.

## Measurement limits

The stream returned one content event per successful response. The recorded
first-content-event and end-to-end distributions include P50/P90/P95/P99;
inter-token latency and TPOT are explicitly unmeasured. There was one run per
coordinate, with no repeated-run throughput confidence interval or quantile
confidence interval measured. These values are diagnostic observations, not a
qualified performance claim. Every successful response matched exactly
`red green blue yellow`; all response usage counts and quality results are in
`quality.json`.

## Reproduction and verification

`capture.go` is the exact bounded capture recipe. It uses the existing
`webbench.RunServingSweep` client extension, requests streaming usage, and adds
a distinct trace for every independent one-turn request. Native inference
receipts are collected separately because the service rejects native-receipt
requests in streaming mode.

The endpoint and bearer credential are supplied privately through
`MODULAR_BENCH_ENDPOINT` and `MODULAR_BENCH_API_KEY`. The recipe accepts
`--model`, `--engine-receipt`, `--capacity`, `--session`, `--out-dir`,
and `--max-seconds`. It retains the original 1-32 range to reproduce this
negative result; do not interpret that range as an admitted-capacity claim.

No server settings were changed to make the benchmark pass. A future qualifying
capture must discover every earlier admission limit before choosing its range,
remain within the effective limit, preserve raw latency samples for uncertainty
estimates, and account for setup cost.

Focused software regression:

```text
go test ./internal/webbench -run 'ServingSweep.*Partial' -count=1
go test ./internal/webbench -count=1
```

The disconfirming-witness branch of #10078 is fulfilled by the real capture and
typed invalid qualification. The positive peak/knee acceptance remains
unestablished; this directory makes no such claim.


Offline readback through the current library:

```text
go run ./experiments/benchmark/runs/by-machine/modular-10078-live/readback.go ./experiments/benchmark/runs/by-machine/modular-10078-live/receipt.json
```
