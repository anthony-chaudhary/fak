# Qwen3.8-27B quantization campaign contract

This directory freezes the comparable evidence contract for issues #8307–#8321.
It does **not** claim that an arm is production-ready merely because it loads.

## Bounded arms

`BF16`, official `FP8`, GGUF `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `IQ4_XS`,
activation-aware `AWQ INT4`, `GPTQ INT4`, and variable-bit `EXL2` are the ten
campaign arms. A substitution requires a dated issue decision and a corpus
version change.

## Evidence rule

A `fak.qwen38-quant-report/1` campaign report must bind immutable model,
checkpoint, artifact, tokenizer, and template hashes; quantizer, runtime, and
fak `module@rev`; argv-form launch command; hardware/software topology; context
and cache settings; exact device residency; and an explicit deny-fallback
policy. It retains failed trials and carries a raw-archive hash, stale-after
date, verdict, and rollback threshold.

Every one of the six workload families in `corpus.json` requires at least three
post-warmup repetitions. Quality is evaluated first. A failed quality trial can
produce `HOLD` or `EXCLUDE`, never `PROMOTE`; performance from such a run is not
a production gain.

### Qualification precedes comparison

This campaign qualifies each quant as a supported platform arm. It does not rank
one quant against another until both arms independently complete the API contract.
The first text request is therefore a canary: model identity must match, the
response must decode, and `usage.completion_tokens` must be a positive integer.
A zero-token or undecodable canary stops before the remaining 17 requests, emits
no campaign report, and names the serving-contract defect to fix. It is not a
`HOLD` data point and cannot feed cross-quant latency, throughput, cache, quality,
or promotion analysis.

`qwen38quant.Validate` repeats the same boundary for imported and hand-assembled
reports: a `CAMPAIGN` with no successful API completion is invalid even when its
verdict says `HOLD`. This prevents a downstream summary from converting an invalid
platform run into a quant-selection recommendation. After each arm qualifies,
cross-arm analysis may compare only like-for-like successful workloads; arm
support remains the primary outcome, while a frontier recommendation is optional
consumer analysis.

## Production soak

`qwen38campaign --soak` extends that campaign contract across at least three
finalists. Each finalist must run the frozen six-family corpus three times plus
the 30 exact-effect coding tasks built into `qwen38quantrun`. The soak also
captures context-pressure, malformed-request, cancellation, restart, and cache-
recovery outcomes with independent identity, residency, and no-fallback
readback. The report records coding latency/throughput, peak memory/power, cache
latency delta, per-arm archive hashes, an overall raw-archive hash, verdict, and
rollback threshold.

The file-backed adapter refuses inline API keys and missing observation or
lifecycle commands. Keys are named through `api_key_env`; command fields are
argv arrays and are never passed through a shell:

```console
qwen38campaign --soak --config soak.json --corpus docs/benchmarks/qwen38-quant/corpus.json --report report.json --archive archive.json
```

A restart command must drain the server's entire process group and prove the
old endpoint is down before starting its replacement. Waiting for only the API
parent PID can leave tensor-parallel workers alive; a readiness probe may then
mistake the stale endpoint for the replacement and corrupt the cache-recovery
witness.

`qwen38quantrun.ValidateSoakReport` rejects fewer than three unique arms, task
drift, incomplete scenario readback, invalid campaign reports, missing metrics
or hashes, and a `PROMOTE` verdict without a fully passing selected arm.
`ValidateSoakArtifacts` additionally binds the supplied raw archive, every arm
archive, and every embedded frozen-corpus campaign to those hashes and results.

## Existing evidence

The 2026-08-19 official FP8 TP2 and native CUDA Q4_K_M artifacts remain useful
acceptance witnesses. Each contains one text, one JSON, and one tool task, not
three repetitions of six workloads. Importers therefore type both as
`HOLD/ACCEPTANCE_ONLY`; neither passes campaign validation.

`qwen38quant.Selfcheck` proves that boundary and proves refusals for corpus
drift, missing immutable hashes, ambiguous fallback, fewer than three repeats,
zero successful API completions, and promotion attached to failing quality:

```console
go test ./internal/qwen38quant
```

The hardware issues may emit reports only after consuming this corpus version.
The final frontier (#8320) may join only validator-clean `CAMPAIGN` reports.
