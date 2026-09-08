<!-- fak-benchmark-key: q8bench-honest-correctness-and-distribution -->
# fix(q8bench): fail closed on missing oracles and compare like-for-like decode distributions

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12375

## Current state

`cmd/q8bench/main.go:256-279` initializes `correctnessOK` to true, ignores JSON decode errors, and leaves the gate true when `oracle.json` is absent, empty, malformed, or contains no prompts. Lines 322-349 then serialize `gate_argmax_exact_vs_hf_oracle: true`, print `argmax-exact=true`, and exit zero with zero checked positions.

The same command stores the minimum fak repetition in a field serialized as `per_token_median_ms` (`main.go:67-72,282-305`) and compares that best-case minimum with Hugging Face medians. A single lucky fak repetition can therefore produce both a mislabeled measurement and an inflated `beats_hf_*` verdict. Existing tests cover reducers but not the CLI gate/report contract.

## Parent context

Parent: #10193. Related verification-quality program: #3831.

## Why now

q8bench currently emits a correctness success with no oracle observations and uses a best-case statistic for the performance reward. Any new quantized optimization can inherit both false-positive gates unless the verifier is repaired first.

## Problem frame

- Priority: P0 correctness and benchmark-integrity defect.
- Centrality: Enabling (truthful native inference performance evidence).
- P1: advanced - a benchmark operator can no longer receive a successful correctness receipt without an oracle observation.
- P2: advanced - like-for-like statistics prevent overstated net performance.
- P3: preserved - both defects remain bounded to one command and one test package.
- P4: advanced - the real report and exit path fail closed and name the compared statistic.

## Core through-line

Parse a required, non-empty oracle -> execute at least one position check -> compute a declared decode distribution statistic for both fak and HF -> emit a report whose field names match the statistic -> return nonzero on missing correctness evidence or drift.

## Working spine

The q8bench oracle reader, decode aggregation, JSON report, and process exit decision form the single working spine.

## Blast radius and affected lanes

- Primary lane: `benchmark`.
- Affected surface: q8bench CLI output and its JSON schema fields.
- Unaffected lanes: model kernels, exporters, gateway, and other benchmark commands.

## Quarantined fallback mechanism

There is no success fallback for missing correctness evidence. Unreadable or invalid HF comparison files may remain explicitly unavailable, but they must not be converted into a win. Keep the change atomic with focused fixtures so trunk stays green.

## Scoped acceptance criteria

- [ ] Missing, unreadable, malformed, empty, and zero-prompt `oracle.json` inputs return nonzero and never emit an exactness pass.
- [ ] A valid non-empty oracle checks at least one position; `total_positions == 0` cannot pass.
- [ ] Fak and HF verdicts use the same named statistic and sample regime.
- [ ] No minimum is serialized in a field named median.
- [ ] JSON and human-readable verdict tests exercise the real report/exit decision.

## Gold-plating boundary

No model export, quant-kernel changes, new benchmark framework, hardware run, historical benchmark rewrite, or performance claim. Do not weaken the argmax-exact requirement.

## Done condition

q8bench cannot emit a correctness pass without a parsed non-empty oracle and observed positions, and every comparative decode verdict is computed from like-for-like, honestly named statistics.

## Witness

The deterministic test witness is the focused q8bench gate and report suite below.

### Non-Forgeable Witness

```text
go test ./cmd/q8bench -run '^TestQ8Bench_(OracleGateFailsClosed|DecodeVerdictUsesLikeForLikeStatistic)$' -count=1
```

Current-state read-back:

```text
rg -n "correctnessOK := true|json.Unmarshal|no oracle.json|per_token_median_ms|mkDec|beatsHF" cmd/q8bench/main.go
```

## Acceptance gate

`go test ./cmd/q8bench -run '^TestQ8Bench_(OracleGateFailsClosed|DecodeVerdictUsesLikeForLikeStatistic)$' -count=1`

## Closure binding

The resolving commit cites this issue in its subject and carries a `(fak benchmark)` trailer.

## Likely files

- `cmd/q8bench/main.go`
- `cmd/q8bench/q8bench_test.go`

## Lane

`benchmark`

## Dependencies and dedupe

Closed #2 validates real exported kernels but does not cover this CLI default-pass path. Closed #9526 only consolidates deterministic ID helpers. Exact searches for `gate_argmax_exact_vs_hf_oracle` and q8bench `per_token_median_ms` found no issue owning these defects.

## Expected steps

5

```routing
lane: benchmark
paths: cmd/q8bench/main.go, cmd/q8bench/q8bench_test.go
expected_steps: 5
```
