# bench — CPU + memory stress witness (`cpumemstress`)

Dependency-free, bounded CPU + memory stress arm added for
[#4371](https://github.com/anthony-chaudhary/fak/issues/4371) (gap found during the
24h CPU server resource campaign, [#4367](https://github.com/anthony-chaudhary/fak/issues/4367)):
`stress-ng` is not installed on CPU server and the campaign may not install packages or
take a privileged config. This arm stresses CPU and memory using **only the Go
standard library** (`crypto/sha256` for CPU, a page-touched byte slab for memory),
so it runs wherever `go test` or the `fak` binary already runs — no apt, no root,
no sysctl. It is an **operator stress witness, not a CI floor**: numbers land
`OBSERVED` with the host named, and it asserts/flips no gate.

Source of truth: `cpumemstress.go` (package doc + the `StressNGGuidance` constant).
This README is the discoverable entry point; it does not restate the guidance,
which ships embedded in every artifact (`stress_ng_when`) so it cannot drift.

## Run it on CPU server (produce the witness for #4367)

The full-size arm is env-gated behind `TestWriteCPUMemArtifact` so a bare
`go test` stays fast; set `FAK_CPUMEM_OUT` to run it and write the JSON witness:

```sh
FAK_CPUMEM_OUT=/tmp/cpumem-da33.json FAK_BENCH_HW=da33 \
  go test ./internal/bench -run TestWriteCPUMemArtifact -count=1
```

`FAK_BENCH_HW` names the host in the artifact (unset → `unspecified`). Attach the
resulting `/tmp/cpumem-da33.json` to #4367. A safety-gate refusal is still a valid
recorded outcome: the run writes `refused: true` with `refused_why` rather than
fabricating a pass.

A zero-config `CPUMemConfig{}` is a valid full CPU server run. Defaults
(`withDefaults`): concurrency `runtime.NumCPU()`, CPU duration `2s`, block `1 MiB`,
slabs `64 MiB / 256 MiB / 1 GiB`, chunk `16 MiB`, timeout `CPUDuration + 60s`.

## What it records (the #4371 field map)

| Done-condition field | Artifact field |
| --- | --- |
| exact duration | `cpu.duration_ns`, `memory[].alloc_ns` / `touch_ns` |
| concurrency | `cpu.workers` |
| allocation size | `memory[].requested_bytes` / `touched_bytes` |
| throughput | `cpu.hashes_per_sec` / `throughput_mb_s`, `memory[].throughput_mb_s` |
| latency distribution | exact `p50/p90/p99/max/mean_ns` over raw samples (not buckets) |
| host load/mem before+after | `host_before` / `host_after` (`/proc/loadavg` + `/proc/meminfo` on Linux; `runtime.MemStats` always) |
| artifact digest | `digest` — SHA-256 of the canonical report with `digest` cleared (self-verifying) |

## Safety gates (bounded, no runaway)

- **load** — refuses to start if `/proc/loadavg` 1-min load per CPU exceeds
  `MaxLoadPerCPU` (default `4.0`).
- **temperature** — best-effort hottest `/sys/class/thermal/thermal_zone*/temp`
  (world-readable, no root); refuses over `MaxTempMilliC` (default `90000` = 90 °C).
  Degrades to an honest `checked:false` where no zone is exposed.
- **timeout** — the whole run is context-deadline bounded (`Timeout`), and each arm
  is duration/size bounded.
- **memory cap** — total slab is capped at `MaxMemFraction` of `MemAvailable`
  (default `0.5`); oversized slabs are **skipped and recorded**, so a 1 GiB request
  cannot OOM a smaller box.

On non-Linux hosts the load/temperature/mem-cap reads are unavailable, so those
gates record `checked:false` (an honest unchecked, not a guessed pass) — they are
live on CPU server/Linux. The two `/proc`-dependent gate tests skip on Windows for the
same reason.

## When to use this vs `stress-ng`

Canonical guidance is the `StressNGGuidance` constant in `cpumemstress.go` (shipped
in every artifact as `stress_ng_when`). In short: **use this arm** when the host
forbids package installation or privileged config (the CPU server case) and you need a
bounded, self-verifying CPU+memory witness that runs anywhere `go test` runs.
**Prefer `stress-ng`** when it *is* installed and you need its breadth — dozens of
stressor methods (VM, cache, I/O, NUMA, thermal), `--cpu-method` micro-kernels, or
matched cross-fleet numbers against an external baseline. This arm is intentionally
two stressors (hash + page-touch): the always-available floor, not the ceiling.

## Witnessed run (CPU server → #4367)

The bounded arm was run on **CPU server** (256 CPU, ~1 TiB RAM, Go 1.26.1) via the recipe
above and the self-verifying artifact was attached to
[#4367](https://github.com/anthony-chaudhary/fak/issues/4367#issuecomment-4981247822).
The run was **not** refused — the load gate was live (`checked:true`, 0.03/CPU) and
the temperature gate degraded to an honest `checked:false` (no `/sys/class/thermal`
zone exposed), exactly as designed. Recorded highlights:

- CPU: 256 workers, 505,513 SHA-256 ops in 2.001 s = **252,591 hashes/s**; latency
  p50 945 µs / p99 1.455 ms.
- Memory: 64 MiB / 256 MiB / 1 GiB slabs all page-touched (none skipped), up to
  ~252 GB/s; host load1 moved 7.79 → 28.07 (real synthetic load applied).
- Artifact digest (canonical report, `digest` cleared):
  `5630e5ffd7fa49ce7f4d87f4785c0a9855f855aabb6e8fcf4020e8c36a481fcd`; file SHA-256
  `1fa3831d4a1d04ad2197a77240e7968d1a782875ec0155b8a9abfd459fd0de43` (4345 bytes).

This distinguishes the native-Go arm from the earlier Python-threaded (GIL-bound)
and OpenSSL `speed` witnesses on #4371 — those measured different things; this is the
dependency-free `cpumemstress` tool itself.

## Lane gate

```sh
go test ./internal/bench -count=1
go vet ./internal/bench
```

### Negation naive baseline

`negbaseline` scores auditable matched affirmative/negated cloze and QA probes through
OpenAI-compatible `fak serve` endpoints. The offline fixture is the default test; live
capture is opt-in:

```bash
FAK_NEGBASELINE_OUT=/tmp/negbaseline.json \
FAK_NEGBASELINE_TARGETS=/tmp/negbaseline-targets.json \
  go test ./internal/bench -run TestNegBaselineCapture -count=1 -v
```

The targets file is a JSON array of `{url, model, parameters, host, surface}` objects.
This is an OBSERVED baseline witness, not a CI gate: it reports plain accuracy,
negated accuracy, their gap, and whether the endpoint gaps widen with model size; it
never asserts that the literature's inverse-scaling signature must reproduce. The
committed witness in `testdata/negation_baseline_observed.json` names the fak-served
models and host used for the run.

### Negation comprehension suite (`negbench`, #4468)

`negbench` is the standing negation-comprehension witness for the negation-operator
epic (#4402): it measures whether negation is *understood*, not merely rephrased,
across four dependency-free (Go stdlib only) task families, each an embedded fixture
set plus an exact, deterministic scorer:

- **negated cloze** — the correct completion depends on the negation
  ("A penguin cannot ___"); a positive-only model falls for the `forbidden` trap word.
- **negated QA** — yes/no questions whose answer flips under negation.
- **do-not / only adherence** — instruction-following where the constraint is a
  prohibition ("do not mention _cat_") or an exclusivity ("respond with **only** ACK").
- **de morgan equivalence** — logical-rewrite pairs scored identical by an embedded
  truth-table evaluator; the deliberately-wrong rewrite fixture must score non-equivalent.

The scored gate (fast, no env) runs every family against a reference oracle (all pass)
and a negation-blind degenerate set (the three response-driven families must show
failures, proving the scorer discriminates rather than always passing):

```sh
go test ./internal/bench -run TestNegBench -count=1
```

To score a real model, feed a `{item-id: response-text}` JSON map and write an OBSERVED
witness (self-verifying digest; flips no gate). With no responses supplied it scores the
reference oracle:

```bash
FAK_NEGBENCH_OUT=/tmp/negbench.json \
FAK_NEGBENCH_MODEL=claude-opus-4-8 FAK_NEGBENCH_HW=win11-fak-bench \
FAK_NEGBENCH_RESPONSES=/tmp/responses.json \
  go test ./internal/bench -run TestNegBenchArtifact -count=1 -v
```

Each family reports per-item pass/fail plus a `passed/total` aggregate and `pass_rate`.
Read the artifact `families[].pass_rate` for the per-family score and `families[].items[]`
for the per-item rows (`id`, `prompt`, `expected`, `response`, `pass`, `why`). Score a
rung's ablation (e.g. `internal/negframe` reframe on vs off) *net* by diffing two
artifacts. The committed witness `testdata/negbench-reference-oracle.json` is the
reference-oracle run (model + host named, `OBSERVED`, `enforced: false`).


## Scheduler token-weight calibration (`tokenweightcal`) — [#5778](https://github.com/anthony-chaudhary/fak/issues/5778)

The scheduler weights in `tokenprofile.DefaultWeights` (`1 / 0.25 / 4`) are explicit
*illustrative policy inputs* from [#5771](https://github.com/anthony-chaudhary/fak/issues/5771),
not hardware truth. This arm fits them from a measured service-time ledger and refuses to
let an illustrative or synthetic input pass itself off as a measurement.

`CalibrateTokenWeights` fits `service_ms ≈ fixed + u·w_u + c·w_c + d·w_d` over the three
classes #5771 separates — uncached prefill, cache read/transfer, decode — by ordinary
least squares on the `fit` split, then scores the result on a never-fitted `holdout`
split against a **tuned** scalar-total baseline (the scalar gets its own least-squares
refit first, so the comparison is not a strawman).

### Producing a ledger on a sanctioned compute node

One JSONL row per observation (`fak-token-service-sample/1`), each declaring
`measured:` or `synthetic:` provenance plus model / engine / hardware / batch. Weights
never pool across those four — a mixed ledger is refused. Then:

```bash
FAK_TOKENCAL_LEDGER=/tmp/token-service-da33.jsonl \
FAK_TOKENCAL_OUT=/tmp/token-weights-da33.json \
FAK_TOKENCAL_BOUND_PCT=5 \
  go test ./internal/bench -run TestWriteTokenWeightCalibration -count=1 -v
```

The artifact (`fak-token-weight-calibration/1`) carries the fitted per-class rates with
95% intervals, `r_squared` / residual std, the held-out MAPE/RMSE for both arms, the
renormalized `scheduler_weights` drop-in for `tokenprofile.Weights`, and a self-verifying
`digest`. A calibration refusal is a valid recorded outcome, not a reason to fabricate.

### What it refuses

- **Uncontrolled shapes** — if the token shapes do not vary the three classes
  independently the normal matrix is singular; no weight is identifiable and the fit is
  refused rather than pseudo-inverted into a confident-looking answer.
- **Split leakage** — a `shape_id` present in both splits.
- **Net-true gain without the receipts** — `EvaluateNetSchedulingValue` is a hard error
  unless the measured alternative *and* the measured per-decision scheduling overhead are
  both supplied. Net value is `RMSE reduction − added overhead`, so an accuracy win that
  costs more to compute than it saves does not count as a gain.
- **Synthetic data claiming hardware truth** — a `synthetic:` ledger can fit and predict
  well and still cannot produce a net-true gain claim. Only a `measured:` ledger can.

`testdata/token_service_synthetic.jsonl` is a committed **synthetic** fixture (26 shapes,
seeded noise), regenerable byte-for-byte from `syntheticTokenRows` via
`FAK_TOKENCAL_FIXTURE_OUT=testdata/token_service_synthetic.jsonl`. It exists to test the
fold and to demonstrate the shape of the gap; it is **not** a hardware measurement and no
device claim rests on it.
