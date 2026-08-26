# Affected-test selection alternatives — 2026-08-10

Status: **WITNESSED WITH ONE UNAVAILABLE ARM**. Issue [#6371](https://github.com/anthony-chaudhary/fak/issues/6371) is backed by the machine-readable record [`affected-test-selection-6371.json`](../benchmarks/affected-test-selection-6371.json) and the pinned fixtures in [`examples/affectedtests-comparison/`](../../examples/affectedtests-comparison).

## Contract and result

Every runnable arm uses the same five-node diamond: changed leaf `internal/a`; direct importers `internal/b` and `internal/c`; top importer `cmd/app`; independent package `internal/isolated`. The oracle is exactly four affected projects: `{internal/a, internal/b, internal/c, cmd/app}`.

| Arm | Available | Precision / recall | Missed / extra | Tests or work targets | Selection wall / CPU / peak RSS | Execution wall / CPU / peak RSS |
|---|---:|---:|---|---:|---|---|
| fak native | yes | 1.00 / 1.00 | 0 / 0 | 0 | 1.719 s / 0.359 s / 77,201,408 B | n/a |
| changed-only Go baseline | yes | 1.00 / 0.25 | 3 / 0 | 1 | mapping not separately timed | 2.274 s / 0.641 s / 61,714,432 B |
| fak + Go test | yes | 1.00 / 1.00 | 0 / 0 | 4 | 1.719 s / 0.359 s / 77,201,408 B | 2.411 s / 12.734 s / 299,900,928 B |
| Bazel query + test-work | yes | 1.00 / 1.00 | 0 / 0 | 4 | 5.824 s / 4.547 s / 383,205,376 B | 8.798 s / 10.969 s / 424,525,824 B |
| Pants changed-since | **no** | n/a / 0.00 | 4 / 0 | 0 | measurement-zero | measurement-zero |
| Nx affected | yes | 1.00 / 1.00 | 0 / 0 | 4 | 3.764 s / 2.266 s / 363,659,264 B | 5.069 s / 2.375 s / 361,299,968 B |
| Gradle configured closure | yes | 1.00 / 1.00 | 0 / 0 | 4 | not a changed-file selector | 9.338 s / 15.141 s / 491,937,792 B |

The graph-aware runnable arms were exact on this synthetic graph; the changed-only tuned baseline missed all three importers. These one-host observations are **not a cross-system performance ranking**: startup/cache semantics differ, the graph is tiny, and Bazel/Gradle are deliberately qualified below. The native in-process selection microbenchmark remains 362.4, 428.5, and 506.5 ns/op (median 428.5 ns/op; 224 B/op; 8 allocs/op); the CLI measurement above includes process and graph-loading overhead.

## Pinned configurations and commands

Run from each fixture directory with all caches and generated output redirected beneath `_scratch/issue-6371`:

```text
# fak repository binary; Go 1.26.5
fak affected --file internal/a/a.go --list
go test -count=1 ./internal/a
go test -count=1 ./cmd/app ./internal/a ./internal/b ./internal/c

# Bazelisk 1.28.1 -> Bazel 9.2.0
bazelisk --batch query 'rdeps(//..., //internal/a:a)'
bazelisk --batch build --noshow_progress //internal/a:a_test //internal/b:b_test //internal/c:c_test //cmd/app:app_test

# Nx 23.1.1; NX_NO_CLOUD=true; npm lockfile committed
nx show projects --affected --files=internal/a/value.js
nx affected -t test --files=internal/a/value.js --outputStyle=static --skip-nx-cache

# Gradle 9.7.0; Eclipse Adoptium JDK 21.0.12+8-LTS
gradle.bat --offline --no-daemon --console=plain :cmd:app:test
```

Bazel 9's built-in fixture uses `filegroup` nodes and four deterministic `genrule` test-work targets so the comparison does not silently pull an external ruleset. It proves reverse-dependency selection and execution closure, **not** native `bazel test` behavior. Gradle core does not provide a generic changed-file selector in this fixture; it proves the explicitly configured `:cmd:app:test` dependency closure and excludes `:internal:isolated:test`.

The Gradle distribution SHA-256 is `84FBBA45C7F4C64ABC77460E1C00F541E9F960E3C7ED2538F1EDE19EACD873AE`; the JDK ZIP SHA-256 is `9BA963EE2371874A74185D18BC7BB2AB9407DF7683300855ED7606E0662321D0`.

## Unavailable Pants arm

`scie-pants` v0.13.2 publishes Linux and macOS assets but no Windows asset. The pinned Linux x86_64 asset (SHA-256 `74A1E53BC50D6EF6CE1BC67BD9F7B48E549505E0A2453AD4D5CCBC72B0BEA874`) was invoked under WSL with its home/cache redirected beneath `_scratch/issue-6371`; bootstrap exceeded the bounded 300-second attempt and produced no selection or test witness. Its selected set and tests are therefore empty, its precision is undefined, recall is zero against the oracle, and all unavailable measurements remain null/measurement-zero. This is availability evidence, not a Pants performance result.

## Resource, setup, and cost accounting

The committed JSON independently records wall time, process-tree CPU, peak RSS, setup elapsed time where a tool emitted it, tests/work targets, and missed/extras. Network bytes are `null` when the sampler could not measure them; they are never inferred from an apparently quiet run. Gradle's benchmark execution used `--offline`, so its run network is zero. The separate WSL Go witness reported zero socket messages, but its cold 207.32-second toolchain run is retained only as a platform witness, not compared with warm Windows observations.

All measured executions used local open-source tools: license, cloud, and billed compute cost were **USD 0**. Hands-on operator time was not instrumented, so operator seconds and an operator-labor total are reported as `null`, not fabricated. Raw logs, downloads, caches, configurations, and failed attempts remain inside `C:\work\fak\_scratch\issue-6371`; reproducible fixtures and the normalized evidence record are committed.
