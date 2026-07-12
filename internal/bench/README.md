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

## Lane gate

```sh
go test ./internal/bench -count=1
go vet ./internal/bench
```
