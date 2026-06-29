# fak kernel — subsystem-latency walkthrough (the in-process-vs-spawned-hook fusion number)

**The same `Fold` decide runs two ways — once in-process (a function call) and once as a
spawned `fak hook` (a full-binary process per decide) — and `fak bench` times both on THIS
machine, apples-to-apples.** The delta is the *process-level fusion win*: collapsing the
spawned-hook boundary turns a per-call **millisecond** into a per-call **microsecond**, a
~3,000–5,000× gap depending on the box. This walkthrough shows an operator how to run the
check, where the headline number lives in `report.json`, what `gate_primary=="pass"` means,
and — loudly — what this number is *not*.

This is the latency-of-the-decide-path companion to the [`turntax`](../turntax/README.md)
(cost-of-malformed-calls) and `fanbench` (fan-out) benchmarks. Those price model *turns*;
this one times the *subsystem boundary* — a different axis entirely. It is the witness
behind **CLAIMS #19** ("the syscall subsystem latency check") and the "no IPC on the decide
path" guarantee (**CLAIMS #14**, `TestNoOsExecOnHotPath`).

> Note on the [issue #319](https://github.com/anthony-chaudhary/fak/issues/319) text, which
> this example resolves — three corrections, in the spirit of the house honesty fences:
> 1. **The number drifts; it is not fixed at 5,583×.** The check measures wall-clock process
>    spawn on *the box it runs on*, so the multiplier is machine- and load-dependent. The
>    issue's headline "2.365 µs / 13.203 ms ⇒ 5,583×" is one historical capture; the current
>    [`CLAIMS.md`](../../CLAIMS.md) #19 row reads **2.427 µs / 6.913 ms (n=100) ⇒ ~2,849×**;
>    a fresh run on this Windows dev box (captured in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md))
>    gave **~3.09 µs / 9.86 ms (n=100) ⇒ ~3,194×**. The *witness* is not the magnitude — it
>    is `gate_primary=="pass"` (in-process beats spawned at all) plus a spawned `p50_ns > 1 ms`
>    floor. All three captures pass it; the µs-vs-ms order-of-magnitude is the durable claim.
> 2. **No `run.sh` build step is needed.** The bundled [`run.sh`](run.sh) drives the prebuilt
>    `fak` binary directly (`FAK_BIN=/path/to/fak`); no Go toolchain at run time. (The issue
>    sketched `./fak bench` implying a local build.)
> 3. **The directory is `examples/bench-latency/`**, not the `examples/bench-subsystem/` the
>    ticket guessed. The witness module is `internal/bench` (+ `internal/metrics` for the
>    report schema), exactly as the ticket noted (`cmd/fak/main.go` `case "bench"`).

## Run it

`fak bench` needs no model, no network, and no key — it replays a frozen tool-call trace
(`testdata/tau2/tau2-smoke.json`, `slice_id: tau2-airline-smoke`) through the kernel twice
(vDSO on/off) and spawns `fak hook` per decide for the baseline. The verdicts are
deterministic; only the nanosecond latency figures are wall-clock and vary per run.

```bash
# with a prebuilt binary (no build):
FAK_BIN=/path/to/fak ./examples/bench-latency/run.sh
```

On Windows with the prebuilt binary, from the repo root (so the cwd-relative `testdata/tau2`
suite resolves):

```powershell
C:\Users\USER\bin\fak.exe bench --suite tau2-smoke --baseline-n 100 --out report.json
```

`--baseline-n N` sets the spawned-hook sample count (default 30; CLAIMS uses 100). The run
writes **`report.json`** (the full A/B artifact) and **`baseline.json`** (the standalone
spawned-hook witness) next to `--out`. Captured output and a field-by-field reading are in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The two transports (same decide, measured twice)

| transport | what it is | how it is timed | typical p50 |
|---|---|---|---|
| **in-process** | the kernel's `Decide` fold called directly — a function call, no process boundary | an inner 2000-rep calibration loop (the fold is faster than the OS clock), divided | **µs** (~2–3 µs) |
| **spawned `fak hook`** | the *same* decide logic reached by spawning the full `fak` binary once per call, piping the call on stdin | wall-clock round-trip of `exec.Command(fak, "hook")`, `n` samples | **ms** (~7–13 ms) |

Both arms run the identical `Fold` decide — the comparison is honest because the only thing
that changes is the *transport*, not the logic. `os/exec` lives only in the baseline harness
(`internal/bench`), **never** on the dispatch hot path; the absence proof
(`TestNoOsExecOnHotPath`) targets `internal/kernel`.

## How to read the µs-vs-ms delta

The headline lives in `report.json` as a pair:

- `vdso_on.p50_ns` — the in-process adjudication p50, in **nanoseconds** (≈ a few thousand → single-digit µs).
- `spawned_hook_baseline.p50_ns` — the spawned-hook p50, in **nanoseconds** (≈ ten million → ~10 ms).
- The console prints `fusion speedup (p50)` = `baseline.p50_ns / vdso_on.p50_ns`. That ratio
  is the fusion number. It is large because one side is microseconds and the other is
  milliseconds — three to four orders of magnitude apart.

`baseline.json` carries the same spawned figure standalone (`p50_ns`, `p50_ms`, `calls`,
`source`) as the second witness.

## The witness gate

Two checks, both deterministic in *direction* (not magnitude):

1. **`report.json` `gate_primary == "pass"`** — set by `metrics.Report.ComputeGate`: pass iff
   `vdso_on.p50_ns < spawned_hook_baseline.p50_ns` and the baseline is non-zero. It is a
   *subsystem regression sentinel* — it confirms the decide path is not accidentally paying a
   per-call process boundary. It does **not** encode a target multiplier.
2. **`baseline.json` `p50_ns > 1 ms`** (i.e. `> 1_000_000`) — the spawned floor is real
   (a full-binary spawn genuinely costs milliseconds), so the comparison is not measuring
   noise. CLAIMS #19 phrases this as `spawned_hook_baseline.p50_ns > 1ms`.

## What this is NOT (CLAIMS #20, verbatim)

> The check is useful as a subsystem regression sentinel: it times the adjudication fold and
> confirms the decide path is not accidentally paying a per-call process boundary. It is
> deliberately **not** a production-readiness, model-quality, serving-throughput, or 45×
> fleet headline.

The `token_delta_pct` field (vDSO on vs off) is reported as a **soft secondary** and never
gates — it is the local-reuse side-effect of the same run, not the subsystem claim. Do not
quote the fusion multiplier as a serving-throughput or fleet number; it is the latency of one
boundary, on one machine, against a deliberately honest local floor.

## Cross-links

- [`CLAIMS.md`](../../CLAIMS.md) #19 (the subsystem latency check) and #20 (what it is not).
- Report schema: `internal/metrics/metrics.go` — `Report`, `Baseline`, `KPIs`, `ComputeGate`.
- Runner: `internal/bench/bench.go` (`RunArm`, `MeasureSpawnedBaseline`, `Run`); CLI in
  `cmd/fak/main.go` (`case "bench"` → `cmdBench`).
- Sibling benchmarks: [`turntax`](../turntax/README.md) (turn-tax cost), `fanbench` (fan-out).
