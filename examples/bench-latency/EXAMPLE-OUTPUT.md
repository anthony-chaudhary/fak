# Captured `fak bench` output

A verbatim run of the command in [`README.md`](README.md), captured with the prebuilt
`fak` **v0.34.0** binary (`go1.26.3 windows/amd64`) from the repo root. The fak side is a
**live kernel** A/B — the in-process p50 is the kernel's own `Decide` fold, and the
spawned-hook p50 is a real wall-clock round-trip of `exec.Command(fak, "hook")` ×100. The
verdicts (`gate_primary`, the bucket shapes, token counts) reproduce exactly; only the
**nanosecond latency figures are wall-clock** and will vary run to run and box to box. The
multiplier on this box was **~3,194×**; CLAIMS #19 reports ~2,849× on its box; the issue's
historical 5,583× capture is a third. All three pass the same witness gate — the µs-vs-ms
*order of magnitude* is the durable claim, not the exact number.

## The run

```console
$ C:\Users\USER\bin\fak.exe bench --suite tau2-smoke --baseline-n 100 --out report.json
== fak bench: tau2-airline-smoke ==
in-process adjudication p50 : 3086 ns
spawned-hook        p50     : 9857700 ns (9.858 ms, n=100)
fusion speedup (p50)        : 3194x
PRIMARY GATE                : pass  (in-process adjudication p50 (3086ns) vs spawned-hook p50 (9857700ns))
secondary token delta       : 47.17% (soft, never gates)
vdso hit-rate               : 0.500   pollution-rate: 0.000
workload hash               : 9f1701415fb4a360   live seam: live_seam_unverified
report written              : report.json
```

## The witness gate (both pass)

```console
report.json   gate_primary == "pass"            -> pass      (in-process beat the spawned floor)
baseline.json p50_ns ( 9857700 ) > 1ms          -> yes       (9.86 ms — a real full-binary spawn)
WITNESS: PASS
```

## `report.json` — the headline fields (annotated)

The pair that *is* the fusion number, plus the gate and provenance. (Full file is ~138 lines;
the histogram buckets and per-arm token counts are elided here.)

```jsonc
{
  "provenance": {
    "app_version": "0.34.0",
    "command": "fak bench --suite tau2-airline-smoke",  // the suite FILE is tau2-smoke; the trace's slice_id is tau2-airline-smoke
    "slice_id": "tau2-airline-smoke",
    "workload_hash": "9f1701415fb4a360",                // identical-workload guard: both arms replay this same hash
    "go_version": "go1.26.3",
    "os": "windows"
  },
  "vdso_on":  { "p50_ns": 3086, ... },                  // <-- IN-PROCESS adjudication p50 (the µs side)
  "vdso_off": { "p50_ns": 2753, ... },                  //     the A/B's other arm (vDSO off) — drives the SOFT token delta, not the gate
  "spawned_hook_baseline": {
    "source": "spawned `fak hook` per decide, this machine",
    "p50_ns": 9857700,                                  // <-- SPAWNED-hook p50 (the ms side) = 9.858 ms
    "p99_ns": 281793000,                                //     p99 is noisy (OS scheduling) — the gate reads p50, not p99
    "calls": 100,
    "spawn_model": "process-per-decide (windows)"
  },
  "kpis": { "tool_call_p50_ns": 3086, "vdso_hit_rate": 0.5, ... },
  "gate_primary": "pass",                               // <-- WITNESS 1: in-process p50 < spawned p50, baseline non-zero
  "primary_detail": "in-process adjudication p50 (3086ns) vs spawned-hook p50 (9857700ns)",
  "token_delta_pct": 47.17,                             //     SOFT secondary (vDSO on vs off) — never gates (CLAIMS #20)
  "dollar_per_task": 0.001485,                          //     illustrative blended $/Mtok — not billed
  "live_seam": "live_seam_unverified"                   //     honest RED flag: no real-engine transcript pinned this run
}
```

The fusion multiplier is just `spawned_hook_baseline.p50_ns / vdso_on.p50_ns`
= `9857700 / 3086` ≈ **3,194×** — three orders of magnitude, because one transport is a
function call (µs) and the other spawns a whole binary (ms).

## `baseline.json` — the standalone spawned-hook witness

```json
{
  "calls": 100,
  "p50_ms": 9.8577,
  "p50_ns": 9857700,
  "source": "spawned `fak hook` per decide, this machine"
}
```

This is **witness 2**: `p50_ns = 9857700 > 1_000_000` (> 1 ms), so the spawned floor is a
real full-binary spawn cost, not measurement noise. CLAIMS #19 phrases the same check as
`spawned_hook_baseline.p50_ns > 1ms`.

## Reading note

`gate_primary` encodes **direction, not magnitude** — it asks only "did the in-process fold
beat the spawned floor?" It is a subsystem regression sentinel for the "no per-call process
boundary on the decide path" property, **not** a serving-throughput, model-quality, or 45×
fleet headline (CLAIMS #20). Quote the order-of-magnitude (µs vs ms), not the exact ratio.
