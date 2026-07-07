---
title: "Full-Span Witnessed Trace + Tail Under Load (G5)"
description: "One process, all four clock bands (B6 plan → B4 admission → B2 turn → B0 decide) in a single causal trace with deny-consequence attribution; plus the µs decide fold's latency distribution quiet vs under same-process synthetic B2/B3 load."
---

# FULLSPAN-TAILLOAD-RESULTS — the dynamic-range composition witness (G5) and the tail-under-load arm (R6)

**Issue:** [#2223](https://github.com/anthony-chaudhary/fak/issues/2223) (epic #2218, gap G5, risk R6 of `docs/notes/DYNAMIC-RANGE-ONE-BINARY-2026-07-01.md`)
**Date:** 2026-07-07
**Machine:** AMD Ryzen 9 9950X (16C/32T), 256 GiB RAM, WSL2 Ubuntu 24.04 (kernel 6.6.114.1-microsoft-standard-WSL2) on a Windows 11 host; go1.26.0 linux/amd64
**Provenance label:** **OBSERVED, research-grade** — real code, one box, one run per artifact; **no gate asserted or flipped** (promotion needs two agreeing runs per the fence, and is explicitly out of scope for #2223)
**Artifacts:** `docs/benchmarks/fullspan/fullspan-trace-20260707.json` · `docs/benchmarks/fullspan/tailload-20260707-run1.json` · `…-run2.json`
**Instrument:** `internal/bench/fullspan.go` (+`_test`), `internal/bench/tailload.go` (+`_test`)

## What this measures (and the gap it closes)

Every number in BENCHMARK-AUTHORITY's dynamic-range story lived in a *separate*
artifact on a *separate* clock: the ns/µs kernel anchors (B0), the seconds-scale
turn benches (B2), the minutes-scale admission (B4), and the days-scale plan
verdicts (B6). Gap G5: **no committed artifact showed the bands operating in one
process as one causal story.** Risk R6: every B0 anchor is a quiet-process
number, but production runs the µs decide fold *concurrently with* streaming and
session bookkeeping on one Go runtime.

Two arms, both hermetic (no network, no model, no fixtures):

1. **Full-span trace** (`RunFullSpan`): ONE process runs a real B6 plan verdict
   (`superloop.Walk`, 3-phase plan), a real B4 admission
   (`dispatchtick.EvaluatePreflight`), a scripted B2 agent turn, and 5 real B0
   `kernel.Decide` calls — every span stamped with band, monotonic-clock
   duration, verdict, and a **walkable parent chain** B6→B4→B2→B0. Every DENY
   span carries an observed **downstream-consequence class**
   (`retry_turn` / `forked_outcome` / `clean_stop`), attributed from what
   actually followed in the trace — R5's "denial-of-work rate" made measurable
   per deny reason.
2. **Tail under load** (`RunTailUnderLoad`): the SAME in-process `kernel.Decide`
   fold the 100 µs distribution gate covers, measured per-call twice in one
   process — quiet, then while the same process drives synthetic B2/B3-shaped
   work (SSE-style streaming churn + shared session-table churn; the issue's
   definition — no external load generator). Percentiles are **exact over raw
   samples** (no histogram buckets — bucketing would quantize exactly the tail
   this arm exists to see).

## Results — full-span trace (one run, one process)

10 spans, all four bands, parent chain verified walkable by
`TestRunFullSpan_FourBandsAndDenyClasses`.

### Per-band attribution (four clocks, never one number)

| Band | Clock | Spans | Total ns in-band |
|---|---|---:|---:|
| B6 plan verdict (`superloop.Walk`) | days | 1 | 30,319 |
| B4 admission (`EvaluatePreflight`) | 5–30 min | 1 | 3,842 |
| B2 turn + dispatch (scripted) | seconds | 3 | 301,192 |
| B0 decide (`kernel.Decide` ×5) | ~0.4–2.4 µs nominal | 5 | 57,004 |

The artifact deliberately has **no** `end_to_end`, `ratio`, or `speedup` field —
the three-clocks fence is enforced *structurally* on the JSON keys by the test
(`assertNoForbiddenKeys`), not just promised in prose.

### Deny → downstream consequence (all three classes in one run)

| Tool | Reason | Consequence | Deny ns | Induced ns | Induced spans |
|---|---|---|---:|---:|---:|
| `write_ledger_entry` | `DEFAULT_DENY` | `retry_turn` | 14,167 | 5,540 | 1 |
| `write_ledger_entry` (retry) | `DEFAULT_DENY` | `forked_outcome` | 5,540 | 55,833 | 2 |
| `shell_rm_rf` | `POLICY_BLOCK` | `clean_stop` | 2,521 | 0 | 0 |

Per-reason deny rates: `DEFAULT_DENY` → {retry_turn: 1, forked_outcome: 1};
`POLICY_BLOCK` → {clean_stop: 1}. A deny is not free even when correct: the
denied-write retry induced a same-tool retry (5.5 µs), the second deny forked
the outcome into a different tool's decide + dispatch (55.8 µs induced), and the
policy block stopped the turn cleanly (0 ns induced) — three structurally
different downstream costs the flat "denies=N" counter could not distinguish.

## Results — tail under load (two runs, 200,000 samples/arm)

Config (defaults): 200k samples + 5k warmup per arm; loaded arm adds 4 streaming
workers + 4 session-churn workers over 512 session ids, all in the measuring
process. Canonical call: `get_user_details` (verified ALLOW before timing — a
silent deny would time the cheaper branch).

| Arm | p50 ns | p90 ns | p99 ns | p99.9 ns | max ns | mean ns | >100 µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| run 1 quiet | 2,904 | 3,498 | 13,483 | 123,513 | 319,423 | 3,709 | 373 (0.19%) |
| run 1 **loaded** | 3,452 | 5,669 | **33,714** | 450,726 | 837,673 | 6,083 | 1,153 (0.58%) |
| run 2 quiet | 2,892 | 3,776 | 14,137 | 118,899 | 318,129 | 3,726 | 353 (0.18%) |
| run 2 **loaded** | 3,388 | 5,336 | **29,533** | 456,049 | 1,346,002 | 5,901 | 1,113 (0.56%) |

**The R6 finding, plainly:** same-process B2/B3-shaped load barely moves the
median (p50 ×1.17–1.19) but degrades the tail disproportionately — p99 ×2.1–2.5,
p99.9 ×3.7–3.8, and the share of calls over the standing gate's 100 µs bar
roughly **triples** (0.18–0.19% → 0.56–0.58%). The quiet-process B0 anchors are
honest for what they name, and they understate the production tail.

**Two-run agreement:** quiet p50 within 0.4%, quiet p99 within 4.9%, loaded p50
within 1.9%, loaded p99 within 14.2% (33.7 vs 29.5 µs — same side of every
threshold, ordering preserved). Consistent but not promotion-grade; per the
fence, no OBSERVED→gate promotion is attempted here.

## Honesty fences

- **Three clocks, never one number.** Per-band totals are reported per band and
  NEVER multiplied or ratioed across bands; the artifact schema carries no
  cross-band field, and the test fails if one appears.
- **The B2 turn is scripted** (deterministic call set) so deny consequences are
  attributable; its span durations are real but carry no model seconds. B6/B4
  spans time real fak code paths (`superloop.Walk`, `EvaluatePreflight`), not
  days/minutes of wall clock — the band labels name the *governing* clock.
- **Trace B0 spans are one-shot costs** (first-call/cold effects included; the
  three deny spans above include first-touch of the deny path). Do NOT compare
  them to the warmed `go test -bench` medians in the M3 Pro anchor row (362 ns
  Decide) — different statistic, different host, different regime. Within this
  sheet, quiet-vs-loaded is the only intended comparison, and the tailload quiet
  arm IS that baseline.
- **Per-call timer overhead** (~2 clock reads/sample) is identical in both arms:
  the quiet-vs-loaded delta is clean; absolute values carry the overhead.
- **Virtualized host.** WSL2 runs under a hypervisor; scheduler/clock jitter
  inflates tails vs bare metal (quiet p99.9 is already ~120 µs here). Numbers
  are host-labeled OBSERVED, not portable constants.
- **Coarse-clock hosts refuse.** On Windows the Go monotonic clock steps
  ~0.5 ms, which would quantize every µs-band duration to 0 or a ~520 µs tick —
  the artifact writers skip with an explicit message rather than write
  quantization noise, and the CI tests gate distribution assertions on measured
  clock resolution (structural assertions always run).
- **No gate asserted or flipped.** The 100 µs bar appears only as context
  (`over_100us` counts); the standing single-flow distribution gate
  (`internal/gateway/adjudication_latency_test.go`) is untouched.

## Reproduce

```sh
# CI-sized smoke (asserts structure everywhere; distributions on fine-clock hosts)
go test ./internal/bench/ -count=1

# Full-span trace artifact (fine-clock host required; paths are absolute
# because `go test` runs with the package dir as cwd)
FAK_BENCH_HW="<cpu, ram, os>" \
FAK_FULLSPAN_OUT="$PWD/docs/benchmarks/fullspan/fullspan-trace-$(date +%Y%m%d).json" \
  go test ./internal/bench/ -run TestWriteFullSpanArtifact -count=1 -v

# Tail-under-load report (run twice; agreement is part of the readout)
FAK_BENCH_HW="<cpu, ram, os>" \
FAK_TAILLOAD_OUT="$PWD/docs/benchmarks/fullspan/tailload-$(date +%Y%m%d)-run1.json" \
  go test ./internal/bench/ -run TestWriteTailLoadArtifact -count=1 -v
```
