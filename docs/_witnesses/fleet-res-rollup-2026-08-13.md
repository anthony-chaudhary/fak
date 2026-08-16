---
title: "Fleet resource accountant witness — 2026-08-13"
description: "Documentation for Fleet resource accountant witness — 2026-08-13, including the captured behavior, operating context, and reproducible fak evidence."
---

# Fleet resource accountant witness — 2026-08-13

Issue: #6557 (rung 1 of epic #6552)
Verdict: **WITNESSED**

## Value frame

- **For:** operators, and every later rung of #6552 — each one is justified by a delta in
  this number.
- **Problem:** `internal/harnessres` metered ONE guarded session (kernel half + wrapped
  child). The fleet total that opened #6552 — 87 processes, 12.90 GiB — was taken with
  `Get-Process` from a shell, so the fleet's own footprint was invisible to the thing that
  spawned it and a before/after claim would have been self-report.
- **Today:** `fak fleet res` walks the fak-owned process tree, classifies each process
  (seat / fak / broker / other), and folds one rollup with host fractions, in text and
  `--json`, banking a row on the existing JSONL ledger.
- **Better because:** the measurement now comes from inside fak, per class, against the
  same host denominator the per-session rows already divide by.
- **Read-only:** this rung kills, reaps, throttles and admits nothing.

## Reproduction

```
fak fleet res                    # text rollup, banks a ledger row
fak fleet res --json             # the same rollup as one JSONL row
fak fleet res --no-ledger        # pure read
fak fleet res --ledger <path>    # bank elsewhere
```

## Captured live fleet — 32 cores / 253.6 GiB host, 2026-08-13

```
fleet resources — 268 processes (262 sampled, 6 unreadable), oldest 17h22m22s
  seat      25 procs, rss 5.8 GiB, private 5.7 GiB, cpu 6816.7s
  fak       73 procs, rss 5.8 GiB, private 7.6 GiB, cpu 2809.1s
  broker    71 procs, rss 2.6 GiB, private 1.8 GiB, cpu 23.2s
  other     99 procs, rss 3.6 GiB, private 1.7 GiB, cpu 157.4s
  total    268 procs, rss 17.7 GiB, private 16.9 GiB, cpu 9806.3s
  host      32 cores / 253.6 GiB ram (166.0 GiB avail)
  fleet/host rss 7.0% of host ram, private 6.7% of host ram, cpu 9806.3s of 2001344.0 core-s (0.49%)
```

`--json`, the same fleet seconds later (seats and brokers churn between runs):

```json
{"schema":"fak-harness-fleet-resources/1","ts":"2026-08-13T20:56:28Z","procs":267,"sampled":267,"unreadable":0,"window_s":62542,"classes":[{"class":"seat","procs":25,"sampled":25,"rss_bytes":6181507072,"private_bytes":6162894848,"cpu_s":6816.84375},{"class":"fak","procs":73,"sampled":73,"rss_bytes":6178361344,"private_bytes":8211767296,"cpu_s":2809.28125},{"class":"broker","procs":71,"sampled":71,"rss_bytes":2796666880,"private_bytes":1962110976,"cpu_s":23.203125},{"class":"other","procs":98,"sampled":98,"rss_bytes":3880185856,"private_bytes":1854947328,"cpu_s":157.75}],"total":{"class":"total","procs":267,"sampled":267,"rss_bytes":19036721152,"private_bytes":18191720448,"cpu_s":9807.078125},"host":{"cores":32,"ram_total_bytes":272252637184,"ram_avail_bytes":178211332096,"core_seconds":2001344,"fleet_rss_pct_of_host_ram":6.99230000080189,"fleet_private_pct_of_host_ram":6.681926256495821,"fleet_cpu_pct_of_host_core_s":0.4900246097122734}}
```

The `broker` row reproduces the finding that motivates the whole epic: 71 MCP/tool hosts
holding 2.6 GiB resident for 23.2 CPU-seconds — a ~0.001 duty cycle, i.e. resident tax
rather than work.

## Classification cross-checked against ground truth

Counted independently from a shell, at the same moment as an earlier run of the same
binary (the fleet churns, so the classes are checked against a simultaneous census rather
than against the numbers above):

| class | rollup | independent count | source |
|---|---|---|---|
| fak | 56 | 56 | `Get-Process` name `fak*` |
| seat | 23 | 23 | `claude` 7 + `codex` 16 |
| broker | 61 | 67 host-wide | `Win32_Process` CommandLine matching `mcp` |

The broker row is *below* the host-wide count because six of those MCP processes have no
fak-owned ancestor, so the tree walk correctly leaves them out — the classifier does not
bill the fleet for every Node process on the box.

## Envelope

| axis | target | witnessed |
|---|---|---|
| fleet seats classified | >= 18 sessions | 25 |
| processes walked | >= 87 count | 267 / 268 |
| host cores | >= 32 cores | 32 |

## Honesty notes

- `unreadable` is the point of the presence bits: processes in the tree that could not be
  opened (protected or exiting) are counted in `procs`, excluded from every byte and CPU
  total, and reported. A partial read cannot masquerade as a small fleet. The two runs
  above show 6 and 0 — it varies with what is exiting at that instant.
- `window_s` is the age of the OLDEST walked process, the only span a fleet's *lifetime*
  CPU totals can honestly be divided by. It over-states available capacity for a fleet
  whose seats are much younger than its oldest fak process, so
  `fleet_cpu_pct_of_host_core_s` is a floor, not a point estimate.
- Private bytes ride alongside resident because resident double-counts pages that N copies
  of one binary already share; the pooling rungs of #6552 must argue against private.
- The binary that produced these numbers was built from a pristine `git archive HEAD` tree
  plus this change, because the shared checkout carries several peers' in-flight edits that
  do not compile (`internal/gateway`, duplicate symbols in `cmd/fak`). Those reds are not
  this diff: `go build ./...` and `go vet ./cmd/fak/ ./internal/harnessres/` are both clean
  on HEAD-plus-this-change.
