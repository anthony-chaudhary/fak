# Issue 9172: hardware net accepted-token-value witness

This directory is a reproducible **decision model, not benchmark evidence or a purchasing
recommendation**. It makes hardware and software options comparable in one unit while keeping
manufacturer/paper facts separate from operator-entered economics and workload assumptions.

## Reproduce and check

From the repository root with PowerShell 7:

```powershell
pwsh -NoProfile -File docs/_witnesses/issue-9172-hardware-net-token-value/calculate.ps1
pwsh -NoProfile -File docs/_witnesses/issue-9172-hardware-net-token-value/calculate.ps1 -Check
```

The first command deterministically regenerates `results.csv`; the second fails if a source ID is
missing, a bounded input is invalid, or the checked-in result differs byte-for-byte (after newline
normalization). No network access, locale-sensitive parsing, clock, or randomness enters the
calculation.

## Provenance boundary

- `sources.csv` is the primary-source ledger. Each row identifies a manufacturer, standards body,
  or primary paper, gives its URL/version and retrieval date, and quotes only the fact used to
  anchor a scenario. A cited peak is not a measured fak rate.
- `scenarios.csv` is entirely **illustrative operator input**, including prices, amortization,
  electricity, utilization, quality acceptance, rejection, migration, availability, reliability,
  workload bytes, effective bandwidth where it differs from a cited peak, and compute caps. The
  `assumption_set=illustrative-v1` label is deliberately carried into output.
- `results.csv` is derived, not sourced. It must not be read as a benchmark, forecast, vendor
  quote, total-cost estimate, or claim that unlike systems have matched output quality.

The representatives cover GPU, CPU, unified memory, HBM/cache, CXL/link, storage/topology, and
software byte reduction. They expose mechanisms; they do not assert that one SKU is the best
implementation of a category. Replace assumptions with local quotes, measurements, quality gates,
and reliability history before making an operator decision.

## Deterministic model

For each scenario (decimal SI units):

```text
physical_bytes/token = logical_bytes/token * byte_reduction_factor
bandwidth_tokens/s   = bandwidth_B/s / physical_bytes/token
raw_tokens/s         = min(bandwidth_tokens/s, compute_cap_tokens/s)
accepted_tokens/hour = raw_tokens/s * 3600 * utilization
                       * accepted_quality * (1 - rejected_work_fraction)
                       * availability
capex/hour           = capex / (amortization_years * 365 * 24)
energy/hour          = power_kW * energy_price * utilization
migration/hour       = migration_cost / migration_amortization_hours
total_cost/hour      = capex/hour + energy/hour + migration/hour
                       + reliability_cost/hour
net accepted-token value = accepted_tokens/hour / total_cost/hour
```

`accepted_quality` is the fraction of generated tokens surviving a fixed task/quality gate, not a
subjective multiplier. `rejected_work_fraction` counts discarded speculation, retries, failed
requests, and other generated work not accepted. `availability` removes outage time; the separate
`reliability_cost_usd_per_hour` prices repair, redundancy, expected data-loss/incident burden, and
on-call overhead. Migration is amortized over the declared evaluation hours. Capex and energy are
always present rather than hidden behind acquisition price or peak tokens/s.

This is intentionally a transparent first-order model. It does not infer latency distributions,
capacity limits, financing/tax effects, cooling/PUE, networking power, software labor, or correlated
failures. Put such costs into the explicit assumption columns or extend the schema before using the
model for a real procurement envelope.

## Falsifiable fak-native experiment contract

**Candidate:** change only decode weight/KV physical bytes by implementing a fak-native
byte-reduction path corresponding to `h100_byte_reduced`; do not route to llama.cpp.

| Contract field | Pre-registered statement |
|---|---|
| Changed term | `physical_bytes_per_accepted_token`; target factor is `0.42` versus the matched fak-native baseline. All capex, power, utilization, availability, and workload terms remain fixed. |
| Envelope | Qwen3.8, one fixed quantized artifact, fixed prompt/output corpus and seeds, decode batch 1 and 8, fixed context buckets, warm resident model, same GPU/power cap/compiler, at least 30 paired runs. Receipt must name the fak-native engine. |
| Prediction | Hardware-counter bytes per accepted token fall at least 45%, while p50 accepted tokens/s rises at least 25% and net accepted tokens/USD rises at least 20% after measured migration and rejected-work costs. |
| Counter-hypothesis | Packing/unpacking, metadata, cache misses, or compute saturation erase byte savings, yielding less than 10% accepted-token throughput gain or higher total bytes per accepted token. |
| Quality risk | Quantization, grouped KV, or altered kernels reduce exact task acceptance or cause long-context regressions. Gate against frozen task scores/logit checks with no more than 1 percentage-point absolute acceptance loss. |
| Waste accounting | Include conversion and load time, draft/verification rejects, retries, warm-up, failed/OOM runs, padding, idle power, and migration engineering time. Report total target + auxiliary bytes/joules per accepted token. |
| Stop rule | Stop after 30 valid pairs, or immediately on correctness failure, >1 point quality loss, repeated corruption, or three OOM/device-reset events. Reject the prediction if any throughput/value threshold misses its paired confidence interval. |
| Rollback | Disable the explicit byte-reduction feature flag, reload the unchanged baseline artifact, rerun five receipt-checked baseline samples, and retain failed receipts/counters. No automatic engine fallback. |

A passing experiment replaces the illustrative byte-reduction, utilization, quality, reject,
power, migration, and availability fields with witnessed values and archives raw receipts. A failure
is equally useful: preserve it and keep `results.csv` labeled illustrative.
