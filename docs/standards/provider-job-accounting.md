---
title: "Provider-neutral completed-job accounting (fak-provider-job-accounting/1)"
description: "Verdict: compare local and API execution only through append-only job receipts that keep commercial terms, provider telemetry, local outcome/quality, and derived accounting separate. The."
---

# Provider-neutral completed-job accounting (`fak-provider-job-accounting/1`)

Issue: [#9575](https://github.com/anthony-chaudhary/fak/issues/9575)

**Verdict:** compare local and API execution only through append-only job receipts that keep commercial terms, provider telemetry, local outcome/quality, and derived accounting separate. The schema is [`provider-job-accounting-schema.json`](provider-job-accounting-schema.json).

## Record and ledger contract

Write one schema-valid record per line in an append-only JSONL ledger. A correction appends a complete replacement record with `corrects_record_id`; it never edits or deletes the earlier record. Stable record kinds are:

- `official_commercial_terms`: a dated provider-published price/spec snapshot with official-source provenance.
- `job_receipt`: one attempted job, whether it completed, failed quality, failed execution, or was cancelled.

A job receipt always has four independent layers:

1. **`official_commercial_terms`** — a reference to a dated terms record, or an explicit `not_applicable`/`unknown`. It is not telemetry.
2. **`raw_provider_counters`** — provider-emitted counters only. Every known counter key is present; an unavailable counter is `null`, never reconstructed from another field or a pricing equation.
3. **`local_job_outcome`** — execution status, the named quality gate and evidence, total wall time, retries, setup, compaction, and operator intervention. This is the matched job result, not a provider price claim.
4. **`derived_accounting`** — attempted-job cost, quality-qualified completed-job cost, qualified wall time, two non-overlapping cost views, formulas, provenance, and uncertainty.

The two cost views describe the same attempted-job total:

- `meter_cost_usd` partitions cost by billable meter (token classes, tools/search/container, local compute, operator time, other).
- `phase_cost_attribution_usd` partitions that same total by job phase (setup, primary attempts, retries, compaction, operator intervention, other). **Do not add the two views together.**

`quality_qualified_completed_job_cost_usd` and `qualified_wall_time_seconds` are non-null only when the job status is `completed` and the quality gate is `pass`. A failed or unrun gate leaves both fields `null`; its spend still appears in `attempted_job_cost_usd`.

## Counter-only provider observations

Use the append-only `provider_counter_observation` kind when a provider witness exposes trustworthy counters but cannot honestly supply a complete job envelope. It preserves the observed per-turn counters and makes every unavailable job field explicit. It is **not** a `job_receipt`: it carries no task revision, local outcome, quality qualification, elapsed time, cost accounting, or realized cache ratio. Those fields remain `null` in each turn and are also named in `unavailable_fields`; they MUST NOT be inferred from the counters.

The first GPT-5.6 Sol observation preserves the already-observed two-turn values from `docs/_witnesses/vcache-provider-calibration-live-2026-08-18.json`: input tokens `28,473` then `39,739`, cached input tokens `3,456` on both turns, and cache-write input tokens `0` on both turns. The row is scrubbed and makes no 200:1 or 300:1 realized-ratio claim.

A quality-qualified `job_receipt` requires the complete workload envelope, an explicit task revision, a local job outcome and quality artifact, wall time, official terms application, raw provider counters for the job scope, and derived accounting with uncertainty. Counter-only evidence can be promoted only by appending a new receipt or correction record; never rewrite the observation.

Next evidence: #9552 must bind a matched provider run to an explicit task revision and complete attempted-job envelope. #9578 must capture the quality result, elapsed time, complete costs, and any realized ratio before the evidence can support a quality-qualified completed-job claim.

## Matched envelopes

The v1 comparison grid is deliberately closed:

- context buckets: `35000`, `64000`, `128000`, `200000` tokens;
- input-to-output ratios: `200` or `300`, representing `200:1` and `300:1`.

The bucket is the declared workload envelope, not a substitute for a provider counter. Actual counters remain raw and nullable.

## Counter and provenance rules

- `input_tokens`, `cached_input_tokens`, `cache_write_tokens`, `output_tokens`, and `reasoning_tokens` are independent raw fields. Preserve the provider's definitions and state `reasoning_token_relation`; do not assume reasoning tokens are additional to output tokens.
- Tool calls, searches, containers, retries, setup, compaction, operator intervention, and wall time have explicit fields even when zero or unavailable.
- Use `0` only for a witnessed zero. Use `null` when the source did not expose the value.
- Provider-reported cost is raw telemetry. A calculated cost belongs only in `derived_accounting`.
- Every derived result names source IDs and an uncertainty class: `exact`, `bounded`, `estimated`, or `unknown`. An estimate must not be relabeled as observed telemetry.
- Fixtures are scrubbed, synthetic examples for schema validation. They are not production measurements or pricing evidence.

## Fixtures

- [`fixtures/provider-job-accounting-gpt56-sol-api.jsonl`](fixtures/provider-job-accounting-gpt56-sol-api.jsonl): official-terms snapshot plus scrubbed synthetic API jobs demonstrating a quality-qualified 64K/200:1 completion and a 35K/300:1 quality failure whose attempted spend remains counted.
- [`fixtures/provider-job-accounting-local.jsonl`](fixtures/provider-job-accounting-local.jsonl): scrubbed synthetic local job at 200K and 300:1 with all provider-only counters explicitly `null`.

GPT-5.6 Sol's dated official-terms snapshot and evidence boundary are documented in [`../notes/gpt-5-6-sol-official-terms-2026-08-27.md`](../notes/gpt-5-6-sol-official-terms-2026-08-27.md). The schema is provider-neutral: Claude and Gemini records for #9588 can use it without changing field names or semantics.
