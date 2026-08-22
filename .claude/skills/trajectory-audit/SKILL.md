---
name: trajectory-audit
description: Audit recent Claude and Codex transcript JSONL with the first-class Go `fak trajectory audit` verb: exact token/cache buckets, source coverage, behavior, deterministic bottlenecks, and baseline regressions. Use for cross-session cost/efficiency and churn questions.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "[--since 7d] [--jsonl OUT] [--md OUT] [--baseline PRIOR.jsonl]"
output_root: none
metadata:
  opencode: claude-only
---

# /trajectory-audit — cross-harness efficiency audit

Use the Go data path for every operational audit:

```bash
fak trajectory audit --since 7d \
  --jsonl trajectory-audit-$(date +%Y%m%d).jsonl \
  --md trajectory-audit-$(date +%Y%m%d).md
```

The verb discovers both supported transcript homes by default:

- Claude: `$CLAUDE_CONFIG_DIR/projects`, else `~/.claude/projects`.
- Codex: `$CODEX_HOME/sessions`, else `~/.codex/sessions`.

Pass `--claude-root` or `--codex-root` only for an explicit alternate corpus or
a pinned fixture. The paths persisted in artifacts are relative to those roots;
machine-private absolute roots are never emitted.

## Exact accounting contract

Every JSONL row uses schema `fak-trajectory-audit/1`. Claude usage is folded once
per unique `message.id`, matching the provider-billed turn identity. Codex
`token_count` records are cumulative: the parser uses the final monotonic
`total_token_usage` record and does not sum repeated `last_token_usage` snapshots.
Codex `input_tokens` includes its cached/cache-write subsets, so fresh input is
the exact subtraction `input - cached - cache_write`. The resulting input,
output, cache-create, and cache-read buckets are disjoint across both harnesses.

An unknown or malformed usage shape is never estimated. The verb writes a
`refusal` row and the markdown diagnostic, prints
`TRAJECTORY_SCHEMA_REFUSED`, and exits non-zero. Treat that as a parser-version
gap, not as zero usage.

## Baseline comparison

Use a prior versioned JSONL artifact as the baseline:

```bash
fak trajectory audit --since 7d --baseline prior.jsonl \
  --jsonl current.jsonl --md current.md
```

The report always emits comparable delta rows for:

- input:output ratio;
- cache-create burden;
- repeated failures; and
- hook p95 latency.

Each is higher-is-worse. A positive comparable delta carries
`"regression": true`; a missing denominator or hook sample is explicit and not
silently converted to zero.

## Reading the result

Read the markdown first, then query the JSONL only for drill-down:

1. Confirm source coverage: roots present, files scanned/discovered, records,
   exact/applied usage records, duplicates, and unsupported records.
2. Read the exact four-bucket totals and input:output/cache-create ratios.
3. Read rank 1 in the `bottleneck` rows. Ranking uses exact accounted tokens;
   ties sort by source, session, then relative path, so the highest-cost row is
   deterministic without inventing cross-vendor prices.
4. Check tool errors, repeated failures, mutation churn, and hook p95 on the
   session rows.
5. If `--baseline` was supplied, surface only the delta rows that are comparable
   regressions.

Do not infer that a missing root means zero activity; the coverage row records
`root_present: false`. Do not sum a vendor price across harnesses: this spine is
token-exact and deliberately does not fabricate a blended dollar rate.

## Python parity boundary

`tools/session_audit.py` remains only the Claude fixture parity oracle while the
Go parser is established. Use it during parser development to verify selected
exact fixture totals; do not use it as the operational audit path and do not
publish its markdown-first output as the cross-harness contract. The pinned
oracle values live beside the Go fixtures under
`internal/trajectory/testdata/audit/`.

## Output

- Versioned JSONL for automation and direct analytical queries.
- Markdown carrying the same totals, coverage, bottleneck, baseline, and
  unsupported-shape evidence.
- A short operator summary naming the exact totals, rank-1 bottleneck, any
  comparable regression, and the artifact paths.

This skill audits existing transcripts and writes only the requested report
artifacts. It does not edit code, plans, or memory.
