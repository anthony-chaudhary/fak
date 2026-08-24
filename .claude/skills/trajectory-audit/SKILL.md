---
name: trajectory-audit
description: Audit recent Claude and Codex transcript JSONL with the first-class Go `fak trajectory audit` verb: exact token/cache buckets, source coverage, behavior, deterministic bottlenecks, and baseline regressions. Use for cross-session cost/efficiency and churn questions.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "[--since 7d] [--user-contains TOPIC] [--jsonl OUT] [--md OUT] [--baseline PRIOR.jsonl]"
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

For a topical cohort, select only user-authored prompts; tool output and injected
repository instructions cannot admit an unrelated session:

```bash
fak trajectory audit --since 4d --user-contains qwen \
  --jsonl qwen-audit.jsonl --md qwen-audit.md
```

The report exposes discovered, matched, and scanned file counts so a topical
claim retains its denominator. `--user-contains` is a case-insensitive literal,
not a regular expression.

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

## Meta-process blocker triage

For a topical cohort, rank these process signals before reading individual sessions:

1. **Scope contamination** — compare matched with discovered files; use
   `--user-contains`, never raw transcript grep, because tool output can echo a topic.
2. **Token concentration** — `top_ten_token_fraction` shows whether a few runaway
   sessions dominate the aggregate; inspect those bottleneck rows first.
3. **Tool failure burden** — `tool_errors/tool_calls` measures execution friction.
4. **Zero-usage launches** — `empty_usage_files` exposes starts that produced no
   billable model progress.
5. **Fragment duplication** — `duplicate_fragments` exposes one session ID spread
   over multiple transcript files; do not count those as independent workers.
6. **Repeated identical failures** — use `repeated_failures`; retry loops need a
   changed hypothesis, not another identical call.
7. **Mutation churn** — use `mutation_churn`; repeated writes to one target indicate
   insufficient repro-first scoping.
8. **Hook latency** — use `hook_p95_ms`; separate gate wait from model/kernel work.
9. **Input amplification** — use `input_output_ratio`; large values warrant context
   paging or a narrower worker packet.
10. **Evidence refusal** — any `refused_records` means the parser found an unsupported
    shape; fix coverage before making a broad efficiency claim.

The first five now have explicit JSON summary fields. The remaining five preserve
existing exact audit fields. Treat each as a hypothesis until the corresponding
session rows or refusal rows identify the concrete cause.

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
5. Audit the signal-bearing sessions against the source transcript. Rank by the
   signal being explained, then inspect at least the top three sessions (or all
   sessions when fewer than three carry it). Keep transcript content private:
   record the session ID, aggregate signal counts, tool names, normalized error
   class, and mutated target only. Do not copy prompts, arguments, outputs, or
   machine-private absolute paths into reports or issues.
6. Classify each witnessed pattern before recommending work:
   - **skill-solvable**: the relevant skill omits a repeatable action that would
     have prevented the failure or churn. Update that canonical skill and add a
     checkable example or assertion in the same change.
   - **tool-solvable**: correct skill use still encounters an avoidable tool or
     telemetry limitation. File or reconcile one deduplicated issue with the
     aggregate witness and done condition.
   - **task-inherent / efficient reuse**: retries, edits, or cache reads are
     justified by the task and next-best alternative. Record no action; a high
     count alone is not debt.
   - **insufficient evidence**: the aggregate points at a session but transcript
     evidence cannot establish cause. File only a bounded instrumentation issue,
     never a speculative guidance change.
7. If `--baseline` was supplied, surface only the delta rows that are comparable
   regressions.

For repeated failures, group normalized error classes across the sampled
sessions. A one-off malformed command is not skill debt. Repeated timeout polls,
invalid invocation shapes, or the same failed recovery sequence become skill
candidates only when the owning skill can name a concrete replacement action.
For mutation churn, distinguish intentional iterative refinement from avoidable
re-touching: check whether the same target was changed repeatedly because the
skill skipped read-before-write, a witness-first repro, or a single coherent
edit. Never infer cause from `mutation_churn` alone.

When the request includes remediation, this audit is the evidence-gathering
phase rather than the stopping point. Apply proven skill-solvable fixes; dedupe
remaining work against open and closed GitHub issues; create issues only for
unresolved, witnessed defects. Put operational artifacts under allocated
scratch or the repository's declared witness path, not loose in the repo root.

Do not infer that a missing root means zero activity; the coverage row records
`root_present: false`. Do not sum a vendor price across harnesses: this spine is
token-exact and deliberately does not fabricate a blended dollar rate.

## Retired Python boundary

The former Python auditor and import API are retired. `fak trajectory audit` is
the only operational audit path. Extend its Go fixtures when a supported
transcript shape changes; never estimate an unsupported shape or revive a
second parser. The checked retirement inventory and intentional omissions live
in `internal/trajectory/PYTHON_PARITY.md`.

## Output

- Versioned JSONL for automation and direct analytical queries.
- Markdown carrying the same totals, coverage, bottleneck, baseline, and
  unsupported-shape evidence.
- A short operator summary naming the exact totals, rank-1 bottleneck, any
  comparable regression, and the artifact paths.

For a report-only request, write only the requested artifacts. For an explicit
remediation request, the output also includes canonical skill edits and
reconciled issue URLs, each tied to privacy-safe aggregate evidence.