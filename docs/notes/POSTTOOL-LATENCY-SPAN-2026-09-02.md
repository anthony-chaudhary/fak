---
title: "Post-tool latency span: tool_result_recorded to next_model_item (#10662)"
description: "The first-class post-tool span in internal/codexlifecycle: definition, disjointness rule, closed band and attribution vocabularies, and the fak session-audit posttool readout."
---

# Post-tool latency span (#10662)

The audited defect: in Codex rollout journals the interval AFTER a tool result
is recorded and BEFORE the next model-emitted record grows with session age and
context size (median ~11s at tool-result ordinal 1-20, ~21s at 101-200, with the
hotspot at 50k-100k input tokens). `internal/codexlifecycle/posttool.go` makes
that interval a first-class span instead of an unnamed hole in the timeline.

## Definition

One PostToolSpan per `function_call_output` / `custom_tool_call_output` record:

    call ──ToolMS──▶ output ──GapMS──▶ next model-emitted record

A model-emitted record is `function_call`, `custom_tool_call`, `token_count`,
`task_complete`, or `turn_aborted` (the same set `decomposeTimeline` anchors on;
`ARecord.PayloadKind` recovers custom calls, so no JSON is re-parsed). A trailing
result with no next model-emitted record is skipped and counted as `tail_skipped`
in the report: an unclosed interval measures nothing.

## Disjointness (the no-double-counting witness)

ToolMS measures call → its own output; GapMS measures that output → the next
model-emitted record. The intervals share exactly one endpoint and cannot
overlap, so ToolMS + GapMS always equals the call → next-record interval. A slow
tool can never be booked as post-tool model latency. Interior `compacted`
records become subspans (`pre_compaction`, then one `compaction` segment per
edge) that tile GapMS exactly.

## Closed vocabularies

- Context band, from the first token_count with input_tokens > 0 at or after the
  gap end (that token_count reports the usage of the request that consumed this
  tool result): `unobserved`, `lt10k`, `10k_25k`, `25k_50k`, `50k_100k`,
  `gte100k`.
- Ordinal bucket, per-session 1-based tool-result ordinal: `1_20`, `21_50`,
  `51_100`, `101_200`, `gte201`.
- Attribution (correlation): `tool_slow` when ToolMS >= GapMS (the wait was the
  tool), `stall_capped` when GapMS > 300s (only the first 300s is model-active,
  the remainder idle, same rule as `decomposeTimeline`), `compaction_in_gap`
  when a compaction fired inside a non-stall gap, else `model_reasoning`.

## Correlation, not causation

Journal timestamps cannot separate provider TTFT from gateway queueing or
harness scheduling inside GapMS. Attribution tokens are CORRELATION aids over
observable structure; the report never claims a causal latency split. Live-path
emit belongs to #10636 and the timing inventory to #10621.

## Running it

    fak session-audit posttool --here --json

Flags mirror `fak session-audit codex`: `--root DIR` (default `~/.codex/sessions`,
honoring `CODEX_HOME`), `--cwd DIR|--here`, `--max N`. The text renderer prints
overall gap/tool percentiles plus one line per non-empty band and ordinal bucket
in canonical order, each with the tool_p50 control beside the gap percentiles.
Regression corpus and expected values: `internal/codexlifecycle/testdata/posttool/issue-10662/`.
