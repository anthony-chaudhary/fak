# Post-compaction regrowth corpus witness — 2026-07-18 (#4768)

Captured output of `fak session compact-audit` over the local Codex rollout corpus
(`~/.codex/sessions`, rollouts modified since 2026-06-15), extending the #4763 witness
(`corpus-witness-2026-07-16.md`) with the per-fire regrowth decomposition. Scrubbed:
aggregate-only, no filesystem paths, no session cwd, no prompt or tool-output bodies —
attribution is measured from row lengths and content hashes at read time, so bodies
never enter a report.

This artifact exists to prove the reproduction the issue asks for: the ~700 observed
>=200k rebounds, the ~22-minute median, the 15/30-minute buckets, and a ranked
content-class attribution of what rebuilt the window.

## Headline (reproduces the issue)

| quantity | issue audit (2026-07-14) | this run (2026-07-18) |
|---|---|---|
| fires with subsequent telemetry | 1,044 | 1,510 |
| rebounds to >=200k resident before session end | 699 | 707 |
| median seconds back to 200k | 1,322 (22.0 min) | 1,325.5 (22.1 min) |
| p90 seconds back to 200k | 2,705 | 2,939 |
| median token samples back to 200k | 70 | 70 |
| rebounds within 15 minutes | 152 | 152 |
| rebounds within 30 minutes | 508 | 508 |
| zero-second (timestamp-suspect) rebounds | 2 | 0 |

The corpus grew between the passes (1,044 -> 1,510 observable fires), yet the median,
the sample count, and both time buckets reproduce almost exactly — the 22-minute
rebound is a stable property of these sessions, not an artifact of one snapshot.

The issue's two zero-second cases do not recur in this window: no rebound crossing
here shows a non-positive wall clock. The analyzer types such crossings
`TIMESTAMP_SUSPECT` (low confidence) and excludes them from every wall-clock
statistic — proven by the `timestamp-ambiguity.jsonl` fixture — rather than
presenting impossible velocity as fact.

## Ranked regrowth attribution (transcript bytes appended post-fire)

1.55 GB of transcript rows were appended inside the 1,510 post-fire windows:

| rank | class | bytes | share | rows | duplicated |
|---|---|---|---|---|---|
| 1 | `tool_result/shell_command` | 1,032,156,451 | 66.4% | 223,643 | 12,458,846 B in 1,587 rows |
| 2 | `reasoning` | 151,378,001 | 9.7% | 62,878 | — |
| 3 | `tool_call/shell_command` | 144,045,987 | 9.3% | 223,699 | 411,546 B in 154 rows |
| 4 | `compaction_summary` | 94,336,036 | 6.1% | 1,512 | — |
| 5 | `message/assistant` | 40,449,589 | 2.6% | 64,991 | — |
| 6 | `tool_call/apply_patch` | 28,846,676 | 1.9% | 11,169 | 247,950 B in 74 rows |
| 7 | `message/user` | 18,360,488 | 1.2% | 3,375 | — |
| 8 | `tool_result/exec` | 8,937,663 | 0.6% | 2,218 | — |
| 9 | `tool_call/update_plan` | 7,889,542 | 0.5% | 9,363 | — |
| 10 | `tool_result/view_image` | 5,915,634 | 0.4% | 32 | — |
| 11 | `instructions` | 4,743,754 | 0.3% | 325 | **3,390,235 B in 134 rows (71%)** |
| 12 | `tool_result/apply_patch` | 4,192,532 | 0.3% | 11,168 | — |
| — | (remaining classes) | 8,444,665 | 0.5% | | |

Dominant classes, read against the anomaly tally
(`REPEATED_TOOL_RESULT` 296 windows, `DUPLICATE_SETUP_REINJECTION` 85,
`OVERSIZED_EVENT` 5, `SUFFIX_RECREATION` 3):

- **Shell tool results are two thirds of all regrowth.** 296 of 1,510 windows
  (19.6%) contain byte-identical repeated tool-result spans under different call
  ids. This is the top independently actionable class — child issue #5254.
- **Instruction/skill reinjection is small in bytes but 71% duplicated** — the same
  setup payload re-enters the window after fires, in 85 windows — child issue #5255.
- **`compaction_summary` averages ~62 KB per fire** — the compactor's own injected
  suffix. That is #3071's suffix-burst axis, so it is linked there rather than
  duplicated into a new child.

## Cache join: fast regrowth is priced net of reuse

Median cache-read share of window input tokens: **94.2%** (fast cohort 96.4%, slow
cohort 89.0%). Codex telemetry reports cache reads (`cached_input_tokens`), not cache
creation; creation-side pricing stays with #2785.

## Fast vs slower rebounds: the work is the same

| cohort | windows | median tool calls | median growth tokens | median cache-read |
|---|---|---|---|---|
| fast (rebound <= 30 min) | 508 | 170 | 215,945 | 96.4% |
| slower / censored | 1,002 | 172 | 81,702 | 89.0% |

Fast-rebound windows run the same tool volume as slow ones — they are not doing a
different (wasteful) kind of work; they simply retain more of it, mostly as repeated
shell output. "Do not optimize away useful work" holds: the lever is deduplicating
repeated result spans, not throttling tool use.

## Scrubbed aggregate (regrowth block)

Full machine form of the regrowth roll-up from this run (the #4763 fields of the same
run match `corpus-witness-2026-07-16.md` in shape and are omitted here):

```json
{
  "regrowth": {
    "fires_with_telemetry": 1510,
    "rebounds": 707,
    "censored": 803,
    "timestamp_suspect": 0,
    "median_seconds_to_rebound": 1325.535,
    "p90_seconds_to_rebound": 2939.012,
    "median_samples_to_rebound": 70,
    "rebounds_within_15min": 152,
    "rebounds_within_30min": 508,
    "median_next_fire_seconds": 1912.5435,
    "median_cache_read_fraction": 0.9416,
    "class_totals": {
      "compaction_summary": {
        "rows": 1512,
        "bytes": 94336036
      },
      "instructions": {
        "rows": 325,
        "bytes": 4743754,
        "dup_rows": 134,
        "dup_bytes": 3390235
      },
      "message/assistant": {
        "rows": 64991,
        "bytes": 40449589
      },
      "message/developer": {
        "rows": 3066,
        "bytes": 1773500,
        "dup_rows": 1,
        "dup_bytes": 2121
      },
      "message/user": {
        "rows": 3375,
        "bytes": 18360488,
        "dup_rows": 1,
        "dup_bytes": 5877
      },
      "reasoning": {
        "rows": 62878,
        "bytes": 151378001
      },
      "tool_call/acme_lane_hint": {
        "rows": 1,
        "bytes": 449
      },
      "tool_call/apply_patch": {
        "rows": 11169,
        "bytes": 28846676,
        "dup_rows": 74,
        "dup_bytes": 247950
      },
      "tool_call/close_agent": {
        "rows": 56,
        "bytes": 23296
      },
      "tool_call/dos_arbitrate": {
        "rows": 520,
        "bytes": 254261
      },
      "tool_call/dos_check_reason": {
        "rows": 6,
        "bytes": 2529
      },
      "tool_call/dos_commit_audit": {
        "rows": 256,
        "bytes": 77843
      },
      "tool_call/dos_doctor": {
        "rows": 45,
        "bytes": 12174
      },
      "tool_call/dos_review": {
        "rows": 99,
        "bytes": 30127
      },
      "tool_call/exec": {
        "rows": 2218,
        "bytes": 2515369,
        "dup_rows": 3,
        "dup_bytes": 10218
      },
      "tool_call/fak_adjudicate": {
        "rows": 3,
        "bytes": 1692
      },
      "tool_call/fak_changes": {
        "rows": 3,
        "bytes": 1110
      },
      "tool_call/fak_memory_drivers": {
        "rows": 1,
        "bytes": 366
      },
      "tool_call/fak_read": {
        "rows": 5,
        "bytes": 2020
      },
      "tool_call/fak_revoke": {
        "rows": 1,
        "bytes": 436
      },
      "tool_call/fak_session_reset": {
        "rows": 1,
        "bytes": 457
      },
      "tool_call/followup_task": {
        "rows": 1,
        "bytes": 1118
      },
      "tool_call/get_goal": {
        "rows": 1092,
        "bytes": 335125
      },
      "tool_call/list_agents": {
        "rows": 2,
        "bytes": 728
      },
      "tool_call/list_mcp_resource_templates": {
        "rows": 2,
        "bytes": 656
      },
      "tool_call/list_mcp_resources": {
        "rows": 2,
        "bytes": 638
      },
      "tool_call/read_mcp_resource": {
        "rows": 4,
        "bytes": 1454
      },
      "tool_call/request_user_input": {
        "rows": 4,
        "bytes": 3953
      },
      "tool_call/send_input": {
        "rows": 2,
        "bytes": 1384
      },
      "tool_call/send_message": {
        "rows": 42,
        "bytes": 50590
      },
      "tool_call/shell_command": {
        "rows": 223699,
        "bytes": 144045987,
        "dup_rows": 154,
        "dup_bytes": 411546
      },
      "tool_call/spawn_agent": {
        "rows": 75,
        "bytes": 96497,
        "dup_rows": 1,
        "dup_bytes": 2092
      },
      "tool_call/update_goal": {
        "rows": 167,
        "bytes": 55874
      },
      "tool_call/update_plan": {
        "rows": 9363,
        "bytes": 7889542
      },
      "tool_call/view_image": {
        "rows": 32,
        "bytes": 10817
      },
      "tool_call/wait": {
        "rows": 1301,
        "bytes": 509614
      },
      "tool_call/wait_agent": {
        "rows": 60,
        "bytes": 27315
      },
      "tool_result/acme_lane_hint": {
        "rows": 1,
        "bytes": 371
      },
      "tool_result/apply_patch": {
        "rows": 11168,
        "bytes": 4192532
      },
      "tool_result/close_agent": {
        "rows": 56,
        "bytes": 149623
      },
      "tool_result/dos_arbitrate": {
        "rows": 520,
        "bytes": 519342
      },
      "tool_result/dos_check_reason": {
        "rows": 6,
        "bytes": 6465
      },
      "tool_result/dos_commit_audit": {
        "rows": 256,
        "bytes": 244929
      },
      "tool_result/dos_doctor": {
        "rows": 45,
        "bytes": 179905
      },
      "tool_result/dos_review": {
        "rows": 99,
        "bytes": 271054
      },
      "tool_result/exec": {
        "rows": 2218,
        "bytes": 8937663,
        "dup_rows": 1,
        "dup_bytes": 4589
      },
      "tool_result/fak_adjudicate": {
        "rows": 3,
        "bytes": 1501
      },
      "tool_result/fak_changes": {
        "rows": 3,
        "bytes": 1265
      },
      "tool_result/fak_memory_drivers": {
        "rows": 1,
        "bytes": 5568
      },
      "tool_result/fak_read": {
        "rows": 5,
        "bytes": 3806
      },
      "tool_result/fak_revoke": {
        "rows": 1,
        "bytes": 455
      },
      "tool_result/fak_session_reset": {
        "rows": 1,
        "bytes": 871
      },
      "tool_result/followup_task": {
        "rows": 1,
        "bytes": 256
      },
      "tool_result/get_goal": {
        "rows": 1092,
        "bytes": 713384
      },
      "tool_result/list_agents": {
        "rows": 2,
        "bytes": 5848
      },
      "tool_result/list_mcp_resource_templates": {
        "rows": 2,
        "bytes": 2418
      },
      "tool_result/list_mcp_resources": {
        "rows": 2,
        "bytes": 46584
      },
      "tool_result/read_mcp_resource": {
        "rows": 4,
        "bytes": 1182
      },
      "tool_result/request_user_input": {
        "rows": 4,
        "bytes": 1220
      },
      "tool_result/send_input": {
        "rows": 2,
        "bytes": 632
      },
      "tool_result/send_message": {
        "rows": 42,
        "bytes": 10752
      },
      "tool_result/shell_command": {
        "rows": 223643,
        "bytes": 1032156451,
        "dup_rows": 1587,
        "dup_bytes": 12458846
      },
      "tool_result/spawn_agent": {
        "rows": 75,
        "bytes": 24874
      },
      "tool_result/unknown": {
        "rows": 1,
        "bytes": 340
      },
      "tool_result/update_goal": {
        "rows": 167,
        "bytes": 153255
      },
      "tool_result/update_plan": {
        "rows": 9362,
        "bytes": 2323832
      },
      "tool_result/view_image": {
        "rows": 32,
        "bytes": 5915634
      },
      "tool_result/wait": {
        "rows": 1299,
        "bytes": 1765331
      },
      "tool_result/wait_agent": {
        "rows": 60,
        "bytes": 95629
      },
      "unknown": {
        "rows": 448,
        "bytes": 961782
      }
    },
    "anomaly_counts": {
      "DUPLICATE_SETUP_REINJECTION": 85,
      "OVERSIZED_EVENT": 5,
      "REPEATED_TOOL_RESULT": 296,
      "SUFFIX_RECREATION": 3
    },
    "fast_cohort": {
      "windows": 508,
      "median_tool_calls": 170,
      "median_turns": 1,
      "median_growth_tokens": 215945,
      "median_cache_read_fraction": 0.9638
    },
    "slow_cohort": {
      "windows": 1002,
      "median_tool_calls": 172,
      "median_turns": 1,
      "median_growth_tokens": 81702,
      "median_cache_read_fraction": 0.8903
    }
  }
}
```

Reproduce with:

```
fak session compact-audit --since 2026-06-15 --json --scrub --aggregate-only
```
