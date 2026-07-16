# Compaction-health corpus witness — 2026-07-16 (#4763)

Captured output of `fak session compact-audit` over the local Codex rollout corpus
(`~/.codex/sessions`, rollouts modified since 2026-06-15). Scrubbed: aggregate-only,
no filesystem paths, no session cwd, no prompt or tool-output bodies — the miner is
body-blind at read time, so there is nothing to redact beyond paths.

This artifact exists to prove the reproduction the issue asks for: the corpus counts,
the median resident-context shed, and that a byte-large / cumulatively-huge session is
read as **repeatedly compacted**, not as "not compacting".

## Headline (reproduces the issue)

| quantity | issue audit (2026-07-14) | this run (2026-07-16) |
|---|---|---|
| fak sessions scanned | 2,125 | 2,398 |
| measurable pre/post fire pairs | 1,041 | 1,304 |
| median pre-fire resident tokens | 242,288 | 240,361 |
| median post-fire resident tokens | 25,568 | 24,609 |
| median shed | 216,158 | 214,610 |
| median post/pre ratio | 0.11 | 0.1106 |

The 27-fire largest session named in the issue
(`019f1be3-94a4-7ba2-87c8-56b934eae312`) is classified `FIRED_WITH_ANOMALIES` with
**27 fires**, a 34.7 MB append-only rollout, 369,265,152 cumulative input tokens, and
peak **resident** context 244,688 / 258,400 — i.e. repeatedly compacted at the ceiling,
exactly as the issue predicts. Reading its file size or cumulative tokens as "unbounded"
is the observability failure this command removes.

## Aggregate

```json
{
  "aggregate": {
    "sessions": 2398,
    "rollout_bytes": 2788432432,
    "fires": 1307,
    "measured_fires": 1304,
    "compacted_sessions": 396,
    "median_pre_tokens": 240361,
    "median_post_tokens": 24609,
    "median_shed_tokens": 214610,
    "median_residual_ratio": 0.1106,
    "anomaly_counts": {
      "FAST_REBOUND": 161,
      "INEFFECTIVE_FIRE": 19,
      "LATE_FIRE": 37,
      "MISSING_POST_WITNESS": 3,
      "NO_FIRE_ABOVE_CEILING": 32,
      "OVERSIZED_RESIDUAL": 4
    },
    "verdict_counts": {
      "FIRED_AND_HELD": 209,
      "FIRED_WITH_ANOMALIES": 187,
      "NO_FIRE_ABOVE_CEILING": 32,
      "NO_FIRE_BOUNDED": 1912,
      "NO_TELEMETRY": 58
    }
  }
}
```

Every anomaly class the Definition of Done enumerates is witnessed on real data:
`NO_FIRE_ABOVE_CEILING`, `LATE_FIRE`, `INEFFECTIVE_FIRE`, `OVERSIZED_RESIDUAL`,
`FAST_REBOUND`, `MISSING_POST_WITNESS` (duplicate-fire is proven by the
`paired-duplicate.jsonl` fixture; no live duplicate occurred in this window).

Reproduce with:

```
fak session compact-audit --since 2026-06-15 --json --scrub --aggregate-only
```
