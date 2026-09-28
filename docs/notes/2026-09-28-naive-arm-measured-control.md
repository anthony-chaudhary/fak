---
title: "The naive dispatch arm as a measured control (fak naive-control)"
description: "internal/naivecontrol and fak naive-control record the 45-character naive arm and the orchestrated dispatch loop on one ruler: git-ancestry-verified ship counts, an explicit MISSING_MEASUREMENT state instead of zeros, and one side-by-side screen. Includes the field mapping, the metrics with no orchestrated counterpart, and the first live measurements."
---

# 2026-09-28: the naive dispatch arm as a measured control

> TL;DR: `fak naive-control` records the naive arm beside the orchestrated loop, with git-verified ship counts.

`docs/ops-unsticking.md` item 6 was "Stand up the naive arm as a measured control.
OPEN". This note records what now exists, how the two arms line up field by field,
and what the first real run measured. Measured at `HEAD 2479ddc55`.

## What shipped

- `internal/naivecontrol`: the row schema (`fak-naive-control/1`), the git-ancestry
  verifier, the pure fold, and the comparison (`fak-naive-control-compare/1`).
- `fak naive-control run | harvest | orchestrated | compare` (`cmd/fak/naive_control.go`).
  The ledger is `.fak/naive-control.jsonl` (gitignored runtime state).

The lifecycle of one naive run:

```
fak naive-control run                    # runs the arm, records picks + HEAD as the base (PLANNED)
# ... a headless session works the picks ...
fak naive-control harvest --run-id ID    # git-verifies the picks (HARVESTED)
fak naive-control orchestrated --since 24h
fak naive-control compare --since 720h
```

Schedule it with the loop runner: `fak loop run --loop naive-control --source cron -- fak naive-control run`.

## The two rules

**A ship count comes only from git.** For every pick, the verifier scans
`base..HEAD` for subjects citing `#N` (the same `dispatchtick.SubjectCitesIssue` key
the orchestrated witness sweep uses). Then every claimed SHA goes through the same
check. That covers worker receipts, harvest receipts, and orchestrated `.witness`
sidecars alike:

1. `git rev-parse --verify --quiet <sha>^{commit}`: exit 1 means no such commit, so
   the claim is `NOT_LANDED`. A fabricated SHA counts as zero here.
2. `assumecheck.GitAncestryDriver`: reachable from HEAD and not reverted is
   `LANDED`. Any git failure is `UNVERIFIABLE`, which is never counted either way.

A pick is `NOT_SHIPPED` only when that pick's own `base..HEAD` scan completed.
Without a base, a missing commit proves nothing, and the pick stays `UNKNOWN`.

**An unread field is `MISSING_MEASUREMENT`, rendered `UNKNOWN`, never 0.** Every metric
is `{"state":"MEASURED","value":N}` or `{"state":"MISSING_MEASUREMENT","reason":"..."}`.
A field that is absent from a decoded row also reads as missing. An arm total is
MEASURED only when every run in it measured the field. Otherwise the partial sum is
shown next to `UNKNOWN`, not in place of it. `compare --since` narrows the window by
recording time, never by value.

## Field mapping against the orchestrated path

| naive-control field | orchestrated source | alignment |
|---|---|---|
| `arm` | none; `loop_id` family `issue-resolve-dispatch/*` separates loops | new field, both arms share one ledger |
| `run_id`, `ts_unix_nano` | `loopmgr.Event.run_id`, `.ts_unix_nano` | aligned names and units |
| `picks_offered` | exited workers (`resolve-*.witness`) in the window | aligned on "issue handed to a worker"; `lane_issue_count` is a candidate pool, not picks, and is not used |
| `picks_shipped` | `.witness` claims, re-verified | aligned: same binding key, same `base..HEAD` scan, same verifier |
| `commits_claimed` | non-null `.witness` `sha` | aligned (both are self-reports) |
| `commits_landed` | nothing aggregated today | partial: `dispatch status` `commits` is an unattributed `rev-list --count`, so it is not used |
| `controller_ms` | sum of `tick_total_ms` in the window (`.fak/loops.jsonl`) | partial: one planning command vs per-tick admission plus spawn |
| `wall_ms` | none | no orchestrated counterpart |
| `cost_usd` | none | no counterpart on either arm |

### Metrics with no orchestrated counterpart

- **`wall_ms` (end-to-end run time).** Orchestrated workers are detached, and no worker
  end event is recorded. The loop `end` event is written at spawn time with status
  `claimed_done`. The `.witness` sidecar has no timestamp. `fak loop economics` sums
  `duration_ms`, which dispatch never emits (0 of 425 events). `tick_total_ms` is
  controller time only, so it maps to `controller_ms`, not to `wall_ms`. The
  orchestrated arm records `wall_ms` as UNKNOWN with that reason.
- **`cost_usd`.** Neither path records a per-run dollar figure. The guard exit
  summary is free text, and all 50 on-disk worker logs say `DOLLAR-BLIND`. The
  `fak dispatch sessions` `cost` is in input-token-equivalents, not USD. Loop-ledger
  metrics are `int64`-only. The naive arm takes cost only from
  `harvest --cost-usd`, for example a `claude -p --output-format json`
  `total_cost_usd`. Nothing is estimated to fill the column.

## First measurements (2026-09-28)

- The literal 45-character arm fails in this checkout. It exits 2: `no issues
  file specified and none found at .fak/issues.json or issues.json`. The run was
  recorded with picks `UNKNOWN` and the reason, and the verb exits 1.
- `--top 10` offers 4 picks. With `--live`, wave 1 was issues 13579, 13587, 13577
  and 13580. The wave size is clamped by adaptive concurrency (default 4,
  `min(cpu/2, 4)`). The arm is also plan-only: without `--spawn-opencode` it
  launches nothing.
- Orchestrated arm, 700h window: 50 picks and 34 claimed SHAs. Every distinct claimed
  SHA verified `LANDED`. By git, 29 picks shipped and 4 did not. 17 picks are
  `UNKNOWN` because their worker left no `.basesha`, so `picks_shipped` reads UNKNOWN
  with the 29 disclosed as a lower bound. Controller time was 1h50m.

## Next checkable steps

1. Give the naive arm an input it can run: pass `--live` or keep `.fak/issues.json`
   fresh. Then schedule `run` daily and `harvest` after each fanout.
2. Write `.basesha` for every orchestrated spawn, so orchestrated picks can be proven
   not shipped instead of reading `UNKNOWN`.
3. Emit a worker end event (or `duration_ms`) from dispatch. That gives `wall_ms` an
   orchestrated counterpart.
