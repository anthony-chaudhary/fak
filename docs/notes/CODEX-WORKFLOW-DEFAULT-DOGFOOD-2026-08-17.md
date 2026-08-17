# Guarded Codex workflow-default dogfood — 2026-08-17

## Verdict

The first paired run proves **uptake**, not productivity: the guarded-default arm launched fak-native guarded Codex workers for 3/5 frozen tasks, explicitly stayed direct for the trivial task, and missed one reasonable multi-step serial task. The explicit-direct control launched 0/5. All three launched workers completed, but this run cannot claim a time or token saving because the control arm intentionally measured routing only and did not execute equivalent task work.

## Frozen protocol

- Task set: five privacy-safe task shapes (`trivial`, `multi-step-serial`, `parallelizable`, `unattended`, `mixed-grind`).
- Same resolver/model envelope: `fak@2a2b8984b540`, Codex `gpt-5.6-sol/xhigh`, worker cap 2, one read-only worker role.
- Guarded-default arm: `--profile auto --launch`.
- Explicit-direct control: `--profile off --launch`; this records a typed decline and launches no worker.
- Frozen task manifest SHA-256: `6B406ABFA9F14A191BF163BB7C8A562704C71A6B6FB2B6E94AF7CE66293C7EA9`.
- Raw local artifacts: `_scratch/codex-workflow-paired-dogfood/run-20260817-1/{tasks.json,results.json,worker-outcomes.json}`. These paths are intentionally local rather than committed because worker logs may contain repository excerpts.

## Paired outcomes

| Task | Shape | Guarded-default | Explicit-direct | Default elapsed | Control elapsed |
|---|---|---|---|---:|---:|
| T1 | trivial | declined: `resolved-direct` | declined: `resolved-direct` | 676 ms | 87 ms |
| T2 | multi-step serial | declined: `resolved-direct` | declined: `resolved-direct` | 201 ms | 153 ms |
| T3 | parallelizable | launched 1 worker | declined: `resolved-direct` | 3,086 ms | 111 ms |
| T4 | unattended | launched 1 worker | declined: `resolved-direct` | 5,974 ms | 48 ms |
| T5 | mixed grind | launched 1 worker | declined: `resolved-direct` | 3,062 ms | 42 ms |

The elapsed values cover plan/decline or launch admission only. They do not represent completed-task latency and must not be compared as a productivity result.

## Error readout

- False-positive orchestration on trivial work: **0/1**.
- False-negative direct execution on reasonable multi-step work: **1/4** (T2). The classifier does not currently recognize that serial three-clause phrase as grind; this is observed classifier debt, not evidence that direct execution was better.
- Positive uptake on non-trivial work: **3/4**.
- Explicit-direct control launch leakage: **0/5**.

## Worker and token evidence

All three launched guarded workers reached a `turn.completed` event:

| Task | Input tokens | Cached input | Output tokens | Final-message SHA-256 |
|---|---:|---:|---:|---|
| T3 | 367,201 | 194,048 | 3,576 | `4F456E89D46BA61423C0DBFAF9134AEBF01E4D4EF2ADDEFCEB5A2A1896340123` |
| T4 | 297,461 | 167,936 | 4,957 | `2A42411C3CB65D71C5D48F3E1CDC56A0D56866BD3E379ED6BF69356C010E14BA` |
| T5 | 312,210 | 180,224 | 4,626 | `B3E27E182BC187BE5F044EDC43E862637AABA717DA5DC9A24D9DE14EE547CC47` |

These are provider-reported usage counters from the guarded worker JSONL. They prove substantial billed-work exposure and cache use; they do **not** prove net-token savings. The control arm did no equivalent task execution, so a causal token comparison is unavailable.

## Provenance boundaries

- **FAK-authored:** first-turn classification/injection witnesses, orchestration invocation and launch receipts, run IDs, typed decline reasons, process IDs, and aggregate joins.
- **Provider-observed:** worker `turn.completed` and usage counters.
- **Not independently witnessed in this protocol:** correctness or usefulness of each worker's report. The final-message digest proves artifact identity only. Independent witness status is therefore **not captured**, and no correctness/productivity claim is made.

This run closes the “did the default cause real workflow invocation and worker launch?” question. It leaves two explicit debts: improve serial multi-step classification, and run a later study where both arms complete equivalent work under an independent outcome grader before reporting productivity or net-token gains.
