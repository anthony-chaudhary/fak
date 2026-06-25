---
name: dos-witness-claim
description: "Verify subagent or worker results before folding them into a synthesis. Use when another agent claims it shipped, created, found, or changed something and your next step would otherwise trust its return string."
---

# dos-witness-claim - fold witnessed effects, not narration

A worker result is a claim, not a fact. Before using it as input to another
prompt, report, plan, or commit, witness the claimed effect from a surface the
worker did not author. Fold only confirmed effects; route unconfirmed results
separately and count them in the denominator.

## Load-Bearing Rule

Never interpolate a worker's return string directly into a synthesis as ground
truth. Extract the checkable effect, gather a non-agent-authored witness, and
fold only the confirmed bucket.

## Step 0 - Discover The Workspace

```bash
dos doctor --workspace . --json
```

Read `paths`, `lanes`, and `stamp` from the result. If a claim names a
`(plan, phase)`, `dos verify` applies this workspace's stamp grammar; do not
grep commit subjects yourself.

## Step 1 - Check Worker Terminal State

If you have a worker transcript path, verify the result is not a harness-authored
death before reading it as a worker result:

```bash
dos verify-result --workspace . --transcript <agent-transcript.jsonl>
```

Route by exit code:

- `0`: terminal result is usable for claim extraction; still witness the effect.
- `3`: worker is dead or synthetic. Do not fold its return string; re-dispatch or
  surface the failed unit.
- `2`: contract error. Fix the transcript wiring; do not treat it as healthy.

If `verify-result` is not available in the installed DOS version, log the gap and
continue to effect witnessing. The gap is not a pass.

## Step 2 - Extract The Checkable Claim

Classify each result into one of these claim types:

| Claim type | Example | Witness |
|---|---|---|
| Git phase | "`PLAN` / `PHASE` shipped" | `dos verify` over git ancestry and stamp grammar |
| Commit claim | "commit `<sha>` did X" | `dos commit-audit <sha>` |
| File/effect claim | "created file X", "sent id Y" | fresh read-back, OS exit code, API GET, or state diff |
| No checkable claim | "done", "looks good" | no fold; surface as `NO_CLAIM` |

Abstain rather than inventing identifiers from prose. A result without a named
effect is not confirmed.

## Step 3 - Witness Git-Backed Claims

For a `(plan, phase)` claim:

```bash
dos verify --workspace . <PLAN> <PHASE> --json
```

Fold only when the exit code is shipped and the JSON carries a positive source
such as `registry` or `grep-subject`. Preserve the source rung in the synthesis;
`grep-subject` is weaker than a registry row.

For a commit claim:

```bash
dos commit-audit --workspace . <sha>
```

Treat `OK` / `diff-witnessed` as foldable shape evidence. Treat
`CLAIM_UNWITNESSED` as residual review work, not as success.

## Step 4 - Witness Non-Git Effects

There is no generic `dos verify-effect` CLI verb. For created files, DB rows,
messages, deploys, or external side effects, gather a fresh read-back from a
surface the worker did not author:

- file claim: fresh filesystem read/stat of the exact path;
- API claim: GET the object by id;
- DB claim: read the row by key;
- command claim: run the relevant command and check its exit/status output.

If the read-back confirms the effect, fold it as `CONFIRMED`. If it contradicts
the claim, route it as `REFUTED`. If no accountable read-back exists, route it as
`UNWITNESSED`; never fold it as success.

## Step 5 - Return A Partition, Not A Story

Return results in four buckets:

- `CONFIRMED`: safe to fold into the next prompt or report.
- `REFUTED` / `NOT_SHIPPED` / `DEAD`: do not fold; re-dispatch or surface.
- `UNWITNESSED`: do not fold; identify the missing witness.
- `NO_CLAIM`: do not fold; ask for a checkable effect if needed.

Carry coverage forward:

```text
confirmed M of N declared results; refuted R; unwitnessed U; no-claim C
```

Downstream synthesis must see the coverage. Do not let `4/7 confirmed` become
`7/7 returned`.

## Anti-Patterns

- Folding `${result}` directly because it is non-empty.
- Asking a model whether a worker's own output looks complete.
- Counting only returned strings in the denominator.
- Treating `UNWITNESSED` or `NO_CLAIM` as probably successful.
- Grepping commit subjects yourself instead of calling `dos verify`.
