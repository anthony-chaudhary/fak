---
name: debt-orchestrator
description: Coordinate bounded, evidence-backed maturity debt work in the current repository, with isolated workers and independent verification. Use for debt burndowns or explicitly requested sustained campaigns.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[repository or lane] [wave size] [explicit campaign goal and budget] (no args = one bounded pass)"
---

# Debt orchestration: select useful work, verify, resume

Retire concrete maturity gaps using the current repository's instructions, ownership
mechanisms, and evidence. The CLI plans work; the coordinator executes and witnesses it.
Use the invoking repository as the default scope. Include a sibling only when the
operator names it or an evidenced dependency requires it; preserve public/private placement.

## Scope and stop condition

Default to one bounded pass with at most one admitted wave. Choose 1–3 small outcomes
and cap workers by available slots, verified disjoint scope, and budget. A recurring
schedule (including a 20-minute cadence) neither supplies a runtime deadline nor proves
that the previous run finished. Inspect active ownership before dispatch; skip held
work and leave a resumable receipt. Do not launch another wave to fill idle capacity.

For an explicitly sustained goal, continue through verified waves within its stated
scope, time/spend limits, and completion criterion. Reassess after each wave; stop at
the achieved goal, exhausted budget, or a checkable blocker. Record the next action
when the pass ends. Never turn an unspecified invocation into a letter-grade campaign.

Read the current objective, latest compact receipt, and changed evidence once. Reuse
an existing matching issue/plan rather than creating a fresh campaign every run. Follow
repository issue-tracking policy within the user's authorization; this skill does not
authorize sending messages. Keep one bounded current-state summary, with links to
owned artifacts; do not replay or append entire transcripts into continuation state.

## Baseline and selection

Resolve explicit absolute repository roots and allocate a unique run scratch directory
through the repository's supported allocator. Check the installed CLI help before using
source-only capabilities. Current source implements `fak debt-orchestrator` as
`fak debt-lanes --plan-waves`; it does not dispatch workers, acquire leases, or land work.

Public scan and plan examples (replace placeholders with resolved paths):

```text
fak debt-lanes --workspace <public-root> --target-repo fak --json
fak debt-orchestrator --workspace <public-root> --target-repo fak --wave-size 3 --max-waves 1 --json
```

For private scope, explicitly pass `--target-repo fak-private --private-root <private-root>`
with the selected workspace. For authorized combined discovery, use the public root as
`--workspace`, `--target-repo both`, and the explicit private root. Folder-name inference
is unreliable in renamed worker checkouts; never infer scope from binary location.

Save the first command's **Report** JSON (`fak.maturity-debt-lane.v1`) as the baseline.
The orchestrator emits **WavePlan** JSON (`fak.debt-orchestrator-wave-plan.v1`), which is
not a compare baseline. `--compare <baseline>` emits text even with `--json`; confirm
schema and matching roots, target-repo, and filters before comparing because the parser does not check them.

Use `--lane`, `--criticality`, and `--min-gap` to narrow candidate work. `--top` limits
hotspot display, not the full set planned. `--max-waves 0` plans all candidate waves;
it does not authorize an endless execution loop. Validate user targets before passing
`--target-grade` or `--target-points` (`--points` alias): malformed grades may be ignored,
and point-based planning can stop on debt in the plan or potential realized points.
Those projections do not establish achieved gains.

Save one plan per fresh scope. Inspect a small candidate batch until the first useful,
ready unit is found; advance through saved candidates on holds. Refresh planning only
when relevant state changes or the candidate set is exhausted. Repeatedly replanning
the same held prefix is not progress.

Apply current priorities before debt rank: all-in-one serving/harness/memory, serving,
harness, then other work. Favor a witnessed blocker in a useful workload and the smallest
end-to-end improvement. Name the outcome, baseline, and independent witness. Debt rank
and test counts are discovery signals, not proof of deployed quality or useful throughput.

## Admit and dispatch

Treat the wave plan as advisory. Its graph scans `internal/`, `pkg/`, and `platform/`
under one workspace, checks direct imports by bare lane name, and samples one lane
journal. It does not prove complete cross-repository, transitive, `cmd/`, or `tools/`
safety. Journal discovery can fail open; lane names can collide across repositories.

Resolve every packet to repository + absolute root + exact write paths. Inspect current
DOS lane trees, dependency annotations, live leases, and contract locks in each affected
repository. Check public SDK/private consumer dependencies separately. Re-arbitrate
immediately before launch. `dos arbitrate` is a decision; acquire a durable lease through
the managed launcher or `dos lease-lane --workspace <root> acquire --lane <lane>`
`--owner <owner> --tree <paths...>` and retain the receipt. Never
substitute an empty lease set or a planner label for ownership. Defer refused scopes.

Use the repository's managed isolated worker flow: public detached
`fak worktree worker prepare|land|reap`, private `fak-flow start|checkpoint|land`.
Bind ownership to a persistent worker/supervisor lifetime, not the short-lived preparation
process. Each packet names one issue/outcome, exact paths, baseline revision, one
appropriate witness, budget, and stop condition. Delegate independent bounded units;
serialize shared contracts, core changes, or uncertain dependencies. Follow the actual
harness's model/worker routing; do not invent worker types or override user model choices.

## Witness and land

Workers return compact evidence: changed paths and revision, executed witness with result
and artifact location, remaining gap or refusal. The coordinator independently reads the
diff and actual result before accepting it. A mock echo, source declaration, successful
launch, or restored context is not execution evidence. If a peer already landed the work,
compare exact candidate files with that commit before attempting another land.

Choose verification proportional to the changed behavior and repository requirements.
Use scoped tests in isolation; execute the real CLI/protocol path for runtime changes.
Expand validation only for a new failure or an unresolved risk. Reuse compatible receipts
bound to the exact candidate; do not require arbitrary 10/50/100 matrices per lane.

For native inference/performance work, follow the current native-inference contract:
new work prefers Qwen3.8, runs fak-native end to end, and names the engine, model,
configuration, workload, and quality constraints in the receipt. Preserve historical
Qwen3.6 evidence and require an explicit task-specific exception for new use. Obtain the
required validity/budget review before expensive runs; measure matched end-to-end cost,
including failed attempts and verification. A correctness fix or maturity lift alone
cannot establish a performance multiplier.

Land only witnessed, owned paths through the repository's sanctioned commit/landing
flow, following its actual trailers and publication rules. Keep private evidence private
and use the repository's supported leak check for public material. Read back the resulting
commit and sync receipt before calling it shipped. Record source delivery, runtime
evidence, performance evidence, and queue state separately in the existing plan.

## Close the pass

Re-scan the same scope and compare the Report baseline. Explain denominator or scope
changes separately; never shrink the denominator to claim retirement. Report observed
outcomes and measured deltas separately from projected points. Keep a compact receipt:

`scope + revisions | accepted outcomes + witnesses | measured delta | held/unfinished | next action`

Release only this run's finished leases and reap only its witnessed landed scratch or
worktrees. Preserve unfinished owned work for recovery. Bulk worktree sweeps, cache
reclamation, and peer cleanup are separate maintenance work, not wave-boundary defaults.
