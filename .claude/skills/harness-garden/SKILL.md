---
name: harness-garden
description: Use when recent trajectories, notes, or refusals should become one evidence-backed harness improvement ticket and at most one guarded worker.
---

# Harness garden

Close one inward-looking improvement loop: **observe -> classify -> subtract or ticket -> admit one worker -> witness**. Compose existing project primitives; do not create another analytics stack, scheduler, or dispatch mechanism.

## Fixed bounds

One pass may update or file **at most one issue** and launch **at most one worker**. It may conclude `drop` when fak already supports the capability or evidence is insufficient. Keep private trajectory content out of public artifacts.

## 1. Observe recent work

Run the supported cross-harness audit into allocated scratch:

```powershell
fak tree-doctor --scratch-dir harness-garden --json
fak trajectory audit --since 7d --jsonl _scratch/harness-garden/recent.jsonl --md _scratch/harness-garden/recent.md
```

Read the report, recent operator notes, relevant guard/refusal evidence, and open/closed issues. Follow [`../trajectory-audit/SKILL.md`](../trajectory-audit/SKILL.md) for sampling, privacy, baseline, and behavior classification.

A nonzero audit is evidence, not an empty report. Preserve its typed refusal and partial counts; never infer that missing output means no bottleneck. If refusal prevents honest diagnosis, make that refusal the candidate defect or stop as `insufficient-evidence`.

## 2. Classify before proposing

Classify the highest-leverage repeated friction as exactly one:

- `subtract`: stale, overlapping, or unnecessary instruction/machinery;
- `config`: an existing primitive is correct but poorly defaulted or connected;
- `skill`: behavior can improve through concise, reusable guidance;
- `tool`: deterministic code or schema support is missing;
- `inherent`: no safe harness change can remove the constraint;
- `insufficient-evidence`: current evidence cannot support a change.

Prefer, in order: no change, deletion, configuration, an existing project primitive, standard library, then new machinery. Name the product defect separately from any guidance-side mitigation. Rank by repeated operator cost and breadth, not novelty.

## 3. Dedupe and choose one durable action

Search open and closed GitHub issues with the observed error class, affected primitive, and desired outcome. Choose exactly one:

1. `drop` - capability is present, evidence is weak, or cost exceeds value;
2. `update` - add the new witness or narrowed done condition to the matching issue;
3. `file` - create one issue only when no duplicate exists.

For `update` or `file`, include:

- **For / Problem / Today / Better because / Witness** against the real next-best alternative;
- centrality and P1-P4 from `docs/problems-we-solve.md`;
- smallest working spine, gold-plating boundary, done condition, operating envelope, privacy boundary, and follow-ons;
- audit command, bounded counts, typed refusals, and artifact paths rather than private transcript content.

Apply [`../ticket-scope/SKILL.md`](../ticket-scope/SKILL.md). Validate through the supported issue-contract surface when available. If unavailable, preserve the full contract instead of inventing a substitute tool.

## 4. Admit one bounded worker

Only after the durable issue exists, define a distinct file tree and ask DOS before launch:

```powershell
dos arbitrate --lane <leaf> --kind keyword --mode exclusive --tree <path>... --output json --explain
pwsh tools/launch_goal_detached.ps1 -PointerFile <allocated-prompt> -Workspace C:\work\fak -WorkKind gardening -PlanOnly
```

The packet names the issue, owned paths, non-goals, required test/witness, and stop condition. Use `fak worktree worker prepare|land|reap` for implementation isolation as required by `AGENTS.md`.

Launch at most one worker after admission and detached preview are green. Keep the spawn preflight enabled. Never use a bypass, raw background process, alternate account, or broader tree to route around an account refusal, schema refusal, lease collision, or another typed gate. On refusal, record the ready-to-run command and typed outcome; that is a prepared packet, not a launch or shipped work.

## 5. Witness and close

A worker self-report is not evidence. Follow [`../dos-witness-claim/SKILL.md`](../dos-witness-claim/SKILL.md), independently read the diff/artifact, run the scope-correct gate, and land only witnessed green work. Reconcile the issue with discoveries and follow-ons.

Report one claim per line:

- **Decision:** `drop | update #N | file #N` with the evidence artifact.
- **Bottleneck:** classification plus bounded count/refusal.
- **Worker:** `not planned | prepared | refused <TOKEN> | launched | witnessed <module@rev>`.
- **Next check:** one command or issue URL.

## Proof scenarios

A pass is not dogfooded until evidence covers:

1. existing capability -> `drop` and no issue;
2. duplicate -> update one issue, no second ticket;
3. absent capability -> one contract-complete issue;
4. trajectory schema refusal -> partial evidence preserved, no false clean claim;
5. account refusal -> worker remains prepared, not launched;
6. lease refusal -> colliding worker does not launch;
7. green worker -> independent witness before landing or done claim.

## When not to use

- External repository or paper scouting -> [`../scout-loop/SKILL.md`](../scout-loop/SKILL.md).
- Broad issue execution -> [`../super-loop/SKILL.md`](../super-loop/SKILL.md).
- Questions without implementation -> [`../question-loop/SKILL.md`](../question-loop/SKILL.md).
- One known ticket -> ticket-specific dispatch, not another garden pass.
