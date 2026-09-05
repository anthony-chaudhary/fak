---
title: "GOAL.md goal specs — the loop's durable cross-turn memory"
description: "How a GOAL.md goal-spec gives a fak loop durable cross-turn memory: unique goal paths, race condition prevention, subagent delegation, and loop routing."
---

# GOAL.md Goal Specs

A `GOAL.md` goal-spec is the loop's durable cross-turn memory: frontmatter names the loop id, the witness criterion that decides completion, and the iteration budget; `# Objective` states one unit of work; `# Plan` is a markdown checklist the agent updates as durable progress; `# Scratch / last-refusal` is where `fak loop drive` appends the prior `NOT_YET` reason before a fresh-context retry. Start from [`docs/templates/GOAL.md`](templates/GOAL.md), keep the objective singular, choose a witness criterion such as `commit-audit`, `verify PLAN PHASE`, `test-witness BASELINE CANDIDATE`, `witness SOURCE SUBJECT`, or `none` for a budgeted/manual loop that cannot auto-witness completion, and run `fak loop drive` so each turn re-reads the file from disk instead of relying on the previous context window.

Active goal specs belong in `_scratch/goals/GOAL.md` or `_scratch/goals/GOAL-<slug>.md` on disk (under the git-ignored `_scratch/` directory), never tracked at the workspace root. Goal specs are noisy, disposable scratchpad memory designed specifically for surviving context compaction and session restarts across turns. They must not pollute the git working tree or shared trunk.

First-class coordination across agents, workers, and operators does not occur through goal specs. Instead, first-class coordination is handled by dedicated, auditable mechanisms:
- **Lane leases** (`dos arbitrate` / `internal/leaseref`): Mutual exclusion over disjoint file trees.
- **Claims** (`CLAIMS.md`): Portfolio-level contracts and verified performance capabilities.
- **Issue comments**: Durable multi-session tracking and human-in-the-loop updates.
- **Commit trailers** (`(fak <leaf>)` & DCO): Cryptographic, referee-verified proof of delivery.

## Unique goal specs per objective

Goal specifications must use unique, slugged filenames rather than a shared static filename. The standard naming convention is:

- **Master goal spec:** `_scratch/goals/GOAL-<slug>.md` (recommended) or `_scratch/goals/GOAL.md`. Never track or commit goal specs at the workspace root.
- **Slug derivation:** Derive a deterministic, kebab-case slug from the objective (for example, "Fix memory leak in auth" becomes `_scratch/goals/GOAL-fix-memory-leak-in-auth.md`).

### Why unique goal files are required

A single static `GOAL.md` file introduces severe operational hazards in multi-agent or concurrent workflows:

1. **Eliminating race conditions and clobbering:** When multiple concurrent agents, background workers, or parallel goals operate in the same repository checkout, a shared `GOAL.md` leads to interleaved writes, lost updates, and destroyed active plans.
2. **Preventing cross-task memory corruption:** Re-using a static file allows stale failure diagnostics, old checklists, and unrelated refusal history from `# Scratch / last-refusal` to bleed into subsequent or concurrent tasks, corrupting agent context.
3. **Preserving durable historical verification evidence:** Overwriting a static file erases the verified historical record. Unique goal files persist on disk as an auditable post-hoc record of what was planned, what was executed across iterations, and which deterministic witness commands satisfied completion.
4. **Keeping the working tree clean:** Keeping goal specs in git-ignored scratch paths (`_scratch/goals/`) ensures noisy agent scratchpad updates do not dirty the shared trunk, trigger merge conflicts, or leak into commits.

## Routing with FAK_GOAL_SPEC and --goal

Direct `fak loop drive` to the active unique goal specification using either the command-line flag or the environment variable:

```bash
# Via environment variable
export FAK_GOAL_SPEC="_scratch/goals/GOAL-<slug>.md"
fak loop drive -- <agent command>

# Via explicit flag
fak loop drive --goal "_scratch/goals/GOAL-<slug>.md" -- <agent command>
```

When `--goal` is omitted, `fak loop drive` checks `FAK_GOAL_SPEC` (defaulting to `_scratch/goals/GOAL.md` or `GOAL.md` if unset). When the driver launches each child turn, it exposes the resolved path back into the child's environment as `FAK_GOAL_SPEC`, along with `FAK_GOAL_OBJECTIVE`, `FAK_GOAL_WITNESS`, `FAK_GOAL_NEXT`, and `FAK_GOAL_LAST_REFUSAL`, anchoring tool execution and supervisor monitoring to the active specification.

## Subagent delegation and child goal specs

Substantive multi-step tasks should delegate child tasks to isolated subagents while the primary coordinator preserves context hygiene. Child tasks maintain their own scoped goal specifications under:

`_scratch/goals/subagents/GOAL-<parent_slug>--sub-<step>.md`

Child goal specs provide three essential containment guarantees:

- **Isolated disk memory:** The child subagent operates against its own checklist and `# Scratch / last-refusal` scratchpad, preventing subagent trial-and-error noise from polluting the coordinator's durable memory or context window.
- **Bounded write scopes:** Child specs restrict the subagent's write surface to a small, explicit file fence (typically 1–3 files within a single package or lane), minimizing blast radius.
- **Independent test witnesses:** Each child spec defines a concrete, deterministic test witness command proving that subtask's specific deliverables before reporting back.

The coordinator independently verifies that the child's witness command passes before checking off the corresponding milestone in the master goal spec.
