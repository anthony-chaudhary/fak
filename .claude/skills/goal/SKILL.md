---
name: goal
description: Autonomous goal-directed execution loop. Establishes an immutable objective pin, durable scratch state (_scratch/goals/GOAL.md & todowrite), a deterministic witness exit-gate, and executes atomic steps until verified.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Write, Grep, Glob, Bash
argument-hint: "[objective] [--witness <command>] [--budget <iters>] [--dry-run]"
---

# /goal — Autonomous Goal-Directed Execution Loop

An autonomous execution skill that drives a high-level objective to verified completion. While awaiting native OpenCode `/goal` runtime primitives, this skill provides a battle-tested goal loop immediately across OpenCode, Claude Code, and Codex.

It unifies the best patterns from existing open-source agent systems:
- **Ralph Loop (Geoffrey Huntley, Vercel `ralph-loop-agent`)**: Durable state on disk (`GOAL.md`), surviving context compaction and session restarts.
- **OpenAI Codex `/goal` & Aider `/architect`**: Two-phase structured planning, maintaining the objective anchor, and active checklist management.
- **All-Hands OpenHands (OpenDevin) & SWE-agent**: Reproduction-first discipline, minimal surgical edits, and external exit-gates (rejecting the self-assessment trap).
- **Fak Goal Specs & Registry**: Native compatibility with `docs/goal-spec.md`, `docs/templates/GOAL.md`, and the `fak goal` registry (`internal/goalregistry`).

---

## The Five Invariants of Goal Execution

1. **The Objective Pin**: The objective statement is pinned verbatim and never mutates during the run. Pinning prevents goal drift (Arike et al., 2025).
2. **State on Disk, Not Ephemeral Memory**: Progress, plans, and scratchpad live in `_scratch/goals/GOAL.md` (or `_scratch/goals/GOAL-<slug>.md`) on disk. Goal specs are noisy, disposable scratchpad memory for surviving context compaction; they are not coordination artifacts. First-class coordination occurs via lane leases (`dos arbitrate` / `internal/leaseref`), claims (`CLAIMS.md`), issue comments, and commit trailers.
3. **External Witness Exit-Gate**: Model proposes, test disposes. The agent never grades its own work. A goal is satisfied only when a deterministic external command (test suite, buildcheck, validator) exits 0.
4. **Atomic S0/S1 Steps (Divide & Conquer, Subdivide, and Scope Abstention)**: Divide and conquer substantive or multi-concern objectives into atomic leaves (1–3 files touched per step). When tasks have independent components, delegate to isolated subagents or workers (`task`: worker, researcher, explore, cross-validator) to prevent coordinator context pollution. Keep exactly one step `in_progress` in `todowrite`. When encountering high-difficulty boundaries (e.g. frozen ABI, kernel SIMD), scope abstention strictly to the bounded aspect: emit a structured `ABSTAIN` record for that boundary while advancing all independent, safe, solvable sub-components (reproduction tests, diagnostics, disjoint packages).
5. **Failure Memory Scratchpad & Persistence**: Genuine guard refusals carrying a closed reason token or unexpected process crashes are recorded in `# Scratch / last-refusal` in `_scratch/goals/GOAL.md` (or `_scratch/goals/GOAL-<slug>.md`). Routine CLI return codes from read-only commands (such as `grep` returning 1 on no match, or `git diff --quiet` detecting changes) are normal tool execution results and must not be logged as failures in `# Scratch / last-refusal`. A refusal or tool crash is diagnostic feedback, not a session abort. Query `fak recover <TOKEN>` for structured recovery, adapt the execution path or decompose the step, and maintain momentum on the pinned objective without repeating identical failing calls.

---

## Execution Protocol

### Step 1: Intake and Objective Pinning

Parse `$ARGUMENTS` (e.g. `/goal Fix memory leak in auth service --witness "go test ./internal/auth/..."`). If no argument was provided, prompt the operator for the singular objective.

Formulate three fields:
- **Objective Pin**: Exactly one clear, measurable end-state.
- **Non-Goals / Scope Fences**: Explicitly state what is NOT part of this run to prevent over-scaffolding and companion abstraction sprawl.
- **Witness Criterion**: The exact deterministic command that proves success (e.g. `go test -v ./internal/gateway/... -run TestGatewayReady` or `fak validate --mine <paths>`).

### Step 2: Initialize Durable Scratch State (_scratch/goals/GOAL.md & todowrite)

Ensure directory exists (`mkdir -p _scratch/goals`) and create or update `_scratch/goals/GOAL.md` (or `_scratch/goals/GOAL-<slug>.md`). Never create or commit a tracked `GOAL.md` at the workspace root. Goal specs are local, disposable execution scratchpad memory. First-class coordination occurs via lane leases (`dos arbitrate`), claims (`CLAIMS.md`), issue comments, and commit trailers (`(fak <leaf>)`).

```markdown
<!-- Active goal spec saved under _scratch/goals/GOAL.md or _scratch/goals/GOAL-<slug>.md (never committed) -->
---
loop: goal
witness: <witness command or criterion>
budget: { max_iters: 20 }
---
# Objective
<Pinned objective statement>

# Non-Goals
- <Explicit exclusion 1>
- <Explicit exclusion 2>

# Plan
- [ ] 1. Reproduce baseline behavior or establish failing test
- [ ] 2. Implement minimal surgical fix or feature
- [ ] 3. Run package-level verification and regression checks
- [ ] 4. Execute final witness command

# Scratch / last-refusal
```

If the `fak` binary is available, optionally register the goal in the canonical registry:
```bash
fak goal create --title "<title>" --summary "<summary>"
```

If running in OpenCode or an environment with `todowrite`, populate the task list with the plan items, marking the first item `in_progress`.

### Step 3: Baseline Reproduction First (SWE-agent Invariant)

Before editing code:
1. Locate relevant files using `Glob`, `Grep`, and `Read`.
2. Execute the existing tests or create a reproduction test case.
3. Capture the baseline failure or diagnostic. A defect is only proven fixed when the test failed before and passes after.

### Step 4: Atomic Sequential Execution

Iterate through plan items sequentially:
1. **Divide and conquer by delegation**: For substantive, complex, or multi-component goals, launch specialized subagents concurrently for independent parts (e.g. `task` with `worker` for implementation, `researcher` for prior art, `cross-validator` for verification). Pull only compact receipts and decisions into the coordinator.
2. Keep only one task `in_progress` in `todowrite`.
3. Confine edits to 1–3 closely related files using `Edit` or `Write`.
4. If a command or tool fails:
   - Routine CLI return codes from read-only commands (such as `grep` returning 1 on no match, or `git diff --quiet` detecting changes) are normal tool execution results and must not be logged as failures in `# Scratch / last-refusal`.
   - Log only genuine guard refusals carrying a closed reason token or unexpected process crashes to `# Scratch / last-refusal` in `_scratch/goals/GOAL.md`.
   - Query structured recovery: if a guard refused the call, run `fak recover <TOKEN>` or `dos man wedge <TOKEN> --explain` to obtain the sanctioned remedy.
   - Handle transient locks: if blocked by `MERGE_IN_PROGRESS`, `COLLISION_RISK`, or `LOCK_BUSY`, unstage paths (`git restore --staged`), wait for quiescence, or switch to an independent disjoint subtask.
   - Decompose rather than abandon: if a subtask is blocked or too complex, divide it into smaller verifiable leaves (e.g. capture baseline reproduction tests first) while keeping the pinned objective intact.
   - Adapt the approach: halt repetition of the identical failing call; select a sanctioned alternative tool or modified arguments.
5. Verify the step using package tests (`go test ./internal/<pkg>/...` or platform test script).
6. Mark the task `- [x]` in `_scratch/goals/GOAL.md` and `completed` in `todowrite`.

### Step 5: Witness Gate Evaluation

Run the declared witness command:
```bash
# Example witness commands:
go test -v ./internal/<pkg>/...
fak validate --mine <path1> <path2>
python tools/<validator>.py --check
```

- **Exit code != 0**: Append the failure to `# Scratch / last-refusal` in `_scratch/goals/GOAL.md`, formulate a targeted fix, and re-run. If the failure exposes an insurmountable high-difficulty boundary, land the verified partial deliverables (e.g. reproduction witness), record a scoped `ABSTAIN` for the specific boundary, and state the exact checkable next step rather than dropping the run.
- **Exit code == 0**: Witness criterion is satisfied. Proceed to completion.

### Step 6: Completion and Evidence Sealing

1. Record final witness evidence and timestamp in `_scratch/goals/GOAL.md`. Files under `_scratch/` are ignored by git (`/_*`) and must never be staged or committed.
2. If registered in `fak goal`:
   ```bash
   fak goal transition --id <goal_id> --lifecycle achieved --evidence-class independent_witness --evidence-ref "<witness-command>"
   ```
3. **Safe Git Sync, Commit, and Push by Default**:
   Do more work by default. When the change is verified green, complete the full delivery lifecycle without waiting for an operator prompt:
   ```bash
   # 1. Pre-flight safe sync with origin/main:
   fak sync reconcile --apply

   # 2. Preview commit subject and lane:
   fak commit --preview -m "<subject> (fak <leaf>)" --path <p1> --path <p2>

   # 3. Stage-and-commit by explicit path (never stage _scratch/):
   fak commit --path <p1> --path <p2> -m "<subject> (fak <leaf>)"

   # 4. Safe push unprompted:
   fak sync push
   # (or combine commit and push: `fak commit --path <p1> ... -m "..." --push`)
   ```
4. Emit a concise, verdict-first completion report (<3 lines):
   - **Line 1**: Goal status and witness confirmation.
   - **Line 2**: Deliverables and touched paths.
   - **Line 3**: Checkable verification command.

---

## Tool Mapping Across Agent Harnesses

| Operation | OpenCode | Claude Code | Codex / fakc |
|---|---|---|---|
| Step tracking | `todowrite` | `TodoWrite` / markdown | `update_plan` |
| File inspection | `read`, `glob`, `grep` | `Read`, `Glob`, `Grep` | `read_file`, `glob` |
| Code editing | `edit`, `write` | `Edit`, `Write` | `edit_file` |
| Execution | `bash` | `Bash` | `shell_command` |
| Subagents | `task` | `Task` | detached worker |

---

## Verification and Witness

Verify this skill definition using the project gates:

```bash
# 1. Structural admission and anti-slop verification:
python tools/skill_slop_scorecard.py .claude/skills/goal/SKILL.md --corpus .claude/skills

# 2. Frontmatter portability check:
python tools/skill_frontmatter_lint.py --check

# 3. Synchronize cross-harness adapter for OpenCode (.agents/skills/goal/SKILL.md):
go run ./cmd/fak-project-assets sync --json

# 4. Verify zero unexplained parity gaps across harnesses:
go run ./cmd/fak-project-assets parity --json
```

Exit code 0 across all checks verifies full portability, compliance with the Agent Skills standard, and zero-gap availability in OpenCode.
