---
name: goal
description: Autonomous goal-directed execution loop. Establishes an immutable objective pin, durable unique disk state (goals/GOAL-<slug>.md & todowrite), deterministic witness exit-gate, and isolated subagent delegation.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, Task
argument-hint: "[objective] [--witness <command>] [--budget <iters>] [--dry-run]"
---

# /goal — Autonomous Goal-Directed Execution Loop

An autonomous execution skill that drives a high-level objective to verified completion. While awaiting native OpenCode `/goal` runtime primitives, this skill provides a battle-tested goal loop immediately across OpenCode, Claude Code, and Codex.

It unifies the best patterns from existing open-source agent systems:
- **Ralph Loop (Geoffrey Huntley, Vercel `ralph-loop-agent`)**: Durable state on disk (`goals/GOAL-<slug>.md`), surviving context compaction and session restarts.
- **OpenAI Codex `/goal` & Aider `/architect`**: Two-phase structured planning, maintaining the objective anchor, and active checklist management.
- **All-Hands OpenHands (OpenDevin) & SWE-agent**: Reproduction-first discipline, minimal surgical edits, and external exit-gates (rejecting the self-assessment trap).
- **Fak Goal Specs & Subagent Hierarchy**: Native compatibility with `docs/goal-spec.md`, `docs/templates/GOAL.md`, the `fak goal` registry (`internal/goalregistry`), and isolated subagent delegation.

---

## Core Invariants of Goal Execution

1. **The Objective Pin**: The objective statement is pinned verbatim and never mutates during the run. Pinning prevents goal drift (Arike et al., 2025).
2. **Unique State on Disk, Not Ephemeral Memory**: Each goal maintains a unique specification file on disk (`goals/GOAL-<slug>.md` or `GOAL-<slug>.md`). Deriving a unique slug from the objective prevents clobbering, race conditions, and memory corruption across concurrent goals or agents while preserving historical evidence across sessions. The environment variable `FAK_GOAL_SPEC` is set to this path so `fak loop drive`, process supervision, and downstream audit tools anchor to the active spec.
3. **External Witness Exit-Gate**: Model proposes, test disposes. The agent never grades its own work. A goal is satisfied only when a deterministic external command (test suite, buildcheck, validator) exits 0.
4. **Subagent Delegation & Coordinator Context Hygiene**: Keep coordinator context clean by delegating substantive implementation and investigation to isolated subagents (`task` in OpenCode or `Task` in Claude Code). Subagents operate within an isolated scope (1–3 files) under child sub-goals (`goals/subagents/GOAL-<parent_slug>--sub-<step>.md`), each carrying its own test witness. The coordinator independently verifies all subagent deliverables before marking checklist items complete in the master goal spec.
5. **Atomic S0/S1 Steps (Subdivide and Scope Abstention)**: Decompose work into single-concern leaves (1–3 files touched per step). Keep exactly one step `in_progress` in `todowrite`. When encountering high-difficulty boundaries (e.g. frozen ABI, kernel SIMD), scope abstention strictly to the bounded aspect: emit a structured `ABSTAIN` record for that boundary while advancing all independent, safe, solvable sub-components (reproduction tests, diagnostics, disjoint packages).
6. **Failure Memory Scratchpad & Persistence**: Genuine guard refusals carrying a closed reason token or unexpected process crashes are recorded in `# Scratch / last-refusal` in `goals/GOAL-<slug>.md`. Routine CLI return codes from read-only commands (such as `grep` returning 1 on no match, or `git diff --quiet` detecting changes) are normal tool execution results and must not be logged as failures in `# Scratch / last-refusal`. A refusal or tool crash is diagnostic feedback, not a session abort. Query `fak recover <TOKEN>` for structured recovery, adapt the execution path or decompose the step, and maintain momentum on the pinned objective without repeating identical failing calls.

---

## Execution Protocol

### Step 1: Intake, Slug Derivation, and Objective Pinning

Parse `$ARGUMENTS` (e.g. `/goal Fix memory leak in auth service --witness "go test ./internal/auth/..."`). If no argument was provided, prompt the operator for the singular objective.

1. **Derive Unique Goal Slug**:
   Derive a deterministic, URL-safe kebab-case slug from the objective statement (e.g. "Fix memory leak in auth service" -> `fix-memory-leak-in-auth-service`).
   - Master goal path: `goals/GOAL-<slug>.md` (or `GOAL-<slug>.md`).
   - Subagent path root: `goals/subagents/GOAL-<slug>--sub-<step>.md`.

2. **Formulate Goal Fields**:
   - **Objective Pin**: Exactly one clear, measurable end-state.
   - **Non-Goals / Scope Fences**: Explicitly state what is NOT part of this run to prevent over-scaffolding and companion abstraction sprawl.
   - **Witness Criterion**: The exact deterministic command that proves success (e.g. `go test -v ./internal/gateway/... -run TestGatewayReady` or `fak validate --mine <paths>`).

### Step 2: Initialize Unique Durable State (`goals/GOAL-<slug>.md` & `todowrite`)

#### Why Unique Goal Files on Disk
A single static `GOAL.md` fails in multi-agent and concurrent environments:
- **Prevents Clobbering**: Multiple concurrent agents or background workers in the same checkout will overwrite a shared `GOAL.md`, destroying active plans.
- **Eliminates Race Conditions**: Concurrent writes to a shared file create lost updates and corrupted execution logs.
- **Isolates Memory Across Runs**: Fresh-context retries or subsequent sessions cannot accidentally inherit stale failure scratchpads from previous unrelated objectives.
- **Preserves Auditable History**: Completed goal specs persist on disk, providing an evidence-backed record of plans, iterations, and witness results.
- **Enables Tool & Runtime Integration**: Setting `export FAK_GOAL_SPEC="goals/GOAL-<slug>.md"` allows `fak loop drive`, process supervision, git review, and `internal/goalregistry` to target the active goal file.

#### File Creation and Environment Binding
Ensure the target directory exists and create `goals/GOAL-<slug>.md`:

```bash
mkdir -p goals goals/subagents
export FAK_GOAL_SPEC="goals/GOAL-<slug>.md"
```

Write the initial master goal spec:

```markdown
---
loop: goal
goal_slug: <slug>
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

### Step 4: Subagent Delegation Protocol

Substantive plan items should be delegated to isolated subagents to preserve coordinator context and enforce blast-radius fences.

#### Coordinator Role vs Subagent Role
- **Coordinator**: Maintains the master goal spec (`goals/GOAL-<slug>.md`), manages the high-level plan, dispatches child workers, independently witnesses outputs, and commits clean results. The coordinator does not accumulate long tool traces, test logs, or build output in its context window.
- **Subagents**: Execute bounded tasks in isolated sessions using `task` (OpenCode) or `Task` (Claude Code). Subagents operate strictly within an assigned file fence (1–3 files) and verify their changes against a scoped witness before returning.

#### Child Sub-Goal Specification (`goals/subagents/GOAL-<parent_slug>--sub-<step>.md`)
For each substantive step in the plan, write a child sub-goal spec:

```markdown
---
parent_goal: goals/GOAL-<parent_slug>.md
sub_step: <step_number_or_name>
witness: <step-scoped test command>
target_files:
  - <path/to/file1>
  - <path/to/file2>
---
# Sub-Goal Objective
<Specific, bounded deliverable for this step>

# Scope Fence
- Work exclusively in: <1-3 assigned files>
- Prohibited: Do not touch root configs, go.mod, go.sum, dos.toml, or sibling packages.

# Witness Command
<Deterministic package-level test command>
```

#### Dispatching the Subagent
Invoke the subagent tool with bounded instructions:
- **OpenCode**: Call `task` with `subagent_type="general"` or targeted agent, passing the path to the child sub-goal file, the permitted file list, and the required test witness.
- **Claude Code**: Call `Task` with the child sub-goal path, bounded prompt, and witness command.

#### Subagent Execution Invariants
1. **Isolated Scope**: The subagent modifies only the 1–3 assigned files using `Edit` or `Write`.
2. **Local Verification**: The subagent runs only package-scoped tests (e.g. `go test -v ./internal/<pkg>/...`), never repository-wide suites.
3. **Compact Report**: The subagent returns a concise summary: touched paths, test witness output, and exit status.

#### Coordinator Independent Verification (Rejecting Self-Reports)
Worker self-reports are not facts ("mocks hide integration bugs"). The coordinator must independently verify subagent deliverables before marking work done:
1. **Inspect Diff Boundaries**: Run `git status` or `git diff --stat` to confirm only the expected 1–3 files were modified.
2. **Execute Independent Witness**: The coordinator directly runs the step's witness command (e.g. `go test -v ./internal/<pkg>/...` or `fak validate --mine <paths>`).
3. **Handle Refusals & Failures**: If the subagent failed or the independent witness fails:
   - Log genuine guard refusals or unexpected crashes to `# Scratch / last-refusal` in `goals/GOAL-<slug>.md`.
   - Query `fak recover <TOKEN>` for structured recovery.
   - Decompose the step further or adjust the child spec rather than repeating identical failing instructions.
4. **Advance Checklist**: Only upon a green independent witness, mark `- [x]` in `goals/GOAL-<slug>.md` and `completed` in `todowrite`.

### Step 5: Witness Gate Evaluation

Run the declared master witness command:
```bash
# Example witness commands:
go test -v ./internal/<pkg>/...
fak validate --mine <path1> <path2>
python tools/<validator>.py --check
```

- **Exit code != 0**: Append the failure to `# Scratch / last-refusal` in `goals/GOAL-<slug>.md`, formulate a targeted fix, and re-run. If the failure exposes an insurmountable high-difficulty boundary, land the verified partial deliverables (e.g. reproduction witness), record a scoped `ABSTAIN` for the specific boundary, and state the exact checkable next step rather than dropping the run.
- **Exit code == 0**: Witness criterion is satisfied. Proceed to completion.

### Step 6: Completion and Evidence Sealing

1. Record final witness evidence and timestamp in `goals/GOAL-<slug>.md`.
2. If registered in `fak goal`:
   ```bash
   fak goal transition --id <goal_id> --lifecycle achieved --evidence-class independent_witness --evidence-ref "<witness-command>"
   ```
3. If committing: follow the project's explicit-path commit standard:
   ```bash
   fak commit --preview -m "<subject> (fak <leaf>)" --path <p1> --path <p2>
   fak commit --path <p1> --path <p2> -m "<subject> (fak <leaf>)"
   ```
4. Emit a concise, verdict-first completion report (<3 lines):
   - **Line 1**: Goal status and witness confirmation (`goals/GOAL-<slug>.md`).
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
