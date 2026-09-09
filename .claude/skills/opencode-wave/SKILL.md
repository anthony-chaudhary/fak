---
name: opencode-wave
description: Spawn and coordinate deterministic, collision-safe OpenCode leaf worker waves with high reasoning effort (--variant high), automated approvals (--auto), mandatory detached worktree isolation (--worktree), pre-flight lane lease arbitration, and automated closed-loop harvest.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, Task
argument-hint: "[--top 10] [--wave-size 4] [--variant high] [--worktree] [--workspace <path>] [--from-issues <file>] [--harvest]"
---

# /opencode-wave — High-Effort Deterministic OpenCode Multi-Agent Wave Dispatch

The campaign coordinator for running multi-issue resolution waves in OpenCode. It takes a cohort of tracked GitHub issues or ticket tranches, verifies pairwise tree-disjoint package boundaries, isolates concurrent workers in detached worktrees, spawns user-visible detached OpenCode processes (`opencode run --variant high --auto`), enforces direct leaf execution within package boundaries (prohibiting nested `task` calls to prevent depth limit recursion failures, #12028), and supervises execution through automated harvest and safe git landing.

---

## Six Invariants of Deterministic OpenCode Wave Dispatch

1. **GitHub Issues & Tranche Grounding First**: Every substantive unit of work must be tracked in a GitHub issue and grounded on ticket specifications before worker execution begins. When targeting companion repositories (e.g. `anthony-chaudhary/fak-private`), explicitly bind candidate issues to the target repo and local ticket tranche (`docs/tickets/.../INDEX.md`).
2. **Mandatory Worktree Isolation**: Dispatched workers touching identical Go packages must execute in detached worker worktrees (`--worktree`) or be partitioned into sequential waves. Concurrent edits to shared packages in the same working tree cause compilation cross-contamination and test failures.
3. **Strict Ban on Loose Scripts (Modular Native Go Dispatch Only)**: Creating new PowerShell (`.ps1`), Bash (`.sh`), or batch (`.bat`) scripts is prohibited. Wave planning, process spawning, and harvest reconciliation must run through first-class Go tooling (`fak issue-orchestrator`).
4. **High-Reasoning Effort & Auto-Approve Flags**: Dispatched child OpenCode sessions run with `--variant high --auto` (or `--dangerously-skip-permissions`). This activates Gemini 3.8 Flash high reasoning mode for complex debugging, concurrency verification, and kernel implementation.
5. **Leaf Worker Direct Execution (No Nested Subagents)**: Dispatched sessions act as depth-1 leaf workers focused on assigned lane and boundary paths. Workers execute deliverables directly within package boundaries, author reproduction tests, and run package verification without calling the `task` tool or attempting nested subagent delegation. The coordinator manages wave fan-out; leaf workers execute directly to prevent recursion depth limit failures (`Subagent depth limit reached (1)`, #12028, #12029).
6. **Automated Closed-Loop Supervision & Harvest**: Waves do not stop at process launch. The coordinator supervises active workers, enforces the 4-state harvest protocol (`StateVerifiedCleared`, `StateResidualReview`, `StateQuietIncomplete`, `StateSpinningStalled`), lands verified worktrees to trunk, and verifies GitHub comment receipts.

---

## Execution Protocol

### Phase 1: Intake & Issue Tranche Ingestion

Query candidate issues from the target repository and pipe directly into structured JSON:

```bash
# Ingest candidate issues from target repository (public fak or private companion)
gh issue list -R anthony-chaudhary/fak-private --limit 15 --json number,title,labels,body > _scratch/issues.json
```

When executing a local campaign tranche, reconcile against the ticket index:
```bash
# Verify ticket definitions and acceptance criteria
cat docs/tickets/<campaign>/INDEX.md
```

### Phase 2: Tree-Disjoint Wave Partitioning & Lease Arbitration

Partition the candidate issue set into ordered, collision-free waves where no two concurrent issues in the same wave touch overlapping packages:

```bash
# Plan safe waves with pairwise tree-disjoint boundaries and lease verification
go run ./cmd/fak issue-orchestrator \
  --workspace . \
  --from-issues _scratch/issues.json \
  --wave-size 4 \
  --plan-waves \
  --worktree \
  --json
```

Verify that each planned wave respects lane leases (`dos arbitrate`) and isolates high-blast-radius core packages into single-worker waves.

### Phase 3: Native Go Wave Dispatch

Launch detached OpenCode worker processes for the target wave using first-class native Go tooling:

```bash
# Dispatch Wave 1 with high reasoning effort and detached worktree isolation
go run ./cmd/fak issue-orchestrator \
  --workspace . \
  --from-issues _scratch/issues.json \
  --spawn-opencode \
  --spawn-wave 1 \
  --wave-size 4 \
  --variant high \
  --worktree \
  --json
```

For private platform components in `fak-private`, specify the companion workspace root:
```bash
go run ./cmd/fak issue-orchestrator \
  --workspace ../fak-private \
  --from-issues _scratch/issues.json \
  --spawn-opencode \
  --spawn-wave 1 \
  --variant high \
  --worktree \
  --json
```

### Phase 4: Live Supervision & Watchdog Monitoring

Audit running worker processes, session database registrations, and log streams:

```bash
# Check registered OpenCode database sessions
opencode session list

# Check running processes and resource utilization
powershell -Command "Get-Process -Name 'opencode' | Select-Object Id, ProcessName, CPU, WorkingSet64, StartTime | Format-Table -AutoSize"

# Inspect real-time execution log of active worker
cat _scratch/logs/opencode-issue-<issue_number>.err.log
```

### Phase 5: Automated Closed-Loop Harvest & Safe Git Landing

Reconcile wave outcomes against the 4-state harvest protocol:

```bash
# Execute automated wave harvest and receipt generation
go run ./cmd/fak issue-orchestrator \
  --workspace . \
  --harvest \
  --json
```

Reconciliation states:
- `StateVerifiedCleared`: Git commit witnessed on worktree, tests pass green, and 5-gate boundary clean. Auto-land worktree to trunk (`fak-flow land` or `fak sync push`) and close/comment on GitHub issue.
- `StateResidualReview`: Diffs exist but uncorroborated or tests failing. Flag for manual inspection.
- `StateQuietIncomplete`: No commits or modifications found. Soft re-queue for next wave.
- `StateSpinningStalled`: Process exceeded turn budget or deadlocked. Terminate PID and release lease.

Verify on-device package tests before closing the wave:
```bash
go test -v ./internal/<changed-pkg>/...
go vet ./internal/<changed-pkg>/...
```

---

## Verification and Witness

Verify this skill definition using the project gates:

```bash
# 1. Structural admission and anti-slop verification:
python tools/skill_slop_scorecard.py .claude/skills/opencode-wave/SKILL.md --corpus .claude/skills

# 2. Frontmatter portability check:
python tools/skill_frontmatter_lint.py --check

# 3. Synchronize cross-harness adapter for OpenCode (.agents/skills/opencode-wave/SKILL.md):
go run ./cmd/fak-project-assets sync --json

# 4. Verify zero unexplained parity gaps across harnesses:
go run ./cmd/fak-project-assets parity --json
```
