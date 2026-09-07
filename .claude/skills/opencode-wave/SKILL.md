---
name: opencode-wave
description: Spawn and coordinate a wave of user-visible, detached OpenCode worker sessions running with high reasoning effort (--variant high), automated approvals (--auto), and instructions to delegate to parallel subagents (task) to resolve tracked GitHub issues end-to-end with deterministic test witnesses and GitHub comment receipts.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, Task
argument-hint: "[--top 10] [--wave-size 4] [--variant high] [--dry-run] [--worktree]"
---

# /opencode-wave — High-Effort OpenCode Multi-Agent Wave Dispatch

The goal-directed campaign coordinator for running multi-issue resolution waves in OpenCode. It takes a cohort of tracked GitHub issues, verifies pairwise tree-disjoint package boundaries, spawns user-visible detached OpenCode processes (`opencode run --variant high --auto`), instructs child sessions to delegate implementation to parallel subagents (`task`: worker, researcher, deep-reason, cross-validator), and monitors execution until verified receipts land on GitHub.

---

## Five Invariants of OpenCode Wave Dispatch

1. **GitHub Issues Tracked First**: Every substantive unit of work must be tracked in a GitHub issue before worker execution begins. If candidate tasks lack issues, create them first via `gh issue create` following the standard contract: Parent epic, Centrality (Core, Enabling, Stewardship), and the For / Problem / Today / Better because / Witness schema.
2. **High-Reasoning Effort by Default**: Child OpenCode workers are launched with `--variant high`. This activates Gemini 3.8 Flash high reasoning mode for complex debugging, concurrency verification, and kernel implementation.
3. **Child Subagent Delegation by Default**: Every child session receives prompt instructions mandating subagent delegation via the `task` tool (`subagent_type="worker"`, `"deep-reason"`, `"researcher"`, `"cross-validator"`). Child coordinators keep their own context clean by delegating compilation, test execution, and research to parallel subagents.
4. **User-Visible & Machine-Auditable**: Child workers run as identifiable processes on the host. Every session registers in `opencode session list`, receives a distinct session title (`Issue #<N>: <title>`), writes logs to `_scratch/logs/`, and posts final verification receipts directly to its GitHub issue (`gh issue comment <N>`).
5. **Pairwise Tree-Disjoint Concurrency**: Workers dispatched in the same wave must touch mutually disjoint packages and directories. Concurrent edits across shared Go packages break compilation. Partition candidate issues into verified disjoint waves using `fak issue-orchestrator --plan-waves`.

---

## Execution Protocol

### Phase 1: Intake & GitHub Issue Verification

Enumerate the target issues or candidate topics:

```bash
# List open issues for the target domain
gh issue list --limit 15 --search "<domain-or-epic>"
```

If candidate tasks do not yet exist on GitHub, create them before launching workers:

```bash
gh issue create \
  --title "<type>(<pkg>): <short-description>" \
  --label "class:dev,priority/P1" \
  --body "## Problem Centrality: Core
Parent: #<epic-number>

### For / Problem / Today / Better because / Witness
- **For**: <target-audience-and-workload>
- **Problem**: <failure-mode-or-performance-bottleneck>
- **Today**: <baseline-behavior-or-current-workaround>
- **Better because**: <measurable-outcome-and-mechanism>
- **Witness**: <exact-deterministic-command-and-exit-code>"
```

### Phase 2: Tree-Disjoint Wave Partitioning

Partition the active issue set into concurrent waves where no two issues in the same wave touch overlapping packages:

```bash
# Plan safe waves bounded to top candidate issues
fak issue-orchestrator --top 10 --wave-size 4 --plan-waves --json
```

Verify that each planned wave respects lane leases (`dos arbitrate`) and isolates high-blast-radius core packages (`internal/abi`, `internal/gateway`, `internal/policy`) into single-worker serial waves.

### Phase 3: Spawn User-Visible OpenCode Worker Sessions

Launch detached OpenCode processes for the selected wave:

```bash
# Native CLI path
fak issue-orchestrator --top 10 --wave-size 4 --spawn-opencode --variant high --json
```

Alternatively, launch detached child sessions using PowerShell with prompt files to prevent argument-splitting traps:

```powershell
$exe = "C:\Users\USER\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe"
$argStr = "run --variant high --auto --title `"Issue #$($issue): $($title)`" `"Resolve GitHub issue #$($issue): please read _scratch/prompts/issue-$($issue).txt and execute the required deliverables using parallel subagents by default, run tests, and post receipts with gh issue comment.`""

Start-Process -FilePath $exe -ArgumentList $argStr -WorkingDirectory (Get-Location) -RedirectStandardOutput "_scratch\logs\opencode-issue-$issue.out.log" -RedirectStandardError "_scratch\logs\opencode-issue-$issue.err.log" -PassThru
```

### Phase 4: Live Monitoring & Audit

Audit running worker processes and active OpenCode database sessions:

```bash
# Check registered sessions
opencode session list

# Check running processes and resource utilization
powershell -Command "Get-Process -Name 'opencode' | Select-Object Id, ProcessName, CPU, WorkingSet64, StartTime | Format-Table -AutoSize"

# Inspect real-time transcript of any active session
opencode export <session_id>
```

### Phase 5: Witness Harvesting & Receipt Verification

Inspect each worker's delivered artifacts and posted GitHub comment receipts:

```bash
# Check comments on the issue
gh issue view <issue> --comments

# Run package verification on changed paths
go test -v ./internal/<changed-pkg>/...
go vet ./internal/<changed-pkg>/...
```

Verify that:
1. The code changes remain bounded to the declared lane and boundary paths.
2. The package test suite exits 0.
3. The worker posted an implementation receipt containing status, touched paths, and test command output.

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
