---
name: super-loop
description: The durable front door to a "super loop" — launching HEADLESS work sessions in BULK, one detached `/goal` worker per DISTINCT rate-limit account, each resolving a top-ranked ready leaf and closing it by witnessed ancestry. Wraps the proven launchers (`tools/issue_dispatch.py --wave`, `tools/launch_wave_detached.ps1`) with the discipline the raw scripts assume: PLAN by default, price the fan-out for tree-collisions AND account-distinctness before a single worker spawns, re-check the no-DoS preflight cap per spawn, and never confuse a launch with a ship. Use when the operator says "launch a wave", "run a super loop", "spin up N workers on the top issues", "fan out headless sessions", "drain the backlog in parallel", or "start the overnight fleet".
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # the commit-by-explicit-path, honesty-boundary, and collision discipline are load-bearing and not portable per-skill
---

# /super-loop — launch headless work sessions in bulk, safely

> **The bulk-headless launcher.** One call fans out N detached `/goal` workers —
> each an independent Claude Code session that survives this shell ending — pointed
> at the top-ranked ready leaves. The dangerous parts (process detachment, account
> pinning, the no-DoS cap, tree-collision pricing) are already solved in the
> launchers this skill drives; the skill's job is to run them in the right order,
> **PLAN first**, and hold the fan-out to the honesty boundary: a launch is not a
> ship — only a witnessed commit on the trunk resolves an issue.

## Two launch paths — pick the one that matches your risk

| Path | What it gives | When |
|---|---|---|
| `python tools/issue_dispatch.py --wave` | Fans out across pairwise **TREE-DISJOINT** lanes in ONE checkout, PRICED (`dos arbitrate`) so a colliding set is caught before launch, preflight re-checked per spawn. Collision-safe by construction. | **Default.** In-repo issue work on the shared trunk. |
| `tools/launch_wave_detached.ps1 -Count N` | Fans out across N **DISTINCT rate-limit accounts** (`fak fleet-accounts wave`), one detached `/goal` worker per pool → N× rate-limit headroom. | High throughput, when you need more usage buckets than one account gives. |

The catch that makes this skill load-bearing: **`launch_wave_detached.ps1` gives
distinct accounts but its workers share ONE working tree** (a single `-Workspace`).
Distinct rate-limit pools ≠ distinct file trees. So the multi-account wave is only
collision-safe if EITHER each worker takes a lane lease (`dos arbitrate`) before it
edits — which the [fuel prompt](../../goal-prompts/resolve-top-issue-witnessed.md)
mandates as step 1 — OR each account runs in its own checkout. Never launch a
multi-account wave whose workers will free-edit the same `cmd/**` / `internal/**`
tree: that poisons `go build` for every sibling on the trunk (the witnessed #1338
failure — two runs, ~52 turns, 0 commits).

## The honesty boundary (do not cross)

- **PLAN by default.** Both launchers spawn NOTHING without an explicit opt-in
  (`--live` / `-Launch`). The dry-run plan is the witnessable artifact; show it and
  get operator approval before spawning real detached workers.
- **A launch is not a ship.** This skill starts processes. It NEVER reports an issue
  resolved. An issue is resolved only when a witnessed commit carrying `Fixes #N`
  lands on the trunk (`dos commit-audit`, `dos verify`) — ground truth the launcher
  cannot fake.
- **The cap is the no-DoS guarantee.** Every spawn passes `dispatch_preflight.py`
  (`SPAWN_OK`) — `issue_dispatch.py` re-checks it per spawn, and
  `launch_goal_detached.ps1` (the spawn point the single AND multi-account wave
  paths share) refuses on any non-`SPAWN_OK` verdict. A `REFUSE_*` is the safety
  floor doing its job — surface it, do not route around it (`-SkipPreflight` is an
  operator-only override; never pass it in a wave). One honest caveat: a just-spawned
  `/goal` worker is stdin-fed and carries no scannable process marker, so the
  per-spawn re-check sees a sibling only once it holds a lane lease — size the wave
  from the plan; do not re-run it to "top up" while workers are still starting.
- **Own the seat.** The launcher strips `ANTHROPIC_*` and the session-identity vars
  before spawning, so a wave launched from inside a `fak guard`ed session cannot
  bleed onto the parent's loopback gateway/seat (the whole-wave same-instant crash;
  child-stderr tell: "claude.ai connectors are disabled because ANTHROPIC_API_KEY …
  is set").

## Step 0 — Orient: is it safe to spawn, and what is the fuel?

```bash
python tools/dispatch_preflight.py --json     # SPAWN_OK  or  REFUSE_{INSPECT,HOST,NO_SEAT,AT_CAP,NO_ACCOUNT}
python tools/dispatch_status.py --fast        # quick pure-local card: workers live vs cap, account availability
```

(`--md` is not a display flag — `--md <path>` WRITES the committed markdown status
doc; the human-readable card is the default stdout output, `--fast` skips the two
gh-backed folds for a quick look.)

If preflight is not `SPAWN_OK`, STOP and report the verdict — the recover step is
the one named in the AGENTS.md guard table (fix the host / re-login / wait for a
seat), not a wider fan-out.

The **fuel** is the `/goal` pointer each headless worker reads
(`.claude/goal-prompts/resolve-top-issue-witnessed.md`) — a self-contained spec:
*take a lane, resolve the top ready leaf, ship it witnessed, close by ancestry,
stop.* Keep it < 4000 chars (the `/goal` cap the launcher enforces).

## Step 1 — Rank the queue (the "top N" fuel, live — never a frozen list)

The backlog re-ranks daily, so the queue is a **live query**, not a committed doc.
The dispatchable surface is the `ready-leaves` / `p0-p1` views; the deterministic
"do next" order is `tools/issue_triage.py`'s integer score (P0 1000 · P1 400 ·
P2 150 · none 60; +300 orphan P0/P1, +40 bug, + idle-age):

```bash
python tools/issue_lane_router.py --view p0-p1 --json      # prioritized leaves → lanes (fall through to ready-leaves)
python tools/issue_triage.py --markdown --out docs/_audits/issue-triage-$(date +%F).md   # ranked, read-only
```

Read the top N (default N = `--max-workers`) rows. These are the leaves the wave
will pick from; each worker selects the top-ranked leaf on the lane it leases.

## Step 2 — Price the fan-out (dry-run) — collisions AND account-distinctness

Never launch blind. Run the launcher in its default PLAN mode and read the plan:

```bash
# Default path — tree-disjoint in-repo wave (priced by dos arbitrate):
python tools/issue_dispatch.py --wave --max-workers <N> --work-kind engineering    # DRY-RUN
```

```powershell
# High-throughput path — N distinct accounts (plan only, no -Launch):
.\tools\launch_wave_detached.ps1 -Count <N> -WorkKind engineering -Workspace C:\work\fak `
  -PointerFile .claude/goal-prompts/resolve-top-issue-witnessed.md
```

Read the plan out loud for the operator:

- **Tree-disjoint (`issue_dispatch --wave`):** confirm each lane's tree is pairwise
  disjoint and none is a self-source lane the arbiter refused. This is
  `dos-plan-price` folded into the launcher — a colliding set is serialized into a
  later wave, not launched.
- **Multi-account (`launch_wave_detached`):** confirm `granted` vs `requested`
  (honest under-fill when fewer distinct accounts are free) and that
  `distinct_pools > 1` (a naive burst collapses to ONE pool and serializes). If the
  workers will share a tree, confirm the fuel's lane-lease step is intact, or launch
  fewer workers.

If the plan shows collisions, an unavailable account, or `distinct_pools == 1`, fix
the partition or wait — do not `--force` / launch anyway.

## Step 3 — Launch the wave (opt-in, operator-approved)

Only after the plan is clean AND the operator approves the real spawn:

```bash
python tools/issue_dispatch.py --wave --max-workers <N> --work-kind engineering --live
```
```powershell
.\tools\launch_wave_detached.ps1 -Count <N> -WorkKind engineering -Launch -Workspace C:\work\fak `
  -PointerFile .claude/goal-prompts/resolve-top-issue-witnessed.md
```

Each spawn re-checks the preflight cap, so the live population still never exceeds
the seat cap even mid-wave. Record what launched: the per-lane account/pool/PID and
the log paths the launcher prints (`.goal-runs/*.pid`, `.dispatch-runs/inflight-*`).

## Step 4 — Watch, witness, and stop (a launch is not a ship)

The workers are detached — they outlive this session. Do NOT poll them in a tight
loop; check back on a cadence with the existing status tools (this skill launches;
it does not re-implement monitoring):

```bash
python tools/dispatch_status.py                 # full fold: live workers, throughput, closure-honesty
```

When a worker claims a leaf done, the truth is git, not the log tail:

```bash
dos commit-audit --json          # the worker's commit CLAIM vs what its DIFF did
dos verify --workspace . <plan> <phase> --json    # a plan/phase actually shipped?
gh issue view <N> --json state,stateReason        # closed by an ancestry `Fixes #N`, not a narration
```

Stop a worker with `Stop-Process -Id <pid>` (the PID is in its `.pid` file). A
worker that produced a witnessed commit and left its lane clean is a complete run;
one that is spinning without net-witnessed gain should be stopped, not left burning
the account.

## Committing (this skill's own writes)

This skill authors/updates the **fuel** and (optionally) an audit note — not the
launched workers' code. Commit only those paths, on the trunk, by explicit path:

```bash
fak commit --preview -m "docs(super-loop): refresh the wave fuel prompt (fak super-loop)" \
  --path .claude/goal-prompts/resolve-top-issue-witnessed.md
fak commit --path .claude/goal-prompts/resolve-top-issue-witnessed.md \
  -m "docs(super-loop): refresh the wave fuel prompt (fak super-loop)"
```

Never `git add -A` (shared multi-session tree). The launched workers commit their
OWN fixes by their own explicit paths — do not sweep their in-flight edits into a
super-loop commit.

## Relationship to the sibling loops (don't reach for the wrong one)

- **`/super-loop`** (this) — DETACHED, BULK, multi-account headless launch. Workers
  survive the session; you launch and walk away. Fuel = a `/goal` pointer.
- **`/dos-dispatch-loop`** — an IN-SESSION dispatch⇄replan cadence on ONE lane, with
  a kernel-decided stop verdict. Use when you want to stay in the loop, not detach.
- **`/dos-dispatch`** — a single lane, end to end, once. The unit `/super-loop`'s
  workers effectively each run.
- **`/run-it-all-night`** — unattended DATA COLLECTION (benchmarks/witnesses), not
  issue-resolution work. Different queue, different acceptance.

## When NOT to use

- **Host not `SPAWN_OK`.** Fix the preflight refusal first; a wave on a dirty host or
  a throttled account just fails N ways instead of one.
- **One issue, one worker.** Use `/dos-dispatch` (or launch a single
  `launch_goal_detached.ps1` — dry-run it first with `-PlanOnly`, the single-spawn
  twin of the wave's default plan mode); a wave is overhead for a single leaf.
- **Self-source churn.** Do not fan out engineering workers that will free-edit
  `cmd/**` / `internal/**` in one shared checkout — that is the build-poisoning
  collision the tree-disjoint `--wave` path exists to prevent.
- **To close issues.** This skill launches work; it never closes an issue. Ancestry
  (`Fixes #N` on the trunk) does that.

## Anti-patterns

- ❌ Launching with `--live` / `-Launch` before showing the dry-run plan and getting
  approval — the plan is the witnessable artifact and the collision check.
- ❌ Treating a naive N-way burst of `launch_goal_detached.ps1` as a wave — it
  resolves the SAME account N times (`distinct_pools == 1`) and serializes on one
  rate-limit bucket. Use `launch_wave_detached.ps1` / `issue_dispatch.py --wave`.
- ❌ A multi-account wave whose workers free-edit a shared tree with no lane lease —
  distinct accounts, colliding files. Keep the fuel's `dos arbitrate` step 1 intact.
- ❌ `--force`-ing past a preflight `REFUSE_*` or an arbiter refuse — the refusal IS
  the no-DoS / no-collision floor.
- ❌ Reporting the backlog "drained" from a launch. Read `dispatch_status.py`
  closure-honesty and `dos commit-audit`; a launch count is not a ship count.
- ❌ Committing a frozen "top 30" list — the queue re-ranks daily; rank it live in
  Step 1 each run.
