---
name: super-loop
description: 'Plan, price, launch, monitor, and reconcile bulk headless issue-resolution work safely. Use when an operator asks for a super loop, worker wave, detached issue workers, backlog draining, capacity/status, stale-worker cleanup, ramp-up, or an overnight fleet. Dispatches end-to-end issue owners, requires explicit launch intent, prices account and tree capacity, verifies effects from git or DOS witnesses, and closes every child, lease, and intent cleanly.'
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

**Codex default.** The native Codex wave is the closest supported Ultracode-like shape:
one guarded Codex process per admitted issue and switcher seat. Never fan out raw `codex exec`
processes or share one interactive `CODEX_HOME` across children.

## Two launch paths — pick the one that matches your risk

| Path | What it gives | When |
|---|---|---|
| `fak dispatch wave` | The native Go wave: preflight cap + account/seat allocation + pairwise **TREE-DISJOINT** lanes priced in-process (`dispatchorder`, no per-lane `dos arbitrate` shell-out). Its hold is narrowed to the **trust-critical** set only (`internal/{abi,kernel,adjudicator,policy,registrations,architest,shipgate}` + `dos.toml`/`.dos`/`policy.json`/`VERSION`), so **core `internal/**` lanes (gateway/engine/agent/compute/…) dispatch by default** — not just docs/tools. Concurrent core work is kept build-safe by the push-seam `TRUNK_WOULD_NOT_COMPILE` gate (`fak hooks pre-push`). | **Default.** In-repo issue work on the shared trunk. |
| `python tools/issue_dispatch.py --wave` | The **legacy** Python wave (a compat shim; `docs/dispatch-loop.md`). It blanket-holds ALL of `cmd/**`+`internal/**`, so it only ever offers docs/tools — the reason a wave "never had real work." Prefer the Go path above. | Legacy / fallback only. |

The catch that makes this skill load-bearing: **the wave gives account session
capacity, but its workers share ONE working tree** (a single `-Workspace`).
Account slots ≠ distinct file trees. So the multi-session wave is only
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

## Issue ownership is end to end

A dispatched worker owns root implementation, not merely reconciliation of work assumed complete.
It inspects whether the issue is unstarted, partial, unwitnessed, or shipped, then performs what is
still owed through witness and clean closeout.

Classify before dispatch: `BOUNDED` goes to one direct owner; `BROAD` goes to a parent ISSUE_OWNER
with reserved capacity for independently executable, tree-disjoint packets; `LEAF_CHILD` is one
bounded packet and cannot orchestrate or close the parent. For BROAD work, the parent MUST use the
managed guarded launcher when delegation is needed and capacity exists. Fan-out is one level deep.

The parent retains the issue claim and closure authority, begins the smallest root spine itself,
records an inspectable execution map, prices child collisions, independently witnesses child
effects, and integrates one coherent result. A refused child lowers concurrency, not ownership.
The parent keeps doing agent-accessible root work and exits only after every child is verified,
parked, or stopped and every owned lease/intent is released.

## Step 0 — Is the loop already running? (decide BEFORE you orient)

The single most common way this skill wastes a turn: it marches orient → reclaim →
rank → price → launch, and only at the *end* notices a **standing dispatcher was
already doing the work.** So the first fact to settle is not "is it safe to spawn" —
it is **"is anything already spawning?"** One cheap, pure-local read answers it:

```bash
fak dispatch status --json   # native Go status fold: live workers, throughput, and closure honesty
# (Legacy fallback: python tools/dispatch_status.py --fast)
```

Read the watchdog fold and the live-worker count, then branch. This is a **gate**,
not a step you pass through on the way to a launch:

| What the card shows | Regime | Do |
|---|---|---|
| `FleetIssueDispatch` installed + enabled, workers live, commits landing | **ALREADY GARDENING** — the standing loop owns the fleet | **STOP — do not launch.** A hand-launched wave beside a live cron is double-dispatch on the same slots and lanes. Jump to *When the loop is already running* below and act there. |
| `FleetIssueDispatch` installed but **stalled** — enabled yet `live=0`, no recent commits, or a throttle fold | **CRON DOWN** — the loop exists but isn't firing | Don't stack a parallel wave on top. **Fixing why the cron is blocked IS the work** (Step 0.5 reclaim rungs, or recover a `REFUSE_*`); then let the cron dispatch. |
| No `FleetIssueDispatch` task at all, **or** the operator explicitly asked for an attended one-shot wave | **HAND-LAUNCH** — nothing standing to collide with | Proceed to Step 0.1 and the launch procedure below. |

Only the third row leads into the rest of this skill. The first two are the whole
answer for their turn — the job there is to *witness and adjust the loop that
already runs*, never to start a second one beside it.

### When the loop is already running

You are not in a launch turn. Do not fall through to Steps 1–3 — those start a
*second* dispatcher. The high-value moves, cheapest first:

- **Report it.** The `dispatch_status.py` card already IS the status — live workers,
  throughput windows, closure honesty. Relay it; don't re-derive it.
- **Harvest it.** `/wave-harvest` reconciles what the standing workers actually
  shipped from git (a launch is not a ship) and re-queues the claimed-but-unshipped.
- **Unblock it.** If the card names a blocker (silent workers, orphan leases, a
  `WEEKLY_CAPPED` seat), fix that one thing — a firing cron then refills itself.

Report what the loop is doing and what you changed, and stop.

## Step 0.1 — Orient the hand-launch (only if the gate sent you here)

Now the launch-safety question — is it safe to spawn, and what is the fuel?

```bash
python tools/dispatch_preflight.py --json     # SPAWN_OK  or  REFUSE_{INSPECT,HOST,NO_SEAT,AT_CAP,NO_ACCOUNT}
```

(`--md` on `dispatch_status.py` is not a display flag — `--md <path>` WRITES the
committed markdown status doc; the human-readable card is the default stdout
output, and `--fast` skips the two gh-backed folds for a quick look.)

**Pick the regime from what you observed — never hard-code "launch a wave":**

| Observation (preflight verdict + status card) | Regime | Where |
|---|---|---|
| `SPAWN_OK`, workers already live and shipping | FULL wave at the remaining headroom | Steps 1–3 |
| `SPAWN_OK` but cold: `live=0`, first wave of the day, or the fuel/launcher just changed | CANARY — 1 worker, promote on evidence | Step 1.5 |
| `REFUSE_HOST`, or the card shows silent workers / orphan leases / dead-PID residue | RECLAIM first, then re-orient | Step 0.5 |
| `REFUSE_AT_CAP` and the live workers are real (leases held, commits landing) | Not a launch turn — watch, don't spawn | Step 4 |
| `REFUSE_NO_SEAT` / `REFUSE_NO_ACCOUNT` / card says `WEEKLY_CAPPED` | WAIT-FOR-RESET — the seat comes back on a window, not on retry | Step 5 signals |
| Operator asked for an overnight / 12h+ run | MARATHON — a cadence of waves, not a bigger wave | Step 5 |

A `REFUSE_*` still means what it always meant: recover per the AGENTS.md guard
table, never route around it. The regimes are the *named* recover/launch paths —
none of them is an override.

The **fuel** is the `/goal` pointer each headless worker reads
(`.claude/goal-prompts/resolve-top-issue-witnessed.md`) — a self-contained spec:
*take a lane, resolve the top ready leaf, ship it witnessed, close by ancestry,
stop.* Keep it < 4000 chars (the `/goal` cap the launcher enforces).

## Step 0.5 — Reclaim: clean up old workers before growing

An `AT_CAP` / `REFUSE_HOST` on a quiet box usually means *residue*, not load:
dead workers still holding inflight markers, silent spinners, orphaned helper
sprawl. Reclaim in three rungs, cheapest first — and re-run Step 0 after, because
**a reclaim is not admission**: the freed capacity has to be witnessed by the
preflight gate, not assumed.

**Rung 1 — free, no kills (safe to run any time):**

```bash
python tools/issue_dispatch.py --no-refresh     # ANY dry-run tick self-heals inflight markers (dead PID / unreadable / >12h TTL)
python tools/stale_work_watchdog.py --live      # GC >7d gitignored ephemera ONLY (.dos/markers|streams|stop-failures, tools/_watchdog); never touches git state
```

**Rung 2 — read the evidence (still no kills):**

```bash
python tools/proc_resource_guard.py --json      # the exact runaway/orphan report REFUSE_HOST is built on
python tools/dispatch_status.py                 # silent-worker (stub log + dead PID) and orphan process/lease folds
```
```powershell
Get-ScheduledTaskInfo FleetRunawayReaper        # a standing reaper may already be on it — don't double-reap
```

**Rung 3 — kill, operator-gated (the same approval bar as `-Launch`):**

```powershell
Stop-Process -Id <pid>    # only a worker the card proves is dead-weight: silent (stub log) or spinning with zero witnessed commits
```
```bash
python tools/proc_resource_guard.py --enact --reap-orphans   # kill FLAGGED runaways/orphans only; protected processes are never killed
```

Never `--enact` from automation or without reading the rung-2 report first, and
never kill a worker that is mid-witnessed-progress — a held lease plus a recent
commit is a live worker, not residue. Killing whatever looks big to route around
`REFUSE_HOST` is the same sin as `-SkipPreflight`.

## Step 1 — Rank the queue (the "top N" fuel, live — never a frozen list)

The backlog re-ranks daily, so the queue is a **live query**, not a committed doc.
The dispatchable surface is the `ready-leaves` / `p0-p1` views; the deterministic
"do next" order is `tools/issue_triage.py`'s integer score (P0 1000 · P1 400 ·
P2 150 · none 60; +300 orphan P0/P1, +40 bug, + idle-age):

```bash
fak dispatch order --json                                   # deterministic candidate ordering & cooldown math
fak issue-orchestrator --top 10                             # top priority leaves and safe wave candidates
fak console issues --state open --json > open-issues.json   # current open issues census
# (Legacy fallback: python tools/issue_lane_router.py --view p0-p1 --json)
```

Read the top N (default N = `--max-workers`) rows. These are the leaves the wave
will pick from; each worker selects the top-ranked leaf on the lane it leases.

## Step 1.5 — Size the wave: ramp rungs (the next tick IS the ramp)

Three pacing knobs now exist, each with a DIFFERENT intent — don't confuse them for
a ramp, and don't collapse them into one:

- `--stagger-s` / `FLEET_LAUNCH_STAGGER_S` (`issue_dispatch.py --wave`, #3610) — spaces
  members inside the prompt-cache TTL so workers 2..N READ the warm ~35.8k floor prefix
  instead of each paying a cache-write. Pairs with `--warm-floor`. Default 0.0 (off).
- `--settle-s` (`cmd/fak/dispatch_wave.go`) — spawn settling in the Go wave driver.
- `Invoke-SpawnPacing` (`launch_wave_detached.ps1`) — jittered anti-burst protection for
  per-account rate limits. Its jitter deliberately DE-synchronizes spawns; keep it.

None of them is a ramp. The ramp primitive is still running the launcher *again*,
smaller first. Pick the rung from evidence, not appetite; `<N>` below is what Steps 2–3 get:

| Rung | When | How |
|---|---|---|
| **CANARY** (1) | cold host, first wave of the day, fuel/launcher just changed, or right after a reclaim | single-tick `issue_dispatch.py --live` (no `--wave`) spawns exactly one; or one `launch_goal_detached.ps1` after `-PlanOnly` |
| **STEP** (2–3) | the canary shipped and headroom is confirmed | `--wave --max-workers 3`; re-run Step 0 + Step 2 before the next rung |
| **FULL** (headroom) | the previous rung landed witnessed commits and the card is healthy | `--wave --max-workers <preflight headroom>` |

Promote on WITNESS, not on time: a canary is promoted when it holds a lane lease
and its first commit passes `dos commit-audit` — not because twenty minutes
passed. Between rungs re-run the preflight: the adaptive host cap rises as load
clears, so each successive plan is honest. A rung that shows stub logs or
throttle folds means drop BACK a rung, not push through. And the Step-0 caveat
still binds — never "top up" while workers are still starting (a just-spawned
worker is invisible to the scan until it holds a lease); a new rung begins only
after the last rung's workers hold leases/markers or have exited.

## Step 2 — Price the fan-out (dry-run) — collisions AND account capacity

Never launch blind. Run the launcher in its default PLAN mode and read the plan:

```powershell
# Ask the switcher for offered seats, then native Go DRY-RUN (no --live).
fak fleet-accounts wave --count <N> --work-kind codex --product codex --json
fak dispatch wave --count <N> --backend codex --work-kind codex --max-workers <N> `
  --goal high-priority --workspace . --json
```

```powershell
# High-throughput path — N account session slots (plan only, no -Launch):
.\tools\launch_wave_detached.ps1 -Count <N> -WorkKind engineering -Workspace C:\work\fak `
  -PointerFile .claude/goal-prompts/resolve-top-issue-witnessed.md
```

Read the plan out loud for the operator:

- **Tree-disjoint (`fak dispatch wave`):** confirm each lane's tree is pairwise
  disjoint and that the plan now includes core `internal/**` lanes (gateway/engine/
  agent/…), not only docs/tools — the trust-critical hold still holds kernel/adjudicator/
  policy/etc. A colliding set is priced out (serialized into a later wave), not launched.
- **Multi-account (`fak dispatch wave`):** use the preceding `fleet-accounts wave`
  receipt as authority; confirm `granted`, `shortfall`, `distinct_pools`, and each lane's
  `config_dir`, `pool`, and `session_slot`. A single pool is not a useful multi-account
  wave unless its explicit session cap permits the offered slots. `status` is health
  context, not allocation.

If the plan shows collisions, an unavailable account, or fewer slots than needed, fix
the partition or wait — do not `--force` / launch anyway.

## Step 3 — Launch the wave (opt-in, operator-approved)

Only after the plan is clean AND the operator approves the real spawn:

```powershell
fak dispatch wave --count <N> --backend codex --work-kind codex --max-workers <N> `
  --goal high-priority --workspace . --live --json
```
```powershell
.\tools\launch_wave_detached.ps1 -Count <N> -WorkKind engineering -Launch -Workspace C:\work\fak `
  -PointerFile .claude/goal-prompts/resolve-top-issue-witnessed.md
```

Each spawn re-checks the preflight cap, so the live population still never exceeds
the account session-slot cap even mid-wave. Record what launched: the per-lane account/pool/PID and
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

## Step 5 — Marathon: runs longer than one wave (12h+ / overnight)

A worker is one-leaf-then-stop by fuel design, so a long run is a **cadence of
waves**, never a long-lived worker or one bigger burst. Two honest shapes:

- **Standing cron (preferred unattended).** If the Step-0 gate found
  `FleetIssueDispatch` live, the overnight run is *already running* — leave it on,
  fix what the card says blocks it, and harvest on a cadence (`/wave-harvest`).
  Don't hand-launch waves beside it; that is the double-dispatch the gate exists
  to prevent.
- **Attended cadence (wave-sized ticks).** Repeat orient → rung-1 reclaim →
  price → wave every 60–90 minutes. Every tick re-runs the Step-0 gate first: a
  cron may have come up (or a preflight verdict gone stale) since the last tick,
  and the queue re-ranks live.

Budget and stop signals — read them each tick, they are the marathon's honesty:

- **`WEEKLY_CAPPED` / seat cooling** — that account is out for the window, and
  waiting IS the correct move; `fak resume status --store <projects-dir>` names
  the earliest fire-eligible session and the exact resume command. Downshift to
  `--work-kind gardening` (tier 2) only if t2 seats are genuinely free.
- **Throughput flat** — the card's 1h/3h/6h/12h/24h trailing windows are the
  witness: launches rising while ships stay flat across two consecutive windows
  means STOP and investigate (silent workers, stub rate), not wave again.
- **Backlog drained** — `ready-leaves` empty means the marathon is DONE; report
  it as done, don't idle-tick.
- **12h markers** — inflight markers auto-expire at 12h; a marker that old is
  residue for rung-1 reclaim, not evidence of a 12-hour worker.

The end-of-marathon report is per-tick launches, witnessed SHIPS (`dos
commit-audit` / closure-honesty), what was reclaimed, and which stop signal ended
the run. Hours elapsed is not a result.

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
- **`/issue-queue`** — the ATOMIC, BOUNDED issue management and queuing pass. Resolves
  1–3 issues with reproduction tests, package-scoped fences, and independent witnessing.
  The issue counterpart of `/debt-clean`.
- **`/issue-orchestrator`** — CAMPAIGN-SCALE multi-wave issue resolution. Partitions the
  backlog into concurrent-safe waves and tracks milestone burndown.
- **`/dos-dispatch-loop`** — an IN-SESSION dispatch⇄replan cadence on ONE lane, with
  a kernel-decided stop verdict. Use when you want to stay in the loop, not detach.
- **`/dos-dispatch`** — a single lane, end to end, once. The unit `/super-loop`'s
  workers effectively each run.
- **`/run-it-all-night`** — unattended DATA COLLECTION (benchmarks/witnesses), not
  issue-resolution work. Different queue, different acceptance.

## When NOT to use

- **Host not `SPAWN_OK`.** Fix the preflight refusal first (the Step-0.5 reclaim
  rungs are the named path); a wave on a dirty host or a throttled account just
  fails N ways instead of one.
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
  resolves without account session-cap accounting and can pile onto one account.
  Use `fak dispatch wave`, `launch_wave_detached.ps1`, or `issue_dispatch.py --wave`.
- ❌ A multi-session wave whose workers free-edit a shared tree with no lane lease —
  account slots, colliding files. Keep the fuel's `dos arbitrate` step 1 intact.
- ❌ `--force`-ing past a preflight `REFUSE_*` or an arbiter refuse — the refusal IS
  the no-DoS / no-collision floor.
- ❌ Reporting the backlog "drained" from a launch. Read `dispatch_status.py`
  closure-honesty and `dos commit-audit`; a launch count is not a ship count.
- ❌ Committing a frozen "top 30" list — the queue re-ranks daily; rank it live in
  Step 1 each run.
- ❌ A full-size wave onto a cold or just-reclaimed host — canary first, and promote
  on witnessed commits, never on elapsed time.
- ❌ `--enact` / `Stop-Process` from automation, or without reading the rung-2
  report — a reclaim kill carries the same operator bar as `-Launch`, and killing
  whatever looks big to clear `REFUSE_HOST` is `-SkipPreflight` by other means.
- ❌ Marching orient → reclaim → rank → price → launch WITHOUT first running the
  Step-0 gate — the whole reason the skill felt repetitive. Settle "is a loop
  already running?" in one read before doing anything else.
- ❌ Running "12 hours" as one bigger burst, or hand-launching waves while the
  `FleetIssueDispatch` cron is live — double-dispatch on the same slots and lanes.
- ❌ Re-waving into flat throughput — launches rising while ships stay flat is a
  stop signal to investigate, not a sizing problem to push through.
