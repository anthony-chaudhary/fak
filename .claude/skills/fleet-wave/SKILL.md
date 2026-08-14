---
name: fleet-wave
description: "Run ONE wave of N fak-guarded ultracode sessions against the top open issues under a closing target and a wall-clock deadline — price, render fuel, launch, monitor, reconcile from git, release. N defaults to 30 (30 issues, 30 sessions, 4 hours). Use when the operator says \"spawn N ultracode sessions\", \"fleet wave\", \"close the top 30 issues in 4 hours\", or asks for a bulk guarded fan-out with a stated goal. The goal-shaped single door over /super-loop (the raw launcher) and /wave-harvest (the reconcile half); the four traps that silently void a whole wave are in the body."
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # commit-by-explicit-path, the honesty boundary, and the seat-hygiene rules are load-bearing and not portable per-skill
---

# /fleet-wave — N guarded ultracode sessions on the top open issues

> **One call, one wave, one stated goal.** `/fleet-wave` takes a target — *close N top
> issues in H hours* — and runs the whole arc: price the fan-out against the live
> ceilings, render the ultracode fuel, launch N detached **`fak guard`ed** workers,
> watch them, reconcile what actually shipped from git, and release what they held.
>
> **Default N = 30.** `/fleet-wave` alone means *30 issues, 30 sessions, 4 hours*.
> `/fleet-wave 12` retargets it. `/fleet-wave 30 8h` retargets the deadline too.

**Relation to the family — this skill does not re-implement any of it.**

| | |
|---|---|
| [`super-loop`](../super-loop/SKILL.md) | the raw bulk launcher and its regimes. `/fleet-wave` is the **goal-shaped** door over it: one wave, one closing target, one deadline. |
| [`wave-harvest`](../wave-harvest/SKILL.md) | the reconcile half. Phase 5 **calls it**; it is not copied here. |
| [`refusals.md`](refusals.md) | the worker refusal + park rules — the one canonical copy, read by every worker **by path**. |
| [`fuel-wave.md`](fuel-wave.md) | the fuel pointer template rendered per wave — budget-aware by construction; `go test ./internal/wavefuel/...` refuses a regression. |
| [`monitor.md`](monitor.md) | the read-only watcher preamble. |
| [`RATIONALE.md`](RATIONALE.md) | the evidence behind every rule here. **No worker loads it** — keep it that way. |

---

## ⛔ The four that make this not a one-liner

Settle all four before Phase 3. Each one has silently voided a whole wave.

**1. `-PointerFile` — FIXED (#5895, `2b2710ee71`), and pass it anyway.**
Both launchers used to default to `.claude/goal-prompts/resolve-tickets-witnessed.md`,
which a rename had deleted from the tree, so `Test-Path` threw `pointer file not found` on
**every** spawn and a bare `-Count 30 -Launch` was zero workers. The default now names
`resolve-top-issue-witnessed.md`, and `TestDetachedLauncherDefaultsSpawnFromABareInvocation`
(`cmd/dispatchworker/guard_test.go`) pins for both launchers that the default pointer
exists and renders under the char cap — a rename cannot silently reintroduce it.
⭐ **Still pass `-PointerFile` explicitly here**, because this skill launches a *per-wave*
rendered pointer (Phase 2) and that filename is the wave's attribution.

**2. The `/goal` condition is hard-capped at 4000 characters.**
`launch_goal_detached.ps1` builds `"/goal " + <pointer body>` and throws when
`$cond.Length > 4000`. ⛔ **That length is in CHARS, so never quote a byte count as the
margin.** Measured: `resolve-top-issue-witnessed.md` is 3988 **bytes** but 3960 **chars**
(14 multibyte glyphs), so the condition renders at 3966 — **34 chars** of headroom, not
the 6 the byte count suggests. The Phase 2 `wc -c` gate below is byte-based and therefore
*conservative*: it refuses early, which is the safe direction, but it is not the number
the launcher compares. ⇒ **The tensorbuild model of concatenating four preambles into the prompt does not
port.** Worker rules live in repo files the worker *reads*
([`refusals.md`](refusals.md), the witnessed spec); the fuel stays a *pointer*. Re-measure
after any edit — the check is one line and it is in Phase 2.

**3. The N you asked for is an aspiration, and the binding term is usually NOT the cap.**
Every spawn re-checks `dispatch_preflight.py`, whose cap is
`min(FAK_MAX_WORKERS, host_cap, dos [supervise].target, seats)`. But the wave then has to
buy *distinct-account session slots* on top of that. Measured dry-run of this exact skill,
2026-08-08, `-Count 30`:

```
WAVE PLAN  requested=30  allocation_requested=12  granted=12  shortfall=18  distinct_pools=3
  preflight: SPAWN_OK  live=0  cap=24  headroom=24  seat_free=12
```

⇒ **A 30-session ask grants 12 on this box** — bounded by `seat_free=12` across 3 distinct
t1 account pools, *not* by the cap of 24, and not by the host (`host_cap=32`, cores-bound).
⛔ **Never quote the cap as the wave size.** Run the plan and read `granted`/`shortfall`;
the binding term moves with seat health (2 seats sat `auth_failed` at this reading).
⭐ **N closes come from refill throughput, not concurrency**: workers exit, `fak dispatch
auto --live` tops the population back toward target. A `REFUSE_*` is never something to
route around; `-SkipPreflight` is an operator-only override that removes the floor for the
whole wave.

⭐ **`-Workspace` — FIXED (#5895, `2b2710ee71`).** Both launchers used to default to a
stale sibling clone missing `tools/proc_resource_guard.py`, so preflight fail-safed to
**`REFUSE_INSPECT`, cap=4, granted=0** and refused the whole wave; `launch_wave_detached.ps1`
pinned that same sibling's `.goal-runs`, which is how a probe reads *another* wave's
artifacts. Both now derive from `$PSScriptRoot` (`tools/` → repo root), and the guard test
refuses any absolute-path default. A bare invocation targets **this** checkout and reaches
the gate. **The workspace is still the tree your workers will share** — pass it explicitly
whenever that is not the checkout you are running from.

**4. Workers here SELF-SELECT their leaf — so only `fak intent` stops double-work.**
Unlike an orchestrator-assigned wave, fak's fuel has each worker pick the top-ranked ready
leaf routed to *its own* lane. The orchestrator therefore **cannot** pre-claim 30 specific
tickets. A lane lease stops two workers editing the same FILES and says nothing about two
workers doing the same TICKET in different files. The fuel makes `fak intent claim`
mandatory per worker for exactly this reason — do not weaken it.

---

## Phase 0 — Gate: is something already gardening? (decide BEFORE anything else)

The commonest way this skill wastes a turn is marching all the way to launch and only then
noticing a standing dispatcher already owns the fleet. One read settles it:

```bash
python tools/dispatch_status.py --fast
```

| card says | do |
|---|---|
| `FleetIssueDispatch` installed + enabled, workers live, commits landing | ⛔ **STOP — do not launch.** A hand-launched wave beside a live cron is double-dispatch on the same slots and lanes. Relay the card; adjust the loop, don't start a second. |
| installed but **stalled** (enabled, `live=0`, no recent commits) | Don't stack on top — **fixing why the cron is blocked IS the work**. |
| no such task, **or** the operator explicitly asked for this attended wave | Proceed to Phase 1. |

⛔ **`lane lease: N held — dead-holder=N` is NOT a blocker and must NOT be reaped.**
Measured 2026-08-08: 48 of 48 lane leases were dead-holder, some 559 h old. The admission
fold `live_leases(config, expire_dead=True)` already elides them at read time for every
consumer, so they block exactly one thing — `lane_lease.acquire()` on that specific lane.
`dos lease-lane release` runs **no liveness check** and its `--owner ""` matches any
holder, so a blind sweep evicts live work (#5859). Watch for a repeated acquire REFUSE on
one lane and escalate *that* lane; do not sweep.

## End-to-end ownership default

A wave dispatches **issue owners**, not finishers waiting for already-complete work. Every selected
issue starts in an unknown state: unstarted, partial, implemented-but-unwitnessed, or shipped. The
owner inspects and moves the actual state to a witnessed ship or a durable, clean handoff.

Price work shape as well as worker count:

- `BOUNDED`: one worker can perform root implementation and witness the issue within its context and
  deadline. It works directly and does not orchestrate.
- `BROAD`: at least two independently executable, tree-disjoint packets are required, or the whole
  issue cannot credibly fit one worker's remaining context/deadline. Dispatch a parent ISSUE_OWNER
  and reserve child capacity. Guarded delegation is mandatory when it is needed and available.
- `LEAF_CHILD`: one bounded child packet. It never orchestrates and never closes the parent.

The parent keeps the ticket claim and closure authority, starts the smallest executable root spine,
prices child tree collisions, and records an ignored execution map under the wave run directory:
acceptance item, evidence, implementation step, exact trees, witness, owner, child identity, lease,
status, and artifact. One-level fan-out is the safety boundary: enough decomposition for broad work
without fork bombs or unowned descendants.

A child-launch or lease refusal lowers concurrency only; it does not erase agent-accessible root
work. The owner continues safe root steps itself. Clean exit requires independent read-back of every
child effect, every child verified/parked/stopped, all owned intents/leases released, and either a
coherent ship or a durable park naming implemented state, missing witness, and exact next command.

## Phase 1 — Price the wave, and state the shortfall out loud

⭐ **The Phase 3 dry-run IS the pricing instrument — do not price by hand.** It folds the
preflight, the seat pool and account distinctness into one line you can quote. Run it
first (Phase 2 renders the fuel it needs), and record the base sha:

```bash
git rev-parse HEAD > .fak/wave/BASE.sha          # ⭐ unrecoverable later — Phase 5 needs it
```

Read `requested` / `granted` / `shortfall` / `distinct_pools` and the preflight's binding
term. Then say the arithmetic plainly, before spawning anything:

> *"You asked for 30. The plan grants 12 — bound by `seat_free=12` across 3 account pools,
> not by the cap of 24. So 12 launch now and the refill door tops up as they exit."*

⛔ A wave that silently under-fills reads as a launch of 30 and gets scored against a
target it never had the capacity to attempt.

**Pricing the closing target.** N closes in H hours is a *throughput* claim, not a
concurrency one. The honest form: `closes_needed / H` versus the witnessed close rate the
box has actually shown — read it from `fak dispatch progress --target <N>` and the
`watch:`/`supply:` folds of the Phase 0 card, never from a worker's log. ⛔ Do not promise
the target off worker count; promise it off rate, and report the gap when the rate is
short. Measured 2026-08-08 the card read `supply: DRAINING arrival −0.167/h vs service
0.083/h` with `loop-closed 106→106` flat over the window — a 30-in-4h target is well above
that rate, which makes it fine to *pursue* and dishonest to *promise*.

## Phase 2 — Render the fuel, and MEASURE it against the cap

```bash
WAVE=fw$(date -u +%m%d%H%M)                       # single-use; never reuse a wave id
DEADLINE=$(date -u -d '+4 hours' +%Y-%m-%dT%H:%MZ)
mkdir -p .fak/wave
sed -e "s/{{WAVE}}/$WAVE/g" -e "s/{{DEADLINE}}/$DEADLINE/g" \
    .claude/skills/fleet-wave/fuel-wave.md > ".fak/wave/$WAVE.md"

# ⛔ THE GATE. Over 4000 and every spawn throws; the wave is zero workers.
b=$(wc -c < ".fak/wave/$WAVE.md"); echo "cond=$((b+6)) cap=4000"; [ $((b+6)) -lt 4000 ] || echo "REFUSING — shrink the fuel"
```

⭐ **The rendered filename is the wave's attribution, for free.** The launcher sets
`$tag = <pointer basename>` and writes `<LogDir>/$tag-$stamp.{out.log,err.log,pid,in.txt}`,
so a per-wave pointer name makes every log, pid crumb and stdin capture self-identifying.
Reuse a pointer name across waves and the previous wave's artifacts answer your probes as
if they were this run's — a stale predecessor vouching for a corpse.

## Phase 3 — Launch. PLAN by default; `-Launch` is the opt-in.

```powershell
# The plan — spawns NOTHING. This is the witnessable artifact AND the pricing; show it first.
.\tools\launch_wave_detached.ps1 -Count 30 -PointerFile ".fak\wave\$WAVE.md" `
    -WorkKind engineering -Workspace "C:\work\fak"

# Then, on approval, actually dispatch:
.\tools\launch_wave_detached.ps1 -Count 30 -PointerFile ".fak\wave\$WAVE.md" `
    -WorkKind engineering -Workspace "C:\work\fak" -Launch
```

⭐ **Both arguments are still worth passing** (§ 1 and § 3), though neither is a landmine
any more: `-PointerFile` because this skill launches a *per-wave* rendered pointer whose
filename is the wave's attribution, and `-Workspace` because the tree your workers share
should be a decision, not an inference. Before #5895 both defaults were wrong and both
failures read as "the wave ran" — that is why they are named here at all.

**What the launcher already solves — do not re-implement any of it.** It asks the switcher
for N *distinct-account* session slots in one call (distinctness by Anthropic
`accountUuid`, so two dirs on one account never both inflate the count), then dispatches
each slot through `launch_goal_detached.ps1`, which owns the dangerous parts:

- ✅ **Guarded by default.** Each worker runs under its **own** `fak guard` gateway
  (`-Guarded`), which is what "fak guarded ultracode session" means here — real tool-floor
  adjudication and a hash-chained audit per worker. Witnessed on this box: **50,055 kernel
  decisions across 513 dispatch sessions.**
- ✅ **Seat hygiene.** It strips `ANTHROPIC_*` and the session-identity vars *before*
  pinning the account. ⛔ **This is why you must NOT hand-wrap workers in your own guard.**
  A parent under `fak guard` exports a loopback `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY`;
  env precedence beats the seat's OAuth, so an inheriting child bills the parent's seat and
  dies the instant the parent gateway exits — the whole-wave same-instant crash. The
  child-stderr tell is `claude.ai connectors are disabled because ANTHROPIC_API_KEY … is set`.
- ✅ **Per-spawn preflight.** The `SPAWN_OK` gate is re-checked at *every* spawn, so the
  wave under-fills mid-flight the moment the host, seat pool, or cap refuses.
- ✅ **stdin-fed prompt.** The condition goes in via a UTF-8 file, never `-ArgumentList` —
  backticked commands and `--flags` would otherwise be re-split and choke claude's parser.

⚠️ **One honest caveat, and it bounds the wave:** a just-spawned worker is stdin-fed and
carries no scannable process marker, so the per-spawn re-check cannot see a sibling until
it holds a lane lease. **Size the wave from the plan; do not re-run the launcher to "top
up" while workers are still starting.** Refill belongs to `fak dispatch auto --live` on a
cadence, which reads live population rather than racing it.

## Phase 4 — Monitor. Watchers see what workers inside the tree cannot.

```bash
RUNS="$(git rev-parse --show-toplevel)/.goal-runs"          # ⭐ the workspace you launched with
ls -t "$RUNS"/$WAVE-*.pid | wc -l                           # spawned
cat "$RUNS"/$WAVE-*.out.log                                 # ⭐ READ THIS FIRST — see below
tail -5 "$RUNS"/$WAVE-*.err.log                             # guard's own exit summary
fak dispatch status --runs-dir "$RUNS" --json
```

⛔ **Derive the runs dir from the workspace you launched with — never hardcode the
sibling checkout.** `launch_goal_detached.ps1` sets `$LogDir = Join-Path $Workspace
'.goal-runs'`, so probing a different tree silently reads *another* wave's artifacts. The
default sibling's `.goal-runs` held **941** of them at last count and ids recycle — that
is the stale predecessor vouching for the corpse, self-inflicted.

⭐ **`out.log` is the highest-value read in the whole phase and it is ~117 bytes.** It
carries the upstream refusal verbatim. When the wave dies together, three causes present
identically and this one line separates them:

| out.log says | cause | next |
|---|---|---|
| nine **distinct** `session <id>` + `BUDGET_CONTEXT_EXHAUSTED` | budget starvation | join the guard session id from `err.log` onto `.fak/nightrun/compaction-health.jsonl` |
| **one** session id repeated across all workers | seat/env collapse | the launcher's strip-then-pin regressed — do not hand-wrap |
| empty, processes actually alive | broken probe | bash `kill -0` lies on native Windows pids; use PowerShell `Get-Process -Id` |

⛔ **Healthy compaction does NOT exonerate the context path.** `anchor_starved=0`,
`solvency_forced=0` and all-correct bail reasons (`under_budget`, `too_few_msgs`,
`burst_unprofitable`) mean the compactor worked *and the session died anyway* — which
indicts the CUMULATIVE drain ceiling (`--context-budget-tokens`), a different knob on a
different scale from the per-turn shed-line. A worker reads a comfortable
`ctx:83.2k/96.0k` on the turn its cumulative budget dies. Never compare the two numbers.

Spawn 2–3 **independent read-only** monitor sessions with [`monitor.md`](monitor.md).
Their job is the three things no worker can see from inside: a **committed-red trunk**, a
**ticket with two owners**, and a **silent death**.

⛔ **A monitor told to "sample every 3 minutes" DIES after pass one.** A session ends when
the model stops calling tools — announcing a wait *is* exiting. The sleep must live inside
one Bash call: `for i in 1 2 3; do { <checks>; } >> $LOG 2>&1; sleep 180; done`, then the
model reads the chunk, judges, and immediately issues the next one.

⛔ **Verify every alert before acting.** A watcher that reports the whole wave dead is far
more often a broken probe than a dead wave — require an explicit `ALL_CLEAR` enumeration
too, or silence is indistinguishable from a broken watcher.

⛔ **Never `Read` a worker's raw transcript** — it is JSONL and will blow up your context.
Read the `.err.log` tail and the report fields.

## Phase 5 — Reconcile from git, then release. Reports understate the work.

```bash
/wave-harvest                                                  # the reconcile half — call it
git log "$(cat .fak/wave/BASE.sha)"..HEAD --pretty='%h %s' --name-only
fak dispatch progress --target 30 --json                       # closes vs the stated goal
fak dispatch closure-audit --workspace . --json
```

- ⭐ **Build the ledger from `git log`, not from reports.** A long, healthy report routinely
  undercounts its own author, because the worker writes its `LANDED:` line before its last
  commits land and never returns. Reconcile dispatched ids against the range diff and treat
  every report as commentary.
- ⭐ **Read every parked tag message — that is where a wave's best work hides.**
  `for t in $(git tag -l "hold/$WAVE-*"); do git log -1 --format='=== %S%n%B' "$t"; done`.
  Then **post those accounts to the tickets yourself**: the worker that did the work is
  gone, and this is the orchestrator's job precisely for that reason.
- ⛔ **Push the `hold/*` refs — a worker physically cannot.** `git push` of a raw ref is
  `REQUIRE_WITNESS`-gated, so every park is durable on this box only until you close the
  loop. Refs live in **two** namespaces and a check over one reads a strand as
  over-published:
  ```bash
  git for-each-ref --format='%(refname)' refs/tags/hold refs/hold | wc -l
  git ls-remote origin 'refs/tags/hold/*' 'refs/hold/*' | sed 's/\^{}$//' | sort -u | wc -l
  ```
  Re-run **both counts after pushing** and report the closed number — the push is not the
  evidence, the recount is.
- **Release the ticket claims.** A lane release does not touch `fak intent`. Finish with
  `fak intent list` showing none of your wave's holders:
  ```bash
  fak intent list | jq -r '.[] | select(.holder|startswith("'"$WAVE"'-")) | .target'
  ```
  The leak is soft — a 3600 s TTL and `fak garden` reap it — but inside that hour they read
  **live** to every peer, which is exactly the window the next wave ranks in.
- ⛔ **Report the target honestly.** *"30 sessions launched"* is not *"30 issues closed."*
  Close the wave with: launched / still live / witnessed closes against the target / parked
  but unlanded / and the shortfall named. A launch is not a ship.

---

## Trap index

| # | trap | tell |
|---|---|---|
| 1 | ~~default `-PointerFile` does not exist~~ **fixed #5895** | was: every spawn throws `pointer file not found`; wave = 0 workers. Now pinned by a guard test |
| 2 | fuel over 4000 chars | `goal condition is N chars (>4000 cap)` — thrown per spawn. **Chars, not bytes** — a byte count understates the margin |
| 3 | asking for N > what seats allow | plan shows `granted` ≪ `requested`; the binding term is usually `seat_free`, **not** `cap` |
| 3b | ~~default `-Workspace` names a sibling clone~~ **fixed #5895** | was: `REFUSE_INSPECT … guard not found: …\tools\proc_resource_guard.py`, cap=4, granted=0. Both launchers now derive from `$PSScriptRoot` |
| 4 | hand-wrapping workers in your own `fak guard` | whole wave dies together when the parent gateway exits |
| 5 | re-running the launcher to "top up" | the per-spawn gate can't see starting siblings — use `fak dispatch auto` |
| 6 | reaping dead-holder lane leases | `dos lease-lane release` has no liveness check; `--owner ""` evicts live work |
| 7 | two workers, one ticket, lanes both green | lanes partition FILES, not TICKETS — gate on `fak intent claim` |
| 8 | `fak intent claim … \| head` | `$?` is the pipe's: a live collision reads rc=0 and stays on the roster |
| 9 | reusing a wave id / pointer name | last wave's logs and pid crumbs answer as if they were this run's |
| 10 | monitor announcing a wait | one pass, clean exit, no further samples — sleep must be inside the Bash call |
| 11 | trusting a worker's `LANDED:` line | report says none; `git log` for that lane says three |
| 12 | `hold/*` refs never pushed | `for-each-ref` count ≫ `ls-remote` count — and only if you check BOTH namespaces |
| 13 | an OPEN issue read as unstarted | the fix is already on the trunk under a commit that never cited it |
| 14 | reporting launches as closes | "30 sessions" ≠ "30 issues" — only a witnessed `Fixes #N` on the trunk closes |
