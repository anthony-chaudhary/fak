---
name: dos-dispatch
description: "Plan and ship the next batch on one lane: run `dos-next-up`, acquire a lease with `dos arbitrate`, gate empty work, dispatch the packet, and archive the run. Use when a single lane should move end to end with collision safety."
---

# dos-dispatch — the generic chained snapshot→ship cycle

> **The concurrency-safe dispatch.** It chains `/dos-next-up` (the packet) to a
> ship, but first takes a **lane lease** through the admission kernel so several
> dispatches on disjoint lanes run in parallel without editing the same files.
> The "may I run on this lane" decision is the kernel's (`dos arbitrate`), not
> inline prose. Every path/lane comes from `dos doctor --json`; nothing is
> hardcoded.

The shape: **discover → take a lane → snapshot → gate → ship → archive.** The
lane taxonomy and the run-dir location are data (`[lanes]`, `[paths]`); the
admission and the gate verdict are kernel syscalls.

## Inputs

- `--lane <name>` (optional) — the lane to dispatch on (a name from the active
  `[lanes]`). Omitted = a bare auto-pick: the arbiter picks a free lane from the
  `autopick` ladder.
- `--leases <json>` (optional) — the live leases other dispatches hold, as a JSON
  list of `{lane, lane_kind, tree}` (the arbiter keys exclusivity on `lane_kind`,
  so include it — `cluster`/`keyword`/`global`). In a real loop these come from a
  status query; for a single dispatch this is usually `[]`.

## Step 0 — Discover the layout + the lane taxonomy

```bash
dos doctor --workspace . --json
```

Read `lanes` (the taxonomy), `paths.next_packets` (packet output), and
`paths.runs` (the run dir to archive under). **Use these; never hardcode a lane
name or a run path.**

## Step 1 — Take a lane lease (the admission kernel)

Ask the kernel whether this dispatch may run on the requested lane, given the
live leases. The arbiter runs the tree-disjointness algebra over the lanes'
declared trees — two dispatches on disjoint trees both ADMIT; overlapping trees
COLLIDE.

```bash
dos arbitrate --workspace . --lane <LANE> --kind cluster --leases '<LIVE_LEASES>'
```

Read the `LaneDecision` JSON: `{outcome, lane, tree, reason, free_clusters, …}`.

- `outcome: "acquire"` → admitted. `lane` is the lane to run on (may differ from
  the request when auto-pick reassigned it); `tree` is its file tree. Proceed.
- `outcome: "refuse"` → not admitted. `reason` explains why; `free_clusters`
  lists lanes you could pick instead. **Stop** (or retry on a free lane). Do not
  force — `--force` is an operator-only override, not an automation default.

The exit code mirrors the outcome (0 = acquire, 1 = refuse), so the screenplay
can branch on it directly.

## Step 2 — Snapshot the portfolio (the packet)

Run `/dos-next-up` scoped to the acquired lane:

```
/dos-next-up --scope <LANE>
```

It writes a packet under `paths.next_packets` and returns its path + a gate
verdict. Capture the packet path and its `.dispositions-<tag>.json` sidecar.

## Step 3 — Gate the empty case (typed verdict)

Before shipping, classify the packet so an empty packet doesn't launch a no-op
ship:

```bash
dos gate --workspace . <paths.next_packets>/.dispositions-<tag>.json
```

Branch on the exit code (the verdict IS the code):

- `0` **LIVE** → there is dispatchable work; proceed to Step 4 (ship).
- `3` **DRAIN** → empty backlog; **skip the ship**, archive a no-op, report drained.
- `4` **STALE-STAMP** → shipped-but-unstamped drift; skip the ship, surface the
  drift for reconciliation (a `/dos-replan` can stamp it).
- `5` **BLOCKED** → picks blocked; skip, surface.
- `6` **RACE** → lost a render race; retry the snapshot once.

## Step 3b — Price multi-agent fan-out (default admission)

If Step 4 would launch more than one worker/pick, price that fan-out before the
first worker starts. This is the default admission step, not an optional
operator check:

```
/dos-plan-price
```

Build the partition from the acquired lane tree plus each pick's declared tree
or scope. Honor the returned action:

- `LAUNCH_ALL` → proceed with the full dispatch list.
- `LAUNCH_SAFE_SET` → launch only the priced safe set; serialize the colliding
  picks into a later wave.
- `REPARTITION_AND_REPRICE` / `SERIALIZE_UNKNOWN_SCOPE` → stop or narrow the
  partition; do not launch a colliding or unknown-scope fan-out.

Record the S0 counts (`collisions_avoided`, `lanes_utilized`,
`serialization_wasted`) with the run archive when the pricing surface provides
them. The reactive `dos arbitrate` lease in Step 1 remains mandatory; pricing
does not replace it.

## Step 4 — Ship the packet (LIVE only)

Launch the packet's dispatch list (the per-pick prompts `/dos-next-up` rendered).
How you ship is host-shaped — the generic baseline launches each pick's prompt as
its own agent. Record what shipped.

## Step 5 — Archive the run

Write a run record under `paths.runs` (the run dir from `dos doctor --json`): the
lane, the packet path, the gate verdict, and what shipped. Commit it with a
generic subject — **read your trunk and ship-grammar from config; do not hardcode
a commit prefix.** (`dos doctor --json`'s `stamp` names the active grammar.)

## Step 6 — Release the lane lease

If you took a cross-process `dos lease` for the archive, release it:

```bash
dos lease --workspace . release <owner>
```

The lane lease itself is advisory state in `live_leases`; a real loop
(`/dos-dispatch-loop`) threads it forward. A single dispatch simply finishes.

## Out-of-scope findings — file an issue, don't widen the lane

Shipping on one lane surfaces work that belongs to another: a bug in a
different tree, a missing test, a doc that drifted from the code. Do not absorb
it into this run's commits — the lease covers ONE lane's tree, and a widened
diff is exactly the collision the admission kernel exists to prevent. Do not
let it evaporate either. If the workspace has a public issue tracker (on a
GitHub-hosted repo, the `gh` CLI), capture it there, then return to the leased
lane:

1. **Dedupe first.** Search before filing — `gh issue list --search
   "<keywords>"` — and comment on the existing issue rather than opening a twin.
2. **File it as a claim.** The body carries a checkable **done-condition** (the
   command or observable that would witness it resolved), a lane guess, and
   where you found it. No done-condition yet = not an issue yet — it is
   design-shaped work for the workspace's planning surface.
3. **Leak-check the drafted body BEFORE posting.** Issue text is public output
   that leaves through a door no tracked-file publication gate scans. Never
   include a machine-absolute path, a hostname, or a personal identifier; write
   paths workspace-relative. If the workspace ships a publication leak-scanner,
   pipe the draft through it first (write the body to a file outside the repo,
   scan it, post with `--body-file`); a hit is a refusal, not a warning.
4. **Close only by ancestry, never by narration.** The honest close is
   `Fixes #N` in the commit BODY of the change that resolves it — the platform
   closes the issue when that commit lands on the trunk, an ancestry check the
   claimant didn't author (the same witness `dos verify` rides). `gh issue
   close` off "I fixed it" is the self-report this kernel exists to refuse;
   non-commit closes (duplicate, wontfix) are operator moves an agent may
   propose, never execute.

## What this skill deliberately does NOT do (no silent gap)

- **No per-pick soft-claim leasing.** It takes a *lane* lease (`dos arbitrate`),
  not the heavy per-pick soft-claim core (`CLAUDE.md` heavy tier). Two loops on
  the same lane serialize on the lane, not on individual picks.
- **No rate-limit resume / focus scheduler.** Those are the host's heavy tier; a
  generic dispatch ships once and archives. `/dos-dispatch-loop` adds the cadence.
- **No host packet template / commit subject.** It reads the ship grammar from
  `[stamp]` and assembles a generic archive record.

**Log the gap, never silently skip it.** The first time the skill would have
reached for one of these (a per-pick soft-claim, a focus-scheduler pick, a
rate-limit resume), emit a one-line `log` naming what it is not doing — so the
capability gap is surfaced at runtime, matching `/dos-dispatch-loop`.

## Worked example (live transcript)

> **The shape, run for real.** `doctor → arbitrate → gate → ship → verify`,
> with the actual `LaneDecision` JSON and the exit codes that branch it. Captured
> against a live DOS workspace (`dos 0.28.0`); copy-paste, then read the RUNG.

```bash
$ dos doctor --workspace . --json
{ ... "lanes": {"concurrent": ["benchmark","docs","examples","scripts","spikes","src","tests"],
  "exclusive": ["global"], "autopick": ["benchmark","docs",...,"tests"]},
  "paths": {"plans_glob":"docs/**/*-plan.md","next_packets":".dos/verdicts","runs":".dos/runs"},
  "stamp": {"style":"grep"}, "overlap_policy": {"active":"prefix"} }
```
The WCR on-ramp — lanes/paths/stamp are DATA. Nothing below is hardcoded.

```bash
$ dos arbitrate --workspace . --lane src
{"auto_picked":true,"free_clusters":[],"lane":"benchmark","lane_kind":"cluster","outcome":"acquire","pick_count":null,"reason":"auto-picked free cluster lane benchmark (requested src was refused: lane src would edit the orchestrator's own running code … (SELF_MODIFY) …).","tree":["benchmark/**"]}
```
exit 0 (**acquire**) — but you asked for `src` and got `benchmark`: the admission
conjunction refused the hint (here SELF_MODIFY — on the kernel's own repo `src/**`
IS the running kernel; a lane contended by a live lease redirects the same way,
with the real reason in the parenthetical), so the kernel redirected rather than
double-book. A free, admissible lane you name is granted directly. Run on
`lane` (= `benchmark`), not your request. exit 1 (**refuse**) would mean stop / pick from `free_clusters`.

```bash
$ dos gate --workspace . .dos/verdicts/.dispositions-benchmark.json
```
The verdict IS the code: `0` LIVE → ship; `3` DRAIN → skip + archive no-op; `4`
STALE-STAMP / `5` BLOCKED → skip + surface; `6` RACE → retry the snapshot once.

```bash
$ dos verify --workspace . docs/82_liveness-oracle-plan liveness --json
{"phase":"liveness","plan":"docs/82_liveness-oracle-plan","rung":"direct","sha":"80d4f30","shipped":true,"source":"grep-subject","summary":"80d4f30 liveness: exclude the BIRTH acquire from the ADVANCING event count"}
```
exit 0 SHIPPED — but `source` is **`grep-subject`**, not `registry`: a commit *subject*
carrying the phase token flips this true even if little was built. Read the rung.

```bash
$ dos verify --workspace . docs/99_runtime-validation-and-the-actuation-boundary halt --json
{"phase":"halt","plan":"docs/99_runtime-validation-and-the-actuation-boundary","shipped":false,"source":"none"}
```
exit 1 NOT_SHIPPED `(source=none)` — git ancestry never stamped it. The oracle closes
the phase from git, never from "I'm done." Then `/release` cuts the version.

## Anti-patterns

- ❌ Launching a ship on a 0-pick packet — gate first; DRAIN/STALE-STAMP skip.
- ❌ `--force`-ing past a refuse in automation — a refuse means a real collision;
  pick a free lane from `free_clusters` or stop.
- ❌ Hardcoding a run dir or a commit prefix — read them from `dos doctor --json`.
- ❌ Absorbing an out-of-scope finding into the lane's commit — or dropping it.
  File an issue (dedupe → done-condition → leak-check), then stay in the leased tree.
