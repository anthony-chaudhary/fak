---
name: wave-harvest
description: The honest closing half of a super loop — after a detached bulk wave (`/super-loop`) has run, HARVEST it: witness what each headless worker actually shipped (not what its log claims), re-queue the leaves that were claimed-done-but-not-shipped, stop workers that are spinning without net gain, and surface any lane a worker stranded dirty. A launch is not a ship, so a bulk loop is only durable if something reconciles its output against git ground truth. Read-mostly — it audits and re-queues; it may stop a spinning PID but never closes an issue by narration. Use when the operator says "harvest the wave", "what did the fleet actually ship", "reconcile the workers", "clean up after the super loop", or on a cadence between waves.
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # the witness-over-self-report and no-narration-close discipline is load-bearing and not portable per-skill
---

# /wave-harvest — reconcile a detached wave against git ground truth

> **The closing half of `/super-loop`.** A wave launches N detached workers; this
> skill answers the only question that matters afterward — *what actually shipped?*
> — from git and the issue tracker, never from a worker's log tail. It re-queues the
> claimed-but-unshipped, stops the spinners, and leaves the tree clean. Without a
> harvest pass a bulk loop drifts: workers narrate "done", the backlog looks drained,
> and nothing is witnessed. This is the pass that keeps an unattended fleet honest.

## The rule this skill enforces

**A launch is not a ship, and a self-report is not a fact.** Every "done" a worker
emitted is a claim. The harvest verdict for each leaf comes from git ancestry
(`dos commit-audit`, `dos verify`) and the issue's real state (closed by a `Fixes #N`
commit on the trunk) — the same witness `/dos-dispatch-loop` Step 3 uses, applied to
detached workers this session can't see.

## Step 0 — Enumerate what the wave launched

The launchers leave durable markers; read them, don't guess:

```bash
python tools/dispatch_status.py --md | head -50    # live vs cap, throughput, CLOSURE HONESTY, orphan cross-check
```

- `.goal-runs/*.pid` / `*.out.log` — one per detached `/goal` worker (`launch_wave_detached.ps1`).
- `.dispatch-runs/inflight-*` — one per in-repo wave worker (`issue_dispatch.py --wave`);
  each names the lane it took.

Build the work-list: for each marker, the `{pid, lane, leaf/issue}` it was launched
on. `dispatch_status.py`'s **closure-honesty** block is the headline — a low
`closure_rate` with a high `CLAIMED_CLOSED` count is exactly the drift this pass exists
to catch.

## Step 1 — Witness each claimed leaf (git, not the log)

For each leaf a worker claims it resolved, get the verdict from ground truth:

```bash
dos commit-audit --json                              # HEAD (or a worker's tip): claim vs diff
dos verify --workspace . <plan> <phase> --json       # a plan/phase actually shipped? (source: registry|grep|none)
gh issue view <N> --json state,stateReason,timelineItems   # closed by an ancestry Fixes #N, or still open?
```

Classify each leaf into one of four buckets:

- **VERIFIED** — a witnessed commit on the trunk, issue closed by ancestry. Done; drop
  it from the residual.
- **QUIET_INCOMPLETE** — the worker CLAIMED done but the oracle says NOT_SHIPPED
  (`dos verify` source=none, or `dos commit-audit` CLAIM_UNWITNESSED, or the issue is
  still open with no resolving commit). **Do NOT believe the claim** — re-queue the leaf
  (Step 3). This is the FQ-336 touch-counts-as-ship false-drain.
- **STRANDED** — a real commit exists but its lane's tree is still dirty
  (`git status --porcelain -- <lane paths>` non-empty) and the worker is gone. Durable
  work left uncommitted; surface it and commit it by explicit path (or leave it if the
  path belongs to a still-live sibling lease — check `dos arbitrate`).
- **HONEST_OPEN** — never claimed, never shipped. Fine; it stays in the queue.

## Step 2 — Stop the spinners (net-gain, not motion)

A detached worker that is still live but producing no witnessed net gain is burning an
account. Judge by witnessed work, not log volume — the `dispatch_status.py` cross-check
flags orphan processes and dead-but-live workers:

- A worker whose leaf is VERIFIED and whose lane is clean has finished — if it is still
  live and spinning (re-picking, narrating), stop it: `Stop-Process -Id <pid>` (the PID
  is in its `.pid` file).
- A worker repeatedly claiming SHIPPED while every claim reconciles QUIET_INCOMPLETE is
  the docs/351 not-ratcheting failure (motion without measured progress). Stop it and
  hand the leaf back to Step 3 — do not let it keep burning the cap.
- Leave a worker that is genuinely mid-flight (advancing commits, clean lane) alone.

## Step 3 — Re-queue the incomplete (the cross-run KEEP)

The QUIET_INCOMPLETE leaves are the ones a naive "the wave ran, backlog drained" would
silently lose. Put them back so the next `/super-loop` wave picks them up — flagged, so
they route to a verifier pass or `/dos-replan`, not blindly re-dispatched:

- If the leaf is a GitHub issue, it is still open (an ancestry close never fired), so it
  re-enters the `ready-leaves` / `p0-p1` surface automatically — just confirm it is not
  falsely marked `in-progress`/assigned by a dead worker, and clear that marker if so.
- Record the re-queue in the harvest note (Step 4) with the witness that demoted it
  (the `dos verify source=none` / `dos commit-audit CLAIM_UNWITNESSED` line), so the next
  wave has the evidence, not just a re-pick.
- **Never** re-dispatch a leaf held by a re-dispatch-invariant reason (draft-class,
  operator-gated, dependency-unmet) — surface it for the operator instead.

## Step 4 — Record the harvest (optional note) and leave the tree clean

If the operator wants a record, write a dated note under `docs/_audits/` (process
residue, kept out of the root index scan):

```text
docs/_audits/wave-harvest-YYYY-MM-DD.md
```

Shape: workers launched (count, accounts/pools), the VERIFIED/QUIET_INCOMPLETE/STRANDED/
HONEST_OPEN tally, the re-queued leaves + their demoting witness, and the PIDs stopped.
Commit only that path, on the trunk, by explicit path (`fak commit --path ... (fak claude)`),
never `git add -A`.

## What this skill does NOT do

- **It does not close issues.** Ancestry (`Fixes #N` on the trunk) closes an issue; a
  `gh issue close` off a harvest is the self-report this whole discipline refuses.
- **It does not launch new work.** That is `/super-loop`. Harvest reconciles; it re-queues
  by leaving open leaves open, not by spawning.
- **It does not re-implement fleet monitoring.** It reads `dispatch_status.py` and the
  markers; it does not build a new status surface.

## Anti-patterns

- ❌ Trusting a worker's `out.log` "done" over `dos verify` / `dos commit-audit`. The log
  is the claim; git is the witness.
- ❌ Reporting the backlog drained from launch/close counts. Read `closure_rate` and the
  per-leaf witness — a `CLAIMED_CLOSED` is not a `TRUE_RESOLVED`.
- ❌ Killing a mid-flight worker that is actually advancing. Stop spinners (no net gain),
  not progress.
- ❌ Dropping QUIET_INCOMPLETE leaves silently. Re-queue them with the demoting witness —
  a silent drop is the false-drain the harvest exists to prevent.
- ❌ Committing a STRANDED lane that belongs to a still-live sibling lease — check
  `dos arbitrate` first; commit only lanes whose worker is gone.
