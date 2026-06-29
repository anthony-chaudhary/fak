# Epic close-out method — how to properly close (or retire) an epic

*2026-06-29.* The open backlog is **epic-heavy**: 57 of 171 open issues (33%) carry
the `epic` label. Many are *not* in-flight work — they are finished-but-unclosed, a
parallel duplicate of a canonical plan, or a perpetual maintenance loop that can never
reach "done." Each of those inflates the apparent scope and hides the real frontier.
This note is the repeatable method for closing an epic *honestly* — and for retiring the
ones that should never be "closed as completed" at all.

## The two failure modes a close-out must avoid

1. **Premature / false close.** Closing an epic whose children were *closed by a docs
   commit* (or just toggled to `CLOSED`) without the code actually shipping. A child's
   `CLOSED/COMPLETED` state and a `- [x]` checkbox are **self-reports**, not witnesses —
   they false-close (see the syspromptmmu epic #1258, whose child issues were closed by
   docs commits while the code stayed UNBUILT). Never close an epic on issue-state alone.
2. **Stale-open.** Leaving an epic open long after its deliverable shipped, because nobody
   re-checked. Roadmap epics lag the code by design; a scout once found 8 already-shipped.

The method below is calibrated so that **neither** a green checkbox nor a stale prose ref
is ever the deciding evidence.

## Step 1 — Build the close-state table (read-only)

Epics encode children as **markdown task-list checkboxes** (`- [ ] #N` / `- [x] #N`),
*not* GitHub native sub-issues (the `sub_issues` API is empty here). Body `#N` refs are
usually real post-migration issue numbers, but **some are stale pre-migration tracker IDs**
— `gh issue view` before trusting one. Compute completion in bulk (one `--state all` pull,
joined locally — per-issue `gh issue view` over hundreds of refs times out):

```bash
gh issue list --state open  --label epic --limit 100 --json number,title,body > epics.json
gh issue list --state all   --limit 4000 --json number,state,stateReason > allstates.json
# then, per epic: count [x]/[ ] boxes, and split body #refs into OPEN vs CLOSED via allstates
```

Classify each epic into one of the three close-out paths below by `(checkbox ratio,
open-children, standing?)`. A clean **A** candidate has *every* box checked **and** zero
open child issues. A **C** epic's own title/intro says "keep … alive / fresh", "ratchet",
"each round" — it has no terminal state.

## Step 2 — Route to one of three paths

### Path A — COMPLETED (witnessed-done)

Pre-conditions, **all** required:
- every child rung is `CLOSED`, and
- the deliverable is **witnessed in the tracked tree** — the artifact exists on disk, the
  claimed ship SHAs are real commits, and `dos commit-audit <sha>` returns `OK /
  diff-witnessed` (not a docs-only or `--allow-empty` commit), and
- any residual is **carved** — either a named follow-on issue, or a *standing debt number a
  scorecard re-surfaces every run* (an acceptable carve: it cannot be silently lost).

Verify the witness, don't assume it:

```bash
for s in <sha…>; do git cat-file -t "$s" && git log -1 --format='%s' "$s"; done   # real commits?
ls <artifact-path>                                                                 # deliverable on disk?
grep -l <thing> tools/scorecard_control_pane.py                                    # actually wired?
```

Then close with the house-style evidence comment (mirrors how #1147 was closed):

```bash
gh issue close <N> --reason completed --comment "Closing **COMPLETED** — <DoD met>.
Resolved by \`<sha>\` … witnessed via dos commit-audit (OK / diff-witnessed).
Residual (carved): <named follow-on / scorecard debt>. Reopen if this does not fully resolve it."
```

*Worked example:* **#1116** (external-onboarding) — 4/4 children closed; SHAs `bad13960`/
`6e230af2`/`1cd4c7ee`/`72cfdc42` verified real + diff-witnessed; `tools/claim_repro_scorecard.py`
present and control-pane-wired; 3 residual un-falsifiable claims left tracked by the
scorecard's own debt number. Closed COMPLETED 2026-06-29.

### Path B — NOT_PLANNED (superseded / deduped)

When an epic is a parallel or absorbed duplicate of a canonical one. Close pointing to the
survivor and fold any unique framing there as comments — keep **one** plan on the trunk per
shared-trunk/DOS discipline:

```bash
gh issue close <N> --reason "not planned" --comment "Superseded by #<canonical> …
Closing this parallel tree; additive framing folded into #<canonical>'s children."
```

*Worked example:* **#1226** → **#1217** (two parallel context-safety epics; the later,
more-granular one survived). Likely current dedup candidates worth a read: the serving
cluster (#50 / #637 / #931 / #805 / #809 / #977) and the agent-OS cluster (#748 appears to
umbrella #749 cron / #750 liveness / #1178 dormancy / #1193 lifecycle — confirm parent-vs-peer
before closing any).

### Path C — STANDING (retire from the deliverable count; do **not** close as done)

A "keep-the-scorecard-alive / keep-the-map-fresh" epic is a **perpetual ratchet**, not a
finite deliverable — it can never reach COMPLETED, and closing it as done would discard the
standing tracker. Proper close-out is to **stop it masquerading as in-flight scope**:
- split its *finite* children into standalone issues (so they show up in the real frontier), and
- drive the recurring part from a `/loop` or `/schedule` routine (the epic body usually already
  prescribes this — e.g. #710 says "run `/stability-score` on a `/loop`"), and
- mark the umbrella as standing (an additive `standing`/`maintenance` label, or a pinned
  tracking issue) so triage can filter `epic AND -standing` for true deliverables.

*Current standing epics:* **#710** (stability scorecard), **#801** (bench-dx scorecard),
**#1050** (industry/vLLM-parity map) — the "keep-the-scorecard-alive" trio; **#1278**
(docs-freshness loop) is the same shape.

## Anti-gaming laws

1. **Witness, never self-report.** A `- [x]` box and a child's `CLOSED` state are claims;
   the deciding evidence is the tracked-tree artifact + a diff-witnessed SHA.
2. **Carve residual, don't bury it.** An epic only closes once every loose end is a named
   follow-on or a scorecard debt number — never an unwritten "someone will finish this."
3. **Standing ≠ done.** Never close a perpetual loop with `--reason completed`; retire it
   from the deliverable count instead.
4. **One plan on the trunk.** Prefer Path B (supersede) over leaving two epics that drift.
