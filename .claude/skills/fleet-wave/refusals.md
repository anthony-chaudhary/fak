# refusals.md — the wave's refusal + park rules

> **Read by every wave worker, by path, from the fuel pointer.** It is not pasted into
> the prompt: `launch_goal_detached.ps1:166` throws above 4000 chars, so the fuel stays a
> *pointer* and the rules live here where they cost nothing per turn.
>
> ⛔ **This is the ONE copy. Do not restate a refusal rule in the fuel pointer or in a
> per-wave addendum.** Tensorbuild's `fleet-wave` lost four workers to a rule that existed
> in one preamble and not the other three; the fix that held was a single canonical file
> read unconditionally. Add rules here and every mode gets them at once.

---

## 1. A guard refusal is per-tool feedback. It is NOT a session stop.

This is the most expensive thing a headless worker gets wrong, and it is expensive
because it looks like obedience. `fak guard` prepends `[fak]` text to a refused tool
call and **the run continues**. A worker that reads a refusal and ends its turn has
converted a two-second detour into a total loss of everything it read.

**When you are refused: reconcile in place, or pick different work. Never end the run,
and never route around the guard.** The refusals you will actually meet:

| refusal | what it means | the move |
|---|---|---|
| `OFF_TRUNK` | you are not on `main` | Return to main: `fak sync apply` (or `git checkout main`). Never open a feature branch — the trunk guard refuses off-trunk commits by design |
| `COLLISION_RISK` | your write overlaps a live sibling's lease | re-read `dos arbitrate --workspace .`; take a leaf in YOUR lane. **Never `--force`** |
| `STALE_BASE_DELETION` | you are deleting a file a peer just landed | `fak sync check --fetch` / re-read the file at HEAD; you are working from a stale base |
| `MERGE_IN_PROGRESS` | a peer merge is in flight | wait and retry the commit; it clears on its own |
| `SELF_MODIFY` / `ESCALATE` | you touched a core-lock path (`policy.json`, `.mcp.json`, `.claude/**`, `dos.toml`) | that work is **not dispatchable as lane work** — Write/Edit are refused on the path too, so the lease is moot. File it as an issue and take a different leaf |
| `PUBLIC_LEAK` | your text carries an absolute path, hostname, or personal identifier | rewrite the text relative-and-anonymous, then re-propose |
| `INTERACTIVE_HANG` | a bare `git commit` (no `-m`/`-F`) opened the editor | always `git commit -s -m "<subject>" -- <paths>`, message flags **before** the `--` |
| `CONCEPT_ADMISSION` | a new Go identifier carries a concept root (`Decision`, `Gate`, `Cache`, `Plan`, `Loop`) with no ledger row | the gate reads the git **index**, not the worktree — stage the ledger row in the same commit, or rename the identifier |
| `REQUIRE_WITNESS` | the call needs a preview-confirm | re-propose the call **byte-identical** and let the kernel attach its own confirm token. Vary one character and you get a fresh token and another pause |

⛔ **`fak commit -s` is not a flag.** It dumps usage and matches no retry pattern, so a
loop that retries on "error" spins forever. Sign-off is `git commit -s`; `fak commit`
takes `--path` and `-m`.

## 2. Commit discipline on a shared trunk

- **By explicit path, always**: `fak commit --path <p> -m "<subject> (fak <leaf>)"`, or
  `git commit -s -m "<subject>" -- <paths>`. ⛔ **Never `git add -A`** — this tree has
  concurrent writers and you will commit a peer's half-finished work.
- **Never reset, rebase, amend, or force-push.** Local `main` has concurrent writers and
  HEAD moves under you. The tree is append-only; to undo, `git revert` forward.
- **The `(fak <leaf>)` trailer is load-bearing.** A bare un-stamped subject stays
  `NOT_SHIPPED` to the referee. Resolve the real leaf from your PATHS
  (`mcp__fak__fak_index_lane`), not from the issue's prose — the assigned lane in a
  ticket is a guess, and routing by tree is how work lands in the wrong leaf.
- **Lead the subject with an action verb** (`add`, `fix`, `implement`). A subject without
  one trips `FLEET_MSG_GUARD` and makes the commit audit ABSTAIN.
- **`fak commit` asserts the path SET, not the CONTENT.** A peer's edit to a file you
  named rides in silently. `git diff HEAD -- <paths>` before you retry a failed commit.

## 2.5 A child refusal is not a parent stop

For a BROAD issue, failure to admit or start one LEAF_CHILD changes the execution plan; it does not prove
the issue is blocked. The ISSUE_OWNER must record the typed refusal, reduce concurrency, and continue root implementation plus
all safe root-spine or undisputed packet work it can execute itself. `BLOCKED` is valid only when no
agent-accessible acceptance step remains.

Before any parent exits, its execution map must account for every spawned child as `VERIFIED`,
`PARKED`, or `STOPPED`, with process/session identity and artifact. A quiet, failed, or rate-limited
child is not reconciled merely because its process disappeared. Release every child lease and parent
ticket intent. If an effect exists but cannot land, park the effect and its witness gap; if no effect
exists, say `nothing` rather than manufacturing a WIP commit.

## 3. Park — the durable channel when you cannot land

You will sometimes hold findings you cannot commit: the lane is held, the path is
core-locked, or the deadline arrived. **Park them.** A `hold/*` tag needs no lease and
never touches the working tree.

```bash
git tag -a "hold/<wave>-<lane>" -m "$(cat <<'EOF'
<what this is, what you measured, what is still owed, and the exact next checkable step>
EOF
)"
```

- ⛔ **NEVER `git tag -f`.** A re-park that force-overwrites is how tensorbuild lost half a
  wave, and three of six workers on another. If the tag exists, add a suffix (`-b`, `-c`).
- ⛔ **Park only YOUR files.** `git status --porcelain <tree>` includes work that was never
  yours. Name the files from your own report; parking a peer's in-flight edits under your
  wave's name misattributes them and can resurrect a half-finished change later.
- **The tag MESSAGE is the deliverable**, not the tag. Put the whole account in it —
  measurements, rationale, and any place the spec turned out to be wrong. The orchestrator
  reads every tag message at reconcile precisely because you are gone by then.
- ⛔ **You cannot publish it.** `git push` of a raw ref is `REQUIRE_WITNESS`-gated and a
  worker has no route to origin. Parking is durable *on this box only* until the
  orchestrator pushes it (SKILL.md Phase 5). Say in your report that you parked.

## 4. The honesty boundary

- **A launch is not a ship; a self-report is not a fact.** Only a witnessed commit on the
  trunk carrying `Fixes #<N>` in the body resolves an issue. Do not `gh issue close` off
  your own say-so.
- **"Landed" ≠ "DoD met."** If the acceptance criteria are not all witnessed, report
  `not yet` with the evidence, the missing witness, and the next checkable step. Do not
  convert an open follow-on into a failure, and do not convert an unproven claim into a
  shipped one.
- **An OPEN issue is not an unstarted issue.** This repo's backlog is full of
  done-but-open tickets. Cross-check `git log --grep "#<N>"` before you build anything:
  the work may already be on the trunk under a commit that never cited it.
- **Do not widen your lane's diff** to absorb out-of-scope findings. File them.
- **Leave the tree clean.** An exiting worker that strands uncommitted work in its lane
  hands the next holder someone else's changes.
