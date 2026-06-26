---
name: skill-score
description: One repeatable pass that keeps the skill pack itself effective — the one surface no other scorecard grades. Runs the skill-effectiveness scorecard (`fak skill-effectiveness-scorecard`) over every .claude/skills/*/SKILL.md, reads the skill-debt work-list, and retires it worst-first by ADDING the real affordance — a sharp "Use when …" trigger, a reference that resolves on disk, the commit-by-path discipline a committing skill owes the shared trunk, a witness step, a scoped allowed-tools — never by spraying a keyword. Re-measures to PROVE skill-debt dropped, regenerates the committed snapshot, and commits only the skill lane by explicit path. An instance of the /score-2x loop pointed at the skill pack; the inward counterpart of agent-readiness (the agent's front door) and quality-score (the code). Use after adding or editing a skill, when a skill cites a tool that no longer exists, or on a /loop cadence to keep every skill discoverable, safe to operate, and trustworthy.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--target 2x|3x]  (no args = full measure + one improvement iteration)"
metadata:
  opencode: claude-only
---

# /skill-score — make "this skill is effective" provable, then prove it moved

> **What this does.** The pack measures fak from every angle, but until this skill
> the *skills themselves* had no measuring stick — "this skill is good" was an
> unfalsifiable claim. This is the skill pack's own checking layer. It makes
> "improve the skills" a repeatable, evidence-grounded pass: measure the skill-debt,
> retire it worst-first by adding the real affordance, then re-measure to prove the
> number dropped. It is an instance of [`/score-2x`](../score-2x/SKILL.md) pointed at
> `.claude/skills/`, and a sibling of the [`scorecard`](../scorecard/SKILL.md) doctrine.

The headline metric is **skill-debt**: the count of concrete, re-derivable affordances
a skill is missing across nine KPIs in four groups (discover / operate / trust /
economy). Driving it to zero means every skill is discoverable, safe to operate, and
trustworthy by construction. The shape is the family's: **run the scorecard → read the
skill-debt work-list → retire debt worst-first (real affordances only) → re-measure and
prove the drop → commit ONLY the skill lane by explicit path.**

---

## The measure (nine KPIs over every `.claude/skills/*/SKILL.md`)

`fak skill-effectiveness-scorecard` folds these into a composite score + the
skill-debt integer. **HARD** KPIs emit skill-debt; the two ECONOMY KPIs are **SOFT**
(they lower the score but never gate — the cheap fix for either is a keyword, which is
gaming).

| KPI | HARD/SOFT | one unit of skill-debt is… |
|---|---|---|
| `description_present` | HARD | a skill with no real `description` front-matter (the trigger surface) |
| `trigger_clause` | HARD | a description that says WHAT but not WHEN — no "Use when / Use to / Use after …" |
| `name_resolves` | HARD | front-matter `name:` ≠ the directory, so `/name` won't invoke it |
| `refs_resolve` | HARD | a cited `tools/*.py` / `docs/*` / sibling `SKILL.md` that does not exist on disk |
| `tools_scoped` | HARD | a skill that COMMITS but declares no `allowed-tools` (least privilege) |
| `commit_discipline` | HARD | a committing skill that never names commit-by-path / `-- <paths>` / never `git add -A` |
| `proof_step` | HARD | a committing skill with no verify / witness / re-measure step |
| `anti_gaming` | **SOFT** | a metric-driving skill with no anti-gaming / honesty clause |
| `context_budget` | **SOFT** | a SKILL.md over ~300 lines (the /clean-skill smell) |

`refs_resolve` is the **ungameable anchor** — a cited helper either exists on disk or
it doesn't, cross-checked against the real tree, so you can't drop it by editing text.
The commit-discipline cluster (`commit_discipline`, `proof_step`, `tools_scoped`) gates
ONLY the skills that commit to the shared trunk — the highest-privilege ones.

## Step 1 — Run the scorecard (it builds your work-list)

```bash
go run ./cmd/fak skill-effectiveness-scorecard            # human scorecard (per-KPI + work-list)
go run ./cmd/fak skill-effectiveness-scorecard --json     # machine payload
go run ./cmd/fak skill-effectiveness-scorecard --json > /tmp/skill-base.json   # record the baseline
```

It exits non-zero whenever skill-debt > 0. Read `corpus.score`, `corpus.skill_debt`,
and `corpus.breakdown` (per-KPI debt, worst first). **Record `skill_debt = N`** before
touching anything — you need the before-number to prove the delta.

## Step 2 — Retire skill-debt worst-first, by ADDING the real affordance

Attack the heaviest KPI first (`breakdown[0]`). Each fix is a genuine improvement to the
skill, not a keyword:

- **`trigger_clause`** — add a real "Use when / Use after …" clause to the description
  that says when the model should fire it. Write the true trigger, not a filler.
- **`refs_resolve`** — the skill cites a tool/doc that moved or was deleted. Fix the
  path to the real file, or remove the dead citation. (Never invent a file to satisfy
  the check — that's the gaming this refuses.)
- **`commit_discipline`** — add the commit-by-path / `-- <paths>` / never-`git add -A`
  rule to a committing skill so it can't sweep a peer's staged files on the shared trunk.
- **`tools_scoped`** — add `allowed-tools:` to a committing skill, declaring its real
  tool surface (least privilege).
- **`proof_step`** — add the verify / witness / re-measure step the skill was missing,
  so "it shipped" rests on evidence.
- **`name_resolves` / `description_present`** — rename to match the dir, or write a real
  description. Structural; ungameable.

Editing a PEER's skill on this shared trunk is fine (additive frontmatter / a clause),
but commit by explicit path so you don't sweep their other in-flight edits (Step 4).

## Step 3 — Re-measure and PROVE the drop

```bash
go run ./cmd/fak skill-effectiveness-scorecard --json
```

Compare the new JSON against the baseline you recorded and state the delta plainly:
`skill-debt N → M (−k), score S → S'`.
Regenerate the committed snapshot so the doc matches the tree (Bash `>` for UTF-8):

```bash
go run ./cmd/fak skill-effectiveness-scorecard --markdown > docs/SKILL-EFFECTIVENESS-SCORECARD.md
```

If a surface SATURATES (skill-debt 0, grade A), harden the bar
per [`/score-2x`](../score-2x/SKILL.md) Step 4 — tighten `DESC_MIN_CHARS`, lower
`CONTEXT_SOFT_MAX`, or promote a SOFT KPI to HARD — then re-pin the control pane.

## Step 4 — Commit ONLY the skill lane, by explicit path

```bash
git pull --no-rebase --no-edit
git add <the SKILL.md files you fixed> cmd/fak/skill_effectiveness.go docs/SKILL-EFFECTIVENESS-SCORECARD.md
git commit -s -m "<subject>" -m "<body: N→M skill-debt, what changed>" -m "(fak <leaf>)"
dos commit-audit HEAD                                    # MUST print [diff-witnessed] / verdict OK
git push
```

- **Commit by explicit path**, never `git add -A` on this shared tree (the exact rule
  the `commit_discipline` KPI enforces — eat the dogfood). Stay on `main`; if a peer's
  `MERGE_HEAD` is set, wait for it to clear, then re-try the pathspec commit.
- End the subject with a `(fak <leaf>)` trailer; `docs(skills):` for a skill-text pass.
- `dos commit-audit HEAD` printing `[diff-witnessed]` is the green light.

---

## The anti-gaming law (the measure is only as honest as the pass)

**Retire a defect by changing the skill, never by gaming the detector.** A missing
trigger is fixed by writing the real trigger, not by pasting "Use when" onto a vague
description; a dead `refs_resolve` is fixed by correcting the path to the real file, NOT
by creating an empty file to satisfy the check; a missing `proof_step` is fixed by
adding a real witness step, not the word "verify". If "fixing" a defect would mean
faking the affordance, stop — that's not a real gap, and weakening the check to make it
green turns the scorecard into theater. `anti_gaming` and `context_budget` are SOFT for
exactly this reason: their cheap fix is cosmetic, so they score but never gate.

## When to run this

- After **adding or editing a skill** (a new SKILL.md lands with debt — retire it).
- When a skill **cites a tool/doc that moved or was deleted** (`refs_resolve` catches it).
- To drive the **skill-2× program** — one halving of skill-debt per focused pass.
- On a **/loop cadence** to keep the pack discoverable, safe, and trustworthy as it grows.

The scorecard is read-only; this skill's only writes are your genuine skill fixes,
`docs/SKILL-EFFECTIVENESS-SCORECARD.md`, and the Go subcommand itself.
