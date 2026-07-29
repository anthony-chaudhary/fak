---
title: "Resident skill-description floor - committed baseline"
description: "The committed baseline and one-way ratchet (#5444) for the resident .claude/skills description floor, part of epic #3229's work to shrink the always-sent token budget."
---

# Resident skill-description floor — committed baseline (#5444)

Part of epic **#3229** (shrink the always-sent context budget). This is the
userland sibling of [MCP tool-schema floor](mcp-tool-floor.md): that page pins the
floor fak's **MCP server** advertises, this one pins the floor fak's
**`.claude/skills` catalog** carries.

## What this number is

Every skill under `.claude/skills/*/SKILL.md` declares a frontmatter
`description`. The skills index holds every one of those descriptions at rest, so
the resident tax of the catalog is the **sum of the description fields**, and it
grows linearly with skill count — one skill at a time, each addition invisible to
its own author.

`fak skill footprint` prices that floor offline and deterministically from the same
shipped `capindex.SkillResolver` cards the `fak skill` verbs read, so the number can
never drift from what the catalog actually holds. The fold lives in
`internal/skillfootprint` (`Fold` / `Measure`) alongside the gate, so the scorecard
and the ratchet are one measurement rather than two estimators.

Regenerate at any time:

```
fak skill footprint                 # human table, heaviest-first
fak skill footprint --json          # schema fak-skill-footprint/1
fak skill footprint --top 8         # just the heaviest N
go test ./internal/skillfootprint   # the enforcing test; -v logs the same figures
```

## Baseline (measured)

```
skill footprint [interactive]: 58 skill(s); resident floor = 47236 bytes (~11809 tokens);
  description floor = 47236 B; name-only floor = 787 B; at-rest card floor = 51409 bytes
```

Heaviest resident descriptions — the trim targets:

| rank | bytes | skill |
|-----:|------:|-------|
| 1 | 1877 | study-repo |
| 2 | 1599 | resume-watchdog-audit |
| 3 | 1471 | scout-loop |
| 4 | 1359 | super-loop |
| 5 | 1332 | field-borrow |
| 6 | 1270 | trajectory-control |
| 7 | 1231 | disambiguation-score |
| 8 | 1223 | stability-score |

The full 58-skill breakdown is what `fak skill footprint --top 0` prints; only the
head is pinned here so a drift is legible in review.

`name-only floor = 787 B` is the size of the headroom: **46.4 kB of the 47.2 kB
resident floor is description prose**, and every skill stays invocable by name
without a single byte of it.

## Provenance (Law A2 — every value carries its provenance)

This floor is denominated in **bytes of frontmatter `description` text**, as parsed
by `internal/capindex`'s `SkillResolver`. Two things follow, and neither may be
quietly dropped when the number is quoted:

- **The `~11809 tokens` figure is an ESTIMATE**, at the house ~4 bytes/token divisor
  (`skillfootprint.BytesPerTokenEstimate`, the same walk as
  `EstimateAnthropicTokens`). It is not a provider-billed count and must never be
  compared against one.
- **The byte floor is fak's MODEL of the resident index**, the `interactive` profile
  #3234 defined — not a witnessed on-the-wire measurement. Harness skill listings
  have been observed rendering project skills **name-only** while built-in skills
  carry their full descriptions, which would put the on-the-wire project-skill cost
  nearer the 787 B name floor than the 47.2 kB description floor. Confirming what a
  given harness build actually ships is open follow-on work; the ratchet is worth
  holding either way, because it is fak's own committed scorecard and because
  pinning today's floor is what turns a later trim into a bankable win instead of
  headroom for the next skill.

## The gate (#5444)

Measuring the floor does not keep it lean. #3234 shipped the measurement and closed;
in the twenty days that followed, the measured floor grew from 36,237 B to 47,236 B
(**+30.4%**) with nothing opposing it. A number that cannot refuse a change is taste,
and taste lost 30% in three weeks.

`internal/skillfootprint.CheckDescriptions` gates the measured floor against a
committed ceiling, `SkillDescriptionBudgetBytes` (currently **47236**), as a one-way
ratchet:

| Direction | Reason | What it means |
|---|---|---|
| measured **>** budget | `SKILL_DESC_BUDGET_EXCEEDED` | a new skill, or a fattened `description`, grew the resident tax |
| measured **<** budget − 2000 | `SKILL_DESC_BUDGET_STALE` | a trim won headroom that was never banked into the constant |

Both tokens are registered in `dos.toml [reasons]`, so `dos_check_reason
SKILL_DESC_BUDGET_EXCEEDED` resolves them as known, refusable gates rather than
UNCLASSIFIED free-text drift.

**How to justify growth.** Raise `SkillDescriptionBudgetBytes` in the *same commit*
as the skill description that grew it, and re-pin the baseline block above. That is
the whole mechanism: the new resident tax becomes a diff line a reviewer sees, bound
to its cause, instead of being discovered a month later as a 30% regression. Prefer
trimming the description first — the skill is still invocable by name, so the
resident prose only has to say **when** to load it, not what it does.

The 2000-byte slack is roughly one heavy skill description (a shade over 4% of the
floor). It absorbs incidental churn — a reworded trigger, a retitled skill — while
still forcing a real reduction to be banked, the same discipline
`internal/pythongate` applies to the `tools/*.py` baseline: the ratchet only ever
tightens. The gate fails **closed**: an unreadable or empty skills tree folds to 0
bytes and refuses as `SKILL_DESC_BUDGET_STALE` rather than greening on a measurement
of nothing.

## Witness

- `internal/skillfootprint.TestMeasureReadsTheRealSkillsTree` prices the **real**
  `.claude/skills` tree and asserts the floor is a faithful partition of the per-skill
  rows and non-trivial — the numbers above are reproducible, not hand-typed.
- `internal/skillfootprint.TestSkillDescriptionBudgetPassesAtHEAD` is the enforcing
  test: the tree as shipped must pass its own committed budget.
- `TestSkillDescriptionBudgetRefusesGrowth` witnesses `SKILL_DESC_BUDGET_EXCEEDED`
  firing on a catalog grown by one fat description;
  `TestSkillDescriptionBudgetDemandsBankedWin` witnesses `SKILL_DESC_BUDGET_STALE`
  refusing an unbanked trim; `TestSkillDescriptionBudgetBandBoundaries` pins the exact
  admit/refuse edges; `TestSkillDescriptionBudgetFailsClosed` pins the fail-closed
  posture.
- `TestCommittedSkillBudgetMatchesMeasuredFloor` proves the ceiling above is a
  measurement, not a hand-typed number that drifted from the tree.
- `TestSkillDescBudgetReasonsRegisteredInDosToml` proves both tokens are declared in
  `dos.toml [reasons]` as refusals, and `TestBaselineDocPinsTheCommittedCeiling`
  proves this page carries the same number the constant does.

## What this issue deliberately does NOT do

- **The `capindex` card split** — `skill_resolver.go` still copies the whole
  `description` into both the at-rest `CardBytes` and the ranking `Trigger`, so the
  "name + one-line intent" index #3234 promised does not exist yet. That is #3234's
  undelivered item 2.
- **The userland description migration** — trimming the heaviest descriptions above.
  That is #3234's undelivered item 3, and this ratchet is what makes it a *bankable*
  win: a trim that is not banked into the constant now reds as
  `SKILL_DESC_BUDGET_STALE`.

Both deserve their own issue. This one only stops the bleeding.

## Cross-links

- **#3229** — epic: shrink the always-sent context budget.
- **#3234** — the measurement this ratchet defends (`fak skill footprint`).
- **#3612** — the `headless` name-only profile (the 787 B floor above).
- [MCP tool-schema floor](mcp-tool-floor.md) — the systemic sibling, with the
  `FLOOR_BUDGET_*` / `DESC_BUDGET_*` ratchets this one is modelled on.
