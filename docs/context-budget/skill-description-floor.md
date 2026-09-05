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
skill footprint [interactive]: 71 skill(s); resident floor = 33982 bytes (~8495 tokens);
  description floor = 33982 B; name-only floor = 956 B; at-rest card floor = 16466 bytes
  at-rest intent slice (#5560): across 71 skill(s)
```

Heaviest resident descriptions — the trim targets:

| rank | bytes | skill |
|-----:|------:|-------|
| 1 | 731 | debt-orchestrator |
| 2 | 686 | field-borrow |
| 3 | 680 | refresh-readme |
| 4 | 657 | study-repo |
| 5 | 652 | refresh-cachedoc-numbers |
| 6 | 648 | milestone-score |
| 7 | 647 | modularize |
| 8 | 645 | issue-orchestrator |

The full 71-skill breakdown is what `fak skill footprint --top 0` prints; only the
head is pinned here so a drift is legible in review.

`name-only floor = 956 B` is the size of the headroom: **33.0 kB of the 34.0 kB
resident floor is description prose**, and every skill stays invocable by name
without a single byte of it.

**Last re-pin: 33415 → 33982 B (+567, 71 skills)** — the live `SkillResolver`
measurement records the current catalog footprint across 71 skills after adding
`issue-queue` and refreshing issue management skills (#11541). The
gate banks that measured baseline; future growth retains the same narrow ratchet
band.

## Provenance (Law A2 — every value carries its provenance)

This floor is denominated in **bytes of frontmatter `description` text**, as parsed
by `internal/capindex`'s `SkillResolver`. Two things follow, and neither may be
quietly dropped when the number is quoted:

- **The `~7931 tokens` figure is an ESTIMATE**, at the house ~4 bytes/token divisor
  (`skillfootprint.BytesPerTokenEstimate`, the same walk as
  `EstimateAnthropicTokens`). It is not a provider-billed count and must never be
  compared against one.
- **The byte floor is fak's MODEL of the resident index**, the `interactive` profile
  #3234 defined — not a witnessed on-the-wire measurement. Harness skill listings
  have been observed rendering project skills **name-only** while built-in skills
  carry their full descriptions, which would put the on-the-wire project-skill cost
  nearer the 894 B name floor than the 31.7 kB description floor. Confirming what a
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
committed ceiling, `SkillDescriptionBudgetBytes` (currently **33982**), as a one-way
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
- For the card split (#5560), in `internal/capindex`:
  `TestResidentCardCarriesIntentNotFullDescription` proves the serialized card holds
  the intent line and no `description` field; `TestFaultStillPagesTheWholeSkillBody`
  proves the elided prose is byte-for-byte recoverable from the faulted body;
  `TestShrinkingTheRankingKeyCostsSelectionQuality` is the A/B above and FAILS if a
  shrunken ranking key ever stops costing selection quality, so the trade is re-decided
  with evidence rather than silently; `TestExplicitFrontmatterIntentWins` pins the
  `intent:` override end to end.

## The card split (#5560) — and why it does NOT move this number

`capindex.CapCard` now separates the two copies of the prose that used to be one
field (`skill_resolver.go`):

| field | who reads it | cost of a byte |
|---|---|---|
| `Intent` | serialized into `CardBytes`; what a listing renders | a **resident token**, paid every session |
| `Trigger` | `Catalog.scoreCard`, `contextq`, `selfquery`'s ranker | **recall** — it never leaves the process |

`Intent` is the leading sentence of the `description`, capped at
`capindex.SkillIntentMaxBytes` (320 B), overridable per skill with a frontmatter
`intent:` line. The full prose is still reachable two ways: as the ranking key, and
in the SKILL.md body that `Fault` pages in on selection. It is a residency split,
not a deletion.

**Measured effect:** the at-rest card floor falls **51,409 B → 14,196 B (−72%)**.

**The ranking key was measured, not assumed.** #5560 asked whether `Trigger` may
shrink too. `capindex.TestShrinkingTheRankingKeyCostsSelectionQuality` builds 56
probes from the corpus itself — rare terms that appear only in the *tail* of a
skill's own description — and ranks them through the shipped `Catalog.RankCards`:

| ranking key | top-1 | probes matching nothing at all |
|---|---:|---:|
| full description (shipped) | 54/56 (96%) | 0 |
| leading sentence only | 4/56 (7%) | 27 |

So `Trigger` keeps the full description. Selection quality, not byte count, decides
that field. `TestEverySkillStaysInvocableByName` pins the other half: name
addressability survives even with *no* trigger at all, because `scoreCard` weights a
name match above a trigger match.

**This is why #5560 left `SkillDescriptionBudgetBytes` unchanged at 47,236.** The gated
`DescFloor` is the sum of the frontmatter `description` fields — the prose a harness
renders into the always-on skill listing, and the thing #5444 exists to refuse the
growth of. The card split moved where fak *serializes* that prose; it did not delete a
byte of frontmatter. Re-pinning the ceiling down onto the derived intent slice would
bank a win that was never won, and would blind the ratchet to a description that
doubles in its tail — the exact growth it was built to catch.

## What is still NOT done

- **The userland description migration** — #3234's undelivered item 3, and the only
  lever that moves the description floor: shortening the frontmatter `description`
  fields themselves. `score-2x` carries the **worked example** of the migration — one
  `intent:` line added, its `description` untouched, so the gated floor is byte-identical
  and only the at-rest card shrank. The 5 skills whose leading sentence still overruns
  the 320 B cap and therefore elides (`curate-cluster`, `issue-triage`, `sota-check`,
  `trajectory-audit`, `wave-harvest`) are named by
  `capindex.TestResidentIntentInventory` and are the next adopters. Sweeping all 58 is
  deliberately not done here.
- **A witnessed on-the-wire count** — see the provenance caveat above. Which slice a
  given harness build actually ships is still unconfirmed, and the card split does not
  change that.

## Cross-links

- **#3229** — epic: shrink the always-sent context budget.
- **#3234** — the measurement this ratchet defends (`fak skill footprint`).
- **#3612** — the `headless` name-only profile (the 894 B floor above).
- **#5560** — the `capindex` residency split above (`Intent` vs `Trigger`).
- [MCP tool-schema floor](mcp-tool-floor.md) — the systemic sibling, with the
  `FLOOR_BUDGET_*` / `DESC_BUDGET_*` ratchets this one is modelled on.
