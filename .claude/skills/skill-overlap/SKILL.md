---
name: skill-overlap
description: Flag content-redundant SKILL.md pairs as merge candidates — pairwise body cosine similarity over the skill pack, sorted highest-first. Use when the pack feels bloated or two skills seem to overlap, to find "skills A and B are N% token-similar — consider merging" candidates before a skill-lifecycle archive pass. Proposal only — never merges, edits, or deletes a skill.
---

# /skill-overlap — flag content-redundant SKILL.md pairs (never merge)

The one context-cost technique from the Comet/Perplexity writeup that fak did
*not* already cover (#3930): **skill-to-skill content redundancy**.
[`skill-lifecycle`](../skill-lifecycle/SKILL.md) archives *dead* skills and
[`skill-score`](../skill-score/SKILL.md) audits over-budget ones — neither measures
whether two *live* skills say the same thing. With 55+ `SKILL.md` files (many
`*-score` scorecards), that overlap is where the pack quietly bloats.

This skill measures it and stops. **Max action is to PRINT candidates** for a
human to judge — like `skill-lifecycle`'s dry-run archive verdicts, it proposes,
it never merges. Deciding two skills are genuinely one is a semantic call the
detector deliberately does not make; there is no `--apply`.

## The witnessed helper

`skill_overlap.py` runs a pairwise token-similarity primitive over every
`.claude/skills/*/SKILL.md` **body** and emits the pairs above a threshold,
sorted by similarity:

```
python .claude/skills/skill-overlap/skill_overlap.py [--threshold T] [--top N] [--json]
```

- **Front-matter is stripped first** — the shared `name:`/`description:` YAML
  skeleton is structural, not redundant, so it must not inflate the score.
- **Similarity is body cosine over token frequency** with a small function-word
  stoplist — "skills A and B are N% token-similar" in the issue's own words. Same
  posture as `fak dup` / `fak traj` (block- and query-level near-dup detection),
  self-contained so the witness runs with no built `fak` binary.
- **Default threshold 0.5**; drop to `--threshold 0.4` to see the long tail.
- **`shared_top`** names the highest-weight tokens both bodies carry — the *why*
  behind each score, so a human can judge the merge without re-reading both.

Exit code is always 0: this is advisory, and "no candidates" is a clean answer.

## How to read a candidate

A high score means *lexical* overlap, not that a merge is correct. Two `*-score`
scorecards may share vocabulary (`debt`, `scorecard`, `commit`) yet tend
genuinely different surfaces. Before proposing a merge:

1. Read both `SKILL.md` bodies. Do they tend the *same* surface, or just share
   the scorecard idiom?
2. If genuinely redundant → open an issue proposing the merge, or route the
   weaker one through [`skill-lifecycle`](../skill-lifecycle/SKILL.md) to archive
   (reversible), never a raw delete.
3. If distinct-but-similar → the overlap is a *discoverability* smell; sharpen
   each `description:`'s "Use when…" trigger via [`skill-score`](../skill-score/SKILL.md)
   so the two stop colliding in recall.

## Witness

`python .claude/skills/skill-overlap/skill_overlap_test.py` — a fixture skill
pack with one deliberately-overlapping pair asserts that pair is surfaced above
the threshold while two unrelated skills are not, plus the front-matter-stripped,
proposal-only, deterministic, and archived-skip invariants. This is the skill's
proof-of-correctness; run it after any change to the detector.
