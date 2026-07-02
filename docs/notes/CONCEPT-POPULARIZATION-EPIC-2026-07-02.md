---
title: "Concept-popularization epic: make the fak/DOS ideas broadly known and attractive"
description: "The decomposition for making fak's core concepts — treat the tool call like a syscall, verify-don't-trust (DOS), the addressable bit-exact KV cache, the default-deny capability gate with prompt-injection quarantine, and the single-binary drop-in — genuinely popular: 50 self-contained tickets across explainer content, visual assets, interactive demos, positioning, social proof, developer onramp, integration recipes, benchmark-as-story, memorable framing, distribution, and an adoption-measurement scorecard."
---

# Concept-popularization epic: make the ideas known and attractive

_Filed 2026-07-02. The existing marketing surface
([`AEO-MARKETING-NEXT-STEPS`](AEO-MARKETING-NEXT-STEPS-2026-07-01.md),
`internal/marketing/aeo.go`) is **mechanism** — answer-engine feeds, schema.org, SEO
front-matter. This epic is the **human-facing** half: the artifacts, stories, demos, and
moves that make a person *want* the concepts and *remember* them. The two are
complements — AEO makes fak findable by machines; this makes it compelling to people._

## The five concepts we are popularizing

Every ticket below serves at least one. When authoring, name which.

1. **Treat the tool call like a syscall** — the model proposes, the kernel disposes. The
   one-line mental model that makes the whole design click.
2. **Verify, don't trust (DOS)** — a false "done" is refused from git evidence, not the
   agent's word; refusals carry a structured reason from a closed vocabulary; recalled
   memory is re-verified at read time. The trust substrate under a fleet of agents.
3. **The addressable, bit-exact KV cache** — reach into the middle of a kept run, evict one
   span, leave the cache bit-for-bit identical (`max|Δ| = 0`); long sessions stop getting
   expensive because the cached prefix survives.
4. **Default-deny capability gate + prompt-injection quarantine** — refusing an irreversible
   action does not depend on *catching* an attack; the lever was never wired. Suspicious
   tool *results* are held out of context by structure, not by a classifier.
5. **One static Go binary, drop-in in front of any agent** — repoint one base URL; the same
   artifact a dev runs on a laptop is what a platform team hardens for a fleet.

## The eleven dimensions (each ticket lands in exactly one)

| Dim | Name | The question it answers |
|---|---|---|
| A | Explainer content | "I read one page and now I *get* it." |
| B | Visual & diagram assets | "One picture and I can re-explain it to a colleague." |
| C | Interactive & runnable demos | "I ran it in 60 seconds and saw the verdict." |
| D | Positioning & comparison | "How is this different from the thing I already use?" |
| E | Social proof & community | "Other people I respect use and vouch for this." |
| F | Developer experience & onramp | "Nothing stood between me and my first win." |
| G | Integration recipes | "It dropped into the exact stack I already run." |
| H | Benchmark-as-story | "The number is real, sourced, and I feel its stakes." |
| I | Memorable framing & naming | "I can repeat the pitch from memory a week later." |
| J | Distribution & channels | "It showed up where I already spend attention." |
| K | Adoption measurement | "We can tell whether any of this is actually working." |

## Rules for every ticket (so each is a self-contained unit of work)

- **One deliverable, one lane.** A ticket a single headless worker can finish and ship
  without touching another ticket's files. Prefer new files in disjoint trees
  (`docs/explainers/*`, `docs/adoption/*`, `examples/*`, `tools/*_scorecard.py`).
- **Honest-scope fence.** No ticket may claim market adoption, a benchmark we have not run,
  or a novelty the 0/29 prior-art audit refutes. Frame the *assembly*, quote *witnessed*
  numbers (the tuned ~4.1×, not the naive 60×), and label simulated as simulated.
- **Acceptance is checkable.** Each ticket names the artifact that proves it done — a file
  that exists, a command that exits 0, a scorecard number that moved.
- **Ship discipline.** Trunk only, commit by explicit path, Conventional-Commits subject
  with a `(fak <leaf>)` stamp. Docs need an `INDEX.md` line; new `docs/*.md` need SEO
  front-matter (`title:`/`description:`) so they don't red the SEO scorecard.

## The 50 tickets

The full enumerated list with per-ticket bodies is filed as GitHub issues under the
`popularization` label. This doc is the durable framing; the issues are the units of work.
Dimension coverage target: A×8, B×5, C×6, D×5, E×5, F×5, G×4, H×4, I×3, J×3, K×2.

## Verify

```
gh issue list --label popularization --state open        # the 50 units of work
python tools/seo_aeo_scorecard.py                        # new docs don't red SEO
```
