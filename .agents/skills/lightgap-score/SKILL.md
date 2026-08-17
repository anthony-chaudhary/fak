---
name: lightgap-score
description: One repeatable pass that answers "is fak actually worth adopting, for whom, and what does it cost you to find out?" on an UNBOUNDED scale anchored at two ends — the next-best option a given buyer would really use, and the best that is physically possible. Runs the Go-backed `fak score lightgap` over a modular data directory (tools/lightgap_scorecard.data/): 8 facets x 7 buyer segments, each cell scored w_net = artanh(beta) - artanh(load), where beta is the fraction of the alternative-to-ceiling gap fak closes and load is the share of that buyer's patience its adoption consumes. Being worse than the alternative is a NEGATIVE score, not a low one; approaching the ceiling diverges. There is deliberately no overall number — a segment can come back ADOPT while another comes back BLOCKED or UNDECIDABLE. Drives one number, lightgap_debt (material comparisons nobody has RUN), which retires only by running an experiment. Regenerates docs/lightgap-scorecard/. Use before positioning work, when a head-to-head lands, when a competitor's ceiling moves, or when someone asks "should WE use this?"
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/lightgap-score/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/lightgap-score/SKILL.md`](../../../.claude/skills/lightgap-score/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.
