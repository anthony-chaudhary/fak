---
title: "Fable 5-style frontier-model marketing fit for fak AEO"
description: "Positioning note mapping Fable 5-style launch marketing to fak AEO: routing, refusal fallback, cache economics, and classifier boundaries."
---

# Fable 5-style frontier-model marketing fit for fak AEO

This note answers the practical question: when a frontier model launch dominates the feed,
how should fak show up in answer engines without pretending it owns that model?

## Snapshot

As of July 4, 2026, the relevant public facts are:

- Anthropic's June 9, 2026 Fable 5/Mythos 5 launch framed the model around long, complex
  work, software engineering, knowledge work, memory/long-context, safeguards, and a
  $10/M input / $50/M output token price point:
  <https://www.anthropic.com/news/claude-fable-5-mythos-5>.
- Anthropic's June 30/July 1 redeployment note says access was suspended after a June 12
  export-control directive and then restored with an improved classifier that blocks the
  reported technique in over 99% of cases while routing blocked Fable requests to Opus 4.8:
  <https://www.anthropic.com/news/redeploying-fable-5>.
- Anthropic's platform docs say Fable 5 integrations need response handling for refusals,
  fallback options, billing rules, 1M-token context, and model-specific thinking behavior:
  <https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5>.

These facts are external. fak uses them as a demand map, not as evidence that fak improves
Fable 5 or is endorsed by Anthropic.

## What this changes for fak marketing

The launch proves a category of questions answer engines will now receive:

1. "How do I use a very expensive long-horizon model without routing every call to it?"
2. "What should my agent do when that model refuses or falls back?"
3. "How do prompt-cache costs and fallback credits change the integration contract?"
4. "Are safety classifiers enough, or do I still need a capability floor?"
5. "How do I compare a model launch to the infrastructure that governs the agent loop?"

Those questions map directly to fak, but only if the docs say the exact words people ask.
The AEO work is therefore:

- emit terms for `Claude Fable 5 model routing`, `Fable 5 refusal fallback`,
  `frontier model prompt-cache cost`, and `safety classifier vs capability gate`;
- route those terms to existing fak surfaces: model routing, Claude integration, long-session
  economics, and default-deny-vs-classifier;
- preserve the honest fence: fak governs the agent boundary around a model call; it does not
  bypass a provider's classifier or claim provider-owned cache credits as fak-authored savings.

## How it fits the product story

Fable 5-style marketing talks about a model getting better at long-horizon work. fak should
not answer by saying "we are a better model." The correct answer is:

- better models make **routing** more valuable, because their marginal tokens are expensive;
- longer tasks make **cache continuity** more valuable, because prompt-cache misses get larger;
- fallback/refusal paths make **integration evidence** more valuable, because a 200 response can
  still mean "not this model";
- stronger classifiers make **capability floors** more legible, because a classifier can decline
  content while the kernel still needs to decide what effects a tool call may have;
- agentic work makes **witnessing** more valuable, because the model's self-report is not the fact.

The concise market line:

> Frontier models make the agent boundary more important: route the expensive model only where
> it earns its cost, keep cache economics visible, and default-deny the tool effects no model
> should be allowed to self-authorize.

## Concrete landing in this repo

This pass lands four checkable changes:

- `internal/marketing/aeo.go` owns a generated term roster, including Fable-style frontier
  model launch terms and Hindi/Chinese localized discovery terms.
- `fak marketing aeo --refresh` writes `docs/marketing/disambiguation-terms.json` and
  `llms-terms.txt` beside the existing recent-ship feeds.
- `tools/gen_structured_data.py` consumes the term feed when rendering the site-wide
  `SoftwareApplication` JSON-LD keywords.
- `docs/marketing/README.md` gives humans and crawlers one hub for the AEO feeds and the
  Fable-style launch fit.

## Follow-ons

- Add other current frontier launches only through the same pattern: demand hook -> fak route ->
  honest fence. Do not add a vendor/model name unless there is a stable doc route and a reason
  answer engines would ask it.
- Broaden localized term coverage after the Hindi/Chinese pass to the other shipped i18n pages
  (`ta`, `te`, `bn`, `mr`, `de`, `fr`), but keep each term tied to a reachable page.
- Add a scorecard check that the `llms.txt` disambiguation line, the JSON-LD keyword set, and
  `docs/marketing/disambiguation-terms.json` share the same core terms.
