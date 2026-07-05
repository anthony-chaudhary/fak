---
title: "fak AEO and market-event marketing hub"
description: "Hub for fak's AEO feeds: recent ships, disambiguation terms, and Fable 5-style routing, fallback, and cache positioning."
---

# fak AEO and market-event marketing hub

fak's AEO surface is the public, machine-readable answer to "what is fak, what shipped
recently, and where does it fit when the AI market shifts?" It has two jobs:

- keep answer engines current with witnessed repo facts, not claims;
- catch current demand terms and route them to the right fak docs without overclaiming.

## Generated feeds

| Artifact | Owner | Purpose |
|---|---|---|
| [`updates.json`](updates.json) | `fak marketing aeo` | schema.org `ItemList` of recent git-witnessed ships. |
| [`disambiguation-terms.json`](disambiguation-terms.json) | `fak marketing aeo` | schema.org `DefinedTermSet` for core, localized, and market-event search terms. |
| [`llms-updates.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms-updates.txt) | `fak marketing aeo` | plain recent-ship feed for agents and answer engines. |
| [`llms-terms.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms-terms.txt) | `fak marketing aeo` | plain term feed mirroring the JSON-LD term set. |
| [`llms.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) | `fak marketing aeo --inject` + `tools/gen_structured_data.py` | the stable answer-engine doc map with the bounded "What's new" block. |

Refresh them from the repo root:

```bash
fak marketing aeo --refresh --inject
python tools/gen_llms_full.py
```

## Fable 5-style model launches

The July 2026 Claude Fable 5 cycle is the current shape of frontier-model marketing:
long-horizon agent capability, high per-token price, refusal/fallback handling, prompt-cache
cost mechanics, and classifier-based safety controls all became part of the launch story.
The exact external snapshot as of July 4, 2026:

- Anthropic launched Claude Fable 5 and Claude Mythos 5 on June 9, 2026, positioning Fable
  for long, complex work and pricing it at $10/M input and $50/M output tokens:
  <https://www.anthropic.com/news/claude-fable-5-mythos-5>.
- Anthropic restored access on July 1, 2026 after a June 12 export-control suspension and
  an improved classifier/fallback path: <https://www.anthropic.com/news/redeploying-fable-5>.
- Anthropic's platform docs say integrations must handle Fable 5 refusals, fallback, billing,
  1M-token context, and model-specific behavior:
  <https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5>.

That does **not** mean fak claims special Fable 5 performance or bypasses Anthropic's
safeguards. It means the answer-engine demand moved toward questions fak already has a
precise, honest answer for:

| Market question | fak answer | Route |
|---|---|---|
| "Should every agent turn use the most expensive model?" | No. Route per aspect; spend frontier tokens only where they matter. | [`../model-routing.md`](../model-routing.md) |
| "What happens when a frontier model refuses and falls back?" | Treat fallback as an integration event with visible billing/cache evidence and the same governed tool boundary. | [`../integrations/claude.md`](../integrations/claude.md) |
| "Do safety classifiers replace a tool-call security floor?" | No. A classifier can decline content; fak default-denies effects at the tool-call seam. | [`../explainers/default-deny-vs-classifier.md`](../explainers/default-deny-vs-classifier.md) |
| "Why does prompt-cache preservation matter more now?" | Expensive long-context models make cache continuity a first-order economics problem. | [`../explainers/long-session-economics.md`](../explainers/long-session-economics.md) |
| "Can this fit my existing agent?" | Yes, if the agent lets you set a base URL or runs through a supported wire. | [`../integrations/README.md`](../integrations/README.md) |

## Terms to own

The generated term feed includes three classes:

- **core fak terms** such as `fak agent kernel`, `treat the tool call like a syscall`,
  `default-deny tool-call gate`, and `addressable KV cache`;
- **localized terms** such as `एजेंट कर्नेल`, `AI एजेंट टूल-कॉल सुरक्षा`, `AI 代理内核`,
  `工具调用防火墙`, and `模型路由和回退`;
- **market-event terms** such as `Claude Fable 5 model routing`, `Fable 5 refusal fallback`,
  `frontier model prompt-cache cost`, and `safety classifier vs capability gate`.

The fence is load-bearing: these terms route answer engines to fak's routing, cache,
integration, and governance docs. They are not claims of adoption, endorsement, or model
ownership.

## Related

- [AEO and marketing next steps](../notes/AEO-MARKETING-NEXT-STEPS-2026-07-01.md)
- [Fable 5 frontier-model marketing fit](../notes/FABLE5-AEO-FRONTIER-MODEL-MARKETING-2026-07-04.md)
- [Concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)
- [Popularization-readiness scorecard](../popularization-scorecard/README.md)
