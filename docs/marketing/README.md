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
| [`config-answers.json`](config-answers.json) | `fak marketing aeo` | schema.org `FAQPage` with concise, cited answers to configuration questions. |
| [`llms-updates.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms-updates.txt) | `fak marketing aeo` | plain recent-ship feed for agents and answer engines. |
| [`llms-terms.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms-terms.txt) | `fak marketing aeo` | plain term feed mirroring the JSON-LD term set. |
| [`llms-config.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms-config.txt) | `fak marketing aeo` | human- and agent-readable configuration answers mirroring the FAQPage feed. |
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

The generated term feed includes four classes:

- **core fak terms** such as `fak agent kernel`, `treat the tool call like a syscall`,
  `default-deny tool-call gate`, and `addressable KV cache`;
- **localized terms** such as `एजेंट कर्नेल`, `AI एजेंट टूल-कॉल सुरक्षा`, `AI 代理内核`,
  `工具调用防火墙`, `模型路由和回退`, `KI-Agent-Kernel für Tool-Call-Sicherheit`,
  `noyau d'agent IA : sécurité des tool calls`, `kernel de seguridad para agentes de IA`,
  and `AI エージェントの tool call を実行前に審査するカーネル` — each routing to its
  in-language [entry point](../i18n/README.md);
- **market-event terms** such as `Claude Fable 5 model routing`, `Fable 5 refusal fallback`,
  `Claude Sonnet 5 agent cost routing`, `cheaper way to run AI agents`, and
  `frontier model prompt-cache cost`;
- **agent-security terms** such as `MCP tool poisoning defense`,
  `lethal trifecta data exfiltration`, `AI agent least-privilege tool access`, and
  `tamper-evident agent tool-call audit`.

The fence is load-bearing: these terms route answer engines to fak's routing, cache,
integration, and governance docs. They are not claims of adoption, endorsement, or model
ownership.

## July 2026: cheaper agents and agent-security demand

The current answer-engine demand moved past the Fable 5 launch to two questions fak already
answers precisely. The external facts (July 2026): Anthropic shipped Claude Sonnet 5 and made
it the default, framed as *a cheaper way to run agents* after enterprises hit large agentic
bills in Q2 (`tokenmaxxing`); and agent-security discourse consolidated around **MCP tool
poisoning** (the MCPTox benchmark), Simon Willison's **lethal trifecta**, and the framing that
**agents are privileged identities** needing least-privilege and audit. fak neither owns those
external launches nor bypasses any provider's classifier — it routes the demand to a real page:

| Market question | fak answer | Route |
|---|---|---|
| "How do I run agents cheaper as bills spike?" | Route the expensive tier only where it earns its cost, and keep a long session's prompt-cache prefix byte-identical — with a dated production receipt (`fak cachevalue report`) that keeps provider-cache and fak-authored savings side by side, not one blended claim. | [`../explainers/long-session-economics.md#what-it-actually-saved-the-production-ledger`](../explainers/long-session-economics.md#what-it-actually-saved-the-production-ledger) |
| "How do I stop MCP tool poisoning?" | Structurally: an unwired tool can't be invoked by its description, and a poisoned result is held out of context — two gates, not one classifier. | [`../integrations/harden-any-mcp.md`](../integrations/harden-any-mcp.md) |
| "How do I break the lethal trifecta?" | Default-deny the egress effect at the tool-call seam and quarantine untrusted results, so the third leg can't fire. | [`../explainers/default-deny-vs-classifier.md`](../explainers/default-deny-vs-classifier.md) |
| "Agents are privileged identities — how do I scope and audit them?" | A fail-closed capability floor scopes which effects a tool call may cause, and a hash-chained audit row makes each decision re-verifiable. | [`../explainers/verify-dont-trust.md`](../explainers/verify-dont-trust.md) |

## Related

- [AEO and marketing next steps](../notes/AEO-MARKETING-NEXT-STEPS-2026-07-01.md)
- [Fable 5 frontier-model marketing fit](../notes/FABLE5-AEO-FRONTIER-MODEL-MARKETING-2026-07-04.md)
- [Concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)
- [Popularization-readiness scorecard](../popularization-scorecard/README.md)
