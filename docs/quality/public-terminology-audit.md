---
title: "Public terminology audit"
description: "The one stable public term fak uses for each concept across README, START-HERE, and integration entry points, with replacement rules and a completion check."
---

# Public terminology audit

> **Audience:** documentation writers and reviewers aligning reader-facing entry points.
> **Lifecycle:** current. **Generation:** `gen/now`. **Support:** maintained as the current public terminology audit.
> **Authority:** this audit owns preferred public labels and replacement rules; the linked product, command, support, claim, and benchmark routes own their facts.
> **Next action:** match each public concept to the table, replace a conflicting label with the preferred term, and run the [completion check](#completion-check).

Use one stable public term for each concept across `README.md`, `START-HERE.md`, integration entry points, and other reader-facing routes. Preserve exact command names, protocol fields, quotations, and historical evidence where fidelity requires them; do not turn those exceptions into competing public labels.

## Default choice

When prose names a concept in this table, use its **preferred public term**. Introduce a narrower technical term only after the public term when the distinction changes the reader's action. If a concept is absent, do not select a label by majority usage: add an authority-backed row before normalizing consumers.

## Preferred public vocabulary

| Concept | Preferred public term | Use it for | Keep these scoped exceptions | Replace or clarify | Authority |
|---|---|---|---|---|---|
| The product | **fak** | The product and CLI as a whole. On first use where orientation matters: **fak, the agent kernel**. | `` `fak` `` for the executable or command prefix. | Do not introduce “FAK platform,” “agent firewall,” or “proxy” as a second product name. | [README definition](../../README.md#fak--the-fused-agent-kernel) |
| Product category | **agent kernel** | The seam that handles an agent's tool calls before execution. | **kernel** after the page has established “agent kernel”; subsystem names in contributor material. | Clarify bare “gateway,” “monitor,” or “middleware” when it refers to the whole product. | [README definition](../../README.md#fak--the-fused-agent-kernel) |
| Agent under kernel control | **managed agent** | An agent whose tool-call path is run through fak. | Framework-specific agent names when selecting an integration. | Replace unexplained “wrapped,” “protected,” or “fak agent” when the intended state is managed. | [managed-agent choices](../../README.md#one-managed-agent-two-ways-to-run-the-kernel) |
| One requested operation | **tool call** | A tool invocation before or after kernel handling. | `tool_call`, protocol field names, exact code identifiers, and quoted source text. | Normalize prose “tool-call,” “function call,” or “action request” unless a protocol distinction matters. | [agent-kernel definition](../../README.md#fak--the-fused-agent-kernel) |
| Local structural decision before execution | **preflight** | The `fak preflight` command and its local allow/deny result. | `` `fak preflight` `` for the exact command; “pre-flight ladder” only when citing the named internal subsystem or ledger heading. | Do not use “pre-check,” “policy test,” and “pre-flight” as interchangeable public names. | [canonical offline proof](../repro-packet.md#reproduce-the-offline-allowdeny-boundary) |
| Policy input | **policy manifest** | The JSON document supplied to policy validation, preflight, or serve paths. | Exact filenames such as `customer-support-readonly-policy.json`; **policy** after the artifact is established. | Clarify “policy file,” “rules file,” or “config” when the JSON manifest is meant. | [policy fixture and validation](../repro-packet.md#witness-1-policy-manifest-validates) |
| Kernel decision process | **adjudication** | The process that produces a decision for a tool call. | **adjudicator** for the component; exact package, type, metric, or command names. | Do not use “moderation,” “filtering,” or “scanning” as synonyms for the decision process. | [adjudication claim](../../CLAIMS.md#adjudication-the-in-process-dos-reference-monitor) |
| Decision outcomes | **ALLOW** and **DENY** | The two public decision outcomes; pair `DENY` with its reason token when available. | Exact JSON field values and reason tokens such as `POLICY_BLOCK`. | Use “blocked” or “refused” only as reader-facing consequences, not as a third decision outcome. | [offline allow/deny witness](../repro-packet.md#result-and-scope) |
| Ways to run fak | **local guard** and **managed endpoint** | The two reader choices: `fak manage` manages one local agent; `fak serve` exposes the managed endpoint. | Exact commands and protocol names; **gateway** for the compatibility surface after the endpoint choice is clear. | Avoid presenting “guard,” “serve,” “gateway,” and “proxy” as four equivalent product modes. | [managed-agent choices](../../README.md#one-managed-agent-two-ways-to-run-the-kernel) |
| Integration compatibility | **integration path** | A supported way to route a named agent or protocol through fak. | Framework, SDK, protocol, and upstream names. | Do not turn “works with,” “compatible,” or a protocol example into an unscoped support promise. | [integration choices](../integrations/README.md#which-agent-do-you-run) |
| Documentation lifecycle | **current**, **preview**, **historical**, **archived** | The page's lifecycle, using exactly one applicable state. | Exact historical labels from preserved evidence. | Do not substitute “latest,” “legacy,” “old,” or “new.” Lifecycle does not imply product support. | [metadata field contract](../templates/documentation-metadata.md#field-contract) |
| Documentation generation | **`gen/now`** or an exact named generation | The documentation stream or exact historical generation. | Exact release, date, commit range, or artifact generation. | Replace vague “latest,” “modern,” or “previous generation.” | [metadata field contract](../templates/documentation-metadata.md#field-contract) |
| Claim maturity | **`[SHIPPED]`**, **`[SIMULATED]`**, **`[STUB]`** | The status of a public capability claim. | None in claim-ledger entries; every claim carries exactly one tag. | Do not translate the tags into “available,” “planned,” or “works” without retaining the authoritative status and scope. | [claim-status ledger](../../CLAIMS.md#claimsmd--the-fak-honesty-ledger) |
| Performance comparison | **tuned baseline**, **fak result**, and **provenance** | A scoped benchmark comparison that identifies the real alternative and whether evidence is witnessed, observed, or modeled. | Benchmark-specific arm labels and metric names after they are defined. | Do not headline a naive baseline, detached number, or generic “faster/cheaper” claim. | [benchmark authority](../../BENCHMARK-AUTHORITY.md#benchmark-authority--single-source-of-truth) |

## Resolve common ambiguous phrases

Do not guess which concept an ambiguous phrase names. Apply the observable rewrite below or stop for its named evidence.

| Ambiguous phrase | Check | Deterministic rewrite or action |
|---|---|---|
| “FAK platform proxy” | Is the phrase naming the product, a run path, or protocol compatibility? | Product: **fak, the agent kernel**. Run path: **local guard** or **managed endpoint**. Compatibility after a run path is named: **fak gateway**. Remove “platform” and “proxy”; if the intended concept is still unclear, stop and identify the reader choice first. |
| “moderation blocked it” | Does the sentence report the process, result, or both? | Process: **adjudication**. Result: **DENY** plus the available reason token. Both: “Adjudication returned `DENY` (`REASON_TOKEN`).” |
| “gateway mode” | Is the reader choosing how to run fak, or describing protocol compatibility after that choice? | Choice: **local guard** or **managed endpoint**, selected by where the agent runs. Compatibility surface: **gateway**, after naming the selected run path; gateway is not a third mode. |
| “latest docs” | Is the claim about page authority, documentation stream, or both? | Authority: lifecycle **current**. Stream: generation **`gen/now`**. State both when both facts matter; neither implies product support. |
| “works today” | Which claim-ledger status and evidence support the capability? | Stop until the claim carries exactly one authoritative `[SHIPPED]`, `[SIMULATED]`, or `[STUB]` tag and its scope/evidence link. |
| “78% faster” or another detached number | What tuned baseline, metric scope, and provenance produced it? | Stop until the statement names the tuned baseline, fak result, scope, provenance, and benchmark-authority link. |
| Two current authorities disagree | Which owning evidence establishes current behavior or support? | Stop normalization, reconcile the owning evidence, then update the authority and its projections together. |
## Apply the audit

For each changed front-door route:

1. **Inventory concepts, not words.** Mark each phrase that names a product, operation, mode, lifecycle, support state, claim, or benchmark relationship.
2. **Resolve authority.** Match the concept to this table and follow its authority link. Use the [documentation authority map](../project/documentation-authority-map.json) when duplicated explanations also need one canonical owner.
3. **Normalize public prose.** Use the preferred term on first meaningful mention. Keep scoped exceptions only where their exact spelling carries protocol, command, code, quotation, or historical meaning.
4. **Preserve the decision.** Keep prerequisites, mode conditions, support scope, claim tags, and evidence links adjacent to the normalized term.
5. **Read back across routes.** An independent reader should assign the same meaning and next action to the term at every changed entry point.

## Exceptions and conflicts

- **Exact interfaces win locally.** Commands, flags, JSON fields, API names, package names, filenames, and reason tokens remain byte-exact and code-formatted.
- **Quotations remain quotations.** Do not silently rewrite quoted evidence; explain the current preferred term outside the quote when needed.
- **Historical routes preserve evidence.** Label their lifecycle and exact generation, name the current successor, and do not let historical terminology redefine a current concept.
- **A real distinction gets a definition.** If two terms describe different observable choices, define both and state the condition that selects each; do not collapse them for cosmetic consistency.
- **A conflict stops normalization.** If current authorities disagree about meaning, support, or behavior, reconcile the owning facts before replacing labels.

## Completion check

- [ ] Every changed public concept maps to one preferred term and one authority.
- [ ] The first meaningful use is understandable without contributor vocabulary.
- [ ] Exact commands, fields, identifiers, quotations, and historical evidence remain faithful.
- [ ] Mode conditions, prerequisites, support scope, claim status, and benchmark provenance remain adjacent.
- [ ] `ALLOW` and `DENY` remain outcomes; consequences and reason tokens are not presented as extra outcomes.
- [ ] Lifecycle and generation terms do not imply product support.
- [ ] Every local authority link and anchor resolves.
- [ ] An independent reader can restate the concept, preferred term, applicable exception, and next action without guessing.

**Next action:** audit one front-door route against the table, normalize each conflicting public label, and run this completion check.
