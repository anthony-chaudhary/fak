---
title: "fak vendor adoption scorecard - Anthropic, OpenAI, and standards"
description: "A dated strategic scorecard for how attractive fak is to Anthropic and OpenAI teams today, plus the smallest work program that can make it roughly 2x easier to adopt internally or as a standard."
---

# Vendor adoption scorecard - Anthropic, OpenAI, and standards

**Date:** 2026-06-25
**Scope:** score how attractive `fak` is to multiple Anthropic and OpenAI teams, then name
the smallest repo/product changes that can make adoption roughly 2x easier. This is a
strategy artifact, not a shipped-claims ledger. For shipped scope, use
[`CLAIMS.md`](../CLAIMS.md), [`docs/PRODUCT-STATUS.md`](PRODUCT-STATUS.md), and the
integration docs under [`docs/integrations/`](integrations/README.md).

## Headline

| Buyer | Current attractiveness | Best fit today | Main blocker | 2x path |
|---|---:|---|---|---|
| Anthropic | **7.9/10** | Claude Code, MCP, enterprise policy, agent security | `fak` is framed as a strong external gateway, not yet as a managed Claude Code / MCP governance profile Anthropic could bless | Package the Claude Code + MCP path as a managed policy/plugin proof pack and propose the governance profile as an MCP security extension |
| OpenAI | **7.2/10** | Codex, Agents SDK tool guardrails, enterprise managed config, Responses-compatible gateways | OpenAI already has sandbox/approval/guardrail primitives, so `fak` must present itself as the external evidence-bearing tool-call kernel, not another guardrail library | Ship a Codex MCP + Agents SDK adapter proof pack and map `fak` verdicts into OpenAI tracing/guardrail concepts |
| Standards/ecosystem | **7.5/10** | MCP tool-result governance, cross-vendor gateway contract, SIEM/audit profile | The standard shape is implicit in docs, not written as a small conformance target | Publish an Agent Tool Governance Gateway profile with conformance fixtures and a "fak is one implementation" stance |

The honest reading: `fak` is already attractive as a **governance boundary**. It is
less attractive as a model-serving engine or benchmark winner, because that is not the
adoption wedge. The 2x opportunity is to stop selling the whole kernel first and instead
make three narrow adopter paths obvious:

1. **Anthropic internal path:** Claude Code / Claude API / MCP teams can adopt `fak` as
   the externalized, auditable tool-call policy plane.
2. **OpenAI internal path:** Codex / Agents SDK / enterprise teams can adopt `fak` as the
   evidence-bearing boundary around tool execution and MCP servers.
3. **Standards path:** MCP and agent-framework ecosystems can adopt a profile that says
   what a governed tool-call gateway must emit, independent of the `fak` implementation.

## Scoring method

Score each target team on a 0-10 scale:

| Dimension | Weight | What earns points |
|---|---:|---|
| Strategic pain | 30% | The team has a real, current need for tool-call safety, audit, policy, MCP governance, or cross-agent consistency |
| Integration fit | 25% | `fak` sits on a wire the team already supports: Anthropic Messages, OpenAI-compatible APIs, MCP, or a tool guardrail boundary |
| Evidence | 20% | The repo already has runnable proofs, tests, claims, or demos that support the pitch |
| Adoption surface | 15% | A team can pilot it without a broad rewrite or ownership fight |
| Standards leverage | 10% | The work could become a reusable protocol/profile rather than a one-off integration |

Current attractiveness is capped by the weakest production-adoption factor: if the
team fit is strong but the path is not packaged as a one-day pilot, the score should
not exceed 8.

## Current-state anchors

These external anchors explain why these teams are plausible buyers. They are not `fak`
claims.

- Anthropic positions Claude Code around tools, MCP, hooks, skills, subagents, managed
  settings, enterprise policy, and security controls:
  [Claude Code overview](https://docs.anthropic.com/en/docs/claude-code/overview),
  [MCP in Claude Code](https://docs.anthropic.com/en/docs/claude-code/mcp),
  [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks),
  [security](https://docs.anthropic.com/en/docs/claude-code/security), and
  [enterprise IAM / managed settings](https://docs.anthropic.com/en/docs/claude-code/iam).
- Anthropic's API surface includes client/server tool use and a Messages API MCP
  connector:
  [tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
  and
  [MCP connector](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector).
- OpenAI Codex is a local/cloud coding-agent surface with sandboxing, approvals,
  MCP configuration, enterprise managed configuration, and security workflows:
  [Codex CLI](https://developers.openai.com/codex/cli),
  [agent approvals and security](https://developers.openai.com/codex/agent-approvals-security),
  [Codex MCP](https://developers.openai.com/codex/mcp),
  [managed configuration](https://developers.openai.com/codex/enterprise/managed-configuration), and
  [Codex Security](https://developers.openai.com/codex/security).
- OpenAI's Agents SDK has tool guardrails, MCP support, and tracing concepts that can
  receive a `fak` verdict mapping:
  [tool guardrails](https://openai.github.io/openai-agents-python/guardrails/),
  [MCP support](https://openai.github.io/openai-agents-python/mcp/), and
  [tracing](https://openai.github.io/openai-agents-python/tracing/).
- MCP is already the shared protocol surface for tool integration. The standards wedge
  should build on the official
  [MCP specification](https://modelcontextprotocol.io/specification/2025-06-18),
  [tools spec](https://modelcontextprotocol.io/specification/2025-06-18/server/tools),
  and
  [security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices).

## Anthropic team score

| Team / buyer | Score | Why it should care | Why `fak` fits | Main gap |
|---|---:|---|---|---|
| Claude Code product / SDK | **8.6** | Claude Code exposes tools, MCP, hooks, subagents, skills, permissions, and managed settings. Tool execution is a core product surface. | `fak serve --stdio`, the Claude guide, dogfood policy, and `fak guard -- claude` already match the product boundary. | The adoption path is not packaged as a managed Claude Code plugin/policy pack with a one-command enterprise rollout story. |
| Claude API / Messages / tools | **7.7** | Tool use and the MCP connector create a server-side tool boundary that benefits from structured policy and result quarantine. | `fak` already fronts Anthropic Messages and can also be asked through MCP-style verdict tools. | The docs need a "Messages API MCP connector plus fak" reference architecture with exact request/response placement. |
| MCP ecosystem / standards | **8.2** | Anthropic has strong incentives for MCP to remain secure, governable, and enterprise-adoptable. | `fak` contributes a concrete governance profile: default deny, structured refusal reasons, result quarantine, revocation, audit. | The profile is implicit; it needs a neutral spec that does not read like "install this repo." |
| Enterprise security / compliance | **8.5** | Enterprise buyers need policy distribution, audit, data containment, and predictable denials. | `fak` is strongest where the model cannot talk past a policy floor, and the repo has product/security scorecards plus runnable offline proof. | Needs an executive proof packet: threat model, first 3 denied calls, log schema, SIEM path, and residual risk table. |
| Infra / cost / long-context | **6.6** | Multi-agent and long-context cost pressure matters, especially for Claude Code teams. | The repo has vCache, prefix-reuse, and planned-context evidence. | The most vendor-relevant cache path still needs a clean Anthropic-specific pilot result, not a broad infrastructure argument. |

**Anthropic overall: 7.9/10.** The highest-confidence wedge is not "replace anything."
It is: make Claude Code and MCP deployments safer to adopt at enterprise scale while
preserving the Claude product surface.

## OpenAI team score

| Team / buyer | Score | Why it should care | Why `fak` fits | Main gap |
|---|---:|---|---|---|
| Codex CLI / IDE / app-server | **7.9** | Codex already has sandboxing, approvals, MCP servers, project config, and enterprise management. It needs composable policy and external tool governance. | `fak` can be added as an MCP server or a gateway in front of OpenAI-compatible local/server flows. | The Codex-specific guide should include a managed-config rollout and a before/after trace showing a refused tool call. |
| OpenAI Agents SDK | **7.5** | Tool guardrails and tracing are natural places to insert pre-execution and post-result checks. | `fak_adjudicate` and `fak_admit` map directly to input/output tool guardrail behavior. | Needs a tiny Python adapter package or example that converts `fak` verdicts into Agents SDK guardrail outcomes and trace spans. |
| Responses / API platform | **6.8** | A governed gateway can protect tool calls across client libraries and model providers. | `fak serve` already speaks OpenAI-compatible surfaces and has gateway docs. | The repo should be explicit about Responses API parity, streaming edge cases, and where `fak` is Chat Completions-only today. |
| Codex Security / enterprise governance | **7.2** | Security buyers care about validated findings, approvals, audit, and controlled rollout. | `fak` has an evidence-first culture and can provide portable tool-call audit artifacts. | Needs a "Codex Security complement" note: `fak` governs runtime tool execution; Codex Security scans code and validates findings. |
| Infra / cache / fleet | **6.5** | OpenAI has huge incentives around token cost, cache hits, and agent throughput. | `fak` has a clear theory around shared-prefix and cross-agent reuse. | Vendor fit is lower unless the benchmark is run on OpenAI-compatible telemetry with a current model/API path. |

**OpenAI overall: 7.2/10.** The strongest wedge is to make `fak` look like an
external, auditable tool-call kernel that complements Codex and the Agents SDK rather
than competing with their built-in sandbox, approval, and guardrail layers.

## The 2x adoption program

Use "2x" as an adoption-path metric, not as a claim that a capped 0-10 score can
double. Today there are about **7 credible one-week pilot paths**:

| Track | Current credible pilots |
|---|---:|
| Anthropic | 3: Claude Code MCP, Claude Code guard/dogfood, enterprise policy/audit |
| OpenAI | 3: Codex MCP, OpenAI-compatible gateway, Agents SDK guardrail mapping |
| Standards | 1: MCP tool-result governance profile |
| **Total** | **7** |

The target is **15 credible pilots** after the following work, a 2.1x expansion:

| Priority | Change | New pilots unlocked | Adoption multiplier |
|---|---|---:|---:|
| P0 | Add a **vendor proof packet**: two one-page internal memos, one for Anthropic and one for OpenAI, each with the first command, denied-call witness, integration boundary, and residual risks | +2 | 1.29x |
| P0 | Add a **Codex + Agents SDK adapter example** that maps `fak_adjudicate` to a tool input guardrail and `fak_admit` to a tool output guardrail | +2 | 1.57x |
| P0 | Add a **Claude Code managed-policy/plugin rollout guide**: managed settings, MCP server entry, hook position, audit-log path, and rollback | +2 | 1.86x |
| P1 | Publish the **Agent Tool Governance Gateway profile**: verdict schema, refusal vocabulary, quarantine semantics, audit fields, and conformance fixtures | +2 | 2.14x |
| P1 | Run a **vendor-specific live pilot proof**: one Claude Code session and one Codex session, each showing allowed work continues after a denied dangerous call | +2 | 2.43x |

**Progress as of 2026-06-25:** the repo now has the vendor proof packet
([Anthropic](vendor/anthropic-internal-pitch.md),
[OpenAI](vendor/openai-internal-pitch.md)), the
[OpenAI Agents SDK guardrail adapter example](../examples/openai-agents-guardrail/),
the [Claude Code managed rollout guide](vendor/claude-code-managed-rollout.md), and
the [Agent Tool Governance Gateway profile](standards/agent-tool-governance-gateway.md)
with [conformance fixtures](standards/fixtures/). That moves the artifact-backed
pilot count from **7 to 15** (2.14x). The remaining P1 evidence gap is live pilot
data proving the same paths during real Claude Code and Codex sessions.

Stop after P1 if the goal is 2x. P2 work can improve depth, but it is not needed to
double the number of credible pilots.

## Standards profile shape

The standard should not be "fak protocol." It should be a small vendor-neutral profile
that any gateway, MCP server, IDE agent, or SDK guardrail can implement.

**Working name:** Agent Tool Governance Gateway profile.

Minimum conformance:

| Field / behavior | Requirement |
|---|---|
| Pre-execution verdict | Every proposed tool call can be answered as `ALLOW`, `DENY`, `TRANSFORM`, or `REQUIRE_WITNESS` before execution. |
| Closed reason vocabulary | Every refusal carries a stable machine token, not only prose. |
| Result admission | Tool results pass through a post-execution admit step before entering model-visible context. |
| Quarantine | A result can be held out of context while preserving a byte-exact audit/recovery path. |
| Revocation | A poisoned or stale external witness can evict dependent cache/context entries. |
| Audit | Every decision carries trace id, tool name, verdict, reason, policy identity, and timestamp. |
| Conformance fixtures | A public fixture proves one allowed call, one denied call, one transformed call, one quarantined result, and one revoked witness. |

`fak` should present itself as one reference implementation of this profile. That is
more attractive to Anthropic and OpenAI than asking either vendor to bless a single
project-specific API.

## What to build next

Do the smallest work that changes adoption probability, in this order:

1. **Done: `docs/vendor/anthropic-internal-pitch.md`.** One page for Claude Code,
   MCP, API tools, enterprise settings, and security, with a pilot command and
   denied-call witness.
2. **Done: `docs/vendor/openai-internal-pitch.md`.** One page for Codex, Agents SDK,
   MCP, managed configuration, and runtime security, with the guardrail mapping table.
3. **Done: `examples/openai-agents-guardrail/`.** Minimal Python example: call
   `fak_adjudicate` before a tool and `fak_admit` after a tool; emit an Agents SDK trace
   mapping if available.
4. **Done: `docs/standards/agent-tool-governance-gateway.md`.** Vendor-neutral
   profile plus conformance JSON fixtures.
5. **Record two live pilot artifacts under `experiments/agent-live/`.** One Claude Code,
   one Codex. Each must show "danger denied, useful task continues."

## Non-goals

- Do not pitch `fak` as a replacement for Claude Code, Codex, the Agents SDK, or the
  MCP spec. The strongest pitch is complement, not replacement.
- Do not lead with raw benchmark speedups. Vendor teams will discount anything that is
  not compared against their own current caching and runtime controls.
- Do not ask a vendor to adopt an implementation before they can adopt the standard
  profile. Lead with the profile; use `fak` as the running proof.

## Success criteria

This adoption effort is working when the repo can answer all of these with concrete
artifacts:

| Question | Evidence required |
|---|---|
| Can an Anthropic team pilot it in one day? | Anthropic pitch page plus Claude Code/MCP managed rollout guide and live denied-call artifact |
| Can an OpenAI team pilot it in one day? | OpenAI pitch page plus Codex/Agents SDK adapter example and live denied-call artifact |
| Can the idea become a standard without forcing `fak` adoption? | Vendor-neutral Agent Tool Governance Gateway profile plus conformance fixtures |
| Does a denied call preserve utility? | Each live pilot shows an unsafe call denied and the useful task still completed |
| Is the residual risk honest? | Each pitch page names what `fak` does not govern on that vendor surface |
