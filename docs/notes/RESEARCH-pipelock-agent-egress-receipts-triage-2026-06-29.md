---
title: "idea-scout triage: Pipelock / agent egress firewall with signed action receipts (#1267) - prior art to cite for mediator-side egress control and external audit evidence; no fak code adopted in this increment (2026-06-29)"
description: "Triage of the idea-scout candidate GitHub repo luckyPipewrench/pipelock: a Go agent-egress firewall that mediates HTTP, WebSocket, CONNECT, MCP, and A2A traffic, scans for secret exfiltration / SSRF / prompt injection, and emits mediator-signed action receipts. Verdict: prior art to cite plus a same-side sibling control. It independently validates fak's mediator-outside-the-agent boundary and witness-not-self-report stance, especially the value of external evidence from the mediator. Its scanner/proxy mechanisms are not adopted: fak's load-bearing floor remains structured tool-call adjudication, IFC sink gating, and result admission, with detectors deliberately additive. Named residual: fak's hash-chained audit journal is tamper-evident but not a pinned-signer action-receipt format; signed receipt export is a candidate follow-on, not shipped here. No change to tools/idea_scout.py."
---

# idea-scout triage - Pipelock / agent egress firewall with signed action receipts (issue #1267)

> Closes the daily idea-scout candidate [#1267](https://github.com/anthony-chaudhary/fak/issues/1267)
> (`tools/idea_scout.py`, filed 2026-06-29). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off - adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + same-side sibling control. Pipelock is not a new
> threat and no scanner/proxy mechanism is adopted in this increment. It is useful
> prior art for two fak positions: put a mediator outside the agent on the action
> boundary, and make the mediator's evidence independently verifiable rather than
> trusting the agent's self-report.**

**Source:** <https://github.com/luckyPipewrench/pipelock> - "Pipelock", an Apache-2.0
core Go repository surfaced by the scout under `mcp-security` (738 stars at filing). Read
from the repository README and docs on 2026-06-29, especially the receipt verification,
security assurance, and in-toto action-receipt profile docs. This is a source-level triage
read, not an audit, benchmark reproduction, or endorsement of its reported coverage.

## What Pipelock is

Pipelock is an agent egress firewall: it sits between an agent and mediated network/tool
transports, then scans traffic crossing that boundary. Its README describes coverage for
HTTP, WebSocket, CONNECT, MCP, and A2A; DLP/prompt-injection/SSRF checks; MCP proxying and
tool policy; process containment modes; a flight recorder; and Ed25519-signed action
receipts / receipt chains that a verifier can check outside the agent runtime.

The important shape for fak is not "another scanner exists." It is the **topology**:
Pipelock's security decision is authored by a mediator outside the agent process and outside
the agent's credentials, then recorded as external evidence. That is the same trust move fak
makes at the tool-call seam: the model proposes, the mediator decides, and the operator
audits the mediator's record rather than the agent's story about what happened.

## Where it maps to fak

| Pipelock surface | fak position |
|---|---|
| Network/egress proxy that mediates agent HTTP, WebSocket, MCP, A2A, and related traffic | Same-side runtime mediator, but at a different primary seam. fak's load-bearing floor is the structured tool-call / result-admit boundary (`fak_adjudicate`, `fak_admit`, `fak guard`, `fak serve`), not a transparent network proxy for every byte an agent process can emit. |
| DLP, SSRF, prompt-injection, and tool-poisoning scanners | Prior art/context, not a mechanism to import. fak already records that lexical/semantic detectors (`normgate`, `ctxmmu`, `wirescreen`) are evadable and therefore additive. The floor remains capability policy + IFC sink gating + result quarantine. |
| MCP request/response scanning and tool-definition change checks | Overlaps fak's result-side stack and change/provenance seams (`AdmitResult`, `/v1/fak/changes`, `revoke`, IFC taint). fak still should not claim it sanitizes all third-party tool descriptions; calls and admitted results are the mediated facts. |
| Ed25519-signed action receipts and hash-chained evidence | Strong prior art for fak's audit story. fak ships a hash-chained audit journal (`internal/journal`) and `/v1/fak/events`, but that journal is tamper-evident, not a pinned-signer receipt format. Pipelock sharpens the external-auditor bar: "which mediator signed which action decision under which policy?" |
| Process sandbox / direct-egress containment | Deployment sibling, not a fak code change. fak can deny mediated tool calls; an operator still needs a real containment/proxying boundary if a child process can bypass the mediated path and open raw network sockets. |

## Triage decision

- **Adopt as a fak capability?** Not in this issue. The scanner/proxy stack is a separate
  product boundary, and importing it would cut against fak's "detector is not the floor"
  discipline. The one capability worth considering later is a **signed external receipt
  export** for fak's existing audit journal: a verifier-pinned Ed25519 envelope over each
  mediated decision, or over batches, so an auditor can bind the hash chain to a trusted
  mediator key. That is a scoped follow-on, not a small triage fix.
- **Defend against as a threat?** Pipelock is not the threat; it is a sibling defense. The
  threats it names - credential exfiltration, prompt injection, SSRF, tool misuse, and MCP
  poisoning - are already in fak's threat model. This source adds independent framing and a
  deployment reminder: fak only governs traffic that actually crosses its mediated seam.
- **Cite as prior art?** Yes. Cite Pipelock for **mediator-side agent egress control** and
  for **signed action receipts as external audit evidence**. It belongs near fak's
  reference-monitor / IFC / witness-not-self-report prior-art line, with the honest fence
  that fak has a hash-chained journal today, not Pipelock-style signed action receipts.

## Named residual

The residual this candidate usefully exposes is **audit provenance, not call gating**. fak's
current journal can detect edits to its row chain, and rows carry action/verdict/reason
digests, but a third-party verifier cannot yet pin those rows to a mediator signing key the
way Pipelock's receipt verifier does. That does not make fak's decision floor weaker; it
limits the external audit claim. Do not describe fak's journal as a signed action-receipt
system until such an export exists.

**Scout calibration (no code change).** The candidate surfaced under `mcp-security`
(score 32) from genuine MCP/security terms, non-trivial stars, and a fresh push. That is a
correct scout hit: new, on-topic, and close enough to fak's runtime-boundary/audit-evidence
work to warrant human triage. No change to `tools/idea_scout.py`.

**Action:** close [#1267](https://github.com/anthony-chaudhary/fak/issues/1267) as
triaged -> **prior art recorded; scanner/proxy mechanisms not adopted; signed receipt export
named as a future audit-evidence follow-on, not shipped here**.
