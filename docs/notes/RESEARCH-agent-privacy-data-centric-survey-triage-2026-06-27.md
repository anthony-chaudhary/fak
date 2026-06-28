---
title: "idea-scout triage: 'Agents That Know Too Much: A Data-Centric Survey of Privacy in LLM Agents' — a field-level survey that taxonomizes the data sources an agent touches, the privacy risk each creates, and the governance mechanisms that address them (it names information-flow control + access control). Verdict: prior art to cite + a real multi-channel threat fak's IFC governance gates UNEVENLY by provenance — the egress/exfil sink and secret-shaped intermediate results are gated-by-provenance (internal/ifc SinkEgress/SinkDestructive + the #552 result-admit gate), but the query-issuance, memory-write, and inter-agent-message channels the survey enumerates are gated-by-capability-only or unmeasured, and fak reports NO privacy-leakage number. Not adopted as a capability (a survey, no kernel mechanism) and not a new threat class (the channels are known); the smallest honest follow-on is a per-channel privacy-posture audit + a candidate leakage metric, FILED not built. The survey is the field umbrella over the source-specific MCPPrivacyDetector triage (#552). No code change (2026-06-27)"
description: "Triage of the idea-scout candidate arXiv:2606.26627 (Nada Lahjouji, Ashwin Gerard Colaco — 'Agents That Know Too Much: A Data-Centric Survey of Privacy in LLM Agents', submitted 2026-06-25): a survey that organizes LLM-agent privacy around the DATA an agent touches rather than by attack type — taxonomizing the data sources (databases, document collections, external APIs, agent memory, cross-session state), the privacy risk each source creates (query leakage, intermediate-result exposure, memory-write vulnerabilities, inter-agent messaging), and the governance mechanisms that address them (it names information-flow control and access control), then maps the benchmarks and identifies what is missing. Verdict: prior art to cite + a real threat whose channels fak's own IFC governance gates UNEVENLY. fak HAS one of the two governance families the survey names: internal/ifc is an information-flow-control mechanism — a provenance taint Ledger (SourceTaint) classifies a tool call into a SinkClass (Egress/Exec/Destructive) and Policy.Gates refuses a tainted->gated-sink flow (DefaultGatedSinks gates EGRESS+DESTRUCTIVE; the CaMeL-style Authorize escape permits explicitly-authorized egress). So the 'final answer' exfil channel and (via the #552 result-admit gate, ctxmmu/normgate) the secret-shaped intermediate-result channel are gated-by-provenance at runtime. But the survey's other enumerated channels are NOT privacy-gated to the same standard: the queries the agent issues are gated by CAPABILITY not by a privacy assessment of the query content; agent memory-writes (cross-session state) ride the MEMORY-VIEW-CONTRACT read-side discipline but are not taint-gated on write; inter-agent messages cross a boundary the taint ledger does not provably propagate across; and — the survey's own 'what is missing' — fak reports NO measured privacy-leakage number (same 'containment is not a measured number' residual as MIRROR #1007). Mechanism NOT adopted: a survey contributes a taxonomy + a benchmark map, not a kernel component; fak adopts no code. The smallest honest follow-on is a per-channel privacy-posture audit that maps the survey's data-source x risk grid onto fak's runtime gates (gated-by-provenance vs gated-by-capability-only vs unmeasured) plus a candidate leakage metric, FILED not built — with the fence that the survey is cited as the field framing + a coverage map, NEVER as evidence fak's privacy containment is complete or measured. The field-level umbrella over the source-specific MCPPrivacyDetector triage (#552), which instantiates one cell (MCP-server returns). Scout-calibration: surfaced under topic prompt-injection-defense (score 47) on genuine prompt-injection/agent(title) terms + freshness (2d, <=30d) — correctly on-topic; no change to tools/idea_scout.py."
---

# idea-scout triage — data-centric survey of privacy in LLM agents (issue #1098)

> Closes the daily idea-scout candidate [#1098](https://github.com/anthony-chaudhary/fak/issues/1098)
> (`tools/idea_scout.py`, filed 2026-06-28). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + a real multi-channel threat fak's own IFC governance
> gates UNEVENLY by provenance. fak HAS one of the two governance families the survey
> names — `internal/ifc` is information-flow control — so the exfil/egress sink and the
> secret-shaped intermediate-result channel are gated-by-provenance at runtime; but the
> query-issuance, memory-write, and inter-agent-message channels the survey enumerates
> are gated-by-capability-only or unmeasured, and fak reports no privacy-leakage number.
> Not adopted as a capability (a survey, no kernel mechanism). The smallest honest
> follow-on — a per-channel privacy-posture audit + a candidate leakage metric — is
> FILED, not built. No code change.**

**Source:** https://arxiv.org/abs/2606.26627 — "Agents That Know Too Much: A
Data-Centric Survey of Privacy in LLM Agents", Nada Lahjouji, Ashwin Gerard Colaco
(submitted 2026-06-25). Read from the arXiv abstract via WebFetch on 2026-06-27; this is
a surface read of the abstract, not a paper audit or a reproduction.

## What it is

A **survey**, not a serving system, a protocol, an attack, or a defense. Its organizing
move is **data-centric**: it taxonomizes LLM-agent privacy around **the data an agent
touches** (it coins *data agent* for an LLM agent that works with data) rather than by
attack type, because the work is "active but scattered across retrieval-augmented
generation, text-to-SQL interfaces, agent memory, prompt injection, access control, and
contextual privacy." It states the threat in channel terms fak already reasons in:
sensitive information leaks "not only through its final answer but through the **queries it
issues**, the **intermediate results it handles**, the **memory it writes**, and the
**messages it exchanges with other agents**." It then (a) taxonomizes data source -> risk
-> **governance mechanism** — naming **information-flow control** and **access control** as
the governance families — (b) **maps the benchmarks** used to measure these risks and
**identifies what is missing**, and (c) sets out the open problems.

It contributes a **taxonomy + a benchmark map + open problems**. It does not propose a
kernel component, a runtime mechanism, an MCP server, or a serving primitive.

## The three triage questions

fak is an **agent kernel**: one Go binary at the tool-call seam that adjudicates every tool
call before it runs — a default-deny **capability floor** plus do-the-shared-setup-once
cross-turn **reuse**. Against that mission:

- **Adopt as a capability? No.** A survey has no mechanism to fold into the kernel. Its
  value is a *map*, not a *part*.
- **Defend against as a threat? The threat is real, already partly contained, and partly
  unmeasured.** The multi-channel leak the survey frames is exactly fak's trust-boundary
  problem — and the next two sections give fak's honest per-channel posture, not a blanket
  "covered."
- **Cite as prior art? Yes** — it is the **field-level umbrella** over fak's own
  data-centric IFC stance and over the source-specific
  [MCPPrivacyDetector triage (#552)](RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md),
  which instantiates exactly one cell of this survey's grid (the MCP-server *return* sink).

## Why this is prior art to cite — fak already runs one of the two governance families

The survey names two governance families: **information-flow control** and **access
control**. fak ships an **information-flow-control** mechanism on the load-bearing path.
`internal/ifc` is a literal IFC decision table:

- a **taint `Ledger`** raises a session's taint by **provenance** (`SourceTaint` /
  `provenance.Taint`) — *who produced this data*, not what it says;
- `Classify` sorts a tool call into a **`SinkClass`** — `SinkEgress` (sends data to an
  external, attacker-reachable destination), `SinkExec` (runs code), `SinkDestructive`
  (irreversibly mutates) — with egress detection on the call's destination args
  (`hasExternalDestination` / `isEgressKey` / `looksExternal`);
- **`Policy.Gates`** refuses a **tainted -> gated-sink** flow. `DefaultGatedSinks` gates the
  **EGRESS** and **DESTRUCTIVE** channels by default (the anti-exfil property: no false
  negatives on egress); `StrictGatedSinks` adds **EXEC** for the untrusted-input threat
  model; the CaMeL-style **`Authorize`** escape permits *explicitly-authorized* legitimate
  egress, default fail-closed.

That is the survey's "information-flow control governance mechanism," shipped. So the
survey is worth citing as **independent field framing** for a load-bearing fak design
choice — fak gates by **provenance**, which is exactly the data-centric lens the survey
adopts — and as the **umbrella taxonomy** above the #552 MCP-privacy note. It is the same
prior-art discipline fak already keeps (`CLAIMS.md` "the contribution is the assembly").

## The honest per-channel posture — what fak gates, and what it does not

The survey's strength for fak is that it **enumerates the channels**, which forces an honest
audit rather than a blanket claim. Against the survey's four leak channels:

| Survey leak channel | fak's runtime posture |
|---|---|
| **final answer** (the model's output) | **Gated-by-provenance.** A tainted session's `SinkEgress` call is refused by `Policy.Gates` (`DefaultGatedSinks`). This is fak's strongest cell. |
| **intermediate results it handles** | **Gated-by-provenance, best-effort.** The result-admit gate (`ctxmmu`/`normgate`, see [#552](RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md)) quarantines secret-shaped tool results before they enter context; `DenyResultsOverTaintCeiling` can hard-refuse an over-ceiling result. Carries the documented `~100% evadable + FP-prone` detector ceiling — a best-effort rung, never the floor. |
| **the queries it issues** | **Gated-by-capability, NOT by a privacy assessment.** fak adjudicates *whether* a query tool may run (capability floor), but does not assess the *content* of an outbound query for over-collection (a SELECT that pulls a sensitive column). The survey's text-to-SQL / RAG-query risk cell is **not** a measured fak gate. |
| **the memory it writes** | **Read-side disciplined, write-side untainted.** fak has a [memory-view contract](MEMORY-VIEW-CONTRACT-2026-06-26.md) on the read side, but a cross-session memory *write* is not taint-gated as a privacy sink. Named residual. |
| **messages it exchanges with other agents** | **Boundary not provably taint-propagating.** The taint `Ledger` is session-scoped; the survey's inter-agent channel crosses a boundary fak does not prove the label survives. Named residual. |

And the survey's own **"what is missing"** lands squarely on fak: fak reports **no measured
privacy-leakage number**. This is the same shape as the
[MIRROR triage (#1007)](RESEARCH-mirror-novelty-mcts-redteam-triage-2026-06-27.md) residual
— *containment is not a measured number* — now stated for the privacy axis specifically.

## Mechanism NOT adopted — and the smallest honest follow-on

- **Mechanism not adopted.** A survey contributes a taxonomy, a benchmark map, and open
  problems — there is no component to fold into a one-binary zero-dep kernel. fak adopts no
  code in this increment.
- **The smallest honest follow-on (FILED, not built):** turn the survey's
  **data-source x risk** grid into a tracked **per-channel privacy-posture audit** for fak —
  one row per channel, each labeled `gated-by-provenance` / `gated-by-capability-only` /
  `unmeasured`, exactly as the table above starts — plus a **candidate privacy-leakage
  metric** (an egress-channel leak-rate under the IFC gate, the privacy sibling of the
  AgentDojo ASR-under-containment number). This is filed as an `ifc`-lane follow-on, not
  built here. **The fence:** the survey is cited as **field framing + a coverage map**,
  **never** as evidence fak's privacy containment is complete or measured — the unmeasured
  and capability-only cells above are explicit, not papered over.

## Triage decision

- **Adopt?** No — a survey, no kernel mechanism.
- **Defend against?** The threat is real and **partly** contained: the egress and
  intermediate-result channels are gated-by-provenance (`internal/ifc` + the #552
  result-admit gate); the query-issuance, memory-write, and inter-agent channels are
  gated-by-capability-only or unmeasured.
- **Cite as prior art?** **Yes** — the field-level data-centric umbrella over fak's IFC
  stance and over the source-specific [#552](RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md)
  note, and independent framing for fak's gate-by-provenance bet.

**Action:** this note is the recorded triage; close
[#1098](https://github.com/anthony-chaudhary/fak/issues/1098) as triaged -> **prior art
recorded + a real multi-channel threat fak's IFC governance gates unevenly, with the
unmeasured/capability-only channels named and a per-channel audit + leakage metric filed,
not built**. No code change in this increment: the right artifact for a survey candidate is
the recorded verdict, the honest per-channel posture, and the named residuals — not a
speculative feature.

**Scout calibration (no code change).** The candidate surfaced under topic
`prompt-injection-defense` (score **47**) on genuine `prompt injection` / `agent (title)`
terms plus a freshness bonus (2d old, <=30d). The match is **on-topic and on-mission** — it
lands on fak's IFC/provenance trust boundary directly — though it is a survey rather than a
serving or security mechanism. The scout behaved **as designed**: it judges *new and
on-topic*, never *worth building*, and handed the call to human triage (see
[`docs/idea-scout.md`](../idea-scout.md)). The scoring was correct, so there is **no change
to `tools/idea_scout.py`** — only this recorded verdict.
