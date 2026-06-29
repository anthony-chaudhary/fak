---
title: "idea-scout triage: FuzzingLabs/mcp-security-hub — a catalog of 38 offensive-security MCP servers (Nmap, SQLMap, Nuclei, Hashcat, Ghidra, IDA, BloodHound, …) wired into AI assistants; a concrete dual-use THREAT-SURFACE EXEMPLAR + prior art to cite, NOT a capability to adopt. This is the canonical, popular, ready-to-attach instantiation of the very surface fak's default-deny capability floor exists to govern: each tool arrives as a structured MCP tool call, none match the guard allowlist or its read-only prefixes, so each stays DEFAULT_DENY at the kernel until an operator explicitly authorizes it — the mechanized form of the repo's OWN stated 'written-authorization + scope + audit-log' human process. Mechanism not adopted (fak is the governor, not the offensive toolbox). Named residual: fak ships no curated offensive-tooling starter policy and no adjudicator test corpus drawn from this inventory — a candidate witnessed gate, FILED not built. No change to tools/idea_scout.py (2026-06-29)"
description: "Triage of the idea-scout candidate GitHub repo FuzzingLabs/mcp-security-hub (595 stars, MIT, Python/Docker, last push 2026-04-08): a growing collection of 38 Dockerized MCP servers across 14 security domains that expose offensive-security tools to AI assistants — reconnaissance (Nmap, Masscan, Shodan, WhatWeb, ZoomEye), web exploitation (Nuclei, SQLMap, Nikto, Ffuf, Burp Suite), binary analysis / RE (Radare2, Ghidra, IDA Pro, Binwalk, Capa, Yara), credential cracking (Hashcat), AD attack pathing (BloodHound), exploit lookup (Searchsploit), fuzzing (Boofuzz, Dharma), cloud/secrets (Trivy, Prowler, Gitleaks), each run as a Docker container behind claude_desktop_config.json. Verdict: a concrete dual-use THREAT-SURFACE EXEMPLAR + prior art to cite; mechanism NOT adopted. It is the first idea-scout candidate that is a real, popular, attach-in-minutes inventory of high-capability offensive tools rather than a paper — exactly the tool inventory fak's default-deny capability floor must classify before it runs. Each tool reaches fak as a structured tool call; none matches the guard default policy's allowlist or its read-only allow-prefixes (read_/get_/search_/list_/lookup_/find_), so each stays DEFAULT_DENY at the kernel (cmd/fak/attest.go:237) until an operator authorizes it — the enforced form of the repo's own README admonition (obtain written authorization, define scope, maintain audit logs), turned from prose into a capability policy + a hash-chained decision journal, with offensive calls further mapping onto fak's IFC sink classes (Exec/Egress/Destructive). Honest fence: fak denies these because they are not allow-listed, NOT because it 'knows' Nmap is offensive (detector-is-not-the-floor) — so it ships no curated offensive-tooling starter policy and no adjudicator test corpus drawn from this catalog; the smallest honest follow-on is that fixture/corpus, FILED not built. Adopt? No — fak is the governor, not the offensive toolbox. Defend against? Indirectly — the threat is an agent with these tools attached and invoked without authorization/scope; the default-deny floor + journal + sink-gating is the containment, and this catalog is the cleanest concrete map of what that floor governs. Cite as prior art? Yes, as a dual-use threat-surface catalog, never as a fak mechanism. Scout calibration: surfaced under mcp-security (score 40) on genuine mcp(title)/server terms + 595 stars + a recent push — a correct, high-signal hit (a real attachable artifact, not an abstract). No change to tools/idea_scout.py."
---

# idea-scout triage — FuzzingLabs/mcp-security-hub: offensive-security MCP tooling (issue #1265)

> Closes the daily idea-scout candidate [#1265](https://github.com/anthony-chaudhary/fak/issues/1265)
> (`tools/idea_scout.py`, filed 2026-06-29). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: a concrete dual-use THREAT-SURFACE EXEMPLAR + prior art to cite; mechanism
> NOT adopted. mcp-security-hub is the first idea-scout candidate that is a real, popular,
> attach-in-minutes inventory of high-capability offensive tools — exactly the tool
> inventory fak's default-deny capability floor exists to govern. fak is the governor, not
> the offensive toolbox: each tool reaches fak as a structured tool call that stays
> `DEFAULT_DENY` at the kernel until an operator authorizes it. No code change.**

**Source:** <https://github.com/FuzzingLabs/mcp-security-hub> — "A growing collection of MCP
servers bringing offensive security tools to AI assistants" (595 stars, MIT, Python/Docker,
last push 2026-04-08). Read from the repository README on 2026-06-29 via WebFetch; this is a
source-level surface read, not an audit, install, or run of any of its servers.

## What it is

A catalog — the README counts **38 MCP servers across 14 security domains** — each wrapping a
well-known offensive-security tool and exposing it to an AI assistant over MCP. Representative
inventory by domain:

- **Reconnaissance:** Nmap, Masscan, Shodan, WhatWeb, ZoomEye, NetworksDB, ProjectDiscovery, ExternalAttacker
- **Web exploitation:** Nuclei, SQLMap, Nikto, Ffuf, Burp Suite, Wayback URLs
- **Binary analysis / RE:** Radare2, Ghidra, IDA Pro, Binwalk, Capa, Yara
- **Credential cracking / AD:** Hashcat, BloodHound
- **Exploit lookup / fuzzing:** Searchsploit, Boofuzz, Dharma
- **Cloud / secrets / SCA:** Trivy, Prowler, RoadRecon, Gitleaks, Semgrep, mcp-scan; plus blockchain (Medusa, Solazy) and threat-intel (VirusTotal, OTX, Maigret, DNSTwist)

Each tool runs as a "Production-ready, Dockerized MCP server," wired into an assistant through
`claude_desktop_config.json` Docker-run entries, with volume mounts for target files and
Docker-Compose orchestration for multi-tool workflows. The README carries an explicit
authorization notice: *"These tools are for authorized security testing only"* — obtain written
authorization from target owners, define scope, maintain audit logs, follow responsible
disclosure, *"Unauthorized access to computer systems is illegal."*

The important shape for fak is **not** "another security toolkit exists." It is that this is the
canonical, popular, ready-to-attach instantiation of a surface fak already names in the abstract:
**high-capability, dual-use tools are now one MCP config block away from any agent**, and once
attached they appear to the runtime as ordinary structured tool calls.

## Where it maps to fak — the load-bearing point

fak is a runtime trust gate at the tool-call seam: the model proposes a call, the kernel
adjudicates it before it runs (default-deny capability floor), and the operator audits the
kernel's decision journal rather than the agent's story. mcp-security-hub is the cleanest
concrete map of *what that floor governs*.

| mcp-security-hub surface | fak position |
|---|---|
| 38 offensive MCP tools attached to an assistant (`nmap_scan`, `sqlmap_run`, `hashcat_crack`, …) | Each arrives as a structured tool call. The guard default policy (`cmd/fak/guard-default-policy.json`) allow-lists only safe tools + **read-only** prefixes (`read_`/`get_`/`search_`/`list_`/`lookup_`/`find_`). An offensive tool name matches neither, so it stays **`DEFAULT_DENY`** at the kernel (`cmd/fak/attest.go:237`) until an operator explicitly authorizes it. |
| README process: "obtain written authorization, define scope, maintain audit logs" | This is the **human-process analogue of what fak mechanizes**. fak turns the README admonition into an *enforced* capability policy (default-deny, operator-authored allows) plus a **hash-chained decision journal** (`fak guard` / `internal/journal`) — a verifiable audit log, not a self-report. |
| Recon queries (Shodan/VirusTotal/ZoomEye) ship target data outward; a scanner/cracker runs code; an exploit run is destructive | Maps onto fak's IFC **sink classes** (`internal/ifc`): Egress / Exec / Destructive. The result-side `SinkGate` governs the *flow*, not just the *call* — a tainted→gated-sink flow is barred regardless of phrasing (`DefaultGatedSinks` = Egress + Destructive). |
| Dual-use: an operator on a real engagement **will** allow Nmap inside scope | The value fak adds there is **governance, not prohibition** — scoped capability allows + the audit journal + sink-gating, exactly the controls the repo asks an operator to apply by hand. |

The honest, fak-specific subtlety: fak denies these tools because they are **not allow-listed**,
**not** because it "knows" Nmap is offensive. That is the correct floor —
[detector-is-not-the-floor](RESEARCH-defensive-misdirection-triage-2026-06-23.md): the kernel
does not read a third-party tool's MCP description for its verdict. But it means fak ships **no
curated offensive-tooling starter policy** and **no adjudicator test corpus** exercising the
floor against this specific, real-world inventory.

## Triage decision

- **Adopt as a fak capability?** **No.** Shipping these servers would put fak on the wrong side
  of its own trust boundary — fak is the *governor* of such tool calls, not the offensive
  toolbox. There is no mechanism here to fold into the kernel.
- **Defend against as a threat?** **Indirectly.** mcp-security-hub is not the adversary; it is
  the **capability-amplification surface**. The threat is *an agent with these tools attached,
  invoked without authorization or scope* (or steered there by injection). fak's default-deny
  capability floor + decision journal + IFC sink-gating is precisely the containment, and this
  candidate is the cleanest concrete catalog of **what** that floor must classify.
- **Cite as prior art?** **Yes — as a dual-use threat-surface catalog, never as a fak
  mechanism.** Cite it where fak argues a default-deny capability floor matters *because*
  high-impact tools are now one MCP config away from any agent. It is the offensive-MCP-tooling
  sibling of the defensive-MCP triages ([MCPPrivacyDetector #552](RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md),
  [ShareLock #911](RESEARCH-sharelock-multitool-threshold-poisoning-triage-2026-06-26.md),
  [Pipelock #1267](RESEARCH-pipelock-agent-egress-receipts-triage-2026-06-29.md)) — the same
  MCP seam, viewed from the *attacker's toolbox* side.

## Named residual — FILED not built

The residual this candidate exposes is **coverage of the floor, not a hole in it**. fak's
default-deny floor already contains the unauthorized-offensive-call case by construction, but:

1. fak ships **no curated "offensive-tooling-MCP" starter capability policy** — an operator
   attaching mcp-security-hub gets default-deny on every server (safe), yet no guidance on
   which to scope-allow vs. keep denied for a given engagement.
2. fak has **no adjudicator test corpus** drawn from this inventory: canonical offensive MCP
   tool names + argument shapes → expected `DEFAULT_DENY` / sink-class, which would turn the
   catalog into a *witnessed* gate (the same move ShareLock/MIRROR motivated for the injection
   battery).

The smallest honest follow-on is that fixture + corpus, **filed not built** — shipping it now
would be an unwitnessed claim that fak "handles" this catalog, when the truthful statement is
that the floor *denies-by-default* and the curated coverage does not yet exist. Do **not**
describe fak as having an offensive-tooling-aware policy until such a fixture lands.

## Scout calibration (no code change)

The candidate surfaced under topic `mcp-security` (score **40**) from genuine `mcp(title)` /
`server` terms, **595 stars** (+5), and a recent push (≤90d). This is a **correct, high-signal
hit** — and a notable one: it is the first recent `mcp-security` candidate that is a real,
attachable *artifact* rather than an abstract paper, so the "is this offensive or defensive?"
ambiguity the scout flags is exactly the on-topic-by-keyword question it is designed to hand to
human triage (it judges *new and on-topic*, never *worth building*; see
[`docs/idea-scout.md`](../idea-scout.md)). The scout behaved **as designed**; no change to
`tools/idea_scout.py`.

**Action:** close [#1265](https://github.com/anthony-chaudhary/fak/issues/1265) as triaged →
**threat-surface exemplar + prior art recorded; mechanism not adopted (fak governs these calls,
it does not ship them); a curated offensive-tooling starter policy + adjudicator test corpus
named as a future witnessed-gate follow-on, not shipped here**.
