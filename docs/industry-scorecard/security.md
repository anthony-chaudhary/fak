---
title: "fak industry scorecard — security"
description: "The security dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# security — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Security & safety (`security`)

### ▼ Prompt-injection defense and agent security (attack-success-rate vs utility-under-attack) — fak: **trails**

*Why it matters:* An agent that calls tools and reads untrusted content is an exfiltration surface; indirect prompt injection is the top agent-security risk. Buyers now require an ASR-vs-utility number, not a vibe, and adaptive attacks routinely break naive defenses, so the bar is benchmark-grounded.

- **SOTA bar:** On AgentDojo, design-by-construction defenses now reach provable security: CaMeL drives successful indirect-prompt-injection attacks to 0 (vs 8 for the next-best tool-filter) while still solving 77% of tasks (vs 84% undefended); MELON (ICML'25) likewise beats prior SOTA defenses. Undefended best agents still see ASR <25%.
- **Leading systems:** CaMeL (Defeating Prompt Injections by Design, 2025), MELON (ICML'25), Meta SecAlign (secure foundation LLM), tool-filter / SecAlign baselines
- **Source:** [https://arxiv.org/abs/2503.18813](https://arxiv.org/abs/2503.18813) (2025-03)
- **fak:** trails — no number (shipped)
- **fak note:** fak's OWN repeated concession: the detector these drivers feed is '~100% evadable on a SOTA evasion battery' and FP-prone on private real transcripts (it sealed 2/59 benign pages as false positives). Detection is DELIBERATELY non-load-bearing — fak's security guarantee is the STRUCTURAL capability floor + containment (which never run the detector), not a detection rate. This TRAILING row carries the honest concession so the security story is never read as a detection-rate win.
- **Trace:** CLAIMS.md (security-substrate ceiling) · internal/agentdojo

### ≈ Tool/agent sandboxing, structural containment, and PII/exfil prevention — fak: **parity**

*Why it matters:* Detectors are evadable, so the durable control is architectural: bound the blast radius so a successful injection still cannot read secrets, write outside scope, or exfiltrate PII. Structural containment is the property that survives an adaptive attacker and that security-conscious buyers actually demand.

- **SOTA bar:** Capability/IFC-based structural containment is the 2025 SOTA: CaMeL (Defeating Prompt Injections by Design) extracts control/data flow so untrusted data can never alter program flow and enforces capability policies at tool-call time (0 successful AgentDojo attacks); MELON adds masked re-execution + tool-comparison provable defense.
- **Leading systems:** CaMeL (capability + control/data-flow IFC, 2025), MELON (masked re-execution, ICML'25), container/microVM sandboxes, IFC / capability allow-lists
- **Source:** [https://arxiv.org/abs/2503.18813](https://arxiv.org/abs/2503.18813) (2025-03)
- **fak:** parity — no number (shipped)
- **fak note:** fak's genuine security strength and arguably its second differentiator after reuse, but PARITY not lead because there is no audited containment leaderboard and the comparison is capability-presence, not a measured ASR-under-containment number. fak ships layered structural controls that hold even when the detector is fooled: a declarative default-deny capability allow-list (closed 12-reason vocab), IFC taint that sink-gates tainted->sink flows pre-call (FIDES/CaMeL-class), kernel-authored provenance (trust taken from the model), plan-CFI, context-MMU quarantine/page-out, and a self-signed provable-deletion certificate + byte-exact causal invalidation. This is the MELON/IFC/capability-allow-list family. Honest fences: self-signed v1 receipts, EvictedCount self-reported, no disposable-microVM tool sandbox, no third-party containment audit. Parity on the layered-containment posture; no number to claim lead.
- **Trace:** CLAIMS.md Adjudication (default-deny manifest, closed 12-reason vocab), Security substrate (IFC Ref.Taint sink-gated, kernel-authored provenance, plan-CFI), Context-MMU (quarantine/page-out), provable-deletion certificate (internal/deletioncert), causal-invalidation witness (commit 0fc39aa)

