---
title: "idea-scout triage: ShareLock — a multi-tool THRESHOLD tool-poisoning attack (Shamir-shared payload dispersed across MCP tool descriptions, reconstructed on a server-update trigger) that provably defeats tool-description-BASED detectors. Verdict: prior art to cite + a real threat whose CONSEQUENCE fak's capability floor already contains; ShareLock is the cleanest external proof that a tool-description detector cannot be the load-bearing floor. The Shamir-sharing attack mechanism is not adopted; no kernel change (2026-06-26)"
description: "Triage of the idea-scout candidate arXiv:2606.27027 (Liwei Liu, Tianzhu Han, Zijian Liu, Zishu Dong, Na Ruan — 'ShareLock: A Stealthy Multi-Tool Threshold Poisoning Attack Against MCP'): a Tool Poisoning Attack (TPA) that fragments its malicious instruction with Shamir's threshold secret-sharing across MULTIPLE innocent-looking MCP tool descriptions, reconstructing the hidden instruction only on a trigger (a server update), claiming information-theoretic stealth, >90% success, and outperformance of single-tool poisoning against tool-description-based detectors. Verdict: prior art to cite + a real threat already contained at the structural floor — ShareLock attacks the DETECTION layer fak deliberately keeps OFF the load-bearing path; its core move (disperse to evade a description scanner) gains nothing against a floor that never reads the descriptions for a verdict and gates the resulting ACTION (default-deny capability lock) and RESULT (result-admit quarantine) instead. Named residual: fak does not screen inbound third-party tool descriptions, and a poisoned description can still induce a POLICY-ALLOWED call with malicious arguments — the standing residual of any capability floor, mitigated (not eliminated) by IFC taint + egress sink-gating. The Shamir-sharing mechanism is not adopted; the smallest honest follow-on is to treat a downstream tool-definition CHANGE as a witnessed provenance event (the reconstruction trigger fires on server update), filed against the witness/change-feed seam. No change to tools/idea_scout.py — the scout surfaced and scored the candidate correctly (topic mcp-security, score 53)."
---

# idea-scout triage — ShareLock / multi-tool threshold tool-poisoning against MCP (issue #911)

> Closes the daily idea-scout candidate [#911](https://github.com/anthony-chaudhary/fak/issues/911)
> (`tools/idea_scout.py`, filed 2026-06-26). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + a real threat whose CONSEQUENCE the capability floor
> already contains. ShareLock attacks the tool-description DETECTION layer fak
> deliberately keeps OFF the load-bearing path; its core innovation (disperse a poison
> across many tools to evade a description scanner) is neutralized-by-construction against
> a floor that never reads the descriptions for a verdict. The Shamir-sharing mechanism is
> not adopted; no kernel change.**

**Source:** https://arxiv.org/abs/2606.27027 — "ShareLock: A Stealthy Multi-Tool Threshold
Poisoning Attack Against MCP", Liwei Liu, Tianzhu Han, Zijian Liu, Zishu Dong, Na Ruan
(submitted 2026-06-25). Read from the arXiv abstract via WebFetch on 2026-06-26; this is a
surface read of the abstract, not a paper audit or a reproduction — the reported numbers
below are as the abstract states them.

## The paper, in one pass

A **threat** paper, in the Tool Poisoning Attack (TPA) family. Its premise: existing
poisoning schemes use a **monolithic plaintext** paradigm — the malicious instruction sits,
whole, inside one tool description — which both manual inspection and automated
tool-description detectors can catch. ShareLock's contribution is to make the poison
**threshold-shared and multi-tool**:

1. The malicious payload is fragmented with **Shamir's (t, n) threshold secret-sharing**
   into `n` shares, each embedded as an **innocent-looking fragment** in a *different* tool
   description. Any subset smaller than the threshold `t` is **information-theoretically**
   uninformative — no individual tool description reveals the attack intent.
2. The shares recombine only when a **reconstruction trigger** activates — the abstract
   names a **server update** — at which point the dispersed fragments reassemble into the
   hidden instruction.
3. The design buys two things the abstract claims: **stealth** (no single description is
   suspicious, so per-tool pattern matching has nothing to flag) and **fault tolerance**
   (the threshold scheme tolerates some shares being absent). It reports **>90% success**
   and **significant outperformance of single-tool poisoning in tool-description-based
   detection** — i.e. it is specifically engineered to defeat description scanners.

The abstract proposes **no defense** — it is an attack-construction + evaluation paper.

## Where fak actually stands

The lens that resolves this paper cleanly is fak's **detector-is-not-the-floor** thesis,
stated verbatim in `CLAIMS.md` ("Honest ceiling, surfaced not hidden: the *detector* …
is ~100% evadable on a SOTA evasion battery … Detection is **deliberately
non-load-bearing** — the structural guarantee is the capability floor + containment, which
never run the detector"). ShareLock is a rigorous, **information-theoretic** instance of
exactly the evasion that line assumes: it does not merely *empirically* slip a scanner, it
*provably* makes the per-description scanner see nothing below the threshold. That makes it
the strongest external evidence to date for **why fak puts the floor on the ACTION, not on
the description text.**

The load-bearing observation: **ShareLock attacks the detection layer; fak's floor is not a
detector.** ShareLock's whole innovation is *disperse-to-evade-a-scanner*. But fak's
capability floor never reads a tool description to reach a verdict — it gates the **call**
against a declarative, default-deny **policy manifest** (which tools the agent may call,
authored as a reviewable file, not a description heuristic). So dispersing the poison across
`n` descriptions gains the attacker **nothing against the floor**: there is no
description-scanner to disperse around. Whatever instruction the shares reconstruct, the
model can only act on it by **emitting a tool call**, and that call faces the same gate
regardless of how cleverly the inducing prompt was assembled.

| ShareLock's frame | fak's position |
|---|---|
| Poison is **dispersed across many tool descriptions** to evade per-tool description detection | fak's floor reads the **call**, not the description. There is no per-description scanner on the load-bearing path to disperse around — the dispersal move is **neutralized by construction**. (`CLAIMS.md` capability-floor units 15, 29.) |
| The reconstructed payload is a **hidden instruction** that induces malicious behavior | An instruction can only act by becoming a **tool call**, and the call is **default-deny** gated against the policy manifest: an unlisted/dangerous call is `Deny(DEFAULT_DENY)`. The poison may *speak*; the action is gated. `fak guard -- claude` arms exactly this floor around a real session (`CLAIMS.md` guard front-door unit). |
| Stealth is **information-theoretic** — provably no signal below threshold | fak does not *contest* the stealth claim — it concedes detectors are ~100% evadable and routes around them. Detection (`ctxmmu`/`normgate`/`wirescreen`) is an **additive, never load-bearing** rung; ShareLock is precisely why it is kept off the floor. |
| Injected instructions that arrive **as data the model reads** | When such content arrives as a tool **result**, the **result-admit gate** quarantines prompt-injection/poison-shaped payloads out of context, raising the session taint high-water mark so a later egress is gated (`CLAIMS.md` result-admit unit 48; durable-moat re-screen unit 67). |
| Trigger fires **on a server update** (the tool surface *changes*) | A changed tool surface is a **provenance event** — exactly the world-state shift fak's witness / change-feed machinery exists to make legible (`fak_refute_witness`, `fak_changes`). fak does not *yet* re-adjudicate on a description change (descriptions are not on the call path), so this is named as the follow-on below, not a shipped defense. |

## The honest residual — name it, don't paper over it

The capability floor contains the **consequence** of a reconstructed instruction; it does
**not** intercept the **injection** itself, and it must not be claimed to. The fences worth
not regressing on:

1. **fak does not screen inbound third-party tool descriptions.** In `fak guard -- claude`
   passthrough, a downstream MCP server's advertised descriptions reach the model verbatim;
   fak adjudicates the *calls* the model then makes, not the advertised description text.
   The gateway's own `tools/list` advertises fak's adjudication primitives
   (`fak_adjudicate` / `fak_execute` / …), not a sanitized re-statement of downstream tools.
2. **A poisoned description can still induce a POLICY-ALLOWED call with malicious
   arguments.** This is the standing residual of *any* capability floor: the floor gates
   *which tool* (and the grammar rung gates the call *shape*), but a semantically-malicious
   call to an *allowed* tool — "read this (sensitive) file", a benign tool bent to a harmful
   end — is not denied by the manifest alone. fak's mitigations that **bite** this residual
   are containment, not interception: **IFC taint + egress sink-gating** (once tainted or
   sensitive bytes are in context, the exfil egress is gated) and **result quarantine**. They
   **reduce** the blast radius; they do not make the injection a non-event.
3. **Adding a tool-description scanner is the wrong fix** — ShareLock *is the proof*. A
   description detector would re-import the ~100% evadable ceiling fak keeps off the
   load-bearing path, and ShareLock defeats exactly that class with an information-theoretic
   guarantee. The right posture is the one fak already holds: floor on the action, contain
   the result, treat detection as additive.

## Triage decision

- **Adopt the Shamir-sharing attack mechanism as a fak capability?** **No.** ShareLock is an
  attack construction, not a capability. There is nothing to fold into the kernel.
- **Defend against (is the threat real for fak)?** **Real — and the consequence is already
  contained at the structural floor, with named residuals.** The reconstructed instruction
  must still produce a tool call, which the default-deny capability floor gates; injected
  content arriving as a result is quarantined; the dispersal-to-evade innovation buys the
  attacker nothing against a floor that never reads the descriptions. The residual (no
  inbound-description screen; a poisoned description can induce a policy-*allowed* malicious
  call, contained but not intercepted by IFC taint + egress gating) is named above, not
  hidden. **No new defense in this increment** — adding a description scanner would be the
  anti-pattern ShareLock itself refutes.
- **Cite as prior art?** **Yes.** ShareLock is the cleanest external, **information-theoretic**
  proof of fak's *detector-is-not-the-floor* thesis: a poison that *provably* defeats
  tool-description detection. It belongs in fak's prior-art lineage alongside
  [defensive misdirection](RESEARCH-defensive-misdirection-triage-2026-06-23.md),
  [the probe-AUC evaluation-protocol triage](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md),
  and [MCPPrivacyDetector](RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md) — each an
  external result that the *detector* is evadable and the structural floor must carry the
  guarantee. ShareLock sharpens it from "empirically evadable" to "provably evadable by
  construction."

**Scout calibration (no code change).** The candidate surfaced under `mcp-security`
(score 53) on the terms *model context protocol / mcp (title) / tool poisoning / server*
plus a freshness bonus (≤30d). This is the scout **working as designed** — it is a genuine,
on-topic MCP-security threat, surfaced and scored correctly, and handed to human triage. No
change to `tools/idea_scout.py`.

**Action:** close #911 as triaged → **prior art cited + a real threat whose consequence the
capability floor already contains; the dispersal-to-evade mechanism is neutralized against a
floor that gates the action, not the description; the Shamir-sharing attack is not adopted**
(this note). No code change in this increment: the right small artifact for a research /
security triage is the recorded verdict + the named residuals, not a half-built
description-scanner that ShareLock itself refutes.

**Next step (the smallest honest follow-on, if pursued):** treat a downstream **tool-definition
change** as a **witnessed provenance event** — ShareLock's reconstruction trigger fires *on a
server update*, so the advertised tool surface changing between sessions is exactly the
world-state shift fak's `fak_refute_witness` / `fak_changes` seam already makes legible. The
scoped increment is to pin the advertised tool surface as a witness and surface a
re-adjudication point when it changes, filed against the witness/change-feed seam — a
**provenance** signal (the surface changed), never a description **scanner** (which ShareLock
proves is evadable). It guards the invariant this note names rather than importing the
attack's evasion target.
