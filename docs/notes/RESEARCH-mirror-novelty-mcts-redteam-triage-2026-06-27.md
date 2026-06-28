---
title: "idea-scout triage: 'MIRROR: Novelty-Constrained Memory-Guided MCTS Red-Teaming for Agentic RAG' — an OFFENSIVE cross-surface attack GENERATOR (memory-guided MCTS conditioned on retrieved context under a deterministic Novelty Gate) for multimodal agentic RAG: text poisoning, image injection, direct-query, orchestrator-level tool manipulation. Triple verdict, the same shape as #909/#911: (1) prior art to cite — it is the SOTA academic form of the adaptive/generative red-team fak's own internal/agentdojo + examples/agentdojo-redteam already embody at a simpler level (3-form paraphrase expander), and it answers the open follow-on from the #909 out-of-band triage by NAME: make fak's ASR battery adaptive, cross-surface, and novelty-constrained, not a static injection set; its deterministic Novelty Gate is the formal sibling of fak's own idea_scout dedup/novelty rungs. (2) A real threat whose CONSEQUENCE the structural floor already contains — MIRROR's whole advantage is NOVELTY to evade template/pattern detectors (it measures 73-84% exact duplication in prior red-teaming and beats it), but fak's full-stack defense taints by PROVENANCE not content, so the egress/destructive sink is barred 'regardless of phrasing' (agentdojo.go ASR full-stack==0) — the novelty axis is invariant against the floor by construction, exactly as ShareLock (#911). (3) Mechanism NOT adopted as a kernel component — fak is a defense; the one-binary zero-dep kernel does not generate attacks (no MCTS, no memory store, no LLM-conditioned candidate generator on the request path). The methodology IS the evaluation upgrade for the red-team battery (the smallest honest follow-on). Honest fence carried from #909: cite MIRROR as a protocol/methodology and a threat-the-floor-contains-by-construction, NEVER as evidence fak's containment is adaptively robust (fak has no measured cross-surface adaptive ASR-under-containment number). (2026-06-27)"
description: "Triage of idea-scout candidate arXiv:2606.26793 (Inderjeet Singh, Andrés Murillo, Motoyoshi Sekiya, Yuki Unno, Junichi Suga — 'MIRROR: Novelty-Constrained Memory-Guided MCTS Red-Teaming for Agentic RAG', submitted 2026-06-25). MIRROR is a unified, cross-surface OFFENSIVE red-teaming framework for MULTIMODAL agentic RAG: it argues the attack surface goes beyond prompt injection to text poisoning, image injection, direct-query attacks, and orchestrator-level tool manipulation; that existing red-teaming is surface-specific and recycles known templates (73-84% exact duplication on text-poisoning benchmarks); and presents a memory-guided Monte Carlo tree search that conditions candidate generation on retrieved context under an explicit novelty constraint, with a deterministic Novelty Gate that rejects any candidate matching the retrieval set under normalized comparison. Verdict (three-part, the shape of the #909 and #911 triages): (1) PRIOR ART TO CITE — it is the academic SOTA for the adaptive, generative red-team that fak already ships in a simpler form (internal/agentdojo's {plain,obfuscated,paraphrased} adaptivity axis + examples/agentdojo-redteam's marker-free generative expander), and it answers BY NAME the open follow-on the #909 out-of-band triage recorded: turn fak's static AgentDojo-style battery into an adaptive, defense-aware, CROSS-SURFACE one. Its deterministic Novelty Gate is the formal sibling of fak's own idea_scout dedup/novelty rungs (seen-cache + Jaccard near-dup rejection) and of the red-team expander's no-marker-word paraphrase goal; its four-surface taxonomy is a coverage map for which surfaces the battery should expand to test (image quarantine #290; orchestrator/tool-manipulation = the gateway/MCP seam). (2) A REAL THREAT WHOSE CONSEQUENCE THE CAPABILITY FLOOR + QUARANTINE ALREADY CONTAIN — MIRROR's entire edge is NOVELTY engineered to evade template/pattern-matching detectors, but fak's full-stack defense taints by PROVENANCE not content: reading untrusted content taints the session and the IFC sink-gate bars the attacker's egress/destructive sink 'regardless of phrasing' (internal/agentdojo/agentdojo.go: ASR(full-stack)==0 vs ASR(detection-only)>0), so a maximally-novel injection that defeats every detector still cannot make a denied capability allowed or a tainted->sink flow pass — the novelty axis is invariant against the structural floor by construction, the same outcome as ShareLock (#911). The novelty advantage targets the detection-only layer fak deliberately keeps OFF the load-bearing path (~100% evadable, by design). (3) MECHANISM NOT ADOPTED — fak is a defense; the kernel does not generate attacks (no MCTS / memory / LLM-conditioned generator on the request path); MIRROR is not imported. The SMALLEST honest follow-on is the evaluation upgrade: expand internal/agentdojo's adaptivity axis toward a novelty-constrained, cross-surface generator and add the image / orchestrator surfaces MIRROR enumerates, turning the #909 'no measured adaptive ASR-under-containment number' fence into a tracked agentdojo-lane task. Honest fence (carried from #909): cite as a protocol to adopt for evaluation and as a threat the floor contains by construction, NEVER as evidence fak's containment is adaptively robust — fak has no measured cross-surface adaptive ASR-under-containment number, and the image-injection / orchestrator surfaces have no such number either. No change to tools/idea_scout.py — surfaced and scored correctly (topic prompt-injection-defense, score 50)."
---

# idea-scout triage — MIRROR: novelty-constrained memory-guided MCTS red-teaming for agentic RAG (issue #1007)

> Closes the daily idea-scout candidate [#1007](https://github.com/anthony-chaudhary/fak/issues/1007)
> (`tools/idea_scout.py`, filed 2026-06-27). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict (three-part): (1) prior art to cite — MIRROR is the SOTA academic form of the
> adaptive, generative red-team fak already ships in simpler form, and it answers BY NAME
> the open follow-on the [#909 out-of-band triage](RESEARCH-out-of-band-injection-defense-taxonomy-triage-2026-06-26.md)
> recorded: make fak's AgentDojo-style ASR battery adaptive, defense-aware, and
> CROSS-SURFACE, not a static injection set. (2) A real threat whose CONSEQUENCE the
> capability floor + quarantine already contain — MIRROR's whole edge is NOVELTY to evade
> template/pattern detectors, but fak's full-stack defense taints by PROVENANCE not content
> (`agentdojo.go`: ASR full-stack == 0 regardless of phrasing), so the novelty axis is
> invariant against the structural floor by construction, exactly as [ShareLock #911](RESEARCH-sharelock-multitool-threshold-poisoning-triage-2026-06-26.md).
> (3) Mechanism NOT adopted — fak is a defense; the one-binary zero-dep kernel does not
> generate attacks. The methodology is the evaluation upgrade for the red-team battery, not
> a kernel feature. Honest fence carried from #909: cite as a protocol and a
> contained-by-construction threat, NEVER as evidence fak's containment is adaptively
> robust (no measured cross-surface adaptive ASR number exists).**

**Source:** https://arxiv.org/abs/2606.26793 — "MIRROR: Novelty-Constrained
Memory-Guided MCTS Red-Teaming for Agentic RAG", Inderjeet Singh, Andrés Murillo,
Motoyoshi Sekiya, Yuki Unno, Junichi Suga (submitted 2026-06-25). Read from the arXiv
abstract surfaced by the scout (the abstract is truncated in the issue body at the
Novelty-Gate sentence); this is a surface read of the abstract, not a paper audit or a
reproduction. The abstract gives the threat-surface taxonomy, the recycled-template
critique with a measured duplication figure, the MCTS+memory+retrieval-conditioning
method, and the deterministic Novelty Gate — not the per-surface ASR tables, the search
budget, or the attack code.

## The paper, in one pass

MIRROR is an **offensive** red-teaming framework, and it makes three moves:

1. **Widen the surface.** Multimodal agentic RAG is attackable on more than prompt
   injection: **text poisoning** (poison the retrieval corpus), **image injection**
   (adversarial content in a retrieved/visual modality), **direct-query attacks**
   (the user turn itself), and **orchestrator-level tool manipulation** (subvert the
   agent's tool-routing / planning layer). Existing red-teams are **surface-specific**.
2. **Name the staleness.** Existing red-teaming **recycles known templates** — measured
   at **73-84% exact duplication** on text-poisoning benchmarks. A red-team that repeats
   itself overstates a defense's robustness for the same reason a static fixture does.
3. **Generate novel attacks under search + memory + a hard novelty constraint.** A
   **memory-guided Monte Carlo tree search** explores the attack space while
   **conditioning candidate generation on retrieved context**, and a **deterministic
   Novelty Gate rejects any candidate that matches the retrieval set under normalized
   comparison** — so the battery cannot fall back into recycled templates.

The contribution is a **unified, cross-surface, novelty-forcing attack GENERATOR**, not a
defense.

## Where fak actually stands

fak already ships the *simpler* version of exactly this idea, and says so in code. The
relevant seam is `internal/agentdojo` — fak's dynamic AgentDojo-style ASR battery — whose
package doc states the premise MIRROR formalizes, almost verbatim:

> *"A defense that passes a fixed corpus can still fail catastrophically against an
> ADAPTIVE attacker who rephrases the payload to evade the very patterns the corpus tested
> — which is precisely the gap AgentDojo was built to measure. So a green poison.json is
> necessary, not sufficient."* — `internal/agentdojo/agentdojo.go`

| MIRROR's move | The shipped fak seam at the same altitude |
|---|---|
| **Generate fresh attacks instead of replaying templates** | `internal/agentdojo` runs a `{vector × adaptivity}` matrix where every attack appears in a **PLAIN**, an **OBFUSCATED**, and a **PARAPHRASED (semantic, no marker word)** form; `examples/agentdojo-redteam` is the runnable face — a generative expander that emits **marker-free rephrasings** and scores them by ASR. MIRROR is the **SOTA generalization** of this expander: MCTS + memory + retrieval-conditioning instead of a fixed paraphrase set. |
| **Deterministic Novelty Gate — reject a candidate matching the retrieval set under normalized comparison** | The same shape as fak's own `tools/idea_scout.py` **dedup/novelty rungs** (durable seen-cache + **Jaccard near-dup rejection** against existing titles/bodies) and the red-team expander's **no-marker-word** paraphrase goal — reject anything that is not *new* under a normalized comparison. fak already runs a deterministic novelty gate, in a different domain. |
| **Image-injection surface** | `internal/model/multimodal.go` (#290): image-bearing input is **fail-closed** — quarantined until the caller explicitly opts into the rollout mode (`MultimodalModeQuarantine`, `MultimodalDecision` allow/quarantine/deny). The surface MIRROR attacks is one fak **governs**, but with **no measured ASR** against an adaptive image-injection generator. |
| **Orchestrator-level tool manipulation** | The agent's tool calls cross fak's **default-deny capability floor** (`internal/adjudicator`, `internal/policy`); a manipulated orchestrator still emits a **tool call**, which is adjudicated against the manifest. Contained at the **action**, but again **no measured cross-surface adaptive ASR** number. |
| **Text poisoning of the retrieval corpus** | Context-MMU **result-admit quarantine** + IFC `Ref.Taint` sink-gating hold a poisoned retrieved result **non-load-bearing** (the [#909 out-of-band](RESEARCH-out-of-band-injection-defense-taxonomy-triage-2026-06-26.md) and [#911 ShareLock](RESEARCH-sharelock-multitool-threshold-poisoning-triage-2026-06-26.md) mappings). |

## The sharp, honest insight

MIRROR moves fak in two directions at once — the same two-sided pattern as the #909 triage.

- **Its novelty advantage is invariant against fak's structural floor — by construction.**
  MIRROR's *entire* edge is engineering **novelty** so a candidate does not match any
  known template — i.e. defeating **pattern / template / detector** matching (the explicit
  point of measuring and beating 73-84% template duplication). But fak's full-stack
  defense does not classify the **content** of the injection at the load-bearing seam: it
  taints by **PROVENANCE**. `internal/agentdojo/agentdojo.go` records the result directly —
  with IFC engaged, **every** attack that reads untrusted content taints the session by
  provenance, so *"the attacker's egress/destructive SINK is barred regardless of phrasing
  — ASR == 0,"* while the **detection-only** rate is `> 0` because a lexical gate cannot be
  complete. A maximally-novel MIRROR candidate that defeats every detector still cannot
  make a **denied capability allowed** or a **tainted→sink** flow pass. So MIRROR's novelty
  axis lands entirely on the **detection-only** layer fak **deliberately keeps off the
  load-bearing path** (the `security.md` row: detection is *"~100% evadable on a SOTA
  evasion battery,"* **non-load-bearing by design**). This is the **same outcome as
  ShareLock (#911)**: an attack that optimizes against the *detector* is neutralized
  against a floor that gates the **action** and the **result**, not the string.

- **It raises the bar fak's OWN evaluation must clear — and it is the named answer to the
  #909 follow-on.** The [#909 out-of-band triage](RESEARCH-out-of-band-injection-defense-taxonomy-triage-2026-06-26.md)
  recorded one open follow-on: fak's static AgentDojo-style battery is *exactly* the
  methodology the literature cautions against, and fak should run an **adaptive
  (defense-aware)** evaluation, not a fixed injection set. MIRROR is the **concrete,
  cross-surface, novelty-forcing generator** that follow-on asked for. It hands fak the
  missing acceptance protocol: a battery whose attacks are **searched (MCTS), memory-
  guided, retrieval-conditioned, novelty-gated, and span all four surfaces** — the bar an
  out-of-band defense should be measured at. The matching honest fence travels with the
  citation: fak has **no measured cross-surface adaptive ASR-under-containment number**
  (the standing `security.md` fence), and the **image-injection** (#290) and
  **orchestrator** surfaces have no such number either — so MIRROR is a **protocol to adopt
  for evaluation and a threat the floor contains by construction**, **never** evidence that
  fak's containment is adaptively robust.

## Triage decision

- **Adopt a new capability? No — fak is a DEFENSE; the kernel does not generate attacks.**
  MIRROR is an offensive generator (MCTS + a memory store + an LLM-conditioned candidate
  producer). None of that belongs on the request path of a one-binary, zero-dep kernel
  (`OUT_OF_DIRECTION` would refuse a generator-in-the-hot-path anyway). The one thing worth
  importing is **evaluation methodology**, recorded below as a follow-on — not a kernel
  feature. (Same disposition as the #909 paper: *describes / strengthens* fak's evaluation,
  is not a mechanism to fold into the kernel.)
- **Defend against (is this a threat)? Yes — and the consequence is already contained.**
  MIRROR is a genuine cross-surface threat *generator*, but its novelty advantage is aimed
  at the **detection** layer fak keeps off the load-bearing path; the **action** (default-
  deny capability floor) and the **result** (IFC `Ref.Taint` sink-gating + Context-MMU
  result-admit quarantine) bar the attacker's sink **regardless of phrasing**
  (`agentdojo.go` ASR full-stack == 0). **Named residual:** *containment* is not *a measured
  number* — fak has no adaptive cross-surface ASR-under-containment metric, and the
  image-injection (#290) / orchestrator surfaces are governed (fail-closed quarantine /
  adjudicated tool call) but **unmeasured against an adaptive generator**. That residual is
  the follow-on below, not a hole in this increment.
- **Cite as prior art? Yes — on two grounds.** (1) It is the **SOTA academic form of fak's
  own dynamic red-team** (`internal/agentdojo` + `examples/agentdojo-redteam`) and the
  **named answer** to the #909 follow-on (make the battery adaptive, defense-aware, and
  cross-surface). (2) Its **deterministic Novelty Gate** is the formal sibling of fak's own
  `idea_scout` dedup/novelty rungs and the red-team expander's no-marker-word goal — a
  worked external instance of "reject anything not new under a normalized comparison."

**Action:** close #1007 as triaged → **prior art cited (the SOTA adaptive, cross-surface,
novelty-constrained red-team GENERATOR that fak's own `internal/agentdojo` /
`examples/agentdojo-redteam` battery already embodies in simpler form, and the named answer
to the [#909](RESEARCH-out-of-band-injection-defense-taxonomy-triage-2026-06-26.md)
follow-on) + a real cross-surface threat whose consequence the structural floor already
contains by construction (novelty defeats detectors, not the provenance-tainting capability
floor + quarantine — `agentdojo.go` ASR full-stack == 0 regardless of phrasing, the same
outcome as [ShareLock #911](RESEARCH-sharelock-multitool-threshold-poisoning-triage-2026-06-26.md));
mechanism not adopted, fak is a defense and the kernel does not generate attacks** (this
note). No code change in this increment: `tools/idea_scout.py` surfaced and scored the
candidate correctly (topic `prompt-injection-defense`, score 50 — prompt injection / tool /
agent in title, fresh ≤30d), and the right small artifact for a research/security triage is
the recorded verdict + the surface-by-surface mapping, not a feature import for a generator
fak (a defense) should not host.

**Next step (the smallest honest follow-on, if pursued):** upgrade fak's evaluation, not the
kernel. Expand `internal/agentdojo`'s adaptivity axis from the fixed `{plain, obfuscated,
paraphrased}` set toward a **novelty-constrained, cross-surface** generator (MIRROR's
Novelty-Gate discipline), and add the **image-injection** (#290 `internal/model/multimodal`)
and **orchestrator-level tool-manipulation** (gateway/MCP) surfaces MIRROR enumerates — so
the battery measures an **adaptive cross-surface ASR-under-containment** number and turns the
standing `docs/industry-scorecard/security.md` "no measured adaptive ASR" fence (also the
open residual of the #909 triage) into a tracked task. Filed as its own **agentdojo-lane**
change (which owns the battery), not built in this triage increment; carried with the fence
that fak must cite MIRROR as a *methodology + a contained threat*, never as a robustness
proof it has not measured.
