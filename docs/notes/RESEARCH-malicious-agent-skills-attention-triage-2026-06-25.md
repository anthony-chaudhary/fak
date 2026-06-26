---
title: "idea-scout triage: Locate-and-Judge — attention-located malicious-skill detection; independent validation of fak's central design choice (gate the CAPABILITY a skill exercises, do not classify its instructions) + an attention-as-audit-signal cross-link to fak's own span-attention research — prior art to cite + a real threat fak already answers a layer below the paper's scanner; the detector is NOT adopted (2026-06-25)"
description: "Triage of the idea-scout candidate arXiv:2606.23416 (Etteib, Lunghi, Bissyandé — 'Detecting Malicious Agent Skills in the Wild using Attention'): malicious marketplace skills (file-based NL-instruction packages running with user privileges) exfiltrate/hijack/persist; the paper's load-bearing claim is that prompt-injection defenses collapse here because a skill IS instructions, so the trusted/untrusted-data boundary they rely on disappears. Locate-and-Judge is a two-stage detector — a cheap locator scores skill spans by instruction-following ATTENTION and keeps top-K, a judge inspects only those — auditing a whole marketplace at order-of-magnitude lower cost than direct LLM scanning. Verdict: prior art to cite (it independently validates WHY fak gates capability rather than classifying instructions — exactly the boundary the paper shows collapses) + a real threat whose ACTION step (exfiltrate=send, hijack=destructive call, persist=write) fak's default-deny capability floor already gates a layer BELOW a static pre-load scanner, with two honest fences (fak's own leaves are build-time-checked Go, not runtime-loaded marketplace skills; and the floor gates actions, not loading). The attention-locator is NOT adopted: it is a best-effort detector (never fak's floor), it has no load surface in fak's kernel, and its core attention-is-importance premise is exactly what fak's own open research #861/#862 is still auditing."
---

# idea-scout triage — Locate-and-Judge / attention-located malicious-skill detection (issue #771)

> Closes the daily idea-scout candidate [#771](https://github.com/anthony-chaudhary/fak/issues/771)
> (`tools/idea_scout.py`, filed 2026-06-25). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — it independently validates fak's central design choice
> (gate the CAPABILITY a skill exercises; do not try to classify its instructions as
> trusted/untrusted) — AND a real threat whose action step fak's default-deny capability
> floor already gates a layer below the paper's static scanner. The attention-based
> detector is NOT adopted.**

**Source:** https://arxiv.org/abs/2606.23416 — "Detecting Malicious Agent Skills in the
Wild using Attention", Bacem Etteib, Daniele Lunghi, Tégawendé F. Bissyandé (submitted
2026-06-22). Read from the arXiv abstract via WebFetch on 2026-06-25; this is a surface
read of the abstract, not a paper audit or a reproduction (the abstract gives "high
precision" / "order-of-magnitude cost reduction" but no recall/precision/F1 figures).

## The paper, in one pass

LLM agents increasingly load **skills** — file-based packages of *natural-language
instructions* written by third parties and distributed through **marketplaces**, that
**execute with the user's privileges**. A single malicious skill can **exfiltrate data,
hijack the agent, or persist as a supply-chain foothold**, turning the marketplace into a
new attack surface for agentic systems.

The load-bearing claim is about *why this is hard*: **prompt-injection defenses do not
carry over.** Those defenses rely on a **boundary between trusted instructions and
untrusted data** — but a skill *is itself a body of instructions*, so an injected command
"sits among many legitimate ones and inherits their authority." The trusted/untrusted-data
boundary the defenses depend on simply does not exist for a skill.

The artifact, **Locate-and-Judge**, is a two-stage detector built for that regime:

1. **Locator (cheap).** Scores the **structural spans** of a skill by the
   **instruction-following attention** each span draws, and retains only the **top-K**.
2. **Judge (costly).** Examines the few retained high-attention spans in detail.

Concentrating the expensive judgment on a handful of high-attention spans lets the detector
**audit an entire marketplace instead of a sample**: an **order-of-magnitude cost
reduction** vs. direct LLM scanning at a *small cost to recall*, and it **dominates keyword
/ regex baselines** at comparable expense. Deployed at marketplace scale it flags skills
with **high precision** (majority manually confirmed malicious), **surfaced dozens of live
malicious skills** — several disguised as benign functionality, many missed by **SkillSpector
and Cisco Skill Scanner** — and the authors release the resulting labeled dataset.

## Where fak actually stands

This is, by topic, the most on-target paper the scout has surfaced for fak's *central
thesis* — and it intersects two distinct fak surfaces at once: the **extension model** (how
fak admits new capability) and an **active research line** (span-level attention as a
signal). The lens that resolves it: **fak does not classify a skill's instructions — it
gates the capability the skill exercises.** That single fact decides every row.

| The paper's frame | fak's position |
|---|---|
| A skill is a **marketplace NL-instruction package** loaded at runtime, executing with user privileges | **fak's own extension model is structurally a different thing.** A fak feature is a **leaf**: a typed, build-time-checked Go registration (`python tools/new_leaf.py`, `internal/registrations`, the additive-only `internal/abi` frozen ABI guarded by `internal/architest`). There is **no `eval` of third-party natural-language instructions** in fak's kernel and no marketplace-load surface — a leaf is reviewed code admitted at build time, not a runtime-loaded instruction file. The supply-chain *foothold* the paper targets does not exist for fak's own capabilities. (Same skill-loader mapping recorded in the [CLAWAUDIT triage](RESEARCH-agent-runtime-source-audit-triage-2026-06-24.md), #585.) |
| **Prompt-injection defenses fail** because a skill is instructions, so the trusted/untrusted-data boundary collapses | **fak never relied on that boundary** — which is exactly why the paper independently validates fak's design choice. fak's floor is **default-deny on *capability*** at `internal/adjudicator` / `internal/policy`, not an instruction-trust classifier. It does not ask "is this instruction trusted?"; it asks "is this *action* permitted?" A malicious instruction can sit among legitimate ones and inherit their authority **as text** and still buy nothing, because the floor screens the **action**, not the prose. |
| The malicious skill **exfiltrates / hijacks / persists** | These are **capabilities**, and that is precisely fak's surface. *Exfiltrate* = an outbound **send** (default-deny without the negotiated `CapA2ASend` in `internal/a2achan`; the pre-send `internal/wirescreen` redactor; the `ifc.SinkGate` egress floor in `internal/tracesink` — the same egress chain the [agentic-surveillance triage](RESEARCH-agentic-surveillance-evasion-triage-2026-06-25.md), #772, walks). *Hijack* = a **destructive call**, refused by structure at `k.Decide`. *Persist* = a **write**, gated like any other capability. fak gates the verb, however the skill phrases the request. |
| The detector **scores spans by instruction-following attention** | Lands squarely on a fak **research** line, not a defense surface. fak has open work auditing whether **witnessed span-attention is a valid importance signal** ([#861](https://github.com/anthony-chaudhary/fak/issues/861), the attention-as-explanation debate), and what confounds it ([#862](https://github.com/anthony-chaudhary/fak/issues/862), attention sinks + positional artifacts), next to the [credit-assignment-over-spans note](RESEARCH-credit-assignment-over-spans-2026-06-25.md) (#864). Locate-and-Judge's locator **inherits exactly those open questions** — its top-K span selection is an attention-as-importance claim of the kind fak's own research has not yet validated. |

So fak and the paper share the whole vocabulary — agents, skills, instructions, exfiltration
— but sit on **different layers of the same stack**. The paper is a **static, pre-load
scanner** that decides *whether to admit a skill at all*; fak is a **runtime capability
floor** that bounds *what an admitted skill can do*. They are complementary, not competing:
a Locate-and-Judge-style scanner is a reasonable *outer* admission rung, and fak's
default-deny floor is the *inner* one that holds even when the scanner's small-recall tail
lets a malicious skill through.

## The sharp, honest insight

The paper's headline — *prompt-injection defenses collapse because a skill is itself
instructions* — is the cleanest external statement of **why fak gates capability instead of
classifying instructions.** Every instruction-trust defense the paper indicts shares one
assumption: that you can draw a line between trusted instructions and untrusted data. fak's
thesis is that for an agent that loads instructions you cannot, so the line must move to the
**action**: deny by structure on the capability, and make the bytes the agent *reads*
(skill text included) **non-load-bearing** for what it *does*. The paper is concurrent,
independent evidence that the instruction-trust line is the wrong place to defend — which is
the argument fak's floor has always made.

The matching honest fence: fak's floor answers the **action** axis, not the **loading**
axis. A malicious skill that only exercises *already-permitted* capabilities, or whose harm
is purely informational within the policy's allowed surface, is **not** something the
capability floor catches — that residual is exactly what a pre-load scanner like
Locate-and-Judge covers. And the *one* place the marketplace threat is live for fak's own
operation is the Claude-Code fleet that loads in-repo `.claude/skills/` SKILL.md packages;
there the mitigation today is **provenance** (the skills are in-tree, version-controlled,
team-authored — not marketplace-sourced), not a scanner.

## Triage decision

- **Adopt Locate-and-Judge as a fak capability? No — on three independent grounds.**
  (1) **No load surface.** fak's extension model is build-time-checked Go leaves, not
  runtime-loaded marketplace skills; there is no in-kernel skill-load point for a pre-load
  scanner to sit on. (2) **It is a detector, and a detector is never fak's floor.** fak's
  standing stance — recorded in the [Tool-Guard triage](RESEARCH-tool-guard-isolated-planning-triage-2026-06-23.md)
  (#527) — is that a best-effort classifier is evadable by design and is a *rung*, never the
  floor (the floor is the capability lock + structural quarantine). Wiring an attention
  detector in *as the defense* would contradict that thesis. (3) **The premise is exactly
  what fak's own research is still auditing.** The locator rests on attention-as-importance,
  the open question in [#861](https://github.com/anthony-chaudhary/fak/issues/861) /
  [#862](https://github.com/anthony-chaudhary/fak/issues/862); fak should not ship an
  attention-importance claim its own work has not yet validated.
- **Defend against (is the threat real for fak)? Real — and fak already answers its action
  step a layer below the paper's scanner, with two honest fences.** The exfiltrate / hijack
  / persist actions are gated by the default-deny capability floor regardless of how the
  malicious instruction is phrased or where it hides among legitimate ones — that is the
  whole point of gating capability rather than classifying instructions. The fences worth
  not overclaiming: (a) the floor gates **actions, not loading** — a skill confined to
  already-permitted capabilities is the residual a pre-load scanner covers; (b) fak's own
  capabilities are **build-time Go leaves**, so the marketplace supply-chain surface is not
  live for the kernel, and for the fleet's `.claude/skills/` it is mitigated by **provenance**,
  not a scanner.
- **Cite as prior art? Yes — strongly, on two axes.** First, it is the cleanest external
  statement that **the instruction-trust boundary collapses for skills**, which is the
  independent justification for fak's choice to **gate capability, not classify
  instructions** — it belongs alongside the [CLAWAUDIT](RESEARCH-agent-runtime-source-audit-triage-2026-06-24.md)
  "the runtime IS the boundary" prior art (#585) and the [Tool-Guard](RESEARCH-tool-guard-isolated-planning-triage-2026-06-23.md)
  description-poisoning dual (#527), in the spirit of `CLAIMS.md`'s `[0/29 novel — the
  contribution is the assembly]` prior-art discipline. Second, it is a real-world data point
  that **span-level attention is usable as an audit signal**, a direct cross-link for fak's
  span-attention research (#861/#862/#864) — with #861's validity caveat applying squarely
  to the locator.

**Action:** close #771 as triaged → **prior art cited (independent validation of fak's
gate-the-capability-not-the-instruction design + an attention-as-audit-signal cross-link) +
a real threat whose action step fak's default-deny floor already gates a layer below a
static pre-load scanner, with the build-time-leaf and actions-not-loading fences named; the
attention detector is explicitly not adopted** (this note). No code change in this
increment: `tools/idea_scout.py` surfaced and scored the candidate correctly (topic
`prompt-injection-defense`, score 50 — a real, on-topic, high-relevance hit), and the right
small artifact for a research/security triage is the recorded verdict + the cross-links, not
an attention detector fak has no load surface for and whose premise its own research has not
yet validated.

**Next step (the smallest honest follow-on, if pursued):** scope a **skill-provenance note
for the fleet's own `.claude/skills/`** — the single place the paper's marketplace threat is
live for fak — stating that fleet skills are in-tree / reviewed (provenance = the
mitigation) and that any *third-party* skill would need a pre-load scan (Locate-and-Judge as
the outer admission rung) *before* fak's runtime capability floor (the inner one) ever sees
its calls. Filed as its own scoped issue with a fixture skill as its witness, not built in
this triage increment.
