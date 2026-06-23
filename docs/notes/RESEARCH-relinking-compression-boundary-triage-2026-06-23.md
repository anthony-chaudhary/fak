---
title: "idea-scout triage: relinking at the compression boundary — fak is structurally immune because its context compression is EXTRACTIVE (verbatim, FaithfulnessProbe=1.0), not abstractive, and its floor gates the resulting CALL not the pre-compression prompt; cite as prior art + a named invariant guarding future generative summarizers (2026-06-23)"
description: "Triage of the idea-scout candidate arXiv:2606.21732 (Liu, Zhang, She, 'Safe to Check, Unsafe to Use: Relinking at the Compression Boundary of LLM Agents'): summarization-based prompt compression shifts the security boundary — filters inspect the pre-compression prompt while the backend acts on a newly generated compressed context — letting a confused-deputy compressor reconvene distributed, locally benign fragments into a complete malicious instruction (relinking). Verdict: prior art to cite + a real threat fak is structurally immune to on two independent grounds (the floor gates the resulting call after compression, not the pre-compression prompt; fak's context compression is extractive-by-construction with a load-bearing FaithfulnessProbe=1.0, so it cannot author an instruction absent from every source fragment). KBRA/Relink not adopted. One named invariant recorded: the immunity is contingent on FaithfulnessProbe staying load-bearing — a future model-backed abstractive summarizer must re-screen its output at the result-admit boundary, never be trusted as a faithful view."
---

# idea-scout triage — relinking at the compression boundary (issue #554)

> Closes the daily idea-scout candidate [#554](https://github.com/anthony-chaudhary/fak/issues/554)
> (`tools/idea_scout.py`, filed 2026-06-23). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + a real threat fak is structurally immune to, on two
> independent grounds. The KBRA / Relink mechanisms are NOT adopted (nothing to bolt
> them onto). One named invariant is recorded to guard future generative summarizers.**

**Source:** https://arxiv.org/abs/2606.21732 — "Safe to Check, Unsafe to Use: Relinking at
the Compression Boundary of LLM Agents", Zesen Liu, Zihan Zhang, Dongdong She (submitted
2026-06-19). Read from the arXiv abstract via WebFetch on 2026-06-23; this is a surface read
of the abstract, not a paper audit or a reproduction.

## The paper, in one pass

Summarization-based prompt compression — an agent shortens a long, distributed context by
having a model **summarize** it — moves the thing the backend acts on. The security boundary
splits: **filters inspect the pre-compression prompt, but the backend acts on a newly
generated compressed context.** The paper names **relinking**: the compressor acts as a
**confused deputy**, summarizing distributed, *locally benign* fragments into a *complete
malicious instruction* that was never present whole in the source. Unlike prompt injection,
no explicitly malicious payload need appear in the source context.

Relinking "arises from summarization itself": **attention** makes separated fragments jointly
available, **pre-training** makes compatible fragments plausible to connect, and
**post-training** favors compact, backend-actionable summaries. The attacker-induced form is
**adversarial relinking**; **Relink** is a DSL tool that splits a payload into benign
fragments (complete payload absent before compression). Across four long-context agent
benchmarks: **86.9% Relink Rate / Backend Action Rate vs 17.0% for clean-split controls**.
Existing defenses fail to capture it; the proposed **KBRA** defense drives residual Backend
Action Rate to **0.0%** (per the abstract; not re-measured here).

## Where fak actually stands

The paper's attack needs two things to both be true of the target: **(1)** a security check
positioned *before* compression that the backend then no longer sees, and **(2)** an
**abstractive** (generative) compressor that can author tokens absent from every source
fragment. fak fails to satisfy *both* preconditions — by construction, on two independent
load-bearing surfaces.

| The paper's frame | fak's position |
|---|---|
| Filter inspects the **pre-compression prompt**; backend acts on the **post-compression** summary → a check/act straddle | fak's floor is **not a prompt filter** at all. The floor is the **capability lock at `k.Decide`**, which adjudicates the *resulting tool CALL* on its **capability** (`SELF_MODIFY`, `POLICY_BLOCK`, …) — and that point sits **after** any context compression. A relinked instruction still has to *emit a call*, and that call is denied on its capability no matter how the context that produced it was compressed or phrased (`CLAIMS.md`, the detector-is-not-the-floor thesis; [`SECURITY-capability-floor`](SECURITY-capability-floor-2026-06-18.md)). There is no "safe to check, unsafe to use" gap because the load-bearing check is not on the prompt. |
| The compressor is **abstractive** — it generates a summary that can reconvene fragments into a new, complete instruction | fak's context compression is **extractive by construction.** `internal/contextq`'s `ViewSummary` is *"a deterministic extractive summary … every byte is verbatim from the source page, so FaithfulnessProbe = 1.0,"* lossy **only in coverage** (`contextq.go:105–121`). `tools/ctxwin`'s reducer elides / dedups / windows to a **recoverable pointer** — *"amputation, not noise-filtering,"* and *"cannot fabricate 2x out of incompressible context"* (`tools/ctxwin.py`). Neither authors a token that was not verbatim in a source fragment, so **neither can synthesize an instruction absent from the source** — the exact move relinking requires. |
| Existing defenses fail; bolt on **KBRA** at the summarization boundary | There is **no abstractive summarization boundary on fak's load-bearing path** to bolt KBRA onto. `FaithfulnessProbe = 1.0` is the witnessed invariant that the compressed view authored nothing — it is the *structural* form of what KBRA tries to recover *behaviorally* after the fact. |

So fak does not "defend against relinking" by detecting it — fak **removes both preconditions
for it**: the floor is on the call (after compression), and the compressor is forbidden from
generating. This is the same shape as the prior idea-scout triages (#526, #553): the field
keeps proposing a detector to recover safety *behaviorally*; fak keeps the unsafe degree of
freedom off the load-bearing path *structurally*.

## Triage decision

- **Adopt KBRA / Relink as a fak capability?** **No.** KBRA is a defense for **abstractive**
  summarization-compression; fak's compression is extractive-with-a-faithfulness-witness, so
  there is no generative summary for KBRA to screen and no relinking surface for Relink to
  attack. Adopting either would mean first *adding* the very abstractive-summarizer hazard the
  paper warns about, then bolting on its detector — the inverse of fak's keep-it-off-the-floor
  discipline.
- **Defend against (is the threat real for fak)?** **Real threat in the field; fak is
  structurally immune today, with one named invariant to guard.** The immunity is not luck —
  it rests on two facts that must stay load-bearing: (a) the capability lock adjudicates the
  *call* after compression, never the pre-compression prompt; (b) `FaithfulnessProbe = 1.0` on
  every materialized context view (`contextq`), and extractive-only reduction (`ctxwin`). The
  **residual fak must not regress on**: the day fak adds a **model-backed abstractive
  summarizer / proposer** to the agentic context path — the ctxplan *model-arm* follow-on
  (`403174d` ships the deterministic, model-free seed and explicitly fences the model arm as
  the higher tier; [`O1-TURN-CONTEXT-PLANNER`](O1-TURN-CONTEXT-PLANNER-2026-06-23.md) S6), a
  generative memory-compactor, or any outbound transform that *rewrites* rather than *elides* —
  that newly generated context is **un-admitted, model-authored content**. It must be
  re-screened at the **result-admit boundary** (`ctxmmu` / `normgate`, `CLAIMS.md` units
  61–70, 109) *after* it is generated, and **never trusted as a faithful view of
  already-checked text**. That is precisely the "act on the post-compression artifact, not the
  pre-compression prompt" boundary this paper proves today's filters miss. Recorded here as an
  invariant to guard, not a gate to build now.
- **Cite as prior art?** **Yes.** "Safe to Check, Unsafe to Use" is the cleanest external
  formalization of the **compression-boundary security shift**, and the strongest external
  argument for *why* two existing fak design choices are load-bearing rather than incidental:
  (1) context views are **extractive with `FaithfulnessProbe = 1.0`**, not abstractive; and
  (2) the floor gates the **resulting call**, not the prompt. It names the failure mode those
  choices avoid. Cited in the spirit of `CLAIMS.md`'s `[0/29 novel — the contribution is the
  assembly]` prior-art discipline.

**Action:** close #554 as triaged → **prior art cited + a real threat fak is structurally
immune to (extractive-by-construction compression + call-not-prompt floor); KBRA/Relink not
adopted; one named invariant recorded.** No code change in this increment: `tools/idea_scout.py`
surfaced and scored the candidate correctly (topic `prompt-injection-defense`, score 53), and
the right small artifact for a research / security triage is the recorded verdict + the named
invariant, not a detector for a hazard fak's compression does not introduce.

**Next step (the smallest honest follow-on, if pursued):** a one-time witness asserting that
**every** materialized context view on the agentic path carries `FaithfulnessProbe == 1.0`
(extractive) *or* is routed through the result-admit gate before it can influence a call
(generative) — i.e. there is no third path where model-authored context reaches `k.Decide`
unscreened. Filed as its own scoped test against `internal/contextq` + the ctxplan model-arm
seam when that arm lands. It guards the invariant this note names rather than importing the
paper's summarize-then-detect loop.
