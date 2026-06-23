---
title: "idea-scout triage: 'When AUC 0.998 Is Not Enough' — empirical support for the detector-is-not-the-floor thesis (2026-06-23)"
description: "Triage of the idea-scout candidate arXiv:2606.22864 (Li, Fan, Zhuang): a cautionary case study showing a hidden-state probe can hit AUC 0.998 on a clean-vs-attack split without that being evidence of malicious-content detection, plus a candidate control set (paired scalar baseline + nuisance-matched visual controls) for what a high AUC does and does not license. Verdict: prior art to cite (it empirically vindicates fak's detector-non-load-bearing stance) AND an evaluation discipline fak should adopt for its own best-effort detector rung — not a threat and not a new capability."
---

# idea-scout triage — "When AUC 0.998 Is Not Enough" (issue #526)

> Closes the daily idea-scout candidate [#526](https://github.com/anthony-chaudhary/fak/issues/526)
> (`tools/idea_scout.py`, filed 2026-06-23). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite (empirical support for the detector-is-not-the-floor
> thesis) AND an evaluation discipline fak should adopt for its own detector rung.**

**Source:** https://arxiv.org/abs/2606.22864 — "When AUC 0.998 Is Not Enough: A Candidate
Evaluation Protocol for Hidden-State Probes of Indirect Prompt Injection in Multimodal
Computer-Use Agents", Yanhang Li, Zhichao Fan, Zexin Zhuang (submitted 2026-06-22). Read
from the arXiv abstract via WebFetch on 2026-06-23; this is a surface read of the abstract,
not a paper audit or a reproduction.

## What it argues

Hidden-state probing — a **linear classifier on a frozen VLM's internal activations** —
has become an attractive way to flag indirect prompt injection (IPI) in multimodal
computer-use agents *before* the agent emits a corrupted action. The paper's load-bearing
claim, on a single-backbone cautionary case study (Qwen2.5-VL-7B on Mind2Web, teacher-forced
replay): **a high probing AUC on a clean-vs-attack split is not, on its own, evidence of
malicious-content detection.** The headline AUC (0.998) can ride on nuisance correlates of
the *injection-surface-present* label rather than on the attack semantics.

It backs that with two post-hoc diagnostics, packaged as a **candidate control set**:

1. **A paired-construction scalar baseline** on text-side injections — a deliberately
   trivial scalar feature whose score on matched clean/attack pairs reveals how much of the
   probe's separability is available *without* reading content.
2. **Same-step, nuisance-matched visual controls** on the overlay surface — controls that
   hold the step fixed and vary only the nuisance, so a probe that is really keying on the
   overlay's *presence* (not its malicious content) is exposed.

Neither diagnostic "licenses an unqualified malicious-content interpretation of the
headline," while leaving room for *partly*-semantic readings. The authors are explicit about
scope: labels are **injection-surface-present, not attack success**, and generalisation
beyond this one backbone/benchmark is stated as a conjecture, not a result.

## Why it surfaced next to fak, and where fak actually stands

fak already runs a best-effort injection-detection rung — `normgate` (the rank-5
canonicalize-and-decode `ResultAdmitter`) and the `internal/agentdojo` ASR-gated red-team
battery. The repo's standing position on that rung is recorded verbatim in `CLAIMS.md`:

> *Honest ceiling, surfaced not hidden: the detector these drivers feed is ~100% evadable on
> a SOTA evasion battery and FP-prone on private real-transcript corpora. Detection is
> **deliberately non-load-bearing** — the structural guarantee is the capability floor +
> containment, which never run the detector; improving detection is additive, not the moat.*

This paper is **independent empirical support for exactly that stance**, arriving from the
opposite direction. fak's caution is *evasion-side* (a detector can be defeated by an
adversary). This paper is *evaluation-side* (a detector's headline number can be defeated by
the **evaluator's own optimism** — a 0.998 AUC that is mostly nuisance-correlation, before
any adversary lifts a finger). Together they bracket the same conclusion: **a high detector
score is never a floor.** A probe can look near-perfect on a clean-vs-attack split and still
not be detecting malicious content — so a security argument that rests on the AUC is unsound,
which is precisely why fak places its guarantee on the capability lock + containment that
*never run the detector*.

## Triage decision

- **Adopt the probe as a capability?** **No.** A frozen-VLM hidden-state probe is the very
  kind of best-effort, evadable detector fak deliberately keeps off the load-bearing path.
  Adding one as a security *floor* would contradict the detector-non-load-bearing thesis.
  fak's existing `normgate`/AgentDojo rung already occupies the "additive detector" slot, and
  the paper is a caution about *trusting* such a rung, not an argument to add another.
- **Adopt the evaluation discipline?** **Yes — this is the operative actionable.** The
  candidate control set is methodology, not mechanism: a **paired-construction scalar
  baseline** and **nuisance-matched controls** are exactly what should gate any future claim
  that `normgate` (or any added probe) "detects" injection rather than separating a nuisance
  proxy. The discipline reinforces how `CLAIMS.md` already reports the rung (ASR/FPR on a
  SOTA battery + private real-transcript FP rate, never a single clean-vs-attack AUC). Scoped
  here as a reporting heuristic for the detector lane, not built in this triage increment
  (see Next step).
- **Defend against?** **No.** This is an evaluation critique, not a threat surface. It does
  not describe an attack on fak; it describes a way evaluators fool themselves.
- **Cite as prior art?** **Yes.** Record it as external, peer-reviewed-track evidence for
  the detector-is-not-the-floor position — a companion to the evasion-side evidence already
  in `CLAIMS.md`. It belongs on the same shelf as the capability-floor security note
  (`docs/notes/SECURITY-capability-floor-2026-06-18.md`) as the *evaluation-side* citation.

**Action:** close #526 as triaged → **prior art cited + evaluation discipline noted** (this
note). No code change in this increment: `tools/idea_scout.py` surfaced and scored the
candidate correctly (topic `prompt-injection-defense`, score 67), and the right small
artifact for a research/security triage is the recorded verdict, not a half-built probe whose
own headline the paper would tell us not to trust.

**Next step (the smallest honest follow-on, if pursued):** add the two diagnostics as a
*detector-evaluation guard* in the `normgate`/`internal/agentdojo` lane — a paired-clean/attack
scalar baseline and a nuisance-matched control split reported alongside any AUC/ASR number,
so a future detector claim cannot ship on a clean-vs-attack split alone. Filed as its own
scoped issue against the detector lane, with a Mind2Web-style nuisance-matched fixture as its
witness. It reuses the existing ASR/FPR reporting harness rather than importing a probe.
