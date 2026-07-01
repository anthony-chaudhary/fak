---
title: "Data residency & compliance — keep inference and data in-country behind one binary"
description: "How fak's self-host-first, fail-closed, default-deny, audit-logged boundary maps to India's DPDP Act and China's PIPL/DSL/CSL — an enforcement boundary you run on infrastructure you control. Not legal advice."
---

# Data residency & compliance — inference and data stay where you put them

Startups in regulated markets — India under the **DPDP Act, 2023** and China under
**PIPL / the Data Security Law / the Cybersecurity Law** — face the same structural
pressure: personal data (and, in China, "important data") should be processed on
infrastructure the operator controls, and cross-border transfer is constrained. A hosted,
cross-border AI API is the awkward shape here. fak is the opposite shape by construction.

> **Fence — this is not legal advice, and fak is not a certified compliance product.**
> fak is a technical *enforcement boundary* you deploy on infrastructure you choose. It
> gives you the controls (locality, deny-by-default, a tamper-evident log) that a
> data-protection program is built from; whether your overall system is compliant is a
> question for your counsel and your DPO, not for this page.

## The mechanism, not a marketing claim

fak's residency story is a direct consequence of four properties it already ships. None of
these are new for this doc — they are the same properties described in the
[README](../../README.md) and [llms.txt](../../llms.txt), read through a residency lens.

1. **Self-host-first.** `fak` is one static Go binary with zero external dependencies. Put
   it in front of a **local** model (`fak guard --gguf …`, or `fak serve` fronting Ollama /
   vLLM / SGLang / llama.cpp on your own hardware) or a **domestic provider**, and the
   inference path never leaves infrastructure you control. Same artifact on a laptop and in
   a fleet — you add flags, not components or third-party services.
2. **Fail-closed residency across backends.** The boundary's default is to *refuse*, not to
   forward-and-hope. An effect the policy did not allow is one that never leaves the box —
   there is no "leaked to an external service by default" path to reason about.
3. **Default-deny capability floor (structural, not a classifier).** Which tools may run and
   which tool *results* may re-enter model context is decided by the policy on the call
   path, in-process. A model cannot request an effect the capability was never wired for.
   Structure beats a recognizer a prompt can argue past — the relevant property when the
   thing you are containing is untrusted model output over personal data.
4. **Tamper-evident audit surface.** Every decision is recorded with an `X-Trace-Id`
   correlation id — an auditable "who asked for what, and what the kernel decided" trail,
   the kind of record a data-protection review asks for. See the
   [gateway API reference](../fak/api-reference.md) and
   [trajectory observability](../observability/trajectory.md).

## Mapping to the two regimes

### India — Digital Personal Data Protection Act, 2023

The DPDP Act pushes fiduciaries toward processing personal data on controlled
infrastructure and being able to demonstrate the safeguards around it. fak's role:

- **Locality:** run the model and tool execution on your own Indian-region infrastructure;
  fak is the in-process gate, not another data processor in the chain.
- **Purpose/So-what limitation as capability:** the allow-list *is* the enforced list of
  what the agent may do with the data — a technical expression of data-minimization.
- **Auditability:** the per-call decision log is evidence for a review, not a promise.

### China — PIPL / Data Security Law / Cybersecurity Law

The Chinese regime constrains cross-border transfer of personal information and "important
data" and expects local security controls. fak's role:

- **In-country processing:** the self-host boundary keeps PI and important data on
  domestic infrastructure; fak fronts the **domestic models teams already run** — Qwen and
  GLM are proven bit-exact in the [in-kernel reference engine](../supported/models.md), and
  DeepSeek / Yi / Baichuan / Kimi and any open-weights model are fronted over the
  OpenAI-compatible wire — so residency does not cost you a model switch.
- **Structural containment:** default-deny means a cross-border effect must be *explicitly*
  wired to exist at all, which is a stronger control than after-the-fact detection.
- **Records:** the audit trail supports the "demonstrate your controls" expectation.

See the localized front doors for the in-language version of this pitch:
[हिन्दी](../i18n/hi/README.md) · [简体中文](../i18n/zh/README.md).

## What this does *not* claim

- It does **not** claim fak makes you compliant. It gives you locality, deny-by-default,
  and an audit trail; the program around them is yours.
- It does **not** claim a certification, a data-processing agreement, or a legal review.
- It does **not** change any benchmark or capability claim — residency is a *deployment
  posture* of the same boundary, achieved with flags (`--policy`, `--gguf`/local upstream,
  `--require-key-env`), not a separate product.

The honest summary: fak is the technical control surface a data-residency posture is built
on — self-host the model, deny by default, log every decision, keep the data on the box —
for exactly the markets where that posture is becoming mandatory. The go-to-market context
is in the [emerging-market adoption note](../notes/CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md).
