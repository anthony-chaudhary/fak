---
title: "The global workspace, negation, and fak's negation operator"
description: "How fak turns the global-workspace result into a negation operator: construct the positive state and broadcast it, so the model selects instead of inverting."
---

# The negation operator: fak's global-workspace research program

A negation operator is a dedicated mechanism that resolves a negative ("not X") into its
positive substitute before a language model has to process it. fak's research program builds
one, and it is grounded in one mechanistic result: transformers have no native operation that
inverts a concept direction in the residual stream, so a model emulates negation with its
scarce, shared workspace — slowly and unreliably. fak's operating rule is therefore to
construct the positive state and broadcast it: the model should only ever select from
positives, with inversion handled by dedicated machinery outside or inside the forward pass.

This note is the program map: the result it builds on, the two shipped surfaces that already
practice the rule, the L0–L4 operator spectrum, and the invariants and measurements that keep
it honest. It also practices what it describes — negative idioms appear below only as fenced
example data, and each section leads with the affordance.

## What is a global workspace in a language model?

A global workspace is the scarce, shared representational stage that Anthropic's
"Verbalizable Representations Form a Global Workspace in Language Models" reports in LLMs:
automatic pattern-completion bypasses it, while deliberate, flexible transforms route through
it and broadcast their result to many downstream circuits. The same work reads and steers
model state through concepts represented as approximately linear directions in the residual
stream — which is the handle that makes an engineered operator conceivable at all. fak treats
this as external research to build on, and states its own claims separately from the paper's.

The practical consequence: anything a model must do deliberately competes for a small serial
resource. Work you can move out of that workspace — by handing the model a precomputed
positive instead of an inversion task — is workspace returned to the actual job.

## Why is negation hard for language models?

Negation is hard because "not X" is an operation on the representation of X, and the
transformer substrate supplies no primitive that performs it. If X is a direction in the
residual stream, producing the state "anything but X" requires the model to emulate an
inversion across forward-pass depth, inside the same scarce workspace every other deliberate
step competes for. Negation is the archetypal deliberate transform.

The NLP literature has documented the symptom for years: Ettinger's "What BERT Is Not" showed
masked models largely insensitive to negation in psycholinguistic cloze tests; Kassner and
Schütze's negated-mispriming probes showed pretrained models completing negated statements
with the very fact being denied; and the Inverse Scaling Prize's negated-QA task (NeQA) showed
larger models sometimes scoring worse on negated questions. The global-workspace result
supplies a mechanism that predicts exactly this failure shape: an emulated, depth-taxed
transform will be under-encoded and inconsistently applied.

## What is fak's operating rule?

The rule is: construct the positive state and broadcast it. Every fak surface that steers a
model — injected rules, step advice, refusal notes, resume prompts, the context window itself
— should carry the state the model needs in positive form, so the model reads what is true and
selects what to do, and inversion work stays out of its workspace. Two shipped surfaces
already run this rule:

- **negframe** ([`internal/negframe`](https://github.com/anthony-chaudhary/fak/blob/main/internal/negframe/negframe.go),
  #3538/#3566) rewrites prohibitions into positive affordances in prose, in token space,
  before tokens reach the model:

  ```text
  before: don't forget to stamp the commit
  after:  remember to stamp the commit
  ```

  It is one machine aimed two ways. `Classify` is the static lint: it reads prose on disk,
  finds negatively-framed spans, and counts the mechanically-reframable ones as hard debt.
  [`Reframe`](https://github.com/anthony-chaudhary/fak/blob/main/internal/negframe/reframe.go)
  is the emit-time pass: it flips only unambiguous idioms, keeps judgement-tier prose
  byte-identical, and admits a rewrite only when every contract token in the original survives
  it (the token-superset fail-safe). It is pure and idempotent, so it is safe on the request
  path.
- **managed-context** treats a session as a query, not a chat: the originating task stays
  pinned at the head, the volatile error/result state is swapped underneath it, and
  superseded history is replaced by its positive residual — the model reads what is true now
  instead of mentally subtracting what used to be true. The reader-facing explainers are
  [you never manage the context window](../explainers/you-never-manage-the-context-window.md)
  and [context shedding](../explainers/context-shedding.md); the same boundary thinking is in
  [a tool call is a syscall](../explainers/tool-call-is-a-syscall.md).

## What is a negation operator?

A negation operator is the dedicated machinery that performs the inversion so the model can
skip emulating it: detect that a negative is in play, apply the operator, and substitute the
positive result directly into what the model consumes — the prompt, the context, or the
residual stream itself. The operator has three parts wherever it lives: a detector (lexical,
parse-time, or an activation probe that notices the negation feature firing), the transform
itself, and a substitution step that writes the positive back. In its functional form it is a
callable primitive — ask "give me the positive form of not-P over domain D" and receive the
substitution, a candidate set, or an explicit UNKNOWN.

## What are the L0–L4 rungs of the negation operator?

The program is a spectrum of rungs, cheapest and shipped at the bottom, architectural and
speculative at the top. Each rung is a place the operator can live.

| Rung | Operator | Where it lives | What it does | Status (July 2026) |
|------|----------|----------------|--------------|--------------------|
| L0 | Lexical reframe | Token space, emit time (negframe) | Rewrites a fixed negative idiom into its positive inverse before tokens reach the model | Shipped keystone; runtime wirings in flight |
| L1 | Semantic negation-normal-form | Preprocessing pass (gateway emit seam) | Parses a constraint, pushes negation inward (De Morgan), re-expresses it as a positive allow-set ("use only {complement}") | Designed |
| L2 | Positive-complement resolution | Complement registry + resolver | Resolves not-X over domain D to the complement D minus {X} and substitutes it, so the model selects instead of inverting; ambiguity yields a candidate set or UNKNOWN | Designed |
| L3 | Activation-space operator | Forward-pass hook in the in-kernel model | Detects the negation feature firing and applies a learned operator: steering/affine map, SAE-feature swap, dedicated micro-adapter, or attention/OV-circuit graft | Research |
| L4 | Native architectural primitive | Model architecture | A cheap automatic negation op — learned involution (N squared = identity), rotation representation, or gated polarity channel — that leaves the deliberate-workspace hot path entirely | Speculative |

L4 is the full circle of the thesis: it would move negation from "deliberate and emulated" to
"automatic and bypassed," the exact axis the global-workspace paper draws.

## How does fak substitute the positive directly?

L2 is the core move. When a negative is unavoidable — the fact really is "the capital is not
Paris" — the operator resolves the positive residual itself instead of passing the inversion
task to the model. Given not-X over a domain D, it returns the complement of X in D: exact
when D is enumerable (the actual capital, from the registry), a candidate set when D is only
partially known, and an explicit UNKNOWN when it is open. What the model then receives is a
positive statement or a short list to select from; the inversion already happened in ordinary
code, where it is cheap, auditable, and testable. Managed-context applies the same move to
history: a stale or superseded span is replaced by its positive residual rather than left in
place for the model to subtract.

## What invariants keep the operator safe?

Four invariants generalize negframe's fail-safe to every rung, and all of them fail closed:

1. **Fabrication is outside the contract.** When the complement is ambiguous or
   non-enumerable, the operator returns the candidate set or an explicit UNKNOWN — a made-up
   positive is a wrong answer wearing a confident face.
2. **Load-bearing prohibitions survive.** A safety constraint must emerge from the operator
   with its force intact; when a rewrite would soften or drop it, the operator emits the
   original byte-identical. This generalizes negframe's token-superset fail-safe into a
   polarity-preservation gate.
3. **Idempotence and equivalence.** operator(operator(x)) equals operator(x), and every
   substitution is admitted only through a semantic-equivalence gate.
4. **The tax is measured, along with correctness.** The payoff claim is freed workspace and
   reduced forward-pass depth, so the instrumentation records the depth/latency cost of naive
   negation versus the operator, alongside accuracy.

## How is this measured?

By A/B ablation, in fak's witness discipline: every rung ships with a paired run — naive
negation versus operator — through the ablation harness, and the claim is the measured delta,
with honest error bars. The evaluation surface is a dedicated negation benchmark: negated
cloze, negated QA (including an inverse-scaling regression check against the NeQA failure
shape), "only/except" instruction adherence, and De Morgan equivalence. For L3, the causal
test is activation patching: the operator counts as real when patching it in makes negated
inputs behave correctly. There are no benchmark numbers to report yet — the harness and the
measurement plan exist; the negation suite is the witness the operator must beat before any
performance claim ships.

## What is shipped today, and what is research?

Shipped and witnessed, as of July 2026:

- The negframe keystone: the negation lexicon, the `Classify` lint with its scorecard debt
  integer, and the emit-time `Reframe` pass with the token-superset fail-safe
  (`internal/negframe`, #3566). Some runtime wirings of the reframe pass are still landing —
  in-flight, listed here as such.
- Managed-context as a default-on pipeline: pinned task, view budget, shedding with the
  cached head kept byte-identical, and restorable dropped spans.

Design-stage: L1 (negation-normal-form) and L2 (complement registry, resolver, swap-in
rewriter, UNKNOWN fallback). Research: L3, in the models fak serves in-kernel, where a
forward-pass hook is possible at all. Speculative: L4, an architectural primitive — named so
the program has a top rung, claimed as exactly that and nothing more.

## FAQ

### Is the negation operator a safety filter?

No — it is a representation transform, orthogonal to content classification. It changes the
form negatives take on the way to the model; a capability floor and its default-deny policy
(see [default-deny vs classifier](../explainers/default-deny-vs-classifier.md)) still decide
what effects a tool call may have.

### Does reframing soften safety rules?

The design commitment is the opposite: only mechanical idioms with an unambiguous positive
inverse are rewritten, and a rewrite is admitted only when every contract token survives it.
Judgement-tier prohibitions stay byte-identical, and the polarity-preservation gate extends
that guarantee up the rungs.

### What happens when the positive complement is ambiguous?

The operator returns the candidate set, or an explicit UNKNOWN when the domain is open.
Ambiguity is surfaced, and the fabricated-positive path is closed by contract.

### Does any shipped rung change model weights?

No. L0–L2 operate entirely outside the model, on tokens and context. Weight- or
activation-touching mechanisms begin at L3, which is research, in-kernel only, and gated on
the causal activation-patching test.

### Why does this belong in an agent kernel rather than a prompt guide?

Because the rule has to hold for machine-assembled strings and machine-managed context, at
volume, with proof. A style guide covers hand-written prose; an operator on the emit path
covers every string and every context view the runtime produces, and its telemetry makes the
negation tax visible instead of anecdotal.

## Sources and prior art

Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (the
workspace result this program builds on); Allyson Ettinger, "What BERT Is Not" (TACL 2020);
Nora Kassner and Hinrich Schütze, "Negated and Misprimed Probes for Pretrained Language
Models" (ACL 2020); the Inverse Scaling Prize's negated-QA task (NeQA), from "Inverse Scaling:
When Bigger Isn't Better" (2023). In-repo companions:
[negframe's package doc](https://github.com/anthony-chaudhary/fak/blob/main/internal/negframe/negframe.go),
the [managed-context explainer](../explainers/you-never-manage-the-context-window.md), and the
AEO program notes
[AEO-MARKETING-NEXT-STEPS-2026-07-01](AEO-MARKETING-NEXT-STEPS-2026-07-01.md) and
[FABLE5-AEO-FRONTIER-MODEL-MARKETING-2026-07-04](FABLE5-AEO-FRONTIER-MODEL-MARKETING-2026-07-04.md).

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "TechArticle",
      "headline": "The global workspace, negation, and fak's negation operator",
      "description": "How fak turns the global-workspace result into a negation operator: construct the positive state and broadcast it, so the model selects instead of inverting.",
      "url": "https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/GLOBAL-WORKSPACE-NEGATION-OPERATOR-AEO-2026-07.md",
      "datePublished": "2026-07-12",
      "author": {"@type": "Organization", "name": "fak project"},
      "about": ["global workspace", "negation operator", "LLM negation", "positive-state construction", "agent kernel"]
    },
    {
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "Is the negation operator a safety filter?",
          "acceptedAnswer": {"@type": "Answer", "text": "No — it is a representation transform, orthogonal to content classification. It changes the form negatives take on the way to the model; a capability floor and its default-deny policy still decide what effects a tool call may have."}
        },
        {
          "@type": "Question",
          "name": "What happens when the positive complement is ambiguous?",
          "acceptedAnswer": {"@type": "Answer", "text": "The operator returns the candidate set, or an explicit UNKNOWN when the domain is open. Ambiguity is surfaced, and the fabricated-positive path is closed by contract."}
        },
        {
          "@type": "Question",
          "name": "Does any shipped rung change model weights?",
          "acceptedAnswer": {"@type": "Answer", "text": "No. L0–L2 operate entirely outside the model, on tokens and context. Weight- or activation-touching mechanisms begin at L3, which is research, in-kernel only, and gated on the causal activation-patching test."}
        }
      ]
    }
  ]
}
</script>
