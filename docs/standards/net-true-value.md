---
title: "Net-true value — how fak decides a gain is real, not noise"
description: "fak's standard for any efficiency/performance claim: a gain is reported only if it survives a six-question rubric — measured against the real alternative (not a strawman), net of the costs it introduces, scope stated, provenance-labeled, reproducible, and realized by default. The same lens reads an incoming industry '5x' / 'save 90% tokens' claim. Each criterion maps to a stick the repo already runs."
---

# Net-true value

A new "5×" or "save 90% of tokens" lands almost every day. Most of it is noise: it
beats a strawman baseline, it holds only in a narrow scope, it is a thing a tuned stack
(or fak) already does, or it is modeled and never measured. This page is the standard fak
holds **its own** claims to, and the lens it reads **other people's** claims with. They are
the same rubric, used in two directions.

The commitment is one sentence: **a gain is net-true at a stated scope, or it isn't a
gain we report.** "Net-true" means the win survives after you subtract the cost the change
itself introduces and compare against the alternative a competent operator would actually
deploy — not the worst thing they could have done instead.

This is not a new gate. It is the *name* for a discipline the repo already runs in pieces
([`CLAIMS.md`](../../CLAIMS.md) tags, the baseline-letter convention in
[`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md), the conflation and default-value
scorecards, the salience register). The table at the end binds each rubric question to the
stick that mechanizes it, so this page is a lens over evidence, not prose on top of it.

## The rubric — six questions a gain must answer

Run any claim — yours before you ship it, or one you read — through these. A claim that
can't answer one of the first five is `not yet`, not a result.

1. **Baseline — measured against the real alternative, not a strawman.** "5× faster"
   answers nothing until you say *than what*. The honest baseline is the best practice a
   competent operator already runs, not the naive thing they'd never ship. When you cite
   the easy baseline, cite the hard one beside it and lead with the hard one: fak's fleet
   reuse is **60.3× vs naive re-send-everything** *and* **4.1× vs a tuned warm-cache
   stack** — the 4.1× is the headline; the 60.3× must never be quoted as the serving win.

2. **Net — after the cost the change itself adds.** A cache that is a win on reuse is a
   *loss* on single use. A quantization that saves bandwidth can cost accuracy. A saved
   tool call can cost a re-decode. A gain stated without its own cost is half a
   measurement. Report the net, including where the net goes negative.

3. **Scope — the conditions it holds under, and the ones it vanishes under.** State both.
   fak's tool-vDSO is cache-favorable in the demo (~50% hits) but ~**0.7%** addressable on
   real tau2-airline — so it's an upside secondary, never a headline. Cross-worker prefill
   reuse is 8.8–9.7× vs naive but only **1.0–1.1×** once each worker already has a warm
   cache. The scope *is* the claim.

4. **Provenance — measured, not modeled (and labeled either way).** Every number carries
   one of WITNESSED (a fact fak authored and controls), OBSERVED (a value relayed from an
   external party), MODELED (a deterministic projection), or SIMULATED (labeled stand-in
   data). A modeled floor is never quoted as a wall-clock; a provider-side miss is never
   blamed on a fak action.

5. **Witness — a third party can re-derive it.** A `go test`, a committed artifact plus a
   reproduce command, a benchmark field that reads back. No witness ⇒ `not yet` with the
   missing evidence named — never an unproven claim dressed as a shipped one.

6. **Realized — on by default, or honestly gated.** A real gain ships enabled (or is
   gated OFF with a stated reason). A value that exists only behind a flag nobody sets is
   not a realized gain; it's a seam. This is the difference between *available* and *true
   net for the operator who installs the defaults*.

## Reading an incoming claim

The four noise patterns, the question that exposes each, and the fak surface that already
encodes the honest answer:

| Pattern | The tell | Ask | fak's honest frame |
|---|---|---|---|
| **Strawman baseline** | "5×" with no "vs what", or vs the naive floor | Against the *tuned* alternative? | baseline letters A=naive / B=tuned / C=fak ([`BENCHMARK-AUTHORITY`](../../BENCHMARK-AUTHORITY.md)) |
| **Narrow scope sold as general** | one cache-favorable trace | Where does it vanish? | lossy/lossless + layer tags ([`awesome-token-efficiency`](../awesome-token-efficiency.md)) |
| **Already in a tuned stack / already in ours** | a "new" trick the engines ship | Marginal over best practice? | the 10 already-shipped SOTA optimizations are the baseline ([`sota-optimizations`](../explainers/sota-optimizations.md)) |
| **Modeled or cherry-picked** | a headline with no artifact | Measured? reproducible? | provenance labels + traceability ([`BENCHMARK-AUTHORITY`](../../BENCHMARK-AUTHORITY.md)) |

The lens cuts constructively too. A method that **survives** the rubric and isn't in fak
doesn't get waved away — it lands in [`awesome-token-efficiency`](../awesome-token-efficiency.md)
with its honest tags and, if it can be safely default-on, a tracking issue. Net-true means
we adopt real gains as readily as we decline noise. The daily intake from
`tools/idea_scout.py` runs this lens; the verdict lands as a triage note, not a slogan.

## How fak holds itself to it (the receipts)

This standard would be decoration if fak's own claims didn't pass it. They are kept honest
by machinery, not goodwill:

- **Per-capability tags** — every `- [` line in [`CLAIMS.md`](../../CLAIMS.md) carries
  exactly one of `[SHIPPED]` / `[SIMULATED]` / `[STUB]`, lint-enforced by `make claims-lint`.
- **Baseline letters + traceability** — every number in
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) pins a baseline letter and traces
  to a commit + artifact; any number quoted elsewhere must trace back here.
- **Provenance, not blame** — the [conflation scorecard](../CONFLATION-SCORECARD.md) checks
  that every reported number labels WITNESSED vs OBSERVED, so a dashboard can't say "fak
  broke the cache" when the provider's cache simply expired.
- **Realized, not shelved** — the [default-value scorecard](../DEFAULT-VALUE-SCORECARD.md)
  reds the tree when a value flag ships OFF without a documented reason.
- **Parked ≠ dropped** — a claim true at its scope but off the live path is retained, not
  deleted, in the [claims salience register](../claims-salience-register.md).
- **Strawman headlines flagged** — `tools/docs_scorecard.py` counts strawman-led headlines
  as doc-debt across the reachable corpus.

A worked own-example: the H100 decode result. The easy story was a missing-feature "5×".
The net-true reading named the gap as **memory bandwidth** (an f32 weight moves ~4 B where
a Q8 weight moves ~1 B), so the headline tok/s stays hardware-gated and the lever is a Q8
decode path, not a press release — see
[`H100-KERNEL-5X-ROADMAP.md`](../benchmarks/H100-KERNEL-5X-ROADMAP.md). Naming the real
bottleneck is worth more than a number measured against the wrong baseline.

## Criterion → stick

What keeps this page honest is that most of it is already mechanized:

| Rubric question | Stick that enforces it | State |
|---|---|---|
| 1 · Real baseline | baseline letters (`BENCHMARK-AUTHORITY`); strawman-headline check (`docs_scorecard`) | enforced |
| 2 · Net of cost | strawman/structure checks + review | partial — no single net-cost stick yet |
| 3 · Scope stated | FAQ stratification caveats; awesome-token-efficiency tags | enforced by convention |
| 4 · Provenance labeled | [conflation scorecard](../CONFLATION-SCORECARD.md) | enforced |
| 5 · Reproducible witness | `make claims-lint`; benchmark traceability | enforced |
| 6 · Realized by default | [default-value scorecard](../DEFAULT-VALUE-SCORECARD.md) | enforced |

## How this is encountered by default

The point of a standard is that nobody has to go looking for it.

- **Agents** meet it in [`AGENTS.md`](../../AGENTS.md) next to the "every claim carries a
  tag" rule — the lens an agent runs before reporting a win, and over the daily idea-scout
  intake before importing one.
- **Humans** meet it in the doc map ([`llms.txt`](../../llms.txt), [`INDEX.md`](../../INDEX.md))
  beside the claims ledger and the benchmark authority, and it is the connective tissue
  under the [charter](../notes/CHARTER.md)'s principle #2 (industry-leading value) and #9
  (win-win-win).

## Honest fences

- This is a **standard plus a lens over existing sticks**. The single grader the standard
  names — a Go subcommand that takes a claim + baseline + witness and returns net-true /
  strawman / `not yet` against the six questions — now ships as **`fak claim-check`**
  (`internal/claimcheck`, the cross-cutting follow-on of epic [#1147](https://github.com/anthony-chaudhary/fak/issues/1147),
  issue [#1171](https://github.com/anthony-chaudhary/fak/issues/1171)). It is a lens, not an
  oracle: it grades whether a claim, as **stated**, answers the six questions — silence on any
  of the first five is `not-yet`, a gain measured against the naive floor is `strawman`. Run
  `fak claim-check --self-test` to grade the built-in honest+strawman corpus. The lens does
  not measure anything; it does not replace the sticks below, it folds them into one verdict.
- Question 2 (net of introduced cost) is the least mechanized: strawman-headline detection
  and the structure checks catch the loud cases, but a claim that quietly omits its own
  cost still relies on review. Closing that gap is the highest-leverage next stick.
- Saying a gain is net-true at a stated scope is not the same as saying it is large. A
  small, honest, reproducible win beats a big one measured against the wrong baseline —
  and this standard exists to keep that ordering.
