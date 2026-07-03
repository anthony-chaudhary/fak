---
title: "fak launch-post draft kit — an honest, ready-to-adapt Show HN / launch post"
description: "A single, self-contained launch-post draft kit for fak: title options, the opening hook, the prosecution-first first comment, a longer launch-post body, the honest what-it-is / what-it-is-NOT framing, a copy-paste TL;DR, and a pre-post checklist. Every claim is witnessed or honest-scoped against the repo's ledger; the naive perf multiplier is never led with; the injection detector is named as evadable-by-design. Draft only — posting is human-owned."
slug: launch-post-draft-kit
keywords:
  - show hn draft
  - launch post template
  - honest launch post
  - agent kernel positioning
  - default-deny tool-call gate
  - prompt injection containment
  - one static Go binary
  - go-to-market launch kit
date: 2026-07-02
---

# fak launch-post draft kit — honest, ready to adapt

> **This is a draft. Posting is human-owned.** Nothing here fires automatically; a person
> decides when, where, and whether to post. The job of this page is that *when* that person
> decides, the strongest **honest** version is already written and the pre-post checklist
> stops an over-claim that would backfire.
>
> Dimension **J — Distribution & channels** of the
> [concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md). This is
> the **one-page, self-contained** launch draft; the deeper, per-channel campaign (r/LocalLLaMA,
> the full Show HN objections list, X, Lobsters, YouTube) lives in
> [`docs/launch/`](https://github.com/anthony-chaudhary/fak/tree/main/docs/launch) and is
> cross-linked below rather than duplicated here.
> Where to *submit* it is [the directory checklist](directories.md); whether it *worked* is the
> [adoption-signals dashboard](signals.md).

## The one rule that governs the whole draft

**Lead with the fence — it's the hook, not the caveat.** The audiences a launch reaches
(Hacker News, r/LocalLLaMA, r/netsec, Lobsters) are reflexively hostile to AI launch-speak
and reward self-skepticism. fak's credibility *is* its fences and its zeros. So the single
move that disarms the AI-slop, naive-benchmark, and security-overclaim reflexes at once:
**make your own first comment the prosecution.** Name the weaknesses before the top
commenter does. Every draft below is written that way.

## Title options

Pick one; lead with the mental model, not the word "security" (security-framed launches read
as fear-marketing to these audiences).

- **Show HN: fak – Treat the LLM as untrusted and the tool call like a syscall (Go, one binary)**
- Show HN: fak – A default-deny capability gate for AI agents, in one static Go binary
- Show HN: fak – Delete a poisoned tool result from the *middle* of a kept model run (max|Δ|=0)

The first is the primary; it matches the vetted asset in
[`docs/launch/show-hn.md`](https://github.com/anthony-chaudhary/fak/blob/main/docs/launch/show-hn.md).

## The hook (one paragraph)

> One ~13 MB static Go binary (Apache-2.0, no external deps, no `go.sum`) sits between an AI
> agent and its tools. On the same in-process call path — no sidecar, no second model — it
> does two things: a **default-deny capability gate** the model can't talk past (an
> irreversible action is refused *by structure*; the lever was never wired, so there's nothing
> to jailbreak toward), and an **addressable, bit-exact KV cache** that lets you cut a poisoned
> tool result out of the *middle* of a kept model run and leave the cache bit-for-bit identical
> (checked at `max|Δ| = 0`).

## First comment (post as author, immediately)

> Disclosure: I built this.
>
> fak is one static Go binary that adjudicates every tool call an agent makes, in-process, on
> the tool-call path. Two mechanisms carry it:
>
> 1. A **default-deny capability gate**. A dangerous tool outside the allow-list can't be
>    called no matter what the model was told — it's refused by structure, not by recognizing an
>    attack.
> 2. **Result quarantine over a bit-exact KV cache**. A poisoned or secret-bearing tool result
>    is held out of context, and the cache can be re-knit so it's identical to a run that never
>    saw the bad span (`max|Δ| = 0`).
>
> 60-second proof — no key, no model, no GPU (the first run compiles the binary; later runs are
> instant):
>
> ```
> go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
> go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
> ```
>
> `fak agent --offline` shows the same two gates in a full agent loop: injection-in-context
> YES→no, destructive-op YES→no, task still completes.
>
> Repo: https://github.com/anthony-chaudhary/fak · live in-browser demos:
> https://anthony-chaudhary.github.io/fak/demos.html
>
> Honest fences up front, because this is the internet:
>
> - **fak is NOT a faster token engine and doesn't try to be.** vLLM / SGLang / llama.cpp win
>   raw throughput. The contrast is *one static binary vs a multi-GB Python/CUDA stack*, not tok/s.
> - The **injection detector is ~100% evadable — by design.** It's a tripwire, not the floor.
>   The floor is the two structural gates above.
> - The perf headline I lead with is **~4.1× less work than a *tuned* warm-cache stack** on a
>   read-heavy, self-hosted fleet — never the bigger naive re-prefill multiplier alone (that
>   denominator is the strawman a perf-literate reader divides away). The reuse win is
>   **self-host + read-heavy only**; an app that just calls a hosted API gets the safety floor,
>   not the savings.
> - I ran a prior-art audit and scored it **0/29 novel** — every primitive here is established.
>   The contribution is the *assembly*: one in-process gate where the tool call is the checkpoint.
> - Power / energy / $ figures in the repo are **simulated** (no power meter on the box). The
>   ~60× and "agent city" frontier numbers are **design targets, not measurements** — labeled
>   as such.
>
> Every number traces to [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
> (each claim tagged `[SHIPPED]` / `[SIMULATED]` / `[STUB]`) and
> [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md).
> The claim I'd most like torn apart is the bit-exact one — reproduce the `max|Δ| = 0`, don't
> trust the receipt.

## Longer launch-post body (blog / Lobsters / newsletter)

Use this when the venue rewards an idea-first essay rather than a link post.

> **Treat the model as untrusted and the tool call as a syscall.**
>
> Most agent security tries to recognize bad text — a classifier that reads a prompt or a tool
> result and guesses "is this an attack?" Recognizers help, but they are not a floor: a
> classifier you can evade is a classifier an attacker will evade. fak moves the load-bearing
> decision somewhere a model can't argue with: the **capability floor**. A dangerous tool that
> isn't on the allow-list simply cannot be called, no matter what the model was convinced of.
> The lever was never wired up, so there is nothing to jailbreak *toward*.
>
> Two independent gates carry it. Call-side: a denied call never reaches the tool runner.
> Result-side: a poisoned or secret-bearing tool *result* is quarantined before it enters the
> model's context. Beating fak means beating two structural gates, not fooling one detector.
>
> The second mechanism is a performance one, and it's honest about its scope. A long agent
> session re-sends its whole transcript every turn; a fleet of agents pays for the same shared
> system prompt over and over. fak does that shared setup work once and reuses it — and, because
> its KV cache is *addressable*, it can reach into the middle of a kept run, evict one span, and
> leave the cache bit-for-bit identical (`max|Δ| = 0`). That's what lets you delete a poisoned
> result *after* the model already saw it without re-prefilling everything after it. On a
> read-heavy, self-hosted fleet that's ~4.1× less work than a *tuned* warm-cache stack. If you
> just call a hosted API, you get the safety floor and none of the savings — and I'd rather say
> that up front than have you find out.
>
> It ships as one ~13 MB static Go binary. `fak guard -- claude` wraps the agent you already
> run; your model, keys, and IDE stay exactly as they are. The prior-art audit scored 0/29
> novel — every primitive is established engineering. The contribution is putting them in one
> in-process gate where the tool call is the checkpoint. Don't trust that framing; run the
> 60-second proof, then try to make the offline demo fire a destructive call. If you can, that's
> a real bug and I want the repro.

Deeper, per-channel variants (r/LocalLLaMA primary post, the full paste-from objections list,
X thread, YouTube script, the talk outline) are in
[`docs/launch/`](https://github.com/anthony-chaudhary/fak/tree/main/docs/launch).

## Honest framing — what fak IS and what it is NOT

Every row below is grounded in the repo; nothing here is aspirational unless it says so.

**fak IS:**

- An **agent kernel / reference-monitor**: an in-process, default-deny tool-call gate fused
  with an addressable, bit-exact KV cache. ([README](../../README.md), [AGENTS.md](../../AGENTS.md))
- **One static Go binary, Apache-2.0**, no external Go deps, drop-in in front of an agent you
  already run by repointing one base URL (Claude Code, Codex, Cursor, any OpenAI / Anthropic /
  MCP client). ([README](../../README.md))
- A **security boundary that doesn't depend on catching an attack** — the capability lock plus
  result quarantine are structural. ([POLICY.md](../../POLICY.md))
- A **read-heavy, self-hosted performance win** from reusing shared setup work across turns and
  agents — honest headline **~4.1× vs a tuned warm-cache stack**. ([BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md))

**fak is NOT:**

- **Not a faster token engine.** vLLM / SGLang / llama.cpp win raw throughput; fak fronts a
  token engine, it does not replace one. ([README Boundaries](../../README.md))
- **Not a prompt-injection classifier you should trust.** The detector is ~100% evadable *by
  design*; it's a tripwire, not the floor.
- **Not a savings story for hosted-API-only apps.** No self-host + read-heavy workload ⇒ safety
  floor only, zero reuse savings; even ~1% write rate can flip the reuse economics negative.
- **Not novel by primitive.** The 0/29 prior-art audit is in the repo; the value is the
  assembly, not an invented mechanism.
- **Not a hardware privilege ring.** "Syscall" / "kernel" are intuition pumps for *the tool call
  is the checkpoint where a default-deny check runs* — an in-process check, not a CPU ring.

**Aspirational / not-yet (label these as such if you mention them):** the ~60× reuse ceiling
and the "agent city" frontier are **design targets, not measurements**; all power / energy / $
figures are **simulated**. Never present either as a shipped result.

## Copy-paste TL;DR

> **fak** — one ~13 MB static Go binary (Apache-2.0) that sits in front of an AI agent and
> adjudicates every tool call like a syscall: a default-deny capability gate the model can't
> talk past, plus quarantine of poisoned tool results over a bit-exact KV cache (`max|Δ| = 0`).
> Drop-in via one base URL (Claude Code / Codex / Cursor / OpenAI / Anthropic / MCP). Not a
> faster token engine; the injection detector is evadable by design and explicitly not the
> floor. 60-second proof needs no key, model, or GPU. https://github.com/anthony-chaudhary/fak

## Pre-post checklist (all must be true before a human posts)

- [ ] **Green CI on the tip you're pointing at.** `make ci` passes on `main` (build + vet +
      test + claims-lint). A launch that links a red tree is the worst possible first impression.
- [ ] **The 60-second proof reproduces on a clean checkout.** Run the two `preflight` commands
      above and confirm `DENY (POLICY_BLOCK)` / `ALLOW` verbatim; run `fak agent --offline` and
      confirm the two YES→no flips. (Last verified against the live binary 2026-06-22 — re-run.)
- [ ] **Live demos load:** https://anthony-chaudhary.github.io/fak/demos.html returns the demo
      page, not a 404. The social-preview card resolves (`visuals/social-preview.png` tracked on `main`).
- [ ] **Every number in the post traces to the ledger.** Cross-check against
      [`CLAIMS.md`](../../CLAIMS.md) and [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md);
      lead with the **tuned ~4.1×**, never the naive multiplier alone.
- [ ] **The fences are in the first comment**, not buried: not-a-token-engine, evadable-detector,
      0/29-novel, simulated-power, design-target frontier numbers, self-host+read-heavy scope.
- [ ] **Authorship disclosed in the first line** ("disclosure: I built this") — it *raises*
      trust in these venues.
- [ ] **No upvote-asking, one venue at a time.** Vote-ring detection silently zeroes votes and
      is a flag magnet; read each venue's self-promo rule first ([landscape research](https://github.com/anthony-chaudhary/fak/blob/main/docs/launch/landscape-research.md)).
- [ ] **Strip the `Provenance & fact-check` appendix** from any asset in `docs/launch/` before
      pasting — it's an internal audit trail, not part of the post.

## Verify

```
# front-matter present and this doc is indexed
grep -n "^title:" docs/adoption/launch-kit.md
grep -n "launch-post-draft-kit\|launch-kit" INDEX.md

# the SEO/AEO scorecard does not regress for this new doc
python tools/seo_aeo_scorecard.py

# the 60-second proof the post promises actually fires
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
```

## What this doc does and does not claim

- **Does:** provide a ready-to-adapt, honesty-fenced launch draft (titles, hook, first
  comment, long body, IS/IS-NOT framing, TL;DR) and a pre-post checklist that blocks an
  over-claim.
- **Does not:** post anything (human-owned), assert fak is adopted, invent a benchmark, or lead
  with the naive perf multiplier. Every witnessed number traces to the ledger; the aspirational
  ones (~60×, "agent city", power/$) are labeled design-target / simulated.
</content>
</invoke>
