---
title: "The fak pitch ladder: one sentence, one paragraph, one page"
description: "The canonical fak pitch at three zoom levels — one sentence (the tweet), one paragraph (the HN comment), one page (the blog intro) — each honest, self-consistent, and quotable."
slug: pitch-ladder
keywords:
  - fak pitch
  - elevator pitch
  - treat the tool call like a syscall
  - the model proposes the kernel disposes
  - agent kernel
  - agent tool firewall
  - one-line summary
  - memorable framing
date: 2026-07-03
---

# The fak pitch ladder: one sentence, one paragraph, one page

> **The one-sentence pitch:** fak treats every agent tool call like a syscall —
> the model proposes, the kernel disposes: one static Go binary that makes the
> same agent loop safer, cheaper, and faster.

This is the same pitch at three zoom levels. Quote the rung that fits your
venue; every rung makes the same claims, so a reader who climbs from the tweet
to the page never catches the pitch contradicting itself. This artifact is
dimension **I — Memorable framing & naming** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md),
and it is the **source for the [README](../../README.md) lead** — if the pitch
changes, change it here first and re-derive the lead.

Every number below is witnessed and traces to
[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) and the honesty ledger
[CLAIMS.md](../../CLAIMS.md). Nothing here claims market adoption, an unrun
benchmark, or a novelty the 0/29 prior-art audit refutes.

## Rung 1 — one sentence (the tweet)

> fak treats every agent tool call like a syscall — the model proposes, the
> kernel disposes: one static Go binary that makes the same agent loop safer,
> cheaper, and faster.

29 words. The anchor is the syscall framing: an OS kernel never trusts a
program's word, and fak applies that same boundary to an AI agent's tool calls.
Everything else fak does unpacks from that one move.

## Rung 2 — one paragraph (the HN comment)

> fak is a fused agent kernel: one static Go binary you drop in front of the
> agent you already run (`fak guard -- claude` — repoint one base URL, keep your
> model, IDE, and keys). It treats every tool call like a syscall — the model
> proposes, the kernel disposes — checking each call against a default-deny
> capability floor and quarantining suspicious tool *results* by structure, so
> refusing an irreversible action never depends on catching the attack. Because
> the kernel owns the session, it also reuses the stable work: on a 50-turn ×
> 5-agent run it does ~4.1× less work than a tuned warm-cache stack, and it can
> evict one span from the middle of a KV cache and leave the rest bit-for-bit
> identical (`max|Δ| = 0`). The decision tax is ~362 ns per call, in-process.
> None of the primitives are new — the assembly into one drop-in binary is the
> point.

Same claims as the sentence, plus the mechanism (drop-in, default-deny,
quarantine), the two witnessed numbers a skeptic will check first (the tuned
~4.1×, never the naive re-send multiplier; the ~362 ns guard tax, measured on
an Apple M3 Pro), and the novelty concession up front.

## Rung 3 — one page (the blog intro)

> Your AI agent runs every tool call on trust. The model decides to delete a
> file, call an API, or push a commit, and the harness just… does it. Every
> defense in that loop is advisory: a system prompt asking nicely, a classifier
> hoping to catch the attack, a human skimming a transcript.
>
> Operating systems solved this problem decades ago. A program never gets to
> perform a dangerous action on its own say-so — it *proposes* a syscall, and
> the kernel *disposes*: checks it against policy, then allows, denies, or
> defers it at a boundary the program cannot reach around. fak applies that
> exact design to AI agents. It is one static Go binary that sits in front of
> the agent you already run — Claude Code, Codex, Cursor, or any
> OpenAI/Anthropic/MCP client. `fak guard -- claude` wraps your normal agent in
> one command; your model, IDE, and keys stay exactly as they are.
>
> From that seat, the kernel enforces five things:
>
> 1. **The tool call is a syscall.** Every call crosses the kernel and gets a
>    verdict — allowed, denied, repaired, quarantined, or deferred — against a
>    default-deny capability floor. The decision runs in-process in ~362 ns
>    (measured, Apple M3 Pro): no network hop, no perceptible tax.
> 2. **Refusal does not depend on detection.** Prompt-injection defense is
>    structural, not probabilistic: suspicious tool *results* are held out of
>    the model's context by quarantine, and an irreversible action is refused
>    because the capability was never granted — not because a classifier caught
>    the attack. The detector can be wrong; the floor holds anyway.
> 3. **Long sessions stop getting more expensive.** The kernel owns the KV
>    cache, so it can reach into the middle of a kept run, evict one span, and
>    leave everything else bit-for-bit identical (`max|Δ| = 0`). On a witnessed
>    50-turn × 5-agent session, that reuse does ~4.1× less work than a *tuned*
>    warm-cache stack — the honest bar, not the ~60× versus a naive re-send
>    loop.
> 4. **"Done" is verified, not trusted.** The DOS substrate re-checks an
>    agent's claims against git evidence: a false "shipped" is refused with a
>    structured reason from a closed vocabulary, and recalled memory is
>    re-verified at read time. The kernel is the part that doesn't believe the
>    agents.
> 5. **It's one binary, and it's yours.** The same static Go artifact a
>    developer runs on a laptop is what a platform team hardens for a fleet —
>    including fully local: it can host a GGUF model in-process, no key, no
>    network.
>
> To be precise about what fak is *not*: it is not a throughput engine (vLLM
> and llama.cpp win that race — fak fronts them instead), and none of its
> primitives are individually novel (a 29-point prior-art audit found 0 new
> ones). The claim is the assembly: kernel-grade discipline — admission,
> isolation, verification, memory management — fused into one drop-in binary
> for the agent loop you already have. The model proposes. The kernel disposes.

## The consistency contract

The rungs must never drift apart. When editing any rung, re-check all three
against this table and against the [README](../../README.md) lead, which is
derived from rung 1:

| Claim | Sentence | Paragraph | Page |
|---|---|---|---|
| Tool call as syscall (model proposes, kernel disposes) | ✔ anchor | ✔ | ✔ |
| One static Go binary, drop-in | ✔ | ✔ (`fak guard -- claude`) | ✔ |
| Safer: default-deny gate + quarantine | "safer" | ✔ mechanism | ✔ + evadable-detector concession |
| Cheaper: reuse, tuned ~4.1×, `max\|Δ\| = 0` | "cheaper" | ✔ numbers | ✔ numbers + naive-multiplier fence |
| Faster: ~362 ns in-process decision | "faster" | ✔ number | ✔ number |
| Verify, don't trust (DOS) | — (no room) | — | ✔ |
| Novelty fence (0/29 — assembly, not invention) | — | ✔ | ✔ |

Hard rules, inherited from the
[epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md): quote the tuned
**~4.1×**, never the naive multiplier unlabeled; no market-adoption claims; no
benchmark not already run; label simulated as simulated.

## Related framing artifacts

- [How to find and name fak](naming.md) — the disambiguated search terms the
  pitch should use (dimension I).
- [Objections & one-line answers](objections.md) — what to say after the pitch
  lands and the pushback starts (dimension I).
- [Social storyboard](social-storyboard.md) — the pitch cut into a
  card-per-concept thread (dimension J).
- [The 4× that's real](stories/the-real-4x.md) — the long-form honest telling
  of the headline number (dimension H).
