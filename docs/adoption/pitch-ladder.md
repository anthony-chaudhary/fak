---
title: "The fak pitch ladder: one sentence, one paragraph, one page"
description: "The canonical fak agent-runtime pitch at three zoom levels — one sentence, one paragraph, and one page — with Fused Agent Kernel retained as the technical architecture."
slug: pitch-ladder
keywords:
  - fak agent runtime
  - fak pitch
  - elevator pitch
  - treat the tool call like a syscall
  - the model proposes the kernel disposes
  - agent kernel
  - agent loop runtime
  - one-line summary
  - memorable framing
date: 2026-07-03
---

# The fak pitch ladder: one sentence, one paragraph, one page

> **The one-sentence pitch:** fak is an agent runtime: the operator-controlled
> boundary for cache and context, model routing, tool authority, memory,
> observability, and native inference—implemented as the Fused Agent Kernel in
> one static Go binary.

This is the same pitch at three zoom levels. Quote the rung that fits your
venue; every rung makes the same claims, so a reader who climbs from the tweet
to the page never catches the pitch contradicting itself. This artifact is
dimension **I — Memorable framing & naming** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md),
and it is the **source for the [README](../../README.md) lead** — if the pitch
changes, change it here first and re-derive the scoped public front doors.

Every number below is witnessed and traces to
[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) and the honesty ledger
[CLAIMS.md](../../CLAIMS.md). Nothing here claims market adoption, an unrun
benchmark, or a novelty the 0/29 prior-art audit refutes.

## Rung 1 — one sentence (the category)

> fak is an agent runtime: the operator-controlled boundary for cache and
> context, model routing, tool authority, memory, observability, and native
> inference—implemented as the Fused Agent Kernel in one static Go binary.

The public category is **agent runtime**. The boundary definition says what the
operator controls; **Fused Agent Kernel** names the technical architecture that
implements it. The syscall framing remains the shortest explanation of the
tool-authority seam, not the category a new reader must learn first.

## Rung 2 — one paragraph (the HN comment)

> fak is an agent runtime: the operator-controlled boundary for cache and
> context, model routing, tool authority, memory, observability, and native
> inference. Its Fused Agent Kernel architecture is one static Go binary you
> drop in front of the agent you already run (`fak manage -- claude` — repoint
> one base URL, keep your
> model, IDE, and keys). It treats every tool call like a syscall — the model
> proposes, the kernel disposes — giving each call a reviewable verdict while
> the kernel reuses the stable work in the session. On a 50-turn × 5-agent run it
> does ~4.1× less work than a tuned warm-cache stack, and it can evict one span
> from the middle of a KV cache and leave the rest bit-for-bit identical
> (`max|Δ| = 0`). The same in-process boundary also routes calls, repairs
> malformed ones, quarantines distrusted results, and checks tool authority from
> policy in ~362 ns per call. None of the primitives are new — the assembly into
> one drop-in binary is the point.

Same claims as the sentence, plus the mechanism (drop-in, verdicts, reuse,
routing, quarantine), the two witnessed numbers a skeptic will check first (the
tuned ~4.1×, never the naive re-send multiplier; the ~362 ns guard tax, measured
on an Apple M3 Pro), and the novelty concession up front.

## Rung 3 — one page (the blog intro)

> Your AI agent spends most of a long session re-paying for setup: the same
> system prompt, the same tools, the same project context, the same retries after
> malformed calls. A fleet pays that cost again for every agent. Meanwhile the
> harness still treats a model's proposed tool call as the thing to run, not the
> checkpoint where cost, routing, policy, and evidence should be decided.
>
> fak is an agent runtime: the operator-controlled boundary for cache and
> context, model routing, tool authority, memory, observability, and native
> inference. Its technical architecture is the Fused Agent Kernel. Operating
> systems solved one part of this problem decades ago: a program never gets to
> perform a dangerous action on its own say-so — it *proposes* a syscall, and
> the kernel *disposes*: checks it against policy, then allows, denies, or
> defers it at a boundary the program cannot reach around. fak applies that
> exact design to AI agents. It is one static Go binary that sits in front of
> the agent you already run — Claude Code, Codex, Cursor, or any
> OpenAI/Anthropic/MCP client. `fak manage -- claude` wraps your normal agent in
> one command; your model, IDE, and keys stay exactly as they are.
>
> From that seat, the kernel provides five things:
>
> 1. **The tool call is a syscall.** Every call crosses the kernel and gets a
>    verdict — allowed, denied, repaired, quarantined, or deferred. The decision
>    runs in-process in ~362 ns (measured, Apple M3 Pro): no network hop, no
>    perceptible tax.
> 2. **The boundary is a control plane.** Tool authority comes from a reviewable
>    policy, suspicious tool *results* can be held out of context, and every
>    verdict is auditable. Refusing an irreversible action does not require a
>    classifier to recognize the bad text first.
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
against this table and the category hierarchy in the [naming authority](naming.md):

| Claim | Sentence | Paragraph | Page |
|---|---|---|---|
| Agent runtime is the public category | ✔ anchor | ✔ | ✔ |
| Fused Agent Kernel / agent kernel is the technical architecture | ✔ | ✔ | ✔ |
| Operator-controlled cache/context, routing, authority, memory, observability, native inference boundary | ✔ | ✔ mechanism | ✔ unpacked |
| Tool call as syscall (model proposes, kernel disposes) | ✔ anchor | ✔ | ✔ |
| One static Go binary, drop-in | ✔ | ✔ (`fak manage -- claude`) | ✔ |
| Controlled: verdicts, policy, quarantine | "controlled" | ✔ mechanism | ✔ + classifier caveat |
| Cheaper: reuse, tuned ~4.1×, `max\|Δ\| = 0` | "cheaper" | ✔ numbers | ✔ numbers + naive-multiplier fence |
| Faster: ~362 ns in-process decision | "faster" | ✔ number | ✔ number |
| Verify, don't trust (DOS) | — (no room) | — | ✔ |
| Novelty fence (0/29 — assembly, not invention) | — | ✔ | ✔ |

Hard rules, inherited from the
[epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md): quote the tuned
**~4.1×**, never the naive multiplier unlabeled; no market-adoption claims; no
benchmark not already run; label simulated as simulated. The category names an
owned control boundary; it does not make roadmap-only orchestration or runtime
features shipped. [`CLAIMS.md`](../../CLAIMS.md) remains the status authority.

## Related framing artifacts

- [How to find and name fak](naming.md) — the public category, technical name,
  and disambiguated search terms the pitch should use (dimension I).
- [Objections & one-line answers](objections.md) — what to say after the pitch
  lands and the pushback starts (dimension I).
- [Social storyboard](social-storyboard.md) — the pitch cut into a
  card-per-concept thread (dimension J).
- [The 4× that's real](stories/the-real-4x.md) — the long-form honest telling
  of the headline number (dimension H).
