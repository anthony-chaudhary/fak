---
title: "fak concept card — one page, five ideas, two commands (printable, PDF-ready)"
description: "The single printable page you hand someone at a meetup: fak's five core concepts (treat the tool call like a syscall; verify don't trust; the addressable bit-exact KV cache; the default-deny capability gate + prompt-injection quarantine; the one static Go binary), each in one sentence with the diagram that carries it, plus the install one-liner and the 60-second proof command. Designed to render to one printed page; witnessed numbers only (the tuned ~4.1x, ~362 ns guard tax, max|delta|=0); honest fences up front."
slug: fak-concept-card
keywords:
  - fak concept card
  - agent kernel cheat sheet
  - tool call as syscall
  - default-deny capability gate
  - bit-exact KV cache
  - one printable page
  - PDF-ready concept card
  - agent infra one-pager
date: 2026-07-03
---

# fak concept card — one page, five ideas, two commands

> **Print this page. Hand it to someone at a meetup.** fak is one ~13 MB static Go
> binary (Apache-2.0) that sits in front of an AI agent and adjudicates every tool call
> like a syscall. The five ideas below are the whole pitch; the two commands prove them in
> 60 seconds with no key, model, or GPU.
>
> Dimension **B — Visual & diagram assets** of the
> [concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md). The
> sibling artifacts: the [pitch ladder](pitch-ladder.md) (1 sentence / 1 paragraph / 1
> page), the [objections card](objections.md), and the
> [social storyboard](social-storyboard.md). This page is the *printable* one-pager that
> compresses all five onto a single sheet.

## The five ideas

**1. Treat the tool call like a syscall.** The model proposes, the kernel disposes — every
tool call is a *request* that crosses a checkpoint, not an action that runs.
→ [syscall-flow diagram](diagrams/syscall-flow.svg)

**2. Verify, don't trust (DOS).** A false "done" is refused from git evidence, not the
agent's word; refusals carry a structured reason from a closed vocabulary; recalled memory
is re-checked at read time.
→ [forgeable-vs-witnessed diagram](diagrams/forgeable-vs-witnessed.svg)

**3. Addressable, bit-exact KV cache.** Reach into the middle of a kept run, evict one
span, leave the cache bit-for-bit identical (`max|Δ| = 0`) — so a poisoned result is cut
out *after* the model saw it without re-prefilling everything after it.
→ [cost-curve diagram](diagrams/cost-curve.svg)

**4. Default-deny capability gate + prompt-injection quarantine.** Refusing an irreversible
action never depends on *catching* an attack — the lever was never wired; and suspicious
tool *results* are held out of context by structure, not by a classifier.
→ [the quarantine arm of the syscall-flow diagram](diagrams/syscall-flow.svg)

**5. One static Go binary, drop-in.** Repoint one base URL; the same ~13 MB binary a dev
runs on a laptop is what a platform team hardens for a fleet — no external deps, no
`go.sum`, no sidecar.
→ [single-binary diagram](diagrams/single-binary.svg)

## Install (one line)

```
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

## The 60-second proof (no key, no model, no GPU)

```
fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
fak agent --offline                                                                                       # -> injection YES→no, destructive YES→no, task still done
```

The first command is denied *by structure* — no model in the loop. The second proves it is
not a blanket block. The third shows both gates inside a full agent loop.

## Honest fences (read these before you repeat the pitch)

- **Not a faster token engine.** vLLM / SGLang / llama.cpp win raw throughput; fak fronts a
  token engine, it does not replace one. The reuse win is **self-host + read-heavy only**
  (~4.1× vs a *tuned* warm-cache stack — never lead with the naive multiplier).
- **The injection detector is ~100% evadable by design** — it is a tripwire, not the floor.
  The floor is the two structural gates in ideas 1 and 4.
- **0/29 prior-art novel.** Every primitive is established engineering; the contribution is
  the *assembly* — one in-process gate where the tool call is the checkpoint.
- Power / energy / $ figures are **simulated**; the ~60× reuse ceiling and the "agent city"
  frontier are **design targets, not measurements** — never present either as a shipped result.

## Go deeper

[README](../../README.md) · [CLAIMS ledger](../../CLAIMS.md) (every claim tagged
`[SHIPPED]` / `[SIMULATED]` / `[STUB]`) · [BENCHMARK-AUTHORITY](../../BENCHMARK-AUTHORITY.md) ·
repo: https://github.com/anthony-chaudhary/fak · live demos: https://anthony-chaudhary.github.io/fak/demos.html

---

> **Single-page layout note.** This file is written to fit one printed page: the five
> ideas, the install line, the proof block, and the fences. Everything below this rule is
> repo wiring (index entry, verify commands, scope statement) and is not part of the
> handout. Export the body above the rule to PDF with any Markdown→PDF tool
> (`pandoc concept-card.md -o concept-card.pdf`, a browser's "Print to PDF", etc.); the
> content density above is tuned to one A4/Letter sheet at default margins.

## Verify

```
# front-matter present and this doc is indexed
grep -n "^title:" docs/adoption/concept-card.md
grep -n "concept-card" INDEX.md

# the SEO/AEO scorecard does not regress for this new doc
python tools/seo_aeo_scorecard.py

# the two proof commands the card promises actually fire (verified 2026-07-03)
fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> verdict=DENY reason=POLICY_BLOCK
fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> verdict=ALLOW reason=NONE
```

## What this doc does and does not claim

- **Does:** provide a single, self-contained, printable one-pager that compresses all five
  core concepts (one sentence each + its carrying diagram), the install one-liner, and the
  60-second proof — honesty-fenced so a stranger can repeat the pitch without overclaiming.
- **Does not:** assert fak is adopted, invent a benchmark, lead with the naive perf
  multiplier, or present the injection detector as the security floor. Every witnessed
  number traces to the ledger; the aspirational ones (~60×, "agent city", power/$) are
  labeled design-target / simulated.
