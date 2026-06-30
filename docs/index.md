---
title: "fak — the agent kernel | cheaper long sessions, the right model per call, a safe floor for AI agents"
description: "fak is one Go binary you put in front of the AI agent you already run — Claude Code, Codex, Cursor, or any OpenAI / Anthropic / MCP client. Cheaper long sessions, the right model per call, fewer wasted turns, an auditable trail — and a hard security floor when you need one."
---

# fak — the Fused Agent Kernel (agent kernel)

**Put one Go binary in front of the AI agent you already run. Keep your model, your IDE, and your tools — gain a handle on the parts of a real agent loop that get expensive or go wrong.**

<!-- hero video — the headline benchmarks as a ~40s reveal. Assets live in /visuals
     (outside the Pages /docs root), so they are referenced by absolute raw URL, the
     same convention as the social-preview image in _config.yml. A real HTML page can
     autoplay the muted mp4; the gif is the fallback for engines that block <video>. -->
<div align="center">
  <video
    src="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/hero-video.mp4"
    poster="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/hero-video-poster.png"
    autoplay loop muted playsinline preload="metadata"
    width="100%" style="max-width:960px;border-radius:12px"
    aria-label="fak — the agent kernel: a ~40 second model-card reveal of the headline benchmarks — the performance spectrum, the turn-tax curves (modeled 9.7x vs the naive floor), the capability matrix, the three-pillar stat card with its honest single-stream fence, and the eight-axis sweep">
    <img
      src="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/hero-video.gif"
      alt="fak headline benchmarks — an animated ~40-second reveal"
      width="100%" style="max-width:960px;border-radius:12px"/>
  </video>
  <br/>
  <sub>The headline benchmarks as a ~40-second reveal · <a href="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/hero-video.mp4">full-resolution MP4</a></sub>
</div>

`fak` is one Go binary you put in front of the AI agent you already run — Claude
Code, Codex, Cursor, or any OpenAI / Anthropic / MCP client. You keep your model,
your IDE, and your tools. You point one base URL at `fak`, and it gives you a handle
on the parts of a real agent loop that get expensive or go wrong:

- **Cheaper long sessions.** A 100k-token conversation re-sends its whole transcript
  every turn. `fak` sheds the old turns while keeping the provider's prompt-cache
  prefix byte-identical, so the discount survives instead of breaking.
- **The right model per call.** Send an easy read to a cheap model and a write-shaped
  call to a careful one, chosen per tool call rather than per whole request.
- **Fewer wasted turns.** A repeated read served locally, a malformed call repaired in
  place, a dead-end branch refused before the agent spends a turn on it.
- **A trail you can audit.** Every decision is a plain verdict (`ALLOW`, `DENY`,
  `TRANSFORM`, or `QUARANTINE`) in JSON logs, an optional hash-chained journal, and
  Prometheus metrics.

> **In one line:** put `fak` in front of the agent you already run. It makes long
> sessions cheaper, routes each call to the right model, keeps unsafe tool results out
> of context, and records every verdict. One binary, no rewrite, no key to start.

It does this by sitting on the tool-call path as a **kernel**: the model *proposes* a
call; `fak` decides whether that call exists, whether its arguments are allowed, whether
the result may enter context, and what gets reused. The same boundary that saves you
tokens is also where a dangerous call gets refused, which is why teams who need a hard
security floor reach for it too ([see below](#for-security-teams)).

<!-- agent-kernel explainer video — the boundary / how-it-works story (P1) as a ~44s reveal,
     built by tools/hero_video_gen.py from the in-repo deterministic diagrams (nothing
     generated). Same /visuals raw-URL convention as the hero video above. -->
<div align="center">
  <video
    src="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/agent-kernel-video.mp4"
    poster="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/agent-kernel-video-poster.png"
    autoplay loop muted playsinline preload="metadata"
    width="100%" style="max-width:960px;border-radius:12px"
    aria-label="fak — the agent kernel: a ~44 second explainer reveal — the agent-kernel card, the tool call as a syscall, the five-gate flow through the kernel, the two-gate security model, the Context MMU, shared-prefix KV, and the 2-D scheduler">
    <img
      src="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/agent-kernel-video.gif"
      alt="fak agent kernel — an animated ~44-second explainer"
      width="100%" style="max-width:960px;border-radius:12px"/>
  </video>
  <br/>
  <sub>How the boundary works — the agent kernel as a ~44-second reveal · <a href="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/agent-kernel-video.mp4">full-resolution MP4</a></sub>
</div>

[**▶ Try the live demos**](demos.html){: .btn } ·
[Open the modular Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb){: .btn } ·
[Install (1 line, no clone)](https://github.com/anthony-chaudhary/fak/blob/main/INSTALL.md){: .btn } ·
[Get started](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md){: .btn } ·
[See the showcase](showcase.html){: .btn } ·
[Read the FAQ](FAQ.md){: .btn } ·
[GitHub repository](https://github.com/anthony-chaudhary/fak){: .btn }

> **▶ See it run.** Start with no-key, no-model demos that run anywhere Go runs:
> `go run ./cmd/dropindemo -print`, `go run ./cmd/guarddemo -print`, or
> `go run ./cmd/tokendemo -print`. Then move to the dedicated security,
> research/science, adoption, memory/serving, and hosted live-model tracks.
> **[Open the demos →](demos.html)** Or open the
> [modular Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb)
> for policy proof, HTTP adjudication, offline value measurement, and an optional
> GPU-backed gateway case. You can also [run your own copy](run-the-demos.md)
> locally, in a container, or on your own cloud VM.

---

## What fak does

The everyday wins first — the reasons most people put `fak` in front of an agent:

- **Cheaper long sessions.** A long conversation re-sends its whole transcript every
  turn, and the provider only discounts it while the cached prefix stays byte-for-byte
  the same. `fak` sheds the un-cacheable middle turns by splicing on the original bytes
  (a memcpy, never a re-marshal), so the prompt-cache discount survives instead of
  breaking. It guarantees prefix byte-identity, and relays the provider's cache number
  rather than claiming it.
- **The right model per call.** `fak route` routes an *aspect* (a tool call, a
  reasoning step, a stage) to a different model, with first-class ensembles
  (`vote`, `best_of`). An easy read goes to a cheap model; a write-shaped call goes to
  a careful one.
- **Fewer wasted turns.** A repeated read is served locally, a malformed call is
  repaired in place, and a dead-end branch is refused before the agent spends a turn on
  it. Shared work is computed once because the KV cache is a kernel object, not a rented
  one.
- **A trail you can audit.** Every decision is a plain verdict (`ALLOW`, `DENY`,
  `TRANSFORM`, or `QUARANTINE`) in JSON logs, an optional hash-chained journal, and
  Prometheus metrics.

And the security floor, for the teams who need one (more in
[For security teams](#for-security-teams)):

- **Stops prompt injection and tool poisoning by structure.** Suspicious tool
  *results* are quarantined out of the model's context entirely; dangerous tools are
  never on the allow-list. Two independent gates, not one evadable classifier.
  Addresses the OWASP Agentic Top-10 and the MCP Top-10 (Tool Poisoning, Memory
  Poisoning).
- **Default-deny capability security.** The permission policy runs *inside* the
  kernel, on the same call path as the tool call. It fails **closed**, not open.
- **Addressable, bit-exact KV cache.** Evict one span from the middle of a kept
  model run — a poisoned result, an expired secret — and leave the cache
  bit-for-bit identical to a run that never saw it (`max|Δ| = 0`). No shipped serving
  engine offers mid-run causal eviction.
- **Cache-efficient agent fleets.** ~4× fewer tokens than a tuned warm-cache stack
  on a 50-turn × 5-agent run; 8.8–9.7× modeled prefill elimination vs the naive
  floor over the real WebVoyager web-agent set (1.0–1.1× vs a tuned per-agent KV).

## See each win in one example

Each idea shrinks to a single worked example. The numbers trace to the
[benchmark authority](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md);
the live versions run on the [demos page](demos.html). Or watch the worked examples as a
[~25-second reveal](showcase.html#by-example).

- **A poisoned turn, removed mid-run.** Quarantine evicts a tool result's K/V from the *middle* of the
  kept run and re-seats every survivor, leaving the cache bit-identical to one that never saw it
  (`max|Δ| = 0`). → [Watch a turn vanish](explainers/addressable-kv-cache.md#a-worked-example-watch-one-turn-vanish-bit-for-bit)
- **More tool calls, more turns saved.** On one 14-call agent trace a naive loop is forced into 9 extra
  model round-trips and a tuned 2026 framework into 5; the kernel resolves them in-syscall, for 0. →
  [The turn that never fires](concepts-and-story.md#three-worked-examples-more-turns-more-agents-more-tool-calls)
- **Pay the shared prefix once.** 5 agents × 50 turns is 250 chances to re-read the setup: naive pays
  250×, a tuned warm cache 5×, fak once: 4.1× vs tuned, 62.0× fewer prefill tokens. →
  [The setup-payments table](concepts-and-story.md#three-worked-examples-more-turns-more-agents-more-tool-calls)
- **More hooks, sooner.** Four checks across 1,000 tool calls is ~28 s of gate latency if you spawn a
  hook per check, or ~10 ms in-process, which is what makes fail-closed the default. →
  [The cost of checking everything](explainers/policy-in-the-kernel.md#a-worked-example-the-cost-of-checking-everything-every-time)

## What fak is not

`fak` is **not** a drop-in replacement for tuned token engines. Use vLLM, SGLang,
llama.cpp, or a hosted provider when raw tokens/sec is the job, and put `fak serve`
in front for the agent boundary: which effects are allowed, which results may enter
memory, when reuse is still legal, what gets audited, and what survives a session
boundary. The in-kernel model path is a correctness/reference engine with narrow
witnessed performance rungs; broad serving-speed claims need a benchmark-authority
row, not a slogan.

## For security teams

If a hard capability floor is *why* you're here — not just a nice-to-have — this is the
load-bearing idea, kept in full.

**Treat the model like an untrusted program, and the tool call like a syscall: the
model proposes, the kernel disposes.** Most agent security tries to recognize bad text.
Recognizers help; they are not the floor. Prompt injection is a text game, and attackers
get turns too. `fak` moves the load-bearing decision to the capability floor: a dangerous
tool outside the allow-list cannot be called, no matter what the model was told.

Two independent gates matter:

- **Call-side gate:** tool names and selected arguments are checked before dispatch, on
  the same call path as the tool call (one address space, no IPC, `default-deny`). A
  denied call never reaches the tool runner, and a check that crashes or times out fails
  **closed**.
- **Result-side gate:** tool output is screened before it enters context. A poisoned or
  secret-bearing result is paged out or quarantined instead of being handed back to the
  model as trusted text. The detector is treated as evadable by design, a bonus rather
  than the floor; the floor is the dangerous lever simply not existing.

The capability floor is the guarantee. Irreversible effects are unwired by default;
untrusted bytes have to pass a gate before they become model context. Read
[Policy in the kernel](explainers/policy-in-the-kernel.md),
[POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md), and
[the security model](https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/security.md).

## Try it in 2 minutes (no key, no model, no GPU)

Get the binary — no clone, no Go toolchain. The installer detects your OS/arch,
downloads the prebuilt static binary for the latest release, verifies its checksum, and
drops `fak` on your PATH:

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak version          # prints the installed version, e.g. 0.34.0
```

Now prove the floor from the bare binary — these need no clone and no `examples/` dir:

```bash
fak preflight --tool refund_payment --args "{}"   # -> DENY  (DEFAULT_DENY): unknown tool, fail-closed
fak preflight --tool search_kb      --args "{}"   # -> ALLOW: a read-shaped name is not blanket-blocked
fak agent --offline                               # runs one task twice — tools wired directly vs. behind fak — and prints the before/after
```

The dangerous action is refused by structure, before any model interpretation matters.
Then wrap the agent you already run — one command, no rewrite, no key to start:

```bash
fak guard -- claude          # or: fak guard --provider openai -- opencode
```

> **Have the source already?** From a clone you can skip the install and run the same
> proof against a named example floor, where the deny is by *argument value*:
> `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"`.
> Full paths in [INSTALL.md](https://github.com/anthony-chaudhary/fak/blob/main/INSTALL.md)
> (one-line installer · manual download · Docker · build-from-source · Windows).

## Learn more

| If you want… | Read |
|---|---|
| **The principles fak is built to satisfy** | [Charter](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/CHARTER.md) |
| **Structured-output decoding SOTA + fak's ride-mode surface (#907)** | [Research note](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/RESEARCH-structured-output-decoding-2026-06-26.md) |
| **Keeping a stable core as models × backends × features multiply** | [Combinatorial-growth epic](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/COMBINATORIAL-GROWTH-EPIC-2026-06-27.md) |
| **The quick answers** | [FAQ](FAQ.md) |
| **A guided first run** | [Tutorial](fak/tutorial.md) |
| **What the words mean** (preflight vs inflight vs prefill; cache rebate / net saving) | [Glossary](glossary.md) |
| **How shared state is split** | [Shared state ladder](shared-state-ladder.md) |
| **A collaborative task state contract** | [Shared task record contract](shared-task-record-contract.md) |
| **How agents discover fak features and memory tools** | [Self-feature query spine](notes/SELF-FEATURE-QUERY-SPINE-2026-06-30.md) |
| **The two core ideas** | [Policy in the kernel](explainers/policy-in-the-kernel.md) · [Addressable KV cache](explainers/addressable-kv-cache.md) |
| **Why a cache-hit % isn't the whole story** | [Context signal-to-noise](explainers/context-signal-to-noise.md) |
| **How fak runs the agent as nested loops** | [Engineering is building loops](explainers/engineering-is-building-loops.md) |
| **Every benchmark number** | [Benchmark authority](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md) |
| **Everything fak supports** | [What fak supports](supported/README.md) — models · features · clouds · APIs/MCP · harnesses · engines |
| **Every machine fak runs on** | [Hardware matrix](HARDWARE-MATRIX.md) (4 platforms · 2 CPU ISAs · 4 GPU backends) |
| **How fak serves at scale** | [Serving plans](serving/README.md) — dual-track · poly-model · hardware-aware & regenerable KV |
| **What's real, what's not** | [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) |
| **The leadership snapshot** (wins · live goal · risks · the one decision) | [Executive roll-up](EXECUTIVE-ROLLUP.md) |
| **A machine-readable map (for LLMs)** | [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) |

---

<sub>License: Apache-2.0 · [Report a vulnerability](https://github.com/anthony-chaudhary/fak/blob/main/SECURITY.md) · Keywords: Fused
Agent Kernel, fak agent kernel, fak serve, fak-certified, agent kernel, agent tool
firewall, AI agent security, prompt injection defense, tool poisoning, capability
security, default-deny permission gate, treat the tool call like a syscall, KV cache,
addressable KV cache, self-hosted LLM, LLM agent fleet, agentic AI, Go.</sub>

<!-- BREADCRUMB-JSONLD:BEGIN (generated by tools/gen_structured_data.py — do not edit by hand) -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "https://anthony-chaudhary.github.io/"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "fak documentation",
      "item": "https://anthony-chaudhary.github.io/fak/"
    }
  ]
}
</script>
<!-- BREADCRUMB-JSONLD:END -->
