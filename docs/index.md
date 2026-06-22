---
title: "fak — the agent kernel | default-deny permission gate + addressable KV cache for AI agents"
description: "fak is an agent kernel for self-hosted LLM agent fleets: an in-process, default-deny permission gate fused with an addressable, bit-exact KV cache. Prompt-injection and tool-poisoning containment, capability security, and cache-efficient inference, in Go."
---

# fak — the agent kernel

**Treat the model like an untrusted program, and the tool call like a syscall.**

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
    aria-label="fak — the agent kernel: a ~40 second model-card reveal of the headline benchmarks — the performance spectrum, the turn-tax curves (measured 9.7x), the capability matrix, the three-pillar stat card with its honest single-stream fence, and the eight-axis sweep">
    <img
      src="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/hero-video.gif"
      alt="fak headline benchmarks — an animated ~40-second reveal"
      width="100%" style="max-width:960px;border-radius:12px"/>
  </video>
  <br/>
  <sub>The headline benchmarks as a ~40-second reveal · <a href="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/hero-video.mp4">full-resolution MP4</a></sub>
</div>

`fak` is an **agent kernel** (an *agent tool firewall*): an in-process,
**default-deny permission gate** for AI agents, fused with an **addressable,
bit-exact KV cache**, written in **Go**. Every tool call an agent makes passes
through a kernel the model doesn't control — the same boundary that enforces
**security** (which effects are allowed, which tool results may enter the model's
context) also drives **performance** (do shared work once instead of every turn).

> **In one line:** prompt-injection containment, capability security, and
> cache-efficient inference for **self-hosted LLM agent fleets** — at one boundary.

[Get started](../GETTING-STARTED.md){: .btn } ·
[See the showcase](showcase.html){: .btn } ·
[Try the live demos](demos.html){: .btn } ·
[Read the FAQ](FAQ.md){: .btn } ·
[GitHub repository](https://github.com/anthony-chaudhary/fak){: .btn }

---

## What fak does

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
  on a 50-turn × 5-agent run; 8.8–9.7× measured prefill elimination on real
  WebVoyager web-agent workloads.

## What fak is not

`fak` is **not** a faster model server. vLLM, SGLang, and llama.cpp win raw throughput
and front-of-prompt prefix caching, and `fak` doesn't try to beat them — it owns the
orthogonal questions they don't: which effects are allowed, which results may enter
memory, when reuse is still legal, and what survives a session boundary. You can run
`fak serve` in front of any of them.

## Try it in 2 minutes (no key, no model, no GPU)

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak agent --offline
```

`refund_payment` returns `DENY (POLICY_BLOCK)`; `agent --offline` runs the same task
twice — tools wired directly vs. behind `fak` — and prints the before/after.

## Learn more

| If you want… | Read |
|---|---|
| **The quick answers** | [FAQ](FAQ.md) |
| **A guided first run** | [Tutorial](fak/tutorial.md) |
| **The two core ideas** | [Policy in the kernel](explainers/policy-in-the-kernel.md) · [Addressable KV cache](explainers/addressable-kv-cache.md) |
| **Every benchmark number** | [Benchmark authority](../BENCHMARK-AUTHORITY.md) |
| **Every machine fak runs on** | [Hardware matrix](HARDWARE-MATRIX.md) (4 platforms · 2 CPU ISAs · 4 GPU backends) |
| **What's real, what's not** | [Claims ledger](../CLAIMS.md) |
| **A machine-readable map (for LLMs)** | [llms.txt](../llms.txt) |

---

<sub>License: Apache-2.0 · [Report a vulnerability](../SECURITY.md) · Keywords: agent
kernel, agent tool firewall, AI agent security, prompt injection defense, tool
poisoning, capability security, default-deny permission gate, KV cache, addressable
KV cache, self-hosted LLM, LLM agent fleet, agentic AI, Go.</sub>
