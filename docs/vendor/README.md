---
title: "Vendor integration paths"
description: "Entry point for accelerator vendors, neo-cloud operators, and foundation-model product teams evaluating fak as an agent-kernel binding layer."
---

# Vendor integration paths

This directory is the vendor-facing map for teams that want fak's binding layer
without becoming fak maintainers. The shared idea is narrow: implement or route
through the boundary fak already owns, then inherit the policy, audit, cache,
context, and parity machinery above it.

## Pick the path that matches your team

| You are | Start here | Why |
|---|---|---|
| Accelerator or chip team | [Neo-silicon onboarding](neo-silicon-onboarding.md) | Shows how a backend plugs into `internal/compute.Backend` and what still needs conformance evidence. |
| Neo-cloud operator | [Neo-cloud reference architecture](neo-cloud-reference-architecture.md) | Shows one control plane over heterogeneous accelerator pools without pretending cross-vendor scheduling is solved. |
| Foundation-model product team | [Anthropic pilot pitch](anthropic-internal-pitch.md) and [OpenAI pilot pitch](openai-internal-pitch.md) | Shows fak as a governed tool-call and audit boundary around existing agent products. |
| Enterprise Claude Code rollout owner | [Claude Code managed rollout](claude-code-managed-rollout.md) | Turns the pilot pitch into managed MCP, policy, audit, and rollback controls. |

Reader-safe public landing pages live outside this excluded vendor directory:
[supported silicon backends](../supported/silicon-backends.md) and
[heterogeneous silicon fleets](../serving/heterogeneous-silicon-fleet.md).

## Search terms this page intentionally owns

Use these terms when linking or describing the binding-layer path:

- `fak backend conformance`
- `fak-certified backend`
- `neo-silicon agent kernel`
- `bring-your-accelerator agent serving`
- `vendor-neutral inference backend`
- `heterogeneous accelerator agent kernel`

The bare word `fak` is noisy. Pair it with one of the terms above or with the
repo's existing agent-kernel language.

## Current honesty fence

The compute HAL seam is real: `internal/compute.Backend`, `Caps`, the runtime
registry, `cpu-ref`, CUDA, and Vulkan are in the repo, with the status and
witness trail summarized in
[hardware portability](../explainers/hardware-portability.md).

The vendor product surface is still being filled in:

- Backend Conformance Kit: planned in issue #1684.
- `fak backend scaffold <name>` generator: planned in issue #1685.
- Hardware-shape-neutral scorecard dimension: planned in issue #1688.
- AEO/marketing generator wiring for these vendor terms: planned in issue #1689.

Until those land, these pages are routing and integration guidance, not a
`fak-certified` mark.
