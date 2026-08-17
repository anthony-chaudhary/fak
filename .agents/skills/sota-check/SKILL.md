---
name: sota-check
description: One repeatable pass that stops fak from re-inventing known kernel art - before writing or optimizing a compute kernel (a quantized GEMM, a fused attention, a KV-cache reuse, a MoE dispatch, a Metal/CUDA kernel), it checks the SOTA prior-art matrix for the production reference (llama.cpp / Marlin / CUTLASS / FlashInfer / vLLM / SGLang / a named paper), decides the route (borrow / bind / stay-minimal), holds the result to the named oracle, and records what was consulted in a Prior-art trailer. Runs `fak sota <op|file>` to surface the reference, reads it, routes deliberately, and (when the matrix has a blind spot) adds the missing row so the next person inherits the map. The inward kernel-engineering counterpart of industry-score (the outward field map). Use before any kernel-optimization commit, when the PRIOR_ART advisory gate fires, when onboarding a new compute operation, or on a /loop cadence to keep the matrix honest against the tree. Preserve useful cohort-specific alternatives as bounded optional modules or recipes when they remain supportable, while excluding genuinely superseded approaches.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/sota-check/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/sota-check/SKILL.md`](../../../.claude/skills/sota-check/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.
