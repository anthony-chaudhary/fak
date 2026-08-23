---
name: sota-check
description: One repeatable pass that stops fak from re-inventing known kernel art - before writing or optimizing a compute kernel (a quantized GEMM, a fused attention, a KV-cache reuse, a MoE dispatch, a Metal/CUDA kernel), it...
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
