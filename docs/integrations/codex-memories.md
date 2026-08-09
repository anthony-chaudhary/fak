---
title: "Codex Memories and fak"
description: "How fak treats the experimental Codex Memories feature: useful cross-session suggestions, but untrusted context that is never closure or test evidence."
---

# Codex Memories and fak

Codex 0.144.1 exposes **Memories** as one experimental, default-off feature. Its
current description combines two behaviors: creating memories from conversations and
bringing relevant memories into later conversations. It does not expose supported,
independent `use-only` and `generate` controls.

## Surface map

| Surface | Role | Trust posture |
|---|---|---|
| `AGENTS.md` | Checked-in contributor contract | Authoritative repository guidance |
| Codex Memories | Experimental cross-session suggestions | Untrusted context; never closure/test evidence |
| fak recall/core images | Kernel-managed resumable context | Revalidate against current repo state |
| `fak memory` (the `memq` query algebra) | Query/inspection boundary | Read-back aid, not proof by itself |
| Skills | Checked-in reusable workflow | Reviewed and versioned |
| MCP/tools | Effectful capabilities | Governed by fak policy and independent witnesses |
| Chronicle | Historical observation | Provenance-bearing context, not current truth |

## Safe defaults

1. Keep Memories disabled for sensitive/private-control sessions.
2. Never treat a generated memory as proof that code shipped, tests passed, or an issue closed.
3. Do not store private hardware/control facts, credentials, raw tool output, or operator-only status.
4. Promote a stable recurring workflow into `AGENTS.md`, docs, or a skill only after independent review.
5. Revalidate every recalled claim against the current tree and external witness.

## Dogfood result

The scrubbed isolated-home study is recorded in
[`experiments/agent-live/codex-memories-dogfood-2026-07-11.json`](../../experiments/agent-live/codex-memories-dogfood-2026-07-11.json).
The enabled postures scheduled a `memory_consolidate_global` job, but it ended in an
error and produced zero stage-1 outputs; a later reuse probe received no memory. The
honest value verdict is therefore **not yet**, not a positive memory claim.
