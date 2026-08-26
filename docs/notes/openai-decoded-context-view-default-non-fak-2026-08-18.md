---
title: "Default OpenAI decoded context view in a non-FAK repository — 2026-08-18"
description: "Verdict: OpenAI/Codex launches already carry a default-on provider-neutral context planner;"
---
# Default OpenAI decoded context view in a non-FAK repository — 2026-08-18

**Verdict:** OpenAI/Codex launches already carry a default-on provider-neutral context planner; launch posture now reports that real seam instead of implying all compaction is inert because the Anthropic byte transform cannot apply.

- Guard and serve default `--ctx-view-budget` to `8000` tokens through `agent.DefaultCtxViewBudget`.
- `TestCtxViewHTTPOnPlansHistoryOnTheWire` captures the live OpenAI-compatible upstream request: the planned wire view contains fewer messages and fewer estimated tokens than the full transcript, preserves pinned system/tool/latest-user spans, and records the rewrite in `CtxValueReport`.
- `fak doctor launch-posture` now reports a separate `decoded-context-view` mechanism as active on OpenAI-compatible, owned-model, and buffered Anthropic paths; `--ctx-view-budget 0` independently disables it.
- Anthropic `compact-history` remains separately named and honestly inert on OpenAI, avoiding a false equivalence between byte-preserving provider-cache compaction and provider-neutral decoded planning.

Structured artifact: [`openai-decoded-context-view-default-non-fak-2026-08-18.json`](../_witnesses/openai-decoded-context-view-default-non-fak-2026-08-18.json).

This retires the provider-neutral compaction part of #8089, not stale-read elision, cold-tool deferral, or cross-backend vCache signaling.
