---
title: "Cross-wire launch posture in a non-FAK repository — 2026-08-19"
description: "Verdict: launch posture now distinguishes provider-neutral runtime effects from Anthropic-only request shaping and explicitly names the passthrough ownership..."
---
# Cross-wire launch posture in a non-FAK repository — 2026-08-19

**Verdict:** launch posture now distinguishes provider-neutral runtime effects from Anthropic-only request shaping and explicitly names the passthrough ownership boundary.

- On guarded Codex/OpenAI, stale-read elision is active through decoded provider-neutral history. `TestDecodedStaleReadElisionReachesOpenAIPlannerInput` remains the behavioral wire witness.
- Cross-backend vCache signals are active independently of request anchoring: normalized provider usage feeds cache-read/cache-write counters, warmth prediction error, and per-family economics. `TestCodexResponsesUsageFeedsVCacheObservation` proves the OpenAI Responses path.
- Cold-tool deferral remains honestly inert on OpenAI/Codex because the current implementation requires Anthropic ToolSearch; no incompatible discovery feature is emulated.
- Passthrough `fak serve` reports bounded code tools inert because the client owns tool execution; native serve and `fak agent` own and arm the bounded catalog.
- Stable JSON now includes the `vcache-signals` mechanism, separating observed feedback from `vcache-anchor` request shaping and calibration freshness.

Structured artifact: [`launch-posture-cross-wire-non-fak-2026-08-19.json`](../_witnesses/launch-posture-cross-wire-non-fak-2026-08-19.json).
