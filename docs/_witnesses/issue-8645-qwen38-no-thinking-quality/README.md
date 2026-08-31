---
title: "Issue #8645 — exact-target no-thinking quality gate"
description: "Frozen 30-task campaign and conservative HOLD until exact Qwen3.8-27B BF16 TP2 evidence exists."
---
# Issue #8645 — exact-target no-thinking quality gate

**Current routing decision: HOLD. Keep thinking enabled outside the arithmetic envelope proven by #8623.**

`campaign.json` freezes 30 deterministic tasks across coding, correlated tool calls,
structured JSON, instruction following, safety/adversarial prompts, and short factual work.
Each task is paired across `enable_thinking=true` and `enable_thinking=false` on the exact
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` BF16 TP2 target. The
campaign pins decoding controls, arm-neutral validation, required raw trial fields, family
quality floors, uncertainty gates, and rollback criteria.

This artifact does **not** claim the campaign ran. The exact target remains blocked on this
control host: the authorized private bridge is installed but lacks its control credentials,
and the sanctioned GCP route requires non-interactive credentials to be refreshed. The local
OpenAI-compatible endpoint exposes `gpt-5.6-sol`, not the pinned Qwen target. No synthetic or
substitute-model output may fill the result fields.

## Decision contract

- Global promotion requires every family to reach a 0.95 pass rate, a Wilson lower bound of
  at least 0.80, and a no-thinking minus thinking pass-rate delta no worse than -0.05.
- A speed metric counts only for pairs where both arms pass their validator.
- A family may be promoted only after its exact-target row clears every family gate; all other
  families retain thinking.
- Any incomplete raw stream, environment mismatch, validator drift, or failed family gate
  retains or restores thinking.

## Evidence lifecycle

- **Promotion evidence:** a complete interleaved run on the pinned BF16 TP2 target with raw
  streams and all required measurements, followed by a typed global or family-scoped decision.
- **Demotion/retirement evidence:** a repeated exact-target family result below any quality
  threshold, or evidence that the frozen validators no longer represent the declared family.
- **Invalidating assumption:** these 30 deterministic tasks predict quality for the broader
  operator task families named by #8645.

The smallest next checkable step is to restore either sanctioned compute credential path, run
`campaign.json` without changing prompts or validators, and publish the raw paired rows plus the
computed family confidence intervals. Until then, the machine-readable decision is `HOLD`.
