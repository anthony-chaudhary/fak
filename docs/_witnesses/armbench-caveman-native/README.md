---
title: "Caveman native control — captured witness (2026-08-14)"
description: "Documentation for Caveman native control — captured witness (2026-08-14), including the captured behavior, operating context, and reproducible fak evidence."
---

# Caveman native control — captured witness (2026-08-14)

## Verdict

The pinned upstream benchmark could not be run on its exact snapshot through the available direct provider account: `claude-sonnet-4-20250514` was rejected as unsupported. The verb records this in `exact-model-unavailable.txt`; no exact-model reproduction value is claimed.

A separately labeled replacement run used `gpt-5.6-sol`, directly against the same provider endpoint for both arms. fak was absent from both inference paths. The 60 complete raw provider responses and paired texts are in `replacement-gpt-5.6-sol/manifest.json`.

## Pinned contract

- source: `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`
- `benchmarks/prompts.json` SHA-256: `773e557f9187363c44e7e5aae2d27268720bcd8772865e119825078b06da93d7`
- `skills/caveman/SKILL.md` SHA-256: `daf9cec496ebd039809d8236f99f17fa1b4beaadf8ce4e2d532d0da51d70afce`
- `benchmarks/run.py` SHA-256: `530a387918418713e64ded97794f41a1ffe6a01e833a69d2cb447bf4640facce`
- unchanged generation settings: temperature 0, maximum output 4096, three trials per prompt and arm
- normal system: `You are a helpful assistant.`; Caveman system: imported `SKILL.md`

## Replacement result versus upstream

| Source | Model | Normal mean of prompt medians | Caveman mean of prompt medians | Output reduction | Semantic gate |
|---|---|---:|---:|---:|---:|
| upstream README | `claude-sonnet-4-20250514` | 1214 | 294 | 65% (published, rounded) | unevaluated upstream |
| reproduced replacement | `gpt-5.6-sol` | 947.3 | 520.2 | 45.1% | 60/60 pass |

The replacement produced 266.7 fewer normal-arm output tokens than upstream (-22.0%), but 226.2 more Caveman-arm tokens (+76.9%), reducing the measured saving by 19.9 percentage points. This is not a reproduction failure or a fak delta: the model differs because the exact snapshot was unavailable, provider tokenization/model behavior differ, and temperature zero is not guaranteed bit-deterministic across hosted inference. Upstream publishes per-prompt medians but no raw trial outputs or provider responses, so deeper trial-level attribution is impossible.

The semantic gate is deterministic and task-specific. It checks necessary concepts or code constructs for each prompt (for example seconds-vs-milliseconds and division by 1000 for JWT expiry, parameterized SQL plus error handling for the security review, and both React error-boundary lifecycle methods, retry, and logging). Every captured replacement response passes. This gate establishes basic answer correctness, not style superiority or a fak claim.
