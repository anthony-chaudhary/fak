---
title: "Calibrated vCache steering in a non-FAK repository — 2026-08-18"
description: "Verdict: fresh measured provider calibration now changes two runtime outcomes rather than only scoring them."
---
# Calibrated vCache steering in a non-FAK repository — 2026-08-18

**Verdict:** fresh measured provider calibration now changes two runtime outcomes rather than only scoring them.

- A temporary non-FAK Git repository was used for the launch-posture run.
- `fak vcache calibrate --samples ... --ledger ...` persisted measured OpenAI/model constants: TTL `30s`, minimum prefix `2048`, and cached-read multiplier `0.2`.
- `fak doctor launch-posture ... --provider openai` reported `vcache-calibration: active` and named the measured steering values.
- `TestFreshMinimumPrefixCalibrationSteersAnchorWireDecision` captures the request bytes: without calibration the default-on star anchor authors `cache_control`; with a fresh measured floor above the request size, the request stays byte-identical.
- `TestFreshReadMultiplierChangesRuntimePricingDecision` proves fresh measured read pricing changes the live cost calculation while an unmeasured value cannot.
- Stale, observation-only, and model-mismatched rows preserve defaults; focused tests cover each fallback.

Structured artifact: [`vcache-calibrated-steering-non-fak-2026-08-18.json`](../_witnesses/vcache-calibrated-steering-non-fak-2026-08-18.json).

This is a steering spine, not closure of #1497: calibrated TTL-tier choice, cache-write multipliers, heartbeat warming, and provider-wide live probes remain open.
