# Calibrated vCache TTL-tier steering in a non-FAK repository — 2026-08-18

**Verdict:** fresh measured provider retention now changes the real Anthropic request bytes used for the 5m-versus-1h cache tier.

- The launch context is a temporary non-FAK Git repository, matching the arbitrary-repository default-on objective.
- `TestFreshMeasuredTTLSteersAnthropicTierDecision` sends the same captured Anthropic request through both decisions. With no trusted calibration, managed cache authors `cache_control.ttl="1h"`. With fresh measured retention of two hours, FAK leaves the provider-default 5m-tier request byte-identical instead of paying to extend it.
- `TestUntrustedTTLCalibrationPreservesStaticTierDecision` proves missing, unmeasured, invalid, and model-mismatched evidence cannot suppress the existing 1h behavior.
- The runtime loader already rejects stale and provider/model-mismatched ledger rows before they reach this request seam.

Structured artifact: [`vcache-calibrated-ttl-tier-non-fak-2026-08-18.json`](vcache-calibrated-ttl-tier-non-fak-2026-08-18.json).

This is request-byte steering, not a claim of a new live Anthropic measurement or provider-billing savings. Cache-write calibration, heartbeat execution, real-session prediction-error metrics, and provider-wide live probes remain open in #1497.
