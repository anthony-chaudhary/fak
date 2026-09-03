---
title: "Harness Gateway startup render witness — 2026-08-29"
description: "Before/after desktop and narrow captures for the typed /debug/vars.startup projection."
---

# Harness Gateway startup render witness — 2026-08-29

Issue: #8241.

## Verdict

The Harness Gateway surface now renders the structured `/debug/vars.startup` block as operator state rather than generic prose: readiness, time-to-ready and unaccounted time, named phases with provenance/stage, native model-load totals and load-path residency, and typed retained messages. A gateway that omits the structured block keeps an explicit legacy fallback.

| Viewport | Before (`c2f2b5ac`) | After |
|---|---|---|
| Desktop, 1440 px | [`gateway-startup-before-desktop-1440-2026-08-29.png`](gateway-startup-before-desktop-1440-2026-08-29.png) | [`gateway-startup-desktop-1440-2026-08-29.png`](gateway-startup-desktop-1440-2026-08-29.png) |
| Narrow, 390 px | [`gateway-startup-before-narrow-390-2026-08-29.png`](gateway-startup-before-narrow-390-2026-08-29.png) | [`gateway-startup-narrow-390-2026-08-29.png`](gateway-startup-narrow-390-2026-08-29.png) |

The before captures use the exact committed `internal/harnessweb` sources at `c2f2b5ac6153e409185fef76e812295b657c8e23`. The after captures use the changed sources. Both read a bounded loopback HTTP gateway fixture; only the after adapter projects its typed startup block. The captured `/api/status` readout is [`gateway-startup-status-2026-08-29.json`](gateway-startup-status-2026-08-29.json).

## Captured typed state

```text
status=ready time_to_ready=8.375s unaccounted=0.125s
phases=planner-init,weights-load,warm-cache
model_load=Qwen3.8 native artifact mode=native tensors=412 bottleneck=device-upload
load_path=Q4_K/resident resident_tensors=398 dequant_tensors=14
messages=model/weights-ready/info,cache/warmup/warn
```

The fixture contains no private model path or credential. Message and producer-controlled labels enter DOM nodes through `textContent`; they are not concatenated into HTML. The render test pins that boundary and the 600 px single-column rule.

## Reproduction gates

```text
go test ./internal/harnessweb -count=1
playwright screenshot --viewport-size="1440,1650" --full-page http://127.0.0.1:18782 gateway-startup-desktop-1440-2026-08-29.png
playwright screenshot --viewport-size="390,844" --full-page http://127.0.0.1:18782 gateway-startup-narrow-390-2026-08-29.png
```

The deterministic adapter test supplies the canonical typed JSON shape. `TestLiveAdapterPreservesOlderGatewayStartupFallback` supplies a legacy response with only `startup_report` and proves structured startup remains absent rather than reconstructed from prose.

## Boundary

This is a real browser capture of the live Harness handler reading a bounded loopback gateway contract fixture. It proves the web projection and responsive layout, not a hardware model load or live CUDA serve; those remain owned by the gateway startup producer's existing witnesses.
