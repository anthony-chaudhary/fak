---
title: "gateway-engine names"
description: "This map positions the current gateway-engine coverage backlog. Each entry names the exact repository symbol;"
---
# gateway-engine names

This map positions the current `gateway-engine` coverage backlog. Each entry names the exact repository symbol; the family label remains the broader domain and is not a substitute for the symbol.

- **`engineresult`** — the exact `gateway-engine` symbol `engineresult`; use this spelling for that operation rather than the undifferentiated family name.
- **`enginespec`** — the exact `gateway-engine` symbol `enginespec`; use this spelling for that operation rather than the undifferentiated family name.
- **`enginevllm`** — the exact `gateway-engine` symbol `enginevllm`; use this spelling for that operation rather than the undifferentiated family name.
- **`externalengineevents`** — the exact `gateway-engine` symbol `externalengineevents`; use this spelling for that operation rather than the undifferentiated family name.
- **`inkernelturntax`** — the exact `gateway-engine` symbol `inkernelturntax`; use this spelling for that operation rather than the undifferentiated family name.
- **`kernelfloor`** — the exact `gateway-engine` symbol `kernelfloor`; use this spelling for that operation rather than the undifferentiated family name.
- **`kernelprincipal`** — the exact `gateway-engine` symbol `kernelprincipal`; use this spelling for that operation rather than the undifferentiated family name.
- **`newsessiongateway`** — the exact `gateway-engine` symbol `newsessiongateway`; use this spelling for that operation rather than the undifferentiated family name.
- **`pagedkernel`** — the exact `gateway-engine` symbol `pagedkernel`; use this spelling for that operation rather than the undifferentiated family name.
- **`qwen35gdnkernelerror`** — the exact `gateway-engine` symbol `qwen35gdnkernelerror`; use this spelling for that operation rather than the undifferentiated family name.
- **`resolvetoolengine`** — the exact `gateway-engine` symbol `resolvetoolengine`; use this spelling for that operation rather than the undifferentiated family name.
- **`sessiongateway`** — the exact `gateway-engine` symbol `sessiongateway`; use this spelling for that operation rather than the undifferentiated family name.
- **`startgatewayusagesnapshotloop`** — the exact `gateway-engine` symbol `startgatewayusagesnapshotloop`; use this spelling for that operation rather than the undifferentiated family name.


### fak_gateway_inference_output_tokens_per_second (blended generation throughput)

fak_gateway_inference_output_tokens_per_second is the gateway Prometheus gauge for cumulative completion tokens divided by full inference wall-clock across served turns, including prefill time.

**Distinct from:** It is blended model generation throughput over full inference time, not the tool vDSO hit path and not decode-only throughput, which subtracts prefill before dividing.
