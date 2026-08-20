---
title: "Microagents through one real fak kernel: captured witness"
description: "Captured proof that two in-process microagents traverse one fak serve kernel with shared gateway, session state, and cooperative scheduling."
---

# Microagent → real fak kernel witness — 2026-08-12

## Scope

This witness proves that two in-process goroutine microagents traverse one running `fak serve` kernel, one shared `microagent.SessionGateway`, one shared `session.Table`, and one cooperative model-call scheduler. The kernel used its deterministic Mock planner; this is **not** evidence of model quality or paid-provider cost.

## Command

```powershell
fak serve --addr 127.0.0.1:18081
fak micro --engine gateway --gateway 127.0.0.1:18081 --model kernel-model `
  --agents 2 --workers 2 --seats 1 --turns 1 --trace-out micro-real-kernel.jsonl --json
```

## Microagent result

```json
{
  "mode": "run",
  "engine": "gateway",
  "slots": 1,
  "agents": 2,
  "done": 2,
  "failed": 0,
  "results": [
    {"id": "micro-000", "steps": 1, "done": true},
    {"id": "micro-001", "steps": 1, "done": true}
  ]
}
```

## Independent kernel read-back

The `fak serve` process emitted two distinct gateway turns and HTTP requests:

```text
{"event":"gateway_inference_turn","model":"mock","prompt_tokens":5,"completion_tokens":24,"total_tokens":29,"trace_id":"gw-1","wire":"openai_chat_completions"}
{"event":"gateway_http_request","method":"POST","path":"/v1/chat/completions","status":200,"trace_id":"gw-1"}
{"event":"gateway_inference_turn","model":"mock","prompt_tokens":5,"completion_tokens":24,"total_tokens":29,"trace_id":"gw-2","wire":"openai_chat_completions"}
{"event":"gateway_http_request","method":"POST","path":"/v1/chat/completions","status":200,"trace_id":"gw-2"}
```

The per-agent trace artifact independently records 29 provider-reported tokens for each agent:

```json
{"trace_id":"micro-000","spans":[{"kind":"seat","label":"acquire","seat":"slot-pool/1"},{"kind":"step","label":"turn 1","tokens":29},{"kind":"verdict","label":"mock-planner","verdict":"ALLOW"}]}
{"trace_id":"micro-001","spans":[{"kind":"seat","label":"acquire","seat":"slot-pool/1"},{"kind":"step","label":"turn 1","tokens":29},{"kind":"verdict","label":"mock-planner","verdict":"ALLOW"}]}
```

## Verdict

**PASS for the kernel-integration claim:** two microagents made two successful `/v1/chat/completions` calls through a real fak gateway process and retained separate usage traces. **NOT YET for quality/cost parity:** #2028/#6520 remain the required paired real-provider benchmark.
