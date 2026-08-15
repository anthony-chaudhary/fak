# Live native UI / agent seam probe — 2026-08-15

Issues: #6910, #1380, parent #6790.

## Sanctioned compute

Node: GCP `fak-realmodel`, zone `us-central1-a`, machine `g2-standard-8`, NVIDIA L4 present on PCI. Running container: `fak-gpu`, model `qwen2.5-0.5b-gpu`, in-kernel CUDA serve on port 8082.

## Live requests

A basic native turn succeeded through the public Anthropic-compatible seam:

```text
POST http://127.0.0.1:8082/v1/messages
{"model":"qwen2.5-0.5b-gpu","max_tokens":32,"messages":[{"role":"user","content":"Reply with the word READY."}]}
```

Observed response: HTTP 200, `READY`, input tokens 14, output tokens 1, gateway trace `gw-965`.

A coding-surface probe then asked the native runtime to inspect its workspace. Observed response: HTTP 200, trace `gw-966`, but the model said it had no tool available and emitted no tool/native-arm witness. The node and model were real; the missing tool-capable work surface, not local hardware, is the blocker.

## Decision

`cmd/harnesswebdemo` can target `/v1/messages` for a real live model turn, but claiming a fak-native coding-harness replacement requires the tool-capable native DoD in #1380 and durable session semantics in #6910. The UI must not fabricate tool events around a gateway response that did not execute a tool.
