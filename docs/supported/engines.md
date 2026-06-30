---
title: "Model serving engines fak supports"
description: "The token engines fak serve fronts over the OpenAI-compatible wire — Ollama, vLLM, SGLang, llm-d, llama.cpp (llama-server), and LM Studio — plus fak's own in-kernel reference engine. fak is the governance and gateway band in front of the engine, not the engine itself."
---

# Model serving engines fak supports

fak does not generate tokens for production. It fronts an engine that does. The gateway
speaks the OpenAI-compatible and Anthropic Messages wires, adjudicates every proposed tool
call, and proxies the request to whatever serves the model.

So "supported engine" has a precise meaning here. An engine is supported when

```bash
fak serve --provider openai --base-url <engine /v1>
```

puts the gate in front of it. The engine keeps serving tokens its own way; fak adds the
capability floor, the result-side quarantine, and the audit trail in front. This page lists
the local engines that wiring covers, then the one engine fak runs itself — the in-kernel
reference engine — and finally the catch-all for anything else that speaks the wire.

## 1. Local / self-hosted engines over the OpenAI-compatible wire

These run on your own box and expose an OpenAI-compatible `/v1` surface. You point
`fak serve --base-url` at the engine, then point your agent at fak. The base URLs below are
the engine defaults from the [compatibility matrix](../integrations/compatibility-matrix.md)
and the [Claude Code guide](../integrations/claude.md); swap host and port to match your
own deployment.

| Engine | Default base URL | fak wiring |
|---|---|---|
| [Ollama](https://docs.ollama.com/api/openai-compatibility) | `http://localhost:11434/v1` | `fak serve --provider openai --base-url http://<host>:11434/v1` (host/port via `OLLAMA_HOST`) |
| [vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) | `http://localhost:8000/v1` | `fak serve --provider openai --base-url http://<host>:8000/v1` (server launched with `vllm serve`, host/port via `--host`/`--port`) |
| [llm-d](../integrations/llm-d.md) | cluster Gateway API route, usually `http://<gateway-host>/v1` | `fak serve --provider openai --base-url http://<llm-d-gateway>/v1` for chat proxy mode. For kernel-dispatched calls and route manifests, use the registered engine id `llm-d` with `FAK_LLMD_BASE_URL`, `FAK_LLMD_MODEL`, and optional `FAK_LLMD_API_KEY` / `FAK_LLMD_METRICS_URL`. |
| [SGLang](https://docs.sglang.ai/backend/openai_api_completions.html) | `http://localhost:30000/v1` | `fak serve --provider openai --base-url http://<host>:30000/v1` (launched via `python3 -m sglang.launch_server`) |
| [llama.cpp (llama-server)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | `http://localhost:8080/v1` | `fak serve --provider openai --base-url http://<host>:8080/v1` (`llama-server -m model.gguf --host 0.0.0.0 --port 8080`) |
| [LM Studio](https://lmstudio.ai/docs/developer/openai-compat) | `http://localhost:1234/v1` | `fak serve --provider openai --base-url http://<host>:1234/v1` (start the server in the Developer tab, port configurable in the app) |
| A local transformers shim (Windows dogfood path) | set by the dogfood launcher | The committed [`dogfood-claude.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/scripts/dogfood-claude.ps1) launcher starts a transformers-backed `local_shim.py` (expected at `experiments/agent-live/local_shim.py`) instead of Ollama, defaulting to `SmolLM2-135M` for CPU-friendly serving. The launcher is committed; the shim helper itself is not. |

Once the engine answers, the wiring is the same for all of them. Verify the upstream with
`curl http://<host>:<port>/v1/models`, start `fak serve` against it, then check fak's own
health at `/healthz`. The [Claude Code guide](../integrations/claude.md) has the full
manual two-terminal walkthrough, including the engine launch commands and the Claude Code
environment variables; the `dogfood-claude.sh` / `dogfood-claude.ps1` launchers automate
the same stack with one command.

If the engine needs provider-specific request fields (for example vLLM, llm-d, or SGLang sampling
knobs), pass them through with `FAK_PROVIDER_EXTRA_BODY_JSON`. The
[serve config reference](../serve-config.md) covers that plus the auth, policy, and timeout
knobs you set for a network-facing deploy — a slow local CPU model in particular needs the
write and planner timeouts raised together.

### llm-d details

llm-d is a Kubernetes serving stack, not a single local worker. It fronts vLLM workers
through the Gateway API / Endpoint Picker Provider path and exposes an OpenAI-compatible
route. Put fak in front of that route when your agent speaks Chat Completions:

```bash
fak serve --addr 0.0.0.0:8080 \
  --provider openai \
  --base-url http://<llm-d-gateway>/v1 \
  --model <served-model> \
  --policy floor.json \
  --require-key-env FAK_GATEWAY_KEY
```

For routes that use fak's syscall dispatch or model-routing manifest, select the
first-class engine id:

```bash
export FAK_LLMD_BASE_URL="http://<llm-d-gateway>/v1"
export FAK_LLMD_MODEL="<served-model>"
fak serve --engine llm-d --model "<served-model>"
```

`FAK_LLM_D_*` aliases are accepted for the same variables. The adapter deliberately
uses llm-d's public OpenAI-compatible frontend and Prometheus/vLLM-style worker signals;
it does not import llm-d internals or claim exact remote KV-span eviction.

Run `fak llmd-smoke --base-url http://<llm-d-gateway>/v1 --model <served-model>` before
putting traffic behind the route. Add `--metrics-url` when the deployment exposes the
worker Prometheus endpoint and you want the smoke report to verify fak's `engine="llm-d"`
metrics normalization too.

The route-manifest preset at
[`examples/routing-presets/llm-d.json`](../../examples/routing-presets/llm-d.json)
uses `llm-d` as the default dispatch target while keeping common sensitivity labels on
`inkernel`. Validate it with `fak route --check examples/routing-presets/llm-d.json`
before using it with `--route-manifest`.

## 2. The in-kernel reference engine

fak also ships an engine of its own: a pure-Go model runner fused into the kernel. You
select it with `--engine inkernel`. Instead of proxying to an upstream, an allowed tool
call is completed by a real greedy decode over a kernel-owned KV cache
(`model.Session.Generate`), wired in as a `RegisterEngine` backend
(`internal/modelengine`). [SHIPPED] in the [claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).

What it is for:

- **A correctness reference.** The forward pass is proven token-for-token against a
  HuggingFace oracle — embedding exact, per-layer cosine 1.000000, final-logits max delta
  about 4.4e-5. The parallel and batched paths are each bit-identical to the serial
  reference. [SHIPPED]
- **A kernel-owned KV cache.** The cache is a Go structure the kernel owns, not an opaque
  arena inside a separate serving process. That ownership is what makes the
  addressable-eviction proofs possible: a quarantine verdict on poison bytes evicts that
  result's K/V span and leaves the cache bit-identical (max delta 0.0) to never having
  seen it. [SHIPPED]
- **A real dispatch path with or without a model export.** With no export it lazily builds
  a deterministic synthetic checkpoint, so the engine runs out of the box. Set
  `FAK_MODEL_DIR` to a real export to load it through the identical dispatch path. The GGUF
  / device-residency knobs (`--gguf`, `FAK_Q4K`, `FAK_GPU_BUDGET_MB`, and the rest) live in
  the [model/compute engine env reference](../model-engine-env.md).

The honest fence: the in-kernel engine is a correctness reference, not a
production-throughput server. The int8/Q8_0 SIMD lane is an in-flight increment, not yet a
`[SHIPPED]` row, and the watt source / token-per-watt telemetry is labelled SIMULATED.
When you need fast production serving, front one of the engines in section 1 instead. For
the architectures the in-kernel engine runs and which rungs are proven bit-exact, see the
[Models](models.md) page.

## 3. Any other OpenAI-compatible server

The list in section 1 is not a closed set. The OpenAI-compatible `/v1/chat/completions`
wire is the field's common denominator, and fak's engine client is base-URL-swappable
local-or-remote with bounded timeout and backoff. [SHIPPED] So any server that exposes that
surface is fronted the same way — point `fak serve --provider openai --base-url` at its
`/v1` and point your agent at fak.

Rather than list engines the repo cannot source, the honest claim is the rule itself: if
the server speaks the OpenAI-compatible wire, fak fronts it. The
[compatibility matrix](../integrations/compatibility-matrix.md) is the sourced reference for
that — its "Model backends & gateways" section carries each engine with its exact base URL
and a source link, and the [integration index](../integrations/README.md) has the universal
"repoint one base URL" recipe and a 60-second offline proof.

One thing to keep honest in the comparison: against a fast engine, fak's difference is
operational surface, not throughput. fak adds the capability floor, the result-side
quarantine, and the decision journal in front of the tokens; it does not make the engine
generate them faster.

## Related: the supported-things pages

- [What fak supports (hub)](README.md) — the index of every "supported" page
- [Models](models.md) — in-kernel architectures + any model you front
- [Features](features.md) — every capability with its shipped / simulated / stub status
- [Clouds & hosted providers](clouds.md) — Anthropic, OpenAI, Gemini, xAI, Bedrock, Vertex, Azure, OpenRouter, Together, Groq, Fireworks
- [APIs, wires & MCP](apis-and-protocols.md) — OpenAI Chat/Responses, Anthropic Messages, Gemini, xAI, MCP, fak-native endpoints
- [Agent harnesses & frameworks](agent-harnesses.md) — Claude Code, Cursor, Codex, Aider, Cline, Roo, LangChain, LlamaIndex, CrewAI, …

## Reference (the witnessed sources behind this page)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 47 sourced harnesses / frameworks / backends / protocols, each with the exact repoint key
- [llm-d upstream](https://github.com/llm-d/llm-d) and [architecture docs](https://llm-d.ai/docs/architecture) — Gateway API / EPP and OpenAI-compatible serving stack
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every capability with one machine-checked tag (shipped / simulated / stub)
- [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) · [CLI reference](../cli-reference.md) · [Hardware matrix](../HARDWARE-MATRIX.md) · [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt)
