---
title: "Choose a fak model-execution backend"
description: "Operator route for choosing a remote model server, a local CPU path, or a GPU path, with current defaults and support boundaries."
---

# Choose a model-execution backend

**Audience:** operators deciding where model inference runs behind fak. fak remains the
policy, routing, cache, and audit layer in every choice; the backend supplies model
tokens.

**Current default:** front an existing OpenAI-compatible model server, local or remote.
That is the maintained operating path for production model serving and lets fak stay
independent of the server and its silicon. **Next action:** choose an engine from the
[serving-engine table](engines.md#1-local--self-hosted-engines-over-the-openai-compatible-wire)
and use that row's wiring route.

## Inspect this binary before loading a model

Run `fak runtime-capabilities` to get the stable `fak-runtime-capabilities/1` JSON report for the current executable and host. The default schema is unchanged. Add `--receipt-schema fak-execution-mode-receipt/1` when a health, audit, deployment, or benchmark consumer needs the uniform seven-mode execution record. The probe does not open or load a model payload. It reports:

- whether the binary and governed control plane are runnable;
- `goos`, `goarch`, build tags, host memory, and every backend that actually registered;
- the always-portable fak-native `cpu-ref` reference tier when present;
- whether an exact `--backend` request is available, unavailable, or unsupported;
- the explicit `supported_cpu_envelopes` catalog for policy-controlled local CPU degradation; and
- the selected engine/backend identity (`fak-native`) without silently substituting another backend.

```console
fak runtime-capabilities
fak runtime-capabilities --backend cuda
fak runtime-capabilities --prefer-backend vulkan --fallback-policy local_cpu_degraded --cpu-envelope qwen25-1p5b-q8-windows-amd64
```

A default build normally reports only `cpu-ref`. CUDA and Vulkan require their build tags and vendor/runtime prerequisites; Metal requires a cgo-enabled Darwin/arm64 build and a reachable Metal device. A missing exact backend request is a diagnostic result, not permission to switch engines. The only built-in degraded path is the explicit `--prefer-backend ... --fallback-policy local_cpu_degraded --cpu-envelope ...` flow: it keeps the requested accelerator reason, selects `cpu-ref` only for a supported envelope, and refuses unsupported or over-budget envelopes before payload load. The command never substitutes `llama.cpp` or another engine.

Remote placement is a separate, pre-payload policy decision. `--placement local_only` (the default) and `prefer_local` never cross a network boundary. `remote_allowed` is considered only after a `--prefer-backend` local failure; an exact `--backend` pin always fails closed. Admission requires an identical `--remote-target` and `--authorize-remote-target`, named provider/engine/model, explicit allowed egress, credential reference name plus presence (never a secret value), verified TLS and proxy state, independently declared reachability, positive timeout and budget, a non-negative retry ceiling, and declared state classes crossing the boundary. The receipt retains local `fak` control-plane ownership while naming remote execution and every gate. The probe opens no network connection, and no provider—including a future fak cloud—has a privileged or automatic path.
### Uniform execution-mode receipt

`fak-execution-mode-receipt/1` is an additive projection over the capability report. It uses exactly seven modes: `local_accelerator`, `local_cpu_degraded`, `remote_backed`, `offline_control_mock`, `offline_model_backed`, `control_only`, and `refused`. The receipt carries binary identity, one health value, status/audit projections with explicit evidence states, local `fak` control-plane ownership, execution identity, fallback reason, offline/egress state, operating envelope, prerequisite witness, and closed transition rules. When both views are independently `observed`, a mode or health mismatch invalidates the receipt. Pre-payload projections and fixtures mark those views `unwitnessed`. Missing evidence is the literal `unknown` or `unwitnessed`; omission never implies readiness.

Every model-backed mode must name the actual engine and model plus a local backend or remote provider. Native/performance receipts additionally require `engine: "fak-native"`; a substituted engine makes the record invalid and refused. `remote_backed` preserves `control_plane_owner: "fak-local"` and the existing explicit remote boundary gates. The projection is pre-payload observation rather than proof that a turn ran.

The CLI's `--execution-mode-fixture MODE` covers all seven schema states deterministically, but each fixture says `certification: "unwitnessed"`. Use real run receipts for deployment certification; unavailable hardware or provider states remain unwitnessed instead of being simulated as successful runs.


## Choose by operating envelope

| Choose | When it fits | Current support boundary | Continue with |
|---|---|---|---|
| **Remote or separately managed server** | You already use a hosted endpoint, vLLM, SGLang, Ollama, llm-d, llama.cpp, LM Studio, or another OpenAI-compatible server. This is the default operator mode. | The provider or engine owns inference and accelerator support. fak fronts its public API; support is scoped to the documented wire and configuration, not the server's internals. | [Serving engines](engines.md), or [clouds and hosted providers](clouds.md) for a provider-native wire. |
| **Local CPU reference** | You need deterministic correctness work, offline development, a portable baseline, or a small-model proof without accelerator requirements. | `cpu-ref` is the shipped correctness floor for fak's in-kernel engine. It is a reference path, not the production-throughput default. | [In-kernel reference engine](engines.md#2-the-in-kernel-reference-engine). |
| **In-kernel GPU backend** | You are evaluating fak's forward pass directly on supported CUDA or Vulkan hardware and can build and run on the matching device. | CUDA and Vulkan are build-tagged `Approx` backends with hardware-specific witnesses. A device-family result does not establish support for every model, shape, driver, or host. | [Hardware portability and build tags](../explainers/hardware-portability.md), then require a non-reference run in the model benchmark. |
| **External GPU serving engine** | You want production GPU throughput while keeping inference in a dedicated serving stack. | GPU lifecycle, kernels, capacity, and model coverage belong to the selected engine; fak's support claim remains the integration wire. | Select vLLM, SGLang, llm-d, or another compatible server in [serving engines](engines.md). |

“Remote” describes where the model server runs; “CPU” and “GPU” describe where its
inference executes. They are not mutually exclusive: a remote server can use either CPU
or GPU. When an operator only needs to choose a fak integration, choose the server/wire
first. Choose an in-kernel silicon backend only when the in-kernel engine itself is the
subject of the run.

## Mode and generation

This page describes the current (`gen/now`) backend-selection route. It covers two
execution modes:

1. **Gateway mode:** fak fronts a local or remote server over a documented provider wire.
   This is the operator default.
2. **In-kernel backend-evaluation mode:** the model benchmark invokes the forward pass
   through `internal/compute.Backend`; `cpu-ref` is the reference implementation, while
   CUDA and Vulkan are opt-in, build-tagged registrations. The linked hardware-portability
   page gives the build and `-require-non-reference` commands. A generic in-kernel run does
   not select one of those accelerator registrations: the optimized legacy prefill/batch
   implementation remains that runtime's default.

Use gateway mode for normal operation. Choose in-kernel backend-evaluation mode only when
the HAL path and its parity witness are the subject of the run.

## Support and lifecycle labels

Backend support is evidence-scoped rather than inferred from a device or vendor name:

- **Shipped wire** means fak currently implements the named API integration.
- **Reference** means the in-kernel backend is the exact correctness floor.
- **Approx** means the backend passed the hardware page's named non-bit-exact parity
  checks (for example, argmax agreement and logit cosine); it does not establish exact
  logits or coverage beyond the witnessed model, shape, build, and device.
- **Partial** means the linked support row lists working and missing methods, shapes, or
  capabilities. **not-yet** means no current witness supports the path.

The maintained support row and its linked witness are the status authority. Add a backend
to this route only after its wire or compute registration and matching witness ship.
Relabel it `Partial` or `not-yet` when that witness no longer covers current code; remove
it from this current route when the maintained support row names a replacement generation
and links the superseding route. For the detailed accelerator fields and conformance vocabulary, see
[supported silicon backends](silicon-backends.md). For machine-specific measurements, use
the [hardware matrix](../HARDWARE-MATRIX.md); those measurements are evidence for the
listed machine and build, not a blanket support promise.

## Decision recap

- For normal operation, front a maintained local or remote model server.
- For portable correctness and offline proofs, use `cpu-ref`.
- For direct in-kernel accelerator evaluation, build the matching CUDA or Vulkan backend
  and require a non-reference witness.
- For production GPU serving, choose a dedicated GPU engine and connect fak over its
  supported wire.

### Captured CPU-degraded governed turn

The issue #9842 witness set includes a real CPU-only in-kernel turn at
[`docs/_witnesses/issue-9842/runtime-capabilities-governed-turn.json`](../_witnesses/issue-9842/runtime-capabilities-governed-turn.json).
The receipt joins the pre-load `local_cpu_degraded` admission to the loaded
Qwen2.5-1.5B-Instruct Q8 artifact and the gateway's `backend=cpu-ref
forward_path=cpu/reference` turn log. The request completed with HTTP 200 and
an `OK` response. It names `engine: fak-native`; no llama.cpp runtime or engine
substitution participated.
