# Harness-independent server builder research decision — 2026-08-19

Issue: [#8133](https://github.com/anthony-chaudhary/fak/issues/8133) (child of [#6777](https://github.com/anthony-chaudhary/fak/issues/6777))  
Decision date: 2026-08-19  
Decision: build a separately owned **server product** whose stable boundary is a non-secret, versioned `ServerReceipt`; do not make a generated harness own an inference process, model download, or deployment substrate.

## Executive decision

FAK should add one narrow local-first path:

1. a declarative server specification selects a model artifact, runtime adapter, transport, and local lifecycle policy;
2. `fak server init` materializes that specification without starting inference;
3. `fak server up` resolves the artifact, launches one supported local adapter, waits for protocol-level readiness, and atomically writes a stable receipt;
4. a separately created harness imports only the receipt, verifies compatibility, and calls the endpoint; and
5. `fak server down` tears down only the process owned by that server instance.

The receipt is the seam. It carries identity and compatibility, not secrets or mutable process internals. The minimum implementation should prove one `llama-server`-style local process and one OpenAI-compatible chat probe. vLLM and LocalAI inform the contract and remain follow-on adapters rather than prerequisites for the spine.

## Research method and dated primary sources

Sources were read on 2026-08-19. Repository links are pinned to the observed commit so later upstream edits cannot silently change this decision's evidence.

- vLLM commit [`db92053e`](https://github.com/vllm-project/vllm/tree/db92053e97b5630a6a36118693b1dffe9b03be36), especially its [OpenAI-compatible server](https://github.com/vllm-project/vllm/blob/db92053e97b5630a6a36118693b1dffe9b03be36/docs/serving/online_serving/openai_compatible_server.md) and [Docker deployment](https://github.com/vllm-project/vllm/blob/db92053e97b5630a6a36118693b1dffe9b03be36/docs/deployment/docker.md) documentation.
- llama.cpp commit [`2e92ecd0`](https://github.com/ggml-org/llama.cpp/tree/2e92ecd0247d25f09797f8fdb044a166522fc05d), especially the pinned [`llama-server` README](https://github.com/ggml-org/llama.cpp/blob/2e92ecd0247d25f09797f8fdb044a166522fc05d/tools/server/README.md).
- LocalAI commit [`34e986de`](https://github.com/mudler/LocalAI/tree/34e986de0aa3f4174fc14b67455a4834e9a5cc2b), especially [model installation/configuration](https://github.com/mudler/LocalAI/blob/34e986de0aa3f4174fc14b67455a4834e9a5cc2b/docs/content/getting-started/models.md), [container startup](https://github.com/mudler/LocalAI/blob/34e986de0aa3f4174fc14b67455a4834e9a5cc2b/docs/content/getting-started/containers.md), and the [health route implementation](https://github.com/mudler/LocalAI/blob/34e986de0aa3f4174fc14b67455a4834e9a5cc2b/core/http/routes/health.go).
- Docker Compose [features and uses](https://docs.docker.com/compose/intro/features-uses/) and OCI Image Specification [runtime configuration](https://specs.opencontainers.org/image-spec/config/) read 2026-08-19, used only as comparison points for lifecycle/config portability.

## Observed facts

These are direct observations from the pinned sources, not claims about what FAK should build.

### vLLM

- `vllm serve` exposes an OpenAI-compatible HTTP server and accepts model/tokenizer configuration plus server-specific CLI arguments.
- The server supports `/v1/models` and OpenAI-style chat/completions surfaces, while some parameters depend on the model repository's generation configuration.
- Its Docker path names image, model cache mount, API surface, IPC/shmem concerns, and optional GPU/runtime arguments separately. The container does not erase model/runtime compatibility work.

### llama.cpp

- `llama-server` is a standalone executable with an OpenAI-compatible HTTP API and a documented `/health` readiness endpoint.
- It can expose model listing, metrics, slots, LoRA adapters, embeddings, reranking, and chat/completions, but those capabilities are individually configurable and must not be inferred merely from “OpenAI-compatible.”
- A local file path plus CLI arguments is enough to launch a minimal server, making it the smallest practical spine adapter.

### LocalAI

- LocalAI presents an OpenAI-compatible API while separating model installation from per-model YAML configuration and backend selection.
- Models may come from a gallery, URI, local file, or OCI source; configuration can name aliases and backend/runtime parameters.
- LocalAI has a health route and container-oriented startup, but its broad backend/gallery surface is substantially larger than the one-adapter spine needed here.

### Deployment comparators

- Compose groups services, configuration, dependencies, health checks, networks, and volumes into a lifecycle unit.
- OCI image config standardizes process defaults such as entrypoint, command, environment, working directory, ports, and volumes; it does not standardize inference capabilities or model compatibility.

## Inferences and decisions

These statements interpret the observations for FAK.

- **Inference:** “OpenAI-compatible” is a transport family, not a sufficient compatibility contract. **Decision:** the receipt must enumerate probed capabilities and protocol revision rather than advertise a boolean.
- **Inference:** artifact acquisition, process construction, and process readiness fail independently. **Decision:** represent them as separate receipt phases/errors and never collapse “downloaded” into “ready.”
- **Inference:** the smallest cross-platform proof is a directly launched executable; containers add a second runtime and mount/accelerator policy. **Decision:** direct local process first, container rendering later.
- **Inference:** harness and server lifecycle coupling would recreate today’s hidden setup. **Decision:** a harness imports a receipt by path and does not start, stop, upgrade, or repair the server.
- **Inference:** a receipt containing bearer tokens would turn discovery into credential distribution. **Decision:** include a credential reference name or auth mode, never secret material.
- **Inference:** a PID alone is forgeable and reusable. **Decision:** ownership state must bind instance ID, spec digest, process start identity, endpoint, and atomic receipt generation; teardown verifies that binding before signaling.

## Current FAK path map (observed 2026-08-19)

| Concern | Current path | What exists | Gap for this decision |
|---|---|---|---|
| Runtime serving | `cmd/fak/serve.go`, `cmd/fak/serve_config.go`, `cmd/fak/serve_stages.go`, `cmd/fak/serve_backend_preflight.go` | A large in-process `fak serve` gateway/runtime with staged startup and backend checks. | No separately initialized server-product manifest or external process ownership receipt. |
| Engine/model seams | `internal/engine/`, `internal/modelengine/`, `internal/modelreg/`, `internal/modelsrc/`, `internal/newmodel/` | Engine selection, model records, source/materialization concepts. | No immutable server-spec resolution that binds one artifact digest to one adapter invocation. |
| Harness creation | `cmd/fak/harness_init.go`, `internal/harnessinit/`, `internal/harnesscreationreceipt/` | Harness initialization and creation receipts. | Creation receipts describe the harness product, not an independently owned inference endpoint. |
| Harness composition | `cmd/fak/harness_compose.go`, `internal/harnesscompose/`, `internal/harnessartifact/` | Typed harness artifact composition and validation. | No import/verify path for a server receipt with protocol/capability constraints. |
| Harness resolution | `cmd/fak/harness_resolve.go`, `internal/harnessresolve/`, `internal/harnessmix/`, `internal/harnessprofile/` | Stack/profile resolution and mix-and-match checks. | Server identity, readiness generation, and capability negotiation are absent from the resolved stack. |
| Gallery/discovery | `cmd/fak/harness_gallery.go`, `internal/harnessgallery/`, `cmd/fak/harness_discover.go`, `internal/harnessdiscover/` | Harness candidates and discovery. | Must not become server lifecycle ownership; at most it may surface receipt locations later. |
| Provider/protocol | `cmd/fak/harness_protocol.go`, `internal/harnessprotocol/`, `internal/modelroute/` | Protocol/provider declarations and routing concepts. | No probe-backed endpoint contract connecting an external server instance to a harness. |
| Operational durability | `internal/runmanifest/`, `internal/provenance/`, `internal/atomicfile/` | Existing patterns for manifests, provenance, and atomic writes. | No server-instance state directory with start/readiness/stop witnesses. |
| Existing deployment examples | `examples/`, `docs/integrations/`, `deploy/` | Demos and deployment guidance for existing FAK surfaces. | No clean-directory independent-server scenario consumed by a separately generated harness. |

This map is intentionally additive: the new `server` product should not rename or absorb `fak serve`. The latter is an existing gateway/runtime; the former is a builder/lifecycle envelope around a separately owned inference server.

## Proposed typed contract

A versioned `ServerSpec` should declare only intent:

- stable server name and instance directory;
- model artifact reference plus required digest;
- runtime adapter and adapter version/constraint;
- protocol family and required capabilities;
- bind policy (`loopback` in the spine), requested port, resource envelope, and credential-reference name; and
- lifecycle mode (`local-process` in the spine).

A generated `ServerReceipt` should report observed state:

- schema version, server/instance IDs, spec digest, generation, and creation time;
- artifact identity/digest and adapter executable identity/version;
- canonical base URL and model alias;
- auth mode/credential reference without secret values;
- capabilities actually probed, protocol revision, readiness timestamp, and probe digest;
- ownership reference sufficient for safe teardown; and
- provenance for authored configuration versus observed runtime facts.

Receipts are written atomically only after readiness succeeds. A failed launch emits a separate diagnostic/result record, not a ready receipt.

## Minimal working spine

**For:** a harness builder with a local GGUF model and a separately installed `llama-server`.  
**Problem:** the harness cannot reproducibly discover whether that endpoint is the right model/runtime or merely an HTTP listener.  
**Today:** the operator starts a process manually, copies a URL into harness configuration, and loses the build/readiness evidence.  
**Better because:** one typed server lifecycle produces a stable, secret-free receipt that a separately initialized harness can validate.  
**Witness:** a clean-directory scenario launches the fixture adapter, observes readiness, imports the receipt into a different harness directory, completes one chat probe, and tears down only the owned process.

Spine sequence:

```text
fak server init --name local-code --adapter llama-server --model <path> --sha256 <digest>
fak server up local-code --receipt <server-dir>/server-receipt.json
fak harness init <harness-dir> --server-receipt <server-dir>/server-receipt.json
fak harness verify-run <harness-dir> --selfcheck
fak server down local-code
```

The implementation may refine flags, but the observable ownership split and receipt handoff are acceptance constraints.

## Boundaries

### In the first cohort

- One direct local-process adapter and loopback HTTP.
- One file artifact with mandatory digest verification.
- OpenAI-compatible model listing, health/readiness, and one chat probe.
- Atomic, non-secret receipt plus safe owned-process teardown.
- Explicit harness import and compatibility failure messages.
- Captured clean-directory end-to-end witness.

### Explicitly out

- Kubernetes, schedulers, multi-node topology, autoscaling, service mesh, or a universal deployment DSL.
- Building or downloading arbitrary server binaries.
- A topology GUI or model marketplace.
- Automatic port exposure beyond loopback, TLS termination, or secret distribution.
- Multi-model hot swap, LoRA management, speculative decoding, batching optimization, or benchmark tuning.
- Harness-owned server start/stop, implicit server discovery, or mutation of a receipt by its consumer.
- Claiming full OpenAI compatibility from endpoint shape alone.

## Issue decomposition

The implementation cohort has exactly six leaves, ordered as a working spine then operating-envelope hardening. Their titles and primary file trees are disjoint from the sibling research tracks under #6777.

1. **Server contract:** add `ServerSpec`/`ServerReceipt` schemas, validation, and atomic encoding under a new `internal/serverproduct/` leaf.
2. **Artifact resolver:** bind one local model file and digest under `internal/serverartifact/`; no network gallery.
3. **Adapter invocation:** render and inspect one `llama-server` process contract under `internal/serveradapter/`; no lifecycle ownership.
4. **Lifecycle CLI:** add `fak server init|up|status|down` and owned-instance state under `cmd/fak/server*.go` plus `internal/serverlifecycle/`.
5. **Harness import:** consume an immutable receipt and fail closed on protocol/capability mismatch under `internal/harnessserver/` and the narrow harness CLI seam.
6. **Independent scenario:** capture clean-directory server→receipt→separate-harness→chat→teardown evidence under `examples/independent-server/` and its focused docs/tests.

Dependencies are deliberate: 1 precedes 2/3; 2 and 3 precede 4; 1 and 4 precede 5; all five precede 6. The cohort should not parallelize leaves that share the contract until the contract lands.

## Alternatives rejected

- **Teach `fak harness init` to launch a server.** Rejected because lifecycle ownership and failure recovery would again be hidden inside the harness product.
- **Make Docker Compose the contract.** Rejected because Compose describes deployment mechanics, not model/protocol compatibility, and excludes a zero-container local spine.
- **Start with vLLM.** Rejected for the first spine because accelerator/container/package prerequisites enlarge the witness; retain it as a later adapter against the same receipt.
- **Adopt LocalAI configuration wholesale.** Rejected because its gallery and backend breadth exceed the problem; borrow separation of model config/backend and explicit health instead.
- **Reuse a bare provider base URL.** Rejected because it carries neither artifact/runtime identity nor witnessed capabilities nor safe lifecycle ownership.

## Completion and independent witness

This research decision is complete only when the note is visible on `origin/main`, exactly six live child issues are visible under #8133, and `fak-dev issue contract` plus `fak-dev issue cohort` accept the captured issue set. Implementation completion belongs to those children, not this research issue.
