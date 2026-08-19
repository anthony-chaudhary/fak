# Harness-bundled model serving research decision — 2026-08-19

Parent: [#8131](https://github.com/anthony-chaudhary/fak/issues/8131)  
Decision date: 2026-08-19  
Scope: serving-side ownership for one generated, local GGUF harness; research and issue decomposition only.

## Decision

FAK should make **model identity, acquisition verification, local-runtime admission, server lifecycle, and lifecycle receipts one typed harness-owned contract**, while keeping the model bytes and third-party runtime out of the generated source tree. The first working spine is deliberately narrow: one pinned GGUF acquired into a content-addressed cache, one explicitly selected local runtime, loopback-only startup, readiness proven by both a health endpoint and a one-token OpenAI-compatible probe, graceful stop, and one receipt consumed by `harness init`, `build`, `selfcheck`, diagnostics, upgrade, and cleanup.

“Bundled” therefore means **the harness bundles the declarative ownership and lifecycle**, not that FAK silently embeds multi-gigabyte weights or promises a GPU. Acquisition is explicit and receipt-bearing. Offline reuse is allowed only after digest verification. Hardware admission is measured before launch and never inferred from a model name.

## Feynman value frame

- **For:** harness builders and operators who want a generated product to run without separately inventing a model-server installation procedure.
- **Problem:** model bytes, runtime compatibility, startup, readiness, shutdown, and updates currently live outside one generated harness contract.
- **Today:** FAK can generate and inspect harness configuration and can serve configured backends, but the handoff between a declared harness model and a live, owned local server is manual.
- **Better because:** the same typed declaration drives acquisition, admission, launch, proof, diagnostics, upgrade, and cleanup, so failures are reproducible and no hidden download or phantom accelerator claim is needed.
- **Witness:** a clean-directory CPU scenario acquires a pinned tiny GGUF with visible consent, verifies its digest, launches loopback-only, passes health plus a one-token request, emits a lifecycle receipt, stops cleanly, then re-runs offline from the verified cache.

## Primary-source observations

All web sources below were read on **2026-08-19**. “Observed” means the behavior or interface is directly documented in the linked primary source. “Inference” is the design conclusion for FAK and is not attributed to the source.

### 1. llama.cpp server

Primary source: [`tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/2e92ecd0247d25f09797f8fdb044a166522fc05d/tools/server/README.md), repository revision [`2e92ecd`](https://github.com/ggml-org/llama.cpp/commit/2e92ecd0247d25f09797f8fdb044a166522fc05d), observed 2026-08-19.

**Observed facts**

- `llama-server` exposes an OpenAI-compatible HTTP server and accepts an explicit local model path with `-m/--model`.
- It also accepts Hugging Face repository/file selectors (`--hf-repo`, `--hf-file`) and a model URL, so acquisition may occur inside the runtime when those modes are chosen.
- Bind address, port, device selection, context size, memory mapping, and related resource controls are explicit CLI inputs.
- `/health` distinguishes loading from ready state; success means the server is ready to accept requests.
- The documented server surface separates process launch from HTTP readiness.

**FAK inference**

- The spine should pass a verified local path, not a remote selector, so download timing, digest, license acknowledgement, and offline behavior remain visible to the harness.
- Process creation is not readiness. FAK needs a bounded health wait and an actual protocol probe before claiming the harness is usable.
- Runtime arguments belong in a typed launch plan and receipt, not an opaque shell string.

### 2. Ollama

Primary sources: [Ollama API documentation](https://github.com/ollama/ollama/blob/0bb09259203ff8f6d361faae1d40c4f83d2a99f7/docs/api.md) and repository revision [`0bb0925`](https://github.com/ollama/ollama/commit/0bb09259203ff8f6d361faae1d40c4f83d2a99f7), observed 2026-08-19.

**Observed facts**

- The API has distinct model operations for pull, create, show, list, copy, and delete rather than treating model availability as server startup.
- Pull progress reports content digests and staged transfer statuses; blob endpoints address content by `sha256:<digest>`.
- Generate/chat requests accept `keep_alive`; an empty request with `keep_alive: 0` unloads a model, making loaded-state lifecycle explicit.
- `show` exposes model details and metadata, while list/process endpoints expose installed and running state.
- Streaming is the default for several operations and can be disabled for a single JSON response.

**FAK inference**

- Acquisition, installed availability, server readiness, and loaded-model readiness must be separate typed states.
- A content digest should be a required model identity input and independently recomputed by FAK, not trusted merely because a server reports one.
- Cleanup policy must distinguish stopping a process, unloading a model, and deleting cached bytes.

### 3. LM Studio / `lms`

Primary sources: [`lms get`](https://lmstudio.ai/docs/cli/local-models/get), [headless mode](https://lmstudio.ai/docs/developer/core/headless), and open-source CLI revision [`lmstudio-ai/lms@07b7252`](https://github.com/lmstudio-ai/lms/commit/07b7252d6de26a3a58c1bb80ed7e75a2b17f5eb6), observed 2026-08-19.

**Observed facts**

- `lms get` makes model discovery/download an explicit command and provides selection and machine-readable modes rather than requiring a GUI-only path.
- The headless flow separates installing/bootstrapping the service from starting the local server, and documents status/control through CLI commands.
- The CLI presents local model and server lifecycle as operator-visible operations suitable for automation.
- Runtime management and model loading are products of the local service, not properties encoded into a generated application source tree.

**FAK inference**

- FAK should expose explicit plan/apply phases and machine-readable receipts instead of downloading as a side effect of `selfcheck` or first inference.
- Runtime ownership must be declared: managed external runtime, operator-supplied runtime, or remote endpoint. The first spine supports one managed local adapter without pretending all runtimes share installation semantics.
- A generated harness should remain portable by storing declarations and receipts, never absolute cache paths or runtime binaries in source control.

## Current FAK path map

The map reflects committed `main` at `90409614e8` plus read-only inspection on 2026-08-19. Peer-dirty files were not treated as shipped evidence.

| Concern | Current path | Present seam | Gap for this decision |
|---|---|---|---|
| Generated harness entry point | `cmd/fak/harness_init.go`, `cmd/fak/harness_init_test.go` | Resolves a harness spec, writes generated product files, and emits init metadata. | No typed model-acquisition/server-lifecycle declaration or clean-directory local-model scenario. |
| Generated Go support | `pkg/harnesskit/kit.go`, `pkg/harnesskit/kit_test.go` | Generated harnesses call stable kit helpers for config/control behavior. | No stable model lifecycle plan/receipt API. |
| Generated resource contract | `cmd/fak/harness_resources.go`, `cmd/fak/harness_resources_guardstops_test.go` | Resource/guard-stop information is already a harness concern. | No local-runtime CPU/RAM/VRAM/disk admission contract tied to a model artifact. |
| Artifact identity/receipts | `internal/harnessartifact/artifact.go`, `internal/harnessartifact/artifact_test.go` | Content-addressed artifact references, digests, provenance, and receipt structures exist. | Not yet specialized into explicit model license/size/cache/acquisition states consumed by harness lifecycle. |
| Model registry | `internal/modelreg/modelreg.go`, `internal/modelreg/modelreg_test.go`; CLI in `cmd/fak/model.go` | Durable model/provider registration and lookup seam. | A registry entry is not yet a verified local GGUF availability or runtime-compatibility receipt. |
| Serve command | `cmd/fak/serve.go`, `cmd/fak/serve_config.go`, `cmd/fak/serve_stages.go`, signal files | Long-lived gateway configuration, staged startup, and shutdown handling exist. | No harness-owned child-runtime supervisor with bounded readiness and ownership-safe stop. |
| Backend preflight | `cmd/fak/serve_backend_preflight.go`, `cmd/fak/serve_backend_preflight_test.go` | Backend checks precede serving. | Does not admit a declared local model against measured disk/RAM/VRAM and runtime capability before acquisition/launch. |
| Gateway/runtime protocol | `internal/gateway/`, `internal/engine/`, `internal/model/` | OpenAI-compatible request path and model/backend abstractions exist. | Readiness is not one receipt that proves health plus an inference probe for a harness-owned process. |
| Harness inspect/verify/release | `cmd/fak/harness_inspect.go`, `cmd/fak/harness_verify_run.go`, `cmd/fak/harness_release.go` | Inspection, run verification, and release surfaces are present. | They do not read one model-serving lifecycle receipt or explain drift, upgrade, and cleanup. |
| Docs/operator contract | `docs/cli-reference.md`, `docs/harness-sdk.md`, generated harness docs | User-facing command and SDK documentation exists. | No no-hidden-download/no-phantom-GPU promise or local-model lifecycle runbook. |

## Minimal working spine

### Declarative input

A generated harness owns one versioned model-serving block:

- immutable model identity: source URL/repository, exact filename/revision, SHA-256, expected bytes, format `gguf`, and license identifier/acknowledgement requirement;
- runtime identity: one supported local adapter, exact executable/version constraint, supported GGUF/runtime capability, and ownership mode;
- launch policy: loopback host, allocated/declared port, context size, CPU/GPU policy, readiness timeout, and graceful-stop timeout;
- cache/update policy: explicit acquisition consent, content-addressed cache root, offline behavior, retention, and upgrade replacement rules.

### End-to-end scenario

1. `harness init` writes the declaration but does **not** download weights.
2. A plan command resolves bytes, license requirement, disk need, runtime compatibility, and measured CPU/RAM/VRAM; it makes no mutation.
3. Apply requires explicit consent, downloads to a temporary file, verifies byte count and SHA-256, then atomically promotes into the content-addressed cache.
4. Start selects a free loopback endpoint, launches the declared runtime with the verified local path, records PID/process identity and sanitized arguments, and refuses remote bind by default.
5. Readiness waits on the runtime health endpoint and then sends one bounded OpenAI-compatible one-token probe for the declared model.
6. A receipt records declaration digest, artifact digest, runtime identity, measured admission facts, endpoint, readiness evidence, timestamps, and ownership token.
7. Stop verifies ownership before signaling, waits boundedly, escalates according to policy, and records outcome without deleting model bytes.
8. Re-running offline uses the verified cached blob and reproduces the same declaration/artifact identity.
9. `selfcheck` and diagnostics consume the receipt and distinguish absent bytes, digest failure, incompatible runtime, insufficient resources, process failure, health timeout, protocol failure, and stale ownership.

### Spine witness

A hermetic test uses a tiny deterministic GGUF fixture and a fake runtime implementing `/health` plus one OpenAI-compatible completion. A captured live clean-directory run proves explicit plan/apply, digest verification, loopback startup, probe, receipt, stop, and offline cache reuse. A negative matrix proves digest mismatch, no consent, insufficient disk/RAM, unavailable requested GPU, occupied port, health timeout, and stale PID all fail closed with typed reasons.

## Boundaries

### Required safety and honesty boundaries

- **No hidden download:** init, build, selfcheck, and first inference cannot silently fetch bytes. Plan is read-only; apply names source, size, digest, cache destination, and license acknowledgement before mutation.
- **No phantom GPU:** `gpu:auto` is a measured policy, not a promise. A requested accelerator must be observed and admitted; otherwise the command returns a typed refusal or an explicitly declared CPU fallback.
- **No unpinned “latest”:** a model is exact revision/file/digest. Updates create a new declaration/receipt and preserve rollback until promotion succeeds.
- **No digest-by-report:** FAK computes SHA-256 over acquired bytes before atomic cache promotion.
- **No unsafe bind:** the managed spine binds loopback only. Remote exposure remains the existing gateway/policy problem, not a convenience default.
- **No PID-only kill:** stop must match an ownership token/process start identity before signaling.
- **No cache deletion on stop:** unload/stop/evict are separate operations; destructive cleanup is previewable and explicit.
- **No license laundering:** FAK records upstream license metadata/acknowledgement but does not assert redistribution rights it cannot prove.
- **No secrets in receipts:** URLs are sanitized and credentials remain in existing secret-provider surfaces.

### Gold-plating boundary

This cohort does **not** build a universal model registry, package manager, GUI marketplace, multi-node accelerator scheduler, arbitrary container manager, model conversion/quantization pipeline, or support matrix for every local server. It stops at one GGUF + one local-runtime adapter + one CPU-capable clean-directory witness. Other formats, remote registries, GPU tuning, and additional adapters require their own evidence after the spine works.

### Separation from adjacent research tracks

The seven children below own **serving lifecycle mechanics only**: declaration-to-local-bytes, runtime admission, child-process readiness/stop, lifecycle receipts, update/cleanup, and the end-to-end witness. They do not own general harness composition, provider routing, session UX, plugin packaging, or unrelated harness distribution domains covered by sibling research tracks.

## Issue decomposition

Exactly seven implementation leaves form the cohort. Titles are domain-prefixed and file scopes are explicit; shared command integration is ordered into later waves rather than presented as parallel-safe.

1. [**#8161 — Model-serving declaration schema and validation**](https://github.com/anthony-chaudhary/fak/issues/8161) — immutable artifact/runtime/launch/cache contract.
2. [**#8162 — Verified GGUF acquisition and cache promotion**](https://github.com/anthony-chaudhary/fak/issues/8162) — explicit plan/apply, digest, atomic cache.
3. [**#8163 — Local-runtime hardware and compatibility admission**](https://github.com/anthony-chaudhary/fak/issues/8163) — measured disk/RAM/VRAM and typed refusal.
4. [**#8164 — Harness-owned local server supervisor**](https://github.com/anthony-chaudhary/fak/issues/8164) — loopback launch, health + inference readiness, ownership-safe stop.
5. [**#8165 — Model-serving lifecycle receipt and diagnostics**](https://github.com/anthony-chaudhary/fak/issues/8165) — one durable truth consumed by selfcheck/inspect.
6. [**#8166 — Pinned upgrade, rollback, and explicit cleanup**](https://github.com/anthony-chaudhary/fak/issues/8166) — non-destructive stop, atomic upgrade, previewable eviction.
7. [**#8167 — Clean-directory bundled-serving spine witness**](https://github.com/anthony-chaudhary/fak/issues/8167) — generated harness scenario, negative matrix, docs.

The links above were independently read back from GitHub on 2026-08-19; the live issue bodies are the authoritative dispatch contracts.

## Validation standard

- Each child names `Parent: #8131`, `gen/now`, `harness-native`, `class:dev`, and priority.
- Each child carries Centrality, all P1–P4 dispositions in accepted `advanced`/`preserved`/`N/A - reason` form, Feynman frame, estimate, contribution, completion standard, target and witnessed envelopes, exact file-tree scope, and an independent witness.
- `fak-dev issue contract` must accept all seven in strict live mode after dedupe evidence.
- `fak-dev issue cohort` must produce a collision-aware wave plan rather than assuming shared `cmd/fak` integration can run concurrently.
- Parent closure requires the pushed note to be visible from `origin/main` and all seven exact child URLs to be independently readable from GitHub.

