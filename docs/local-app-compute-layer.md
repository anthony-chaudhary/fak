# Ship local inference as an app capability, not a localhost server

**Status:** proposed product contract for issue [#9131](https://github.com/anthony-chaudhary/fak/issues/9131)
**v1:** signed macOS apps on Apple Silicon; fak-native Metal; no cloud control plane

## Verdict

The valuable product is an **embeddable compute appliance for one app**. A developer adds a small SDK and signed helper to the app bundle. The helper owns model selection, installation, admission, execution, updates, resource budgets, and recovery. The SDK keeps a familiar model-call shape while exposing facts a cloud API hides: whether a request can run locally, what it costs the user's machine, what is happening now, why execution moved elsewhere, and whether the result met the app's quality and latency envelope.

An OpenAI-compatible localhost endpoint is useful for migration, but is not the product. It does not solve distribution, user trust, device diversity, quality guarantees, resource contention, upgrades, support, or explicit fallback.

> **The app declares an outcome envelope; fak turns each supported Mac into a measured, app-scoped inference appliance and returns a verifiable execution receipt.**

## Customer and alternative

**For:** one engineer shipping an agentic desktop product to roughly 10,000 users, initially on Apple Silicon Macs.

**Problem:** moving work from a cloud API to customer machines currently means becoming a runtime vendor: package an engine, choose/download models, classify hardware, prevent memory pressure, retain quality, manage helper lifecycle, explain network use, recover from corruption, and create support diagnostics.

**Today:** keep Gemini/OpenAI/Codex as default; tell users to install Ollama or LM Studio; embed llama.cpp; or assemble MLX/Core ML and custom lifecycle code. Each can execute tokens. None is an app-complete compute contract.

**Better because:** one integration supplies a signed app-scoped runtime, deterministic admission, measured envelopes, visible local/cloud decisions, typed events, safe updates, and support receipts.

**Witness:** a clean Mac installs a sample job-application app, completes onboarding, executes a real task through fak-native Metal, displays lifecycle/execution events, produces a scrubbed receipt naming engine and envelope, and exercises explicit cloud handoff. No separate runtime installation or terminal command is required.

**Centrality: Core.** This turns fak's engine, gateway, policy, cache, and observability into an adoptable product. P1: local execution is the primary path. P2: a developer ships a real app outcome. P3: one integration replaces assembly and support of a separate runtime. P4: install-to-ready, device coverage, local accepted-result share, latency/quality, handoff reason, and update behavior are visible.

## What developers ship

1. **`FakKit` Swift SDK:** typed requests, structured output, cancellation, backpressure, events, receipts, and an OpenAI-wire migration adapter.
2. **Signed `fak-runtime` helper:** app-scoped XPC service or supervised sidecar, authenticated client, native Metal engine, isolated failure domain.
3. **Compute manifest:** signed tasks, model packs, quality/context/latency bounds, storage/memory budgets, local/cloud policy, and update channels.
4. **Model packs:** content-addressed signed manifests and quantized weights fetched after install; resumable, revocable, independently updateable.

The v1 golden path is a Swift package plus app-bundled helper. Tauri/Electron/native apps can launch the same helper later. Users should not install or operate a separate general-purpose server.

```swift
let compute = try await FakKit.start(manifest: "JobApplyCompute")
for await event in compute.events { productState.consume(event) }

let result = try await compute.run(
    task: "job-application-tailor",
    input: applicationContext,
    constraints: .init(privacy: .localRequired,
                       deadline: .seconds(20),
                       qualityFloor: "job-apply-v1"))

support.attach(result.receipt.scrubbed())
```

For existing OpenAI-wire code, migration changes transport/base URL and model alias while retaining streaming, tool calls, schema output, cancellation, usage, and stable errors. Production adoption then adds a compute manifest, readiness/handoff UI, and receipts.

## Product primitives

### Outcome contracts, not model names

The app asks for `job-application-tailor@1`, not a particular Qwen quantization. A task contract declares input/output schemas, tool grammar, context/output bounds, quality fixture and floor, latency class, data-locality rule, eligible pack/runtime versions, memory/disk/energy/concurrency limits, and handoff authority. Models and kernels can evolve without an app release, but cannot silently weaken behavior.

### Deterministic device admission

At onboarding and after material changes, fak runs a representative fixture and emits:

- `ready`: quality, memory, and latency envelopes witnessed;
- `ready_degraded`: only named tasks or bounds admitted;
- `download_required` or `warming`;
- `temporarily_unavailable`: pressure, battery, thermal, disk, or competing work;
- `unsupported`: no declared task can be met.

Admission uses measured machine facts, not only chip labels. UI can say “Local for extraction and drafting; approval needed for long-form cloud review,” rather than presenting a misleading global percentage.

### Resource coexistence is correctness

The compute allowance covers memory high-water mark, pressure response, foreground/background concurrency, queue bounds, battery/AC and Low Power Mode, thermal downshift, disk reservation/eviction, and foreground latency. User modes are Automatic, Prefer local, Local only, and Pause local. A request that violates this allowance is not locally capable now: it queues, selects an already-qualified smaller envelope, or offers handoff.

### Explicit handoff, never invisible fallback

Fallback has three policies: eligibility (may this data leave?), trigger (deadline, unsupported feature, quality miss, pressure, fault, or user choice), and authority (pre-consented, ask, or never). A handoff event names the reason, data classes, destination chosen by the app, consequence, and alternatives. The app owns cloud credentials and provider UX; fak returns a normalized handoff package and preserves request identity. Receipts record every attempt and never label remote work local.

Measure local-eligible, local-attempted, local-accepted, and local-result-used shares plus reason-coded handoffs. A single “percent local” conceals quality rejects and fallback loops.

### Developer event plane

Events are versioned, ordered per operation, replayable from a bounded local journal, and exposed as Swift `AsyncSequence`, callback, or local SSE.

| Family | Examples | Product use |
|---|---|---|
| lifecycle | `runtime.starting`, `runtime.ready`, `runtime.recovering` | readiness and recovery |
| assets | `pack.required`, `download.progress`, `pack.verified`, `pack.evicted` | onboarding and storage UI |
| admission | `task.ready`, `task.degraded`, `task.unsupported` | feature gates |
| scheduling | `request.queued`, `request.started`, `pressure.changed` | progress and cancellation |
| generation | `output.delta`, `tool.requested`, `usage.updated` | agent interaction |
| handoff | `handoff.proposed`, `handoff.approved`, `handoff.denied` | cloud transition |
| terminal | `request.completed`, `request.failed`, `request.cancelled` | deterministic state |
| maintenance | `update.available`, `update.applied`, `rollback.applied` | safe lifecycle |

Every event includes an ID, operation ID, monotonic sequence, timestamps, task contract, runtime/pack revisions, privacy class, and stable reason code. Prompt and output text are absent by default.

### Receipts are the trust and support primitive

Each operation returns a scrubbed receipt naming local/remote/mixed execution; actual fak-native engine; task/runtime/kernel/pack revisions; admitted envelope; observed TTFT, decode rate, total latency, queue and peak memory; token/cache facts; quality verification; retries/downshifts/handoffs; policy and user authority; and terminal diagnostic code. Receipts omit prompts, output, paths, usernames, and stable device identifiers. Debug bundles require separate user preview and consent.

### Guarantees are bounded envelopes

fak cannot promise one token rate across every Mac. It can guarantee that readiness requires a representative fixture; admitted work remains inside declared bounds; updates pass fixture canaries before activation; misses are typed rather than silently changing quality or engine; last-known-good packs remain rollbackable; local-required data never enters handoff; and native/performance paths never silently execute through llama.cpp. Device classes (M1–M4 and memory bands initially) remain subordinate to per-device admission.

### Installation and updates are product UI

First run verifies helper/manifest, inspects machine and task needs, shows exact download/storage/privacy/capability, fetches a signed resumable starter pack, verifies/stages atomically, runs fixtures, activates only on success, retains last known good, and emits readiness per task. Runtime updates ride normal app updates in v1. Model packs use a signed app-controlled channel with staged activation, fixture gating, revocation, and rollback.

### Failure is a designed state

Stable classes include incompatible OS/runtime, insufficient memory/disk, unavailable/corrupt/revoked pack, admission failure, deadline risk, pressure, helper restart, schema/tool mismatch, quality reject, and handoff prohibited/declined/failed. Each maps to an app action and safe receipt. Crash loops trip a circuit breaker while the host app stays usable.

## Job-application app example

The manifest defines:

- `resume-extract@1`: local required, structured, small/interactive;
- `job-fit@1`: local preferred, quality-gated, interactive;
- `application-draft@1`: local preferred, medium context, cloud only after consent;
- `browser-action-plan@1`: local required for planning; browser tools remain under fak policy.

The app bundles FakKit, helper, and manifest. First run downloads the smallest pack covering initial tasks and witnesses readiness. Features appear as each task becomes ready instead of waiting behind one global spinner.

Request flow: validate schema/privacy/admission/allowance; reserve memory and deadline; stream scheduling and output events; execute via fak-native Metal and mediated tools; verify schema and task quality; return output and receipt. If the envelope cannot be met, return a handoff proposal rather than disguising a cloud retry.

**Day-one migration:** change transport/base URL, task alias, and retain stream/tool/schema behavior.
**Production adoption:** add manifest, consume readiness/handoff events, render download/privacy/resource states, attach receipts.
A no-code local replacement is not promised because an app that hides fallback and resource states cannot be trustworthy.

## v1 boundary

- macOS 14+ signed/notarized app integration; Apple Silicon execution only;
- Swift SDK, app-scoped helper, compute-manifest schema;
- one signed model pack selected for supported 16 GB+ machines;
- OpenAI-compatible streaming adapter plus native task API;
- readiness, download, generation, pressure, handoff, and terminal events;
- local-only/local-preferred policy and app-owned remote callback;
- bounded receipt and local diagnostic export;
- job-application sample with structured output and one mediated tool;
- measured admission and rollback-gated updates.

Defer cloud control, Windows/Linux, arbitrary BYO models, LAN serving, cross-app sharing, provider billing, and a model marketplace.

## Smallest working spine

A signed sample app installs on a clean declared Apple Silicon Mac. It starts the bundled helper, downloads/verifies one pack, runs admission, and executes `resume-extract@1` plus `application-draft@1` through fak-native Metal. UI renders typed events. A second fixture forces local unavailability and demonstrates app-owned, user-approved remote handoff. Both runs emit scrubbed receipts.

Capture install-to-ready time/bytes; Mac/OS/memory/power envelope; fixture quality; TTFT/latency/tokens per second/peak memory/disk; restart and cancellation; absence of separate runtime install; receipt engine and location truth; and rollback after a deliberately rejected pack update.

Compose rather than reopen existing work: native Qwen/Metal [#8011](https://github.com/anthony-chaudhary/fak/issues/8011), resource admission [#8163](https://github.com/anthony-chaudhary/fak/issues/8163), router coexistence [#8176](https://github.com/anthony-chaudhary/fak/issues/8176), lifecycle demo [#8555](https://github.com/anthony-chaudhary/fak/issues/8555), and receipt/outcome joining [#8402](https://github.com/anthony-chaudhary/fak/issues/8402).

## Borrowed field patterns

- Apple's Foundation Models framework validates task-native guided generation and tools, but system model availability does not replace an app-controlled quality/performance envelope: <https://developer.apple.com/documentation/foundationmodels>.
- MLX/MLX-LM demonstrate Apple unified-memory-native execution and conversion; fak should borrow ergonomics while owning application guarantees: <https://github.com/ml-explore/mlx> and <https://github.com/ml-explore/mlx-lm>.
- Ollama validates a simple local API and named artifacts, but a separately operated general runtime is not app embedding: <https://docs.ollama.com/api/introduction>.
- LM Studio validates OpenAI-wire migration and shows that endpoint compatibility is table stakes: <https://lmstudio.ai/docs/developer/openai-compat>.
- llama.cpp is benchmark/reference and borrowing source; the fak-native invariant still governs product execution: <https://github.com/ggml-org/llama.cpp>.
- Tauri sidecars validate bundled helper distribution; fak adds signing, authenticated app scope, lifecycle, update, and receipt contracts: <https://v2.tauri.app/develop/sidecar/>.
- OpenAI streaming and structured output are migration baselines, not the durable abstraction: <https://platform.openai.com/docs/api-reference/responses>.

## Moat and measures

The moat is not Metal token generation. It is the outcome graph across diverse machines: task contracts and quality fixtures; per-device admission; coexistence scheduling; signed updates proven before activation; events/receipts joined to app outcomes; privacy-preserving local analytics; and policy-mediated tools at the same checkpoint as performance.

For each task and device class measure install-to-ready success/time; eligible, attempted, accepted, and used local shares; reason-coded handoffs; p50/p95 TTFT and end-to-end latency; peak memory/disk and foreground impact; crash-free operations and rollback; incidents resolved from receipts; and cloud cost avoided only for accepted local results net of retries and verification. Optimize accepted outcomes per customer-visible resource cost, not a global local percentage.
