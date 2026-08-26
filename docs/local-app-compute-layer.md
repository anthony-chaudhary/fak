---
title: "Ship local inference as an app capability, not a localhost server"
description: "Architecture and operating contract for fak local applications, including browser, daemon, accelerator, security, and offline boundaries."
---

# Ship local inference as an app capability, not a localhost server

**One-line product:** An app bundles fak once, declares the outcomes it needs, and gets measured local execution with explicit handoff when the Mac cannot meet the contract.

**Status:** proposed product contract for issue [#9131](https://github.com/anthony-chaudhary/fak/issues/9131)

## TL;DR

The product is an **app-scoped compute appliance** for Apple Silicon. It is not a daemon that users install and operate.

A developer adds a Swift SDK, signed helper, compute manifest, and model-pack channel to the app. fak then owns the difficult local-compute work:

- install and supervise the runtime;
- qualify each task on the current Mac;
- acquire and verify model packs;
- protect foreground memory, battery, and latency;
- execute through fak-native Metal;
- expose typed progress and lifecycle events;
- propose cloud handoff instead of hiding fallback;
- return a scrubbed receipt that proves what ran where.

The durable abstraction is an **outcome contract** such as `application-draft@1`. An OpenAI-compatible endpoint is only the migration bridge.

## Customer, problem, and alternative

**For:** one engineer shipping an agentic desktop product to about 10,000 users. The first release supports Apple Silicon Macs.

**Problem:** moving work from a cloud API to customer machines makes the app vendor responsible for a runtime. The vendor must package an engine, choose models, qualify hardware, manage memory pressure, preserve quality, supervise a helper, explain network use, recover from bad updates, and support diverse machines.

**Today:** keep Gemini or OpenAI as the default; ask users to install Ollama or LM Studio; embed llama.cpp; or assemble MLX and custom lifecycle code. Each option can generate tokens. None provides an app-complete compute contract.

**Better because:** one integration provides a signed app-scoped runtime. It also provides measured envelopes, visible local-versus-cloud decisions, safe updates, and support receipts.

**Witness:** a clean Mac installs a sample job-application app and completes onboarding. The app runs a real task through fak-native Metal, displays lifecycle and execution events, and produces a scrubbed receipt. A separate fixture demonstrates explicit cloud handoff. No terminal command or separate runtime installation is required.

**Centrality: Core.** Local execution is the primary path (P1). A real app outcome completes (P2). One integration replaces runtime assembly and support (P3). Install-to-result evidence makes adoption visible (P4).

## What the app vendor ships

1. **`FakKit` Swift SDK.** It provides typed requests, structured output, streaming, cancellation, backpressure, events, receipts, and an OpenAI-wire adapter.
2. **Signed `fak-runtime` helper.** This is an app-scoped XPC service or supervised sidecar. It authenticates its client, isolates crashes, and runs the native Metal engine.
3. **Compute manifest.** This signed file declares tasks, eligible model packs, quality and latency bounds, resource budgets, locality rules, and update channels.
4. **Model packs.** Signed manifests point to content-addressed weights. Downloads are resumable, revocable, and independent of app releases.

The v1 golden path is a Swift package plus an app-bundled helper. Tauri, Electron, and other native hosts can launch the same helper later. Users never need to operate a general-purpose model server.

```swift
let compute = try await FakKit.start(manifest: "JobApplyCompute")
for await event in compute.events { productState.consume(event) }

let result = try await compute.run(
    task: "job-application-tailor",
    input: applicationContext,
    constraints: .init(
        privacy: .localRequired,
        deadline: .seconds(20),
        qualityFloor: "job-apply-v1"))

support.attach(result.receipt.scrubbed())
```

Existing OpenAI-wire code can first change its transport, base URL, and model alias. Streaming, tool calls, schema output, cancellation, usage, and stable errors remain available. Production adoption then adds a compute manifest, readiness UI, handoff UI, and receipts.

## Product primitives

### Outcome contracts, not model names

The app requests `job-application-tailor@1`, not a specific quantization. Its task contract declares:

- input, output, and tool schemas;
- context and output bounds;
- a quality fixture and acceptance floor;
- a latency class and data-locality rule;
- eligible pack and runtime versions;
- memory, disk, energy, and concurrency limits;
- authority for remote handoff.

Models and kernels can change without an app release. They cannot silently weaken the declared behavior.

### Deterministic admission per task

At onboarding, fak inspects the machine and runs a representative fixture. It repeats admission after a material runtime, pack, OS, or hardware change. Each task receives one state:

- `ready`: quality, memory, and latency envelopes passed;
- `ready_degraded`: only named bounds passed;
- `download_required` or `warming`;
- `temporarily_unavailable`: current pressure, battery, thermal, disk, or contention prevents use;
- `unsupported`: no declared envelope passed.

Admission uses measurements from this machine, not only its chip label. The app can say, “Local for extraction and drafting; approval needed for long-form cloud review.” It does not show one misleading global support percentage.

### Resource coexistence is part of correctness

A local result is not successful if it makes the host app or the rest of the Mac unusable. The compute allowance therefore covers:

- memory high-water marks and pressure response;
- foreground and background concurrency;
- queue bounds and cancellation;
- battery state, AC power, and Low Power Mode;
- thermal downshift;
- disk reservation and eviction;
- foreground latency impact.

User modes are Automatic, Prefer local, Local only, and Pause local. Work that exceeds the current allowance can queue or select an already-qualified smaller envelope. Otherwise fak proposes handoff.

### Explicit handoff, never invisible fallback

Remote handoff has three separate decisions:

1. **Eligibility:** may these data classes leave the device?
2. **Trigger:** did the local path hit a deadline risk, unsupported feature, quality miss, pressure event, fault, or user request?
3. **Authority:** was handoff pre-consented, must the app ask now, or is it forbidden?

A handoff event names the reason, affected data classes, destination class, consequence, and alternatives. The app owns cloud credentials, billing, provider selection, and consent UX. fak preserves request identity and returns a normalized handoff package.

Receipts record every local and remote attempt. Remote work is never labeled local.

### Measure locality honestly

A single “percent local” hides quality rejects and fallback loops. Measure four separate shares:

- **eligible:** policy and admitted capability allowed a local attempt;
- **attempted:** local execution started;
- **accepted:** the result met the task contract;
- **used:** the app used the accepted local result.

Every gap carries a reason code. Cloud savings count only accepted, used local results after retry and verification cost.

### Developer event plane

Events are versioned and ordered per operation. A bounded local journal supports replay. The SDK exposes Swift `AsyncSequence`; callbacks and local SSE are adapters.

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

Every event carries an event ID, operation ID, sequence, timestamp, task contract, relevant revisions, privacy class, and stable reason code. Prompt and output text are absent by default.

### Receipts are the trust and support primitive

Each operation returns a scrubbed receipt. It records:

- local, remote, or mixed execution;
- the actual fak-native engine;
- task, runtime, kernel, and pack revisions;
- admitted and observed envelopes;
- time to first token, decode rate, queue time, and total latency;
- peak memory and token/cache facts;
- quality verification;
- retries, downshifts, and handoffs;
- policy authority and terminal diagnostic code.

Receipts omit prompts, outputs, private paths, usernames, and stable device identifiers. A debug bundle requires a separate user preview and consent step.

### Guarantees are bounded envelopes

fak cannot promise one token rate across every Mac. It can promise the following:

- readiness requires a representative task fixture;
- admitted work stays inside declared bounds or ends with a typed miss;
- updates pass fixture canaries before activation;
- a miss never silently changes quality, locality, or engine;
- last-known-good packs remain rollbackable;
- local-required data never enters a handoff package;
- native performance work never silently executes through llama.cpp.

Initial device classes can use M1–M4 and memory bands for planning. Per-device admission remains authoritative.

### Installation and updates are product UI

First run verifies the helper and manifest, then inspects task needs. The app shows the exact download size, storage reservation, privacy boundary, and expected capability. fak fetches the starter pack, verifies it, stages it atomically, runs fixtures, and activates it only after success. Tasks become available independently rather than waiting behind one global spinner.

Runtime updates ride normal signed and notarized app updates in v1. Model packs use a signed app-controlled channel. That channel supports staged activation, fixture gating, revocation, and rollback.

The app manifest also carries model license and redistribution metadata. Installation refuses a pack when the vendor lacks the declared entitlement or when required attribution is missing. The app vendor, not fak, owns commercial model terms.

### App sandbox and local security boundary

The helper accepts only authenticated requests from its host app. It binds to XPC or a loopback endpoint protected by a per-install capability. It does not expose a general LAN service. Sandboxed file access uses explicit app-provided handles or mediated tools. Code signing, notarization, helper identity, pack signatures, and downgrade protection are part of admission.

### Offline behavior and data lifecycle

After the required pack is present, local-required tasks work without a network connection. Downloads pause and resume safely. Remote-eligible work returns a typed `network_unavailable` state rather than spinning.

Uninstall and “delete local AI data” remove app-scoped packs, journals, receipts, and credentials according to the manifest. Shared caches are out of scope for v1. Diagnostic export always has a preview step.

### Failure is a designed state

Stable failure classes cover incompatible OS or runtime, insufficient memory or disk, unavailable packs, corrupt or revoked packs, admission failure, deadline risk, pressure, helper restart, schema mismatch, tool mismatch, quality reject, and handoff failure. Each class maps to an app action and safe receipt. Crash loops trip a circuit breaker while the host app remains usable.

### Developer testing and compatibility

FakKit ships an in-process fake and deterministic fixture runner. Product teams can test every readiness, progress, pressure, failure, and handoff state without downloading a model. Golden-wire fixtures cover the supported OpenAI subset.

The SDK, manifest, event, reason-code, and receipt schemas use explicit versions. Minor revisions are additive. Breaking revisions require a new major contract and a migration window. The product publishes the supported runtime-pack-SDK matrix.

## Job-application app example

The app declares four outcome contracts:

| Task | Locality and envelope |
|---|---|
| `resume-extract@1` | local required, structured, small, interactive |
| `job-fit@1` | local preferred, quality-gated, interactive |
| `application-draft@1` | local preferred, medium context, cloud only after consent |
| `browser-action-plan@1` | local required for planning; browser tools remain under fak policy |

The app bundles FakKit, its helper, and the manifest. First run downloads the smallest pack that covers the initial tasks. Admission makes each feature available as soon as its fixture passes.

For each request, fak validates the schema, privacy rule, admission state, and current resource allowance. It reserves memory and deadline budget, streams events, and executes through fak-native Metal. Mediated tools remain behind the existing fak policy floor. Finally, it verifies the output schema and task-quality rule before returning the output and receipt.

If the envelope cannot be met, the app receives a handoff proposal. It never receives a disguised cloud retry.

**Day-one migration:** change transport, base URL, and task alias while retaining stream, tool, and schema behavior.

**Production adoption:** add the manifest; consume readiness and handoff events; render download and resource states; attach receipts to support flows.

A no-code replacement is not promised. An app that hides fallback and resource state cannot provide a trustworthy local product.

## v1 boundary

Ship:

- macOS 14+ signed and notarized app integration;
- Apple Silicon execution only;
- Swift SDK and app-scoped helper;
- compute-manifest and task-contract schemas;
- one commercially usable signed pack for supported 16 GB+ machines;
- OpenAI-compatible streaming adapter plus native task API;
- readiness, asset, generation, pressure, handoff, maintenance, and terminal events;
- Local only and Prefer local policy with an app-owned remote callback;
- scrubbed receipts and consented diagnostic export;
- deterministic test fake and compatibility fixtures;
- a job-application sample with structured output and one mediated tool;
- measured admission and rollback-gated updates.

Defer cloud control, Windows, Linux, arbitrary bring-your-own models, LAN serving, cross-app sharing, provider billing, shared caches, and a model marketplace.

## Smallest working spine

A signed sample app installs on a clean supported Mac. It starts the bundled helper, downloads and verifies one pack, runs admission, then executes `resume-extract@1` and `application-draft@1` through fak-native Metal. The UI renders typed events.

A second fixture forces local unavailability. The app shows the reason and completes a user-approved remote handoff. Both runs emit scrubbed receipts.

The captured witness includes:

- install-to-ready time and bytes;
- Mac, OS, memory, and power envelope;
- fixture quality result;
- time to first token, total latency, decode rate, memory, and disk;
- restart, offline, and cancellation behavior;
- no separate runtime installation;
- truthful engine and location fields in each receipt;
- rollback after a deliberately rejected pack update;
- uninstall and local-data deletion.

Compose existing work instead of reopening it: native Qwen/Metal [#8011](https://github.com/anthony-chaudhary/fak/issues/8011), resource admission [#8163](https://github.com/anthony-chaudhary/fak/issues/8163), router coexistence [#8176](https://github.com/anthony-chaudhary/fak/issues/8176), lifecycle demo [#8555](https://github.com/anthony-chaudhary/fak/issues/8555), and outcome receipts [#8402](https://github.com/anthony-chaudhary/fak/issues/8402).

## Borrowed field patterns

- Apple Foundation Models validates task-native guided generation and tools. System-model availability does not replace an app-controlled quality envelope: <https://developer.apple.com/documentation/foundationmodels>.
- MLX and MLX-LM demonstrate unified-memory-native execution and friendly model workflows. fak should borrow the ergonomics while retaining its own guarantees: <https://github.com/ml-explore/mlx> and <https://github.com/ml-explore/mlx-lm>.
- Ollama validates a simple local API and named artifacts. A separately operated general runtime is not app embedding: <https://docs.ollama.com/api/introduction>.
- LM Studio validates OpenAI-wire migration. Endpoint compatibility is table stakes: <https://lmstudio.ai/docs/developer/openai-compat>.
- llama.cpp remains a benchmark, reference, and borrowing source. The fak-native invariant still governs product execution: <https://github.com/ggml-org/llama.cpp>.
- Tauri sidecars validate bundled-helper distribution. fak adds authenticated app scope, lifecycle, signed updates, admission, and receipts: <https://v2.tauri.app/develop/sidecar/>.
- OpenAI streaming and structured output are migration baselines, not the durable abstraction: <https://platform.openai.com/docs/api-reference/responses>.

## Moat, economics, and measures

Metal token generation alone is not the moat. The compounding asset is the outcome graph across diverse machines:

- task contracts and quality fixtures;
- per-device admission history;
- coexistence scheduling;
- signed updates proven before activation;
- receipts joined to app outcomes;
- privacy-preserving local analytics;
- policy-mediated tools at the same checkpoint as performance.

For each task and device class, measure install-to-ready success and time; local eligibility, attempts, acceptance, and use; reason-coded handoffs; p50 and p95 latency; peak memory and disk; foreground impact; crash-free operations; rollback success; and support resolution from receipts.

The vendor-facing unit is **accepted local outcomes per active device-month**. Cost accounting includes CDN bytes, storage, verification, retries, support, and any remote attempt. Report cloud cost avoided only after those costs. Optimize customer-visible outcomes per resource cost, not raw tokens or one global local percentage.
