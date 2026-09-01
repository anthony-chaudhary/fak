---
title: "Builder contract ladder"
description: "The stability, ownership, and compatibility map for products built on fak."
---

# Builder contract ladder

This is the entry point for teams building durable harnesses, UIs, workers, and integrations on fak. Choose the highest stable level that solves the job; descend only when the extra coupling is intentional.

## Value and centrality

- **For:** teams building durable harnesses, UIs, workers, and integrations.
- **Problem:** fak has useful primitives, but builders lacked one stability and ownership map.
- **Today:** builders infer boundaries from scattered documents and private implementation packages.
- **Better because:** one ladder keeps integrations on stable semantic envelopes, prevents provider/internal coupling, and makes asynchronous ownership explicit.
- **Witness:** the source-backed tables below, deterministic documentation/link checks, and runnable examples as they land through the linked gap issues.

This is **Core** work. fak owns the semantic center: governance, run and input identity, event meaning, compatibility, and witnessed outcomes. Builders own domain projections, durable application state, and the choice and operation of external brokers.

## Builder jobs to be done

Builders need to launch and supervise governed runs; render one run in CLI, terminal, web, or embedded views; resume event consumption after disconnects; dispatch work through an existing queue; add bounded tools and extensions; preserve identity and proof across transports/providers; and replay fixtures for evaluation and incident diagnosis.

## The five levels

| Level | Use it when | Stable surface | Coupling and promise |
|---|---|---|---|
| 1. CLI/process | Shells, CI, operators, and polyglot prototypes need process isolation. | Documented commands, exit behavior, and named machine-readable modes. | Lowest source coupling. Human text is not a parsing contract. |
| 2. Semantic run protocol | A UI, consumer, or worker must understand a run independent of transport or language. | Versioned identity, checkpoints, events, ordering, replay, compatibility, cancellation, approval, and sensitivity in [the protocol](harness-protocol.md). | The stable center. Consumers accept additive unknown events and never infer provider objects. |
| 3. Public `pkg/harnesskit` Go API | A Go product wants typed values and approved adapters in-process. | Exported contracts in [the Go contract](harness-kit-contract.md). | Only the package's documented maturity applies; no promise covers `internal/`. |
| 4. Language-neutral sidecar protocol | A non-Go or separately deployed product needs level-2 semantics over a boundary. | Versioned wire envelopes and negotiated capabilities that preserve semantic IDs and acknowledgements. | The protocol, not a particular HTTP, stdio, broker, or generated client, is the contract. |
| 5. Internal implementation leaves | fak maintainers change kernel mechanics. | None for external builders. | May change at any time; this is not an advanced integration tier. |

Prefer levels 1-4 in order. Descend only when the earlier level cannot satisfy the job.

## Guarantees and non-guarantees

At the semantic boundary fak guarantees stable run/input identity independent of provider objects; documented event ordering scope; replay-safe identity; bounded credit/backpressure; meaningful cancellation and sensitivity markers; additive-event compatibility; kernel-owned governance; and witnessed outcomes tied to evidence rather than self-report.

The exact event fields, validation rules, ordering scope, and compatibility behavior remain authoritative in [Headless run protocol v1](harness-protocol.md). This page assigns ownership rather than duplicating that specification.

fak does **not** guarantee exactly-once delivery or exactly-once external effects; provide a built-in durable broker, workflow DSL, application database, or second scheduler; promise cross-stream ordering unless specified; retain events forever; permit unbounded replay/buffering; stabilize provider SDK request/response objects; or stabilize `internal/agentqueue`, `internal/leasequeue`, `internal/harnessprotocol`, or any other `internal/` package.

The delivery contract is **at least once with idempotent effects**, never exactly once.

## Async ownership

| Concern | fak owns | Builder owns |
|---|---|---|
| Identity | Run, input, event, checkpoint, and witnessed-outcome identity | Mapping IDs to domain records and effect keys |
| Event meaning | Envelope schema, ordering scope, compatibility, sensitivity, cancellation, and resume semantics | Projection shape, query model, retention, and presentation |
| Persistence | Validation of semantic cursors/checkpoints at fak boundaries | Atomic persistence of projection/effect state with the committed cursor |
| Backpressure | Credit/batch semantics and bounded production | Consumer concurrency, storage capacity, retry timing, and admission policy |
| Queues | Narrow adapters preserving identity, governance, and receipts | Broker choice, topology, durability, credentials, operations, and dead-letter policy |
| Effects | Stable identifiers and witnessed completion semantics | Idempotency table/outbox/inbox and domain transaction boundaries |

### Delivery, replay, cancellation, and sensitivity

1. Treat every delivery as repeatable. Deduplicate by stable semantic identity, not payload text or broker offset alone.
2. Apply a projection/effect and advance its semantic cursor atomically. If one transaction is impossible, use an inbox/outbox or equivalent recoverable protocol.
3. Resume after the last committed semantic cursor and request bounded credit; unbounded replay is not flow control.
4. Propagate cancellation promptly and make cleanup idempotent because delivery and cancellation can race.
5. Preserve sensitivity metadata through queues and projections; retain redacted/minimal fields when raw payload is unnecessary.
6. Ignore compatible unknown additive events and reject incompatible versions explicitly.

### Two acknowledgements, two meanings

A **broker message acknowledgement** tells Kafka, NATS, SQS, or another transport what to redeliver. A **semantic event-cursor acknowledgement** tells the consumer which fak run event was durably and idempotently applied. They are not interchangeable.

Safe order: receive broker message → validate envelope → idempotently apply the domain change and commit semantic cursor → acknowledge broker message. A crash before broker acknowledgement may redeliver an already committed event; idempotency makes that harmless. A broker offset alone cannot prove the domain effect committed.

## Queue ownership decision table

| Need | Recommended owner | Why |
|---|---|---|
| One local/interactive run | fak CLI/process or in-process harness | No external queue is needed. |
| Durable enterprise work dispatch | Builder's broker plus a narrow fak adapter | The builder owns availability, credentials, topology, retention, and operations. |
| Semantic replay for projections | fak protocol cursor plus builder persistence | Broker position does not replace semantic acknowledgement. |
| Workflow dependencies/business retries | Builder orchestration system | fak should not become a workflow DSL or second scheduler. |
| Kernel-local resource admission | fak internal scheduler | Execution governance is not a public general-purpose queue. |
| Cross-broker portability | Small adapter around stable job/run/input IDs | Bridge required operations only; do not normalize every broker feature into fak. |

Adapters should expose only the job, delivery/lease, acknowledgement, retry, and dead-letter concepts required. Broker-native tuning remains builder configuration.

## Extension trust and runtime boundaries

An extension is untrusted input to governance, not inherited kernel authority. [Extension seams](extension-seams.md) require declared capabilities, pre-execution validation, explicitly scoped services, preserved cancellation/quotas/sensitivity/audit identity, containment of panics/hangs/malformed or oversized output, and typed attributable outcomes. Prefer process/sidecar isolation for third-party or failure-prone code. Provider SDK objects stay behind provider adapters.

## Public versus internal API

| Surface | Builder status | Notes |
|---|---|---|
| Documented `fak` CLI machine modes | Public | Default for process and polyglot integration. |
| [Headless run protocol](harness-protocol.md) envelopes | Public semantic contract | Authoritative center for events and replay. |
| `pkg/harnesskit` | Public Go contract | Use exported, documented contracts at their stated maturity. |
| Documented versioned sidecar/wire protocol | Public | Must preserve semantic protocol; generated clients are replaceable. |
| `internal/agentqueue` | Internal | Never publish/import as the queue contract. |
| `internal/leasequeue` | Internal | Kernel coordination detail, not broker abstraction. |
| `internal/harnessprotocol` | Internal | Implementation of public semantics, not the import surface. |
| Other `internal/*` and provider SDK structs | Internal/private | No external compatibility guarantee. |

## Runnable example coverage

| Builder job | Runnable evidence | Status / gap |
|---|---|---|
| Synchronous CLI/process integration | Existing CLI and integration examples | Available in `examples/` and `docs/integrations/`. |
| Replay-safe event consumer | Public-envelope disconnect/redelivery witness | Gap: [#10576](https://github.com/anthony-chaudhary/fak/issues/10576). |
| BYO-queue worker | Separate queue/event acknowledgements | Gap: [#10578](https://github.com/anthony-chaudhary/fak/issues/10578). |
| Terminal UI | Run projection rendered in a TUI | Gap: [#10561](https://github.com/anthony-chaudhary/fak/issues/10561). |
| Web UI | Run projection rendered over a web boundary | Gap: [#10562](https://github.com/anthony-chaudhary/fak/issues/10562). |
| Embedded Go agent | Product importing only public `pkg/harnesskit` | Gap: [#10563](https://github.com/anthony-chaudhary/fak/issues/10563). |
| Custom tool/MCP extension | Scoped registration and authority proof | Existing owner: [#3265](https://github.com/anthony-chaudhary/fak/issues/3265); do not duplicate. |
| Replay/evaluation | Deterministic fixtures and expected outcomes | Gap: [#6804](https://github.com/anthony-chaudhary/fak/issues/6804). |
| Extension failure isolation | Panic/hang/malformed-output containment | Gap: [#10428](https://github.com/anthony-chaudhary/fak/issues/10428). |

A row is available only when a runnable artifact demonstrates the boundary. Planned examples are not shipped capability.

## Selection rule

Start with CLI/process. Move to the semantic protocol when several views or transports must agree. Use `pkg/harnesskit` for typed Go embedding. Add a sidecar only when language or deployment isolation requires it. If a design needs an internal leaf, first identify the smaller missing public semantic contract; do not publish the internal type by default.
