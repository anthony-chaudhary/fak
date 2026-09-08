<!-- fak-contexttrace-key: native-kv-lifecycle-v1 -->

```routing
lane: cachemeta
paths: ["internal/model/**", "internal/cachemeta/**"]
expected_steps: 8
```

# feat(contexttrace): trace live native context and KV lifecycle transitions

## Problem

For an agent asking why a turn reused, recomputed, rolled back, restored, or lost context, current facts are split between session state, KV stores, prefix snapshots, lifecycle policy, and aggregate cache events. Today a final cache length or hit counter cannot reconstruct which semantic context span became which KV blocks, which positions were accepted, or which transition removed them. Better because one bounded live trace joins context identity to native KV lifecycle events—fill, append, reuse, snapshot, clone, restore, move, rollback, evict, and free—without logging prompt text or KV bytes. The witness executes a tiny native session through prefill, snapshot, speculative append/rollback, restore, and close and reconstructs the exact surviving prefix after every transition.

## Parent context

First-class public `fak` dataflow tracing goal. This leaf covers native KV ownership and lifecycle by adapting domain actions to the canonical trace ABI and replay engine; it remains distinct from provider cache attribution.

## Why now

#8447, #5260, and #9307 established cache-event vocabulary, while #8586 still lacks a live native context lifecycle witness through restore, reuse, rollback, and free. The shared trace contracts and deterministic replay primitive must land first so native lifecycle evidence is composable and replay means one thing.

## Current state

Inspected at public `fak` commit `325ade2db2a7b1ea556d03d89293162a85cd1e9e`:

- `internal/model/hal.go:736-784` appends live per-layer K/V into the backend store, but no normalized event binds the append to semantic context/token lineage.
- `internal/model/kv.go:41-97` snapshots host/device KV and recurrent state; `internal/model/kv.go:100-170` clones/restores ownership; `internal/model/kv.go:173-185` frees device state. These real transitions are not captured as one lifecycle artifact.
- `internal/model/kv.go:1300-1316` truncates speculative KV by token count but emits no accepted/rejected position edge.
- `internal/cachemeta/cache_event_consolidator.go:15-92` defines stable logical block and source event identity, while `:162-245` consolidates only logical store/remove visibility. It does not represent native append/snapshot/restore/rollback ownership flow.

## Dependencies and schedule

- Hard prerequisite: `typed-causal-envelope-v1` for causal identity and sequence.
- Hard prerequisite: `many-to-many-value-lineage-v1` for context/KV range derivation and ownership transforms.
- Hard prerequisite: `bounded-private-evidence-policy-v1` for capture, privacy, loss accounting, and evidence ceilings.
- Hard prerequisite: TICKET-05 `deterministic-state-reconstruction-v1`; this issue supplies a domain reducer and must not create another replay/result vocabulary.
- Schedule after TICKET-08. TICKET-06, TICKET-08, and TICKET-09 serialize in that order on `internal/model/**`.

## Centrality and P1-P4

Centrality: Core

P1: advanced - makes context continuity, reuse, and loss visible while keeping text and tokens digest-safe.

P2: advanced - distinguishes true reuse from recompute and records byte/event overhead without treating modeled savings as observed.

P3: advanced - every restore or reuse is bound to model, tokenizer, serialization, position/RoPE, cache geometry epoch, and backend; incompatible state fails closed.

P4: advanced - a bounded ordered artifact explains native KV ownership and survives partial or failing session paths.

## Scope

Trace native context and KV lifecycle transitions across `internal/model` and `internal/cachemeta` through adapters to public `pkg/traceabi` and the TICKET-05 replay engine; exclude provider-managed cache attribution and changes to the shared contracts.

## Core through-line

1. Reuse `CacheLogicalBlockKey` and source-local sequence identity as domain correlations on `traceabi.CausalEnvelope`, with session/context span, token-position interval, layer coverage, tier, precision, owner, and parent transition.
2. Emit domain action payloads for `fill_begin/fill_commit`, `append`, `reuse`, `snapshot`, `clone`, `restore`, `move`, `rollback`, `evict`, and `free`; use canonical lineage transforms for input/output ranges and distinguish intent, successful completion, refusal, and partial failure.
3. Bind semantic context by digest and interval only—no prompt text, token values, K/V payloads, pointers, or private cache keys.
4. Record before/after token lengths and accepted/rejected ranges so speculative rollback and compaction are reconstructible.
5. Feed visibility-changing transitions through the existing source-aware consolidator and reconstruct them with a TICKET-05 domain reducer rather than creating a second store/remove ledger or replay engine.

## Gold-plating boundary

No provider prompt-cache hint tracing, cache policy optimizer, cross-host transfer implementation, graphical timeline, semantic memory database, raw token/KV dump, or new eviction policy. Start with one native session and one deterministic backend fixture. Provider-owned cache provenance remains #7308.

## Done condition

- [ ] One model/cachemeta adapter maps append/reuse/snapshot/clone/restore/move/rollback/evict/free domain actions, before/after lengths, and causal parents into canonical trace-ABI records; no alternate envelope or lineage graph exists.
- [ ] Identity includes model, tokenizer/serializer, positional regime, cache geometry epoch, backend, and digest-safe context interval.
- [ ] A speculative rollback states accepted and rejected position ranges and leaves the reconstructed prefix exact.
- [ ] Snapshot/clone/restore ownership transfer and final free are balanced; leaks and double-free events are detectable.
- [ ] Visibility changes reuse `CacheEventConsolidator`; duplicate/reordered source events remain deterministic.
- [ ] Disabled tracing is inert; enabled recording uses the canonical capture policy and reports drops/overhead in its canonical receipt.
- [ ] No prompt, token value, KV byte, pointer, or private identifier appears in the artifact.

## Witness

The deterministic command and expected invariants are specified in `## Verifiable Witness` below.

## Verifiable Witness

```text
go test ./internal/model ./internal/cachemeta -run 'TestNativeKVLifecycleTrace' -count=1
```

The deterministic fixture performs prefill of a short prefix, snapshot/clone, two speculative appends, rollback of the rejected suffix, restore into a fresh session, one reuse, and close. The assertion replays only the emitted events and proves exact token lengths, ownership balance, final visibility, and surviving prefix identity after each stage.

## Prior-art source, date, license, and disposition

- Observed `2026-09-08T16:09:42Z`; source event `2026-09-04T02:22:50Z`; shipped source: vLLM KV event contract in `vllm/distributed/kv_events.py@d9e2b5238d4a42341db3374226890a801357c646` ([commit](https://github.com/vllm-project/vllm/commit/d9e2b5238d4a42341db3374226890a801357c646)).
- License: [Apache-2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE). Disposition: **ADAPT / DEFAULT** — preserve explicit block lifecycle identity and session correlation while implementing FAK-native events and ownership semantics. Platform context: vLLM V1 distributed KV events on main; FAK already pins an older compatible baseline in closed #8447. Refresh trigger: vLLM KV event schema/session identity change.
- Transferable principle: cache truth is an ordered lifecycle of identified blocks, not an aggregate hit counter. FAK opportunity: join that lifecycle to native semantic context, rollback, and snapshot ownership. Disconfirming check: if existing native events can replay the full fixture without reading session memory, close as duplicate.

## Dedupe

Searched public open/closed KV event, cache lifecycle, context trace, and prefix lifecycle issues. Closed #8447 pins external vLLM event provenance; closed #5260 adds tier/priority block diffs; closed #9307 consolidates multiple producers; none instruments FAK’s native `Session` append/snapshot/rollback/restore path. Open #7308 is provider-owned prompt-cache hint provenance and remains separate. Open #8586 consumes broader harness context lifecycle events; this issue supplies native inference materialization evidence. #10184 is runtime scheduler flight recording, not KV lifecycle.

## Definition of done

The resolving commit wires real native lifecycle emission, replays the deterministic session through the TICKET-05 reducer contract solely from canonical records, proves ownership balance, and passes privacy/bound tests. A new event struct, duplicate replay engine, or schema with no live emit sites is incomplete.

## Working spine

Semantic context span → per-layer native KV append → prefix snapshot/clone → speculative suffix → rollback → restore/reuse → free, normalized through cachemeta identity.

## Acceptance gate

The focused witness and `go vet ./internal/model ./internal/cachemeta` pass; `fak validate --mine` is green; artifact scrub assertions prove payload absence; event replay reaches the same final prefix length/identity as the live session.

## Likely files

- `internal/model/hal.go`
- `internal/model/kv.go`
- A narrowly named `internal/model/kv_trace.go`
- `internal/cachemeta/cache_event_consolidator.go` or a lifecycle companion
- Focused `_test.go` files in the same packages
- `docs/tickets/tracing-dataflow/TICKET-09.md`

## Lane

`cachemeta`; two Go packages: `internal/model` and `internal/cachemeta`. Dispatch after TICKET-08; it is the final leaf in the TICKET-06 → TICKET-08 → TICKET-09 model-tree serialization.

## Closure binding

Close only from the signed resolving commit with the replayed lifecycle receipt. Do not mark provider-cache, hardware-transfer, or physical-throughput criteria complete from this software fixture.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12266
