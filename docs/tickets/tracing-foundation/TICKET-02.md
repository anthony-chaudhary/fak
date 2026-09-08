# feat(trajectory): preserve many-to-many value lineage across split, fuse, repack, expand, and contract

```routing
lane: trajectory
paths:
  - pkg/traceabi/**
expected_steps: 8
```

<!-- fak-trace-foundation-key: many-to-many-value-lineage-v1 -->

## Parent context

#6053. Companion dependent on the typed causal envelope ticket `typed-causal-envelope-v1`. It supplies the value-flow graph needed by tensor, quantization, context, and cache tracers. It is deliberately separate from #10269's evidence-independence quorum and #6876's claim-to-artifact liveness graph.

## Why this is next

The existing quantization tracer proves that representation flow is useful, but further tensor/context tracers need one identity-preserving map before each encodes a different lineage dialect.

## Problem

For an agent diagnosing a wrong value or unexpected byte count, event parentage says which operation followed which, but not which logical values became which outputs. Tensor layouts split and fuse ranges; quantization repacks codes and expands values; contractions combine many weights and activations into one output. Today those transformations are either aggregate strings or one-off sampled fields. Better because a bounded many-to-many lineage record lets the agent walk from any output digest/range back to every contributing logical source without treating array offsets or addresses as identity. Witness: one fixture executes split -> repack -> fuse -> expand -> contract and reconstructs exact predecessors and successors after shuffled input order.

## Centrality and P1-P4

**Centrality: Enabling.** This is the reusable value-identity substrate for the first-class tracers, not an end-user tracer by itself.

- P1 Context: advanced - an agent retrieves the minimal upstream/downstream lineage slice for one value instead of loading a full tensor or trace.
- P2 Net value: preserved - the map exposes real representation bytes and fan-in/fan-out; no bandwidth or speed claim follows from lineage alone.
- P3 Adaptation: advanced - all nodes and edges are content-addressed, range-bounded, typed, and cycle-checked; missing lineage stays unknown.
- P4 Operations: advanced - one graph vocabulary spans unpack/repack/expand/contract and later context/cache transformations.

## Current state

At `f21eb416d92f03286747f84e888fb7dd1038c7a9`:

- `internal/trajectory/event.go:72-85 (Event)` supports `ParentIDs []string`, which is many-parent event causality but has no value node, representation, byte/element range, or transformation edge.
- `internal/model/materialization_trace.go:5-13 (MaterializationStep)` records aggregate `Stage`, `From`, `To`, and `Bytes`; it cannot identify the specific source/output values or express a many-input contraction.
- `internal/model/token_lineage.go:9-12,25-45 (tokenLineage)` stores one exact token ID per resident KV position, a valuable one-to-one invariant that does not represent split/fuse/repack derivation.
- `internal/bench/quanttrace.go:16-32 (QuantTraceSample)` maps selected Q4_K logical indexes to source/repacked byte offsets and expanded values, while `:73-145 (TraceQ4K)` is intentionally one format and at most 32 samples.

The capability is **PARTIAL**: domain-local mappings exist, but there is no reusable many-to-many value lineage contract.

## Core through-line

Add a bounded, canonical `ValueLineage` graph to public `pkg/traceabi`:

- `ValueRef`: logical ID, version, representation, digest, unit (`byte`, `element`, `token`, `page`), half-open range, optional shape/dtype/layout labels;
- `Transform`: transform ID, causal span ID, closed operation (`copy`, `split`, `fuse`, `pack`, `unpack`, `repack`, `quantize`, `dequantize`, `expand`, `contract`, `transpose`, `evict`, `restore`), zero-or-more inputs, one-or-more outputs, exactness (`bit_exact`, `value_exact`, `lossy`, `unknown`), and evidence reference;
- deterministic predecessor/successor queries by logical ID plus range;
- explicit omission/loss rows when a recorder samples or drops mappings.

The first slice is data and algorithms only. A compact fixture should demonstrate transformations equivalent in shape to Q4_K unpack/repack/expand/contract, but it must not move model code or claim production instrumentation.

## Working spine

1. Define versioned value, range, transform, and omission records.
2. Validate nonempty logical IDs, digests, units, valid half-open ranges, and known operation/exactness values.
3. Reject duplicate node versions, dangling references, self-derivation, and cycles.
4. Canonicalize node/transform/input/output ordering so source order cannot change the graph digest.
5. Implement bounded upstream/downstream traversal with max depth and max result count.
6. Preserve many-to-many fan-in and fan-out rather than flattening to a single parent.
7. Emit explicit `truncated`/`unknown` loss evidence when bounds cut the graph.
8. Prove a split/repack/fuse/expand/contract fixture and one lossy/unknown edge.

## Gold-plating boundary

Do not instrument every kernel, store raw tensors, infer lineage from matching values, build a graph database, or generalize into claim/evidence quorum. Do not modify `internal/model` or `internal/bench` in this issue. Keep the exported package data-only, stdlib-only, and free of private semantics. One in-memory/JSON graph with bounded traversal is the tracer bullet.

## Dedupe

Search across open and closed public issues on 2026-09-08 found:

- #10269: transitive `derived_from` roots prevent circular or same-root evidence from satisfying quorum; it owns evidence independence, not runtime value ranges.
- #6876: `supports`/`supersedes`/`derived_from` edges determine whether proof artifacts remain reachable from claims; it owns artifact liveness.
- #6528: reconstructs one derived context view; it can later project its operators into this graph but remains context-specific.
- #8609 (closed): bounded device-event traces for kernel replay; its event schema has operation timing, not value derivation.
- #9071 (closed): token-indexed decode traces; token indexing is a consumer, not the shared transformation map.

No existing issue defines many-to-many logical value/range lineage across representation transformations. The dedupe key is intentionally `value-lineage`, not generic `derived-lineage`.

## Prior-art ledger

- **Source:** OpenLineage `spec/facets/ColumnLineageDatasetFacet.json:13-49,54-98@9ac19298c8ef518ed10c7a3af547c8ac86c2e012`.
- **Source event:** commit `9ac19298c8ef518ed10c7a3af547c8ac86c2e012`, 2026-09-08T12:36:21Z; source state: shipped schema on `main`.
- **Observed at:** 2026-09-08T16:10:19Z.
- **License:** Apache-2.0.
- **Disposition:** **ADAPT / DEFAULT**. Reuse the principle that one output field lists all contributing input fields plus transformation descriptors; adapt identifiers to tensor/context values and half-open ranges. No upstream code or prose copied.
- **Refresh trigger:** OpenLineage column-lineage facet revision or a fak consumer proving the closed operation vocabulary insufficient.
- **Spirit extension:** field lineage's many-input output mapping generalizes cleanly to fused kernels and contractions when identity is logical/content-addressed rather than tied to storage addresses.

## Done condition

The public trace ABI can encode, validate, digest, and boundedly traverse a deterministic many-to-many value lineage graph without raw payloads.

## Definition of done

- [ ] `ValueRef`, `ValueRange`, `Transform`, `LineageLoss`, and schema descriptors are versioned.
- [ ] Every transform supports multiple inputs and outputs and names exactness.
- [ ] Validation rejects bad ranges, duplicate versions, dangling refs, cycles, and self-derivation.
- [ ] Canonical graph bytes and digest are invariant to input slice/map order.
- [ ] Upstream and downstream traversal accept explicit depth/result bounds.
- [ ] Bound exhaustion returns partial results plus typed truncation evidence.
- [ ] A golden split/repack/fuse/expand/contract fixture retains all contributing sources.
- [ ] A lossy transform never reports bit- or value-exact lineage.
- [ ] No raw tensor/value payload is required or retained.

## Verifiable Witness

```bash
go test ./pkg/traceabi -run 'TestValueLineage|TestValueLineageManyToMany|TestValueLineageBoundedTraversal|TestExternalLineageConsumer' -count=1
```

The golden graph is built twice from differently ordered rows and must produce identical bytes/digest and the same predecessor/successor sets.

## Witness

Run the deterministic `go test ./pkg/traceabi -run 'TestValueLineage|TestValueLineageManyToMany|TestValueLineageBoundedTraversal|TestExternalLineageConsumer' -count=1` gate above.

## Acceptance gate

Given output `y[0:8]` from a contraction whose weights passed through split, repack, fuse, and expand, the query returns every contributing source range and activation range, names each transformation/exactness class, and explicitly marks any sampled-away mapping; no address or slice index is treated as durable identity.

## Likely files

- `pkg/traceabi/lineage.go`
- `pkg/traceabi/lineage_test.go`
- `pkg/traceabi/testdata/value-lineage-transform-chain.json`

## Lane

trajectory

Schedule after `typed-causal-envelope-v1` to avoid same-lane overlap.

## Expected steps

8

## Closure binding

The resolving commit cites this issue and uses `(fak traceabi)`. Report `pkg/traceabi@rev`, the canonical fixture digest, and the bounded traversal witness.

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12259
