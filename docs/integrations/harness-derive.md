# Derive a harness from one verified import plus a typed delta

`fak harness derive` is the short path from “this support harness is useful” to “this is my product.” It imports an immutable resolved lock, verifies its identity and launch shape, applies only a bounded typed delta, and emits another ordinary product lock accepted by generated `fak harness init` products.

```text
fak harness derive \
  --from support.lock.json \
  --set instruction:response-style=detailed \
  --layer my-support \
  --output my-support.lock.json

fak harness preview --current support.lock.json --candidate my-support.lock.json
fak harness inspect --lock my-support.lock.json
go run ./cmd/product --selfcheck --product-lock my-support.lock.json
```

The base manifest and layer-selection files are not required. The base lock is the imported product; the `--set` expression is the local source code of the variant.

## What “verified” means

Derivation is not an arbitrary patch operation. Before writing output, fak:

- verifies the base lock's canonical SHA-256 identity;
- requires the capability to exist in that exact base;
- refuses locked and mandatory capabilities;
- permits replacement only where the shipped launch adapter executes the represented capability (currently instruction values);
- sorts deltas before application so equivalent input order produces the same lock ID;
- retains original provenance in `derive:<local-layer> (from <upstream-source>)` and in the asset trace; and
- recomputes and verifies the standard product-lock identity after the delta.

The command writes `<output>.derive.json` by default as a lineage receipt. It binds the exact base ID, result ID, target environment, normalized delta, verifier contract, and rebuild command. Use `--expect-base sha256:...` in automation to reject a source path that has moved to a different upstream revision. An upstream update therefore becomes a deliberate derive → preview → admit decision, never a silent float.

Policy widening, free-form append operations, executable plugin injection, and edits to asset kinds without a conformant launch adapter are refused. A successful derivation means the resulting lock is structurally launchable through the represented adapters; it does not turn receipt-only asset kinds into executable behavior.

## Why this is more than plugins

A plugin mechanism answers whether code can be loaded. Derivation answers the operational questions needed for a default-useful product:

- exactly which product and revision was imported;
- which effective behavior changed and which behavior stayed inherited;
- whether company locks and mandatory workflows survived;
- whether the same environment, component graph, and resource budget remain admitted;
- whether the result is reproducible and launchable through shipped adapters; and
- how to inspect, preview, rebuild, update, and roll back it.

Multiple whole-harness imports and component conflict resolution are tracked separately in #7229. Immutable publication, update channels, signatures, cache, and rollback are tracked in #7230.
