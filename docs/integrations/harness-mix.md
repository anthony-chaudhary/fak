# Mix verified harnesses without plugin roulette

`fak harness mix` combines independently resolved `v1alpha2` product locks. It emits the same product-lock contract consumed by `harness inspect`, `harness preview`, `harness derive`, and generated product launch.

```text
fak harness mix \
  --import support.lock.json \
  --import cited-research.lock.json \
  --context-budget 12000 \
  --memory-budget-mib 1024 \
  --worker-budget 4 \
  --output support-research.lock.json

fak harness inspect --lock support-research.lock.json
go run ./cmd/microharnessdemo --selfcheck --product-lock support-research.lock.json
```

This is intentionally stricter than loading two plugins:

- every import has a verified immutable identity and the same target OS, architecture, and harness contract;
- every component carries requires, provides, conflicts, compatibility, cost, and runtime-adapter evidence;
- byte-identical components are deduplicated before context, memory, and worker costs are summed;
- transitive requirements and declared conflicts are checked over the combined component graph;
- identical effective assets collapse, policy denials may become stricter, and all other overlapping capabilities require an explicit choice before mixing;
- policy floors, mandatory status, and secret boundaries cannot be resolved by load order; and
- the mix receipt pins sorted import IDs, the resulting lock ID, deduplicated components, and the rebuild command.

Equivalent import order produces the same lock ID. A legacy `v1alpha1` lock can still launch and can be used as a derivation base, but cannot be mixed because it did not retain the evidence needed to prove cross-product compatibility. Re-resolve it from source first.

## What the guarantee covers

A successful mix proves that the represented component graph, target environment, policy/secret overlap, declared resource limits, and adapter evidence are mutually admissible. The resulting lock is structurally accepted by the shipped generated-product runtime. It does not certify undeclared external service behavior or make receipt-only asset kinds executable; those remain bounded by each adapter's conformance evidence.
