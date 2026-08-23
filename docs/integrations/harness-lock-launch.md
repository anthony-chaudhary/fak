# Lock-driven external harness launch

Generated `fak harness init` products accept the immutable product lock emitted by `fak harness resolve`:

```text
fak harness select ... > selection.json
fak harness resolve ... > resolution.json
# extract resolution.lock as product.lock.json
go run ./cmd/microharnessdemo --selfcheck --product-lock product.lock.json
```

Before constructing the public `pkg/harnesskit` product, the generated runtime validates the lock schema, required component and asset provenance, opaque secret references, and canonical SHA-256 lock identity. Tampered or malformed locks fail before the offline turn starts.

A successful launch projects the lock into the public harness profile, emits `harness.locked` with a `fak.harness-launch-receipt/v1alpha1` receipt, then runs the product. The receipt pins lock ID, product ID, selected profile/layers, each `kind/id@source` asset, and each `component@version#digest`. Instruction assets visibly alter the deterministic model response, making domain-specific launch behavior capturable without a provider or GPU.

The clean-room witness generates one external product, launches it with a legal lock and a coding lock, and observes different profile, instruction, asset, component, and lock receipts. Removing the two lock files and launching stock `--selfcheck` restores the generated default with no `harness.locked` event and no copied host configuration. The generator still preserves user-owned `product/config.go` and README bytes on upgrade.

`pkg/harnesskit` also exposes lock parsing/projection for manually built external products. It verifies canonical identity and refuses tampered content or secret assets without opaque references. The generated runtime intentionally vendors the small admission/projection adapter into its generator-owned file so products pinned to the existing public pseudo-version continue to build; upgrading the generator refreshes that adapter without touching user-owned files.

This spine projects only represented typed assets: profile layers and instructions affect the deterministic product, while tools, memory, policy, routes, secrets, workflows, and UI remain pinned in the launch receipt for their future adapters. It does not claim that inert assets execute merely because they are locked. Black-box adapter conformance remains #6793, and novelty/privilege preview remains #6902.
