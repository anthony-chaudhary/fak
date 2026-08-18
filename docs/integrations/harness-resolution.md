# Deterministic harness product resolution

`fak harness resolve` closes the pre-launch chain:

```text
fak harness select ... > selection.json
fak harness resolve \
  --manifest product.json --selection selection.json \
  --os linux --arch amd64 --contract v1
```

The `fak.harness-product/v1alpha1` manifest combines selected typed assets with versioned components, capability providers, dependency ranges, conflicts, compatibility, and aggregate resource budgets. Successful resolution emits an immutable `fak.harness-product-lock/v1alpha1` receipt.

## Resolution order

1. Compose selected company/team/person/repo/project/domain/task assets through the kind-specific merge contract.
2. Check product compatibility against OS, architecture, and harness contract.
3. Bind every dependency capability to exactly one provider satisfying its version range.
4. Reject missing and ambiguous providers and dependency cycles.
5. Run the generic `internal/stackresolve` closure/conflict engine over the concrete bindings.
6. Check every selected component's compatibility and aggregate context-token, memory-MiB, and worker costs against the manifest ceiling.
7. Canonically sort selected components/assets/explanations and hash the lock body.

The lock pins component ID, semantic version, SHA-256 digest, source provenance, provider, selection reason, provided capabilities, effective typed assets, asset-merge trace, compatibility environment, actual resource consumption, and dependency decisions. Its ID is SHA-256 over canonical JSON with the ID field blank, so changing a digest, source, selected asset, environment, or decision changes the lock.

## Explanation and failure behavior

`explain` entries answer why each root/provider/dependency capability exists and include the requested version range. Errors name missing requirements, ambiguous providers, incompatible ranges, cycles, component conflicts, compatibility mismatches, budget overrun, or typed-asset privilege violations. No lock is emitted on refusal.

The property witness permutes component catalog order and asset-layer declaration order while retaining selected precedence; the complete result and lock digest remain identical. Adversarial witnesses cover missing dependencies, incompatible ranges, ambiguous providers, cycles, conflicts, OS/architecture/contract incompatibility, resource ceilings, and policy widening/locked-floor changes.

## Boundaries

Version constraints currently support one comparator (`==`, `>=`, `<=`, `>`, or `<`) against strict `MAJOR.MINOR.PATCH`; compound semver ranges and prereleases are not accepted. Provider ambiguity fails rather than ranking candidates. The resolver emits an inert lock and does not load adapters or project host files. External launch wiring remains #6901, conformance certification remains #6793, and richer package/skill distribution remains #6796.

## Downstream composition evidence

New resolutions emit `fak.harness-product-lock/v1alpha2`. In addition to the launch fields retained by `v1alpha1`, each selected component now keeps its typed requirements, conflicts, target compatibility, resource cost, and runtime-adapter conformance declarations. These fields participate in the canonical lock ID and are deterministically ordered.

`v1alpha1` locks remain valid launch and derivation inputs. They are deliberately **not mixable**: the old schema discarded facts needed to prove that two independently resolved products are compatible. Rebuild the old lock from its source manifest with the current `fak harness resolve`; fak never invents missing compatibility or adapter evidence.

A `v1alpha2` lock is mix-ready only when every selected component declares a compatibility contract and at least one represented runtime adapter. `harness resolve` can still emit a launchable lock without these declarations, but downstream mix admission refuses it with the exact component and missing evidence. This separates “works as the product that was resolved” from the stronger “contains enough evidence to combine with an independently resolved product.”
