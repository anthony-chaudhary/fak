---
title: "Native performance evidence correlation"
description: "Operational contract for resolving a native-performance dashboard point to its scrubbed evidence without exposing private infrastructure details."
---

# Native-performance evidence correlation

Status: shipped package contract for issue #9812. This is the bounded lookup seam between a
native-performance dashboard point and its scrubbed evidence. It complements the metric contract
in [`native-performance-contract.md`](native-performance-contract.md); it does not add metric
series or labels.

## Result

A producer builds one `nativeperfcorrelation.Input` after a fak-native run completes. Adding the
input to a bounded `Index` returns an opaque `npc1_...` correlation key. A dashboard may carry that
key in a Prometheus exemplar or an artifact URL. Looking up the key returns one record joining:

- request and run fingerprints;
- the source commit;
- receipt, trace, and profile fingerprints plus their scrubbed artifact locators and SHA-256
  digests;
- the explicit `fak-native` engine, backend, model, and optional quantization identity; and
- the owning `module@rev`, for example `internal/engine@r652+g1f75c56d`.

The key and all fingerprints are deterministic SHA-256-derived values. Artifact order is
canonicalized before key generation, so producer iteration order cannot change the key.

## Metric boundary

**Do not put the correlation key, fingerprints, commit, artifact locator, model identifier, or
`module@rev` in Prometheus labels.** They are high-cardinality lookup data. Existing bounded labels
such as `engine="native"`, `operation`, `backend`, and `outcome` remain governed by the native
performance metric contract.

Attach only the opaque key at the evidence boundary:

```text
metric sample
  bounded labels: engine="native", operation="decode", backend="cuda"
  exemplar/link:  correlation_key="npc1_..."
                         |
                         v
              bounded scrubbed index
                         |
       receipt + trace + profile + source identity
```

`Index.Exemplar` returns the package's complete metric attachment contract: a
`correlation_key`. The package deliberately has no helper that converts record fields to labels.

## Producer contract

The producer must supply all of the following:

1. Non-empty request, run, receipt, trace, and profile identities. They are accepted only as input
   and become `sha256:<hex>` fingerprints before storage.
2. A lowercase hexadecimal source commit.
3. `EngineIdentity{Name: "fak-native", Backend: ..., Model: ..., Quantization: ...}`. Backend,
   model, and quantization are bounded public identities, not endpoints, credentials, filesystem
   paths, or free-form command lines.
4. A `module@rev` matching `module/path@rN+g<sha>`.
5. Exactly one receipt, trace, and profile artifact. Each has a relative locator rooted under
   `artifacts/` and a lowercase SHA-256 content digest.

Use a per-stream capacity derived from the evidence retention budget. Insertion at capacity evicts
the oldest record. Re-adding identical evidence is idempotent. A key already occupied by different
evidence returns `ErrCollision`; it never overwrites either side silently.

A minimal producer flow is:

```go
index, err := nativeperfcorrelation.NewIndex(1024)
if err != nil { /* configuration error */ }
record, err := index.Add(input)
if err != nil { /* reject incomplete or unsafe evidence */ }
// Persist index.Snapshot() as JSON and attach record.Key as an exemplar or link.
```

The JSON schema identifier is `fak/native-performance-correlation/v1`. Snapshots are returned in
oldest-to-newest insertion order, making persisted indexes stable and diffable. Persist them with
the same atomic-write and retention controls as the evidence bundle.

## Lookup and artifact verification

Starting from a dashboard exemplar or link:

1. Read `correlation_key` and call `Index.Lookup`.
2. Confirm `engine.name == "fak-native"`; a different or absent engine is not native-performance
   evidence.
3. Inspect the model/backend identity and `module_at_rev` before comparing runs.
4. Resolve each relative locator against the configured scrubbed artifact root.
5. Call `VerifyArtifact` for receipt, trace, and profile. It streams through a byte limit and
   checks the indexed SHA-256 digest.

Lookup returns `ErrNotFound` after absence or retention eviction. Verification distinguishes
`ErrArtifactMissing`, `ErrArtifactTooLarge`, and `ErrDigestMismatch`. These outcomes make a broken
evidence chain explicit rather than returning a neighboring or stale artifact.

The index is in-memory by design. A storage adapter may JSON-encode `Snapshot`, but must preserve
capacity, insertion order, schema, and exact collision rejection when loading records. It must not
fall back to prefix matching, nearest timestamps, or a different engine's evidence.

## Scrubbing and private-data boundary

The public index contains no raw request, run, receipt, trace, or profile IDs. It also rejects:

- absolute paths, traversal, URI-like locators, and locators outside `artifacts/`;
- path segments containing whitespace, query strings, credentials, or host syntax;
- free-form or endpoint-shaped engine identities; and
- incomplete artifact sets.

Fingerprinting is defense in depth, not permission to feed secrets into observability. Producers
must scrub receipt, trace, and profile contents before hashing or publishing them. Private hostnames,
GPU-server details, credentials, raw prompts, customer identifiers, and private repository paths
remain outside the public bundle. The locator is a logical path inside an already-scrubbed artifact
root, never a machine path or download URL.

## Witnesses

`internal/nativeperfcorrelation/correlation_test.go` proves:

- deterministic joins, explicit fak-native/module identity, exemplar shape, and oldest-first bounded
  eviction;
- collision rejection with no silent overwrite;
- exact digest verification plus distinct missing and mismatched artifact failures; and
- serialized-record redaction plus rejection of absolute/traversal locators and private-shaped
  engine data.
