# Safe public registry lifecycle

`fak profile continuity registry-selfcheck` captures the local-first lifecycle for #6603. The implementation consumes the `fak.portability/v1` package from #6598/#6599, runs every public payload through the #6600 egress gate, and leaves #6601 sync and #6602 organization governance as the metadata-distribution and authorization planes.

## Trust boundary

`internal/portability.Registry` has only immutable `Put` and exact-version `Get`. `LocalRegistry` is the first transport; a hosted repository, OCI/artifact adapter, or federated registry implements the same interface. Transport retrieval never implies trust. `Inspect` locally verifies signed identity, immutable package digest, provenance, license, sensitivity, compatibility, dependency pins, permissions/hooks, signer status, freshness, and anti-replay sequence before install.

Publish is a dry-run unless the caller explicitly commits. The dry-run invokes the public egress gate before package identity or registry writes. Committed publish requires complete metadata and an Ed25519 signature, and an existing version cannot be replaced.

## Consumer journey

1. **Inspect** is non-executing and displays provenance, dependency pins, sensitivity, compatibility, signature result, permissions, hooks, deprecation, yank, and revocation state.
2. **Resolve** walks exact namespace/name/version/digest dependencies and emits a deterministically sorted `fak.lock/v1` permission preview. Digest mismatch is dependency confusion, not an alternate candidate.
3. **Install** records the verified reference and digest inactive. Activation is a separate explicit transaction.
4. **Update** requires a strictly newer sequence and presents semantic permission changes, object add/remove diff, breaking changes, migration, and rollback before approval. Pins remain represented by the old lockfile for rollback.
5. **Deprecation** remains visible but installable. Yank/revocation deny new installs. A synced revocation or compromise fast path deactivates and quarantines existing installs while retaining local evidence.

## Fail-closed cases

Malformed/unknown metadata, path-like identities, signed-identity mismatch, tamper, stale/expired metadata, retired or unknown signers, replay/downgrade, namespace takeover, immutable-version overwrite, yanked/revoked content, and dependency digest mismatch are typed denials. Explanations name fields and reason classes, never excluded payload values; egress denial names only the managed object ID.

Run the captured live journey:

```text
fak profile continuity registry-selfcheck
fak profile continuity registry-selfcheck --json
```
