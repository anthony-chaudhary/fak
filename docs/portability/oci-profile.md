# fak OCI collection profile v1

Status: minimal working profile for issue #6627. The profile carries a collection through OCI Distribution 1.1 while keeping import and pull **inert**. Activation remains a separate, digest-pinned decision.

## Value frame

- **For:** operators moving mixed fak objects between OCI-compatible stores.
- **Problem:** native objects need content-addressed transport without replacing their schemas or turning a registry into activation authority.
- **Today:** ad-hoc bundles lose descriptor identity, unknown object bytes, and detached supply-chain metadata.
- **Better because:** OCI manifests, descriptors, subjects, and referrers carry native bytes with fail-closed validation.
- **Witness:** `TestGoldenLayoutRegistryProxyAndReferrers` exercises an OCI layout and an in-process conformant registry by digest.

## Profile table

| Role | Media type | Required content |
|---|---|---|
| OCI manifest | `application/vnd.oci.image.manifest.v1+json` | OCI schema version 2; `artifactType` is the config media type |
| Collection config v1 | `application/vnd.fak.collection.config.v1+json` | `fak.oci.collection/v1`, collection name/version, ordered typed object records and dependencies |
| Skill layer | `application/vnd.fak.skill.v1+json` | Native skill payload, unchanged |
| Workflow layer | `application/vnd.fak.workflow.v1+json` | Native workflow payload, unchanged |
| Policy layer | `application/vnd.fak.policy.v1+json` | Native policy payload, unchanged |
| MCP service layer | `application/vnd.fak.mcp.server.v1+json` | Native MCP `server.json` metadata only for an object whose kind is `mcp-service` |
| Unknown native layer | its native `type/subtype` | Preserved byte-for-byte and descriptor-for-descriptor; never executed |
| Signature | `application/vnd.dev.cosign.simplesigning.v1+json` | Detached subject/referrer of the collection manifest |
| Attestation | `application/vnd.in-toto+json` | Detached subject/referrer |
| SBOM | `application/vnd.cyclonedx+json` | Detached subject/referrer |
| Replacement/revocation | `application/vnd.fak.collection.statement.v1+json` | Detached statement naming the superseded or revoked manifest as `subject` |

Every layer descriptor carries `org.opencontainers.image.title` (clean relative collection path) and `io.fak.object.kind`. Config and descriptor media types must agree. Digest (`sha256`) and size are checked before config interpretation. Missing blobs, unknown collection/config media types, path escapes, unresolved dependencies, and dependency cycles fail closed with typed errors.

## Transport and activation boundary

1. A mutable tag may be used only for discovery. `HEAD /v2/<name>/manifests/<tag>` resolves it to `Docker-Content-Digest`.
2. Pull, inspect, plan, receipts, and later activation name that digest. `Pull` rejects tags; there is no tag-only activation path.
3. Pull/import validates all descriptors and dependencies and returns an activation-neutral receipt: `manifest_digest`, ordered `layer_digests`, and `activated:false`.
4. Referrers use OCI Distribution 1.1 `GET /v2/<name>/referrers/<digest>`. Registries without that API use the OCI artifact convention fallback tag `sha256-<hex>` containing an OCI index. Both paths return descriptors only and execute nothing.
5. Registries are byte stores, not policy or activation authorities. Signature, attestation, SBOM, replacement, and revocation interpretation belongs to a later verifier/planner.

## MCP `server.json` bridge

The bridge is deliberately narrow. It accepts/exports `server.json` only for `mcp-service` objects and preserves native `$schema`, `name`, `version`, `packages`, `remotes`, `repository`, and `status`. The server document remains a typed layer payload. It is never the collection manifest or config and cannot describe non-MCP objects.

## Golden fixture and tests

`fixtures/mixed-layout/` is a checked-in, hand-readable OCI image layout with a skill, MCP service metadata, and an unknown binary layer. `collection.source.json` is the authoring view; `index.json`, the manifest/config blobs, and payload blobs are the transport witness. Tests push/pull via `httptest`, resolve a mutable tag before digest pull, proxy/re-export unchanged bytes/annotations/descriptors, discover a detached referrer by standard and fallback paths, and assert zero activation calls.

Run:

```text
go test ./internal/ociartifact
```

This is intentionally not a universal package manager: it does not install, merge, resolve remote dependency ranges, execute hooks, or activate content.
