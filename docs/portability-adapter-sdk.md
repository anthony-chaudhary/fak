# Portability adapter SDK

The `internal/portability` adapter seam keeps object-specific meaning outside the
stable package/context contract. An adapter declares a deterministic identity,
version, sensitivity, compatibility, capabilities, support status, and any
semantic degradation. It receives only a caller-provided `State`; no filesystem
root, environment, network, or credential reader is available to adapter code.

## Contract

`Adapter` provides registration/discovery, read, validation, deterministic
preview/apply/rollback, migration, structural diff, dependency enumeration,
identity, and canonical export. Apply returns a receipt and rolls back completed
changes when a write is interrupted. Repeating preview after apply produces an
empty plan. Errors use `portability.Error` (`code`, `kind`, `operation`, `field`,
`message`) rather than provider text.

Unknown kinds resolve to an inactive opaque adapter. Their valid JSON and future
fields round-trip canonically with `active=false`; activation and state-changing
operations fail with `UNSUPPORTED`. Credential-like values fail closed with
`SENSITIVE_DATA` and are never emitted.

## Reference support

Run `go run ./cmd/portability-adapter-selfcheck (matrix is included)` for the machine-readable matrix. Shipped
references are:

- full: `skill`, `policy`, `receipt`;
- partial with explicit degradation: `workflow`, `profile`, `tool-binding`,
  `model-binding`;
- local-only: `session`, `checkpoint`.

A managed-kind registry must call `RequireCoverage` with every registered kind;
a newly added kind then fails until it has an adapter/status declaration.

## Add an adapter

```text
go run ./cmd/portability-adapter-selfcheck --skeleton my-kind
```

Use the emitted registration helper as the starting point, declare all metadata,
then call `portability.RunConformance` from the adapter test. Do not grant an
adapter ambient roots or secrets. Preserve unrecognized JSON in `Record.Unknown`.

## Witness

```text
go run ./cmd/portability-adapter-selfcheck
```

The live selfcheck runs the same golden lifecycle through every reference adapter:
round-trip and future-field preservation, deterministic identity/export, hostile
input and secret rejection, dependency ordering, schema evolution,
apply/rollback/idempotence, interruption recovery, precedence conflict, and
partial-translation status. Captured output is in
`docs/_witnesses/portability-adapter-selfcheck-2026-08-13.json`.

