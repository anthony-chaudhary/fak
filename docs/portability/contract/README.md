---
title: "Managed-agent portability contract v1"
description: "Reference documentation for Managed-agent portability contract v1, preserving the page's implementation details, evidence, and operating context."
---

# Managed-agent portability contract v1

`fak.portability/v1` is the minimal public interchange contract identified by inventory #6595. The executable reference is `internal/portabilitycontract`; the wire schema is `v1.schema.json`. It is a contract, not a registry or universal serializer.

## Six concepts

- **Object** is one versioned typed payload. `stable_id` follows the logical object across edits; `content_id` is SHA-256 over canonical `{type, type_version, payload}`. Provenance, dependencies, scope, sensitivity, compatibility, precedence, signatures, migrations, receipts, and namespaced extensions are explicit.
- **Collection** is a versioned dependency-bearing set of Objects.
- **Context** records a machine's active collections and selected stable IDs. Machine-local selection never contributes to package identity.
- **Package** is the transport envelope. Its content-derived identity excludes local fields, signatures, receipts, machine-scoped objects, and secret objects by construction.
- **Channel** declares transport kind, endpoint, capabilities, compatibility, and extensions without changing the Package.
- **Transaction** is a previewed `apply`, `switch`, or `merge`: expected state and preview hash provide optimistic atomicity; an idempotency key suppresses repeats; a journal receipt, recovery strategy, and inverse operations make interruption recoverable and rollback explicit.

## Safety and compatibility

Readers reject unknown schema majors and unknown **critical** extensions. Unknown object `type` values are never activated: their payload and non-critical extensions round-trip inertly. A writer must not infer executable behavior from an unregistered type. Known v1 types are intentionally the inventory vocabulary: skill, policy, session, loop, model-binding, instruction, hook, MCP server, plugin, account, and secret reference.

Scope resolution is fixed and deterministic:

```
machine > user > project > team > corporate > public
```

Within a scope, larger explicit `precedence` wins; `stable_id` ascending is the final tie-break. Dependencies are stable IDs, optionally pinned by content ID. A translation is exact only when its degradation list is empty. Every loss names the object, unsupported feature, severity, meaning lost, and safe fallback.

Migration declares `from`, `to`, stable migration ID, and reversibility. A non-reversible migration is rollbackable only while a before-state receipt remains available. Signatures authenticate an already-derived identity and are excluded from that identity, avoiding self-reference.

## Fixtures and executable witness

- `representative.golden.json`: mixed policy/skill, inert future type, and excluded machine secret.
- `hostile.unknown-critical.json`: hostile activation request behind an unknown critical extension; rejected.
- `invalid.schema.json`: unsupported schema; rejected.
- `compatibility.golden.json`: normative reader/writer, identity, precedence, transaction, translation, and migration rules.

Run:

```text
go test ./internal/portabilitycontract
go run ./cmd/portabilitycontract --check internal/portabilitycontract/testdata/representative.golden.json
go run ./cmd/portabilitycontract --explain internal/portabilitycontract/testdata/representative.golden.json
```

The last command emits the captured `representative.explain.golden.txt`, intended to be understandable without Go source.
