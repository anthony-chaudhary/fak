# Security fleet governance demo

From the repository root:

```text
go run ./examples/security-fleet-governance
```

The demo uses temporary files and the local fak source only. It needs no API key, model, network, running gateway, or durable machine mutation. It verifies the central floor, derived attestation, canonical `corp://` references, an ArgRules-only team narrowing, local read-back, and rollback.

Every step is a local `go run` of the kernel CLI (`policy --check`, `attest`, `preflight`, `land-rule`); the first run pays a one-time kernel compile, after which the sequence completes in a few seconds. It exits 0 only when every check holds. The preflight verdicts (`ALLOW`/`DENY`) are pure functions of the policy file and tool call, so the run is deterministic and safe to re-run: the demo writes only to a temp directory that is removed on exit, and the policy mutation (`--land` / `--rollback`) operates on a throwaway copy, never on the repo's manifests.

## Scope — what this does not claim

The demo exercises the local policy control surface against two fixture manifests; what it does not claim: it does not claim production identity integration (no SSO, no real attestation authority, no enterprise IdP), it does not claim the reload endpoint is a real gateway (it is an in-process stub returning 204), and it does not claim the fixture central floor is the *correct* security policy — only that the checks behave as specified. It is not a benchmark and makes no performance claim.

See [`../../docs/security-fleet-governance.md`](../../docs/security-fleet-governance.md) for the control boundary and enterprise rollout composition.
