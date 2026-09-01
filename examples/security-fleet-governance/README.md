# Security fleet governance demo

From the repository root:

```text
go run ./examples/security-fleet-governance
```

The demo uses temporary files and the local fak source only. It needs no API key, model, network, running gateway, or durable machine mutation. It verifies the central floor, derived attestation, canonical `corp://` references, an ArgRules-only team narrowing, local read-back, and rollback.

See [`../../docs/security-fleet-governance.md`](../../docs/security-fleet-governance.md) for the control boundary and enterprise rollout composition.
