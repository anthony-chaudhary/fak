# extseamsdemo — extension trust-seam catalog

Print the production catalog that maps extension use cases to the least-privileged attachment and trust boundary:

```bash
go run ./cmd/extseamsdemo
```

The output is deterministic and offline. It helps operators choose between out-of-process agent hooks, lazy capability resolution, reviewed in-process extensions, policy data, and independently witnessed improvement proposals.

## What this demo does not claim

This demo does not claim to install, execute, sandbox, or certify an extension. It presents the checked-in trust taxonomy only. See [`../../CLAIMS.md`](../../CLAIMS.md) for the project honesty ledger.

The command creates no files or external state, so no cleanup is required.
