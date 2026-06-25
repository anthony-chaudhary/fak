# deletioncert - provable-deletion certificate

`deletioncert` is a no-model, deterministic proof that a selected KV span can be
evicted so the surviving context is byte-identical to a run that never saw it, then
bound into a tamper-evident deletion certificate.

## Prerequisites

Requires Go only from the repository root. It runs offline over a tiny synthetic model
and a temporary in-process journal; no model weights, GPU, API key, network service, or
durable output file is required. The run completes in a few seconds and returns exit
code 0 only if the eviction, certificate verification, and tamper-rejection checks pass.

## Run it

```bash
go run ./cmd/deletioncert -selfcheck
go run ./cmd/deletioncert -selfcheck -out deletioncert.json
```

The optional `-out` path writes the minted certificate JSON. The default run creates
only a temporary journal and removes it before exit.

## What you see

```text
== fak provable-deletion demo ==
never-saw  continuation = [11 11 11 11 11 11]
kept-secret continuation = [11 11 11 11 11 109]
evicted    continuation = [11 11 11 11 11 11]
PROVEN: evicted == never-saw (max|Delta|=0); kept-secret differs.
VERIFY (intact)        -> valid=true sig=true anchor=true bound=true equiv=true
VERIFY (cert forged)   -> valid=false
VERIFY (scope forged)  -> valid=false
VERIFY (journal rewrit)-> valid=false
OK - provable-deletion certificate minted, verified, and tamper-rejected.
```

`go test ./cmd/deletioncert` now runs the selfcheck path as a regression guard in
addition to the pure helper tests.

## Scope

This demo does not claim third-party timestamping, independent issuance, or a production
deletion service. The certificate is self-signed v1; it proves integrity of the recorded
facts and fail-closed verification over the synthetic-model witness. For the full honesty
fences, see [`../../CLAIMS.md`](../../CLAIMS.md) and [`../../docs/proofs/deletioncert.md`](../../docs/proofs/deletioncert.md).
