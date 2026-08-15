# Hardware and quantization support-graph spine

**Date:** 2026-08-15 · **Issue:** [#6895](https://github.com/anthony-chaudhary/fak/issues/6895)

`internal/supportgraph` replaces an ambiguous “AWQ supported” boolean with an exact normalized tuple: artifact hash, model architecture, quant scheme, layout, backend, kernel, runtime, and hardware baseline. Each edge carries required and recommended baseline facts plus provenance, support state, proof tier, expiry, fallback, and penalty.

The representative fixture returns:

```text
L4 exact tuple: supported — current witnessed device-decode evidence
T4 exact tuple: unsupported — witnessed refusal overrides a vendor support claim
  fallback=awq-portable-cpu penalty=latency unmeasured
old tuple: stale — all matching evidence expired
unknown layout: unknown — no exact support edge; not evaluated
SELFCHECK PASS
```

Authority policy is intentionally small: the highest current proof tier decides; equal-tier contradictory states return `conflict`; expired evidence yields `stale`; absence yields `unknown`, never unsupported. This fixture is synthetic schema evidence, not a claim that the named artifact actually ran in the lab. Real ingestion remains #6896.

`ArtifactRuntimeRequest.Tuple` is the narrow #6224 bridge: the artifact-runtime adjudicator remains owner of its decision and supplies normalized identities; supportgraph does not import or duplicate peer-owned `quantmeta`.

Reproduce:

```bash
go test -count=1 ./internal/supportgraph ./cmd/supportgraphdemo
go vet ./internal/supportgraph ./cmd/supportgraphdemo
go run ./cmd/supportgraphdemo -selfcheck
```
