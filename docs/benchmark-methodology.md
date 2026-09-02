---
title: "Benchmark route: current authority and archived evidence"
description: "Which fak benchmark page is current authority, which pages are historical evidence, and the reproduction command to run before quoting any result."
---

# Benchmark route: current authority and archived evidence

**Audience:** evaluators deciding which benchmark page is current authority, which pages are historical evidence, and what to reproduce before quoting a result.

**Lifecycle:** current

**Generation:** gen/now

**Owner:** documentation

**Authority:** [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) owns current result selection, tuned baselines, honesty fences, and reproduction commands. This page classifies the surrounding benchmark collections; it does not promote their numbers.

**Next action:** start with the current authority, match one result to your workload and hardware, and run its named reproduction command. Open an archived collection only when the authority links to it or when you are investigating that recorded generation.

## Choose the route

| Route | Classification | Use it for | Do not infer |
|---|---|---|---|
| [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) | **Current authority · lifecycle: current · generation: gen/now** | Selecting a current scoped result, its tuned alternative, honesty fences, artifact, and reproduce command. | That one scoped result applies to another model, device, concurrency shape, or run mode. |
| [`docs/benchmarks/`](benchmarks/README.md) | **Evidence archive · mixed recorded generations** | Finding individual result sheets, runbooks, contracts, and historical witnesses by topic. A sheet is current only when the authority explicitly selects it. | Directory recency, filename, or a large number makes a page current authority. |
| [`docs/benchmarks/BENCH-INFRA-INDEX.md`](benchmarks/BENCH-INFRA-INDEX.md) | **Infrastructure reference · current workflow, mixed run generations** | Operating the benchmark catalog, machine registry, schemas, run identifiers, and data layout. | Infrastructure documentation validates or promotes a benchmark result. |
| [`docs/benchmarks/LEGACY-BENCH-INDEX.md`](benchmarks/LEGACY-BENCH-INDEX.md) | **Legacy collection index · archived** | Tracing older benchmark narratives and reproductions that still matter to their recorded generation. | Its “primary” labels override the current authority or describe current product support. |
| [`docs/production-benchmark-methodology.md`](production-benchmark-methodology.md) | **Methodology reference · current when selected by the authority** | Understanding controlled production comparison requirements and evidence quality. | A methodology document by itself proves a performance gain. |

## Classification rules

1. **Current authority is singular.** Use `BENCHMARK-AUTHORITY.md` to select a result; neighboring indexes cannot supersede it.
2. **Generation belongs to the evidence.** Preserve the date, commit, artifact, model, hardware, mode, and workload recorded by a result sheet. `gen/now` on this route does not relabel old runs as current.
3. **Lifecycle and support are different.** `current`, `archived`, or `mixed` describes documentation authority. It does not claim a model, cloud, or backend is supported.
4. **A tuned alternative is mandatory.** Quote the tuned baseline as the headline. A naive stateless arm may provide context but is not the decision-grade alternative.
5. **Reproduction gates promotion.** If the selected authority row lacks an available artifact or reproducible command for the claimed scope, report `not yet`.

## Preflight a task environment before launch

A benchmark task that declares a typed environment contract must be matched to an
observed compute receipt before model or provider spend:

```text
fak benchmarks preflight --requirement task-environment.json --receipt compute-receipt.json
```

The contract covers OS, architecture, immutable image identity, minimum vCPU,
RAM, disk, GPU class/count, network posture, post-boot software identity,
license posture, and input-data identity. The receipt carries the same observed
axes plus provider, node, source, and probe time. Blank or mutable identities are
unknown and fail closed; numeric shortfalls are `insufficient`; absent identities
are `missing`; and a provider capability that violates a task prohibition is
`forbidden`.

An accepted result prints content hashes for both the requirement and receipt.
Persist those hashes in the benchmark result packet so the score stays bound to
the admitted environment. A caller may attach remediation from the existing
fleet registry to a refusal; the matcher does not create or provision a second
node registry.

Existing `internal/benchcatalog` rows remain compatible: `Need=offline|weights|dataset`
continues to drive list, describe, and legacy run behavior. That coarse label is
not converted into environment proof. `fak benchmarks preflight <legacy-name>
--receipt ...` therefore returns `BENCH_REQUIREMENT_UNKNOWN` until the row gains
a typed task contract; operators can preflight a standalone contract with
`--requirement` during that migration.

## Evaluator completion check

Before quoting a benchmark, record all six fields:

- current authority row or anchor;
- measured outcome and units;
- tuned alternative;
- model, hardware, mode, workload, and warm-state scope;
- artifact and exact reproduction command;
- evidence status (`witnessed`, `observed`, `modeled`, or `not yet`).

Then grade the proposed statement with `fak claim-check` under the [net-true-value standard](standards/net-true-value.md). A historical page remains valid evidence for its stated generation, but it becomes a current headline only through the authority route.
