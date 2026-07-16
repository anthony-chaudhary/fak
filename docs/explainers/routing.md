---
title: "Model routing"
description: "The current operator route for choosing fak model-routing policy, validating its support surface, and separating advisory decisions from execution."
slug: routing
keywords:
  - fak model routing
  - model route manifest
  - model selection
  - ensemble routing
  - routing oracle
  - provider accounts
date: 2026-07-16
---

# Model routing

**For operators choosing which model plan should handle a workload:** use a reviewed
routing manifest as the policy authority. Fak classifies each request aspect, applies the
first matching rule, and returns either one model or an ensemble plan. Start by validating
the manifest you intend to deploy:

```console
fak route --check routing.json
```

That is the one next check for this page. The output lists every admitted rule and the
fallback, so an operator can review the actual support surface before using it.

## Explicit default

Use **manifest routing** for operated workloads. Commit or otherwise review
`routing.json`, validate it with `--check`, and pass the same file with
`--manifest routing.json` when asking the oracle for a decision. If no rule matches, the
manifest's named default model is selected; the decision reports
`no rule matched; fail-closed default model`. This is deterministic fallback, not an
implicit attempt to guess a provider or model.

For a local exploration, `fak route` can use its built-in starter manifest, and
`fak route --dump` can materialize that starter. Treat it as an example baseline, not as
an operator-approved production policy.

```console
fak route --manifest routing.json --aspect request --json
fak route --manifest routing.json --aspect tool_call --tool write_repository --json
```

The JSON decision names the matched rule, primary model, ensemble members and reduction
when applicable, classified subject, reason, and rough cost lens.

## Choose the routing mode

| Workload need | Choice | Support boundary |
|---|---|---|
| One reviewed model policy for request aspects | Manifest routing with a single-model fallback | Current, deterministic oracle behavior. |
| A high-risk or specialist aspect needs several opinions | An `ensemble` rule with an explicit reduction such as `vote`, `judge`, `unanimous`, or `best_of` | The oracle emits and can simulate the plan; the caller still owns actual member execution and reduction. |
| Routed model IDs must resolve to provider accounts | Supply `--accounts`, validate with `--accounts-check`, and prove coverage with `--accounts-cover` | Binding and credential-readiness inspection are current; secret values are never printed. |
| Capacity blocks one target | Supply the current block and declared alternates through the capacity flags | The oracle chooses only from explicit inputs; it does not invent health or capacity. |
| The decision must also choose harness and session lifecycle | Use `fak execution-route` | This composes harness, model, and session decisions; it does not turn them into one model string. |

`request`, `tool_call`, `query`, `state`, `step`, and `scout` are built-in aspects; a
manifest may also use a declared custom aspect. Constraints such as complexity, latency,
prompt length, labels, and tool name are match inputs, not hidden heuristics.

## Decision versus execution

`fak route` is an **offline routing oracle**. It validates policy and produces a model
plan; it does not call providers, execute ensemble members, or silently replace the model
on an existing gateway request. The process that invokes the gateway or agent must consume
the decision and bind routed IDs to supported upstream accounts.

Use [Execution routing](../execution-routing.md) when the choice also includes a harness
or session action. Use the [router integration boundary](../integrations/routers.md) when
an external router, gateway, or serving backend will execute the plan. Provider-specific
credentials, endpoint compatibility, and maturity remain governed by the relevant
[integration route](../integrations/README.md), not by a successful oracle decision alone.

## Generation and lifecycle

This page is the **current (`gen/now`) operator route** for the shipped model-policy
oracle. Runtime authority lives in [`internal/modelroute`](../../internal/modelroute/) and
the `fak route` CLI. [Model routing](../model-routing.md) is the deeper contract for
manifest fields, precedence, ensemble reductions, account binding, and benchmark evidence.

The built-in manifest is a generated starter shipped with the binary. A deployed manifest
has its own review lifecycle: validate before use, change rules and defaults explicitly,
and re-run account coverage whenever routed IDs change. Promote new modes here only after
the CLI and consumer wire are both witnessed. Demote or replace guidance when a newer
runtime authority supersedes it. Research notes and dated proposals explain design history;
they do not extend current support.

## Deeper routes

- [Model routing contract](../model-routing.md) — rule schema, matching precedence, ensembles, accounts, and evidence.
- [Execution routing](../execution-routing.md) — compose harness, model, and session choices.
- [Router integrations](../integrations/routers.md) — external execution boundary.
- [Agent routing schema](../standards/agent-routing-schema.md) — portable agent-dispatch routing, a separate schema from model policy.
