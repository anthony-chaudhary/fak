# Hosted Policy + Audit Control Plane — Architecture Brief

> **Status: architecture brief (RFC), not a built feature. Adoption-gated** — sequenced
> *after* the MCP wedge and the context-debugger product ship (see
> [`docs/dispatch-status.md`](../dispatch-status.md)). This document satisfies the
> epic-level acceptance criterion "a written architecture brief for the hosted plane" of
> [#576](https://github.com/anthony-chaudhary/fak/issues/576). It describes a *destination*,
> not the next thing to build. Nothing here is a shipped capability; every "what we already
> have" claim is verified against the binary below.

## 1. The thesis (open-core)

`fak` is an *agent kernel*: one Go binary that adjudicates every tool call before it runs —
deny by structure, repair malformed calls, quarantine poisoned results. The eventual revenue
model is **open-core**: the binary stays **Apache-2.0** and free; the paid layer is the thing
teams do not want to build themselves — a **hosted, multi-tenant policy + audit control
plane** over the audit stream the binary already emits. This is the HashiCorp / GitLab shape,
and it fits because the data plane is already a clean single binary emitting structured,
correlatable audit.

The defensible story is structural, not statistical: the capability lock **fails closed and is
non-bypassable** — refusing an irreversible action does not depend on *catching* an attack; the
lever was never wired up. That is a compliance claim a classifier / guardrail vendor
structurally cannot make, and it is what an enterprise control plane sells.

## 2. What the data plane already emits (verified in the binary)

The hosted plane sits on top of an audit surface that already exists today. Every endpoint
below is present in `fak serve`; auth is `--require-key-env` (bearer / `x-api-key`), enforced
on every route except `/healthz`.

| Surface | Where it lives | What it carries |
|---|---|---|
| `GET /healthz` | `internal/gateway/http.go:51` (auth-exempt, `:159`) | Unauthenticated liveness + planner kind. |
| `GET /metrics` | `internal/gateway/http.go:52` | Prometheus exposition: HTTP counters, syscall / operation counters, vDSO + KV-cache gauges. Auth-gated. |
| `GET /debug/vars` | `internal/gateway/http.go:53` | expvar JSON diagnostics (HTTP, operations, syscall). Auth-gated. |
| `GET /v1/fak/events` | `internal/gateway/http.go:43` | Structured event/journal stream with a `?since=` cursor — the per-turn adjudication record. |
| `X-Trace-Id` | `internal/gateway/messages.go:141`, `metrics.go:877` | Correlation header, minted or honored per request — the join key across logs. |
| JSON access logs | gateway serve path | One structured record per request, carrying the trace id. |
| Policy manifests | `internal/policy`, [`POLICY.md`](../../POLICY.md) | `--policy FILE` JSON, validated against the **closed reason vocabulary**, with `--dump` ↔ `--check` round-trip. |

Deployment, scaling, and HA guidance (sticky `trace_id` routing) already exists in
[`deployment-guide.md`](deployment-guide.md) and [`advanced-topics.md`](advanced-topics.md).

## 3. Data flow: binary → control plane

The plane is a **read-side aggregator** over the audit the binary already emits; it never
sits in the request path and never weakens the capability floor.

```
   ┌──────────── fak binary (data plane, Apache-2.0) ────────────┐
   │  adjudicator ──► /v1/fak/events  (per-turn allow/deny +     │
   │                   reason token + args hash)                  │
   │                ──► JSON access log (+ X-Trace-Id)            │
   │                ──► /metrics (Prometheus), /debug/vars        │
   └──────────────────────────┬──────────────────────────────────┘
            (out-of-band)      │  ship via sidecar / OTel collector
                              ▼
   ┌──────────── hosted control plane (paid, multi-tenant) ──────┐
   │  ingest ► tenant/project partition (RBAC) ► retention store  │
   │  policy lifecycle + drift (review / diff against deployed    │
   │    floors) ► SIEM export ► fleet metrics dashboard          │
   │  billing + entitlement (the open-core boundary)              │
   └──────────────────────────────────────────────────────────────┘
```

The join key across every record is `X-Trace-Id`; the per-turn verdict and its structured
*reason token* (from the closed vocabulary) are what make the audit machine-queryable rather
than free text. Because the floor fails closed, the plane reports on decisions the binary
already made structurally — it is not a second classifier asked to re-catch attacks.

## 4. The open-core boundary

| Free in the binary (Apache-2.0) | Paid in the hosted plane |
|---|---|
| The default-deny capability floor + closed reason vocabulary. | Multi-tenant auth + org/project model (RBAC over policies). |
| Single-key `--require-key-env` auth. | Policy lifecycle: review workflow, who-changed-the-allow-list, drift/diff alerts. |
| Local `/metrics`, `/debug/vars`, `/v1/fak/events`, `X-Trace-Id`. | Audit export / SIEM integration: retention + search across replicas. |
| `--policy FILE`, `--dump` / `--check`. | Fleet metrics dashboard aggregated across replicas. |
| One binary, self-hosted. | Billing + entitlement; the hosted control plane as a service. |

The line is deliberate: everything needed to *run* a safe agent stays free; everything needed
to *govern a fleet* of them across an organization is the paid layer.

## 5. Child sub-tasks (the build-out, not in this brief)

These are the six tracks the plane decomposes into. Each is intended to be filed as its own
issue and linked from [#576](https://github.com/anthony-chaudhary/fak/issues/576):

1. **Multi-tenant auth + org model** — tenants, projects, RBAC over policies.
2. **Policy lifecycle + drift detection** — review workflow, diff/drift alerts against deployed floors (likely parent of [#202](https://github.com/anthony-chaudhary/fak/issues/202)).
3. **Audit export / SIEM integration** — ship the JSON access logs + `X-Trace-Id` stream; retention; search.
4. **Fleet metrics dashboard** — aggregates `/metrics` across replicas (builds on [#216](https://github.com/anthony-chaudhary/fak/issues/216) / [#196](https://github.com/anthony-chaudhary/fak/issues/196)).
5. **Billing + entitlement** — the open-core boundary itself.
6. **Open-core licensing decision** — Apache-2.0 vs hosted-layer licensing (see the `licensing` label).

## 6. Honest fences

- This monetizes **governance + audit**, not inference. The binary is not a faster model
  server; the plane sells the boundary, the audit, and the compliance story.
- Power / energy numbers are **simulated** and must never enter a pricing or marketing claim.
- The capability lock's "fails-closed, non-bypassable" property is the moat; it must not be
  diluted by any plane feature that re-introduces a catch-based gate.

## 7. Epic acceptance-criteria status ([#576](https://github.com/anthony-chaudhary/fak/issues/576))

| Criterion | Status |
|---|---|
| Written architecture brief for the hosted plane (data flow + open-core boundary). | **Done** — this document. |
| Each child sub-task filed as its own issue and linked here. | **Blocked** — requires GitHub issue-tracker write access; tracked in #576. |
| Explicit "adoption-gated" note (sequenced after wedge + debugger). | **Done** — §Status above. |

## Related

- [#216](https://github.com/anthony-chaudhary/fak/issues/216) · [#196](https://github.com/anthony-chaudhary/fak/issues/196) · [#328](https://github.com/anthony-chaudhary/fak/issues/328) — the foundational observability work this builds on.
- [`observability.md`](observability.md) · [`security.md`](security.md) · [`deployment-guide.md`](deployment-guide.md) · [`advanced-topics.md`](advanced-topics.md) · [`POLICY.md`](../../POLICY.md).
