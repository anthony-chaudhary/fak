---
title: "fak Track F integration/tooling-parity tracker: the 8 children, honest on-disk status, and the gate that keeps the epic open"
description: "Umbrella tracker for epic #302 (Track F - Integration/Tooling Parity). Records the CORRECT child issue map (the epic body's links are stale internal-tracker ids), each child's honest shipped/partial/unimplemented status with deciding files, and the named gate — published-artifact + unbuilt-surface acceptances no single host change can flip — that keeps the roll-up open. No number is invented."
---

# Track F — Integration/Tooling Parity (epic #302) — status tracker

> **Umbrella tracker for [#302](https://github.com/anthony-chaudhary/fak/issues/302)** —
> *"OpenAI API parity, prompt caching, metrics/telemetry, MCP ecosystem, developer
> tooling, deployment guides, SDK generation, policy editor GUI."*
>
> **#302 is a pure roll-up: it closes only when all 8 children close.** This doc does not
> claim the epic is done — it records the **honest on-disk state** of each child so an
> operator (or the next agent) can see exactly what shipped, what is wired-but-acceptance-open,
> and what is not built at all. **House rule: every claim comes from the tree + `git`, not a
> worker's say-so.** Where an acceptance is a *published artifact* (a registry-pushed Docker
> image, three published SDK packages) or a *whole unbuilt surface* (a web GUI, real
> OpenTelemetry spans), the cell says so plainly; nothing is asserted as shipped that the tree
> does not show. Written **2026-06-25** on a win32 dev box.

---

## 0. The child map — the epic's own links are stale (read this first)

The migrated epic body (#302) lists its children as **#348, #351, #353, #356, #358, #361,
#364, #367**. Those numbers are **wrong** — they are pre-migration internal-tracker ids that
GitHub renumbered on import. **#348 does not resolve at all**, and the rest now point at
unrelated issues (e.g. #353 is a *closed* `examples: healthcare/PHI agent policy`, #361 is a
*closed* GPU-env-knobs docs issue, #367 is a *closed* `demo(simpledemo)` repetition bug). The
epic's note *"cross-references are auto-maintained"* is therefore **not holding** for this
epic. The **real** children — the issues actually tagged `track/F-integration-tooling` with an
`F-00x` slug — are:

| Slug | Live issue | Priority | Title | Epic body shows (WRONG) |
|---|---|---|---|---|
| **F-001** | [#221](https://github.com/anthony-chaudhary/fak/issues/221) | P0 | OpenAI API Feature Parity *(CLOSED)* | #348 *(does not resolve)* |
| **F-002** | [#218](https://github.com/anthony-chaudhary/fak/issues/218) | P1 | Prompt Caching Features | #351 |
| **F-003** | [#216](https://github.com/anthony-chaudhary/fak/issues/216) | P1 | Metrics/Telemetry Production | #353 |
| **F-004** | [#213](https://github.com/anthony-chaudhary/fak/issues/213) | P2 | MCP Ecosystem Expansion | #356 |
| **F-005** | [#211](https://github.com/anthony-chaudhary/fak/issues/211) | P2 | Developer Tooling | #358 |
| **F-006** | [#208](https://github.com/anthony-chaudhary/fak/issues/208) | P2 | Deployment Guides | #361 |
| **F-007** | [#205](https://github.com/anthony-chaudhary/fak/issues/205) | P2 | SDK Generation | #364 |
| **F-008** | [#202](https://github.com/anthony-chaudhary/fak/issues/202) | P2 | Policy Editor GUI | #367 |

> The children's own migrated bodies still cite the **internal** epic id *"Epic #266"*; the
> live GitHub umbrella is **#302**. Both refer to the same Track F.

This doc keys everything off the **live** numbers above.

> **Operator action (the one GitHub-side fix):** for #302's *"closes when all children close"*
> mechanism to actually fire, repoint the epic body's checkbox list from the stale ids to the
> live ones — `#348→#221, #351→#218, #353→#216, #356→#213, #358→#211, #361→#208, #364→#205,
> #367→#202`. Until then the epic tracks eight unrelated (and several closed) issues and will
> never auto-close from its children. This in-repo tracker is the authoritative map in the
> meantime. (Same stale-link defect as the sibling epics — see
> [`track-b-performance-parity-tracking-306.md`](track-b-performance-parity-tracking-306.md)
> and [`track-d-agent-framework-parity-tracking-304.md`](track-d-agent-framework-parity-tracking-304.md).)

---

## 1. Honest status — each child, verified against the working tree (2026-06-25)

Status is read from the tree + `git log`, not from any worker's say-so. "Wired" means the
capability exists and is reachable today; "partial" means a real subset of the acceptance
shipped while a named, separable piece did not.

| Slug · issue | On-disk state | What is real today | What the acceptance still wants | Deciding file(s) |
|---|---|---|---|---|
| **F-001** #221 OpenAI parity (P0) | 🟢 **CLOSED — shipped** | `/v1/chat/completions` (incl. **streaming SSE**), `/v1/embeddings`, `/v1/moderations`, `/v1/models`, `/v1/messages` all routed and served; per-call array batching witnessed (`TestEmbeddingsBatchReturnsPerItemResults`, `TestModerationsBatchReturnsPerItemResults`); OpenAI client shape witnessed end-to-end. | — (closed) | [`internal/gateway/http.go`](../../internal/gateway/http.go) · [`internal/gateway/openai_parity_test.go`](../../internal/gateway/openai_parity_test.go) · [`internal/gateway/embeddings.go`](../../internal/gateway/embeddings.go) |
| **F-002** #218 prompt caching (P1) | 🟡 **Cache + hit/miss metrics shipped; control-API/TTL/pricing open** | The **vCache** virtual-API-cache spine (epic **#715**, CLOSED) caches over upstream providers with a cost-gated correctness law; **cache hit/miss + provider hit-ratio render on `/metrics`** (`fak_cache_*` via the unified cachemeta fold). `fak vcache` proves the recall cost-gate deterministically. | A user-facing **cache-control header/param** matching Anthropic/OpenAI `cache_control`, a **configurable TTL** knob, and a **pricing model** — vCache budgets at the *uncached* price (hits are realized rebates), not a TTL/cache-control surface. | [`cmd/fak/vcache.go`](../../cmd/fak/vcache.go) · [`internal/cachemeta/provider.go`](../../internal/cachemeta/provider.go) · [`internal/cachemeta/stream_metrics.go`](../../internal/cachemeta/stream_metrics.go) |
| **F-003** #216 metrics/telemetry (P1) | 🟡 **Prometheus + Grafana + alerts shipped; OTel spans open** | A full **Prometheus exposition** `/metrics` endpoint (`fak_gateway_*`/`fak_vdso_*`/`fak_cache_*` with `# HELP`/`# TYPE`, histograms, build-info); a **Grafana** stack (provisioned dashboards + `gen_dashboard.py`); **alert/SLO definitions** in `prometheus-alerts.yml`. | **OpenTelemetry spans/tracing** — there is **no `go.opentelemetry.io` import**; tracing today is an internal `tracesink` + the `/v1/fak/trace` endpoint, not OTel-wire spans. 3 of 4 acceptance boxes are met; OTel is the one gap. | [`internal/gateway/metrics.go`](../../internal/gateway/metrics.go) · [`tools/grafana/gen_dashboard.py`](../../tools/grafana/gen_dashboard.py) · [`tools/grafana/prometheus-alerts.yml`](../../tools/grafana/prometheus-alerts.yml) |
| **F-004** #213 MCP expansion (P2) | 🟡 **fak governs MCP; the registry/discovery surface is unbuilt** | fak **governs** MCP traffic: a `/mcp` gateway endpoint (openapi `paths./mcp`) plus a governance example and a worked client (`examples/mcp`, `examples/mcp-client`). | The F-004 *expansion* features — **tool auto-discovery, a tool registry, resource access, prompt templates, a server directory** — are not built. This is partly an architecture-fit question: fak is the *gate* in front of an MCP server, not an MCP registry/provider. | [`examples/mcp/README.md`](../../examples/mcp/README.md) · [`examples/mcp-client/client.py`](../../examples/mcp-client/client.py) · [`docs/fak/openapi.yaml`](../../docs/fak/openapi.yaml) (`/mcp`) |
| **F-005** #211 developer tooling (P2) | 🟡 **debug/doctor/bench shipped; dedicated `profile`/`test` verbs absent** | `fak debug`, `fak doctor`, `fak headroom`, and a bench family (`fak bench`/`webbench`/`routebench`/`swebench`) ship as verbs; the dev-workflow contract lives in `AGENTS.md`/`CONTRIBUTING.md`. | A dedicated **`fak profile`** verb and a **`fak test` runner** verb as named in the issue — neither is registered in `cmd/fak/main.go`'s verb switch. | [`cmd/fak/debug.go`](../../cmd/fak/debug.go) · [`cmd/fak/doctor.go`](../../cmd/fak/doctor.go) · [`cmd/fak/headroom.go`](../../cmd/fak/headroom.go) |
| **F-006** #208 deployment guides (P2) | 🟡 **Guides shipped (Docker/Compose/K8s/bare-metal/checklist); published image open** | A complete `deployment-guide.md` with **Docker** build/run-hardened, **Docker Compose**, **Kubernetes** (live `deploy/k8s/` manifests + kustomization), **bare-metal** (systemd unit, hardened sandbox), and a **production-readiness checklist**. | An **official, registry-published Docker image** (the guide builds from a clone, no `ghcr.io`-pushed artifact) — a release-infra task, not a doc. Also formally `Depends on` F-003 (#216). | [`docs/fak/deployment-guide.md`](../../docs/fak/deployment-guide.md) · [`Dockerfile`](../../Dockerfile) · [`deploy/k8s/fak.yaml`](../../deploy/k8s/fak.yaml) · [`deploy/k8s/kustomization.yaml`](../../deploy/k8s/kustomization.yaml) |
| **F-007** #205 SDK generation (P2) | 🟡 **OpenAPI spec shipped; published SDKs absent** | A generated **OpenAPI 3 spec** (`docs/fak/openapi.yaml`, v0.30.0) covering the chat/embeddings/moderations/models/messages + `/v1/fak/*` + `/mcp` surface. The Go module itself is importable as a client. | **Published Python, TypeScript, and Go SDK packages** generated from the spec — no generated SDK packages exist in-tree or on a registry. | [`docs/fak/openapi.yaml`](../../docs/fak/openapi.yaml) |
| **F-008** #202 policy editor GUI (P2) | 🔴 **Not implemented** | `fak policy` CLI authoring + JSON-schema validation exist on the command line. | A **web-based** visual editor (React/Next.js frontend, real-time validation, visual tool builder, export-to-JSON) — no `.tsx`/`.jsx`/web frontend in the tree. A 14–21-day effort by the issue's own estimate. | — (no web UI) |

Legend: 🟢 done · 🟡 partial (real subset shipped, named piece open) · 🔴 not implemented.

**Roll-up: 1 / 8 children closed (#221).** Of the 7 open: two are essentially *acceptance-
complete except one published artifact* (F-006 a registry image, F-003 OTel spans), two are
*real-subset partials* (F-002 cache-control API; F-005 the `profile`/`test` verbs), one is
*spec-shipped-SDKs-pending* (F-007), one is an *architecture-fit unbuilt surface* (F-004), and
one is an *unbuilt web GUI* (F-008). So #302 stays **OPEN** — correctly.

---

## 2. The gate — why #302 cannot honestly close here

The seven open acceptance gates fall into three classes, and **none is a single in-repo change
this host can land without fabricating a shipped artifact:**

1. **Published-artifact acceptances (F-006 official Docker image, F-007 three published SDKs).**
   The *content* is done — the deployment guides are written and the OpenAPI spec is generated —
   but "official Docker images" and "Python/TypeScript/Go SDK **published**" mean release-infra
   artifacts pushed to a registry. Asserting them closed without the published packages would be
   the exact overclaim `make claims-lint` and the witness ledger reject.

2. **Whole-surface features (F-003 OpenTelemetry spans, F-004 MCP registry/discovery, F-008 the
   web policy editor).** These are not "shipped minus a flag" — there is no implementation to
   point at. Real OTel spans need the `go.opentelemetry.io` SDK wired through the gateway; the
   MCP expansion needs a registry/discovery/prompt-template surface (and a prior architecture-fit
   decision, since fak *gates* MCP rather than *providing* it); the GUI is a 14–21-day React app.

3. **Real-subset partials (F-002 cache-control/TTL/pricing API, F-005 the `fak profile`/`fak test`
   verbs).** A genuine subset shipped (cache hit/miss metrics; `debug`/`doctor`/bench), but the
   named remaining piece is a separate leaf, not a one-line completion.

**The single honest gate, stated plainly:** epic #302 closes only when all 8 children meet their
acceptance, and **7 are open** — spanning two published-artifact gaps, three unbuilt surfaces, and
two real-subset partials. No code or doc change on this host can flip those bits without either
publishing a registry artifact this box cannot publish or shipping a multi-day feature. So the
correct deliverable is this tracker, not a closed epic.

---

## 3. Smallest next step per child (for the agent that picks one up)

| Child | Smallest honest next step | Where it runs |
|---|---|---|
| F-001 #221 | Done (closed). Optional: add an OpenAI async `/v1/batches` endpoint if true batch-API parity (not just per-call arrays) is later required | host-tractable |
| F-002 #218 | Add a `cache_control`-style request param + a configurable TTL knob over the vCache spine, surface a pricing/rebate line in the cache stats; the `fak_cache_*` hit/miss metrics already exist to validate it | host-tractable |
| F-003 #216 | Wire the `go.opentelemetry.io` SDK through the gateway request path (spans around adjudicate/serve), keeping the existing `tracesink` as the local sink; Prometheus + Grafana + alerts are already shipped | host-tractable |
| F-004 #213 | Decide the architecture fit first (does fak host an MCP registry, or only govern upstream MCP servers?); if hosting, add tool auto-discovery + a server directory as the first leaf | host-tractable (design first) |
| F-005 #211 | Register a `fak profile` verb (pprof capture over a serve/bench run) and a `fak test` runner verb wrapping the witness suites; both follow the existing `cmd/fak` verb pattern | host-tractable |
| F-006 #208 | Add a CI release job that builds and pushes an `official` image to a registry (e.g. `ghcr.io`), then cite the pulled tag from the guide; the guides themselves are complete | a CI/release runner |
| F-007 #205 | Run an OpenAPI generator over `docs/fak/openapi.yaml` for Python/TS/Go, smoke each against a live `fak serve`, publish the three packages | a CI/release runner |
| F-008 #202 | Scaffold the React/Next.js policy editor as a separate `web/` app (or sibling repo): load the policy JSON schema, real-time validate, export to JSON | host-tractable (large) |

---

## 4. Provenance

- **Child set:** `gh issue list --label track/F-integration-tooling --state all` (live,
  2026-06-25); acceptance criteria from each child body (#221/#218/#216/#213/#211/#208/#205/#202).
- **Stale-link evidence:** `gh issue view` on the epic-body ids — #348 does not resolve; #353,
  #361, #367 are unrelated *closed* `examples:`/docs/demo issues, not the F-00x titles they
  stand in for.
- **On-disk evidence:** `internal/gateway/http.go` + `openai_parity_test.go` + `embeddings.go`
  (F-001); `cmd/fak/vcache.go` + `internal/cachemeta/{provider,stream_metrics}.go` (F-002);
  `internal/gateway/metrics.go` + `tools/grafana/{gen_dashboard.py,prometheus-alerts.yml}` (F-003);
  `examples/mcp` + `examples/mcp-client` + `docs/fak/openapi.yaml` `/mcp` (F-004);
  `cmd/fak/{debug,doctor,headroom}.go` (F-005); `docs/fak/deployment-guide.md` + `Dockerfile` +
  `deploy/k8s/` (F-006); `docs/fak/openapi.yaml` v0.30.0 (F-007); no web frontend in-tree (F-008).
- **OTel-absence check:** `rg 'go.opentelemetry.io|otel\.'` over the Go tree returns no SDK import
  (only the internal `tracesink`), grounding the F-003 OTel gap.
- **Honesty rails:** `make claims-lint`, BENCHMARK-AUTHORITY — every "shipped" cell here points at
  a real file; every "open" cell names the missing artifact rather than asserting it done.

## 5. See also

- [`docs/notes/track-b-performance-parity-tracking-306.md`](track-b-performance-parity-tracking-306.md) ·
  [`docs/notes/track-d-agent-framework-parity-tracking-304.md`](track-d-agent-framework-parity-tracking-304.md) —
  sibling parity trackers in the same house format (same stale-link defect, same honest-status gate).
- [`docs/fak/deployment-guide.md`](../fak/deployment-guide.md) — the F-006 guides; [`docs/fak/openapi.yaml`](../fak/openapi.yaml) — the F-007 spec.
- [`docs/integrations/README.md`](../integrations/README.md) — the wire-level integration front door (repoint-the-base-URL) that grounds the OpenAI-compatible surface F-001/F-002 build on.
- Epic **#715** (vCache, the F-002 caching spine, CLOSED).
