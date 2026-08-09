---
title: "fak operator route: deploy, observe, recover, upgrade"
description: "The current production route for operators deploying fak serve, observing it, recovering service, and choosing an upgrade or rollback anchor."
---

# Operator route

This page is for **operators responsible for a running `fak serve` gateway**. It
routes production work to the maintained runbooks without sending you through
contributor setup or design history.

**Next action:** after starting or reaching the gateway, run this health check:

```bash
curl -s http://127.0.0.1:8080/healthz
```

A healthy process returns JSON with `"ok":true`. Change the host and port to the
address you deployed. This is the one initial check; use the task route below for
the work that follows.

## Production task route

| Task | Start here | What the route establishes |
|---|---|---|
| **Deploy** | [Deployment chooser](../deployment.md) | Choose local proof, fleet, cloud GPU, private lab GPU, or air-gapped operation by workload. For the gateway command and authentication baseline, continue to the [`fak serve` quickstart](../fak/server-quickstart.md); [`server-config.md`](../fak/server-config.md) is the flag and environment-variable authority. |
| **Observe** | [Observability guide](../fak/observability.md) | Scrape `GET /metrics`, inspect `GET /debug/vars`, and correlate structured access logs with `trace_id`. These surfaces are on by default and omit request bodies, tool arguments, and result content. |
| **Recover** | [Server troubleshooting](../fak/server-troubleshooting.md) | Match the observed symptom before changing the service. Diagnose port, memory, GPU, model, policy, and startup failures; then re-run `/healthz`. For a release regression, restore a previously promoted stable anchor rather than inventing an application-level rollback. |
| **Upgrade** | [Install-path chooser](../adoption/install-paths.md) | Reuse the installation channel you operate. `go install ...@latest` selects the newest tag, while source builds select the checked-out revision. [Stable releases](https://github.com/anthony-chaudhary/fak/blob/main/docs/stable-releases/README.md) are sparse, evidence-backed rollback anchors; rolling `vX.Y.Z` tags are the faster release channel. |

## Operating mode and defaults

The production gateway mode is `fak serve`: an OpenAI-compatible HTTP gateway in
front of either a remote OpenAI-compatible model server or an in-kernel GGUF
model. The smallest local deployment binds `127.0.0.1:8080`; a network-facing
deployment must make its exposure, authentication, model, and policy choices
explicit. The server quickstart gives the runnable forms, while the server-config
guide remains authoritative when examples and flags differ.

Do not treat the local no-model proof as a production deployment. If the workload
requires CUDA, private lab access, DC networking, or heavy CPU, the
[deployment chooser](../deployment.md) routes it to sanctioned compute. Runtime
support is bounded by the documented server flags, endpoints, backends, and
installation channels; contributor scripts and historical notes are not operator
contracts.

## Recovery and upgrade order

1. Read `/healthz`, then metrics, logs, and `/debug/vars` before restarting or
   changing configuration.
2. Follow the troubleshooting route for the symptom and verify `/healthz` again.
3. If a newly installed release is the cause, select a known release artifact or
   a promoted `stable/<codename>` anchor. Stable tags identify commits; they do
   not provide an in-process rollback command.
4. Reapply the deployment's explicit authentication, policy, model, and address
   configuration, then repeat the health and observability checks.

This ordering keeps diagnosis separate from mutation and makes the release or
configuration change visible to the operator.

## Generation, lifecycle, and support boundary

| Context | Operator meaning |
|---|---|
| **Generation** | This is the current `gen/now` route. It indexes shipped operator surfaces; it does not generate configuration or change a running service. |
| **Lifecycle** | Deploy, observe, recover, and upgrade are recurring production tasks. Rolling releases move quickly; promoted stable tags are retained as rollback anchors. |
| **Support** | The linked quickstart, server configuration, observability, troubleshooting, install, and stable-release pages are the scoped authorities. Historical notes and contributor workflows may explain decisions but do not override them. |
| **Runtime authority** | The running binary's `fak serve --help`, its configured environment, and its health/telemetry responses determine actual behavior. Validate those at the deployed revision. |

For model and policy choices made before deployment, use the current
[model-routing route](../explainers/routing.md) and [policy route](../policy.md).
For supported execution backends, use the [backend chooser](../supported/backends.md).
