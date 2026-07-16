---
title: "Choose a fak deployment envelope"
description: "Operator route for choosing a local, fleet, cloud, or air-gapped fak deployment and continuing to its maintained runbook."
---

# Choose a fak deployment envelope

**Audience:** operators deciding where and how to run fak. The same `fak serve` kernel
runs in every envelope; topology, model location, network boundary, and operational
ownership determine the route.

**Current default:** start with one `fak serve` process and one separately managed model
endpoint. **Next action:** choose the row below that matches your network and ownership
requirements, then run the linked route's first health check.

## Choose by requirements

| Envelope | Choose it when | Current support boundary | Maintained route and first check |
|---|---|---|---|
| **Local, one process** | One operator is proving the gateway, developing an integration, or serving a trusted local workload. | One process owns policy, gateway state, and local observability. This route does not establish multi-node availability. The model may be local or remote. | Follow the [single-process deployment](fak/deployment-guide.md#option-a-one-process-single-host). Confirm `GET /healthz` returns `200`. |
| **Fleet or Kubernetes** | Multiple replicas, service discovery, rolling updates, or an internal platform team owns availability. | The repository ships a Kubernetes base and GPU overlay. Your platform still owns ingress, identity, persistent state, secrets, and the model-serving service. | Apply the [Kubernetes manifests](../deploy/k8s/README.md); require `kubectl rollout status deployment/fak` to succeed, then port-forward the Service and require `curl -f http://127.0.0.1:8080/healthz` to succeed. |
| **Cloud GPU host** | You rent or own a GPU node and want fak beside an in-kernel model or a co-located vLLM/SGLang server. | This row covers one GPU host. Provider networking, disks, credentials, quotas, and production HA remain operator responsibilities. For a cloud fleet, use the fleet row on cloud nodes. | Follow the [single-process production guide](fak/deployment-guide.md#production-single-host), configure its documented GPU-backed engine, bind health locally or to an authenticated internal network, and require `curl -f http://127.0.0.1:8080/healthz` to succeed. |
| **Air-gapped or disconnected** | Runtime traffic, model weights, policies, and evidence must remain inside a controlled network boundary. | The runtime can use local policy and a local model endpoint without provider calls. You must pre-stage the fak binary, model/runtime artifacts, policy, trust material, and updates; the 60-second offline proof is a kernel proof, not a production topology. | Run the [offline proof](repro-packet.md#the-60-second-offline-proof) inside the boundary, then apply the single-process or fleet topology with only local endpoints. Capture an operator-owned firewall or network-policy egress-denial check separately; `/healthz` does not prove isolation. |

“Local” and “cloud” describe placement; “one process” and “fleet” describe topology;
“air-gapped” describes the network boundary. They can combine. For example, an air-gapped
Kubernetes fleet uses the fleet topology but replaces every external dependency with a
pre-staged local one.

## Mode and model placement

Two serving modes are current:

1. **Gateway mode (operator default):** `fak serve` fronts a local or remote model server
   over a supported provider wire. The model can run on CPU or GPU independently of where
   fak runs.
2. **In-kernel mode:** fak loads a supported model path directly. Use this for the bounded
   model and hardware envelopes documented by the in-kernel engine, not as an implied
   replacement for a dedicated production model server.

Choose model execution separately from deployment topology. Use the
[backend-selection route](supported/backends.md) to decide between a remote server, CPU
reference work, and GPU serving.

## Generation, support, and lifecycle

This is the current (`gen/now`) deployment front door. Its authority is routing: each
linked runbook owns its commands, prerequisites, supported provider wires, model paths,
and hardware boundary. Pin the released binary or container version selected by that
runbook; a page at current trunk does not make an older artifact current. A route is
current only while its first health check and referenced versioned artifacts exist at the
linked location.

Add an envelope after a runnable path and health witness ship. Relabel or remove it when
its artifacts or health witness no longer cover current trunk, and point to the
superseding route. Historical experiments and dated fleet notes are evidence, not current
operator instructions unless a current runbook links them explicitly.

Across all envelopes, `/healthz` proves process readiness, not model quality, policy
fitness, capacity, or high availability. Before production traffic, continue through the
[deployment checklist](fak/deployment-guide.md#production-checklist), including policy,
TLS, authentication, secrets, observability, backup, and rollback ownership.

## Decision recap

- Start local and single-process unless availability or platform ownership requires a
  fleet.
- Use the cloud GPU route when fak and model serving belong on a rented accelerator host.
- Treat air-gapped operation as a dependency-staging and network-boundary requirement,
  then choose either the single-process or fleet topology inside that boundary.
- Verify the linked route's first health check before adding production traffic.
