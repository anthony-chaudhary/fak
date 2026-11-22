---
title: "Run a token factory on fak: the neocloud operator route"
description: "For neocloud and GPU-cloud operators selling inference: prove a fak gateway with no GPU in five minutes, then size, deploy, cap, observe and measure a token factory on your own nodes, with the shipped status of every piece."
---

# Run a token factory on fak

**Audience:** neocloud, GPU-cloud and inference-platform operators who rent or own
accelerators and sell tokens — a *token factory* (also called an inference factory).
You already have GPUs and probably a model server. This page is the shortest route
from a clone of this repository to a governed, observable, OpenAI-compatible endpoint
on your own nodes, and it says plainly what fak does not do for you yet.

**What fak is in that picture:** one static Go binary, `fak serve`, that puts an
OpenAI-compatible and Anthropic-compatible front door in front of the model server
you already run (vLLM, SGLang, llama.cpp, Ollama, or a hosted provider), or serves a
supported GGUF model in-kernel on a rented card. It adds authentication, spend and
session ceilings with typed backpressure, Prometheus metrics, an audit trail, and a
capability floor on agent tool calls. Apache-2.0, no control-plane dependency.

**Next action:** run the five-minute proof below. It needs no GPU, no model and no key.

## Five-minute proof, no GPU

This proves the wiring your customers will call. The planner is a scripted mock, so
the responses are **not model output and not performance evidence**.

```bash
# The CPU build is enough for this proof. The default GPU variant refuses hosts
# where nvidia-smi works (use the CUDA container there) and has no linux/arm64 archive.
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh -s -- --variant cpu

# Terminal 1: an offline gateway on loopback
fak serve --mock --addr 127.0.0.1:8080

# Terminal 2: health, model list, one completion, metrics
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mock","messages":[{"role":"user","content":"Say hello"}],"max_tokens":16}'
curl -s http://127.0.0.1:8080/metrics | grep -c '^fak_'
```

Expect `/healthz` to return `"ok":true` with `"serving":{"mode":"mock","real_model_output":false,...}`,
the completion to carry a `usage` object (`prompt_tokens`, `completion_tokens`,
`total_tokens`), and `/metrics` to expose several hundred `fak_` series. Started
non-interactively with neither a model nor `--mock`, `fak serve` exits with an error
instead of serving; on a terminal it prompts for a model. Mock mode is always labelled
in `/healthz` and in the startup log, so check `real_model_output` before trusting a
deployment.

## The route: prove, size, deploy, cap, observe, measure

| Step | What you do | Where it is maintained |
|---|---|---|
| **1. Front your model server** | Point `--base-url` at the `/v1` endpoint of the engine you already run and add `--require-key-env`. fak adjudicates; your engine generates. Per-provider recipes for CoreWeave, Lambda, RunPod, Crusoe, Vast.ai and Nebius are in the neo-cloud page. | [deploy on a rented GPU cloud](../fak/neo-cloud-deploy.md) · [`fak serve` quickstart](../fak/server-quickstart.md) · [flag and env authority](../fak/server-config.md) · [backend chooser](../supported/backends.md) |
| **2. Or serve in-kernel on the card** | `--gguf PATH` loads a supported GGUF into the in-kernel engine. Use it inside the documented model and hardware envelope, not as an implied replacement for a dedicated production model server. | [in-kernel model](../fak/in-kernel-model.md) · [engines](../supported/engines.md) · [models](../supported/models.md) |
| **3. Size the box before load** | `fak serve --gguf … --plan-json` prints the header-derived weight and KV footprint before any load. `fak serve --gguf … --fit auto` jointly picks context, KV dtype and slots against a stated workload (`--workload-*`) and SLO (`--slo-*`), and refuses a live boot with a typed reason before load instead of guessing. | [capacity and sizing](../fak/deployment-guide.md#capacity--sizing-how-big-a-box-and-when-it-saturates) |
| **4. Deploy** | Kubernetes base plus a one-GPU in-kernel overlay, a Helm chart wrapping both, a hardened systemd unit, and one CUDA image compiled for every supported card architecture (sm_80 through sm_120), with an optional `CUDA_ARCH` build argument to narrow it. | [deployment chooser](../deployment.md) · [Kubernetes manifests](../../deploy/k8s/README.md) · [Helm chart](../../deploy/helm/fak/) · [systemd](../../deploy/systemd/README.md) |
| **5. Meter, cap and shed load** | `fak serve --billing <manifest>` mounts a multi-tenant chain: tenant keys (`fak keys mint`), a price book, a per-tenant ledger with reserve and settle, and HTTP `402` when a balance cannot cover a request. Separately, `--spend-cap tenant:<t>=<N>` (tokens) caps a tenant and needs `--spend-scope-trace` to name the tenant scope, or the gateway refuses to start; `FAK_MAX_SESSIONS` sets a session ceiling. Past the ceiling a new session gets HTTP `503` with `Retry-After` and the closed reason `SESSION_CEILING_SATURATED`; sessions already in flight are never refused. | [billing quickstart](../billing/quickstart.md) · [scale-out state model](../fak/deployment-guide.md#scale-out-state-model-running-n-replicas) |
| **6. Observe** | Scrape `GET /metrics`: token counters and rates under `fak_gateway_inference_*` (completion tokens, decode tokens per second, cached-prompt hit ratio), session saturation as `fak_gateway_session_saturation` and `fak_gateway_session_ceiling` with the live count in `fak_gateway_sessions_live`, per-operation latency histograms. Access logs carry a `trace_id` and omit request bodies. | [observability guide](../fak/observability.md) |
| **7. Measure your own goodput** | `cmd/loadgen` is a stdlib-only load generator for any OpenAI-compatible endpoint: a concurrency sweep with Poisson, gamma or constant arrivals that writes usage-based output tokens per second, first-content timing and latency percentiles to a `MATRIX.json`. Run it against the raw engine and against fak in front of it, on your node; pass `-api-key-env` when the gateway requires a key. For SLO-based goodput and a capacity search, `fak webbench serving` replays agent requests against both. | [benchmark index](../benchmarks/README.md) · [benchmark authority](../../BENCHMARK-AUTHORITY.md) |
| **8. Keep it running** | Health, recovery order, upgrade and rollback anchors. | [operator route](README.md) · [troubleshooting](../fak/server-troubleshooting.md) |

Step 7 as a command from a clone of this repository (Go toolchain required), against the
proof gateway from above or any real endpoint:

```bash
go run ./cmd/loadgen -url http://127.0.0.1:8080/v1/chat/completions \
  -model mock -concurrency 1,4,16 -requests-per 32 -out MATRIX.json
```

Report goodput against an SLO you declare (first-token and completion deadlines), not
raw throughput: a sweep point that misses its deadline is capacity you cannot sell.

## What a token factory needs, and where fak stands

Status words follow the [claims ledger](../../CLAIMS.md): **shipped** means real code
closed by a mechanical witness; **limited** means shipped inside a stated boundary;
**yours** means the repository does not provide it.

| Need | Status | Detail |
|---|---|---|
| OpenAI-compatible and Anthropic-compatible endpoint | Shipped | `/v1/chat/completions` and `/v1/messages`; provider wires in [APIs and protocols](../supported/apis-and-protocols.md). |
| Front an existing vLLM, SGLang, llama.cpp or Ollama server | Shipped | Proxy mode over the OpenAI-compatible wire; the common production shape. |
| Bearer-key authentication | Shipped | `--require-key-env`: every route needs the key except the probes (`/healthz`, `/readyz`) and the key-proof endpoint; a few discovery and read-only pages are also open to loopback callers that arrive without proxy headers. TLS terminates at your ingress. |
| Prometheus metrics, access log, trace correlation | Shipped | On by default; no request bodies or tool arguments are logged. |
| Session ceiling with typed backpressure | Shipped | Inert until `FAK_MAX_SESSIONS` is set. |
| Spend caps and rate limits | Limited | Enforced **per process**. With N replicas the fleet-wide allowance is N times the per-replica value; set each replica to `C / N`. There is no shared distributed ledger. |
| Multi-replica serving | Limited | Sessions, budgets and prefix warmth are process-local; use sticky routing by tenant. |
| Kubernetes, Helm, systemd, CUDA image | Shipped | Your platform still owns ingress, identity, persistent state, secrets and high availability. |
| In-kernel GPU serving | Limited | Bounded model and hardware envelopes; several CUDA prefill paths are still below llama.cpp on the same card. Read the status column of the [latest hardware results](../../README.md) before planning capacity on it. |
| Continuous batching in-kernel | Limited | Wired into decode and code-verified; no accepted throughput receipt yet. See [batching configuration](../fak/batching-config.md). |
| Prefix (KV) reuse in-kernel | Limited | Measured first-token and hit-rate receipts exist ([radix attention results](../benchmarks/RADIXATTENTION-RESULTS.md)); CUDA paged-KV prefix sharing is opt-in (`FAK_CUDA_PAGED_KV=1`). |
| Capability floor on agent tool calls | Shipped | Useful when your customers run agents: every proposed tool call is adjudicated against a policy manifest before it runs. See [policy](../policy.md). |
| Usage metering, rating, prepaid balances | Limited | `fak serve --billing` with a price book, per-tenant unit ledger, per-tenant rollups and manifest rate limits. State is local to the gateway's `state_dir`; read the [billing quickstart](../billing/quickstart.md) for the boundary. |
| Tenant keys and quotas | Limited | CLI only: `fak keys mint`, `topup`, `balance`, `revoke`, `list` over a key store hashed at rest. Outside billing mode, keys are operator-provisioned environment values. |
| Invoicing, payment collection, signup portal, quota UI | Yours | Not in this repository. |
| Multi-region control plane, autoscaling policy | Yours | A hosted control plane exists only as an [architecture brief](../fak/hosted-control-plane.md). |
| Gateway cost in front of an engine | Limited | One published sweep: fak in front of SGLang on an 8-GPU server reached 0.75× raw SGLang throughput at concurrency 64 and 0.97× at 128, with a larger tax in the mid-range ([results](../benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md)). Measure your own engine with step 7. |
| Matched SLO-goodput comparison of in-kernel serving versus vLLM or SGLang | Not yet | None is published. |

## What to send back

The fastest way to make this route better for the next operator is a result from a
real node. Open an issue with the card, the engine and version you fronted, the
`MATRIX.json` from step 7 for the raw engine and for fak in front of it, and the
first thing on this page that was wrong or missing. Contribution rules are in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Generation, support and lifecycle

This is a current (`gen/now`) routing page. Its authority is routing: each linked
runbook owns its commands, flags, supported wires and hardware boundary, and the
running binary's `fak serve --help` decides actual behavior at the revision you
deploy. A row in the status table changes only when its linked authority changes.
The five-minute proof is a kernel wiring proof, not a production topology and not a
capacity statement.
