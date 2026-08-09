---
title: "Deploy fak on a rented GPU cloud (CoreWeave / Lambda / RunPod / Crusoe / Vast / Nebius)"
description: "The operator quickstart for running the fak gateway ON a rented GPU box or GPU k8s pool — in-kernel on the GPU, or proxy in front of a co-located vLLM/SGLang. The opposite shape from fronting a hosted model API."
---

# Deploy fak on a rented GPU cloud

This page is the copy-paste path for **standing up `fak` on a GPU you rent** from a
neo cloud — CoreWeave, Lambda, RunPod, Crusoe, Vast.ai, Nebius. It is the operator's
"get to a running gateway on the box" quickstart, not a marketing page.

It sits between two neighbours that look similar and are not:

- [Clouds & hosted providers](../supported/clouds.md) is the **opposite** shape: it
  fronts a hosted model **API** (`fak serve --provider openai --base-url <cloud /v1>`
  for Bedrock, Vertex, Groq, Together, Fireworks). No GPU of yours — you proxy someone
  else's tokens.
- [Neo-cloud reference architecture](https://github.com/anthony-chaudhary/fak/blob/main/docs/vendor/neo-cloud-reference-architecture.md) is
  the **binding layer**: how a neo-cloud operator exposes many accelerator *backends*
  through one control plane. That is about the backends behind the gateway; this page
  is about standing the gateway up on one rented box.

Everything generic to running `fak serve` in production — the auth floor
(`--require-key-env`), the policy JSON, TLS termination, timeouts, the readiness
checklist — lives in the [deployment guide](deployment-guide.md). This page is only the
**rented-GPU delta**: which image, which arch, and the two host shapes.

## The honest core: two host primitives, six storefronts

A rented GPU cloud hands you exactly one of two things, and every provider below is a
storefront over one of them:

1. **A raw GPU VM** (an on-demand instance, or a marketplace container host). You get
   `ssh` + `nvidia-smi`. → Install the binary and run it, or `docker run --gpus all`.
2. **A GPU Kubernetes pool** (managed k8s with the NVIDIA device plugin installed). You
   get `kubectl` and nodes advertising `nvidia.com/gpu`. → Apply an overlay.

So there are only two recipes to learn. Pick your shape first, then jump to your
provider for the login-level specifics.

## Two shapes, one page

### Shape 1 — in-kernel on the rented GPU

`fak` runs the model itself from a mounted `.gguf` (`--engine inkernel --backend cuda`).
No upstream provider, so no provider key and no `--base-url`. This needs the GPU-serving
image, built from [`Dockerfile.cuda`](https://github.com/anthony-chaudhary/fak/blob/main/Dockerfile.cuda)
(the default `Dockerfile` stays the lean, GPU-free static image for Shape 2).

> **Image status (read this first).** The versioned `-cuda` image is **not yet published**
> — the first `<version>-cuda` tag publishes with the first release cut after
> `Dockerfile.cuda` landed (via `.github/workflows/release-cuda-container.yml`). Until
> then, **build your own**. The plain build takes no arch flag and covers every
> supported card:
>
> ```bash
> docker build -f Dockerfile.cuda -t REGISTRY/fak:cuda .
> ```
>
> With `CUDA_ARCH` unset the builder compiles a cubin for **every** arch in
> `internal/compute/cuda_arch.txt` — `sm_80` (A100) · `sm_89` (Ada L4/L40, RTX 40xx) ·
> `sm_90` (H100/H200) · `sm_100` (B100/B200) · `sm_120` (consumer Blackwell, RTX 50xx) —
> plus a `compute_120` PTX floor the driver JITs forward onto a newer card. That is the
> build to use when you do not know in advance which card you will be handed.
>
> Passing `--build-arg CUDA_ARCH=sm_90` **narrows** the image to one arch — smaller, and
> the right call on a provider where you know the card you get. The value is validated against
> `cuda_arch.txt` and fails the build if it is not in that set. A narrowed image carries
> no PTX floor, so an arch/card mismatch is the first thing to check when a single-arch
> GPU pod won't decode.

On a **raw GPU VM** (the NVIDIA Container Toolkit installed by the provider image):

```bash
docker run --rm --gpus all -p 8080:8080 REGISTRY/fak:cuda \
    serve --addr 0.0.0.0:8080 --gguf /models/model.gguf \
    --engine inkernel --backend cuda \
    --require-key-env FAK_GATEWAY_KEY --policy /etc/fak/policy.json
```

On a **GPU k8s pool**, apply the in-kernel overlay (it carries the Secret, policy floor,
a weights PVC, and GPU-aware probes/timeouts already tuned for slow local decode):

```bash
kubectl apply -k deploy/k8s/overlays/gpu
```

The overlay ([`deploy/k8s/overlays/gpu/`](https://github.com/anthony-chaudhary/fak/blob/main/deploy/k8s/overlays/gpu/fak-gpu.yaml),
documented in [`deploy/k8s/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/deploy/k8s/README.md))
assumes two prerequisites `fak` does **not** install: the NVIDIA device plugin (or GPU
Operator) is already on the cluster, and a `.gguf` is loaded onto the `fak-weights`
claim at `/weights/model.gguf`. Pin `<version>-cuda` in its `kustomization.yaml`
`images:` block once a versioned tag exists.

### Shape 2 — proxy in front of a co-located model server

You run a real serving engine (vLLM / SGLang) on the box and put `fak` in front to
adjudicate. `fak` does not touch the GPU here, so the **lean static image is enough** —
no CUDA build:

```bash
# vLLM (or SGLang) is already serving an OpenAI-compatible endpoint on :8000
fak serve --provider openai --base-url http://localhost:8000/v1 \
    --addr 0.0.0.0:8080 --require-key-env FAK_GATEWAY_KEY --policy policy.json
```

This is the same `--provider openai --base-url …` mechanic as fronting a hosted API —
see [clouds.md](../supported/clouds.md) and the [deployment guide](deployment-guide.md)
— the only difference is the upstream lives on the same rented box. On k8s, this is the
proxy base at [`deploy/k8s/`](https://github.com/anthony-chaudhary/fak/blob/main/deploy/k8s/)
(`kubectl apply -k deploy/k8s`), which applies cleanly alongside the GPU overlay.

## Per-provider recipes

Each provider is one of the two primitives above. The provider-specific part is only
*how you get the box*; the `fak` command does not change.

| Provider | Primitive you rent | Shape 1 (in-kernel) | Shape 2 (proxy) |
|---|---|---|---|
| **CoreWeave** | CKS (managed k8s) | `kubectl apply -k deploy/k8s/overlays/gpu` | `kubectl apply -k deploy/k8s` + your vLLM Deployment |
| **Lambda Cloud** | On-demand GPU VM, or their managed k8s | VM: `install.sh` + a systemd unit running `fak serve … --engine inkernel --backend cuda`, or `docker run --gpus all`. k8s: the GPU overlay | VM/k8s: static `fak` in front of local vLLM |
| **RunPod** | GPU Pod (container-native) | Launch the `-cuda` image as the pod image with the `serve … --engine inkernel --backend cuda` command; expose 8080 as an HTTP port | Two containers/templates: vLLM + the static `fak` image |
| **Crusoe** | GPU VM, or managed k8s | Same two doors as Lambda (VM: `install.sh`+systemd or `docker --gpus all`; k8s: the GPU overlay) | Static `fak` in front of local vLLM |
| **Vast.ai** | Marketplace GPU instance (Docker-native, most heterogeneous) | Launch the `-cuda` image as the instance image, map 8080 — ship the **default all-arch build**, since the card changes with every bid | Static `fak` in front of local vLLM |
| **Nebius** | GPU VM, or managed k8s | Same two doors as Lambda | Static `fak` in front of local vLLM |

### Per-provider verification status — read before you quote a row

Every recipe above is derived from the committed manifests, **not** from a run on that
provider. No provider below has an end-to-end witness yet (a gateway on that provider's
GPU answering an adjudicated request), so every row is `not yet` — a dogfood path to run
and report, not a verified claim. Parent epic: [#1678](https://github.com/anthony-chaudhary/fak/issues/1678).

| Provider | End-to-end witness | What is actually established | Provider-side prerequisite `fak` does not install |
|---|---|---|---|
| **CoreWeave** | `not yet` | The GPU overlay applies as a self-contained stack; CKS is standard k8s + GPU nodes | NVIDIA device plugin / GPU Operator on the pool (CKS ships it); a `.gguf` on the `fak-weights` claim |
| **Lambda Cloud** | `not yet` | `install.sh` + systemd and `docker run --gpus all` are the repo's documented VM paths | NVIDIA driver + Container Toolkit on the VM image |
| **RunPod** | `not yet` | Container-native: takes an image + command, which is what both shapes need | A pod template exposing 8080 as an HTTP port |
| **Crusoe** | `not yet` | Same two doors as Lambda; no Crusoe-specific step is known to be required | NVIDIA driver + Container Toolkit (VM) or device plugin (k8s) |
| **Vast.ai** | `not yet` | Marketplace instances are Docker-native | Nothing extra with the default all-arch image; a narrowed `CUDA_ARCH` build must match the card you win |
| **Nebius** | `not yet` | Same two doors as Lambda | NVIDIA driver + Container Toolkit (VM) or device plugin (k8s) |

No provider here requires a non-default `runtimeClass` as far as the manifests show; if
yours does, that is a gap in this page, not in your cluster — file it against #1678.

**CoreWeave (CKS).** Managed Kubernetes with GPU nodes; the device plugin ships with the
pool. Both shapes are pure `kubectl apply -k` — the GPU overlay for in-kernel, the proxy
base for a co-located engine. Confirm `nvidia.com/gpu` is allocatable before you apply:
`kubectl get nodes -o json | jq '.items[].status.allocatable'`.

**Lambda Cloud.** Two doors. On a bare on-demand GPU **VM**, install the binary with the
repo's `install.sh`, drop a systemd unit that runs `fak serve … --require-key-env …`, or
skip systemd and `docker run --gpus all` the CUDA image. On Lambda's **managed k8s**, use
the GPU overlay exactly as CoreWeave.

**RunPod / Vast.ai.** Container-native marketplaces — you hand them an image and a
command, not a VM to configure. Point the pod/instance at your `-cuda` build with the
`serve … --engine inkernel --backend cuda` args and expose port 8080. On Vast.ai the
rented card's arch is whatever you win the bid on — consumer RTX 50xx (`sm_120`)
included — so this is the one place to ship the **default all-arch image** rather than
a narrowed `CUDA_ARCH` build.

**Crusoe / Nebius.** GPU VMs or managed k8s — identical to the Lambda doors: VM →
`install.sh`+systemd or `docker --gpus all`; k8s → the GPU overlay.

## Always set the auth floor

A rented box is a public IP. Every recipe above carries `--require-key-env` and a policy
JSON for a reason: a network-facing `fak serve` with no key and no policy is an open tool-
call relay. `/healthz` is the only route that answers unauthenticated (it drives the
probes). TLS terminates at your Ingress / load balancer — `fak` speaks plain HTTP behind
it. The full auth/policy/binding checklist is in the
[deployment guide](deployment-guide.md); do not ship without it.

## Status, promotion evidence, and one assumption to watch

This page ships as **gen/next** — a near-term foundation, honest about what is witnessed
today versus what still needs a live run.

- **Witnessed at the artifact level.** `Dockerfile.cuda` (#2661) builds the in-kernel
  image, and the GPU overlay (#2662) applies as a self-contained k8s stack. Both are in
  the tree and their recipes above are copied from the manifests, not invented.
- **Not yet witnessed end-to-end.** No per-provider "fak decoding on a rented GPU on
  provider X" run is captured here yet, and the versioned `-cuda` image is not yet
  published. Treat every provider recipe as a **dogfood path**: run it, and report the
  result back so the row can earn promotion evidence.
- **Promotion evidence (what moves this toward gen/now):** a published `<version>-cuda`
  tag from `release-cuda-container.yml`, plus at least one witnessed end-to-end run per
  provider (a gateway answering an adjudicated request from a `.gguf` on that provider's
  GPU). Demotion/retirement evidence: if the neo-cloud in-kernel path is superseded by
  the binding-layer control plane in the
  [reference architecture](https://github.com/anthony-chaudhary/fak/blob/main/docs/vendor/neo-cloud-reference-architecture.md), this quickstart
  folds into that page instead of standing alone.
- **Invalidating assumption to watch:** that the **default all-arch build** is the right
  thing to hand a renter. It buys card-independence (five cubins + a `compute_120` PTX
  floor) at the cost of build time and image size, and nobody has measured that cost on a
  rented box — where you often build *on* the metered instance itself. If compiling five
  cubins turns out to dominate the time-to-first-token on a fresh rental, the guidance
  above flips: narrow to the bid's `CUDA_ARCH` and accept the per-card rebuild.

## See also

- [Deployment guide](deployment-guide.md) — the generic `fak serve` production mechanics
  (Docker, Compose, Kubernetes, bare metal; the auth/policy/binding checklist).
- [Clouds & hosted providers](../supported/clouds.md) — the opposite shape: fronting a
  hosted model API.
- [Neo-cloud reference architecture](https://github.com/anthony-chaudhary/fak/blob/main/docs/vendor/neo-cloud-reference-architecture.md) — the
  binding layer for exposing heterogeneous accelerator backends.
- [GPU forward pass](../../GPU.md) — what the in-kernel CUDA decode actually runs, with
  the honest gap to llama.cpp.
