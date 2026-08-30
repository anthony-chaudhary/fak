---
title: "Go inference runtimes on GitHub — execution-boundary inventory"
description: "A 2026-08-28 source-pinned inventory separating native Go execution, Go-governed native backends, binding frameworks, and research implementations."
---

# Go inference runtimes on GitHub

**Verdict:** the useful Go inference field is not one category. This inventory found **23
source-pinned repositories** across four execution-boundary tiers. Eight materially execute model
or tensor work as Go, three use Go to govern foreign native engines, nine are bindings or
backend-selecting frameworks, and three are representative research implementations. “No CGo” is
not sufficient evidence of native Go computation: `purego` projects still load ONNX Runtime or
llama.cpp, and one static-Go runtime is generated from llama.cpp through Wasm.

Machine authority:
[`go-inference-runtimes-github-2026-08-28.json`](go-inference-runtimes-github-2026-08-28.json).
Tracked by [#9846](https://github.com/anthony-chaudhary/fak/issues/9846). Observed at the
inventory cutoff `2026-08-28T20:30:00Z`; revisions, not moving branch names, are the evidence
anchors.

## How to read the tiers

- **Native Go engine** means model or tensor execution materially runs as Go code. It does not
  imply that every driver or accelerator layer is Go, and generated/transpiled provenance is
  stated explicitly.
- **Go-governed native backend** means Go owns loading, serving, lifecycle, scheduling, or
  caching while a declared native backend owns tensor math.
- **Go binding/framework** means an embeddable Go API, importer, or pipeline selects or calls an
  execution backend. The backend may be pure Go, CGo, `purego`, XLA, libtorch, or ORT.
- **Research demo** means a reproducible model-specific implementation whose main value is
  educational or architectural rather than a broad supported runtime.

## Inventory

| Repository @ pinned revision | Tier | Where inference actually executes | Scope and current posture |
|---|---|---|---|
| [`ollama/ollama@f96e7aa`](https://github.com/ollama/ollama/tree/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a) | Go-governed native backend | Go defines model graphs and orchestration; dynamically loaded MLX and the separate llama.cpp path own tensor execution. | Broad production local-LLM product; active, MIT. Hybrid reference, not pure-Go kernels. |
| [`mudler/LocalAI@0cdc31d`](https://github.com/mudler/LocalAI/tree/0cdc31dcb30cf1464a7402b2065e4c9bc3fe3646) | Go-governed native backend | Go owns API/configuration, lifecycle, coalescing, RPC and supervision; C++, Rust, Python, remote, or native-library workers execute models. | Very broad text/media/backend marketplace; active, MIT plus backend licenses. HEAD moved after cutoff. |
| [`ardanlabs/kronk@6fadebd`](https://github.com/ardanlabs/kronk/tree/6fadebdfeb9e8c40dd3fd237fca2e8253cfcf51e) | Go-governed native backend | Go owns embeddable APIs, lifecycle, batching and queues; llama.cpp, whisper.cpp and stable-diffusion.cpp bindings own compute. | Multimodal embedded/server runtime; active, Apache-2.0 plus dependency licenses. |
| [`born-ml/born@b1237ec`](https://github.com/born-ml/born/tree/b1237ec5135d1108f56af6dfd5068638068ad8d3) | Native Go engine | Go/assembly CPU backend and Go-owned WGSL through WebGPU; Go LLaMA graph and tensor/autograd runtime. | GGUF, SafeTensors, partial ONNX; emerging Apache-2.0 framework. |
| [`townsendmerino/goinfer@bf6bdc3`](https://github.com/townsendmerino/goinfer/tree/bf6bdc3afc633ac8c118a7a982fc00c4ff8d2bdc) | Native Go engine | Go CPU decoder, loaders, quantization and fallback; optional cgo-free CUDA/PTX, Metal/purego and WebGPU. | Broad decoder-model coverage; active but only eight stars at cutoff, MIT. Treat performance claims as unwitnessed. |
| [`rostamlabs/rembed@19d673b`](https://github.com/rostamlabs/rembed/tree/19d673b357bd8c244aa1a6d17d83769224fa98c9) | Native Go engine | CPU Go/assembly encoder, attention, pooling and SIMD tensor kernels. | Focused SafeTensors/HF embedding runtime; active-emerging, Apache-2.0. |
| [`magomedcoder/gogguf@c157545`](https://github.com/magomedcoder/gogguf/tree/c157545050743f73356da7ea9db7532e9129013e) | Native Go engine | Pure-Go quantized CPU model paths; optional CUDA path uses CGo. | Broad GGUF quantization/model claims, no release and four stars; experimental MIT reference. |
| [`nlpodyssey/spago@3130dda`](https://github.com/nlpodyssey/spago/tree/3130dda657565c0b6b8e494070d8c35a037eee4e) | Native Go engine | Self-contained Go/assembly CPU tensor, autograd and neural layers. | Mature custom ML/NLP substrate, but paused and without ready-to-use support for current LLM formats. BSD-2-Clause plus assembly provenance. |
| [`gorgonia/gorgonia@d7a3ce2`](https://github.com/gorgonia/gorgonia/tree/d7a3ce27c9a1ffbee531d7850485229584db8d97) | Native Go engine | Go graph compiler and VMs; CPU Go/tensor/BLAS, optional CUDA/cuDNN/cuBLAS integration. | General differentiable graphs, not a modern pretrained-LLM loader; maintenance watch, Apache-2.0. |
| [`zerfoo/zerfoo@a6c9dcb`](https://github.com/zerfoo/zerfoo/tree/a6c9dcbdc0117f929269c4d90bf4204c74afa372) | Native Go engine | Go GGUF/model/generation/CPU path plus repository-owned CUDA and Metal source. | Extremely broad active claims but five-star adoption; Apache-2.0, independent verification required. |
| [`goccy/go-llama@6b608ae`](https://github.com/goccy/go-llama/tree/6b608ae6947a4f6a27b2f05f85cb0c56ac3253b9) | Native Go engine, translated provenance | llama.cpp is compiled to Wasm and translated/generated into Go; no CGo or Wasm runtime remains. | Static CPU GGUF runtime with wasm32/4 GiB constraints; active-emerging, MIT. Not Go-authored kernels. |
| [`gomlx/gomlx@f847c57`](https://github.com/gomlx/gomlx/tree/f847c57c9a4e10c1c66c737bbc44625fd4d1e538) | Go binding/framework | Go graph/model layer selects a portable pure-Go backend, OpenXLA/PJRT, or ONNX Runtime. | Broad custom ML with HF/ONNX companions; active Apache-2.0 framework. Classify results per selected backend. |
| [`knights-analytics/hugot@e085914`](https://github.com/knights-analytics/hugot/tree/e08591494e5383c534e1c3cef92ac9d766e461b0) | Go binding/framework | Go pipelines select native ORT, native XLA, or GoMLX’s pure-Go operator backend. | HF-style inference pipelines; active Apache-2.0. The Go route has narrower operator coverage. |
| [`yalue/onnxruntime_go@8ce7fa5`](https://github.com/yalue/onnxruntime_go/tree/8ce7fa5882bb3ac038bacdd53bfe02da3691899f) | Go binding/framework | A CGo shim loads ONNX Runtime; ORT owns graph optimization and computation. | Principal current general ONNX binding in this corpus; active MIT wrapper plus ORT provenance. |
| [`sugarme/gotch@db66155`](https://github.com/sugarme/gotch/tree/db661550e5a88582a40830e54bb1777e6afd32e7) | Go binding/framework | CGo enters libtorch; PyTorch C++ owns tensor/model computation. | TorchScript/PyTorch and companion transformer models; maintenance watch, Apache-2.0 wrapper. |
| [`oramasearch/onnx-go@5befeb8`](https://github.com/oramasearch/onnx-go/tree/5befeb870169e8cd10af28c1552ed14662f82390) | Go binding/framework | Go decodes ONNX into a pluggable backend; the backend, commonly Gorgonia, executes it. | Format/import bridge rather than standalone executor; maintenance watch, MIT. |
| [`hybridgroup/yzma@c4fed78`](https://github.com/hybridgroup/yzma/tree/c4fed7865c4c5cb116d7ae9105a1765bc0398803) | Go binding/framework | `purego` loads llama/ggml/mtmd; llama.cpp owns loading and computation. | Strong current no-CGo llama.cpp binding; active, Apache-2.0 plus MIT portions/upstream provenance. ABI drift is the core risk. |
| [`go-skynet/go-llama.cpp@6a8041e`](https://github.com/go-skynet/go-llama.cpp/tree/6a8041ef6b46d4712afc3ae791d1c2d73da0ad1c) | Go binding/framework | CGo enters binding C++; an older pinned llama.cpp tree owns inference. | Important historical high-level binding; legacy-maintenance, MIT plus upstream provenance. |
| [`benedoc-inc/onnxer@d3f86f8`](https://github.com/benedoc-inc/onnxer/tree/d3f86f897e42e0f5544494d62bfe5d8ead6370fd) | Go binding/framework | `purego` loads ORT/ORT GenAI; native libraries own computation. | Emerging versioned multi-runtime design; four-star fork with mixed file-level license signals. Inspire-only/watch. |
| [`dianlight/gollama.cpp@6376d0d`](https://github.com/dianlight/gollama.cpp/tree/6376d0d4e0a2d59f6f048a7c1ca121a5c1129604) | Go binding/framework | `purego` resolves native llama/ggml symbols; llama.cpp owns tensor/decode computation. | Active lineage reference incorporated in part by Yzma; MIT plus upstream provenance. |
| [`nikolaydubina/llama2.go@836881d`](https://github.com/nikolaydubina/llama2.go/tree/836881ddaeb0d96d8c11efe3cd59dbc5a8ea15d2) | Research demo | Pure-Go CPU float32 Llama-2/llama2.c forward path. | Archived, MIT, useful as a compact model-specific reference. |
| [`adalkiran/llama-nuts-and-bolts@ccbe25e`](https://github.com/adalkiran/llama-nuts-and-bolts/tree/ccbe25e9fdc0f2ca80c6a5de9d7eb8c82554bf89) | Research demo | Pure-Go CPU/goroutine implementation over official Llama 3.1 weights. | Strong educational end-to-end Llama-3 reference; stable Apache-2.0 project. |
| [`divy-sh/llama3-go@a68a81f`](https://github.com/divy-sh/llama3-go/tree/a68a81fe918bd20f43cce845cdcc362b16bce236) | Research demo | Pure-Go CPU float32 GGUF/Llama path. | Experimental, two stars, no tests found, and **no license: DO-NOT-USE**. |

## What the inventory changes

1. **“Written in Go” is not an execution claim.** LocalAI, Kronk, Yzma, Gotch, and the
   ONNX bindings are useful Go runtimes or libraries, but their model math remains in foreign
   native engines.
2. **“No CGo” is also not an execution claim.** Yzma and Onnxer use dynamic FFI; they remove
   build-time CGo without moving computation into Go.
3. **Native Go now spans more than teaching code.** Born, goinfer, rembed, GoGGUF, Zerfoo,
   Gorgonia and Spago contain real Go execution. Their maturity, model breadth and independent
   performance evidence vary sharply, so the inventory does not rank them.
4. **Provenance is orthogonal to runtime language.** goccy/go-llama executes translated Go but
   derives the engine from llama.cpp. That is meaningfully different from both a dynamic
   llama.cpp binding and a Go-authored tensor runtime.
5. **Backend-selecting frameworks need per-route receipts.** GoMLX and Hugot can take a native
   Go path or a foreign accelerated path. A benchmark or support claim that omits the selected
   backend is incomplete.

## Discovery coverage and exclusions

The dated search ledger covered Go + ONNX, tensor inference, Llama inference, ML frameworks,
PyTorch bindings, deep-learning/machine-learning topics, pure-Go neural networks, GGUF, ONNX
Runtime, and TensorFlow Lite. Each surviving family was then pinned and read through code,
tests, license/provenance, releases/history, and material issue/PR state. Exact query strings and
reported result counts are in the machine file.

Explicitly excluded were remote API clients, agent/RAG frameworks, Kubernetes operators,
application-specific model consumers, duplicate model libraries that introduce no new execution
boundary, and downstream products such as CSG Lite, QuenchForge, and Docker Model Runner.
Companion modules (`gomlx/go-huggingface`, GoMLX compute backends, `nlpodyssey/cybertron`,
`sugarme/transformer`) are attributed to their parent execution family instead of double-counted.

GitHub search can miss Go subtrees in non-Go-primary monorepos and newly created or poorly
described repositories. The completeness claim is therefore family-level: no material execution
boundary found in this pass remains unopened. It is not a claim that every GitHub repository was
enumerated.

## Evidence fences

- This is an inventory, **not a performance or quality ranking**. Star counts are snapshot
  discovery metadata only.
- Source model-support and performance statements are not converted into fak claims. A runtime
  needs a separate matched model, format, hardware, quality, and operating-envelope witness.
- Licenses were read for classification, but this is not legal advice. `DIRECT-PORT` or `ADAPT`
  still requires exact-file provenance and obligation review at implementation time.
- The fak-native invariant remains unchanged: foreign runtimes are benchmark, parity,
  interoperability, or borrowing references unless explicitly selected; they are not silent
  fallback engines for fak-native work.
- Re-pin moving branches before acting. LocalAI demonstrated the need during this pass by moving
  after its studied revision was captured.
