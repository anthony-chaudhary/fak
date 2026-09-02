---
title: "fak FAQ — The in-kernel model engine"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# The in-kernel model engine

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

When the configured upstream is the real Anthropic API, the `/v1/messages` route forwards the client's original request bytes byte-for-byte and authenticates with the client's own `x-api-key`, a transparent hop. This passthrough preserves the `cache_control` prefix, so a real upstream cache hit reaches the client's `cache_read_input_tokens` accounting. The kernel boundary still runs: proposed tool calls are adjudicated and inbound results screened, but the downstream request body itself is not re-serialized in this anthropic-to-anthropic case. Note `max_tokens` is required on the `/v1/messages` wire, unlike the OpenAI surface.

## The in-kernel model engine

The optional in-process engine that loads a GGUF and runs the forward pass inside the kernel — a bit-exact correctness reference, with its honest scope stated plainly.

## What is the in-kernel model engine?

The in-kernel model engine is a from-scratch, pure-Go transformer forward pass that loads a GGUF or safetensors checkpoint directly into the process address space and runs decode in-process. It is a correctness reference, not a hardened production serving engine, so its load-bearing claim is bit-exact and argmax-exact agreement with a HuggingFace oracle rather than throughput. It ships as the `inkernel` engine (the default), where an allowed tool call is completed by a real greedy decode over the kernel-owned KV cache; with no real weights loaded it builds a tiny deterministic synthetic checkpoint so CI runs offline. Reach for a tuned engine like vLLM, SGLang, or llama.cpp when you need serving-grade tokens per second.

## Why does fak own a model engine at all if it isn't trying to be fast?

`fak` carries its own engine so the KV cache can be a kernel-owned Go object instead of a tensor pool rented behind a serving engine's HTTP boundary. Owning the cache as a plain data structure is what makes provable span eviction and cross-session splice real operations: when a result is quarantined, the kernel can physically evict that span and the model becomes mechanically incapable of attending to it, verified byte-identical to never having seen it at `max|Δ| = 0`. The engine exists to make that boundary demonstrable end-to-end, not to win raw throughput; for production tokens you front a real engine.

## What exactly does "bit-exact vs a HuggingFace oracle" prove, and what is still unproven?

It proves that, on Llama-family weights, `fak`'s forward pass matches the HuggingFace reference to the last bit: on SmolLM2-135M the hidden-state cosine is 1.000000 at every checked layer, the argmax matches at every position, and the final-logit `max|Δ|` is about 4.4e-5. That parity is currently witnessed green for Llama only. Non-Llama families route through the same oracle harness but skip for want of on-node fixtures, so cross-family parity is honestly un-witnessed; real-GGUF-weight end-to-end parity is also open; and `fak`'s greedy decode of Qwen3.6-27B is refuted, diverging from llama.cpp at the third token from accumulated f32 drift. First-token parity holds there, multi-token continuation does not.

## How does the engine load a GGUF file into the kernel's address space?

The GGUF loader is a read-only parser that maps the checkpoint, normalizes GGUF tensor names to the canonical HuggingFace-Llama naming, and then chooses a resident representation. The exact/f32 loader dequantizes supported F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K, and Q3_K blocks to f32 before the model runs. The lean serving loader instead keeps big matmul weights as resident Q8_0 tensors and drops their f32 copies; small tensors and f32-sensitive state remain f32. The resident-Q4_K path keeps eligible Q4_K tensors native and routes the rest through Q8. Layout and dequant correctness are proven on synthetic fixtures; end-to-end HuggingFace-oracle parity of real GGUF weights is gated behind an opt-in smoke flag and skips on the build box, so treat it as open. A safetensors path also exists, reinterpreting little-endian f32 tensors zero-copy and erroring if a tensor's dtype is not f32.

## What does the --gguf flag actually do when I run fak serve?

`fak serve --gguf` preloads a GGUF checkpoint at boot into the `inkernel` engine and, with no `--base-url` set, serves `/v1/chat/completions` and `/v1/messages` directly from the in-kernel model using the GGUF's embedded tokenizer. The load mode is explicit: the host default is the lean-Q8 profile, `FAK_Q4K=1` selects the resident-Q4_K path, and `--backend` selects a device path. A device backend that advertises quantized `UploadDtype` uses mixed precision: Q8 resident weights with f32 activations and KV rows. A backend without quantized upload falls back to f32 resident weights. `--cpu-offload-experts` uses the same Q8 device representation for dense/device weights while keeping experts host-resident. You can also pass an `hf://` URI and `fak model load` resolves it to a locally cached file with checksum verification. The engine is a correctness reference, so prefer fronting a real server with `--base-url` for production serving; the `--gguf` path is for self-host correctness and the cache-reuse wins, not throughput.

## Is the in-kernel model what serves my chat responses by default?

Not unless you explicitly load real weights; by default the `inkernel` engine builds a small deterministic synthetic checkpoint so the kernel and CI run with no model export. That synthetic model is a 3-layer byte-level map with no natural-language tokenizer that decodes a fixed sixteen tokens, so it is not a chat surface; it exists to prove the kernel wiring at the tensor layer. To serve real generations you load weights via `FAK_MODEL_DIR` or `fak serve --gguf`, which run through the identical dispatch path. If you instead set `--base-url`, the gateway proxies an upstream provider and the in-kernel engine is not in the generation path at all.

## Why is the forward pass written in deliberately slow scalar Go?

The primitives are intentionally scalar and in-order so the f32 bit-exact correctness rungs survive across architectures and call sites. The RMS-norm uses a serial sum-of-squares that must not be reordered, the matmul and dot products run in fixed order, and float32 casts pin the RoPE rotation against fused-multiply-add so it stays bit-identical everywhere. Faster approximations like `fastExp32` and `fastSilu` exist but are used only by the Q8 decode path, never by the exact f32 serial-equivalence path. This is a correctness-first design choice and a direct reason the engine is not a throughput contender.

## Does the compute HAL let me run GGUF-quantized weights on a GPU backend?

Yes, on the quantized-upload path: a backend that advertises `UploadDtype` can consume Q8_0 resident weight tensors, and `fak serve --gguf --backend` uses that path instead of forcing the checkpoint through f32 resident weights. Be precise about the claim: this is mixed precision, not pure int8 inference. Resident weights are Q8 where the backend supports it, while activations, logits, and HAL KV rows remain f32. The legacy exact path and f32-only backends still fetch/upload f32 weights, and a quant-only manifest still fails if you route it through that f32 fetch path. The default `cpu-ref` backend remains the scalar pure-Go reference held to `max|Δ| = 0`; device backends register as correctness-witnessed `Approx` peers, not as the default engine.

## What do the GPU backends actually prove, and do they make fak faster?

The GPU backends prove numerical correctness, not serving-grade speed, and several are slower than llama.cpp. The `gpucheck` witness loads a real f32 safetensors checkpoint, decodes the same prompt on the pure-Go f32 reference and through the HAL on a device backend, and asserts the two greedy token streams agree. On the record: AMD Vulkan is argmax-exact but roughly 58× slower than llama.cpp CPU at f32; NVIDIA CUDA on a small model that fits reaches a single-stream dead-even with llama.cpp Q8_0 but at f32, which is four times the bytes, and large-model parity is not claimed; Apple Metal is argmax-exact with throughput explicitly not yet claimed. These are correctness peers, so claiming throughput parity would be false.

## How does the engine connect a quarantined result to actually evicting it from the model's attention?

Because the KV cache is a kernel-owned Go structure, one detector verdict drives two enforcement media: the context-MMU bars the poisoned bytes from text context, and the KV-MMU bars the corresponding K/V span from attention state. The cache keeps pre-RoPE keys, so removing a span from the middle re-derives each survivor's key in a single clean rotation at its new position, leaving the kept sequence byte-identical to never having seen the evicted span. This bridge is proven bit-exact on a synthetic model in `internal/kvmmu` and is honestly not yet wired into the live `fak agent` HTTP loop; the real-weights numerics are proven separately by the `internal/model` oracle. It is the durable, hard-to-commoditize leg: prefix-cost wins erode as hardware loosens, but provably removing a span and proving it is gone does not.

## When should I use the in-kernel engine versus fronting a real serving engine?

