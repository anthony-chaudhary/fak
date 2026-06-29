# GPU smoke example output

These are hardware-captured result excerpts already committed in the repo, reshaped
to match what the smoke scripts print. They are not a fresh run from this checkout.
Run `run-vulkan.sh` or `run-cuda.sh` on the target machine to produce local logs under
`.cache/gpu-smoke/`.

## AMD / Vulkan, Radeon RX 7600

Source: [VULKAN-AMD-RESULTS.md](../../docs/benchmarks/VULKAN-AMD-RESULTS.md) and
[CLAIMS.md](../../CLAIMS.md#engine).

```text
gpu-smoke(vulkan): hf=/path/to/HuggingFaceTB/SmolLM2-135M-Instruct/snapshot

== argmax-exact witness ==
OK - greedy argmax-exact over 10 tokens vs f32 legacy
gpu-smoke(vulkan): argmax-exact PASS vs cpu-ref

== decode throughput ==
[fak] decode: 394 ms/tok (2.5 tok/s)
gpu-smoke(vulkan): decode_tok_per_sec=2.5 per_token_median_ms=394 report=.cache/gpu-smoke/vulkan/modelbench-vulkan.json
```

Honest read: this proves the AMD Vulkan backend runs the model correctly on the
RX 7600. It is not throughput parity. The Vulkan writeup records
prefill-logit cosine 1.0 and argmax-exact decode, while throughput remains far
behind llama.cpp depending on precision and baseline.

## NVIDIA / CUDA, RTX 4070 Laptop

Source: [GPU.md](../../GPU.md) section 3b and
[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md).

```text
gpu-smoke(cuda): hf=/path/to/HuggingFaceTB/SmolLM2-135M-Instruct/snapshot
gpu-smoke(cuda): FAK_CUDA_GRAPH=1

== argmax-exact witness ==
OK - greedy argmax-exact over 10 tokens vs f32 legacy
gpu-smoke(cuda): argmax-exact PASS vs cpu-ref

== decode throughput ==
[fak] decode: 8.3 ms/tok (119-120 tok/s)
gpu-smoke(cuda): decode_tok_per_sec=119.8 per_token_median_ms=8.3 report=.cache/gpu-smoke/cuda/modelbench-cuda.json
```

Honest read: the documented CUDA graph path reaches about 119-120 tok/s on
SmolLM2-135M on the RTX 4070, matching llama.cpp Q8_0 for that model while
running fak f32. It does not claim that every larger model fits or reaches parity.
