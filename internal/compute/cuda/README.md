# CUDA implementation modules

`cuda_kernels.cu` is the sole NVCC compile entry. It owns common includes, error macros,
and process-wide state, then includes these implementation fragments in dependency order:

| Module | Responsibility |
|---|---|
| `runtime_memory.cuh` | initialization, allocator pools, and transfers |
| `runtime_services.cuh` | profiling, device services, and transfer witnesses |
| `model_candidates.cuh` | embedding gather and conditional v4.1 candidates |
| `qwen35_gdn.cuh` | Qwen3.5/3.6 Gated-DeltaNet operations |
| `q4k_mmq.cuh` | Q4_K int8 tensor-core MMQ for prompt panels adapted from llama.cpp (opt-in `mmq` arm) |
| `quant_matmul.cuh` | cuBLAS and native quantized matrix operations |
| `transformer_ops.cuh` | normalization, RoPE, gating, elementwise, and KV writes |
| `fattn_mma_hd256.cuh` | Qwen3.8 hd256 tensor-core prompt attention adapted from llama.cpp (opt-in) |
| `attention.cuh` | decode, flash, and prompt-panel attention |
| `spec_graph.cuh` | speculative/DSA attention, argmax, and graph capture |
| `weight_formats.cuh` | AWQ, GPTQ, and cuBLASLt NVFP4 paths |

The `.cuh` suffix denotes a private implementation fragment, not a standalone or public
header. All fragments intentionally compile as one translation unit. They share `g_stream`,
allocation pools, cuBLAS handles, graph-capture state, counters, static helpers, and
conditional feature state. Include order is therefore part of the implementation contract;
do not compile fragments separately or reorder them without first designing explicit state
ownership, declarations, and linkage.

## Adding or moving an operation

1. Put complete top-level definitions in the narrowest existing module.
2. If a new module is warranted, extract a contiguous block at a natural boundary and add
   its include at the same position in `cuda_kernels.cu`.
3. Preserve function bodies and conditional-compilation scopes. Adjust only nested relative
   include paths required by the `cuda/` directory.
4. Keep the flat ABI declaration in `cuda_backend.h` and its cgo binding in sync.
5. Update source-inspection tests to read the owning module rather than assuming every symbol
   is textually present in `cuda_kernels.cu`.
6. Run the GPU-free `internal/compute` and `internal/computebuild` tests, ABI parity checks,
   and the sanctioned CUDA build and hardware acceptance gates.
