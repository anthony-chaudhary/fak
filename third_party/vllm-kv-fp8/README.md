# vLLM FP8 KV mechanism provenance

This bounded adapter targets Fak #13982. It is source preparation, not a
backend capability, serving activation, graph implementation, or hardware result.

Upstream: https://github.com/vllm-project/vllm
Commit: fa3b9d1377219247060d3c8b49eae199c9e7e154
License: Apache-2.0. The pinned root LICENSE is included verbatim. Root NOTICE
at that pin returned 404. Preserve applicable source notices if the adapted
dependency closure is expanded.

Source files and Git blobs:
- csrc/quantization/w8a8/fp8/common.cu
  837948fdd2a7a99ff21ed0592542de925920d7b1
- csrc/quantization/w8a8/fp8/common.cuh
  143be698cc82512ffef558e7fc14c920cfad5e9c
- csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh
  77aabe4651119bf027d003efe98ffd950b4b0db8
- csrc/quantization/utils.cuh
  6bb9b9fc5635ea3aaff63ac4bbb8494b8796830a
- LICENSE
  261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64

Adaptations:
1. A donor row becomes a single (position, KV head, K-or-V plane).
2. Remove PyTorch, CUB, ROCm and paged/slot-mapping dispatch dependencies.
3. Keep F32 amax/448, with scale one for all-zero rows and the least positive
   F32 scale only when division underflows to zero. This deliberately differs
   from the donor's max(amax/448, 1/(448*512)).
4. Reject nonfinite input/results through separate numerical status, never
   hide NaNs through clamping.
5. Use the scalar exact-widen-then-F32-multiply conversion. Do not use the
   donor's scaled half/half2 vector paths, which lose range and precision.
6. Retain F32 attention arithmetic and the existing Fak recurrence; this is
   not FP8 tensor-core attention or a GQA-reuse optimization.
7. The pure Go plane oracle reuses Fak's existing rfqEncodeE4M3 from the
   independent RMSNorm/FP8 reference, whose OCP subnormal grid is 2^-9.

The native fragment is not included in the production translation unit yet.
Before inclusion, the admitted native build must establish strict no-FTZ,
precise F32 division/multiply flags; defining the marker alone is insufficient.
The coherent five-plane owner/status protocol, full ingress preflight for
borrowed views, rotary binding and staged eviction must be implemented and
independently reviewed. Graph capture remains refused. Do not declare a
realized tier from the enum, oracle, native source or modeled bytes alone.
