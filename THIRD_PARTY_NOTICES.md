# Third-party notices

## JigSawPT expert cache ownership

`internal/model/expert_checkpoint_tier.go` adapts loading-entry ownership,
complete publication and failed-fill rollback from `src/llama-moe-stream.cpp`
(lines 205–245, 267–280, 751–778) and `src/llama-moe-stream.h` (lines 243–251)
in JigSawPT/llama.cpp at revision
`8ce0968539b1a6795f3413c2783dfa5646b4688a`:
https://github.com/JigSawPT/llama.cpp/tree/8ce0968539b1a6795f3413c2783dfa5646b4688a

Changed for Fak: immutable weight buffers/views, per-key waiting callers and the existing
bounded residency pool. Upstream treats a loading entry as an independent-read
miss; sharing that read is the Fak adaptation. No upstream device scheduler is
included. The upstream MIT license follows:

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## TensorFold benchmark evidence

`internal/perfledger/bench_evidence.go` adapts `tools/bench_evidence.py` from
TensorFold at revision `a6aa4782bf4ffe87ccee4a0a5cbf2efb2dd01cba`:
https://github.com/ashhart/TensorFold/blob/a6aa4782bf4ffe87ccee4a0a5cbf2efb2dd01cba/tools/bench_evidence.py

Changed for Fak: Go receipt types, strict JSON count parsing, and integration into
the existing gateway performance ledger. No original Python benchmark client or
third-party kernel/model code is included.

The upstream NOTICE is reproduced below. Its referenced files are upstream files.

```text
TensorFold
Copyright 2026 TensorFold contributors
https://github.com/ashhart/TensorFold

TensorFold is licensed under the Apache License, Version 2.0 (LICENSE) from 0.6.0.
Releases up to 0.5.0 were MIT-licensed, and code written before 0.6.0 keeps its
MIT notice in LICENSES/MIT.txt. Third-party code and its licenses are listed in
THIRD_PARTY_NOTICES.md.
```

The full upstream LICENSE at the pinned revision follows:

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```


## Score-aware expert-cache eviction policies

`internal/deepseekv4moe/cache_policy_bench.go` `CachePolicyLFRU` (`replayOnlineLFRU`,
`pickLFRUVictim`) reimplements the LFRU victim rule (`score = freq / age`, lower score
evicted) from `tinyserve/cache_policy.py` `LFRUPolicy` at revision
`e1n00r/tinyserve@9277e683` (branch `master`):
https://github.com/e1n00r/tinyserve/blob/master/tinyserve/cache_policy.py

`CachePolicyScoreEMA` (`replayOnlineScoreEMA`, `pickScoreEMAVictim`) ports the
`S = alpha * TopP(s) + (1 - alpha) * S` score-EMA update and lowest-score eviction from
`KScoreAwareCache` in `ktransformers/operators/experts.py:662-712` at revision
`PKU-SEC-Lab/HybriMoE@508a85b9a43dff37124370d8816e411e74827803`:
https://github.com/PKU-SEC-Lab/HybriMoE/blob/508a85b9a43dff37124370d8816e411e74827803/ktransformers/operators/experts.py

Changed for Fak: the policies are reimplemented over the package's offline
unit-group access stream and its deterministic tie-break (`groupLess`), not over
a torch tensor store or the upstream ring. The optional `CachePolicyStream.Scores` and `fak expert-trace policy` use selected
route weights emitted by Fak's router. These are selected expert weights, not
HybriMoE's full-distribution top-p scores. Fak decays all tracked scores once per flattened
expert access, whereas upstream updates one layer's selected scores together.
An omitted score array preserves the prior access-indicator (1.0) experiment;
`alpha` remains 0.5. The CLI requires scores and labels its capacity as global
uniform expert slots. No serving eviction policy is changed. No upstream storage, scheduler,
kernel or test fixture is copied.

The HybriMoE Apache-2.0 license follows:

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The tinyserve MIT license follows:

```text
MIT License

Copyright (c) 2026 tinyserve contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Expert route identity normalization

`internal/model/expert_route_trace_analyze.go` adapts the grouped distinct-expert
request contract from Mixtral-offloading at revision
`ce545188b804238f0b23a59fc45e6a8f8b390c40`:
https://github.com/dvmazur/mixtral-offloading/blob/ce545188b804238f0b23a59fc45e6a8f8b390c40/src/custom_layers.py#L292
https://github.com/dvmazur/mixtral-offloading/blob/ce545188b804238f0b23a59fc45e6a8f8b390c40/src/expert_cache.py#L134

The equivalent set contract was compared with MoE-Infinity at revision
`0f96d563db2dda5f1834bf667f6fcc44f45b2b9d` (Apache-2.0):
https://github.com/EfficientMoE/MoE-Infinity/blob/0f96d563db2dda5f1834bf667f6fcc44f45b2b9d/moe_infinity/spec_decode/_prefetch_route.py#L52
https://github.com/EfficientMoE/MoE-Infinity/blob/0f96d563db2dda5f1834bf667f6fcc44f45b2b9d/LICENSE
No upstream NOTICE file appears in either pinned source tree.

Changed for Fak: reuse the existing order-preserving Go `dedupeInts` helper in
offline route export and inferred TopK. Later invocations remain separate accesses.
No upstream cache, prefetch implementation, dependency or test fixture is copied.
The Mixtral MIT copyright and permission notice follows:

```text
MIT License

Copyright (c) 2023 Artyom Eliseev, Denis Mazur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## ggml GGUF type traits

The supported GGUF block geometry table in `internal/ggufload/gguf_type_traits.go`
is translated from `ggml/src/ggml.c` (`type_traits` and `ggml_get_type_traits`) at
llama.cpp revision `83078fec0db82d6b5a00d9599062c38c39145755`:
https://github.com/ggml-org/llama.cpp/blob/83078fec0db82d6b5a00d9599062c38c39145755/ggml/src/ggml.c
Block layout definitions are in `ggml/src/ggml-common.h` at the same revision.
The translation omits function pointers and retains the loader's existing local
format geometry, including its 128-element Q2_0 layout.

The V4.1 activation comparison and deliberate-divergence controls also adapt
the backend graph node-comparison and sentinel process from
`tests/test-backend-ops.cpp` at revision
`fd55cb98fe29d3df0054d3f3e6e6f15958ea72ab`:
https://github.com/ggml-org/llama.cpp/blob/fd55cb98fe29d3df0054d3f3e6e6f15958ea72ab/tests/test-backend-ops.cpp
That upstream source uses the same MIT license below.

`fak-dev kernel-oracle` (`cmd/fak-dev/kernel_oracle_parse.go`) runs no ggml code: it
parses the console perf rows of an externally built `test-backend-ops` binary
(`print_perf_console`, `test_generic_op` parameter strings and `--test-file`
input written by `tests/test-export-graph-ops.cpp`), as read at revision
`dff600487` (2026-09-09).

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DeepSeek-V3 / JoyAI pre-tokenizer and Unicode classification

The DeepSeek-V3 / JoyAI regex sequence in
`internal/tokenizer/pretok_deepseek3.go` is adapted from
`src/llama-vocab.cpp` and `src/unicode.cpp` in vcruz305/llama.cpp at
revision `5210c7c5ed61dddaee6ed476623abf4b63093d16`.
The supplementary letter and symbol classification ranges, and their
source-derived regression expectations, are taken from `src/unicode-data.cpp`
at the same pinned revision:
https://github.com/vcruz305/llama.cpp/blob/5210c7c5ed61dddaee6ed476623abf4b63093d16/src/unicode-data.cpp

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## vLLM offloader pinned-buffer identity and dependency ordering

The checkpoint expert-ring fill in `internal/model/expert_checkpoint_tier.go`
(`ExpertCheckpointTier.Prefetch`, the bounded fill pool and the ascending-offset
dispatch order) and the directory-only read-span accessor
`ggufExpertSource.expertReadSpan` in `internal/model/gguf_expert_source.go` adapt
the static-buffer-pool identity rule and the prefetch/sync dependency-ordering
contract from the vLLM offloader at revision
`f47792af7fba76eb8e8f6a564b35cfad316d39f2`:
https://github.com/vllm-project/vllm/tree/f47792af7fba76eb8e8f6a564b35cfad316d39f2
- `vllm/model_executor/offloader/prefetch.py` (`ParamInfo.key` — a staging buffer
  is keyed by parameter identity (name, shape, stride, dtype) and reused across
  layers; `StaticBufferPool`, `_get_next_prefetch_index`).
- `vllm/model_executor/offloader/prefetch_ops.py` (`wait_prefetch` / `start_prefetch`
  declare a mutated argument so the compiler cannot reorder a wait ahead of the
  copy it guards).
- `vllm/model_executor/offloader/uva.py` (pinned staging storage).

Changed for Fak: the port is expressed over a synchronous `io.ReaderAt` ring fill
rather than a CUDA copy stream. Identity becomes the absolute checkpoint byte
range derived from GGUF directory geometry, which also supplies the dispatch
order; the dependency edge is the existing one-shot in-flight `expertCheckpointLoad`
handoff a demand fault joins; and a fixed 8-worker pool bounds the fan-out a
layer's misses may have instead of one goroutine per miss. No vLLM Python source,
CUDA kernel, or runtime scheduling code is included.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
```

The full Apache License, Version 2.0 text is reproduced in the
"TensorFold benchmark evidence" section above (that section carries the complete
license body shared by both Apache-2.0 notices). Copyright: the vLLM project
contributors, Licensed under the Apache License, Version 2.0.

## llama.cpp mul_mat_vec_id and llama_moe_cache slot map

The V4.1 routed-expert id kernels `internal/compute/shaders/v41_q2k_matvec_id.comp`
and `internal/compute/shaders/v41_q3k_matvec_id.comp`, and the id dispatch in
`internal/compute/vulkan_shim.cpp` (`fvk_v41_matvec_id_f32`), adapt the
ggml-vulkan `mul_mat_vec_id` dispatch shape (grid Y is the selected expert, read
from an ids buffer) from `ggml/src/ggml-vulkan/ggml-vulkan.cpp` and
`ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl` at llama.cpp revision
`ccdbe290533f68154ede46e0618622eebcb6c070`:
https://github.com/ggml-org/llama.cpp/blob/ccdbe290533f68154ede46e0618622eebcb6c070/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl
The optional paired gate/up token kernel
`internal/compute/shaders/v41_q2k_token_pair.comp` reuses the existing
`v41_q2k_token_panel.comp` quantized decode, ordered accumulation and reduction
from this same pinned MIT prior art. Its two workgroup-X banks select separate
weight and output buffers without changing the per-workgroup arithmetic.

The ids carry ring slots remapped through a per-layer slot map, after
`src/llama-moe-cache.h` from llama.cpp PR #29887 (merge
`d6cf9acb25e6c657d6c6a4645ab7b69fce119995`).

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## antirez/ds4 fused routed-FFN kernels

The V4.1 fused routed-FFN kernels `internal/compute/shaders/v41_routed_gate_up_swiglu_id.comp`
and `internal/compute/shaders/v41_routed_down_sum.comp`, the dispatch entrypoints in
`internal/compute/vulkan_shim.cpp` (`fvk_v41_routed_gate_up_swiglu_id_f32`,
`fvk_v41_routed_down_sum_f32`), and the Go shape in `internal/compute/v41_routed_fused.go`
adapt the fused MoE expert kernel shape from `ds4_cuda.cu` at revision
`fc80bd695da76ee14bcb86820bf900a5cdfda806` (MIT):
- `moe_gate_up_mid_decode_lut_qwarp32_kernel:22143` does gate + up + SwiGLU + route weight
  into one `mid` buffer.
- `moe_down_sum_qwarp32_kernel:23869` and `moe_down_q4K_sum6_qwarp32_kernel:24099` do the
  down projection plus the routed expert sum.
- Launch sequence: `:27230-27360`.
https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu

Changed for Fak: no upstream code is copied. The kernels are GLSL reimplementations over the
P3 ids/slot-map binding, one workgroup owns one output row, and the quantization body plus
per-row accumulation order are the project's own `v41_q2k_matvec_id.comp` / `v41_q3k_matvec_id.comp`
term for term so the fused path is ULP- or bit-parity with the unfused id path rather than a
new arithmetic contract. The fused gate/up kernel keeps the id kernel's per-superblock FMA
association; the down/sum epilogue reproduces the id down + BF16 publication + sum chain
bitwise. The ds4 MIT copyright and permission notice follows:

```text
MIT License

Copyright (c) 2025-2026 antirez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DeepSeek V4.1 DSpark residency port study

`internal/model/v41_dspark_residency.go` records a revision-anchored port STUDY of the DSpark
draft-resident / target-streamed speculative-decoding shape. It adapts the shape (and only the
shape) of `python/sglang/srt/models/deepseek_v4_dspark.py` from sgl-project/sglang at revision
`45418d7dc648d0d71501b8b905a2499a40e9b971` (Apache-2.0):
https://github.com/sgl-project/sglang/blob/45418d7dc648d0d71501b8b905a2499a40e9b971/python/sglang/srt/models/deepseek_v4_dspark.py

It also records the draft-resident/target-streamed split and the UPSTREAM-REPORTED code-vs-prose
acceptance figures from evolvemarketingitalia/ds4 at revision
`5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75` (MIT):
https://github.com/evolvemarketingitalia/ds4/blob/5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75/docs/STRIX_HALO_DSPARK.md
https://github.com/evolvemarketingitalia/ds4/blob/5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75/docs/SPECULATIVE_DECODING.md

Changed for Fak: no upstream code is copied. Only the residency split, the census byte shape and
the reported acceptance figures are recorded as typed, fail-closed Go metadata, and the study
refuses a single-node resident claim that does not fit the host budget. No decoder, drafter,
kernel, or verify loop is ported. SGLang is Apache-2.0 and is used as an ADAPT reference only;
its NOTICE obligations are satisfied by the citation above. The evolve ds4 MIT copyright and
permission notice follows:

```text
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## freetoken-mlx adaptive segmented LRU replay

The optional offline `slru` replay arm in `internal/deepseekv4moe/cache_policy_slru.go`
is adapted from `stevezkw1998/freetoken-mlx`, commit
`2d5bd339ab599a36e663ef90a813fc0adc5ad0a9`,
`python/freetoken/mlx_cache.py:45-98`:
https://github.com/stevezkw1998/freetoken-mlx/blob/2d5bd339ab599a36e663ef90a813fc0adc5ad0a9/python/freetoken/mlx_cache.py

Licensed under Apache License, Version 2.0 (upstream `LICENSE`), copyright and
attribution retained from the FreeToken/freetoken-mlx contributors. License text:
https://www.apache.org/licenses/LICENSE-2.0
The upstream source file has no individual copyright header and the acquired
repository has no root NOTICE file. This adaptation replaces Python ordered
maps with Go ordered lists, computes the 10%/80% protected target without integer
overflow, and stores only touched resident groups. It preserves the switch after
more than four full-capacity request windows. It is an offline experiment,
without MLX materialization, serving eviction or a hardware performance claim.

## MLX-LM declared RoPE scaling

The model configuration handling and linear RoPE compatibility in
`internal/model/config.go` and `internal/model/arch.go` adapt the declared
scaling contract in `ml-explore/mlx-lm`, commit
`8d47a50482e27ad972e545f94f7ecb9e73a44635`,
`mlx_lm/models/qwen3_moe.py` and `mlx_lm/models/rope_utils.py:290-307`.
Source: https://github.com/ml-explore/mlx-lm/tree/8d47a50482e27ad972e545f94f7ecb9e73a44635

```text
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Vulkan drained batch command-buffer reuse

`internal/compute/vulkan_shim.cpp` adapts the completed-command reuse pattern from
`ggml/src/ggml-vulkan/ggml-vulkan.cpp` lines 1114–1137 in llama.cpp at revision
`5e03a1ff450cae37a774684de55d14fb48b1f90b`:
https://github.com/ggml-org/llama.cpp/blob/5e03a1ff450cae37a774684de55d14fb48b1f90b/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L1114-L1137

Changed for Fak: one batch-only idle slot, individual command-buffer reset, and
successful-fence-only retention with checked failure quarantine. Restore and
one-shot commands keep their own ownership. No upstream runtime is included.
The upstream MIT license follows.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ggml-vulkan i-quant matrix-vector kernels

The native i-quant decode kernels `internal/compute/shaders/iq1s_matvec.comp`,
`iq2s_matvec.comp`, `iq2xs_matvec.comp`, `iq2xxs_matvec.comp`, `iq3s_matvec.comp`,
`iq3xxs_matvec.comp` and `iq4xs_matvec.comp` port the codebook tables, block layouts
and grid/sign/scale decode from `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`,
`dequant_funcs.glsl` and the `mul_mat_vec_iq*.comp` shaders, and the per-row
`subgroupAdd` reduction from `mul_mat_vec_base.glsl`, at llama.cpp revision
`dff60048744f99cb0af68d02be2314456b3269dc`:
https://github.com/ggml-org/llama.cpp/tree/dff60048744f99cb0af68d02be2314456b3269dc/ggml/src/ggml-vulkan/vulkan-shaders

Changed for Fak: the bindings, push constants and residual/gate-up modes follow
`q2k_matvec.comp`, and each subgroup owns a group of rows and strides over row
groups. No upstream runtime is included. The upstream MIT license follows.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
