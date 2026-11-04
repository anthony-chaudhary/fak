# Third-party notices

## vLLM native stream boundary collector

`internal/gateway/delta_coalescer.go` adapts the non-merging boundary rule of
`StreamingInputOutputCollector` in `vllm/v1/engine/output_processor.py` at
`e5196e2959a42908313510f55fa21f213a48802e`:
https://github.com/vllm-project/vllm/blob/e5196e2959a42908313510f55fa21f213a48802e/vllm/v1/engine/output_processor.py

Copyright contributors to the vLLM project. SPDX-License-Identifier: Apache-2.0.
Changed for Fak: Go synchronization, bounded pending/in-flight byte and event
budgets, explicit failure/cancellation/join, and serialized native lifecycle SSE
barriers instead of vLLM streaming-input completion boundaries. No Python runtime
or model/backend implementation is included. The Apache-2.0 license reproduced
below for the vLLM paged-attention adaptation also applies to this adaptation.

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

Full token SHA-256 validation and short-prefix consistency are updated from
revision `9e8c368b0b4c2c7892a0c9f0f302d7c164016209` of the same source:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/tools/bench_evidence.py

Changed for Fak: Go receipt types, strict JSON count parsing, full token digest
validation with a compatible legacy constructor, and integration into
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

## TensorFold decode/prefill wall-time share

`internal/agent/inkernel_qwen_coordinator.go` and
`internal/agent/inkernel_v41_coordinator.go` adapt the lane-host fill
scheduling idea from TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/lane_host_fill.zig
plus its operator knobs documented at
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/docs/api.md#prefill-lane-host-fill-scheduling

The adapted contract: while decoders are live, a prompt step taking `t`
nanoseconds owes them `t*(1-p)/p` of decode time (`p` is the prompt's wall-time
share, floored at 0.01); decode rounds pay the debt down; a decode share outside
(0,1) — including 0 — keeps whole-prompt arrival ordering; and an optional row
budget caps the rows a visit fills beside live decoders. In this codebase that
lands as the `inKernelDecodeShare` type plus two operator knobs,
`FAK_INKERNEL_DECODE_SHARE` and `FAK_INKERNEL_PROMPT_ROW_BUDGET`, wired into the
coalescer admission paths of both in-kernel decode cohorts.

Changed for Fak: idiomatic Go re-expression of the scheduling contract over
fak's prepare/decode visits instead of Zig lane-host fill rounds, int64 debt
saturation instead of unbounded floats, and env-var configuration instead of
process-global flags. No upstream source code is copied; the adaptation is
witnessed by `TestDecodeShareDebtOwedAndZeroKeepsWholePromptOrder` in
`internal/agent/inkernel_decode_share_test.go`. The upstream Apache-2.0 license
is reproduced in the TensorFold benchmark evidence section above.

## TensorFold queued disk-state read-ahead (l3kv prefetch)

`internal/l3kv/prefetch_queue.go` and the read side of `internal/l3kv/
async_job.go` (`AsyncStore.Prefetch` / `RestorePrefetched`) adapt the lane-host
fetch contract from TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/lane_host_fetch.zig
plus its admission caller
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/lane_host_admit.zig
and operator documentation at
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/docs/api.md#L661

The adapted contract: while other streams run, a queued request whose durable
state is on disk has it read ahead on separate I/O workers; held reads wait
with a bounded timeout (default 30 s) and fall back to admission restore or
fresh prefill; concurrent-read permits are capped (default 2, 0 disables)
with an optional resident-memory cap; a cancelled waiter does not cancel the
read — it keeps its permit until the I/O returns; and stop cancels reads. In
this codebase that lands as the `PrefetchQueue` type plus the `Prefetch`,
`RestorePrefetched`, `PrefetchStats` and `StopPrefetch` methods on `AsyncStore`.

Changed for Fak: idiomatic Go re-expression over fak's span-keyed `Store`
interface instead of Zig lane-host state, sentinel-error fallback signals
(`ErrPrefetchTimeout`, `ErrPrefetchDisabled`, `ErrPrefetchStopped`) instead of
Zig error unions, and per-instance configuration instead of process-global
tunables. No upstream source code is copied; the adaptation is witnessed by
`TestRestorePrefetchPermitTimeoutFallback` in
`internal/l3kv/prefetch_queue_test.go`. The upstream Apache-2.0 license is
reproduced in the TensorFold benchmark evidence section above.

## TensorFold CUDA graph-cache policy

`internal/compute/graph_cache_policy.go` `GraphCachePolicy` adapts the keyed
graph-cache policy (`capture-once per key, replay on hit; address-fingerprint
drop-all; LRU cap with min_keep floor; all-rank agree vote gates misses;
declined vote and capture OOM fall back eager; peer capture failure disables;
environment off`) from `zig/src/cuda/graph_cache.zig` `Cache.run` at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209` of the same source:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/cuda/graph_cache.zig

Changed for Fak: idiomatic Go re-expression against an injected engine
interface so the whole policy is testable on host CPU with a fake engine,
string keys instead of u32, and explicit sentinel errors for constructor
misconfiguration. No Zig code, device kernel or release-binary claim is
included; the upstream bound is preserved that its only concrete callers are
experimental DeepSeek-V4.1 paths no release binary serves. The policy is not
yet wired into the live `cudaBackend` capture path.

The pinned repository's root Apache-2.0 LICENSE
(Git `9e8c368b0b4c2c7892a0c9f0f302d7c164016209`) is reproduced byte-exactly
in the existing "TensorFold benchmark evidence" license section of this file
and also applies here.

## TensorFold GDN segment-mark recurrent state

`internal/compute/gdn_marks.go` adapts the markList/slotKernel discipline from
TensorFold at revision `9e8c368b0b4c2c7892a0c9f0f302d7c164016209`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/deltanet_kernels.zig#L9
plus the GDN recurrence context at
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/gdn.zig#L117

The adapted contract: the GDN recurrent state is snapshotted into a bounded
store at segment marks; a mark list keeps at most 4 ascending non-zero rows,
each <= the segment's total row count, refusing empty, late, repeated or
fifth marks; and tree state selects the narrowest slot kernel whose
representable range holds the state's peak magnitude. In this codebase that
lands as `GDNMarkList`, `GDNSlotKernel` / `SelectGDNSlotKernel` and
`GDNMarkedRecurrence` beside the existing GDN dispatch seam
(`Wave32GatedDeltaNetStep` / `tryDeltaNetSIMD`).

Changed for Fak: idiomatic Go re-expression over fak's float32 recurrent-state
buffers instead of Zig slices, sentinel errors instead of Zig error unions, a
fixed IEEE-format slot ladder (E4M3/binary16/bfloat16/binary32) instead of
Zig's device-dependent slot table, and defensive snapshot copies. No upstream
source code is copied; the adaptation is witnessed by
`TestGDNMarkListOrderAndSlotSelection` in
`internal/compute/gdn_marks_test.go`. The upstream Apache-2.0 license is
reproduced in the TensorFold benchmark evidence section above.

## TensorFold distributed message fabric (dist transport vtable)

`internal/model/dist_transport.go` adapts the cluster transport contract from
TensorFold at revision `9e8c368b0b4c2c7892a0c9f0f302d7c164016209`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/cluster/transport.zig
plus its error classification and link-state handling at
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/cluster/fabric_link.zig#L79

The adapted contract: what a control plane needs from a link layer is exactly
four things — a bounded send, a poll of the next message from any link,
every link's up/down state, and the largest payload a link carries — with
messages on one link arriving IN ORDER OR NOT AT ALL, oversize payloads
refused before sending (`TooLarge`), and observed link failures mapped to
closed classes (peer timeout or down is one retryable class; anything else is
corrupt). In this codebase that lands as the `DistTransport` interface
(`Send`, `Recv`, `Links`, `Limit`) plus `NewDistTransport`, which rides a
formed `DistComm` group's existing framed connections (the same
writeFrame/readFrame seams `writeRequest`/`readRequest` use) with control
frames tagged `opTransportCtrl` = 6, and the closed sentinel classes
`ErrTooLarge`, `ErrLinkDown`, `ErrPeerTimeout`, `ErrCorrupt` with
`ClassifyDistTransportError` mapping raw link-layer failures onto them.

Changed for Fak: idiomatic Go re-expression over fak's loopback/stream TCP
framed connections instead of Zig endpoint buffers, per-link FIFO mailboxes
fed by one reader goroutine per peer (non-blocking `Recv` poll instead of a
blocking any-link receive), and parking of non-control frames so the
collective read paths drain them in arrival order. Physical RDMA stays open;
this adaptation is loopback/stream TCP only. No upstream source code is
copied; the adaptation is witnessed by
`TestDistTransportPayloadLimitLinkState` in
`internal/model/dist_transport_test.go`. The upstream Apache-2.0 license is
reproduced in the TensorFold benchmark evidence section above.

## vLLM chunkwise Qwen host Gated-DeltaNet

`internal/model/qwen35_gdn_chunked_host.go` adapts the chunk-64 causal
triangular solve, initial/final recurrent-state schedule and distinct query
readout from vLLM at revision
`5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`:

- `csrc/cpu/sgl-kernels/fla.cpp`, intra-chunk solve/recompute and inter-chunk
  state/readout kernels, and `chunk_gated_delta_rule_cpu`.
  https://github.com/vllm-project/vllm/blob/5c8ae7c74aaa99d1f7fbeea942fd7282e3413810/csrc/cpu/sgl-kernels/fla.cpp
  The pinned source header credits its adaptation from SGLang's CPU kernels.
- `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`, the ordinary
  causal-convolution / initial-state / final-state consumer.
  https://github.com/vllm-project/vllm/blob/5c8ae7c74aaa99d1f7fbeea942fd7282e3413810/vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py

Copyright contributors to the vLLM project. SPDX-License-Identifier: Apache-2.0.
Modified for FAK: native Go/F32 rectangular key/value state, fused application
of the unit-lower-triangular inverse, products of existing rounded scalar decay
(including exact-zero resets), bounded private candidate output/state with
all-head publication only after success, and ordinary Q8/mixed-Q4K host-prefill
consumers. The default-off qualification path changes reduction order. No
Torch, BLAS, BF16/AMX kernel, global workspace or device acceleration is copied
or claimed. Existing causal convolution, decode and fallback remain FAK's.

The pinned repository has no root NOTICE file. Its root Apache-2.0 LICENSE
(Git `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`) is reproduced byte-exactly
in the existing TensorFold license section of this file and also applies here.
Source and adaptation details are in `qwen35_gdn_chunked_host.README.md`.

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

The CUDA IQ2_XXS / IQ2_XS GEMV in `internal/compute/cuda/iq_gemv.cuh` ports the
i-quant decode of `vec_dot_iq2_xxs_q8_1` and `vec_dot_iq2_xs_q8_1`
(`ggml/src/ggml-cuda/vecdotq.cuh`) and the `iq2xxs_grid`, `iq2xs_grid` and
`ksigns_iq2xs` tables (`ggml/src/ggml-common.h`) at llama.cpp revision
`f7c01c5226cac1de1052c17920ff38cdefea0ae6`:
https://github.com/ggml-org/llama.cpp/blob/f7c01c5226cac1de1052c17920ff38cdefea0ae6/ggml/src/ggml-cuda/vecdotq.cuh
That upstream source uses the same MIT license below.

The CUDA IQ1_M / IQ4_NL GEMV in the same file ports the per-thread decode of
`dequantize_block_iq1_m` and `dequantize_block_iq4_nl` (`ggml/src/ggml-cuda/convert.cu`,
decode-identical to `vec_dot_iq1_m_q8_1` / `vec_dot_iq4_nl_q8_1` in
`ggml/src/ggml-cuda/vecdotq.cuh`), the `iq1m_scale_t` super-scale layout and `IQ1M_DELTA`,
and the `iq1s_grid_gpu` and `kvalues_iq4nl` tables (`ggml/src/ggml-common.h`) at llama.cpp
revision `571d0d540df04f25298d0e159e520d9fc62ed121`:
https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/ggml/src/ggml-cuda/convert.cu
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

### Split-phase remote snapshot transfer ownership

The remote snapshot transaction in `internal/radixkv/remote_transfer.go`,
`remote_l3.go`, the scoped tree, and the ordinary InKernelPlanner tiered lookup
and pressure paths adapts prepare/pin, submitted-job ownership, validated
completion, and failed-writer lifetime retention from the vLLM tiering manager:
https://github.com/vllm-project/vllm/blob/5c8ae7c74aaa99d1f7fbeea942fd7282e3413810/vllm/v1/kv_offload/tiering/manager.py

SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project

Changed for Fak: a single synchronous transfer job, not an asynchronous Python
worker pipeline. Explicit held-lock APIs prepare immutable store/configuration,
record, namespace and snapshot-reference identity under the existing planner
mutex, release that mutex for remote transport and independently owned restore,
then reacquire it to validate publication. Configuration generations invalidate
late results even when model and namespace strings are unchanged. Cancellation
does not release the charged job until the actual store callback retires; busy
work cannot consume a breaker probe or authorize source eviction. Stale restored
owners are closed exactly once. A stale successful Put may leave an external
orphan because the existing SnapshotStore has no Delete; no remote rollback is
claimed. Local snapshot capture/serialization and local/SSD work remain locked.
No new endpoint, credential, tenant permission, background queue or device path
is introduced. This is distinct from retained host-swap capacity and dense
prefix payload sharing. Runtime, race and performance qualification remain
separate from source publication.

The full Apache License, Version 2.0 reproduced in the "TensorFold benchmark
evidence" section above applies to this adaptation.

### Checkpoint expert-ring adaptation

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

## antirez/ds4 (DwarfStar) V4 Flash CUDA kernels

Pinned revision: `80ebbc396aee40eedc1d829222f3362d10fa4c6c` (MIT); the license text below is the
upstream `LICENSE` at that revision. The routed-FFN adaptation predates the pin and cites the
revision it was read at.

### Fused routed-FFN kernels

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
bitwise.

### Compressor ring and gated pool

`internal/compute/cuda/v41_compressor.cuh` ports `compressor_store_kernel` (`ds4_cuda.cu:5784`) and
`compressor_update_pool_kernel` (`ds4_cuda.cu:5904`) at the pinned revision; `internal/compute/cuda_v41_compressor.go`
is the thin adapter behind `V41CompressorRingBackend` / `V41CompressorPoolBackend`.
https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/ds4_cuda.cu

Changed for Fak: one token per store into an explicit ring slot with the APE add and ratio-4
overlap row offset left to the caller; only the generic pooling branch is kept, with ds4's
128-entry local candidate arrays replaced by direct strided reads of the same values in the same
order (no rounded operation changes); a separate per-dimension non-finite status post-pass.
The pooling arithmetic (`expf`, `den += w`, `acc += kv * w`, `acc / den`) is upstream's.

The ds4 MIT copyright and permission notice follows:

```text
MIT License

Copyright (c) 2026 The ds4.c authors
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

## antirez/ds4 V4.1 router top-k (CUDA)

`internal/compute/cuda_v41_router.cu` (`k_v41_router_argmax_topk`, `fak_v41_router_topk_f32`) and
its Go binding `internal/compute/cuda_v41_router.go` adapt `router_select_warp_topk_kernel` and
`router_select_kernel` from `ds4_cuda.cu` at revision
`80ebbc396aee40eedc1d829222f3362d10fa4c6c` (MIT, The ds4.c authors):
https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/ds4_cuda.cu

Kept from upstream: one warp per token row with four rows per block, k rounds of a
`__shfl_xor_sync` argmax that prefers the lower expert on equal values, and the hash mode that
takes a row's experts from the token-to-expert table instead of selecting them. Changed for Fak:
expert count, k (up to 8) and route scale are parameters instead of 256/6/1.5; each lane keeps
an insertion-ordered top-k over a strided expert subset so the shuffle rounds merge lane heads;
the score is the project's `v41_router_topk.comp` sqrt(softplus) formula, normalization is the
in-order F32 sum (+1e-20 on the full geometry) instead of a 2^-14 floor, and per-row fault
records follow the existing `V41RouterTopKBackend` contract. The ds4 copyright and permission
notice is reproduced in the section above.

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

## ggml-vulkan cooperative-matrix prefill GEMM

The prefill GEMM `internal/compute/shaders/mm_cm1_quant.glsl` (compiled once per weight
format by the `mm_cm1_q2k.comp`, `mm_cm1_q4k.comp`, `mm_cm1_iq1s.comp`, `mm_cm1_iq2xxs.comp`,
`mm_cm1_iq2xs.comp`, `mm_cm1_iq2s.comp`, `mm_cm1_iq3xxs.comp`, `mm_cm1_iq3s.comp` and
`mm_cm1_iq4xs.comp` wrappers) adapts the cooperative-matrix path of
`ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` (shared A/B tiles, row-major A and
column-major B `coopMatLoad`, per-subgroup accumulator grid, staged edge store) and the
dequantize-into-shared-memory structure of `mul_mm_funcs.glsl` and `dequant_funcs.glsl`, at
llama.cpp revision `f74e2fa6a31cc33ab4ee4ae516641cfe62df3924`:
https://github.com/ggml-org/llama.cpp/tree/f74e2fa6a31cc33ab4ee4ae516641cfe62df3924/ggml/src/ggml-vulkan/vulkan-shaders

The i-quant codebooks in `mm_cm1_iq_tables.glsl` are copied from the i-quant
matrix-vector kernels above. Changed for Fak: fixed 128x128x32 tiles instead of
specialization constants, scalar fp16 shared tiles, the Fak three-binding ABI and push
constants, and a per-format 16-weight decode that follows the matching `iq*_matvec.comp`.
No upstream runtime is included. The upstream license is the MIT license reproduced in the
section above.

## ggml-vulkan split-k flash attention for grouped-query decode

The decode attention kernel `internal/compute/shaders/attention_split_kv.comp` adapts the
scalar path of `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` (grouped-query heads
packed as the rows of one workgroup, a fixed number of lanes per key with a shuffle-xor
score reduction, per-lane online softmax over its own keys combined once per workgroup,
split-k partial output with per-row max and sum) and the combine of
`flash_attn_split_k_reduce.comp`, at llama.cpp revision
`f74e2fa6a31cc33ab4ee4ae516641cfe62df3924`:
https://github.com/ggml-org/llama.cpp/tree/f74e2fa6a31cc33ab4ee4ae516641cfe62df3924/ggml/src/ggml-vulkan/vulkan-shaders

Changed for Fak: F32 row-major KV bound directly with no mask, dequantization, sinks or
softcap; one workgroup per (KV chunk, KV head); fixed 16 lanes per key, 32-key tiles, up
to 8 query rows and head dim 256 instead of specialization constants; the partial and
reduce passes share one module selected by a push-constant mode; a single chunk writes
the normalized output directly. No upstream runtime is included. The upstream license is
the MIT license reproduced in the i-quant matrix-vector section above.

## DeepGEMM mega gate and TileKernels top-k gate (V4.1 fused router)

The fused router kernel `internal/compute/shaders/v41_router_gate_topk.comp` adapts the
structure of DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh` (the
router gate projection and the top-k selection in one launch, so the gate logits never
leave the device) at DeepGEMM revision `1fc9c66975ee322d628b3288534ca0e9ce12a64d`, vendored
in DeepInfer revision `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`:
https://github.com/deepseek-ai/DeepGEMM/tree/1fc9c66975ee322d628b3288534ca0e9ce12a64d
and takes its selection semantics reference (sqrt-softplus score, correction bias, ties to
the lower expert, normalized routed scaling) from TileKernels
`tile_kernels/moe/moe_topk_gate_forward_cuda.py` at revision
`c1cf914a82e6901c679445f872051ce44ac2cc63`:
https://github.com/deepseek-ai/TileKernels/tree/c1cf914a82e6901c679445f872051ce44ac2cc63

Changed for Fak: Vulkan GLSL instead of CUDA/TileLang; the F32 gate and one-token decode
sum of Fak's `matmul.comp` instead of the BF16 tensor-core GEMM, so the logits equal Fak's
existing projection bit for bit; the last arriving workgroup routes the row instead of a
thread-block cluster; Fak's in-order score sum, `+1e-20` rule and fault status record from
`v41_router_topk.comp`; no expert groups, masks, forced routing or EPLB mapping. No
upstream runtime is included. The upstream licenses follow:

```text
MIT License

Copyright (c) 2025 DeepSeek

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

TileKernels carries the same MIT license text with `Copyright (c) 2026 DeepSeek`.

## DeepInfer device-resident compressor ring

The V4.1 decode compressor ring (`internal/compute/shaders/v41_compressor_ring.comp`,
`internal/compute/v41_compressor_ring.go`, `internal/model/v41_compressor_ring_device.go`)
adapts the store-then-compute structure of `compress_and_update_state_decode_tl`
(`deepinfer/model_executor/layers/tilelang_ops/cuda/compress.py`) from deepseek-ai/DeepInfer at
revision `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (Apache-2.0):
https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/compress.py

Changed for Fak: no upstream code is copied. One Vulkan dispatch stores a decode token's projected
(kv, score) row into a per-layer F32 ring row and, for the token that closes a group, pools the
ring in the same dispatch; the pool keeps Fak's exact binary32 `v41_compressor_pool.comp`
arithmetic instead of fast-math exp, writes F32 instead of BF16, carries a per-dimension
non-finite status, and serves one sequence per session instead of a slot-indexed state cache.
The host attention state remains the source of truth and reseeds stale ring rows. Licensed under
the Apache License, Version 2.0 (upstream `LICENSE`); the upstream repository has no NOTICE file.
License text: https://www.apache.org/licenses/LICENSE-2.0

## DeepGEMM and TileKernels mHC prenorm split

Modes 5-7 of `internal/compute/shaders/v41_mhc.comp`, the dispatch entrypoint
`fvk_v41_mhc_prenorm_split_f32` in `internal/compute/vulkan_shim.cpp`, and the Go shape in
`internal/compute/v41_mhc_prenorm.go` adapt the split of the DeepSeek V4 manifold
hyper-connection pre-norm into a projection pass that also produces the row sum of squares
and a second pass that applies `rsqrt(sqrsum / (4 * hidden) + eps)` to the mixes:
- `deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh:43`
  (`sm100_tf32_hc_prenorm_gemm_impl`, the `[m, 24]` mix GEMM with the fused `sqr_sum` and
  split-K partials) from DeepGEMM at revision `1fc9c66975ee322d628b3288534ca0e9ce12a64d`
  as vendored by DeepInfer (Apache-2.0); DeepGEMM is MIT, Copyright (c) 2025 DeepSeek.
- `tile_kernels/mhc/norm_fn_cuda.py:67` (`mhc_reduce_partials_and_rmsnorm_fwd_cuda`) from
  TileKernels at revision `c1cf914a82e6901c679445f872051ce44ac2cc63`; MIT, Copyright (c)
  2026 DeepSeek.

Changed for Fak: no upstream code is copied. Upstream multiplies in TF32 on tensor-core MMA
and sums split-K partials; Fak keeps its existing binary32 arithmetic so the coefficients
stay bitwise those of the one-token MatMul followed by the rms-split: an F32 weight runs
`matmul.comp`'s 1024-wide chunk partials summed in chunk order, a Q2_K weight runs
`q2k_matvec.comp`'s lanes and tree, and the sum of squares is the serial index-order sum of
`rmsSplit`, computed by one workgroup beside the projection workgroups instead of after
them. Vulkan has no grid barrier, so the split remains a second dispatch. The upstream
license text is the MIT license reproduced in the antirez/ds4 section above.


## FlashInfer NVFP4 W4A16 numerical staging

`internal/compute/nvfp4_backend.go` adapts the local E2M1 × E4M3FN-to-BF16
conversion and delayed FP32 tensor-scale staging from FlashInfer at revision
`866c80d0c70a90624b4c2056d12d1543d709609e`:

- [`e2m1x16_e4m3_to_bf16x16` (line 18)](https://github.com/flashinfer-ai/flashinfer/blob/866c80d0c70a90624b4c2056d12d1543d709609e/flashinfer/fused_moe/cute_dsl/blackwell/moe_w4a16_utils.py#L18),
  Copyright (c) 2026 FlashInfer contributors, SPDX-License-Identifier: Apache-2.0.
- [`_dequantize` (line 44)](https://github.com/flashinfer-ai/flashinfer/blob/866c80d0c70a90624b4c2056d12d1543d709609e/tests/moe_ep/w4a16_reference.py#L44)
  and the reference's FP32 matmul followed by `fc1_alpha` / `fc2_alpha`
  (lines 303–304 and 324–325), covered by the upstream Apache-2.0 license.

Changed for Fak: owned row-major packed host tensors, strict shape/scale/input
validation, F32-to-BF16 round-to-nearest-even activation conversion, a streaming
scalar Go reduction with separately rounded FP32 products and additions, and
one tensor-scale multiply per output. This preserves the numerical staging;
it does not reproduce GPU MMA reduction order or establish a Blackwell kernel,
serving integration, or performance result. The BSD-3-Clause SM100 kernel is
not included. The existing early-scale CPU reference keeps its contract.

The full Apache License, Version 2.0 text is reproduced in the
[TensorFold benchmark evidence section](#tensorfold-benchmark-evidence) above.
The upstream NOTICE follows; its KDA references describe upstream files, not
files incorporated by this adaptation.

```text
FlashInfer
Copyright 2025-2026 NVIDIA
Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)

-------------------------------------------------------------------------------------------------
Some of the code in this project are adapted from other open-source projects with different
licenses. This product also bundles some third-party components under other open source licenses.
See licenses/ for text of these licenses.

The persistent CuTe DSL KDA kernels are adapted from humanfia/kda-for-kda-release,
Copyright (c) 2026 KDA Team, under the MIT License (licenses/LICENSE.kda-for-kda).

The static PTX KDA kernels are adapted from NVlabs/kda (260927-kda-for-kda),
Copyright (c) 2026 KDA Team, under the MIT License (LICENSE.kda-for-kda.txt).

The TIRx KDA kernels are adapted from NVlabs/kda (260927-kda-for-kda),
Copyright (c) 2026 KDA Team, under the MIT License (licenses/LICENSE.kda-for-kda).
The split kernel also retains its original Apache-2.0 TIRx-author attribution.
Source: https://github.com/NVlabs/kda/tree/260927-kda-for-kda
```

## llama.cpp tiled CPU attention for ordinary Qwen prefill

Adapted from ggml-org/llama.cpp commit `74bca3950054eed1795f525cb9cf06bfe796492d`,
`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_flash_attn_ext_tiled`:
https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/ggml/src/ggml-cpu/ops.cpp

FAK borrows the multi-query/KV tile, packed K/V reuse and running maximum/sum/output
rescale. Its adaptation retains FAK's F32 fdot/saxpy, causal/window and softcap
semantics, uses an invocation-owned dead projection loan, and is default-off.
It does not import donor SIMD GEMM, quantized KV, sinks, ALiBi or a runtime dependency.
Changed online rounding and all performance claims require separate qualification.

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

## llama.cpp fattn-mma-f16 hd256 prompt attention

The Qwen3.8 prompt-attention kernel `internal/compute/cuda/fattn_mma_hd256.cuh` adapts
one instantiation (DKQ = DV = 256, ncols1 = 8, ncols2 = 8, the Ampere config row) of
`ggml/src/ggml-cuda/fattn-mma-f16.cuh`, with the tile, `ldmatrix` and `mma.sync`
primitives it uses from `mma.cuh` and the `cp.async` helpers from `cp-async.cuh`, at
llama.cpp revision `0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd`:
https://github.com/ggml-org/llama.cpp/tree/0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd/ggml/src/ggml-cuda

Changed for Fak: every C++17 `if constexpr` branch resolved for the one instantiation so
the code compiles as C++14; an implicit causal bound and causal KV-tile skip replace the
mask tensor; f32 K/V are staged once per call into a zero-padded f16 copy; Q and the
output stay f32 in Fak's row-major layout; no ALiBi, softcap, sinks, stream-k split or
fixup buffer. No upstream runtime is included. The upstream license follows.

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

## llama.cpp Q4_K MMQ prompt-panel GEMM

The opt-in `mmq` arm of the CUDA Q4_K prompt-panel GEMM, `internal/compute/cuda/q4k_mmq.cuh`,
adapts the Q4_K instantiation of `ggml/src/ggml-cuda/mmq.cuh` with its Q4_K tile loader
(`mmq-load-tiles.cuh`), the `vec_dot_q8_1_q8_1_mma` dot product (`mmq-vec-dot.cuh`), the
Ampere Q4_K config rows (`mmq-config-ampere.cuh`), the int8 tile, `ldmatrix` and `mma.sync`
primitives it uses (`mma.cuh`), the DS4 `quantize_mmq_q8_1` activation quantizer
(`quantize.cu`) and the fast-division helpers (`common.cuh`), at llama.cpp revision
`0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd`:
https://github.com/ggml-org/llama.cpp/tree/0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd/ggml/src/ggml-cuda

The useful-work Stream-K grid cap additionally adapts upstream change
`4c1004c49221b07c3b8108977e3d1991fc9f45cc`, retained in `mmq.cuh` at the current-source
refresh `0ec4027a2a0183de87c007069adddc06e665f7de` (2026-10-31):
https://github.com/ggml-org/llama.cpp/commit/4c1004c49221b07c3b8108977e3d1991fc9f45cc
Fak limits this adaptation to Q4_K, whose 256-element K iteration is one quant block;
it preserves the existing nonempty partition/fixup order and uses checked wide host
arithmetic. Other formats and the existing opt-in arm selection are unchanged. This
source port does not claim a measured GPU performance improvement.

Changed for Fak: only the Q4_K instantiation on the NVIDIA MMA data layout (sm_80 and newer);
no MoE ids, channel or sample broadcast, or NVFP4 scale plumbing; scratch comes from Fak's
device pool. No upstream runtime is included. The upstream license follows.

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

## llama.cpp Q4_K MMVQ decode GEMV

The default `mmvq` arm of the CUDA Q4_K decode GEMV (`FAK_CUDA_Q4K_DECODE`, escape hatch `f32`),
`internal/compute/cuda/q4k_mmvq.cuh`, adapts the Q4_K instantiation of `mul_mat_vec_q` with its
NVIDIA generic parameter rows (`ggml/src/ggml-cuda/mmvq.cu`), the `vec_dot_q4_K_q8_1` DP4A dot
product (`vecdotq.cuh`) and the `quantize_q8_1` activation quantizer (`quantize.cu`), at
llama.cpp revision `0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd`:
https://github.com/ggml-org/llama.cpp/tree/0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd/ggml/src/ggml-cuda

Changed for Fak: only Q4_K for one to three activation columns; no fusion, MoE ids, channel
or sample broadcast; the second row of a two-row block clamps its weight read; the q8_1 scratch
is a persistent buffer. No upstream runtime is included. The upstream license is the MIT
License reproduced under "llama.cpp Q4_K MMQ prompt-panel GEMM" above
(Copyright (c) 2023-2026 The ggml authors).

## llama.cpp Q3_K MMVQ decode GEMV

The resident CUDA Q3_K matrix multiply, `internal/compute/cuda/q3k_mmvq.cuh`, adapts the
`block_q3_K` layout (`ggml/src/ggml-common.h`), `get_int_b2` / `get_int_b4`,
`vec_dot_q3_K_q8_1_impl_mmvq` and `vec_dot_q3_K_q8_1` (`ggml/src/ggml-cuda/vecdotq.cuh`), the
Q3_K instantiation of `mul_mat_vec_q` with its NVIDIA generic parameter rows (`mmvq.cu`) and
`dequantize_block_q3_K` with its 64-thread launcher (`convert.cu`), at llama.cpp revision
`571d0d540df04f25298d0e159e520d9fc62ed121`:
https://github.com/ggml-org/llama.cpp/tree/571d0d540df04f25298d0e159e520d9fc62ed121/ggml/src/ggml-cuda

Changed for Fak: only Q3_K, f32 dequant output (prompt panels of 128+ rows) and one to three
activation columns per MMVQ launch (the opt-in decode arm); no fusion, MoE ids, channel or
sample broadcast; the second row of a two-row block clamps its weight read; the q8_1 scratch is a
persistent buffer, and the q8_1 quantizer is the one already ported for Q4_K. No upstream runtime
is included. The upstream license is the MIT License reproduced under "llama.cpp Q4_K MMQ
prompt-panel GEMM" above (Copyright (c) 2023-2026 The ggml authors).

## llama.cpp MUL_MAT_ID MMVQ routed-expert GEMV

The CUDA DeepSeek-V4.1 routed-expert kernels, `internal/compute/cuda_v41_routed_mmid.cu`
(`mul_mat_vec_q_id`, `mul_mat_vec_q_down_sum`, `fak_v41_mmid_*`) with the Go binding
`internal/compute/cuda_v41_routed_mmid.go`, port the ids path of `mul_mat_vec_q`
(`channel_x = ids[channel_dst]`, grid Y the selected expert) and its fused gate + GLU epilogue
(`fusion.gate`, `tmp_gate`) from `ggml/src/ggml-cuda/mmvq.cu`, plus `block_q2_K`
(`ggml/src/ggml-common.h`) and `vec_dot_q2_K_q8_1_impl_mmvq` / `vec_dot_q2_K_q8_1`
(`ggml/src/ggml-cuda/vecdotq.cuh`), at llama.cpp revision
`571d0d540df04f25298d0e159e520d9fc62ed121`:
https://github.com/ggml-org/llama.cpp/tree/571d0d540df04f25298d0e159e520d9fc62ed121/ggml/src/ggml-cuda

Changed for Fak: one activation column per launch on the NVIDIA generic parameter row (4 warps,
one row per CUDA block); Q2_K and Q3_K only (Q3_K reuses the dot already ported in
`cuda/q3k_mmvq.cuh`); the ids are Fak's V4.1 routed-id word image and index a by-value table of
up to eight ring-page pointers instead of a 3-D expert bank; the GLU epilogue is the V4.1
checked clamp-SwiGLU with per-pick BF16 fault records; and a down kernel loops the picks in
routed-sum order to fold the checked BF16 publication and the ordered F32 sum into its
epilogue. No upstream runtime is included. The upstream license is the MIT License reproduced
under "llama.cpp Q4_K MMQ prompt-panel GEMM" above (Copyright (c) 2023-2026 The ggml authors).

The clamped SwiGLU lane (`routed_swiglu_lane` in `internal/compute/cuda_v41_routed_swiglu.cu`,
shared by the CUDA routed SwiGLU candidate and these kernels) follows `swiglu_kernel` from
antirez/ds4 `ds4_cuda.cu` at revision `80ebbc396aee40eedc1d829222f3362d10fa4c6c` (MIT, The
ds4.c authors): `g = min(g, c)`, `u = clamp(u, -c, c)`, `silu(g) * u * weight`:
https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/ds4_cuda.cu
Fak adds the BF16 rounding of gate/up and of the weighted product and the two fault stages. The
ds4 copyright and permission notice is reproduced under "antirez/ds4 (DwarfStar) V4 Flash CUDA
kernels" above.

## llama.cpp IQ4_XS resident GEMV

The resident CUDA IQ4_XS matrix multiply, `internal/compute/cuda/iq4xs_mmvq.cuh`, adapts the
`block_iq4_xs` layout and `kvalues_iq4nl` table (`ggml/src/ggml-common.h`), the portable
`get_int_from_table_16`, `VDR_IQ4_XS_Q8_1_MMVQ` and `vec_dot_iq4_xs_q8_1`
(`ggml/src/ggml-cuda/vecdotq.cuh`), the IQ4_XS instantiation of `mul_mat_vec_q` with its NVIDIA
generic parameter rows (`mmvq.cu`) and the per-thread split of `dequantize_block_iq4_xs`
(`convert.cu`), at llama.cpp revision `571d0d540df04f25298d0e159e520d9fc62ed121`:
https://github.com/ggml-org/llama.cpp/tree/571d0d540df04f25298d0e159e520d9fc62ed121/ggml/src/ggml-cuda

Changed for Fak: only IQ4_XS; an f32-activation warp-per-row GEMV using the dequantize split;
one to three activation columns per MMVQ launch (the opt-in decode arm); prompt panels of 32+
rows decode into fp16 hi/lo halves for Fak's compensated tensor-core GEMM; no fusion, MoE ids,
channel or sample broadcast; the second row of a two-row block clamps its weight read; the q8_1
scratch is a persistent buffer, and the q8_1 quantizer is the one already ported for Q4_K. No
upstream runtime is included. The upstream license is the MIT License reproduced under "llama.cpp
Q4_K MMQ prompt-panel GEMM" above (Copyright (c) 2023-2026 The ggml authors).


## NVIDIA cuBLASLt FP8 sample

The experimental Q4_K FP8 arm in `internal/compute/cuda/q4k_fp8lt.cuh` adapts the
per-tensor scaling, matrix descriptor and vendor-GEMM flow from
`cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu` at
NVIDIA/CUDALibrarySamples revision `9b347f161cce5cd2a51a0406b140f2a2d81df024`.
Source: https://github.com/NVIDIA/CUDALibrarySamples/blob/9b347f161cce5cd2a51a0406b140f2a2d81df024/cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu

That revision carries Apache-2.0 in its source header and root LICENSE.md.
No separate NOTICE exists in the root, cuBLASLt or sample directory. The Apache
license is included in this repository LICENSE; the upstream header follows.

Modified for Fak: Q4_K and activation staging, F32 output, shared handle and
workspace, explicit stream, status returns, and sm_89/sm_90-only admission.
No upstream executable or inference runtime is used.

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```


## Strata owner-safe routed/shared expert overlap

`internal/model/moe_shared_overlap.go` adapts the current-router-ID, separate-output,
launch/join/combine control pattern from Niko1221/Strata at revision
`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`:
- `include/strata/core/hit_hook.hpp`
- `src/core/session.cpp` (hit launch, CPU work and combine ordering)
- `src/core/expert_source.cpp` (current expert IDs and async input lifetime)

https://github.com/Niko1221/Strata/tree/c62d5e3a15b40ea51c96aae9e8d36928965f53ab

Copyright (c) 2026 Niko1221 and the Strata contributors.
SPDX-License-Identifier: MIT
The complete copyright, permission and warranty notice is retained in
`internal/model/STRATA_LICENSE.txt`.

Adapted for Fak's existing whole-layer GLM host spill: only a prepared CPU
batch may run on the worker; Session/backend operations remain on their owning
goroutine. Private outputs join before retirement and preserve current router
IDs, weights, dtype and reduction order. The explicit host-payload budget is
default off. This is a Go ownership adaptation, not a copied CUDA kernel,
modified routing/quantization policy, device qualification or performance claim.
## Strata / Project Maya Engram CLOCK replacement

`internal/model/v41_cache.go` adapts `EgRowCache`'s hand/reference-bit replacement
from `src/dsv41/ds4_engram_io.hpp:38-90` in Niko1221/Strata at
`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`:
https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_engram_io.hpp

Source SHA256: `21db12eb7f19613552e4bc6e55634672cf4e02f575d34adab100fbd9b26fb246`.
The source originates in Project Maya v1.1.17 (upstream notice identifies commit
`8d9a9b5d`, https://github.com/mw00/project-maya); Strata's port history records
its later changes. Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya
contributors, including needmorevram, and Niko1221 and the Strata contributors.

Changed for Fak: Go maps and existing caller mutex, typed default-off model
attachment, a bounded reference-bit sweep, byte-budget/fixed-row admission,
and immutable separately copied payloads instead of reusable mutable block
slots. No upstream I/O thread pool, GPU code, model weights, or quantization is
included. Default expert/Engram LRU and the ordinary loader budget are unchanged.

The upstream `third_party/project-maya/NOTICE.md` is retained verbatim below;
its paths and release history refer to the upstream repository.

```text
# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.
```

The full MIT license from `third_party/project-maya/LICENSE` at that revision
follows (the upstream attribution/notice applies to this adaptation):

```text
MIT License

Copyright (c) 2026 Peasant Smith and the Project Maya contributors
Copyright (c) 2026 Niko1221 and the Strata contributors

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

## vLLM paged attention v1/v2

The opt-in CUDA paged KV store's decode attention, `internal/compute/cuda/paged_attention_vllm.cuh`,
adapts `paged_attention_kernel`, `paged_attention_v1_kernel`, `paged_attention_v2_kernel`
and `paged_attention_v2_reduce_kernel` from `csrc/attention/attention_kernels.cuh`, the
float vector helpers of `attention_generic.cuh`, `attention_utils.cuh` and
`dtype_float32.cuh`, the warp-shuffle macros of `csrc/cuda_compat.h`, the launcher
v1/v2 selection rule of `paged_attention_v1.cu` / `paged_attention_v2.cu`, and
`reshape_and_cache_kernel` from `csrc/cache_kernels.cu`, at vLLM tag `v0.11.0`
(`b8b302cde434df8c9289a2b465406b47ebab1c2d`):
https://github.com/vllm-project/vllm/tree/b8b302cde434df8c9289a2b465406b47ebab1c2d/csrc/attention
The kernel is itself adapted by vLLM from NVIDIA FasterTransformer
`decoder_masked_multihead_attention_template.hpp` (release/v5.3_tag), Apache-2.0.

Changed for Fak: only the float / float / kAuto / non-block-sparse instantiation is kept,
so its `if constexpr` arms are resolved (C++14) and the fp8/fp16/bf16 dtype headers,
ALiBi and block-sparse parameters are dropped; torch launchers are replaced by extern "C"
adapters; the cache write takes one contiguous in-page run instead of a slot-mapping
tensor and optionally writes a second key-layout plane (pre-RoPE K); the namespace is
`fak_vllm_pa`. The row gather out of the paged layout is Fak-native glue. No vLLM Python
source or runtime is included.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
```

The full Apache License, Version 2.0 text is reproduced in the
"TensorFold benchmark evidence" section above. Copyright (c) 2023, The vLLM team;
Copyright (c) 2020-2023, NVIDIA CORPORATION. Licensed under the Apache License,
Version 2.0.

## vLLM token-weighted prefix cache telemetry

`internal/gateway/serving_metrics.go` adapts the token-unit numerator/denominator
telemetry model from vLLM at revision
`a20280f30e3169d33a7e21d1d117e9719bace00b`:
- `vllm/v1/metrics/loggers.py` (local prefix-cache query/hit counters and distinct external counters)
- `vllm/v1/metrics/stats.py` (`PrefixCacheStats` and token-weighted `CachingMetrics.hit_rate`)

https://github.com/vllm-project/vllm/tree/a20280f30e3169d33a7e21d1d117e9719bace00b

Copyright contributors to the vLLM project.
SPDX-License-Identifier: Apache-2.0

Changed for Fak: optional counter presence in the existing Go serving schema;
direct-gauge precedence; finite/nonnegative aggregate validation; local/external
tier separation; atomic prefix source replacement; omission rather than deriving
a native token rate from provider request-cache events. No vLLM execution engine,
Python dependencies, sliding request window, cache reset operation, or inference
policy is included. This is an ADAPT of the licensed schema/ratio model, not a
literal translation of the upstream logger or scheduler.
The upstream root LICENSE is Apache-2.0 and matches this repository's LICENSE.
Both studied source files carry the SPDX contributor notices above. No root
NOTICE exists at this exact revision (GitHub contents API returned 404).

## vLLM packed heterogeneous host prefill

The optional packed host-prefill path in `internal/model/batch_prefill_packed.go`
and `internal/modelengine/nativesched_packed_prefill.go` adapts cumulative query
offsets, per-request computed positions, and final-row logits selection from
vLLM's `GPUModelRunner._prepare_inputs`:

- Repository: https://github.com/vllm-project/vllm
- Revision: `5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`
- Source: `vllm/v1/worker/gpu_model_runner.py`, blob
  `d98498a4f97bfd89700829fd36df71bbab6437f5`
- License: Apache-2.0, root LICENSE blob
  `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`
- Copyright: Copyright contributors to the vLLM project.

Changed for Fak: bounded synchronous Go CPU FP32/Q8-weight projection panels
reuse existing per-request causal attention and independently owned FP32 caches.
The default-off `fak_vllm_packed_prefill` qualification tag selects the existing
native syscall Engine scheduler, with at most 32 total prompt tokens per cohort,
full-prompt admission reservations, and existing cancel/Close session retirement.
Unsupported execution policies, observers, layouts and storage retain their
historical paths; no GPU, asynchronous transfer, chat-planner or speed claim is made.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this adaptation except in compliance with the License. A copy is available at
https://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and
limitations under the License.

## vLLM bounded reverse-KMP prompt lookup

`internal/model/ngram.go`, `longestRecentTokenSuffix`, adapts
`_find_longest_matched_ngram_and_propose_tokens` from
`vllm/v1/spec_decode/ngram_proposer.py` at revision
`85591000403fc3bc3b4d0303e3a97bedab007763`:
https://github.com/vllm-project/vllm/blob/85591000403fc3bc3b4d0303e3a97bedab007763/vllm/v1/spec_decode/ngram_proposer.py

SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project

Modified for Fak: translate the bounded reverse-KMP scan to Go slice indexing,
keep the most-recent occurrence on equal-length matches rather than vLLM's
earliest occurrence, stop at the first maximum-length match, and use a small
local prefix table for default settings. The existing copied-proposal,
configuration, default-off, and target-verification contracts are unchanged.
The full Apache License, Version 2.0 is supplied in the root `LICENSE`.

## NVIDIA cuBLASLt experimental SM120 MXFP8 WKV baseline

`internal/compute/cuda/blockscaled_wkv.cuh` adapts the descriptor/heuristic
flow from `cuBLASLt/LtMxfp8Matmul/sample_cublasLt_LtMxfp8Matmul.cu` in
NVIDIA/CUDALibrarySamples at revision
`9b347f161cce5cd2a51a0406b140f2a2d81df024`:
https://github.com/NVIDIA/CUDALibrarySamples/blob/9b347f161cce5cd2a51a0406b140f2a2d81df024/cuBLASLt/LtMxfp8Matmul/sample_cublasLt_LtMxfp8Matmul.cu

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Modified for Fak: exact V4.1 wkv_b TN geometry, E4M3/UE8M0 inputs with F32
output, SM120-only runtime opt-in and toolkit/version refusal, one bounded
vendor heuristic, owned buffers and checked stream/capture/cleanup boundaries.
The activation cast is the existing Fak SIMT kernel; recipe32-to-vendor scale
repacking is a byte permutation, not an upstream GEMM kernel. The weight scale
adapter reuses the existing attributed TensorRT-LLM byte-layout helper.
No NVIDIA sample build system, model loader, or new serving policy is included.
The donor copyright and Apache-2.0 header are retained in the adapted source.
The complete Apache License, Version 2.0 is supplied in root `LICENSE` and
reproduced in the TensorFold section above. No separate donor NOTICE exists
at this revision. This experimental backend is unqualified; it is not evidence
of B300 acceptance or an SM120 performance gain.

## Strata owner-safe routed/shared expert overlap

`internal/model/moe_shared_overlap.go` adapts the current-router-ID, separate-output,
launch/join/combine control pattern from Niko1221/Strata at revision
`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`:
- `include/strata/core/hit_hook.hpp`
- `src/core/session.cpp` (hit launch, CPU work and combine ordering)
- `src/core/expert_source.cpp` (current expert IDs and async input lifetime)

https://github.com/Niko1221/Strata/tree/c62d5e3a15b40ea51c96aae9e8d36928965f53ab

Copyright (c) 2026 Niko1221 and the Strata contributors.
SPDX-License-Identifier: MIT
The complete copyright, permission and warranty notice is retained in
`internal/model/STRATA_LICENSE.txt`.

Adapted for Fak's existing whole-layer GLM host spill: only a prepared CPU
batch may run on the worker; Session/backend operations remain on their owning
goroutine. Private outputs join before retirement and preserve current router
IDs, weights, dtype and reduction order. The explicit host-payload budget is
default off. This is a Go ownership adaptation, not a copied CUDA kernel,
modified routing/quantization policy, device qualification or performance claim.

### vLLM multi-page CUDA KV append addressing (2026-10-31)

`internal/compute/cuda/paged_kv_write_map.h` and the multi-page writer in
`internal/compute/cuda/paged_attention_vllm.cuh` adapt the per-token slot resolution
and K/V cache address arithmetic of `reshape_and_cache_kernel` in
[vLLM `csrc/cache_kernels.cu`](https://github.com/vllm-project/vllm/blob/e5196e2959a42908313510f55fa21f213a48802e/csrc/cache_kernels.cu),
commit `e5196e2959a42908313510f55fa21f213a48802e` (Apache License 2.0).
Copyright (c) 2023, The vLLM team.

Fak replaces the device slot-mapping tensor with at most 64 physical page IDs
passed by value, keeps F32 storage and its optional pre-RoPE K plane, and retains
its existing single-page writer and page/COW ownership. Host admission and Go
batch preparation/order control are Fak-specific. This does not reattribute the
older v0.11.0 paged attention implementation or claim a measured speedup.
The upstream Apache-2.0 license text is reproduced in the existing vLLM notice.


## llama.cpp causal hidden-output and pooling contract

The optional causal embeddings provider adapts final-normalization pooling and
LM-head bypass from ggml-org/llama.cpp commit
cd8427d018c07eefab6c110f7f115c31ec4c6157, src/llama-graph.cpp.
FAK retains its own CPU layer/weight ownership implementation and explicit admission.
No trained embedding quality or device equivalence is implied.

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


## vLLM bounded native CPU beam search

`internal/beam/search.go` and its ordinary native CPU chat adapter in
`internal/agent/inkernel_beam.go` adapt the live/completed beam lifecycle,
top-2B expansion and generated-length score from vLLM revision
`5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`:

- `vllm/entrypoints/generate/beam_search/offline.py`
  (Git `e458ba99282ff683593e7ddbedf726d0c8612857`)
- `vllm/entrypoints/generate/beam_search/utils.py`
  (Git `d009008cb5a060939b26535b419ccfeea09d1673`)

https://github.com/vllm-project/vllm/tree/5c8ae7c74aaa99d1f7fbeea942fd7282e3413810/vllm/entrypoints/generate/beam_search

Copyright contributors to the vLLM project. SPDX-License-Identifier: Apache-2.0.
Modified for FAK: native Go float64 finite log-softmax, stable deterministic
ties, fixed request/work bounds, typed nonstreaming CPU selection, fresh
owned Session replay and all-or-error ranked publication. No vLLM device
execution, KV-sharing efficiency or model qualification is claimed.

The pinned repository has no root NOTICE. Its root Apache-2.0 LICENSE
(Git `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`) is reproduced byte-exactly
in the existing TensorFold license section of this file and also applies here.
See `internal/beam/README.md` for the consumer, adaptations and execution debt.


## llama.cpp score-based SPM tokenizer

Adapted from ggml-org/llama.cpp at cd8427d018c07eefab6c110f7f115c31ec4c6157,
src/llama-vocab.cpp (score-priority UTF-8 symbol merging and byte fallback).
FAK wires an explicit llama-spm-v1 JSON profile through its immutable tokenizer,
trusted-fragment and raw-completion contracts. This does not add embedded GGUF,
.model or generic Unigram interchange. Legacy BPE remains unchanged. Generic
FAK encoding leaves BOS/EOS to the selected raw prompt caller.

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


## llama.cpp bounded native Jinja chat-template interpreter

- Upstream: ggml-org/llama.cpp, commit
  `cd8427d018c07eefab6c110f7f115c31ec4c6157` (MIT).
- Source: `common/jinja/lexer.cpp`, `parser.cpp`, `runtime.cpp`,
  `string.cpp`, `string.h` and `value.cpp`; full license retained at
  `internal/agent/JINJA-LICENSE.txt`.
- Adaptation: bounded tag/expression/statement parsing, request-local
  scope/macro/namespace execution and input-marked string composition, wired to
  the ordinary native prompt preparation and segmented tokenizer consumer.
- FAK-specific changes: independent source/model digest binding, explicit
  construction-time opt-in, closed builtin/profile admission, finite resource
  limits, redacted refusal, and conservative special-token provenance.
  No Python runtime, model, tokenizer, executable or network behavior is imported.

## TensorFold cluster admission planning

`cmd/fak/cluster.go` (the `fak cluster info --plan` admission planner: `fitsAt`,
`maxStreams`, `maxContext`, `chunkRows`, `planAdmission`) adapts the converged-round
admission math from `zig/src/cluster/admission.zig` in TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209` (Apache-2.0):
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/cluster/admission.zig

The per-rank needs-plus-KV-plus-margin budget check against each rank's GPU limit,
the monotone stream/context binary searches, and the 16-step prompt-chunk sizing
within (1+share) of decode-only round time are re-expressed in idiomatic Go over
fak's own TPPlan tiling and a declared round-cost model. No upstream Zig code
blocks, cost model, traffic link or round composition are copied; the planning
surface keeps the upstream honesty line that admission planning never proves
physical serving.

The pinned repository's Apache-2.0 LICENSE is reproduced in the existing TensorFold
license section of this file and also applies here.

## TensorFold NVFP4 host reference interpretation

`internal/compute/nvfp4.go` (`DecodeNVFP4Stacks`, stacked multi-expert input to the
NVFP4 CPU reference) adapts the host dequantization semantics from
`zig/src/core/quant/nvfp4.zig` in TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209` (Apache-2.0):
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/quant/nvfp4.zig

Per-expert global scales follow the Qwen4 device-weight loading in
`zig/src/families/qwen4/device_weights.zig` at the same revision:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/families/qwen4/device_weights.zig

Contiguous expert stacks, e2m1 low-nibble-first packing, one e4m3 scale per 16
inputs, per-stack global scaling, exact subnormal-scale decoding, and
shape/stack mismatch refusals are re-expressed in idiomatic Go over fak's
existing E2M1/E4M3 tables. No upstream Zig code blocks are copied.

The pinned repository's Apache-2.0 LICENSE is reproduced in the existing TensorFold
license section of this file and also applies here.

## TensorFold shared-attention geometry contract

`internal/model/shared_attention_geometry.go` adapts the shared-attention
geometry contract from `zig/src/core/shared_attention.zig:17` in TensorFold at
revision `9e8c368b0b4c2c7892a0c9f0f302d7c164016209`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/shared_attention.zig

The adapted contract: query heads divide KV heads (GQA) with head dim bounded
at 256 and 32-aligned; the rotary range fits the head and is even; the byte
mask[row*stride+key] alone decides which keys each row sees; and cache appends
go to unique physical slots on pairwise-disjoint K/V/cache stores. `ForwardTP`
in `internal/model/tensor_parallel_forward.go` consults the shape half before
native dispatch when `TPConfig.SharedAttentionGeometry` is set, and
`internal/model/shared_attention_geometry_test.go` pins the sentinel refusals.

Changed for Fak: idiomatic Go over fak's float32 tensors with sentinel errors
(`ErrSharedAttentionShape`, `ErrSharedAttentionView`,
`ErrSharedAttentionAppend*`, `ErrSharedAttentionAliased`, ...) instead of Zig
error sets; buffer identity by opaque store ID with fail-closed empty-ID
aliasing; and an opt-in ForwardTP gate so the CPU reference path keeps serving
toy fixture geometries the kernel contract does not admit. No Zig runtime,
kernel, or model code is included.

The upstream NOTICE and full Apache License, Version 2.0 are reproduced in the
"TensorFold benchmark evidence" section above (same pinned repository and
revision); they apply to this adaptation as well.

## TensorFold external drafter attachment (per-stream DFlash2 trees)

`internal/agent/inkernel_dflash2.go` (`AttachDFlash2Tree` and the
`dflash2TreeStream` owner) adapts the external-drafter attachment idea from
`zig/src/core/lanes/drafted.zig:35` in TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/lanes/drafted.zig
plus the drafter initialization at
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/native/cuda_drafted.zig
and the operator-facing `--drafter` documentation at
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/README.md

The adapted contract: the target is told the drafter's feature taps before any
forward; kept target rows are absorbed into drafter state; the drafter stream
lifecycle is owned end to end; and offered chain/tree nodes are bounded by the
target width, with unsupported tree requests refusing (the upstream
TreesNotBuilt/NotDrafting equivalents). Per-stream DFlash2 trees do not
constitute a generic ability to load arbitrary neural drafters.

Changed for Fak: idiomatic Go re-expression over the existing `EnableDFlash2`
serving-domain seam instead of Zig stream wrappers, the authenticated target's
verification panel as the width bound, and closed sentinel errors
(`ErrDFlash2TreesNotBuilt`, `ErrDFlash2NotDrafting`,
`ErrDFlash2TreeWidthExceeded`) matched with `errors.Is`. No upstream source
code is copied; the adaptation is witnessed by `TestDFlash2TreeWidthBoundRefusal`
in `internal/agent/dflash2_tree_test.go`.

The upstream NOTICE and full Apache License, Version 2.0 are reproduced in the
"TensorFold benchmark evidence" section above (same pinned repository and
revision); they apply to this adaptation as well.

## TensorFold durable write admission budget

`internal/agent/inkernel_warm_disk.go` adapts the staging-admission
reserve/backoff idea from `zig/src/core/lanes/learned_disk.zig:61`
(`Admission.reserve`) in TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209`, with the caller pattern from
`zig/src/sessions/jobs.zig:113` and the byte-permit documentation at
`docs/api.md:647`:
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/lanes/learned_disk.zig

The adapted contract: durable warm-disk writes reserve their byte budget
before staging against a pool sized from the store cap and the volume's
measured free bytes; unknown free space or an unaffordable request refuses
fail-closed; each refusal arms an exponential backoff (1 s base, doubled per
failure, capped at 60 s after 7 failures); finish releases the reservation, a
landed write clears the ladder, unaffordable writes are skipped rather than
faulted, incomplete records stay unpublishable through the store's existing
temp+fsync+rename commit, and in-flight reservations drain within a shutdown
deadline. `internal/agent/warm_disk_budget_test.go` pins the sentinel
refusals and the backoff ladder.

Changed for Fak: idiomatic Go over fak's existing warm-disk seam
(`beginWarmDiskCapture` / `persistWarmPrefixDisk`) with a per-store-directory
permit pool and sentinel errors instead of Zig error sets; the pool is measured
from fak's own store byte cap and `compute.DiskInfo` free-space probe. No Zig
runtime, kernel, or model code is included.

The upstream NOTICE and full Apache License, Version 2.0 are reproduced in the
"TensorFold benchmark evidence" section above (same pinned repository and
revision); they apply to this adaptation as well.

## TensorFold eviction-spill (l3kv Spiller)

Adapted from ashhart/TensorFold @9e8c368b0b4c2c7892a0c9f0f302d7c164016209 (Apache-2.0):
the eviction-spill contract — an optional spill copy of durable KV state before
releasing RAM, reusing the durable restore path, capped on disk/staging/time;
checkpoint slots bound retained RAM states.

Source files: zig/src/native/disk_spill.zig (spill record + caps),
zig/src/server/cli.zig:518-523 (--spill-gib/--checkpoint-slots flags),
zig/src/core/prompt_cache.zig:191 (spill-before-release call site).

What was adapted into internal/l3kv:
- Spiller (spill.go): owned-copy staging through Store.Put under the span's
  residency digest, with per-entry and total-disk byte caps and a bounded Put
  window; the checkpoint-slot knob is carried/validated here, enforced on the
  retaining side.
- AsyncStore seam (async_job.go): EnableSpill/Spill/SpillStats mirror the
  Prefetch read-side seam.

No Zig runtime, kernel, or model code is included.

The upstream NOTICE and full Apache License, Version 2.0 are reproduced in the
"TensorFold benchmark evidence" section above (same pinned repository and
revision); they apply to this adaptation as well.

## TensorFold bounded top-k selection

`internal/agent/inkernel_sampling.go` (`topKRankOrder`, shared sweep
`topKSelectHeap`) adapts the bounded top-k selection contract from
`zig/src/core/lanes/sampling_selection.zig` (line 7, with `sampling.zig` lines
116/134) in ashhart/TensorFold at revision
`9e8c368b0b4c2c7892a0c9f0f302d7c164016209` (Apache-2.0):
https://github.com/ashhart/TensorFold/blob/9e8c368b0b4c2c7892a0c9f0f302d7c164016209/zig/src/core/lanes/sampling_selection.zig

Greedy best-by-value-descending-then-id-ascending picks, null-on-any-non-finite
row, and allocation of only the k selected indices are re-expressed in idiomatic
Go over fak's existing samplerWorkspace hot path: the non-finite rejection maps
to the package's ErrNonFiniteLogits sentinel, and the bounded min-heap sweep is
shared with the pre-existing topKTruncate seam so no parallel selection path is
introduced. No upstream Zig code blocks are copied.

The pinned repository's Apache-2.0 LICENSE is reproduced in the existing TensorFold
license section of this file and also applies here.
