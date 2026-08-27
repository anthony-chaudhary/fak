# vLLM ticket closure

Machine ledger: `ledger.json` (`sha256:31d4b6e64ba8a4328014c0530d1257613aeec3cdf12ccf90da1cbb16091c7ec0`). Parent study: #9268.

## Verdict

- **Complete:** 193 join clusters have dispositions; 5 uncovered actionable clusters map through 2 priority candidates to 2 open captured tickets; closure leftovers are 0.
- **Ticket accounting:** created=2, reused=0. Here “created” means constructed specifically for these candidates before this offline build; the build itself performs no GitHub mutation.
- **Honesty:** landed/open/partial/conflict/obsolete source dispositions remain separate counts; only the five `uncovered` clusters enter the two new tickets.

## Corpus coverage

- vLLM: **53,848 records** at `f18d0ba90d972a852a351c98be3f42b31372cfe4`, cutoff `2026-08-26T22:35:00Z` (index `sha256:75577e65370ac201f29efcf6ce38c15b7bf48c0d793fbfb946407a4045de7c14`).
- Related systems: **14/14 forge captures complete**, totaling **121,063 records**. Runtime-tree partial and inaccessible classes remain explicit below.
- Final FAK ticket corpus: **9,537 records**, including open issues #9377 and #9378.

## Source receipts

| source | schema | revision / cutoff | count | checksum |
|---|---|---|---:|---|
| `docs/research/vllm-priority-2026-08-27/ledger.json` | `fak.study-priority-ledger/1` | anthony-chaudhary/fak@b4103047b27d94bcb5995bbcae912f99d661857f<br>2026-08-27T04:36:00Z | 2 | `sha256:df1f2114211a78b15f88decad3bb02de1e3ff790e150bd65c8ec55b73d68a417` |
| `docs/research/vllm-fak-join-2026-08-27/ledger.json` | `fak.study-join-ledger/1` | anthony-chaudhary/fak@b4103047b27d94bcb5995bbcae912f99d661857f<br>2026-08-27T04:36:00Z | 193 | `sha256:2e7b2a6c64a3b2a48a6a49db0bed2086e82327d4b41c150b3fa297f424b6413d` |
| `fak-forge.json` | `fak-studyforge-corpus/1` | 8effa952450f642aa7169910249ae756d8580607<br>2026-08-27T05:52:00Z | 9537 | `sha256:1507a16160b4c9a95fa5057e9e1f3d1bbdccc6709daa9fe8bd7ff2f392f11b1c` |
| `docs/research/inventory/vllm-related-system-adjacency-v1.json` | `fak-study-adjacency/1` |  | 14 | `sha256:be5cde2c2ed4df1bde2a069d889884204138d51189fa3f5ecd1ab04c35254f34` |

Capture receipt: repository `anthony-chaudhary/fak`, revision `8effa952450f642aa7169910249ae756d8580607`, cutoff `2026-08-27T05:52:00Z`, status `complete`, records=9537, sources=[issues pulls discussions releases labels milestones], receipt `sha256:a34377af672312ea0a4dfcd6b54845a6e3166f089fdb1d644b6541116325c6e5`, index `sha256:35c6e6f1d5bb5bd1927c38d5614bfa445bcca8fb6e227e1287cba2c049362d7a`.

## Closure counts

| disposition | total | actionable |
|---|---:|---:|
| `landed` | 4 | 4 |
| `open_exact` | 2 | 2 |
| `partial` | 168 | 159 |
| `conflict` | 13 | 13 |
| `obsolete` | 1 | 0 |
| `uncovered` | 5 | 5 |

Selected/unmapped=0; unclassified=0; unmapped actionable=0; closure leftovers=0.

## Ticket mapping and queue

| rank | candidate | issue | horizon | source clusters |
|---:|---|---|---|---|
| 1 | `native-vllm-ir` | [#9377](https://github.com/anthony-chaudhary/fak/issues/9377) | `now` | `architecture_runtime:body:vllm-ir`, `architecture_runtime:label:vllm-ir`, `architecture_runtime:title:vllm-ir`, `kernels_compilation:label:vllm-ir` |
| 2 | `allocator-fragmentation` | [#9378](https://github.com/anthony-chaudhary/fak/issues/9378) | `next` | `memory_residency:body:allocator-fragmentation` |

Each issue is captured once, open at the forge cutoff, carries the parent and stable-cluster links, required issue sections, native engine/model constraint, expected horizon labels, and a unique queue slot. Dependencies are empty for both candidates.

## Adjacency coverage

Adjacency `vllm-related-system-adjacency-2026-08-26-v1` has 14 members and 29 complete, 4 partial, and 9 inaccessible/missing source-class receipts.

### Partial classes

| repository | class | status | note |
|---|---|---|---|
| `ai-dynamo/dynamo` | `runtime_tree` | `partial` | Reused docs/research/inventory/ai-dynamo-dynamo.json at stale pin f494601e; 5,401 files were exhaustive there, not at the cutoff pin. |
| `flashinfer-ai/flashinfer` | `runtime_tree` | `partial` | Reused docs/research/inventory/flashinfer-ai-flashinfer.json at stale pin fb28d724; exact cutoff tree remains uncaptured. |
| `llm-d/llm-d` | `runtime_tree` | `partial` | Reused docs/research/inventory/llm-d-llm-d.json at stale pin 3243fcf1; the exact cutoff tree is not captured. |
| `vllm-project/speculators` | `runtime_tree` | `partial` | Reused docs/research/inventory/vllm-project-speculators.json at stale pin 0faffeb3; exact cutoff tree is not inventoried. |

### Inaccessible / missing classes

| repository | class | status | note |
|---|---|---|---|
| `NVIDIA/TensorRT-LLM` | `runtime_tree` | `missing` | No machine-readable runtime tree inventory exists for c2f5f319; docs/notes/CONCEPT-STUDY-TENSORRT-LLM-2026-07-18.md is stale source guidance only. |
| `sgl-project/sglang` | `runtime_tree` | `missing` | No standard inventory for full sgl-project/sglang at 7f27bf47; existing mini-sglang map is a different repository and cannot satisfy this member. |
| `vllm-project/afd-plugin` | `runtime_tree` | `missing` | No pinned runtime tree slice is retained in git or allocated scratch for this run. |
| `vllm-project/aibrix` | `runtime_tree` | `missing` | No pinned runtime tree inventory is present. |
| `vllm-project/production-stack` | `runtime_tree` | `missing` | No standard pinned runtime tree inventory was produced in this run. |
| `vllm-project/router` | `runtime_tree` | `missing` | No standard pinned runtime tree inventory was produced in this run. |
| `vllm-project/semantic-router` | `runtime_tree` | `missing` | No pinned runtime tree inventory is present. |
| `vllm-project/vllm-gguf-plugin` | `runtime_tree` | `missing` | No pinned runtime tree inventory is present. |
| `vllm-project/vllm-metal` | `runtime_tree` | `missing` | No standard pinned runtime tree inventory was produced in this run. |

## Sampling evidence

| cluster | disposition | actionable | artifacts | confidence | evidence |
|---|---|---|---:|---|---|
| `apis_tool_calling_structured_output:body:guided-decoding` | `landed` | true | 2 | `explicit-map+captured-closed+history` | `sha256:e8b8aaf79d2322e1ee2176103c869c3af6be6977f2e92b74b12db5f089ae17a0` |
| `apis_tool_calling_structured_output:title:structured-output` | `open_exact` | true | 3 | `exact-title` | `sha256:8e9e6da3c3301b1a53ed8adf982b754504db15cc2c3813008e4c4435c7148a0f` |
| `apis_tool_calling_structured_output:body:json-schema` | `partial` | true | 8 | `lexical-candidate` | `sha256:99424d7388bd05d9c49eeec93dc8315edfa14a6b3594755549c1e3db551d0237` |
| `model_backend_hardware:title:model-support` | `conflict` | true | 3 | `ambiguous` | `sha256:686e36b0551b796feb6a69a521adea022101c0e834e3b45072770b1faac31613` |
| `explicit_non_candidate:disposition:release-metadata-noncandidate` | `obsolete` | false | 0 | `source-explicit-noncandidate` | `sha256:d289037c999007f3924e4c9eb90b5564c918e99978c6ad571ea639ae96599780` |

## Refresh obligations

- Recapture the complete anthony-chaudhary/fak study-forge corpus before relying on issue open state after either mapped issue changes.
- Rebuild the study-link ledger when its classification index, FAK forge capture, repository revision, or adjacency manifest changes.
- Rebuild the study-priority ledger when the uncovered actionable set, hard gates, rubric, dependencies, or queue order changes.
- Run fak study-tickets build and validate whenever any recorded source checksum changes.
- Refresh this closure if #9377 or #9378 changes state, title, body, labels, horizon, source-cluster links, or dependencies.
- classification source checksum: sha256:b62813994a053eba39e85a817c0e2f68f0401ae9f6023cde66bb9684d6b7ddaa
