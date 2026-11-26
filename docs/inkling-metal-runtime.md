# The vendored Inkling Metal runtime: what it is, what fak owns, and how to maintain it

**Audience:** maintainers who have to change, debug, trim, or replace the Apple GPU path for Inkling-2-Mini. This page maps the code. It is not a benchmark sheet.

**Evidence labels used on this page.** *Measured* means a number recorded from a run or a byte-compare. *Read from code* means the statement comes from reading the sources named next to it and has not been confirmed by a profile or an ablation. Line counts are `wc -l` over tracked files and include comments and blank lines.

## Summary

- On an Apple GPU, `fak run <Inkling-2-Mini GGUF>` executes the model inside a C++/Metal runtime vendored from `incoai/splash` at commit `2afc7337aacc044722fa95e4cb3da75c281a19a8` (Apache-2.0). It lives in `third_party/inklingmetal/runtime`: 275 files, about 67k lines, of which about 4.1k are a fak-generated shader index.
- fak owns everything outside the forward pass (CLI, tokenizer, prompt template, admission, GPU lease, the speculation loop and its acceptance check) and a native adapter in `internal/inklingmetal` (about 1.7k lines of C++ and 0.6k of Go) that replaces the upstream serving engine.
- *Measured* on an M3 Pro with `Inkling-2-Mini-UD-Q3_K_XL.gguf`: the vendored path decodes at about 40 tok/s. fak's own Metal wiring for the same model (`ggufload.OpenInkling2LoopMetal`, matrix products only on the GPU) logged 7–9 decode tok/s and is now only the fallback.
- The vendored runtime and fak's own Metal library (`internal/metalgemm`, about 52k lines, used by the Qwen models) do not share code. Each solves the same problems separately.
- The vendored path is the default today, but it is not the intended hot path: it is kept as the comparison bar and as a learning source, then removed. The staged, gated retirement plan and the current scoreboard are in [`inkling-metal-vendored-retirement.md`](inkling-metal-vendored-retirement.md).

## 1. What executes, and where the boundary is

### Call path

1. `cmd/fak/run_inkling2.go` (`openInkling2Run`) detects the architecture, builds the tokenizer from the GGUF (`runInkling2Tokenizer`), and resolves the backend (`inkling2RunBackend`).
2. If the architecture is `inkling2_loop` and `inklingmetal.Available()` is true, it sizes the load with `inklingmetal.Estimate`, admits it through `admitLocalMetalModel` (memory plan plus the exclusive GPU lease in `internal/gpulease`), and loads with `inklingmetal.Open`. Otherwise it falls back to `ggufload.OpenInkling2LoopMetal`.
3. `internal/agent/inkling2_run.go` (`NewInkling2MetalRunPlanner`, `inkling2RunPlanner.Complete`) renders the chat template, tokenizes, opens a session sized to prompt plus output, and picks a decode loop.
4. Greedy requests (temperature 0, no penalties) go to `ggufload.Inkling2MetalGreedy` in `internal/ggufload/inkling2_metal_speculative.go`, which drives draft, verify, and commit through the session.
5. Sampled requests read the full logits row back each token (`Session.BeginBlock` then `Session.Commit(1)`) and sample on the host with the agent sampler. This path does not speculate.

### What fak owns

| Concern | Where | Notes |
|---|---|---|
| CLI, backend selection, fallback | `cmd/fak/run_inkling2.go` | Refuses split GGUFs on the Metal path. |
| Tokenizer and chat template | `internal/tokenizer`, `internal/agent/inkling2_run.go` (`inkling2RunPrompt`, `inkling2RunReply`) | The vendored tree has no tokenizer; upstream's lived in its engine, which was not vendored. |
| Memory admission and GPU lease | `internal/localadmission`, `internal/gpulease`, `cmd/fak/run_inkling2.go` | The byte plan itself is computed natively by `fak_ink_plan`. |
| Planner | `internal/agent/inkling2_run.go` | One planner type covers host, CUDA, and Metal. |
| Speculation driver | `internal/ggufload/inkling2_metal_speculative.go` | Chooses prompt-lookup or neural draft, checks ancestry, applies EOS and the output budget. |
| Acceptance rule | `internal/polymodel` (`AcceptGreedy`, `AcceptTree`), `internal/model/ngram.go` (`NgramDrafter`) | The GPU also computes acceptance; the host recomputes it and refuses a mismatch. |
| Host sampling | `internal/agent` (`sampleLogitsWithPenaltyWS`) | Non-greedy requests only. |
| Go binding | `internal/inklingmetal/model.go`, `internal/inklingmetal/model_darwin_arm64.go`, `internal/inklingmetal/model_stub.go` | `Model`, `Session`, one-idle-session reuse, argument validation. |
| Native session | `internal/inklingmetal/target_darwin_arm64.cc` | `Weights`, `FakInkSession`: owns arenas, the KV page table, the committed token list. |
| Draft lane, draft budget, tree verify | `internal/inklingmetal/fak_draft_lane.hpp`, `internal/inklingmetal/fak_draft_budget.hpp`, `internal/inklingmetal/fak_tree_verify.hpp` and their `_darwin_arm64.cc` files | fak-maintained, but the three `internal/inklingmetal/fak_*_darwin_arm64.cc` files each carry a comment saying they adapt upstream `model/Runtime.cpp` (the `.hpp` headers carry no such comment). Treat them as derived code under the same license, not as independent designs. |
| Shader embedding | `internal/inklingmetal/cmd/genshaders`, `third_party/inklingmetal/runtime/metal/FakEmbeddedShaders.hpp`, `third_party/inklingmetal/runtime/metal/FakShaderIndex.inc` | Compiles MSL source at first use instead of loading upstream's prebuilt metallib. |
| Build glue | 60 `internal/inklingmetal/op_*_darwin_arm64.cc` files | Each includes one vendored `.cpp`/`.mm` so cgo compiles it. `third_party/inklingmetal/sources_darwin_arm64.go` embeds the tree so the Go build cache sees vendored edits. |

### What is vendored

Everything under `third_party/inklingmetal/runtime`: the Metal backend, the kernel ABI, the shaders, the operators, the model definitions (target and draft), the GGUF reader and repacker, and the Neural Engine support.

*Measured* by byte-compare against upstream at the pinned commit: 272 of 275 runtime files are identical. The three that differ are `metal/MetalBackend.mm` (about ten changed lines that route pipeline creation to embedded shader source when no metallib path is given) and the two fak-added shader files above.

Upstream `runtime/` has 377 files. The 102 not vendored are mostly the `engine/` directory (the serving loop, request batching, KV pool, and prefix cache) and `main.mm`. This matters for section 8: several limits exist because the engine that provided those features was replaced by a single-session adapter, not because the operators cannot do them.

The build gate is in `internal/inklingmetal/platform.h`: the runtime compiles only against a macOS 26.4 SDK with `__has_embed`. Otherwise `internal/inklingmetal/platform_darwin_arm64.cc` supplies stubs and `fak_ink_available` returns 0.

### The bridge (`internal/inklingmetal/bridge.h`)

Every function returns a `strdup`'d error string or null. Every session call takes the model-wide device mutex.

| Call | What it does |
|---|---|
| `fak_ink_available` | True when the device passes the vendored capability probe on macOS 26.4 or later. |
| `fak_ink_plan` | Plans packed image, scratch, state, and KV bytes for a context length without loading tensors. |
| `fak_ink_model` | Creates the Metal backend, repacks the GGUF into GPU images, and loads the optional DFlash2 draft. |
| `fak_ink_free_model` | Releases the model. |
| `fak_ink_session` | Allocates one request's arenas, page table, logits buffer, verifier, and draft lane for a token capacity. |
| `fak_ink_reset` | Clears request state so an idle session can serve a new request within its allocated capacity. |
| `fak_ink_free_session` | Releases a session under the device mutex. |
| `fak_ink_step` | Prefills a prompt in chunks, or stages a causal block of up to 32 rows; returns logits or argmax per scored row. |
| `fak_ink_commit` | Keeps the first N staged rows and discards the rest. |
| `fak_ink_tree` | Stages a caller-built candidate tree and returns the target's argmax at each row. |
| `fak_ink_commit_tree` | Keeps the root-to-leaf path of a staged tree; leaf -1 discards it. |
| `fak_ink_propose` | Runs the draft model and returns a linear proposal. |
| `fak_ink_draft_tree` | Runs the draft model and returns a candidate tree. |
| `fak_ink_propose_verify` | Draft and target verify in one submission; the caller then commits a leaf. |
| `fak_ink_cycle` | Draft, target verify, argmax, acceptance, and state commit in one submission; returns the tree, per-row target ids, and the retained-row mask. |

`fak_ink_cycle` is the production call. The others exist for the stepwise paths and tests.

## 2. Layer map of `third_party/inklingmetal/runtime`

Sizes are tracked files and lines at the vendored revision.

| Layer | Path | Files / lines | Responsibility |
|---|---|---|---|
| Metal backend | `metal/MetalBackend.mm`, `metal/MetalBackend.hpp`, `metal/Allocation.hpp`, `metal/DeviceCapabilities.hpp`, `metal/CommandWatchdog.hpp`, `metal/Residency.hpp` | about 4.0k lines excluding the generated index | Device, buffers and views, pipeline compilation and cache, command encoding and submission, completion tickets, GPU timing, residency. |
| CommandGraph | `metal/CommandGraph.hpp` | 145 lines | An ordered list of compute dispatches with parameter payloads, address tables, function-constant specialization, and event or host steps. Operators append to a graph; the backend encodes it. |
| ABI | `metal/abi` | 30 files / 3.7k | C structs and constants shared by host C++ and MSL: quant formats, KV extents, verify-tree rows, sampling layouts. The contract between operators and kernels. |
| Kernels | `metal/kernels` | 78 files / 16.8k | MSL compute shaders. `shared` (30 files, 7.0k) serves prefill and decode; `decode` (5 files, 4.0k) holds the staged decode tiles, sampling, and the draft; `prefill` (4 files, 0.8k); `common` (38 headers, 5.0k) is the tile, quant-format, and attention code the others include. |
| Operators | `ops` | 57 files / 12.9k | Host-side planners that turn one logical operation into dispatches: `Linear`, `LinearGguf`, `MoE`, `PagedAttention`, `RelativeAttention`, `ShortConvolution`, `MessageGate`, `BlockResidualMix`, `Sampling`, `KernelSelection`, `KernelCalibration`, `PageStorage`, `ExecutionPlans`. |
| Model | `model` | 90 files / 23.9k | Model families and loading: `Inkling2Loop*` (the target), `DFlashDraft` (the draft), `QwenTarget` (the shared target interface every family implements), `RuntimeArenas`, `GgufFile`, `GgufImage`, `GgufPreparation`, `WeightImages`, plus the speculation cost and acceptance models (`StepCosts`, `VerifyBudget`, `LookupProposals`) and the SSD tiering and expert-streaming stores. |
| ANE | `ane` | 5 files / 1.4k | Core ML program build and GPU-to-Neural-Engine handoff. |
| Top level | `AwakeClock.hpp`, `Checked.hpp`, `StderrLine.hpp`, `Supervision.hpp` | 4 files / 0.3k | Small utilities. |

`model/Inkling2Loop.hpp` is the best single description of the architecture as the runtime sees it: 24 weight layers, layers 6–17 looped three times, 48 layer applications per token, each with its own KV and convolution state; 20 sliding-window layers and 4 full-history layers; a 128-expert top-8 MoE.

## 3. Why it is fast

All of the following is read from code unless a number is given. No ablation has been run in fak to rank these, so the order is the order a token flows through them, not a ranking of contribution.

### One fused GPU submission per decode cycle

`TreeVerify::cycle` in `internal/inklingmetal/fak_tree_verify_darwin_arm64.cc` builds two command graphs and submits them as one command with `MetalBackend::submitCommandPartsAsync`. The first part runs the draft model. While the GPU runs it, the host encodes the second part: the target forward over the draft's tree rows, argmax, acceptance (`ops::Sampling::addAcceptance`), KV path commit, convolution-state commit, and the draft's own state update. The host waits once. A cycle that accepts several tokens therefore costs one submit and one wait, and the draft's output never returns to the host before the target consumes it: the selector writes the tree rows on the device and the verify reads them there.

For contrast, fak's first Metal wiring for this model sent each matrix product to the GPU separately and ran the rest of the forward on the host. That difference in submission granularity is the most likely main cause of the measured 7–9 versus about 40 tok/s, but this has not been isolated.

### GPU-resident arenas and double-buffered state

`model/RuntimeArenas.hpp` defines `PrefillArena` and `DecodeArena`: every intermediate tensor of a step has a fixed slot allocated once per session, so a step allocates nothing. Convolution state is held in two buffers (`current`, `next` in `FakInkSession`); a step reads `current`, writes `next`, and the host swaps handles only on commit. KV writes land at or above the committed position, and position alone gates visibility, so a rejected speculation needs no copy or undo.

### Load-time repack of quantized formats into plane images

`model/GgufImage.hpp` plans, from GGUF metadata alone, one image per layer; `model/GgufPreparation.hpp` writes it. Quantized rows are not used in llama.cpp's block layout. A GPU pass (`metal/kernels/shared/gguf_repack.metal`) rewrites them into per-format planes padded to whole tiles, so the decode kernels read codes and scales at fixed strides. Some tensors are converted at load to what the kernels read directly (bf16 or F32 copies, transposed relative-bias tables, reordered convolution taps). Staging is pipelined through two buffers so file reads overlap the GPU's writes.

The cost is load time and a second, GPU-format copy of the weights; `fak_ink_plan` reports that copy as `target_images`.

### Row-count kernel selection with per-GPU calibration

`ops/KernelSelection.hpp` chooses, for each projection and each row count, among several compiled tiles: a register tile, staged decode tiles at different heights, int8-activation tiles, and one-pass or two-pass forms for a gate/up pair. `ops/KernelCalibration.hpp` holds the measured constants that decide the choice, per GPU class and core count: how many input splits to use, how many threadgroups per core saturate memory bandwidth, where the int8 path wins. The constants were measured by upstream on its devices. *Read from code:* an M3 Pro selects a calibration row by GPU class and core count; nobody at fak has re-measured that row.

This is the part least like fak's own Metal code, and the part most likely to be wrong on a device upstream did not measure.

### Expert-grouped MoE tiles

`ops/MoE.hpp` and `metal/kernels/shared/moe.metal`, `metal/kernels/shared/moe_gguf.metal`: the router selects top-k experts per row on the GPU, a grouping pass sorts rows by expert, and the expert projections then run as tiles over rows that share an expert, so rows that route to the same expert share its tile. With one row this changes little; with 8–16 verify rows the step reads the union of selected experts rather than 8 experts per row. The size of this effect has not been measured in fak.

### Paged int8 KV with split attention

`ops/PagedKv.hpp`, `ops/PageStorage.hpp`, `ops/PagedAttention.hpp`, `ops/RelativeAttention.hpp`. K and V are stored as int8 with separate scale planes (`kv::Format::Int8`, set in `Weights` in `internal/inklingmetal/target_darwin_arm64.cc`), in fixed-size pages grouped into extents. Kernels reach pages through a GPU page table of buffer addresses, so growing context adds an extent without rebinding or copying. Attention over a long history is split into several partial reductions that run in parallel and are combined, with the split count planned per layer from history length and core count. Inkling's 20 sliding-window layers read at most a 1024-key window; only the 4 full layers pay for full history.

### Tree-shaped speculative verify with the DFlash draft and online cost models

The draft (`model/DFlashDraft.hpp`, checkpoint directory `Inkling-2-Mini-DFlash2` next to the GGUF) is a separate small model whose layer count is read from its checkpoint at load (`DFlashDraftLayout.layers`), not fixed in code. It reads hidden rows the target captures at three of its layer applications plus its final norm, and proposes a block of following tokens in one pass. fak requests a tree up to 4 candidates wide (`TreeWidth: 4` in `internal/agent/inkling2_run.go`). The target verifies the whole tree in one forward: each row attends to committed history plus its own ancestors (`metal/abi/VerifyTree.h`).

How many rows to verify is chosen per cycle by `DraftBudget::prepare` in `internal/inklingmetal/fak_draft_budget_darwin_arm64.cc`. It maximizes expected accepted tokens per millisecond using a cost line fitted online from observed GPU time per row count (`model/StepCosts.hpp`), acceptance probabilities calibrated online from the draft's confidence (`model/VerifyBudget.hpp`), and a prompt-lookup index whose match-length acceptance rates are also learned online (`model/LookupProposals.hpp`). `DraftBudget::observe` feeds each cycle's outcome back.

The host independently recomputes the accepted path from the returned per-row target ids (`inkling2VerifyMetalTree`) and refuses the cycle if the GPU's retained-row mask disagrees. Output is therefore the target's greedy output regardless of what the draft proposes; the draft changes speed only. *Measured* at vendoring time: token ids matched on two short prompts and a 128-token prompt.

### Residency management

`metal/Residency.hpp`: all buffers sit in one Metal residency set that a 500 ms heartbeat keeps wired while requests arrive and releases after an idle keep-alive. The set is deliberately not attached to the command queue, because Metal would then make all of it resident before every command buffer; upstream's comment records that this slowed decode several-fold near the working-set limit. Each command instead declares only the extents its page tables reach.

## 4. Carried but idle for Inkling

These subsystems compile into the binary and are not exercised by the Inkling path as fak drives it. Line counts are sums over the named files and are estimates; "idle" is read from code (the adapter passes a null `AneFfn`, an empty `SpilledKvReads`, and a default GGUF plan), not from coverage data.

| Subsystem | Main files | Approx. lines | Why idle here |
|---|---|---|---|
| Neural Engine FFN | `ane`, `ops/AneFfn.cpp`, `ops/AneFfn.hpp`, `ops/AneFfnCalibration.cpp`, `ops/AneFfnCalibration.hpp`, `metal/kernels/prefill/ane_ffn.metal` | 3.7k | Offloads dense FFNs during prefill; Inkling's FFN is MoE and the adapter passes no ANE split. |
| SSD KV tiering | `model/SlotFile.cpp`, `model/KvBlockCache.cpp`, `model/KvHistoryStream.cpp`, `model/SlotRecycler.cpp`, `model/SsdReaders.cpp`, `model/SpilledKvReads.cpp`, `ops/KvSlotFormat.cpp` and headers | 4.5k | Spills cold KV pages to disk and reads blocks back; fak allocates only in-memory Full pages. |
| Expert streaming | `model/ExpertStore.cpp`, `model/SparseRowStore.cpp`, `model/ExpertChunkLayout.hpp` and headers | 2.8k | Streams MoE experts from SSD instead of loading them; the fak load keeps all experts resident. |
| GDN (gated delta) layers | `ops/GDN.cpp`, `ops/GDN.hpp`, `metal/kernels/decode/gdn.metal`, `metal/kernels/prefill/gdn.metal` | 1.7k | A recurrent mixer of the Qwen hybrid families; Inkling has none. |
| Other model families | `model/Qwen3_8.cpp`, `model/Qwen4Exp.hpp`, `model/Qwen3_6Moe.hpp`, `model/Flint.cpp`, `model/QwenHybridLayout.hpp`, plus shaders such as `metal/kernels/shared/hyper_connection.metal`, `metal/kernels/shared/ngram_embedding.metal`, `metal/kernels/shared/sparse_attention.metal`, `metal/kernels/shared/block_indexer.metal` | 2–3k | Layouts, loaders, and kernels for models fak does not run through this runtime. |
| Vision | `ops/Vision.hpp`, `model/QwenVision.hpp`, `metal/kernels/shared/vision.metal` | 0.4k | Image input for other families. |
| Safetensors / MLX loading | `model/SafetensorsCheckpoint.mm`, `model/SafetensorsImage.hpp`, `model/SafetensorsTarget.hpp`, branches inside `model/GgufImage.cpp` | 0.4k dedicated | fak loads the target from GGUF. The DFlash2 draft checkpoint does go through `model/DraftCheckpoint.cpp`, so do not remove checkpoint reading wholesale. |

Roughly 15k of the 63k upstream-derived lines, about a quarter. The shared headers are the obstacle to removing them: `model/QwenTarget.hpp` is the interface every family implements and it names ANE, spilled-KV, and GDN types in its signatures, so trimming means editing vendored interfaces, which ends byte-identity with upstream (see section 9).

Two of these are not dead weight in principle. SSD KV tiering and expert streaming are the upstream answers to "the model or the context does not fit", which fak answers elsewhere with its own mechanisms. They are idle because fak has not connected them, not because they are useless.

## 5. Vendored subsystem versus fak's own equivalent

"Connected" asks whether the vendored path uses the fak component. In every row the answer is no: the vendored runtime opens the GGUF by path and does everything else itself. The only fak code in the loop is on the host side of the bridge.

| Problem | Vendored | fak's own | Connected |
|---|---|---|---|
| GGUF parsing and tensor access | `model/GgufFile.cpp`, `model/GgufImage.cpp` | `internal/ggufload` | No. fak parses the same file a second time for the tokenizer and architecture check. |
| Quantized weights on GPU | `model/GgufPreparation.cpp`, `ops/LinearGguf.cpp`, `metal/kernels/shared/gguf_linear.metal`, `metal/kernels/shared/gguf_repack.metal` | `internal/metalgemm/packed_gguf.go`, `internal/metalgemm/packed_gguf_darwin.m`, `internal/metalgemm/packed_gguf_gemv.metal`, `internal/metalgemm/packed_gguf_dequant.metal` | No. Different on-GPU layouts: repacked plane images versus views over the mapped GGUF blocks. |
| MoE experts on GPU | `ops/MoE.cpp`, `metal/kernels/shared/moe_gguf.metal` | `internal/metalgemm/packed_gguf_routed_darwin.go`, `internal/metalgemm/packed_gguf_moe.metal` | No. |
| Command batching | `metal/CommandGraph.hpp`, `metal/MetalBackend.mm` | `internal/metalgemm/graph.go`, `internal/metalgemm/icb_replay_darwin.go`, `internal/metalgemm/icb_dispatch.go` | No. Both batch a step into few submissions. Going by file names and roles, fak's replays a recorded indirect command buffer where upstream re-encodes each step; the two have not been compared in detail. |
| Device-resident KV | `ops/PagedKv.hpp`, `ops/PageStorage.cpp` | `internal/metalgemm/graph_devicekv.go` (q8 KV covered by `internal/metalgemm/graph_kvq8_longctx_darwin_test.go`) | No. |
| Attention kernels | `ops/PagedAttention.cpp`, `ops/RelativeAttention.cpp` | `internal/metalgemm/flash_attn_darwin.go`, `internal/metalgemm/flash_decode_split.metal` | No. fak's own Metal wiring for Inkling runs attention on the host. |
| KV persistence and tiering | `model/SlotFile.cpp`, `model/KvBlockCache.cpp` (idle) | `internal/kv` | No. |
| Fit and memory planning | `fak_ink_plan` over `model/RuntimeArenas.cpp`; kernel choice in `ops/KernelCalibration.hpp` | `internal/fitplan`, `internal/localadmission` | Partly. fak's admission consumes the native byte plan as an opaque total; it does not know its parts or the calibration. |
| Sampling | `ops/Sampling.cpp`, `metal/kernels/decode/sampling.metal` (greedy argmax and acceptance used; stochastic sampling unused) | `internal/agent/inkernel_sampling.go` and the agent sampler used by `internal/agent/inkling2_run.go` | No. Sampled requests bypass the GPU sampler and lose speculation. |
| Speculative decoding policy | `model/VerifyBudget.cpp`, `model/StepCosts.cpp`, `model/LookupProposals.cpp` | `internal/polymodel/specdecode.go`, `internal/polymodel/specdecodetree.go`, `internal/polymodel/specdepthconfidence.go`, `internal/model/ngram.go` | Partly. fak's acceptance functions re-check the result; the budget decision is the vendored one. Both sides also run their own prompt-lookup drafter. |
| Draft model | `model/DFlashDraft.cpp` | the DSpark draft in `internal/ggufload/inkling2dspark.go` | No. Different draft models per platform. |

The practical consequence: an improvement to `internal/metalgemm` does not speed up Inkling on Mac, and an upstream improvement to the vendored runtime does not speed up Qwen on Mac.

## 6. The CUDA Inkling path, for contrast

The CUDA path was written in fak rather than vendored, and it is structured as additions to shared code:

- Device kernels are headers in the shared CUDA library. `internal/compute/cuda_kernels.cu` includes five directly: `internal/compute/cuda/inkling2_loop.cuh`, `internal/compute/cuda/inkling2_prefill.cuh`, `internal/compute/cuda/inkling2_dspark.cuh`, `internal/compute/cuda/inkling2_dspark_markov.cuh`, and `internal/compute/cuda/inkling2_verify_rows.cuh`. Two more, `internal/compute/cuda/inkling2_attend_split.cuh` and `internal/compute/cuda/inkling2_expert_gemv.cuh`, come in transitively through the first of those. The entry points are declared in `internal/compute/cuda_backend.h`.
- The Go seam is `internal/compute/cuda_inkling2.go`; the device session is `internal/ggufload/inkling2loop_cuda_device.go`.
- The forward's structure (which layer runs when, loop passes, tree rows) is shared host code in `internal/model/inkling2loop_forward.go`, `internal/model/inkling2loop_passes.go`, `internal/model/inkling2loop_spine.go`, and `internal/model/inkling2loop_tree.go`, which the CPU path also runs.

*Measured:* about 5.4k fak-authored lines; decode climbed from 6.3 to 151 tok/s over 22 commits, with three small cited adaptations from other projects.

| | CUDA path | Mac path |
|---|---|---|
| Origin | Written in fak | Vendored |
| Size | about 5.4k lines | about 63k upstream-derived plus 1.7k adapter |
| Shares the backend library with other models | Yes | No |
| Shares the host forward with the CPU path | Yes | No; the forward is `model/Inkling2LoopTarget.cpp` |
| Time to first fast result | 22 commits | One commit |
| fak engineers can explain each kernel | Yes | Not yet |

Neither column is the right answer by itself. Vendoring reached 40 tok/s 25 minutes after the home-grown attempt logged 7–9. The CUDA path took longer and left fak with code it understands and can extend to the next model.

## 7. The backend-neutral seam

*Read from code.* The CUDA path does not call CUDA from the model code. It goes through four Go interfaces in `internal/compute`:

| Interface | Defined in | Covers |
|---|---|---|
| `Inkling2Device` | `internal/compute/inkling2_device.go` | The op set of the resident decode forward: every op enqueues on the device and nothing returns to the host except through explicit read calls. |
| `Inkling2PrefillDevice` | `internal/compute/inkling2_prefill_device.go` | The same ops batched along the token axis for a prompt chunk. |
| `Inkling2VerifyDevice` | `internal/compute/inkling2_prefill_device.go` | What a speculative verify chunk adds: logged convolution history and the commit that rewinds to the kept prefix. |
| `Inkling2DraftDevice` | `internal/compute/inkling2_prefill_device.go` | The DSpark draft's forward. |

The CUDA backend is the only implementation today (`internal/compute/cuda_inkling2.go`, `internal/compute/cuda_inkling2_prefill.go`). The drivers are backend-neutral host code in `internal/ggufload`: `internal/ggufload/inkling2loop_resident.go` opens a resident session on any backend that satisfies `Inkling2Device` and upgrades to batched prefill when it also satisfies `Inkling2PrefillDevice`; `internal/ggufload/inkling2loop_prefill.go`, `internal/ggufload/inkling2loop_spec.go`, and `internal/ggufload/inkling2loop_draft_dev.go` use the prefill, verify, and draft interfaces the same way.

`internal/metalgemm` implements none of these interfaces yet. The planned integration path for fak's own Metal forward is to implement them there, so the existing resident drivers run on Metal unchanged, the way they run on CUDA. This is a plan, not something in the tree. The vendored runtime is the bar that implementation is measured against: same file, same prompt, identical greedy token ids, and the vendored decode rate. It stays the default until that bar is met.

## 8. Known limits

Each is read from code unless marked measured.

| Limit | Cause | Where |
|---|---|---|
| One request at a time per model | `geometry.target.lanes = 1`; every bridge call takes `Weights::device`; Go adds `Model.mu` and `Session.mu`. Upstream batches lanes in its engine, which was not vendored. | `internal/inklingmetal/target_darwin_arm64.cc`, `internal/inklingmetal/model_darwin_arm64.go` |
| One process at a time per GPU | *Measured:* a second concurrent `fak run` is refused by the exclusive GPU lease. | `internal/gpulease` |
| Session capacity is prompt plus `max_tokens` | `Complete` calls `NewSession(len(ids) + maxTokens)`; a block that would pass capacity is refused. | `internal/agent/inkling2_run.go` |
| KV pages are always Full and in memory | The adapter allocates extents in the default form and never spills or slims; memory grows linearly with context for all 48 applications, including sliding-window layers whose old pages upstream can slim. | `internal/inklingmetal/target_darwin_arm64.cc` (`ensurePages`) |
| No prefix reuse across requests | Each request resets the session and prefills the whole prompt; a multi-turn chat re-runs every earlier turn. Only the allocation is reused (`Model.idle`). | `internal/inklingmetal/model_darwin_arm64.go` (`NewSession`), `fak_ink_reset` |
| Speculation is greedy-only | Any temperature or penalty takes the one-token path with a full logits readback per token. | `internal/agent/inkling2_run.go` |
| No tools, no thinking budget | `Complete` refuses them. | `internal/agent/inkling2_run.go` |
| Single-file GGUF only | Split shards are refused on the Metal path. | `cmd/fak/run_inkling2.go` |
| `fak serve --gguf` refuses on Mac | *Measured:* "Metal memory estimate refused before model load: quant capability refusal: derive active tensor roles: gguf: missing inkling2_loop.attention.head_count". Separately, `RefuseInkling2LoopBackend` admits only `cuda`. Serving goes through the generic loader, which does not know this adapter. | `internal/model/arch_support.go` |
| macOS 26.4 SDK and runtime, Apple GPU that passes the capability probe | MSL 4.0 and `__has_embed`. | `internal/inklingmetal/platform.h` |
| Hardware test is opt-in | Skips unless `FAK_INKLING2_TEST_GGUF` is set, so ordinary CI does not exercise the vendored path. | `internal/inklingmetal/hardware_darwin_arm64_test.go` |
| Kernel calibration is upstream's | The M3 Pro row was not re-measured by fak. | `ops/KernelCalibration.hpp` |

## 9. Maturity ladder

A rung describes how well fak controls the code, not how fast it runs. Speed is fixed by rung 0; each later rung must hold the measured bar (about 40 tok/s on the M3 Pro reference file, identical greedy token ids) or it is a regression.

| Rung | State | Exit test | Status |
|---|---|---|---|
| 0. Vendored | Upstream code runs behind a narrow bridge. License and notices preserved, revision pinned. | `fak run` produces the target's greedy output at the measured rate. | **Done.** |
| 1. Tracked and understood | Provenance is machine-checkable: pinned revision, per-file identity against upstream, the list of local modifications. A maintainer can name what each layer does and which parts are idle. The hardware test runs somewhere on a schedule. | A script reproduces the 272/275 byte-compare; this page stays accurate; a profile attributes cycle time to draft, verify, and commit. | **In progress.** This page is the "understood" half. The byte-compare is recorded but not automated. |
| 2. Trimmed, bridge widened | Idle subsystems removed or compiled out. The bridge exposes what serving needs: several lanes, prefix retention, sampled speculation, a memory plan broken into parts. | Binary and source shrink by the idle share; `fak serve` admits the model; multi-turn chat does not re-prefill. | Not started. Costs byte-identity with upstream, so upstream fixes become manual ports. Do this only after rung 1 makes that cost visible. |
| 3. Subsystems swapped to fak | Individual vendored subsystems replaced by fak's, one at a time, each behind a parity test: GGUF reading by `internal/ggufload`, admission by `internal/fitplan`, sampling by the agent sampler, KV tiering by `internal/kv`. | Each swap holds the rate and the token ids. | Not started. The table in section 5 is the work list. |
| 4. Shared library | Inkling on Mac is a model definition over `internal/metalgemm`, the way Inkling on CUDA is a set of headers in the shared CUDA library; concretely, `internal/metalgemm` implements the interfaces in section 7. The techniques in section 3 that `internal/metalgemm` lacks have been ported into it and benefit the Qwen models too. | The vendored tree is no longer the default; it remains as the comparison bar until the shared path matches it, then is removed. | In progress. Gates in [`inkling-metal-vendored-retirement.md`](inkling-metal-vendored-retirement.md). |

Rungs 2–4 are options, not commitments. Rung 1 is the one that should not be skipped: without it the default Mac path for this model is 63k lines nobody at fak can vouch for, pinned to a revision nobody is watching.

## 10. Lessons for the next port

These follow from the comparison in section 6 and apply beyond this model.

1. **Look at what exists before building, in both directions.** Before the home-grown Metal wiring was written, a runtime that already ran this exact model fast on this exact hardware was available under a permissive license. Before vendoring, fak's own Metal library already had packed-GGUF kernels, command batching, and device KV. As far as the commit history shows, neither was consulted against the other; that is an inference from the order and content of the commits, not a recorded fact. The check is cheap: for each technique the target needs, name the upstream file and the fak file that does it. Section 5 is that table, written after the fact.
2. **Vendoring buys a result; it does not buy understanding.** One commit moved the number from 7–9 to about 40 tok/s. It also added a second Metal stack that shares nothing with the first. Budget the rung 1 work as part of the vendoring decision, not as optional follow-up.
3. **Index vendored code by technique, not only by file.** A provenance record that says "275 files from revision X" answers the license question. It does not answer "where is the int8 KV" or "do we already have this". An index keyed by technique, with the upstream location, the fak equivalent, and whether they are connected, answers both and makes duplication visible.
4. **The reason a port is slow is usually structural.** fak's first wiring was not slow because its kernels were bad; it was slow because it offloaded matrix products and kept the control flow on the host. The vendored runtime's advantage is where the boundary sits: the whole cycle on the device, one wait. That lesson transfers to `internal/metalgemm` without copying any code.
5. **Note what was left behind.** Upstream's engine was not vendored, and with it went batching, prefix caching, and KV tiering policy. Three of the limits in section 8 trace to that one decision. When vendoring part of a system, record which capabilities lived in the part that was dropped.
6. **Keep the fast path as the bar while replacing it.** Whatever happens at rungs 2–4, the vendored path at rung 0 is a correct, fast reference on real hardware. Every replacement is measured against it on the same file and prompt.
7. **A calibration table is data, not code.** `ops/KernelCalibration.hpp` encodes measurements from someone else's devices. Vendored constants of this kind need their own check on local hardware; byte-identity with upstream says nothing about whether they are right here.
