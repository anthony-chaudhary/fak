# Concept Study: hasso5703/dgx-spark-qwen38 — High-Performance Qwen3.8 GB10 Serving Stack

**Source:** https://github.com/hasso5703/dgx-spark-qwen38  
**Pinned Revision:** `a08dea9cd8cda4cee61c007d4176f7b97908de8e`  
**Commit Date:** 2026-09-03T20:09:02Z  
**Study Date:** 2026-09-03  
**Author:** hasso5703 (with community contributions from hashd1ve, MiaAI-Lab, r0b0tlab, RadixArk, BBuf)  
**License:** MIT License (`LICENSE:1-21@a08dea9cd8cda4cee61c007d4176f7b97908de8e`) over Apache-2.0 SGLang sources  
**Tracking Issue:** [#10952](https://github.com/anthony-chaudhary/fak/issues/10952)  
**Parent Epics:** [#9433](https://github.com/anthony-chaudhary/fak/issues/9433) (Qwen3.8 / GLM-5.3-Flash native architecture), [#10193](https://github.com/anthony-chaudhary/fak/issues/10193) (native inference performance), [#2236](https://github.com/anthony-chaudhary/fak/issues/2236) (inference engine superset)  
**Study Depth:** Deep (exhaustive inspection of overlay modules, Triton kernels, proxy implementation, speculative decoding modifications, and deployment automation)  
**Completeness Critic:** Verified — all primary components (`flash-sglang/qwen4_exp.py`, `flash-sglang/sm121_varlen.py`, `flash-sglang/qwen_sparse_attn_backend.py`, `flash-sglang/kda_kernels/`, `keepalive-proxy.py`, `dflash2/sglang/`, `bench.sh`, `bench-matrix.sh`, `needle.sh`, `patch-yarn.py`, `patch-template.py`, `oc-fit-limits.py`, and service templates) inspected at `file:line@a08dea9cd8cda4cee61c007d4176f7b97908de8e`.

---

## Executive Summary

`hasso5703/dgx-spark-qwen38` is a production-grade, hardened serving stack engineered specifically for serving the **Qwen3.8** model family on a single **NVIDIA DGX Spark** workstation (GB10 Grace Blackwell, 128 GB Unified Memory Architecture). While the broader open-source ecosystem runs Qwen3.8-Flash-Next (176B/6B hybrid MoE) across multi-node clusters or dual DGX Spark systems (TP2), this repository achieves single-node serving with full 262K context, native vision support, lossless speculative decoding, and working prefix caching.

The project demonstrates exceptional systems engineering rigor, solving critical edge cases that cause silent failure in long-running agent workloads:
1. **Zero-Copy NVMe Memory-Mapped Embedding Backing (51B PLE Table)**: Bypasses the ~48 GB pinned host RAM requirement for Qwen3.8-Flash-Next's Position-Linked Embedding (n-gram) table by mapping it from NVMe via `torch.from_file(shared=True)` with explicit `madvise(MADV_RANDOM)` and persistent sidecar completion markers. Leveraging GB10's coherent CPU-GPU memory bus, GPU gather kernels dereference pageable host pointers directly without table residency, incurring less than 3% runtime overhead.
2. **SM121 QSA Sparse-Decode Kernel Specialization**: Discovers and resolves a silent long-context data corruption failure in upstream FlashInfer's TRT-LLM sparse-decode kernel on SM121 (Blackwell GB10), where requests beyond 100k tokens produce runs of token ID 0 (`!`). Replaces the broken path with a specialized Triton packed-varlen kernel and an upstream-merged KDA package (`PR #36845`), validated by 11/11 exact needle retrievals at 120k tokens.
3. **Agent-Conforming Keepalive & Cancellation Reverse Proxy (`keepalive-proxy.py`)**: Resolves client timeout drops caused by SGLang's tool-call parser buffering arguments (measured at 127 seconds of silence during a 400-line file write). Injects official Anthropic `event: ping` frames on `/v1/messages` and authentic empty chunks (`{"choices":[]}`) on OpenAI `/v1/chat/completions` at 10-second intervals. Simultaneously eliminates "zombie generations" on the GPU by actively issuing POST `/abort_request` to SGLang when an HTTP client disconnects, and trips an early abort if a corrupted kernel emits 128 consecutive token-0 characters.
4. **Elimination of the FlashInfer "Boot Lottery"**: Proves that default FlashInfer kernel autotuning on GB10 creates massive non-deterministic performance swings (92 to 111 tok/s aggregate throughput across boots for identical configurations). Eliminates the variance with `--disable-flashinfer-autotune`, stabilizing throughput to ±1.6% across machine reboots.
5. **DFlash2 In-Place Quantized `lm_head` Execution**: Solves a critical out-of-memory crash on GB10 where standard DFlash2 candidate selection dequantized the full 152k vocabulary `lm_head` into dense BF16 (allocating 2.5–5 GB during CUDA graph capture). Executes `lm_head.quant_method.apply` in place on NVFP4 weights with contiguous logit masking and radix top-k.

---

## Subsystem Coverage & Architecture Analysis

### 1. NVMe mmap PLE Table Backing (`flash-sglang/qwen4_exp.py`)

Qwen3.8-Flash-Next incorporates a 51-billion parameter Position-Linked Embedding (PLE) n-gram table requiring approximately 48 GiB of storage. Under the default SGLang and vLLM runtimes, this table must reside in pinned host RAM. On a unified-memory workstation with 128 GB shared between CPU and GPU, allocating 48 GB to pinned host RAM starves the GPU KV cache, restricting context window to less than 32k tokens.

In `flash-sglang/qwen4_exp.py:750-875@a08dea9cd8cda4cee61c007d4176f7b97908de8e`, the author introduces a file-backed memory-mapped store:

```python
# flash-sglang/qwen4_exp.py:805-834
def _alloc_ple_table(shape, dtype):
    ...
    _PLE_MMAP_DIR = os.environ.get("SGLANG_QWEN4_PLE_MMAP_DIR", "").strip()
    if not _PLE_MMAP_DIR:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    os.makedirs(_PLE_MMAP_DIR, exist_ok=True)
    path = os.path.join(_PLE_MMAP_DIR, "ple_table_%d_%d.bin" % (numel, nbytes))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        with open(path, "wb") as f:
            f.truncate(nbytes)
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
```

#### Memory Management & `madvise` Discipline
A naive `mmap` on Linux causes catastrophic I/O thrashing during random token gathering due to default kernel readahead pulling in 128 KiB–256 KiB pages for every 160-byte row access. The author implements a dual-mode access pattern:
1. **Population Phase (`MADV_SEQUENTIAL`)**: During initial table generation at boot, `_ple_madvise(storage.data_ptr(), nbytes, 2, "MADV_SEQUENTIAL")` enables sequential kernel prefetching while rows are populated.
2. **Durable Marker Synchronization (`msync` + Atomic Rename)**: Upon completion, `_ple_table_complete()` calls `libc.msync(..., MS_SYNC)` and writes an atomic sidecar file `path + ".complete"` containing `tag = "{MODEL_REV}|{numel}|{nbytes}|{dtype}"`.
3. **Serving Phase (`MADV_RANDOM`)**: When the sidecar tag matches the model revision on subsequent boots, table population is completely skipped. `_ple_madvise(storage.data_ptr(), nbytes, 1, "MADV_RANDOM")` is invoked, forcing the kernel to fault in only the specific 4 KiB page containing the requested 160-byte row. Upstream measurements confirmed a **~560× reduction in disk I/O traffic per token**.
4. **Coherent Memory Dereferencing**: On the NVIDIA GB10 Grace Blackwell architecture, the CPU and GPU share a unified physical address space over high-speed NVLink C2C. The GPU CUDA gather kernel directly dereferences the pageable host virtual address; the Linux page cache handles paging rows on demand without requiring explicit pinned residency.

---

### 2. SM121 QSA Sparse-Decode Triton Kernel (`flash-sglang/sm121_varlen.py`)

In Qwen3.8-Flash-Next, 11 of the 45 layers utilize Qwen Sparse Attention (QSA). The QSA decode backend compacts each request's selected KV rows into a packed buffer and issues one query row per request (`max_seqlen_q=1`).

#### The Silent Corruption Failure
On SM121 (Blackwell GB10), FlashAttention-4's CuTe varlen kernel fails to compile for this specific decode call shape. In early revisions (v1.5), operators bypassed this by widening the FlashInfer TRT-LLM sparse-decode gate from `is_sm100_supported()` to include SM120/SM121. As discovered by `hashd1ve` and documented in `flash-sglang/ATTRIBUTION.md:72-89`, this routed decode to FlashInfer's XQA kernel, which silently corrupted KV cache state at long sequence lengths:
- **120k tokens:** 1 in 4 requests corrupted into repeating sequences of token ID 0 (`!`).
- **210k tokens:** 4 in 4 requests corrupted completely into token ID 0.

#### The Dedicated Triton Varlen Fix
`flash-sglang/sm121_varlen.py:22-102` implements a dedicated, numerically stable Triton kernel (`_qsa_one_query_varlen_kernel`) specifically for the packed one-query contract:

```python
# flash-sglang/sm121_varlen.py:22-47
@triton.jit
def _qsa_one_query_varlen_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, PADDED_HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr, ...
):
    sequence_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    query_idx = tl.load(cu_seqlens_q_ptr + sequence_idx)
    kv_start = tl.load(cu_seqlens_k_ptr + sequence_idx)
    kv_end = tl.load(cu_seqlens_k_ptr + sequence_idx + 1)
```

In `flash-sglang/qwen_sparse_attn_backend.py:74-100`, the resolver dynamically routes SM121 decode:
- When inputs conform to the KDA contract (BF16, head dim 256, 24Q/2KV or 12Q/1KV, batch $\le 128$, selected KV $\le 2055$), it executes the optimized `qwen38_qsa_sm121` split-K kernel (SGLang PR #36845).
- For all other tensor layouts, it falls back to `qsa_sm121_varlen_attention` in `sm121_varlen.py`.
- Empirical validation on the reference box confirmed **11/11 exact needle retrievals at 120k prompt tokens**, zero token-0 corruption, and decode rates of 38.8 tok/s (code) and 36.7 tok/s (math).

---

### 3. Keepalive & Cancellation Reverse Proxy (`keepalive-proxy.py`)

The proxy sits on port 30001 in front of SGLang (port 30000). It operates without modifying payload semantics or logging prompt text, addressing three fatal failure modes in production agent systems.

#### A. Preventing Client Stall Timeouts During Tool Call Generation
When an LLM generates a tool call, SGLang's tool-call parser buffers arguments until the full JSON object is parsed. On long file edits (e.g. generating a 400-line diff or file write), the proxy measured **127 seconds of complete network silence**. Standard agent harnesses (Claude Code, OpenCode, Vercel AI SDK) feature aggressive stall detectors that drop connections after 120–180 seconds of silence.

Crucially, the author discovered that naive SSE comments crash modern clients:
- **Claude / Anthropic dialect (`/v1/messages`)**: Sending an SSE comment (`: keepalive\n\n`) causes Claude-family parsers to crash 10–20 seconds after receipt. The proxy instead injects the official Anthropic ping event:
  ```python
  # keepalive-proxy.py:498
  b'event: ping\ndata: {"type": "ping"}\n\n'
  ```
- **OpenAI dialect (`/v1/chat/completions`)**: OpenCode and AI SDK stall detectors disregard SSE comments entirely, dropping the stream anyway. The proxy injects an authentic empty chunk:
  ```python
  # keepalive-proxy.py:499
  b'data: {"id":"keepalive","object":"chat.completion.chunk","created":0,"model":"keepalive","choices":[]}\n\n'
  ```
- **Boundary Safety**: Keepalives are injected strictly on `\n\n` SSE event boundaries via a buffered worker thread (`pump()` using `read1(8192)`). Injecting keepalives mid-event corrupts JSON framing.

#### B. Active Zombie Generation Abort (`/abort_request`)
In SGLang, closing a client TCP connection does not immediately terminate an in-flight autoregressive decode; the engine continues generating tokens until `max_tokens` is exhausted. On a single-worker workstation, a disconnected client leaves a "zombie generation" monopolizing 100% of the GPU compute for minutes.

`keepalive-proxy.py:403-420` captures the engine's internal request ID (`_rid`) from the initial SSE event (`id` field). When a client disconnects (`BrokenPipeError`, `ConnectionResetError`, or timeout), the proxy immediately spawns a daemon thread to issue an explicit cancel:
```python
# keepalive-proxy.py:403-420
def _abort_upstream(self, why: str):
    rid = getattr(self, "_rid", None)
    if not rid:
        return
    def go():
        try:
            hdrs = {"Content-Type": "application/json"}
            auth = self.headers.get("Authorization")
            if auth: hdrs["Authorization"] = auth
            req = urllib.request.Request(UPSTREAM + "/abort_request",
                                         json.dumps({"rid": rid}).encode(), hdrs)
            urllib.request.urlopen(req, timeout=5).read()
            log(f"aborted upstream rid={rid} ({why})")
        except Exception as e:
            log(f"abort_request failed for rid={rid}: {e}")
    threading.Thread(target=go, daemon=True).start()
```

#### C. Mid-Stream Decode Corruption Circuit Breaker
If an underlying kernel experiences state corruption or livelock, Qwen models emit endless runs of token ID 0 (`!`). `keepalive-proxy.py:382-401` monitors the text delta across streaming events:
- If consecutive `!` characters exceed `CORRUPTION_RUN = 128`, it aborts upstream generation via `/abort_request`.
- Emits a typed error payload:
  ```json
  {"error": {"type": "corrupted_output", "code": "corrupted_output", "message": "keepalive-proxy: the engine emitted 128 consecutive '!' characters..."}}
  ```
  followed by `data: [DONE]\n\n`, cleanly terminating client execution instead of returning garbage.

#### D. Oversize Request Pre-Gate & Scheduler Wedge Protection
In SGLang, submitting a prompt that exceeds the available KV pool does not return a clean HTTP 400; it wedges the internal scheduler into an unrecoverable queue lock (`#queue-req: 1, #running-req: 0`), rendering `/abort_request` unresponsive (`not found in rid_to_state`, SGLang issue #36333).
The proxy queries `/get_server_info` to cache `max_total_num_tokens`, invalidates the cache immediately if the upstream is unreachable or restarts, and validates incoming prompts against `prompt_limit(pool)` (using the engine's `/tokenize` endpoint when payload size exceeds 200 KB). Over-budget requests receive an immediate HTTP 400 `context_too_long`, protecting the backend scheduler from deadlocking.

---

### 4. DFlash2 Draft Block 8 Tuning & Quantized `lm_head` Fix (`dflash2/`)

For the 27B model, `hasso5703/dgx-spark-qwen38` deploys **DFlash2** block-diffusion speculative decoding (`z-lab/Qwen3.8-27B-DFlash2`), delivering 50 tok/s greedy median and 148 tok/s at 8 concurrent streams.

#### The Dense Dequantization OOM Hazard
In the reference DFlash2 implementation, the draft model computes candidate tokens by taking the target model's `lm_head` and dequantizing it to a dense BF16/FP32 matrix. For a 152,064 vocabulary model, allocating this dense matrix during CUDA graph capture requires 2.5–5 GB of contiguous VRAM. On GB10 unified memory, this allocation spiked total usage past earlyoom limits, causing hard system reboots.

#### The In-Place Quantized Candidate Selector
`dflash2/sglang/srt/models/dflash.py:940-955@a08dea9cd8cda4cee61c007d4176f7b97908de8e` patches the candidate generator to execute directly against the quantized NVFP4 linear weights:

```python
# dflash2/sglang/srt/models/dflash.py:940-955
weight = getattr(lm_head, "weight", None)
quant_method = getattr(lm_head, "quant_method", None)
if should_apply_lm_head_quant_method(lm_head, quant_method):
    # Upstream fix (sglang #35496): flashinfer's radix top-k must not see a
    # cropped (non-contiguous) view of the padded local vocab; keep the
    # logits contiguous and mask the padded tail out of the top-k instead.
    local_logits = quant_method.apply(lm_head, hidden, None).contiguous()
    if local_logits.shape[-1] > num_org:
        local_logits[:, num_org:] = float("-inf")
elif is_dense_head_weight(weight):
    local_logits = torch.matmul(hidden.to(weight.dtype), weight[:num_org].T)
```
- In multi-GPU / Tensor Parallel mode, each rank performs a local top-k over its vocab shard, all-gathers only $K$ logits/IDs rather than the entire vocabulary, and performs a global top-k reduction, cutting communication from $O(\text{vocab})$ to $O(\text{TP} \times K)$.
- The draft block parameter is pinned to **Block 8**. Empirical sweeps by `r0b0tlab` and `hasso5703` proved that draft block 8 achieves optimal acceptance, whereas block 9 experiences catastrophic acceptance collapse.

---

### 5. Deployment Configurations & Benchmarking Methodology

#### Host Memory Settle Loop (`qwen38-flash-launch.sh.template:34-55`)
On unified memory, SGLang measures available memory at the exact microsecond of engine initialization to compute the KV token pool size. If background processes or a previous container teardown are still releasing memory, the engine under-allocates the pool for the lifetime of that container.
The launcher loops for up to 180 seconds inspecting `/proc/meminfo` until `MemAvailable >= 96 GiB`. Furthermore, the server pins `--max-total-tokens 190000` (189,952 post-alignment), converting a variable boot allocation into a deterministic, frozen KV capacity.

#### Poisoned-Table Recovery Protocol (`qwen38-flash-launch.sh.template:18-32`)
Writing a 48 GiB PLE backing file takes several minutes. If a server reboot or power failure interrupts the write, the half-written table causes subsequent boots to spin in an infinite kernel hang. The startup script touches `$PLE_DIR/.loading` prior to initialization and spawns a background monitor that clears `.loading` only when `/health` returns HTTP 200. If `.loading` is detected at startup, the script automatically purges `ple_table_*` and initiates a clean rebuild.

#### Thinking-Aware Benchmarking (`bench.sh:59-69`)
Standard benchmarking scripts measure time-to-first-token by checking `delta.content`. However, Qwen3.8 emits thinking tokens in `delta.reasoning_content` (SGLang) or `delta.reasoning` (vLLM). Naive scripts ignore the thinking phase, starting the timer only when markdown text begins while counting the entire token total, artificially inflating calculated tok/s by 300–500%. `bench.sh` explicitly monitors both reasoning and content channels.

---

## Author Worldview Reconstruction & Systems Tradeoffs

### 1. Single-Node Workstation Reality vs Hyperscale Abstractions
Hyperscale serving frameworks (vLLM, Dynamo, Mooncake) assume homogeneous data-center nodes with discrete GPU VRAM (e.g. H100 SXM5 with 80 GB dedicated HBM3), out-of-band RDMA fabrics, and independent front-end load balancers.
The author's worldview is anchored in the reality of **workstation-class Unified Memory Architecture (NVIDIA DGX Spark / GB10)**:
- Memory is a single shared pool of 128 GB LPDDR5X. An allocation spike on the GPU directly starves the Linux kernel, systemd, SSH, and developer tools.
- SGLang's internal memory profiler fails to account for 25–40 GB of transient allocations during FlashInfer FP8 autotuning and CUDA graph capture.
- Running `--mem-fraction-static 0.80` (the official SGLang cookbook setting) causes instant out-of-memory kernel panics under multi-client concurrency. The author enforces `--mem-fraction-static 0.50` (native) or `0.70` (1m context mode) inside a strict Docker `--memory 100g` container cgroup.

### 2. Eliminating the FlashInfer "Boot Lottery"
In standard deployments, FlashInfer runs an online empirical autotuner on every engine boot, benchmarking micro-kernels across arbitrary input sizes and caching the selection to disk. On GB10, however, the disk cache was advisory; subtle timing variations resulted in different kernel selections across boots.
- **The Empirical Finding:** Across identical hardware and identical benchmarks, aggregate 8-stream throughput fluctuated between **92 tok/s and 111 tok/s** purely based on the autotuner's selection lottery.
- **The Solution:** Passing `--disable-flashinfer-autotune` stabilizes throughput across boots to **100–104 tok/s (±1.6%)** and cuts container boot times by approximately 2 minutes.

### 3. Agent Harness Pragmatism
The author views the LLM server not as an isolated benchmark target, but as a component embedded inside an active coding loop (specifically driving OpenCode and Claude Code):
- An engine that wedges under context overflow is unacceptable; it must fail fast with HTTP 400.
- Tool argument generation causes long silent pauses; the transport layer must emit dialect-compliant heartbeats.
- Client disconnects must actively kill GPU kernels; running zombie decodes wastes the workstation's only accelerator.

### 4. Speculative Decoding: The Floating-Point Divergence Truth
A critical insight in `BENCHMARKS.md:339-369` debunks community claims regarding "lossless" speculative decoding. While speculative verification is mathematically lossless under infinite precision, in IEEE-754 floating-point arithmetic, block verification reorders reduction trees. This creates minor numerical variations ($\approx 10^{-6}$) that flip near-tie argmax decisions early in the reasoning trace.
The author proved that both DSpark and DFlash2 diverge from pure autoregressive generation on 10 out of 10 prompts within 2% to 33% of the generation length. However, large-n evaluations (GSM8K at 188/200, IFEval at 81.4%) demonstrate that this represents normal sampling within the model's numeric manifold rather than quality degradation.

---

## Concrete Borrow Candidates Grounded in Source Code

The following 5 borrow candidates represent high-value architectural techniques directly applicable to the `fak` kernel:

### Candidate 1: Dialect-Conforming SSE Stream Keepalives (Anthropic Ping & OpenAI Empty Chunk)

- **Source Anchor:** `keepalive-proxy.py:494-517@a08dea9cd8cda4cee61c007d4176f7b97908de8e`
- **The Specific Axis:** Transport-level harness liveness signaling during tool-call argument buffering without violating client parser grammar.
- **Their Tradeoff & Context:** SGLang buffers tool arguments for up to 127 seconds. Emitting raw SSE comment lines (`: keepalive`) crashes Claude Code / Claude-family parsers, while OpenCode / AI SDK stall detectors ignore comments entirely and abort after 140s. The author injects official `event: ping\ndata: {"type": "ping"}\n\n` for Anthropic and `data: {"choices":[]}\n\n` for OpenAI at 10s intervals, strictly at event boundaries.
- **`fak` Comparison & Seam:**
  - `internal/gateway/messages_stream_planner.go:146-165` implements Anthropic stream pinging (`anthropicStreamPingInterval`).
  - However, in `internal/gateway/stream_proxy.go:209-254` (`streamChatLive` for OpenAI `/v1/chat/completions`), `fak` emits only comment frames (`: fak-heartbeat {...}`), which are ignored by OpenCode/AI SDK and choke Claude parsers. Furthermore, `streamChatLive` suppresses heartbeats prior to the first token, meaning tool argument buffering on an upstream proxy still triggers client timeouts!
- **Disposition:** `INSPIRE-ONLY` (clean-room Go in `internal/gateway/stream_proxy.go`).
- **Witness on Axis:** `PARTIAL-on-axis`.

---

### Candidate 2: Active Upstream Request Cancellation on Client Disconnect (`/abort_request`)

- **Source Anchor:** `keepalive-proxy.py:403-420@a08dea9cd8cda4cee61c007d4176f7b97908de8e`
- **The Specific Axis:** Elimination of GPU compute waste from orphaned "zombie" generations following client disconnection.
- **Their Tradeoff & Context:** Closing an HTTP client socket to an inference engine (SGLang/vLLM) does not abort model execution; the server decodes until `max_tokens`. The author captures the upstream request ID from the initial chunk and issues an asynchronous POST to `/abort_request` immediately upon connection termination.
- **`fak` Comparison & Seam:**
  - In `internal/gateway/stream_proxy.go` and `internal/gateway/gateway.go`, when `ctx.Done()` fires or client write returns an error, `fak` halts its local reader loop, but never issues an explicit `/abort_request` or cancellation webhook to upstream backend engines (`BaseURL` proxy routes).
- **Disposition:** `INSPIRE-ONLY` (clean-room Go in `internal/gateway`).
- **Witness on Axis:** `ABSENT-on-axis`.

---

### Candidate 3: Streaming Decode Corruption Tripwire for Token-0 Runaway (`CORRUPTION_RUN=128`)

- **Source Anchor:** `keepalive-proxy.py:88-89,243-250,382-402,530-541@a08dea9cd8cda4cee61c007d4176f7b97908de8e`
- **The Specific Axis:** In-stream circuit-breaking and graceful failure recovery for hardware/kernel state degradation.
- **Their Tradeoff & Context:** When SM121 kernels or low-level CUDA graph states corrupt, Qwen models emit endless runs of token ID 0 (`!`). Rather than letting a user or harness ingest thousands of exclamation marks, the proxy tracks consecutive marker characters across stream chunks, aborts upstream, and emits a structured `corrupted_output` error event followed by `data: [DONE]`.
- **`fak` Comparison & Seam:**
  - `fak` has `internal/answershape/answershape.go` to detect repetitive text. However, `answershape.go:36-40` *explicitly exempts* non-alphanumeric runs ("a run or tiling of PURELY non-alphanumeric fill characters... is structural formatting, not a loop"). Thus, `answershape` is blind to token-0 runaway (`!!!!!...`).
  - Moreover, `answershape` operates post-hoc on completed strings or write buffers, whereas `keepalive-proxy.py` operates live on streaming deltas.
- **Disposition:** `INSPIRE-ONLY` (clean-room Go in `internal/gateway/stream_proxy.go` or `internal/answershape`).
- **Witness on Axis:** `ABSENT-on-axis`.

---

### Candidate 4: Dual-Pattern Memory-Mapped Large Table Backing with Completion Sidecars

- **Source Anchor:** `flash-sglang/qwen4_exp.py:760-875@a08dea9cd8cda4cee61c007d4176f7b97908de8e`
- **The Specific Axis:** Zero-copy large weight/table offloading on unified memory architectures without pinned host RAM residency.
- **Their Tradeoff & Context:** Pinned host RAM consumes physical memory on unified architectures. Mapping via `torch.from_file` with `MADV_SEQUENTIAL` for builder population, `msync(MS_SYNC)` plus atomic `.complete` sidecar marker with SHA verification, and `MADV_RANDOM` for runtime inference reduces disk I/O by 560× while preserving host memory for the KV cache.
- **`fak` Comparison & Seam:**
  - `internal/metalgemm/q4k.go:208-256` supports mapped spans on Apple Silicon (`UploadQ4KMappedSpan`), and `cmd/modelbench/qwen38_paged_swap.go` models paged swapping. However, `fak` lacks persistent sidecar-verified table caching with phase-aware `madvise` policies (`MADV_SEQUENTIAL` $\to$ `MADV_RANDOM`) for auxiliary tables (e.g. n-gram tables, large router tables).
- **Disposition:** `INSPIRE-ONLY` (clean-room Go/C).
- **Witness on Axis:** `PARTIAL-on-axis`.

---

### Candidate 5: In-Place Quantized Target `lm_head` Application for Speculative Drafters

- **Source Anchor:** `dflash2/sglang/srt/models/dflash.py:940-955@a08dea9cd8cda4cee61c007d4176f7b97908de8e`
- **The Specific Axis:** Memory footprint and CUDA graph capture safety during speculative candidate generation on low-precision targets.
- **Their Tradeoff & Context:** Dense dequantization of the target model's 152k vocabulary `lm_head` during draft graph capture allocates 2.5–5 GB VRAM, triggering system reboots on GB10. Invoking `lm_head.quant_method.apply` in place on NVFP4 weights with contiguous logit masking and radix top-k avoids dequantization completely.
- **`fak` Comparison & Seam:**
  - `fak`'s speculative decoding kernels (`cmd/fak/model_qwen38_ladder.go`, `internal/polymodel/polymodel.go:410-450`, `internal/metalgemm`). When implementing speculative drafting over low-precision models, `fak` must enforce in-place quantized projection rather than transient dequantization.
- **Disposition:** `INSPIRE-ONLY` (clean-room Go/CUDA/Metal).
- **Witness on Axis:** `PARTIAL-on-axis`.

---

## Empirical Benchmarks & Hardware Receipts

The following empirical measurements were captured directly from the reference DGX Spark (GB10 128 GB) environment running revision `a08dea9cd8cda4cee61c007d4176f7b97908de8e`:

### 1. Speculative Decoding Throughput (Qwen3.8-27B NVFP4, Batch 1 Decode)

| Workload Domain | DFlash2 (v1.2+, Block 8) | DSpark (v1.1) | Stable-MTP Baseline |
|---|---|---|---|
| **Agentic Coding (Code, Diffs, Tools)** | **32–40 tok/s** (bench.sh: 41–47) | 28–36 tok/s | 24–28 tok/s |
| **Math & Structured Reasoning** | **41–44 tok/s** (bench.sh: 52–57, peak 60) | 38–42 tok/s | 24–33 tok/s |
| **Technical Explanations (French)** | **26 tok/s** | 23–25 tok/s | ~22 tok/s |
| **Free-Form Prose (EN / FR / DE)** | **22 / 20 / 17 tok/s** | 17 / 14 / 13 tok/s | 17–20 tok/s |
| **8 Concurrent Streams (Aggregate)** | **135–148 tok/s** | 100–104 tok/s | ~92 tok/s |
| **32 Concurrent Streams (Aggregate)** | **258 tok/s** | Not measured | Not measured |

### 2. Qwen3.8-Flash-Next 176B on Single DGX Spark GB10

| Measurement Axis | Measured Receipt | Notes |
|---|---|---|
| **Prefix Cache Re-serve (30K tokens)** | **18.4 s cold $\to$ 0.5 s cached (36.8×)** | SGLang Radix Cache with Mamba extra buffer |
| **Agent Turn (30K prefix + fresh query)** | **~3.0 s (5.8× speedup)** | Typical multi-turn agentic conversational shape |
| **Decode Throughput (Reasoning)** | **34.2 tok/s** (up to 42 on short context) | Speculative NEXTN (MTP BF16, 3 steps, top-k 1) |
| **Decode Throughput (Free Prose)** | **20.3 tok/s** | Unassisted autoregressive floor |
| **Cold Prefill Rate** | **~1,500–2,000 tok/s** | Triton prefill attention backend |
| **Long Context Needle Retrieval (`needle.sh`)** | **11/11 exact retrievals at 120k tokens** | Host memory floor 14.6 GiB (9 GiB below idle) |
| **Maximum Single-Prompt Context** | **128,000 tokens** | Hard ceiling enforced by proxy `PROMPT_CEILING_TOKENS` |

### 3. Elimination of FlashInfer Autotune Variance

| Configuration | 8-Stream Throughput Range | Inter-Boot Variance | Boot Time |
|---|---|---|---|
| Default FlashInfer Autotune | 92.0 – 111.0 tok/s | ±15.0% | ~7–9 minutes |
| `--disable-flashinfer-autotune` | **100.0 – 104.0 tok/s** | **±1.6%** | **~5–7 minutes (-2 min)** |

---

## License Disposition & Legal Provenance

1. **Root Repository License**: MIT License, Copyright (c) 2026 hasso5703 (`LICENSE:1-21@a08dea9cd8cda4cee61c007d4176f7b97908de8e`).
2. **Vendored Overlay Patches**:
   - `flash-sglang/`: MIT patches by hashd1ve over Apache-2.0 SGLang base.
   - `dflash2/`: MIT quantized `lm_head` patch by MiaAI-Lab over Apache-2.0 SGLang sources.
   - `flash-sglang/kda_kernels/`: Apache-2.0 kernel implementation by BBuf (merged upstream in SGLang PR #36845).
3. **FAK Licensing Compatibility**:
   - `fak` is licensed under Apache-2.0. Both MIT and Apache-2.0 upstream licenses are fully compatible with `fak`.
   - In accordance with `fak` contributor doctrine, all candidates are designated as **`INSPIRE-ONLY`**. Clean-room Go implementations within `internal/gateway` and `internal/answershape` will borrow the mathematical mechanisms and architectural principles with source attribution, vendoring zero foreign Python or Triton code.

---

## Concrete Follow-up Implementation Tickets

- Issue: `feat(gateway): dialect-conforming SSE stream keepalives (Anthropic ping & OpenAI empty chunk)` (Candidate 1)
- Issue: `feat(gateway): active upstream request cancellation (/abort_request) on client socket disconnect` (Candidate 2)
- Issue: `feat(gateway): streaming decode corruption tripwire for token-0 runaway (CORRUPTION_RUN=128)` (Candidate 3)
- Issue: `feat(model): dual-pattern memory-mapped large table backing with completion sidecars` (Candidate 4)
- Issue: `feat(compute): in-place quantized target lm_head application for speculative drafters` (Candidate 5)

---

## Registration Trail & Index Updates

- **Durable Study Note:** `docs/notes/CONCEPT-STUDY-HASSO5703-DGX-SPARK-2026-09-03.md` (this file).
- **Monitored Registry:** Added entry for `hasso5703/dgx-spark-qwen38` at revision `a08dea9cd8cda4cee61c007d4176f7b97908de8e` in `docs/research/monitored-repositories.json`.
- **Master Documentation Index:** Linked from `INDEX.md` under *Notes & research (`docs/notes/`)*.
