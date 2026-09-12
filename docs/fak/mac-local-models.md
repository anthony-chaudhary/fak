---
title: "Run local models on Mac (Apple Silicon Metal) and interactive chat"
description: "How to run local models (Qwen3.8-27B and peers) natively on Apple Silicon with Metal GPU acceleration and chat via interactive REPL or OpenAI-compatible gateway."
---

# Run local models on Mac (Apple Silicon Metal) and interactive chat

**fak** runs open models directly on Apple Silicon unified memory using native Metal compute kernels. No Python, no PyTorch, and no separate runtime daemon required — one native binary runs the model, accelerates KV-cache prefix reuse, and secures tool execution behind a default-deny capability floor.

> **TL;DR — Two ways to chat with Qwen3.8 on macOS:**
>
> 1. **Instant Interactive REPL (daemon-less, single command):**
>    ```bash
>    fak run qwen38
>    ```
> 2. **Metal GPU-Accelerated Server + Chat Client (zero extra flags required):**
>    ```bash
>    # Terminal 1: launch Metal GPU server (Metal & resident Q4_K auto-selected)
>    fak serve --gguf qwen38:27b-q4
>
>    # Terminal 2: chat through the gateway (auto-connects to :8080 and detects model)
>    fak chat
>    ```

---

## Build and qualify the native Metal binary

Metal support is compiled into the native `darwin/arm64` build with CGo enabled.
The installer now selects the Metal archive by default on Apple Silicon; CPU
archives require explicit `--variant cpu` and cannot pass Metal qualification.
The v0.54.0 Metal archive is available as a backfill from its exact tagged source,
without creating a new release. On Apple Silicon, use `FAK_VERSION=0.54.0 sh install.sh`
to select it. Missing assets for other versions produce an actionable error.
Vulkan publication remains pending its release and hardware gate. The commands
below build and qualify the native Metal backend from source. MLX remains a
comparison runtime.
Install Xcode Command Line Tools and Go 1.26+, then build from a clean
committed checkout (or a managed worker) to a unique temporary path and inspect the actual
Mac and backend:
A pure-Go binary never fails silently: `fak serve`, `fak run`, and `fak up` stamp a
`backend=cpu (metal-not-compiled)` or `backend=cpu (metal-no-device)` skip-reason line at startup, and
`fak gpucheck -backend metal` exits non-zero with the same two-state reason.

```bash
xcode-select -p
go version
system_profiler SPHardwareDataType
sysctl -n hw.memsize

FAK_BIN="$(mktemp -d)/fak"
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 \
  go build -trimpath -o "${FAK_BIN}" ./cmd/fak
file "${FAK_BIN}"
"${FAK_BIN}" hil --probe --json
"${FAK_BIN}" hil --json
```

The discovery probe passes when it reports `kind: "metal"`, `architecture: "arm64"`,
`physical_available: true`, `memory_unified: true`, the device name, and a non-zero Metal
working-set size. The second command executes the physical micro-dose suite and must report
`all_passed: true`. `mps_available` may be false; the Q4_K native shader path does not require
MPS.

Then run a fail-loud native server smoke. Use the weight estimate below to plan capacity,
then confirm total process headroom on the intended Mac and workload:

```bash
# Terminal 1
"${FAK_BIN}" serve --gguf qwen38:27b-q4 --metal --addr 127.0.0.1:18080

# Terminal 2
curl -fsS http://127.0.0.1:18080/healthz
curl -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38:27b-q4","messages":[{"role":"user","content":"Reply with ready."}],"max_tokens":1,"temperature":0,"fak":{"native_inference_receipt":true}}'
```

Acceptance requires a successful health response with `engine: "inkernel"` and a response
receipt under `fak.native_inference_receipt` with `backend: "metal"`, a non-empty
`forward_path`, and `fallback_active: false`.

FAK detects the Metal device and its usable working-set size at runtime rather than matching
an M-series product name. Apple's [Metal feature-set tables](https://developer.apple.com/metal/capabilities/)
map M3 to Apple GPU family 9 and M5 to family 10, so M5 compatibility is expected from the
runtime contract. That is not a performance result. A base M5 cannot be assumed to equal or
beat every M3 Pro or Max: GPU configuration, memory bandwidth and capacity, cooling, and the
workload all affect the result.

For a performance claim, measure both physical Macs with the same source revision and build
flags, model artifact digest and quantization, prompt/context/output-token envelope, sampling,
concurrency, warm-up/repetition policy, and power state. Record the hardware probe and native
receipt for each run. Physical M5 qualification is tracked in
[#12681](https://github.com/anthony-chaudhary/fak/issues/12681); until it lands, retain the M3
figures later in this guide as historical M3 evidence.

---

## 1. Unified memory model sizing

Apple Silicon uses unified memory shared between the CPU, GPU, operating system, and other
applications. The values below are approximate model-weight sizes. They are useful download
and first-pass capacity estimates, but they do not measure total process footprint or prove
that a model will run without swap.

| Model Alias | Quantization | Approximate Model Weights | What This Estimate Covers |
|---|---|---:|---|
| `qwen2.5-coder:3b` or `qwen2.5-coder:7b` | Q4_K_M | 1.8 GB - 4.5 GB | Model weights only |
| `qwen38` (`qwen38:27b-q2k`) | UD-Q2_K_XL | 9.4 GB | Model weights only |
| `qwen38:27b-q4` | Q4_K_M | ~17.1 GB (15.93 GiB) | Pinned artifact weights only; context length and concurrency add runtime memory |

Allow additional memory for the KV cache, Metal buffers, command and scratch allocations,
the Go heap, model loading, macOS, and other applications. Confirm the ready-state and peak
process footprint on the intended model, context length, concurrency, and hardware before
treating an estimate as an operating envelope.

A 2026-09-09 native-v2 diagnostic candidate illustrates the gap. On a physical M3 Pro with
36 GiB, Qwen3.8-27B Q4_K_M at 4K context reached a ready-state process footprint of
29,821,769,400 bytes (27.77 GiB), above its 24 GiB candidate review limit, before any chat
request. This was not a clean-trunk result: binary digest prefix `504d37fb` came from base
`7a62f603d` plus an uncommitted managed candidate bundle. It does not establish M5 memory or
performance. Runtime memory investigation is tracked in
[#12684](https://github.com/anthony-chaudhary/fak/issues/12684), and physical M5 qualification
remains [#12681](https://github.com/anthony-chaudhary/fak/issues/12681).

---

## 2. Inspect and pull models

Check available model aliases, local cache state, and download progress with `fak`'s built-in discovery tools:

```bash
# List all known model aliases and whether they are cached locally
fak ls

# Inspect the canonical default model identity (Qwen3.8-27B Q4_K_M)
fak model-default

# Pull weights into local cache on demand (resumable)
fak pull qwen38          # Qwen3.8-27B UD-Q2_K_XL (~9.4 GB)
fak pull qwen38:27b-q4   # Qwen3.8-27B Q4_K_M (~17.1 GB)
```

Downloads are stored in `~/.cache/fak-models/hub/` or `~/Library/Caches/fak-models/hub/`.

---

## 3. Option A: Instant interactive REPL (`fak run`)

`fak run <model>` provides an Ollama-style chat REPL without launching a background server or network listener:

```bash
# Interactive multi-turn chat (Ctrl-D or Ctrl-C to exit)
fak run qwen38

# Or with the approximately 17.1 GB Q4_K_M model
fak run qwen38:27b-q4

# One-shot command-line query
fak run qwen38 "Explain RadixAttention prefix caching in one sentence"
```

### Witnessed prefix cache reuse
On every turn, `fak run` prints the kernel's **witnessed KV-prefix reuse** to `stderr` (keeping `stdout` pipe-clean):
```text
  cache: reused 128/160 prompt tok (80% frozen, by=vdso) — computed 32
```
Subsequent conversation turns automatically reuse the prior conversation context from cache.

---

## 4. Option B: Metal GPU server + chat (`fak serve` + `fak chat`)

To serve through native Metal on Apple Silicon, launch the Metal GPU server. This enables:
- Native Metal GPU prefill and decode kernels (`metalgemm`).
- In-kernel RadixAttention prefix caching for repeated prefixes.
- OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` endpoints.

### Step 1: Start the server

Metal GPU acceleration and resident Q4_K tensor decoding are auto-selected on Apple Silicon, and `--gguf` accepts model aliases directly without manual pulls or environment variables:

```bash
fak serve --gguf qwen38:27b-q4
```

Verify the server is ready:
```bash
curl -s http://127.0.0.1:8080/healthz
# Expected acceptance shape: {"engine":"inkernel","model":"qwen38:27b-q4","ok":true}
```

### Step 2: Chat with the server

Use `fak chat` as an interactive client. When run locally, it auto-connects to the server on `:8080` and auto-detects the loaded model:

```bash
fak chat
```

Or connect explicitly if using custom ports, remote hosts, or model routing:
```bash
fak chat --base-url http://127.0.0.1:8080/v1 --model qwen38:27b-q4
```

Or query with standard `curl`:
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38:27b-q4",
    "messages": [{"role": "user", "content": "Write a Go function to reverse a slice."}]
  }'
```

### Step 3: Run Pi coding agent directly against the Metal server

Connect the [Pi](https://github.com/earendil-works/pi) terminal coding agent to `fak serve` ("raw" without guard):

```bash
# Terminal 1: start server and write ~/.pi/agent/models.json
fak serve --gguf qwen38:27b-q4 --pi

# Terminal 2: launch Pi connected directly to the backend
fak pi
# or run pi CLI directly: pi --provider fak --model qwen38:27b-q4
```

---

## 5. Option C: Turnkey Apple Silicon Auto-Provisioner (`fak up`)

`fak up` probes your Mac's unified memory using `macfit`, estimates a model tier and
reservation headroom, and starts the server with an interactive REPL. The plan is an
estimate; it does not prove runtime peak memory or guarantee that macOS will not swap. Confirm
the loaded model's process footprint and memory pressure under the intended context and
concurrency before relying on the estimated headroom:

```bash
fak up
```

Flags:
- `fak up --dry-run`: profile unified memory and print the allocation plan without loading weights.
- `fak up --headless`: run the background server without opening the terminal REPL.
- `fak up --model 27B`: override auto-selected model tier.

---

## 6. Option D: Protect coding agents with local Qwen3.8 (`fak guard`)

Wrap existing coding agents (Claude Code, Codex, OpenCode) with your local Qwen3.8 model. The kernel provides in-memory model execution while enforcing a default-deny capability floor:

```bash
# Guard Claude Code using local Qwen3.8 GGUF
fak guard --gguf qwen38:27b -- claude

# Guard Codex using local Qwen3.8 GGUF
fak guard --gguf qwen38:27b -- codex
```

Every tool call proposed by the agent is adjudicated locally before execution, blocking path escapes and destructive operations.

---

## 7. Option E: Direct Claude Code harness on local Mac backend (`fak claude` — raw, without guard)

If you prefer running Claude Code directly against `fak serve` without the local guard wrapper:

```bash
# Terminal 1: run Metal GPU server with Claude configuration
fak serve --gguf qwen38:27b-q4 --claude

# Terminal 2: launch raw Claude Code connected to fak serve
fak claude
```

Use `fak claude --dry-run` to preview the injected Anthropic environment, or `fak claude config --write` to update `.claude/settings.json`.

---

## 8. Option F: Direct Codex harness on local Mac backend (`fak codex --raw` — raw, without guard)

If you prefer running OpenAI Codex directly against `fak serve` without the local guard wrapper:

```bash
# Terminal 1: run Metal GPU server with Codex configuration
fak serve --gguf qwen38:27b-q4 --codex

# Terminal 2: launch raw Codex connected to fak serve
fak codex --raw
```

Or connect existing Codex CLI installations manually or via config generation:
```bash
# Write or update ~/.codex/config.toml with fak serve backend provider
fak codex config --write

# Or preview the config.toml snippet
fak codex config

# Run single headless probe turn with raw Codex
fak codex --raw --probe "Explain prefix caching in one sentence"

# Dry-run to inspect exact CLI invocation and -c overrides
fak codex --raw --dry-run
```

---

## 9. Performance verification on Mac

To verify Apple Silicon Metal performance on your Mac against committed baseline contracts:

```bash
make mac-perf
```

This runs on-device Metal GEMV/GEMM microbenchmarks and validates the 3-way Mac comparison packet (fak-native vs llama.cpp vs MLX). For full benchmark data and methodology, see:
- [Three-way Mac benchmark (M3 Pro)](../notes/MAC-THREEWAY-BENCH-2026-09-03.md)
- [Mac many-agent shared-prefix cache-value study](../notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md)
- [Qwen3.8-27B latest benchmark details](../benchmarks/QWEN38-27B-LATEST.md)
