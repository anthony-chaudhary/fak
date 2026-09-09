---
title: "Run local models on Mac (Apple Silicon Metal) and interactive chat"
description: "How to run local models (Qwen3.8-27B and peers) natively on Apple Silicon with Metal GPU acceleration and chat via interactive REPL or OpenAI-compatible gateway."
---

# Run local models on Mac (Apple Silicon Metal) and interactive chat

**fak** runs open models directly on Apple Silicon unified memory using native Metal compute kernels. No Python, no PyTorch, and no separate runtime daemon required — one static binary runs the model, accelerates KV-cache prefix reuse, and secures tool execution behind a default-deny capability floor.

> **TL;DR — The two fastest ways to chat with Qwen3.8 right now on macOS:**
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

## 1. Unified memory model sizing

Apple Silicon uses unified memory shared between CPU and GPU. Check your Mac's RAM and pick the matching model tier:

| Mac RAM | Recommended Model Alias | Quantization | Resident Size | Optimal Use |
|---|---|---|---|---|
| **8 GB – 16 GB** | `qwen2.5-coder:3b` or `qwen2.5-coder:7b` | Q4_K_M | 1.8 GB – 4.5 GB | Fast exploration, lightweight coding |
| **32 GB – 36 GB** | `qwen38` (`qwen38:27b-q2k`) | UD-Q2_K_XL | 9.4 GB | Fast 27B inference with ample KV headroom |
| **36 GB – 48 GB** | `qwen38:27b-q4` | Q4_K_M | 16.3 GB | Canonical 27B benchmark weight; high precision |
| **64 GB+** | `qwen38:27b-q4` (large context) | Q4_K_M | 16.3 GB + KV | Large multi-agent concurrency (16+ workers) |

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
fak pull qwen38:27b-q4   # Qwen3.8-27B Q4_K_M (~16.3 GB)
```

Downloads are stored in `~/.cache/fak-models/hub/` or `~/Library/Caches/fak-models/hub/`.

---

## 3. Option A: Instant interactive REPL (`fak run`)

`fak run <model>` provides an Ollama-style chat REPL without launching a background server or network listener:

```bash
# Interactive multi-turn chat (Ctrl-D or Ctrl-C to exit)
fak run qwen38

# Or with the 16.3 GB Q4_K_M model
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

For maximum performance on Apple Silicon, launch the native Metal GPU server. This enables:
- Metal 4 GPU prefill and decode kernels (`metalgemm`).
- In-kernel RadixAttention prefix caching (>190× TTFT speedup on repeated prefixes).
- OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` endpoints.

### Step 1: Start the server

Metal GPU acceleration and resident Q4_K tensor decoding are auto-selected on Apple Silicon, and `--gguf` accepts model aliases directly without manual pulls or environment variables:

```bash
fak serve --gguf qwen38:27b-q4
```

Verify the server is ready:
```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"qwen38:27b-q4","ok":true}
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

`fak up` probes your Mac's unified memory using `macfit`, auto-selects the optimal model tier ensuring at least 20% memory headroom to prevent swap, and starts the server with an interactive REPL:

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
