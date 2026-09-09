---
title: "Pi + fak: Terminal Coding Agent with Mac Metal Backend"
description: "Run Pi (earendil-works/pi) as a terminal coding agent with fak serve on Apple Silicon Mac as backend — native Metal GPU acceleration, in-kernel RadixAttention prefix caching, and one-touch models.json configuration."
---

# Pi + fak Integration Guide

This guide explains how to run [Pi](https://github.com/earendil-works/pi) (by Mario Zechner / Can Bölük) as a terminal coding harness with `fak serve` on macOS (Apple Silicon Metal) as the inference backend ("raw" without guard).

Pi is an extensible, minimal terminal coding harness featuring an interactive TUI, session tree branching (`/tree`), compaction, prompt templates, and the Agent Skills standard. `fak serve` runs local open models (such as Qwen3.8-27B) natively on Apple Silicon unified memory using Metal GPU acceleration, serving an OpenAI-compatible `/v1/chat/completions` API with in-kernel prefix caching for ultra-fast multi-turn TTFT.

---

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Pi Coding Agent                      │
│      (interactive TUI, session tree, skills, tools)     │
└─────────────────────────────────────────────────────────┘
                            │
               OpenAI Chat Completions
               POST http://127.0.0.1:8080/v1/chat/completions
                            ▼
┌─────────────────────────────────────────────────────────┐
│              fak serve (macOS Backend)                  │
│   • Apple Silicon Metal GPU kernels (metalgemm)         │
│   • In-kernel RadixAttention KV prefix caching          │
│   • Native Q4_K / UD-Q2_K_XL tensor execution           │
│   • Zero Python / PyTorch / daemon dependencies         │
└─────────────────────────────────────────────────────────┘
```

**Key benefits of the Pi + fak serve combination:**

- **Native Apple Silicon Metal performance:** Zero-copy unified memory model execution with Metal 4 compute kernels on Apple Silicon (M1/M2/M3/M4).
- **In-kernel prefix caching:** Automatically reuses previous turns' KV-cache tokens across conversation trees, slashing time-to-first-token (>190× TTFT speedup).
- **One-touch configuration:** `fak serve --pi` or `fak pi config --write` automatically registers the `fak` provider in `~/.pi/agent/models.json` without clobbering any existing providers.
- **Raw direct connection:** Direct connection between Pi and the serving engine without guard interception overhead.

---

## Prerequisites

1. **Install fak:**
   ```bash
   git clone https://github.com/anthony-chaudhary/fak && cd fak
   go build -o fak ./cmd/fak
   ```

2. **Install Pi:**
   ```bash
   npm install -g --ignore-scripts @earendil-works/pi-coding-agent
   # or via curl:
   # curl -fsSL https://pi.dev/install.sh | sh
   ```

---

## Workflow A — Dedicated Launcher: `fak pi`

The fastest way to launch Pi connected to `fak serve` backend:

```bash
# 1. Interactive terminal session (auto-configures models.json and connects to fak serve)
fak pi

# 2. Dry-run preview (inspect the exact Pi command without executing)
fak pi --dry-run

# 3. Headless one-shot probe turn
fak pi --probe "Explain Apple Silicon unified memory architecture"

# 4. Explicit model override
fak pi --model qwen38:27b-q4

# 5. Reasoning / thinking level control
fak pi --thinking high
```

`fak pi` automatically:
1. Probes the backend at `http://127.0.0.1:8080/v1` (`/healthz` and `/v1/models`) to verify it is reachable and detect the served model.
2. Updates `~/.pi/agent/models.json` with the `fak` provider and served model ID (preserving all existing providers).
3. Launches `pi --provider fak --model <model>` directly as a child process.

---

## Workflow B — Two-Terminal Serving Flow

If you prefer running a dedicated model server in one terminal and chatting with Pi in another:

### Terminal 1 — Start the Metal GPU server:
```bash
fak serve --gguf qwen38:27b-q4 --pi
```

*(On macOS Apple Silicon, `--gguf` auto-selects Metal GPU acceleration and resident Q4_K tensor decoding. The `--pi` flag writes or updates `~/.pi/agent/models.json` with provider `"fak"` and model `qwen38:27b-q4` before listening on `:8080`).*

Verify the backend is live:
```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"qwen38:27b-q4","ok":true}
```

### Terminal 2 — Run Pi:
```bash
pi --provider fak --model qwen38:27b-q4
# or simply:
fak pi
```

---

## Config Helpers: `fak pi config` & `fak serve --pi-config`

Inspect or write Pi configuration without starting a server or child process:

```bash
# Preview Pi's models.json configuration snippet:
fak pi config

# Preview with custom port or model:
fak pi config --addr 127.0.0.1:9000 --model qwen38:27b-q4

# Write or update ~/.pi/agent/models.json:
fak pi config --write

# Target a custom directory or file:
fak pi config --write --path ~/.pi/agent/models.json
```

Or from `fak serve`:
```bash
# Print models.json and exit:
fak serve --pi-config

# Write models.json and exit:
fak serve --write-pi-config
```

---

## Pi `models.json` Format

Pi reads custom provider definitions from `~/.pi/agent/models.json` (or `$PI_CODING_AGENT_DIR/models.json`). `fak` configures:

```json
{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [
        {
          "id": "qwen38:27b-q4",
          "name": "Qwen 3.8 27B Q4_K_M (fak serve)",
          "reasoning": false,
          "input": [
            "text"
          ],
          "contextWindow": 131072,
          "maxTokens": 16384,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "compat": {
            "supportsDeveloperRole": false
          }
        }
      ]
    }
  }
}
```

### Configuration fields explained

| Field | Value | Reason |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8080/v1` | Root endpoint of the `fak serve` gateway. |
| `api` | `openai-completions` | Pi streams tool calling via OpenAI Chat Completions API. |
| `apiKey` | `"fak"` | Placeholder token; satisfies Pi's auth check for model availability in `/model`. |
| `compat.supportsDeveloperRole` | `false` | Instructs Pi to send system instructions with `role: "system"` rather than `role: "developer"`. |
| `contextWindow` | `131072` | In-kernel RadixAttention cache supports 128k+ tokens. |

---

## Apple Silicon Unified Memory Sizing

Pick the model tier that matches your Mac's RAM:

| Mac RAM | Model Alias | Quantization | Resident Footprint | Optimal Use |
|---|---|---|---|---|
| **8 GB – 16 GB** | `qwen2.5-coder:7b` | Q4_K_M | ~4.5 GB | Lightweight coding, fast turns |
| **32 GB – 36 GB** | `qwen38` (`qwen38:27b-q2k`) | UD-Q2_K_XL | ~9.4 GB | Fast 27B inference with ample KV headroom |
| **36 GB – 48 GB** | `qwen38:27b-q4` | Q4_K_M | ~16.3 GB | Canonical 27B benchmark weight; high precision |
| **64 GB+** | `qwen38:27b-q4` (large context) | Q4_K_M | 16.3 GB + KV | Deep multi-turn agent sessions |

---

## Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| `fak pi: fak serve backend at http://127.0.0.1:8080/v1 is not responding` | No backend server is running on port 8080. | Start the server in another terminal: `fak serve --gguf qwen38:27b-q4 --pi` |
| Pi `/model` does not list `fak` models | `models.json` is missing or in wrong path. | Run `fak pi config --write` or verify `~/.pi/agent/models.json` exists. |
| `pi: command not found` | Pi CLI is not installed globally or not on `$PATH`. | Run `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` or pass `--command /path/to/pi`. |
| Connection refused on custom port | `fak serve` was bound to a non-default address. | Run `fak pi --addr 127.0.0.1:<port>` or export `FAK_SERVE_ADDR=127.0.0.1:<port>`. |
