---
title: "Qwen3.6-27B on AMD Radeon RX 7600 via Vulkan"
description: "Witnessed run of Qwen3.6-27B Q4_K_M on an AMD/Vulkan Windows desktop, proving the model loads and serves all three fak surfaces despite a GDN perf caveat."
---

# Qwen3.6-27B on AMD/Vulkan Windows desktop

Witnessed 2026-06-19 on `node-desktop-b`.

## Hardware and backend

- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads, AVX512-capable.
- RAM: 272 GB physical.
- GPU devices seen by llama.cpp b9673 WinGet Vulkan build:
  - `Vulkan0`: AMD Radeon RX 7600, 8176 MiB.
  - `Vulkan1`: AMD Radeon(TM) Graphics, integrated UMA.
- Model: `Qwen3.6-27B-Q4_K_M.gguf`, 16,547,398,784 bytes, from
  `lmstudio-community/Qwen3.6-27B-GGUF`.
- Server command shape:
  `llama-server -m Qwen3.6-27B-Q4_K_M.gguf --host 127.0.0.1 --port 8131 --ctx-size 8192 --n-gpu-layers 20 --fit on`.

## Load result

`llama-server` loaded the model and exposed `/v1/models` with:

- `n_vocab`: 248320
- `n_ctx`: 8192
- `n_ctx_train`: 262144
- `n_embd`: 5120
- `n_params`: 26895998464
- `size`: 16536406016

The backend ran as a hybrid CPU/Vulkan setup, but llama.cpp logged a real perf caveat:

```text
sched_reserve: layer 0 is assigned to device CPU but the fused Gated Delta Net tensor is assigned to device Vulkan0 (usually due to missing support)
fused Gated Delta Net (chunked) not supported, set to disabled
```

So this proves the model runs here, but also explains why absolute throughput is well below
the M3 Pro llama.cpp reference in `QWEN36-PARITY-RESULTS.md`.

## End-to-end fak surface proof

Full smoke command:

```powershell
python tools\qwen36_surface_smoke.py `
  --base-url http://127.0.0.1:8131/v1 `
  --model Qwen3.6-27B-Q4_K_M.gguf `
  --node-name amd-vulkan-local-full `
  --gateway-chat `
  --perf-decode-baseline-tps 7.29 `
  --out fak\experiments\qwen36\amd-vulkan-local-full.json `
  --markdown fak\experiments\qwen36\amd-vulkan-local-full.md `
  --model-timeout-s 600 `
  --agent-timeout-s 900 `
  --http-timeout-s 20
```

Result: **3/3 surfaces passed**.

| surface | proof |
|---|---|
| `agent` | live `fak agent`, one turn per arm, real model `Qwen3.6-27B-Q4_K_M.gguf`, 3 tool calls per arm, report written |
| `gateway-openai` | `/v1/models` listed the model and `/v1/chat/completions` returned `OK` |
| `mcp-http` | `initialize`, `tools/list`, and `tools/call fak_adjudicate` all returned JSON-RPC results |

Artifacts:

- `fak/experiments/qwen36/amd-vulkan-local-full.json`
- `fak/experiments/qwen36/amd-vulkan-local-full.md`
- `fak/experiments/qwen36/qwen36-agent-amd-vulkan-local-full.json`

## Standalone packet path

For another AMD/Vulkan Windows test bench, generate and send the explicit Vulkan
packet instead of using the NVIDIA wrapper. The broader multi-GPU serving plan is the GPU model ladder.

```powershell
python tools\qwen36_node_packet.py --profile vulkan --report-target auto
```

For a watched Tailscale node, force the packet profile when the node registry does
not carry GPU facts:

```powershell
python tools\qwen36_watch_nodes.py `
  --node <tailnet-node> `
  --send-packet `
  --packet-profile vulkan `
  --gateway-chat `
  --perf-decode-baseline-tps 7.29
```

The generated node wrapper runs `qwen36_node_server.py --profile vulkan`, which keeps
the AMD/llama.cpp shape used above: partial Vulkan offload, `--fit on`, tailnet-only
bind, preflight-first start, and returned `qwen36-reports/` logs.

## Performance

Gateway chat report:

- Prompt tokens: 15
- Completion tokens: 125
- Wall time: 46.351 s
- Gateway-estimated decode: 2.697 tok/s
- llama.cpp decode: 2.80 tok/s
- Ratio to the M3 Pro llama.cpp reference decode bar (7.29 tok/s): 0.38x

llama.cpp server timing for that same gateway chat:

```text
prompt eval time = 1673.25 ms / 15 tokens (8.96 tokens per second)
eval time        = 44589.79 ms / 125 tokens (2.80 tokens per second)
total time       = 46263.04 ms / 140 tokens
```

Interpretation:

- **End-to-end chat works on this AMD CPU/GPU setup.**
- **fak gateway overhead is near parity with llama.cpp on the same setup**:
  2.697 tok/s wall-clock through fak vs 2.80 tok/s from llama.cpp timing, about 0.96x.
- **Absolute local decode is not at the M3 Pro reference bar**:
  2.80 tok/s vs 7.29 tok/s, about 0.38x, due at least in part to llama.cpp's current Vulkan
  Gated-DeltaNet fallback.

## Pure-fak in-kernel speed parity sweep

After the Q8 hybrid fresh-prefill and head-parallel Gated-DeltaNet scan work, the pure-fak
in-kernel runtime now has a direct local microbench against this same WinGet llama.cpp Vulkan
build and the same GGUF.

Commands:

```powershell
go -C fak run ./cmd/modelbench `
  -lean `
  -gguf C:\Users\USER\.cache\fak-models\gguf\Qwen3.6-27B-Q4_K_M.gguf `
  -prefill-sizes 16,64,256 `
  -prefill-reps 1 `
  -decode-prompt 16 `
  -decode-steps 4 `
  -decode-reps 1 `
  -out experiments\qwen36\native-gguf-q8-hybrid-headscan-p16-64-256-20260619.json

& C:\Users\USER\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-bench.exe `
  -m C:\Users\USER\.cache\fak-models\gguf\Qwen3.6-27B-Q4_K_M.gguf `
  -p 16,64,256 `
  -n 1 `
  -r 1 `
  -o json > fak\experiments\qwen36\llamacpp-vulkan-qwen36-pp16-64-256-tg1-20260619.json
```

| Workload | pure-fak Q8 in-kernel | llama.cpp Vulkan b9673 | Ratio |
|---|---:|---:|---:|
| Prefill P16 | 14.86 tok/s | 5.20 tok/s | 2.86x |
| Prefill P64 | 27.46 tok/s | 14.57 tok/s | 1.88x |
| Prefill P256 | 31.34 tok/s | 9.95 tok/s | 3.15x |
| Decode TG1 | 1.24 tok/s | 0.99 tok/s | 1.25x |

Artifacts:

- `fak/experiments/qwen36/native-gguf-q8-hybrid-headscan-p16-64-256-20260619.json`
- `fak/experiments/qwen36/native-gguf-q8-hybrid-batch-p16-64-256-20260619.json`
- `fak/experiments/qwen36/llamacpp-vulkan-qwen36-pp16-64-256-tg1-20260619.json`

Extended sweep:

| Workload | pure-fak Q8 in-kernel | llama.cpp Vulkan b9673 | Ratio |
|---|---:|---:|---:|
| Prefill P512 | 30.28 tok/s | 9.32 tok/s | 3.25x |
| Prefill P1024 | 29.67 tok/s | 9.31 tok/s | 3.19x |
| Decode D16 | 1.15 tok/s | 0.99 tok/s | 1.16x |

Extended artifacts:

- `fak/experiments/qwen36/native-gguf-q8-hybrid-headscan-p512-1024-dp256-d16-20260619.json`
- `fak/experiments/qwen36/llamacpp-vulkan-qwen36-pp512-1024-tg16-20260619.json`
- `fak/experiments/qwen36/qwen36-perf-gate-amd-20260619.json`
- `fak/experiments/qwen36/qwen36-perf-gate-amd-20260619.md`

Interpretation:

- **Single-stream microbench speed parity is met on this AMD CPU/GPU setup** for the measured
  P16/P64/P256/P512/P1024 prefill points and the measured TG1/D16 decode points.
- The parity statement is machine-checked by `python tools/qwen36_perf_gate.py`; the current
  gate report is PASS at a 1.0x minimum fak/llama.cpp ratio.
- This is still not a broad quality/logit-parity claim. The real-artifact logit oracle and
  longer prompt/decode sweeps remain separate gates.
- The llama.cpp baseline is the WinGet b9673 Vulkan build with `n_gpu_layers=-1`; the live
  server path above may use `--fit on --n-gpu-layers 20`, so compare artifact-to-artifact.
