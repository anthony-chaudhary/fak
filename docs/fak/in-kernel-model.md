---
title: "Tier 2: run the fused in-kernel model"
description: "Run fak's own pure-Go forward pass inside the kernel: synthetic checkpoints, the one-command SmolLM2-135M export, a Qwen3.6 GGUF smoke through cmd/fakchat, and in-kernel chat through fak serve."
---

# Tier 2 — run the fused in-kernel model

**Audience:** people working on `fak`'s own inference kernel. **Nothing on this page is
needed to use `fak`.** If you want `fak` in front of the agent or model you already run,
[`GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
is the whole route — Tier 0 (the offline proof) and Tier 1 (`fak serve` in front of any
OpenAI-compatible model server) end there, and this page is the optional continuation.

**Prerequisites:** an installed `fak` binary
([Get the binary](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#1-get-the-binary)).
The synthetic path below needs nothing else — no GPU, no API key, no network. Real
weights additionally need **Python 3.10+**; the fetch script creates a venv and installs
`torch`/`transformers` for you.

> This page holds what used to be §4 of `GETTING-STARTED.md`, moved here so the newcomer
> route reads end to end without it (#5468). The material below is unchanged.

The kernel can dispatch an allowed tool call to a **real pure-Go SmolLM2 forward pass it
owns** (`--engine inkernel`), decoding over a kernel-owned KV cache. This is the deepest
fusion: the model runs inside the kernel address space, and it's reachable via
`/v1/fak/syscall`.

## Synthetic weights — instant, zero download

By default `--engine inkernel` runs a small **deterministic synthetic checkpoint**, so the
decode path works with no model export:

```bash
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-inkernel &
# stop it later with:  kill %1  (bash)  /  Stop-Process  (PowerShell)  /  Ctrl-C if foreground.
# Windows has no '&'/'kill %1': run it in its own window via Start-Process / `start` instead.

curl -s http://127.0.0.1:8137/healthz
# {"engine":"inkernel","model":"smollm2-inkernel","ok":true}

# the fak-native wire key is "arguments" (NOT "args" — an unknown key is silently dropped):
curl -s -X POST http://127.0.0.1:8137/v1/fak/syscall \
  -H 'Content-Type: application/json' \
  -d '{"tool":"read_file","arguments":{"path":"notes.txt"}}'
# {"verdict":{"kind":"ALLOW","by":"monitor"},
#  "result":{"status":"OK",
#    "content":"{\"tool\":\"read_file\",\"engine\":\"inkernel\",\"model\":\"smollm2-inkernel\",\"generated_tokens\":[125,125,...,125]}",
#    "meta":{"engine":"inkernel","ifc_taint":"trusted","input_tokens":"29","output_tokens":"16"}}}
```

This exercises the **real** in-kernel prefill+decode loop over the kernel-owned KV cache.
The *weights* are random-init synthetic, so the tokens are meaningless: it proves the
dispatch+decode path rather than output quality.

## Real SmolLM2-135M weights — one command

The fused model loads a checkpoint exported from HuggingFace (`config.json` +
`manifest.json` + `weights.f32`). One script does the whole export:

```bash
# from fak/ :
./scripts/fetch-model.sh                       # macOS/Linux/WSL/git-bash
#   - or on Windows PowerShell:
#   ./scripts/fetch-model.ps1

# on success the script prints the exact two lines to run — copy them:
export FAK_MODEL_DIR="$PWD/internal/model/.cache/smollm2-135m"
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-135m
```

`FAK_MODEL_DIR` is what actually selects the real weights; `--model` is just the id
advertised on `/v1/models` and `/healthz` (a free-form label).

The script creates a Python venv, installs `torch`/`transformers`/`numpy` (CPU is enough),
downloads `HuggingFaceTB/SmolLM2-135M-Instruct`, and runs
`internal/model/export_oracle.py` into `internal/model/.cache/smollm2-135m`
(git-ignored; regenerable). Preview without doing the work:

```bash
./scripts/fetch-model.sh --check               # report Python + what it would export
FAK_EXPORT_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct ./scripts/fetch-model.sh   # a different model
```

Point any verb that uses the engine at the real weights with `FAK_MODEL_DIR`; if the load
fails the engine falls back to the synthetic checkpoint rather than wedging.

## Expert smoke: Qwen3.6-27B on pure fak

For the Qwen3.6 goal lane, `cmd/fakchat` can run the real local
`Qwen3.6-27B.q4_k_m.gguf` through fak's own in-kernel Gated-DeltaNet path. This does
not use `fak serve`, llama.cpp, Ollama, or an OpenAI-compatible upstream.

```bash
go run ./cmd/fakchat \
  --gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  --tokenizer ~/.cache/fak-models/tokenizers/qwen3.6 \
  --prompt "Say OK." \
  --max-new 1
```

On the witnessed M3 Pro run this loaded the model in about 75 s, peaked at about
25.8 GB RSS, prefilling 22 tokens at about 0.5 tok/s and decoding one cached token at
about 0.1 tok/s. The first greedy token is `<think>`, matching llama.cpp for the same
ChatML prompt. Treat this as a runnability/debug smoke; the current speed bar and the
remaining broader logit-oracle work are tracked in
[`docs/benchmarks/QWEN36-PARITY-RESULTS.md`](../benchmarks/QWEN36-PARITY-RESULTS.md) and
[`docs/benchmarks/FAK-NATIVE-QWEN35-RESULTS.md`](../benchmarks/FAK-NATIVE-QWEN35-RESULTS.md).

## In-kernel CHAT through `fak serve` (both OpenAI + Anthropic wires)

`fak serve` can serve the in-kernel model as a **real chat backend** that goes beyond the
byte-tokenized `/v1/fak/syscall` dispatch demo. With `--gguf` and **no** `--base-url`
(a separate `--tokenizer` is optional; the GGUF's embedded tokenizer is used when
omitted), the gateway routes BOTH `/v1/chat/completions` (OpenAI wire) AND
`/v1/messages` (Anthropic wire) through the in-kernel model via `internal/tokenizer`
+ the `cmd/fakchat` ChatML→Prefill→Step recipe (factored into `agent.InKernelPlanner`).
This is the "test fak locally with the model up" path: fak's own engine as the chat
backend, with no llama-server/Ollama proxy.

```bash
FAK_Q4K=1 ./fak serve --addr 127.0.0.1:8137 \
  --gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  --tokenizer ~/.cache/fak-models/tokenizers/qwen3.6 \
  --model qwen3.6-27b-q4k
# then from another terminal — both work, same model:
curl -s localhost:8137/v1/chat/completions -d '{"model":"x","messages":[{"role":"user","content":"Say OK."}]}'
curl -s localhost:8137/v1/messages        -d '{"model":"x","max_tokens":48,"messages":[{"role":"user","content":"Say OK."}]}'
```

Witnessed on M3 Pro / Qwen3.6-27B q4_k_m: `/v1/chat/completions` returns
`<think>\n\n</think>\n\nOK`; `/v1/messages` returns a live reasoning trace. Decode
depth/sampling default to a greedy 256-token turn (`FAK_INKERNEL_MAX_TOKENS` /
`FAK_INKERNEL_TEMP` / `FAK_INKERNEL_SEED` override). The planner emits **text** today
(no structured tool-call emission yet), so the gateway's adjudication layer still runs on
whatever the caller proposed. `--base-url` (Tier 1 proxy) wins if both are set.

> **Honest caveat (why Tier 2 is not a production chat server).** The
> `fak serve --engine inkernel` SmolLM2 path is proven correct at the *tensor* layer
> against a HuggingFace oracle, and `/v1/fak/syscall` feeds it a bounded **byte-level**
> prompt. `cmd/fakchat` is a separate command-line harness for tokenizer-backed local
> model experiments, including the Qwen3.6 smoke above. These paths make model state
> first-class kernel-owned state; they are not production serving engines. For practical
> chat-quality serving, use **Tier 1** — [`fak serve` in front of any OpenAI-compatible
> model server](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#3-tier-1--put-fak-in-front-of-a-real-model-the-practical-serving-path).
> (This matches the scope in
> [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) and the
> README's honesty ledger.)

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fetch-model.sh`: `need python3` | Install Python 3.10+ or set `PYTHON=/path/to/python`. |
| `fetch-model`: offline / can't reach HuggingFace | The export needs network for the first download; the script forces `HF_HUB_OFFLINE=0`. Re-run once online; the HF cache makes repeats offline-safe. |
| `address already in use` on `fak serve` | Pick another `--addr` port. |

The Tier 0 / Tier 1 symptoms (Go toolchain, Windows test binaries, trace paths) stay in
[the getting-started troubleshooting table](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#troubleshooting).

## Where to go next

- [`GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
  — the install, offline-proof, and practical-serving route this page was split out of.
- [tutorial.md](tutorial.md) — the guided first session, with the real captured output of
  every command.
- [`GPU.md`](https://github.com/anthony-chaudhary/fak/blob/main/GPU.md) — the in-kernel
  Llama decode on the GPU, witnessed against the CPU reference.
- [`docs/benchmarks/IN-KERNEL-MODEL-RESULTS.md`](../benchmarks/IN-KERNEL-MODEL-RESULTS.md) —
  the *evidence* sheet for this path (the forward pass proven rung-by-rung against
  HuggingFace transformers). This page is the how-to; that one is the proof.
- [`docs/benchmarks/`](../benchmarks/README.md) — every per-run sheet behind the numbers
  quoted above.
- [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every
  capability tagged `[SHIPPED]` / `[SIMULATED]` / `[STUB]`.
