# Start Here: Run AI on Your Computer

This page gets you from zero to chatting with a local AI model in under 10 minutes.

## What you can do

After following these steps, you'll have an AI running on your own computer that works
offline, costs nothing (no API keys, no cloud bills), keeps your data on your machine,
and runs on CPU — no GPU needed for small models.

## Pick your path

| I want to... | Follow this |
|---------------|-------------|
| **Prove the safety gate in 60 seconds** (no model, no download, no key) | [See it in 2 minutes](README.md#see-it-in-2-minutes-no-key-no-model-no-gpu) — one structural DENY |
| **See the gate stop a live attack** (Go only, ~1 min, no downloads) | [AgentDojo red-team demo](examples/agentdojo-redteam/README.md) |
| **I'm a coding agent** (build/test/run + the rules) | [AGENTS.md](AGENTS.md) |
| **Chat with a local AI** (most fun — needs a ~1.6 GB model download) | [Simple Demo](cmd/simpledemo/README.md) — 5 minutes |
| **Follow a guided first session** (real output at every step) | [Tutorial](docs/fak/tutorial.md) — 15 minutes ⭐ |
| **Put a safety gate in front of my AI** | [Getting Started](GETTING-STARTED.md) — 10 minutes |
| **Understand what fak actually does** | [Main README](README.md) |
| **See the performance benchmarks** | [Benchmark Authority](BENCHMARK-AUTHORITY.md) |

## Quick: Try the chat demo (5 minutes)

### 1. Download a model

Pick one:
- **[Qwen2.5-1.5B-Q8](https://huggingface.co/mradermacher/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q8_0.gguf)** (1.6 GB) — Fast, good quality
- **[Qwen2.5-3B-Q8](https://huggingface.co/mradermacher/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q8_0.gguf)** (3.2 GB) — Better quality

Also download: **[tokenizer.json](https://huggingface.co/mradermacher/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/tokenizer.json)**

Save both to the same folder (e.g., `~/Downloads/` or `C:\Users\You\Downloads\`).

### 2. Run it

**Linux/macOS (one line):**
```bash
go run ./cmd/simpledemo -gguf ~/Downloads/Qwen2.5-1.5B-Instruct-Q8_0.gguf -tok ~/Downloads
```

**Windows PowerShell (one line):**
```powershell
go run ./cmd/simpledemo -gguf $env:USERPROFILE\Downloads\Qwen2.5-1.5B-Instruct-Q8_0.gguf -tok $env:USERPROFILE\Downloads
```

### 3. Chat!

```
You: Explain quantum computing like I'm 12
AI: Imagine a regular computer is like a light switch — it's either ON (1) or OFF (0)...
```

## What is fak?

**fak** is **one Go binary** that sits between your AI agents and the tools they call.
Everything runs inside that one process — the permission gate, the cache, the quarantine,
the metrics — so there are no sidecars, no separate authorizer, and no multi-tier ops:

- **Self-contained** — one static Go binary, zero external dependencies, no complex setup
- **Safer** — puts every action behind a permission gate the model can't talk past
- **Cheaper for fleets** — does the shared setup work once instead of every turn

For fleets of AI agents that share setup (long system prompts, tool lists), the savings
compound: the first agent pays for the shared work, everyone after reads it for free.

## How fast is it?

On a measured 50-turn × 5-agent session, `fak` did in ~19 minutes what a **naive
re-send-everything loop** does in ~19 hours — a **60× gain against that naive baseline**.
Against a *tuned* warm-cache stack the honest gain is a few-fold (~4×); the eye-catching
60× is only versus the naive pattern, whose cost balloons because it reprocesses the whole
growing conversation every turn. The reuse win is **self-host only** and applies to
read-heavy fleets.

See [`fak/BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) for every number traced to
its commit and artifact.

## Next steps

1. Run the [Simple Demo](cmd/simpledemo/README.md)
2. Read [Getting Started](GETTING-STARTED.md) for the full feature set
3. Explore [examples](examples/) of safety policies and tool gates
4. Check the [main README](README.md) for architecture and benchmarks

## Requirements

- **Go 1.26+** (auto-downloads if you have an older version)
- **4-8 GB RAM** (depends on model size)
- That's it!

---

**Lost?** Each subdirectory has its own README with detailed instructions. Start with [Simple Demo](cmd/simpledemo/README.md).
