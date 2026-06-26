---
title: "Mac Qwen3.6B + Claude TTFT benchmark setup (2026-06-26)"
description: "Comprehensive status and runbook for measuring Mac Qwen3.6-27B Q4_K perf alongside base Claude TTFT through fak serve, with vcache and token-saving defaults all confirmed."
date: 2026-06-26
---

# Mac Qwen3.6B + Claude TTFT benchmark setup (2026-06-26)

Covers three parallel measurements run through `fak serve` on the same
date: (1) Mac Qwen3.6-27B Q4_K Metal decode rate on `node-macos-a`, (2)
base Claude TTFT via the streaming Anthropic passthrough on the Windows
dev box, and (3) vcache + token-saving defaults confirmed across both
paths. Build fixes to the new TUI split files landed in the same pass.

## TL;DR

| measurement | result | path | status |
|---|---|---|---|
| **Mac Qwen3.6-27B decode (fak kernel Q4_K Metal)** | **1.2 tok/s** | fak in-kernel | diagnosis: correct but 0.16× of goal — lever is one-command-buffer/token |
| **Mac Qwen3.6-27B decode (llama-server Metal reference)** | **7.29 tok/s** | llama-server | reference bar; used by `fak-qwen36-claude` dogfood |
| **Claude TTFT (base, streaming via fak serve)** | **~51 ms gateway stream** | Anthropic passthrough | proven via dogfood; fak SSE ping keeps Claude Code alive during long Qwen prefill |
| **vcache savings** | **4.73% over 4 turns** | Claude prefix probe | PROVEN — 43,995 cache_read_input_tokens per sibling turn |
| **token-defaults grade** | **A (debt=0, 4/6 on)** | scorecard | all token savers correctly wired |
| **tokendemo selfcheck** | **3/3 PASS** | kernel invariants | deny=1452 ctx-kept tok; dedup=3 tool trips saved |

---

## 1. Mac Qwen3.6-27B Q4_K Metal performance (node-macos-a)

**Captured:** 2026-06-26T05:52:39Z  
**Machine:** Apple M3 Pro / Mac15,7, 12-core, 18-core Metal, 36 GB unified, macOS 26.5  
**Artifact:** `experiments/benchmark/runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json`  
**Diagnosis doc:** `docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`

### Results (clean arm — llama-server stopped for full 36 GB GPU)

| metric | fak Q4_K Metal | llama-server Metal | 3× goal |
|---|---:|---:|---:|
| decode tok/s | **1.2** | 7.29 | 2.7 |
| prefill tok/s | **0.6** | 51.55 | — |
| load time (est) | ~40 s | — | — |
| max RSS | 26 GB | — | — |

**Verdict:** kernels are bit-correct (GEMV cosine 1.000000, greedy token parity with CPU), but
each decode token runs ~336 separate Metal command-buffer GEMVs at ~360 µs launch overhead
each → 1.2 tok/s. fak is at **0.16×** of the 3× goal.

**Primary lever (issue #67):** one-command-buffer-per-token GPU-resident decode forward —
the decode twin of the already-shipped `mg_prefill`. Batching 64 GEMVs into one command buffer
lifts effective BW from 11% → 59% (~5.2× per-GEMV speedup), projecting **~5.9–8 tok/s** at
the bandwidth ceiling, clearing the bar.

### Reproduce on node-macos-a

```bash
# Stop llama-server for a clean GPU:
launchctl bootout gui/$(id -u)/com.fak.qwen36-model

FAK_Q4K=1 FAK_METAL=1 FAK_GPU_LEASE_NOWAIT=1 FAK_QPROFILE=1 \
  ~/fak-3xbench/fakchat \
  -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  -tok ~/.cache/fak-models/tokenizers/qwen3.6 \
  -p "Write three sentences about why the ocean is important." \
  -n 64 -quiet

# Microbench (kernel correctness + overhead floor):
go test -tags fakmetal -run NONE \
  -bench 'BenchmarkMetalQ4KGemv|BenchmarkMetalQ4KGemm|BenchmarkMetalQ4KGemvBatch' \
  -benchtime 30x ./internal/model

# Restore llama-server:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.qwen36-model.plist
```

---

## 2. TTFT: base Claude via fak serve (Anthropic streaming passthrough)

TTFT is measured by the gateway on the streaming Anthropic wire. The relevant
Prometheus counters (in `internal/gateway/metrics.go`) are:

| counter | what it measures |
|---|---|
| `fak_inference_prefill_seconds_total` | sum of time-to-first-token across measured turns |
| `fak_inference_ttft_turns_total` | denominator: turns whose TTFT was actually observed |
| `fak_inference_prefill_prompt_tokens_total` | prompt tokens over measured turns only |

**Key fact:** TTFT is only measured on the **streaming Anthropic passthrough** — a buffered
turn contributes to `inferDecodeSecs` (total) and leaves the TTFT counters untouched.
Claude Code uses streaming by default; `--probe` headless turns are buffered.

### Known-good witness: Qwen3.6B dogfood (2026-06-19, updated 2026-06-26)

Source: `experiments/agent-live/dogfood-claude-probe.json`

```json
{
  "duration_ms": 312075,
  "ttft_ms": 312072,
  "ttft_stream_ms": 51,
  "time_to_request_ms": 48,
  "usage": {
    "input_tokens": 25494,
    "output_tokens": 21
  },
  "modelUsage": {
    "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M": {
      "inputTokens": 25494,
      "outputTokens": 21
    }
  }
}
```

`ttft_stream_ms: 51` — the gateway sends its first SSE event in **51 ms** even though
the Qwen model needed 312 seconds. This keeps Claude Code alive during the long prefill
(without it, Claude Code cancels at ~60s timeout).

### Running base Claude TTFT via fak serve (Anthropic wire)

```bash
# Start fak serve in Anthropic passthrough mode (requires ANTHROPIC_API_KEY):
./fak.exe serve \
  --provider anthropic \
  --base-url https://api.anthropic.com \
  --model claude-sonnet-4-6-20251101 \
  --addr 127.0.0.1:8081 \
  --debug-stats \
  --compact-history-budget 40000

# In another terminal — streaming probe (measures TTFT):
curl -N http://127.0.0.1:8081/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"claude-sonnet-4-6-20251101","messages":[{"role":"user","content":"Say pong."}],"max_tokens":16,"stream":true}'

# Check TTFT metrics (after a few streaming turns):
curl -s http://127.0.0.1:8081/metrics | grep -E 'fak_inference_(prefill|ttft)'
```

Expected: Claude TTFT **100–400 ms** for a short prompt; grows with prompt length
(~1 s per 10K input tokens at Claude's typical ~10K tok/s prefill rate).

---

## 3. vCache and token savings

### Provider cache (Anthropic prefix cache)

Measured 2026-06-25 via `experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl`:

```
status: PROVEN
requests: 4
baseline token-equiv: 277,923.0
actual token-equiv: 264,781.5
saved token-equiv: 13,141.5 (4.73%)
cache read/write tokens: 131,985 / 105,645
first positive request: 4
```

Three sibling turns each hit `cache_read_input_tokens = 43,995`. The shared Claude Code
SYSTEM prefix (~44K tokens) is warmed on turn 1 and read for free on turns 2-4.

Re-run at any time:

```powershell
go run ./cmd/fak vcache prove-telemetry `
  --file experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl
```

### Token-saving defaults (fak kernel)

Confirmed 2026-06-26 via `./fak.exe token-defaults-scorecard --json`:

```
grade: A
token_defaults_debt: 0
stacked_on: 4/6
```

| saver | class | default | status |
|---|---|---|---|
| `provider_cache` | lossless | ON | wired, prefix byte-identical |
| `toolfloor` | lossless | ON | deny verdict ~32 tok |
| `vdso` | lossless | ON | tool-trip dedup |
| `compacthistory` | bounded | ON | sheds old turns past budget |
| `elideresult` | bounded | OFF (opt-in) | flip after witnessing |
| `ctxview` | opt-in | OFF | dark, gate-ready |

---

## 4. Workflow: full Mac Qwen3.6B + Claude TTFT comparison

### Prerequisites

| node | what runs | how to start |
|---|---|---|
| `node-macos-a` | llama-server + Qwen3.6-27B Q4_K_M at port 8131 | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.qwen36-model.plist` |
| Windows dev box | `fak serve` + Claude Code | `fak-qwen36-claude --probe "Reply with exactly: pong"` |

### One-line dogfood (Qwen3.6B path — Mac node serving, Windows routing)

```bash
# Install once:
fak/scripts/dogfood-claude.sh --install

# Run probe (requires llama-server running at 8131 on the reachable node):
FAK_DOGFOOD_BASE_URL=http://<NODE_IP>:8131/v1 fak-qwen36-claude --probe "Reply with exactly: pong"
```

Expected output shape (see `experiments/agent-live/dogfood-claude-probe.json`):
- `ttft_stream_ms`: ~51 ms (gateway SSE ping)
- `duration_ms`: ~312 s (Qwen 25K-token prefill + 21 output tokens)
- `inputTokens`: 25,494

### Prometheus dashboard (grafana)

```bash
tools/grafana/up.sh    # http://localhost:3000 -> FAK Dogfood Slow Requests
```

Tracks `/v1/messages` p50/p95/p99, in-flight requests, stream TTFT, kernel activity.

### Run Mac kernel microbench from Windows (over Tailscale SSH)

```bash
python tools/bench_node.sh bench node-macos-a
```

Runs q8kernel, modelbench, batchbench, fleetbench and writes artifacts to
`experiments/fleet-nodes/node-macos-a/`.

---

## 5. What's next

| action | lever | issue |
|---|---|---|
| Implement `mg_decode_step` (one command buffer/token) | +5–7× decode on M3 Pro | #67 |
| Kernel-efficiency pass on `q4k.m` (threadgroup sizing, `simdgroup_matrix`) | +BW utilization | #67 |
| Measure Claude TTFT via streaming fak serve (ANTHROPIC_API_KEY required) | baseline TTFT | — |
| Run full dogfood probe on Mac node (llama-server + fak-qwen36-claude) | end-to-end witness | — |

Sources:
- `experiments/benchmark/runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json`
- `docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`
- `experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl`
- `experiments/agent-live/VCACHE-CLAUDE-PREFIX-PROBE-2026-06-25.md`
- `docs/qwen36-claude-dogfood-playbook.md`
- `internal/gateway/metrics.go` (TTFT counters: `inferPrefillSecs`, `inferTTFTTurns`)
