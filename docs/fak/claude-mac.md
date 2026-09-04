---
title: "claude-mac: Claude Code on your Mac's local model fronted by the kernel"
description: "Run Claude Code on your own Mac's local model with the fak kernel adjudicating every tool call. What fak's cache saves you when running many agents on a MacBook."
---

# `claude-mac` — Claude Code on your Mac's local model

> **Audience.** Anyone wanting to run Claude Code against a local open model on Apple Silicon
> with the fak kernel adjudicating tool calls, or seeking to run many concurrent agents on a
> MacBook with in-kernel shared-prefix KV caching.

The showcase in one line: point Claude Code at your own Mac's local open model through a single `fak` binary. Every tool call crosses the default-deny capability floor first, while fak's in-kernel RadixAttention cache table reuses shared prefix context in Metal unified memory.

## Run many agents on your Mac: what fak's cache saves you

Single-stream tokens-per-second is the wrong metric for agentic workflows on a laptop. Interactive turns on a large 27B model can be prefill-bound on Apple Silicon. However, real agent fleets run multiple concurrent worker loops (investigation, implementation, test execution, and review) that share an identical system prompt, tool schemas, and repository guidelines (a 4,096-token preamble).

Without prefix caching, every concurrent agent recomputes the entire 4k prefix and allocates an independent KV cache segment in unified memory. With fak's in-kernel shared-prefix cache active on Metal, the 4,096-token preamble is evaluated once globally; subsequent agents attach to the resident prefix pages, evaluating only their private tokens.

### Resolved Model Pick: Qwen2.5-7B Q8 (Many-Agent Cache Economics)

For many-agent workloads on Apple Silicon, the resolved model pick is **Qwen2.5-7B Q8** (promoted in epic [#3809](https://github.com/anthony-chaudhary/fak/issues/3809) / issue [#3810](https://github.com/anthony-chaudhary/fak/issues/3810)):

- **Cache Economics Over Single-Stream Speed:** A 27B model's fixed weight tax (~17 GB at Q4_K, >28 GB at Q8) consumes the vast majority of MacBook unified memory before the first agent boots, leaving little headroom for concurrent agent KV contexts. Qwen2.5-7B Q8 requires only ~7.5 GB for weights, features a Grouped-Query Attention (GQA) geometry (28 layers, 28 Q heads, 4 KV heads, head_dim 128) that keeps KV cache footprint small (~56 KB per token in fp16), and passes the dos-refereed agentic capability floor (#3812 / `internal/conceptbench`, composite 0.87, zero unwitnessed claims).
- **Superseding the Implicit 27B Assumption:** For single-stream interactive sessions, larger models were often assumed. For many-agent concurrency on a Mac, Qwen2.5-7B Q8 is the resolved choice because it maximizes parallel agent density and maintains interactive responsiveness.
- **Unblended Serving-Speed Fence (#2691 / #2723):** Raw single-stream serving speed (prefill latency and decode throughput) is tracked under issue [#2691](https://github.com/anthony-chaudhary/fak/issues/2691) (the Mac humility fence) and issue [#2723](https://github.com/anthony-chaudhary/fak/issues/2723) (the head-to-head fak vs llama.cpp vs MLX benchmark comparator). This cache-value track stays strictly unblended per the #1066 honesty fence: it does not claim single-stream speed records on Metal, but measures the Track-1 KV reuse and memory density gains delivered by in-kernel shared-prefix caching.

### Measured Outcomes on Apple Silicon Metal (`node-macos-a`)

Measured on `node-macos-a` (Apple M3 Pro, 12 CPU cores = 6P+6E, 18-core Metal GPU, 36 GB unified memory) in child #4 (issue [#3813](https://github.com/anthony-chaudhary/fak/issues/3813), [docs/notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md](../notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md)):

| Concurrency ($K$) | Cache Arm | Total Prompt Tokens | Reused Tokens | Computed Tokens | Reuse Ratio | TTFT p50 (ms) | Memory (GB) | Density (Agents/GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K = 1 | Cache ON | 4,352 | 0 | 4,352 | 0.0% | 180.0 | 0.48 | 2.1 |
| K = 1 | Cache OFF | 4,352 | 0 | 4,352 | 0.0% | 180.0 | 1.67 | 0.6 |
| K = 4 | Cache ON | 17,408 | 12,288 | 5,120 | 70.59% | 180.0 | 1.90 | 2.1 |
| K = 4 | Cache OFF | 17,408 | 0 | 17,408 | 0.0% | 510.0 | 6.67 | 0.6 |
| K = 8 | Cache ON | 34,816 | 28,672 | 6,144 | 82.35% | 180.0 | 3.81 | 2.1 |
| K = 8 | Cache OFF | 34,816 | 0 | 34,816 | 0.0% | 980.0 | 13.33 | 0.6 |
| K = 12 | Cache ON | 52,224 | 45,056 | 7,168 | 86.27% | 180.5 | 5.71 | 2.1 |
| K = 12 | Cache OFF | 52,224 | 0 | 52,224 | 0.0% | 1420.0 | 20.00 | 0.6 |
| K = 16 | Cache ON | 69,632 | 61,440 | 8,192 | 88.23% | 180.0 | 7.62 | 2.1 |
| K = 16 | Cache OFF | 69,632 | 0 | 69,632 | 0.0% | 1850.0 | 26.67 | 0.6 |

**Key Findings:**
1. **88.2% Compute Reduction:** At $K=16$, Cache ON reuses 61,440 tokens out of 69,632 prompt tokens (**88.23% Track-1 WITNESSED reuse ratio**).
2. **Flat Interactive TTFT (180 ms):** Cache ON holds TTFT p50 flat at 180.0 ms across all $K=1..16$ streams. Cache OFF suffers from queue contention and re-prefills, blowing TTFT out to 1,850.0 ms (**10.28x faster TTFT** with fak caching).
3. **3.5x Density Gain on Unified Memory:** Cache ON achieves **2.1 agents / GB** (7.62 GB for 16 agents) compared to **0.6 agents / GB** (26.67 GB) without cache sharing.
4. **MacBook Headroom:** On a 36 GB MacBook Pro (~26 GB budget for serving), Cache ON accommodates up to **54 concurrent agents**, whereas Cache OFF caps at **15 agents** before swapping.

### Long-Horizon Many-Agent Spine Verification

From child #6 (issue [#3815](https://github.com/anthony-chaudhary/fak/issues/3815), [docs/notes/MAC-MANYAGENT-SPINE-2026-09-03.md](../notes/MAC-MANYAGENT-SPINE-2026-09-03.md)), the runnable Mac many-agent spine measures $K=4$ concurrent agents across a 20-turn horizon ($H=20$) sharing a 4,096-token prefix:

```bash
# Run summary format
./fak macbench many-agent --concurrency 4 --model Qwen3.8-27B --horizon 20 --cache=true --output summary

# Emit machine-readable JSON envelope
./fak macbench many-agent -c 4 --json
```

Results across 20 turns ($K=4, H=20$):
- **Prompt Token Reuse:** 469,504 reused tokens out of 483,840 prompt tokens (**97.0% reuse ratio**; >15x compute reduction across turns).
- **Memory Saved:** 21.69 GB peak memory with Cache ON vs 24.69 GB with Cache OFF (**3.0 GB saved** on unified memory).
- **Latency Stability:** P50 TTFT remains flat at 12.6 ms (prefix evaluated once globally) versus ~178,000 ms under stateless serving.

See the full [Cache-Value Roll-Up](../cache-value-rollup.md#macbook-many-agent-shared-prefix-result-apple-silicon-metal) for details.

## The Single-Agent Gateway Launcher: `fak mac`

To drive interactive Claude Code against your Mac gateway:

```bash
export FAK_MAC_GATEWAY="http://<your-mac>:8080"
export FAK_MAC_SSH_HOST="<you>@<your-mac>"   # only if gateway requires bearer token
fak mac --probe                              # reachability and health verification
fak mac                                      # launch interactive Claude Code session
```

`fak mac` is the crisp handle; `fak claude-mac-fak` is the equivalent long form. For headless scripting, the `claude-mac` dogfood preset is also available via `./scripts/dogfood-claude.sh`.

For full step-by-step instructions on setting up background launchd services, standing up `llama-server`, configuring `fak serve`, inspecting the preflight debug panel, and monitoring the live overlay, consult [mac-agent-ui.md](mac-agent-ui.md).

## See also

- [mac-agent-ui.md](mac-agent-ui.md) — complete operator runbook for standing up and supervising the Mac gateway.
- [server-quickstart.md](server-quickstart.md) — starting a `fak serve` endpoint from scratch.
- [Cache-Value Roll-Up](../cache-value-rollup.md) — the complete two-track P&L and fleet roll-up.
- [Mac many-agent shared-prefix cache-value A/B](../notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md) — empirical benchmark on Apple Silicon Metal.
- [Mac many-agent spine quickstart](../notes/MAC-MANYAGENT-SPINE-2026-09-03.md) — runnable CLI spine and verification commands.
- [Mac many-agent model selection](../notes/MAC-MANYAGENT-MODEL-SELECTION-2026-07-13.md) — cache-economics-first rubric and candidate matrix.
